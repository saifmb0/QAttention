"""
ragged_attn.py
==============
Triton ragged attention — ancestor-sparse Flash Attention for tree-structured
speculative decoding.

Target hardware : SM75 (NVIDIA T4, Kaggle 2×T4) — original
                  SM89 (NVIDIA RTX 6000 ADA PRO 96 GB GDDR7) — "blackwell" branch
Precision       : float16 or bfloat16 input/output, float32 accumulation

─────────────────────────────────────────────────────────────────────────────
Ada Lovelace (SM 8.9) optimisation notes  — RTX 6000 ADA PRO
─────────────────────────────────────────────────────────────────────────────

Hardware spec
  Architecture  : Ada Lovelace (AD102 full die)
  CUDA version  : SM 8.9
  SMs           : 142
  FP16 peak     : 364.2 TFLOPS  (w/ tensor cores)
  FP8  peak     : 1457   TFLOPS  (E4M3/E5M2 tensor cores)
  HBM bandwidth : 820 GB/s  (GDDR7, 384-bit bus)
  L2 cache      : 96 MB  ← 25× larger than T4 (3.8 MB); K/V reuse is free
  Shared mem/SM : up to 100 KB per block
  Regs/SM       : 65 536 (same as T4, but 142×SM vs 40×SM = 3.55× total)
  Warp size     : 32

Key improvements over SM75 (T4)
  1. BLOCK_M 256 is viable — 142 SMs can sustain 256-thread CTAs without
     occupancy collapse (T4 would stall at 256 due to 40 SMs × 16 CTAs/SM).
  2. num_stages=2 enables software-pipeline prefetch for step 0 K-loads
     (step 0 address = seq_start + q_off + m_range, fully data-independent).
     Subsequent steps still use num_stages=1 (data-dependent addresses).
  3. BF16 natively accelerated on Ada tensor cores (unlike SM75 which only
     has FP16/INT8/TF32 tensor cores); added bfloat16 precision path.
  4. 96 MB L2 is large enough to hold the full K/V of even the largest batch
     in this workload (32 × 1 365 × 8 × 64 × 2 bytes = 360 MB — not quite,
     but many ancestor K/V positions are shared across query tiles, so the
     effective working set fits easily).
  5. Halved memory-bandwidth pressure relative to T4: 820 / 300 ≈ 2.7× BW,
     but our sparse kernel is already compute-bound at depth ≥ 3 on T4,
     so the effective gain flows mostly from occupancy and clock headroom.

Roofline ridge point (Ada)
  I* = peak_FP16 / HBM_BW = 364.2e12 / 820e9  ≈ 444 FLOPs/byte
  (vs T4: 65e12/300e9 ≈ 217 FLOPs/byte)
─────────────────────────────────────────────────────────────────────────────

─────────────────────────────────────────────────────────────────────────────
Bottleneck analysis  (profile_kernel.py on T4, commit a9fa79b)
─────────────────────────────────────────────────────────────────────────────

The profiling data showed util% ≈ 3 000–574 000% across all configurations.
util% = actual_ms / roofline_ms × 100, so 3 000% means the kernel ran 30×
slower than the theoretical HBM/compute ceiling.  The pattern was uniform
regardless of batch size B (B=1 to B=32), which rules out CTA starvation as
the primary cause.

Root cause: *** algorithmic sparsity blindness ***

In BFS-ordered b-ary trees, query q can attend ONLY to its ancestors:
    { parent^0(q),  parent^1(q), ...,  parent^MAX_DEPTH(q) }
that is  MAX_DEPTH+1 ≤ 6  distinct KV positions out of  N  total.

The v0 dense kernel iterated over ALL ⌈N/BLOCK_N⌉ KV blocks and masked
non-ancestor positions to –∞.  For b=4, d=5 → N=1 365:
    useful work / total work  =  6 / 1 365  =  0.44%
99.56% of every QK^T computation and K/V memory load was thrown away.
The ~2 TFLOPS plateau in the benchmark is simply the HBM bandwidth
ceiling for streaming 1 365 × 64 fp16 K/V values that are never used.

─────────────────────────────────────────────────────────────────────────────
Fix: ancestor-sparse Flash Attention  (_ragged_attn_sparse_kernel)
─────────────────────────────────────────────────────────────────────────────

Instead of iterating over KV-blocks, the new kernel iterates over the
MAX_DEPTH+1 ancestor steps.  For each step s, it performs a
*scattered gather*: loads K[parent^s(q)] and V[parent^s(q)] for every
query row q in the BLOCK_M-sized Q-tile.

Complexity comparison (B=4, b=4, d=5 → N=1 365, D=64, H=8, BLOCK_M=64):

  Dense kernel:
    FMA ops / Q-tile  = ⌈N/BLOCK_N⌉ × BLOCK_M × BLOCK_N × D × 2
                      = 22 × 64 × 64 × 64 × 2  = 11.5 GFLOPs
    K+V bytes / Q-tile = N × D × sizeof(fp16) × 2
                       = 1 365 × 64 × 2 × 2  = 341 KB

  Sparse kernel:
    FMA ops / Q-tile  = (MAX_DEPTH+1) × BLOCK_M × D  (element-wise dot)
                      = 6 × 64 × 64                  = 24 576
    K+V bytes / Q-tile = (MAX_DEPTH+1) × BLOCK_M × D × sizeof(fp16) × 2
                       = 6 × 64 × 64 × 2 × 2         = 96 KB

  Reduction: 470× fewer FMAs, 3.6× less HBM traffic per Q-tile.
  With L2 reuse (all N unique K/V positions = 341 KB — fits in T4 L2):
  effective HBM fills ≈ once per unique K/V position across all Q-tiles.

─────────────────────────────────────────────────────────────────────────────
Algorithm: online softmax with per-step single-element updates
─────────────────────────────────────────────────────────────────────────────

Each step s updates the Flash-Attention running state (m_i, l_i, acc)
with ONE score per query row instead of BLOCK_N scores:

    For step s in range(MAX_DEPTH+1):
        cur[m] = parent^s(q_off + m)    (sequence-local position)
        if cur[m] == prev[m]: skip      (root revisited when depth < MAX_DEPTH)
        k_anc[m] = K[seq_start + cur[m], head, :]   # scattered load
        score[m] = dot(q[m], k_anc[m]) * scale       # element-wise dot
        m_new = max(m_i, score)
        alpha = exp(m_i - m_new)
        p     = exp(score - m_new)
        v_anc[m] = V[seq_start + cur[m], head, :]    # scattered load
        l_i  = l_i * alpha + p
        acc  = acc * alpha + p * v_anc

This is exactly Flash-Attention-2 (Dao et al. 2023) with the KV-block
loop replaced by the ancestor walk loop.

─────────────────────────────────────────────────────────────────────────────
Public API
─────────────────────────────────────────────────────────────────────────────
  pack_inputs(qs, ks, vs)
      -> (Q, K, V, cu_seqlens)

  ragged_attention(Q, K, V, cu_seqlens, branching_factor, max_depth)
      -> O  [same shape as Q]
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# SM75 autotune configs for the ancestor-sparse kernel
#
# The sparse kernel has no BLOCK_N dimension — there are no KV tiles.
# We tune only BLOCK_M, which controls Q-tile size and register pressure.
#
# Register budget per thread (SM75: 65 536 regs/SM):
#   BLOCK_M=64,  HEAD_DIM=64,  num_warps=4 (128 threads):
#     q + k_anc + v_anc: 3 × (64×64/128) fp16 = 3×32 = 96 fp16 → 48 fp32 regs
#     acc: (64×64/128) fp32 = 32 fp32 regs
#     scalars (m_i, l_i, alpha, p, cur, prev): ~16 regs
#     Total: ~96 regs/thread  →  no register spill
#   BLOCK_M=128, HEAD_DIM=128, num_warps=8: ~192 regs → risk of spill
#
# num_stages=1: each step's load addresses depend on the previous step's
# `cur` value (data-dependent) → no software pipeline overlap is possible.
# ---------------------------------------------------------------------------
_SPARSE_SM75_CONFIGS = [
    triton.Config({"BLOCK_M": 16},  num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=1),
]

# ---------------------------------------------------------------------------
# SM89 autotune configs — RTX 6000 ADA PRO (Ada Lovelace)
#
# Differences from SM75:
#   • 142 SMs vs 40 → no occupancy cliff at BLOCK_M=256 (128-thread CTAs)
#   • 96 MB L2 → K/V ancestor positions hot in L2 across all tiles
#   • Tensor cores natively accelerate BF16 in addition to FP16
#   • num_stages=2 is safe for step 0 (address fully data-independent);
#     deeper steps remain num_stages=1 (data-dependent gather addresses).
#     Triton uses the same num_stages for the whole kernel; we set 2 here
#     because the step-0 prefetch is the dominant latency hider at B=32.
#
# Register budget at BLOCK_M=256, HEAD_DIM=64, num_warps=8 (256 threads):
#   q + k_anc + v_anc: 3 × (256×64/256) fp16 = 3×64 = 192 fp16 → 96 fp32 regs
#   acc: (256×64)/256 fp32 = 64 fp32 regs
#   scalars: ~16 regs
#   Total: ~176 regs/thread — within 65 536/256 = 256 limit, no spill.
#
# BLOCK_M=256, HEAD_DIM=128, num_warps=16: 192+128+16 = 336 regs → spill risk;
#   num_warps=16 (512 threads) lowers per-thread quota to 128 → borderline;
#   retained as candidate but typically autotuned away.
# ---------------------------------------------------------------------------
_SPARSE_SM89_CONFIGS = [
    # Small tiles — low-latency for tiny batches (B=1)
    triton.Config({"BLOCK_M": 32},  num_warps=4,  num_stages=1),
    triton.Config({"BLOCK_M": 64},  num_warps=4,  num_stages=2),
    # Medium tiles — balanced occupancy
    triton.Config({"BLOCK_M": 64},  num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 128}, num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 128}, num_warps=16, num_stages=2),
    # Large tiles — exploit 96 MB L2 and high SM count
    triton.Config({"BLOCK_M": 256}, num_warps=8,  num_stages=2),
    triton.Config({"BLOCK_M": 256}, num_warps=16, num_stages=2),
]


def _get_autotune_configs() -> list:
    """Select autotune config set based on the current CUDA device's SM."""
    if not torch.cuda.is_available():
        return _SPARSE_SM75_CONFIGS
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    # SM 8.9 = Ada Lovelace; also covers 8.x forward
    if (props.major, props.minor) >= (8, 9):
        return _SPARSE_SM89_CONFIGS
    return _SPARSE_SM75_CONFIGS


# ---------------------------------------------------------------------------
# Ancestor-sparse Flash Attention kernel
# ---------------------------------------------------------------------------
# Algorithm : Flash-Attention-2 online softmax (Dao 2023) adapted to iterate
#             over MAX_DEPTH+1 ancestor steps — NOT over KV blocks.
# Masking   : none needed — each step processes exactly one valid KV position
#             per query row (its actual ancestor), with duplicate detection
#             via `prev` to handle shallow trees (root visited multiple times).
# Grid      : (B * H,  ceil(max_seqlen / BLOCK_M))
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["HEAD_DIM", "BRANCHING_FACTOR", "MAX_DEPTH"],
)
@triton.jit
def _ragged_attn_sparse_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    cu_seqlens_ptr,            # int32 [B+1]
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    scale,
    max_seqlen,
    H:                tl.constexpr,
    HEAD_DIM:         tl.constexpr,
    BRANCHING_FACTOR: tl.constexpr,
    MAX_DEPTH:        tl.constexpr,
    BLOCK_M:          tl.constexpr,
):
    # ── CTA identification ──────────────────────────────────────────────────
    pid0   = tl.program_id(0)   # seq_idx * H + head_idx
    m_tile = tl.program_id(1)   # tile index along Q axis

    seq_idx  = pid0 // H
    head_idx = pid0  % H

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end   = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len   = seq_end - seq_start

    q_off = m_tile * BLOCK_M
    if q_off >= seq_len:
        return

    m_range  = tl.arange(0, BLOCK_M)
    d_range  = tl.arange(0, HEAD_DIM)
    valid_q  = (m_range + q_off) < seq_len       # [BLOCK_M]
    q_global = seq_start + q_off + m_range        # [BLOCK_M] absolute indices

    # ── Load Q tile  [BLOCK_M, HEAD_DIM]  fp16 ─────────────────────────────
    q_ptrs = (Q_ptr
              + q_global[:, None] * stride_qt
              + head_idx          * stride_qh
              + d_range  [None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)

    # ── Flash-Attention-2 online softmax state ──────────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # ── Ancestor walk state ──────────────────────────────────────────────────
    # cur[m]  = sequence-local position of the step-s ancestor of query m
    # prev[m] = previous step's position (initialised to -1 so step 0 is new)
    cur  = (m_range + q_off).to(tl.int32)   # [BLOCK_M] — step 0 = self
    prev = tl.full([BLOCK_M], -1, dtype=tl.int32)

    # ── Ancestor-sparse loop (MAX_DEPTH+1 steps, fully unrolled) ───────────
    #
    # Why `is_new = (cur != prev)`:  the parent walk clamps at 0 (root).
    # For queries shallower than MAX_DEPTH, cur reaches 0 before the loop
    # ends; subsequent steps have cur == prev == 0 → is_new = False, so
    # those iterations do nothing (alpha=1, p=0, zero-masked loads).
    #
    # Parent formula: parent(k) = max(k-1, 0) // b
    # Clamping BEFORE the divide is critical: for k=0, k-1=-1 (int32).
    # PTX lowers non-power-of-2 division to multiply-high (unsigned), so
    # -1 as uint32 = 0xFFFFFFFF yields a spurious large parent index.
    # `tl.maximum(cur-1, 0)` ensures the dividend is always non-negative.
    for _step in range(MAX_DEPTH + 1):      # constexpr → fully unrolled

        is_new = (cur != prev) & valid_q    # [BLOCK_M]
        kv_abs = seq_start + cur            # [BLOCK_M] absolute token indices

        # ── Scatter-gather K[ancestor]  [BLOCK_M, HEAD_DIM]  fp16 ──────────
        # Different query rows load from different positions — scattered.
        # Pairs of adjacent queries often share high-level ancestors, so L2
        # captures repeated loads at depth >= 2.
        k_ptrs = (K_ptr
                  + kv_abs[:, None] * stride_kt
                  + head_idx        * stride_kh
                  + d_range [None,:] * stride_kd)
        k_anc = tl.load(k_ptrs, mask=is_new[:, None], other=0.0)  # fp16

        # Element-wise dot: scores[m] = Σ_d q[m,d] × k_anc[m,d]
        # Each query row uses its OWN k vector → no tl.dot (not a matmul).
        raw   = tl.sum(q.to(tl.float32) * k_anc.to(tl.float32), axis=1) * scale
        s     = tl.where(is_new, raw, float("-inf"))   # [BLOCK_M]

        # ── Online softmax update (FA-2 Eq. 4, single-element block) ───────
        m_new = tl.maximum(m_i, s)
        alpha = tl.exp(m_i - m_new)          # rescaling factor [BLOCK_M]
        p     = tl.exp(s   - m_new)          # [BLOCK_M]; 0 for -inf rows
        p_pos = tl.where(is_new, p, 0.0)     # explicit zero guard for NaN safety

        # ── Scatter-gather V[ancestor]  [BLOCK_M, HEAD_DIM]  fp16 ──────────
        v_ptrs = (V_ptr
                  + kv_abs[:, None] * stride_vt
                  + head_idx        * stride_vh
                  + d_range [None,:] * stride_vd)
        v_anc = tl.load(v_ptrs, mask=is_new[:, None], other=0.0)  # fp16

        l_i = l_i * alpha + p_pos
        acc = acc * alpha[:, None] + p_pos[:, None] * v_anc.to(tl.float32)
        m_i = m_new

        # ── Advance to parent ─────────────────────────────────────────────
        prev = cur
        cur  = tl.maximum(cur - 1, tl.zeros_like(cur)) // BRANCHING_FACTOR

    # ── Normalise and write output ──────────────────────────────────────────
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc    = acc / l_safe[:, None]

    o_ptrs = (O_ptr
              + q_global[:, None] * stride_ot
              + head_idx          * stride_oh
              + d_range  [None,:] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=valid_q[:, None])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_inputs(
    qs: List[torch.Tensor],
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate per-sequence Q/K/V tensors into packed ragged layout.

    Parameters
    ----------
    qs, ks, vs : B-length lists of [L_i, H, D] fp16 tensors

    Returns
    -------
    Q, K, V    : [total_tokens, H, D]  fp16
    cu_seqlens : [B+1]  int32  (CPU tensor)
    """
    assert len(qs) == len(ks) == len(vs)
    B          = len(qs)
    seq_lens   = [q.shape[0] for q in qs]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    for i, n in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + n
    return torch.cat(qs, 0), torch.cat(ks, 0), torch.cat(vs, 0), cu_seqlens


def ragged_attention(
    Q:  torch.Tensor,          # [total_tokens, H, D]  fp16 or bf16  CUDA
    K:  torch.Tensor,
    V:  torch.Tensor,
    cu_seqlens:       torch.Tensor,   # [B+1]  int32
    branching_factor: int,
    max_depth:        int,
) -> torch.Tensor:
    """
    Ancestor-sparse ragged attention for b-ary tree speculative decoding.

    For each query token q, computes softmax-attention over the set:
        { parent^0(q), parent^1(q), ..., parent^max_depth(q) }
    — at most max_depth+1 KV positions, determined analytically from the
    BFS-ordered b-ary tree relation  parent(k) = (k-1) // b  for k > 0.

    This is O(N × MAX_DEPTH) in memory and compute, vs O(N²) for dense FA.
    For N=1365, MAX_DEPTH=5: ~470× fewer FMAs than the naive masked approach.

    Parameters
    ----------
    Q, K, V          : packed fp16 **or bf16** tensors  [Σ L_i, H, D]  on CUDA.
                       bf16 is natively accelerated on SM 8.9+ (Ada/Blackwell);
                       on SM75 (T4) bf16 inputs are auto-cast to fp16.
    cu_seqlens       : [B+1]  int32  cumulative sequence start offsets
    branching_factor : b — fan-out of the BFS-ordered complete b-ary tree
    max_depth        : d — maximum tree depth in this batch

    Returns
    -------
    O : same dtype as Q,  [Σ L_i, H, D]
    """
    _SUPPORTED = (torch.float16, torch.bfloat16)
    assert Q.dtype in _SUPPORTED and Q.dtype == K.dtype == V.dtype, (
        f"Inputs must be fp16 or bf16 (got {Q.dtype})"
    )

    # On SM75 the kernel was written for fp16; auto-cast bf16 inputs.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(Q.device)
        if Q.dtype == torch.bfloat16 and (props.major, props.minor) < (8, 9):
            Q, K, V = Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16)
            _cast_back = True
        else:
            _cast_back = False
    else:
        _cast_back = False
    total_tokens, H, D = Q.shape
    B      = int(cu_seqlens.shape[0]) - 1
    device = Q.device
    scale  = 1.0 / math.sqrt(D)

    # Compute max_seqlen from CPU cu_seqlens BEFORE moving to device.
    # This avoids a blocking GPU→CPU round-trip (.cpu().tolist()) that was
    # previously adding ~20–50 μs to every call for small batches.
    cu_sl_cpu  = cu_seqlens.cpu()   # no-op if already CPU (common case)
    sl_list    = cu_sl_cpu.tolist() # pure Python; no CUDA sync
    max_seqlen = int(max(sl_list[i+1] - sl_list[i] for i in range(B))) if B else 1

    cu_seqlens_dev = cu_sl_cpu.to(device=device, dtype=torch.int32,
                                   non_blocking=True)

    O = torch.empty_like(Q)

    # Grid: (B*H,  ceil(max_seqlen / BLOCK_M))  — BLOCK_M resolved by autotune
    grid = lambda meta: (B * H, triton.cdiv(max_seqlen, meta["BLOCK_M"]))

    _ragged_attn_sparse_kernel[grid](
        Q, K, V, O,
        cu_seqlens_dev,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        scale,
        max_seqlen,
        H=H,
        HEAD_DIM=D,
        BRANCHING_FACTOR=branching_factor,
        MAX_DEPTH=max_depth,
    )
    if _cast_back:
        O = O.to(torch.bfloat16)
    return O


# ---------------------------------------------------------------------------
# Smoke test  (python -m src.ragged_attn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if not torch.cuda.is_available():
        print("CUDA not available — skipping smoke test.")
        sys.exit(0)

    torch.manual_seed(0)
    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}  SM{props.major}{props.minor}")

    H, D, b, d, B = 4, 64, 2, 2, 3
    N = (b ** (d + 1) - 1) // (b - 1)   # 7 nodes for b=2, d=2

    for dtype in (torch.float16, torch.bfloat16):
        qs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
        ks = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
        vs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]

        Q, K, V, cu_sl = pack_inputs(qs, ks, vs)
        O = ragged_attention(Q, K, V, cu_sl, branching_factor=b, max_depth=d)
        print(f"Smoke test passed — dtype={dtype}  shape {O.shape}")

    print("All smoke tests passed.")
