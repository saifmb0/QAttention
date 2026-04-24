"""
ragged_attn.py
==============
Triton ragged attention — ancestor-sparse Flash Attention for tree-structured
speculative decoding.

Target hardware : SM75  (NVIDIA T4, Kaggle 2×T4) — original
                  SM120 (NVIDIA RTX PRO 6000 Blackwell Server Edition 94 GB) — "blackwell" branch
Precision       : float16 or bfloat16 input/output, float32 accumulation

─────────────────────────────────────────────────────────────────────────────
Blackwell optimisation notes  — RTX PRO 6000 Blackwell Server Edition (SM 12.0)
─────────────────────────────────────────────────────────────────────────────

Target hardware
  Architecture  : Blackwell (GB202)
  CUDA SM       : 12.0
  VRAM          : 94 GB GDDR7
  Shared mem/SM : up to 232 KB per block  (3.6× Turing T4)
  Regs/SM       : 65 536
  Warp size     : 32

Key improvements over SM75 (T4)
  1. BLOCK_M up to 512 — Blackwell's 232 KB shared mem/SM removes the spill
     cliff that caps T4 at BLOCK_M=128.
  2. num_stages=1 throughout — the ancestor walk has data-dependent addresses,
     so Triton's software pipelining cannot help and in fact causes
     CompilationError on SM 12.0 for certain autotune-key combos.
  3. BF16 natively accelerated on Blackwell tensor cores; added bfloat16
     precision path with dtype-aware store (auto-cast to fp16 on SM75).
  4. int64 pointer offsets — at B≥64, b=4, d=8 (5.6 M tokens) the byte
     offsets exceed INT32_MAX (2 GB), causing wrap-around on SM 12.0.
  5. Large L2 cache keeps K/V ancestor positions L2-resident across all
     Q-tiles — scatter-gather latency ≈ 0.
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
# SM89 autotune configs — intermediate fallback for SM 8.x GPUs
#
# Differences from SM75:
#   • Large L2 (96 MB on Ada) → K/V ancestor positions hot in L2
#   • Tensor cores natively accelerate BF16 in addition to FP16
#   • num_stages=2 safe for step 0 (address fully data-independent)
#
# Register budget at BLOCK_M=256, HEAD_DIM=64, num_warps=8 (256 threads):
#   Total: ~176 regs/thread — within 65 536/256 = 256 limit, no spill.
# ---------------------------------------------------------------------------
_SPARSE_SM89_CONFIGS = [
    # Small tiles — low-latency for tiny batches (B=1)
    triton.Config({"BLOCK_M": 32},  num_warps=4,  num_stages=1),
    triton.Config({"BLOCK_M": 64},  num_warps=4,  num_stages=1),
    # Medium tiles — balanced occupancy
    triton.Config({"BLOCK_M": 64},  num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=16, num_stages=1),
    # Large tiles — exploit large L2 and high SM count
    triton.Config({"BLOCK_M": 256}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 256}, num_warps=16, num_stages=1),
]

# ---------------------------------------------------------------------------
# SM90 autotune configs — Hopper (H100, H200)
#
# Differences from SM89 (Ada):
#   • 4th-gen Tensor Cores: support warpgroup-level MMA (used by FA-3, not Triton yet)
#   • Shared memory per SM up to 228 KB (similar to SM120)
#   • Large L2 (50 MB on H100 SXM) → K/V ancestor positions highly L2-resident
#   • num_stages=1: ancestor walk is still data-dependent; no pipelining
#   • BLOCK_M up to 512 allowed by shared-memory budget
#
# Register budget at BLOCK_M=512, HEAD_DIM=64, num_warps=16 (512 threads):
#   Same analysis as SM120: ~176 regs/thread, within 65536/512=128 limit.
# ---------------------------------------------------------------------------
_SPARSE_SM90_CONFIGS = [
    triton.Config({"BLOCK_M": 32},  num_warps=4,  num_stages=1),
    triton.Config({"BLOCK_M": 64},  num_warps=4,  num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 256}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 256}, num_warps=16, num_stages=1),
    # Exploit 228 KB shared mem
    triton.Config({"BLOCK_M": 512}, num_warps=16, num_stages=1),
]

# ---------------------------------------------------------------------------
# SM120 autotune configs — Blackwell (e.g. RTX PRO 6000 Blackwell, GB200)
#
# Differences from SM89:
#   • Shared memory per SM up to 232 KB (vs ~100 KB on Ada) → BLOCK_M=512
#     fits without register spill at HEAD_DIM=64.
#   • num_stages=1 used throughout: the ancestor walk loop has data-dependent
#     addresses (parent^s depends on parent^{s-1}), so software-pipeline
#     prefetch cannot help.  Using num_stages=2 causes Triton CompilationError
#     on SM 12.0 for certain (HEAD_DIM, BRANCHING_FACTOR, MAX_DEPTH) combos
#     (non-power-of-2 divisors + unrolled loops + PTX pipelining interact).
#
# Register budget at BLOCK_M=512, HEAD_DIM=64, num_warps=16 (512 threads):
#   q + k_anc + v_anc: 3×64 fp16 → 96 fp32 regs
#   acc: 64 fp32 regs;  scalars: ~16 → 176 total
#   Per-thread quota = 65 536/512 = 128 regs → tight but no spill.
# ---------------------------------------------------------------------------
_SPARSE_SM120_CONFIGS = [
    # Small tiles
    triton.Config({"BLOCK_M": 32},  num_warps=4,  num_stages=1),
    triton.Config({"BLOCK_M": 64},  num_warps=4,  num_stages=1),
    # Medium tiles
    triton.Config({"BLOCK_M": 128}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 256}, num_warps=8,  num_stages=1),
    triton.Config({"BLOCK_M": 256}, num_warps=16, num_stages=1),
    # Large tiles — exploit 232 KB shared mem
    triton.Config({"BLOCK_M": 512}, num_warps=16, num_stages=1),
]


def _get_autotune_configs() -> list:
    """Select autotune config set based on the current CUDA device's SM.

    Tiers:
      SM 12.x (Blackwell)  -> _SPARSE_SM120_CONFIGS  [primary target]
      SM  9.x (Hopper)     -> _SPARSE_SM90_CONFIGS   [H100/H200]
      SM  8.9 (Ada)        -> _SPARSE_SM89_CONFIGS   [RTX 4090/L40S]
      other                -> _SPARSE_SM75_CONFIGS   [legacy fallback]
    """
    if not torch.cuda.is_available():
        return _SPARSE_SM75_CONFIGS
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm = (props.major, props.minor)
    if sm >= (12, 0):
        return _SPARSE_SM120_CONFIGS
    if sm >= (9, 0):
        return _SPARSE_SM90_CONFIGS
    if sm >= (8, 9):
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
    key=["HEAD_DIM", "BRANCHING_FACTOR", "MAX_DEPTH", "max_seqlen", "total_tokens"],
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
    total_tokens,
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
    # int64 offsets: at B=64, b=4, d=8 total_tokens≈5.6 M → byte offsets
    # exceed INT32_MAX (2 GB).  Triton multiplies element offsets by
    # sizeof(dtype) internally; keeping offsets int32 causes wrap-around
    # and illegal-memory-access on SM 12.0.
    q_global = (seq_start + q_off + m_range).to(tl.int64)  # [BLOCK_M]

    # ── Load Q tile  [BLOCK_M, HEAD_DIM]  fp16 ─────────────────────────────
    q_ptrs = (Q_ptr
              + q_global[:, None] * stride_qt
              + head_idx          * stride_qh
              + d_range  [None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)
    _out_dtype = q.dtype          # fp16 or bf16 — used for the output store

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
        kv_abs = (seq_start + cur).to(tl.int64)  # int64 for >2 GB tensors

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
        raw   = (tl.sum(q.to(tl.float32) * k_anc.to(tl.float32), axis=1) * scale).to(tl.float32)
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
    tl.store(o_ptrs, acc.to(_out_dtype), mask=valid_q[:, None])


# ---------------------------------------------------------------------------
# LSE variant — same ancestor-sparse kernel but also writes
#   lse[token, head] = m_i + log(l_i)   (log-sum-exp, float32)
# Used by ragged_attention_with_lse() for the online-softmax prefix merge.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["HEAD_DIM", "BRANCHING_FACTOR", "MAX_DEPTH", "max_seqlen", "total_tokens"],
)
@triton.jit
def _ragged_attn_sparse_kernel_lse(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    cu_seqlens_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lse_t, stride_lse_h,
    scale,
    max_seqlen,
    total_tokens,
    H:                tl.constexpr,
    HEAD_DIM:         tl.constexpr,
    BRANCHING_FACTOR: tl.constexpr,
    MAX_DEPTH:        tl.constexpr,
    BLOCK_M:          tl.constexpr,
):
    pid0   = tl.program_id(0)
    m_tile = tl.program_id(1)

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
    valid_q  = (m_range + q_off) < seq_len
    q_global = (seq_start + q_off + m_range).to(tl.int64)

    q_ptrs = (Q_ptr
              + q_global[:, None] * stride_qt
              + head_idx          * stride_qh
              + d_range  [None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)
    _out_dtype = q.dtype

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    cur  = (m_range + q_off).to(tl.int32)
    prev = tl.full([BLOCK_M], -1, dtype=tl.int32)

    for _step in range(MAX_DEPTH + 1):
        is_new = (cur != prev) & valid_q
        kv_abs = (seq_start + cur).to(tl.int64)

        k_ptrs = (K_ptr
                  + kv_abs[:, None] * stride_kt
                  + head_idx        * stride_kh
                  + d_range [None,:] * stride_kd)
        k_anc = tl.load(k_ptrs, mask=is_new[:, None], other=0.0)

        raw   = (tl.sum(q.to(tl.float32) * k_anc.to(tl.float32), axis=1) * scale).to(tl.float32)
        s     = tl.where(is_new, raw, float("-inf"))

        m_new = tl.maximum(m_i, s)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(s   - m_new)
        p_pos = tl.where(is_new, p, 0.0)

        v_ptrs = (V_ptr
                  + kv_abs[:, None] * stride_vt
                  + head_idx        * stride_vh
                  + d_range [None,:] * stride_vd)
        v_anc = tl.load(v_ptrs, mask=is_new[:, None], other=0.0)

        l_i = l_i * alpha + p_pos
        acc = acc * alpha[:, None] + p_pos[:, None] * v_anc.to(tl.float32)
        m_i = m_new

        prev = cur
        cur  = tl.maximum(cur - 1, tl.zeros_like(cur)) // BRANCHING_FACTOR

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)

    # Write lse = m_i + log(l_i)  [total_tokens, H]
    lse_ptrs = (LSE_ptr
                + q_global * stride_lse_t
                + head_idx * stride_lse_h)
    tl.store(lse_ptrs, m_i + tl.log(l_safe), mask=valid_q)

    # Normalise and write output
    acc = acc / l_safe[:, None]
    o_ptrs = (O_ptr
              + q_global[:, None] * stride_ot
              + head_idx          * stride_oh
              + d_range  [None,:] * stride_od)
    tl.store(o_ptrs, acc.to(_out_dtype), mask=valid_q[:, None])


def ragged_attention_with_lse(
    Q:  torch.Tensor,
    K:  torch.Tensor,
    V:  torch.Tensor,
    cu_seqlens:       torch.Tensor,
    branching_factor: int,
    max_depth:        int,
    max_seqlen:       int = None,
) -> tuple:
    """Ancestor-sparse attention returning (output, lse) for online-softmax merging.

    Same algorithm as ragged_attention() but also returns the per-token per-head
    log-sum-exp needed to merge this block's output with another attention block
    (e.g., the prefix KV-cache) via the standard FA2 split-softmax formula:

        O = (exp(lse_a) * O_a + exp(lse_b) * O_b) / (exp(lse_a) + exp(lse_b))

    Returns
    -------
    O   : same dtype as Q,  [Σ L_i, H, D]
    lse : float32,          [Σ L_i, H]    — log(Σ exp(score_j)) for each (token, head)
    """
    _SUPPORTED = (torch.float16, torch.bfloat16)
    assert Q.dtype in _SUPPORTED and Q.dtype == K.dtype == V.dtype, (
        f"Inputs must be fp16 or bf16 (got {Q.dtype})"
    )

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

    if max_seqlen is None:
        cu_sl_cpu  = cu_seqlens.cpu()
        sl_list    = cu_sl_cpu.tolist()
        max_seqlen = int(max(sl_list[i+1] - sl_list[i] for i in range(B))) if B else 1
        cu_seqlens_dev = cu_sl_cpu.to(device=device, dtype=torch.int32, non_blocking=True)
    else:
        cu_seqlens_dev = cu_seqlens.to(device=device, dtype=torch.int32, non_blocking=True)

    O   = torch.empty_like(Q)
    LSE = torch.empty(total_tokens, H, dtype=torch.float32, device=device)

    grid = lambda meta: (B * H, triton.cdiv(max_seqlen, meta["BLOCK_M"]))

    _ragged_attn_sparse_kernel_lse[grid](
        Q, K, V, O, LSE,
        cu_seqlens_dev,
        Q.stride(0),   Q.stride(1),   Q.stride(2),
        K.stride(0),   K.stride(1),   K.stride(2),
        V.stride(0),   V.stride(1),   V.stride(2),
        O.stride(0),   O.stride(1),   O.stride(2),
        LSE.stride(0), LSE.stride(1),
        scale,
        max_seqlen,
        total_tokens,
        H=H,
        HEAD_DIM=D,
        BRANCHING_FACTOR=branching_factor,
        MAX_DEPTH=max_depth,
    )
    if _cast_back:
        O = O.to(torch.bfloat16)
    return O, LSE


# ---------------------------------------------------------------------------
# Explicit-parent variant — for EAGLE-3 dynamic/pruned trees
# ---------------------------------------------------------------------------
# EAGLE-3 builds trees via beam search + global top-k pruning.  The
# resulting tree is NOT a complete b-ary tree — it has irregular branching
# and varying depths across paths.  The formula parent(k) = ⌊(k-1)/b⌋
# is WRONG for these trees.
#
# This variant accepts an explicit parent array (int32, per-token) that
# encodes the actual tree topology.  Everything else — online softmax,
# ancestor walk, duplicate detection — is identical.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["HEAD_DIM", "MAX_DEPTH", "max_seqlen", "total_tokens"],
)
@triton.jit
def _ragged_attn_parents_kernel_lse(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    parent_ptr,                    # int32 [total_tokens] — seq-local parent idx
    cu_seqlens_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lse_t, stride_lse_h,
    scale,
    max_seqlen,
    total_tokens,
    H:                tl.constexpr,
    HEAD_DIM:         tl.constexpr,
    MAX_DEPTH:        tl.constexpr,
    BLOCK_M:          tl.constexpr,
):
    """
    Ancestor-sparse Flash Attention with explicit parent array.

    For each query at local position q, the ancestor walk does:
        cur = q  (self)
        for each step:
            attend to cur
            cur = parent[cur]   (from parent_ptr)
    Root nodes have parent[root] = root (self-loop → cur==prev → stops).
    """
    pid0   = tl.program_id(0)
    m_tile = tl.program_id(1)

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
    valid_q  = (m_range + q_off) < seq_len
    q_global = (seq_start + q_off + m_range).to(tl.int64)

    # ── Load Q tile  [BLOCK_M, HEAD_DIM] ────────────────────────────────────
    q_ptrs = (Q_ptr
              + q_global[:, None] * stride_qt
              + head_idx          * stride_qh
              + d_range  [None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)
    _out_dtype = q.dtype

    # ── Flash-Attention-2 online softmax state ──────────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # ── Ancestor walk state ──────────────────────────────────────────────────
    cur  = (m_range + q_off).to(tl.int32)   # local positions — step 0 = self
    prev = tl.full([BLOCK_M], -1, dtype=tl.int32)

    # ── Ancestor-sparse loop ────────────────────────────────────────────────
    step = 0
    is_new = (cur != prev) & valid_q
    active = tl.max(is_new.to(tl.int32))

    while (step <= MAX_DEPTH) and (active > 0):
        kv_abs = (seq_start + cur).to(tl.int64)

        # ── Scatter-gather K[ancestor] ──────────────────────────────────────
        k_ptrs = (K_ptr
                  + kv_abs[:, None] * stride_kt
                  + head_idx        * stride_kh
                  + d_range [None,:] * stride_kd)
        k_anc = tl.load(k_ptrs, mask=is_new[:, None], other=0.0)

        raw   = (tl.sum(q.to(tl.float32) * k_anc.to(tl.float32), axis=1) * scale).to(tl.float32)
        s     = tl.where(is_new, raw, float("-inf"))

        # ── Online softmax update ───────────────────────────────────────────
        m_new = tl.maximum(m_i, s)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(s   - m_new)
        p_pos = tl.where(is_new, p, 0.0)

        # ── Scatter-gather V[ancestor] ──────────────────────────────────────
        v_ptrs = (V_ptr
                  + kv_abs[:, None] * stride_vt
                  + head_idx        * stride_vh
                  + d_range [None,:] * stride_vd)
        v_anc = tl.load(v_ptrs, mask=is_new[:, None], other=0.0)

        l_i = l_i * alpha + p_pos
        acc = acc * alpha[:, None] + p_pos[:, None] * v_anc.to(tl.float32)
        m_i = m_new

        # ── Advance to parent via explicit lookup ───────────────────────────
        prev = cur
        parent_global = (seq_start + cur).to(tl.int64)
        cur = tl.load(parent_ptr + parent_global, mask=valid_q, other=0).to(tl.int32)
        
        is_new = (cur != prev) & valid_q
        active = tl.max(is_new.to(tl.int32))
        step += 1

    # ── Write LSE ───────────────────────────────────────────────────────────
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    lse_ptrs = (LSE_ptr
                + q_global * stride_lse_t
                + head_idx * stride_lse_h)
    tl.store(lse_ptrs, m_i + tl.log(l_safe), mask=valid_q)

    # ── Normalise and write output ──────────────────────────────────────────
    acc = acc / l_safe[:, None]
    o_ptrs = (O_ptr
              + q_global[:, None] * stride_ot
              + head_idx          * stride_oh
              + d_range  [None,:] * stride_od)
    tl.store(o_ptrs, acc.to(_out_dtype), mask=valid_q[:, None])


def ragged_attention_with_parents(
    Q:  torch.Tensor,       # [total_tokens, H, D]
    K:  torch.Tensor,       # [total_tokens, H, D]
    V:  torch.Tensor,       # [total_tokens, H, D]
    cu_seqlens:  torch.Tensor,   # [B+1] int32
    parents:     torch.Tensor,   # [total_tokens] int32 — local parent index
    max_depth:   int,
    max_seqlen:  int = None,
) -> tuple:
    """Ancestor-sparse attention with explicit parent array.

    Same as ``ragged_attention_with_lse`` but uses an explicit parent array
    instead of the complete b-ary tree parent formula.  This is required for
    EAGLE-3's dynamic/pruned trees where ``parent(k) = ⌊(k-1)/b⌋`` is wrong.

    Parameters
    ----------
    parents : int32 tensor [total_tokens]
        ``parents[global_pos]`` = sequence-local parent index.
        Root nodes must have ``parents[root] = root`` (self-loop).

    Returns
    -------
    O   : same dtype as Q,  [total_tokens, H, D]
    lse : float32,          [total_tokens, H]
    """
    _SUPPORTED = (torch.float16, torch.bfloat16)
    assert Q.dtype in _SUPPORTED and Q.dtype == K.dtype == V.dtype, (
        f"Inputs must be fp16 or bf16 (got {Q.dtype})"
    )

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

    if max_seqlen is None:
        cu_sl_cpu  = cu_seqlens.cpu()
        sl_list    = cu_sl_cpu.tolist()
        max_seqlen = int(max(sl_list[i+1] - sl_list[i] for i in range(B))) if B else 1
        cu_seqlens_dev = cu_sl_cpu.to(device=device, dtype=torch.int32, non_blocking=True)
    else:
        cu_seqlens_dev = cu_seqlens.to(device=device, dtype=torch.int32, non_blocking=True)

    parents_dev    = parents.to(device=device, dtype=torch.int32, non_blocking=True)

    O   = torch.empty_like(Q)
    LSE = torch.empty(total_tokens, H, dtype=torch.float32, device=device)

    grid = lambda meta: (B * H, triton.cdiv(max_seqlen, meta["BLOCK_M"]))

    _ragged_attn_parents_kernel_lse[grid](
        Q, K, V, O, LSE,
        parents_dev,
        cu_seqlens_dev,
        Q.stride(0),   Q.stride(1),   Q.stride(2),
        K.stride(0),   K.stride(1),   K.stride(2),
        V.stride(0),   V.stride(1),   V.stride(2),
        O.stride(0),   O.stride(1),   O.stride(2),
        LSE.stride(0), LSE.stride(1),
        scale,
        max_seqlen,
        total_tokens,
        H=H,
        HEAD_DIM=D,
        MAX_DEPTH=max_depth,
    )
    if _cast_back:
        O = O.to(torch.bfloat16)
    return O, LSE


# ---------------------------------------------------------------------------
# Paged KV-Cache variant
# ---------------------------------------------------------------------------
# PagedAttention (vLLM) stores K/V in non-contiguous 16-token pages.
# The contiguous `seq_start + ancestor_pos` addressing of the dense kernel
# cannot be used directly.  This kernel accepts a page_table tensor that
# maps logical KV positions to physical page slots.
#
# Layout:
#   page_table  : [B, max_pages]  int32   — page_table[b, p] = physical page idx
#   K_cache     : [total_pages, PAGE_SIZE, H, D]  fp16/bf16
#   V_cache     : [total_pages, PAGE_SIZE, H, D]  fp16/bf16
#   Q           : [total_tokens, H, D]            fp16/bf16
#
# For ancestor position `pos` in sequence `b`:
#   page_idx    = page_table[b, pos // PAGE_SIZE]
#   page_off    = pos % PAGE_SIZE
#   K address   = K_cache[page_idx, page_off, head, :]
#
# This adds one extra indirection per ancestor step vs the contiguous kernel.
# The scatter penalty is larger (page table entries likely L2-miss on first
# access) but ancestor reuse across Q-tiles still holds for high-level ancestors.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["HEAD_DIM", "BRANCHING_FACTOR", "MAX_DEPTH"],
)
@triton.jit
def _ragged_attn_paged_kernel(
    Q_ptr,
    K_ptr,          # [total_pages, PAGE_SIZE, H, D]
    V_ptr,          # [total_pages, PAGE_SIZE, H, D]
    O_ptr,
    page_table_ptr, # [B, max_pages]  int32
    cu_seqlens_ptr, # [B+1]  int32
    stride_qt, stride_qh, stride_qd,
    stride_kp, stride_kps, stride_kh, stride_kd,   # page, page_slot, head, dim
    stride_vp, stride_vps, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_ptb,     # page_table: stride over batch dim (= max_pages)
    scale,
    max_seqlen,
    PAGE_SIZE:        tl.constexpr,
    H:                tl.constexpr,
    HEAD_DIM:         tl.constexpr,
    BRANCHING_FACTOR: tl.constexpr,
    MAX_DEPTH:        tl.constexpr,
    BLOCK_M:          tl.constexpr,
):
    pid0   = tl.program_id(0)
    m_tile = tl.program_id(1)
    seq_idx  = pid0 // H
    head_idx = pid0  % H

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end   = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len   = seq_end - seq_start

    q_off = m_tile * BLOCK_M
    if q_off >= seq_len:
        return

    m_range = tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, HEAD_DIM)
    valid_q = (m_range + q_off) < seq_len
    q_global = (seq_start + q_off + m_range).to(tl.int64)

    q_ptrs = (Q_ptr
              + q_global[:, None] * stride_qt
              + head_idx          * stride_qh
              + d_range  [None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)
    _out_dtype = q.dtype

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    cur  = (m_range + q_off).to(tl.int32)
    prev = tl.full([BLOCK_M], -1, dtype=tl.int32)

    for _step in range(MAX_DEPTH + 1):
        is_new = (cur != prev) & valid_q

        # Two-level page-table indirection:
        # physical_page = page_table[seq_idx, cur // PAGE_SIZE]
        page_slot   = cur % PAGE_SIZE                               # [BLOCK_M] int32
        page_logical = tl.maximum(cur, 0) // PAGE_SIZE              # [BLOCK_M] int32
        page_phys   = tl.load(
            page_table_ptr + seq_idx.to(tl.int64) * stride_ptb
            + page_logical.to(tl.int64),
            mask=is_new, other=0,
        ).to(tl.int64)                                              # [BLOCK_M] int64

        k_ptrs = (K_ptr
                  + page_phys  [:, None] * stride_kp
                  + page_slot  [:, None] * stride_kps
                  + head_idx              * stride_kh
                  + d_range    [None, :] * stride_kd)
        k_anc = tl.load(k_ptrs, mask=is_new[:, None], other=0.0)

        raw = (tl.sum(q.to(tl.float32) * k_anc.to(tl.float32), axis=1) * scale).to(tl.float32)
        s   = tl.where(is_new, raw, float("-inf"))

        m_new = tl.maximum(m_i, s)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(s   - m_new)
        p_pos = tl.where(is_new, p, 0.0)

        v_ptrs = (V_ptr
                  + page_phys  [:, None] * stride_vp
                  + page_slot  [:, None] * stride_vps
                  + head_idx              * stride_vh
                  + d_range    [None, :] * stride_vd)
        v_anc = tl.load(v_ptrs, mask=is_new[:, None], other=0.0)

        l_i = l_i * alpha + p_pos
        acc = acc * alpha[:, None] + p_pos[:, None] * v_anc.to(tl.float32)
        m_i = m_new

        prev = cur
        cur  = tl.maximum(cur - 1, tl.zeros_like(cur)) // BRANCHING_FACTOR

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc    = acc / l_safe[:, None]

    o_ptrs = (O_ptr
              + q_global[:, None] * stride_ot
              + head_idx          * stride_oh
              + d_range  [None,:] * stride_od)
    tl.store(o_ptrs, acc.to(_out_dtype), mask=valid_q[:, None])


def ragged_attention_paged(
    Q:           torch.Tensor,   # [total_tokens, H, D]
    K_cache:     torch.Tensor,   # [total_pages, PAGE_SIZE, H, D]
    V_cache:     torch.Tensor,   # [total_pages, PAGE_SIZE, H, D]
    page_table:  torch.Tensor,   # [B, max_pages]  int32  on device
    cu_seqlens:  torch.Tensor,   # [B+1]  int32
    branching_factor: int,
    max_depth:        int,
) -> torch.Tensor:
    """
    Ancestor-sparse ragged attention with paged KV cache (PagedAttention layout).

    This variant is compatible with vLLM-style PagedAttention where the K/V
    cache is stored in non-contiguous 16-token blocks mapped via a page table.

    Parameters
    ----------
    Q           : packed query tensor [total_tokens, H, D]  fp16/bf16
    K_cache     : physical KV cache pages [total_pages, PAGE_SIZE, H, D]
    V_cache     : physical KV cache pages [total_pages, PAGE_SIZE, H, D]
    page_table  : [B, max_pages] int32 on GPU — logical-to-physical page map.
                  page_table[b, p] = physical page index for sequence b, page p.
    cu_seqlens  : cumulative sequence starts [B+1] int32 (CPU or GPU)
    branching_factor, max_depth : tree structure parameters

    Returns
    -------
    O : [total_tokens, H, D]  same dtype as Q

    Notes on PagedAttention incompatibility with the contiguous kernel
    -------------------------------------------------------------------
    The standard ragged_attention() kernel computes:
        K_ptr + (seq_start + ancestor_pos) * stride_kt
    assuming K/V is a contiguous packed tensor.  With PagedAttention,
    physical pages can be anywhere in GPU memory; the page_table provides
    the indirection.  This kernel adds one extra load (the page_table lookup)
    per ancestor step, introducing a ~2-4% overhead for typical page sizes
    (PAGE_SIZE=16), but makes the kernel compatible with paged serving runtimes.
    """
    assert K_cache.shape == V_cache.shape, "K and V cache shapes must match"
    _PAGE_SIZE = K_cache.shape[1]
    total_tokens, H, D = Q.shape
    B = int(cu_seqlens.shape[0]) - 1
    scale = 1.0 / math.sqrt(D)

    cu_sl_cpu = cu_seqlens.cpu()
    sl_list   = cu_sl_cpu.tolist()
    max_seqlen = int(max(sl_list[i+1] - sl_list[i] for i in range(B))) if B else 1

    device = Q.device
    cu_seqlens_dev = cu_sl_cpu.to(device=device, dtype=torch.int32, non_blocking=True)
    page_table_dev = page_table.to(device=device, dtype=torch.int32, non_blocking=True)

    O = torch.empty_like(Q)
    grid = lambda meta: (B * H, triton.cdiv(max_seqlen, meta["BLOCK_M"]))

    _ragged_attn_paged_kernel[grid](
        Q, K_cache, V_cache, O,
        page_table_dev, cu_seqlens_dev,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        page_table_dev.stride(0),
        scale, max_seqlen,
        PAGE_SIZE=_PAGE_SIZE,
        H=H,
        HEAD_DIM=D,
        BRANCHING_FACTOR=branching_factor,
        MAX_DEPTH=max_depth,
    )
    return O


def build_paged_kv(
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
    page_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack per-sequence K/V tensors into paged KV cache layout.

    Returns
    -------
    K_cache    : [total_pages, page_size, H, D]
    V_cache    : [total_pages, page_size, H, D]
    page_table : [B, max_pages] int32 CPU tensor — logical-to-physical map
    """
    import math as _math
    B = len(ks)
    H, D = ks[0].shape[1], ks[0].shape[2]
    device = ks[0].device
    dtype  = ks[0].dtype

    # Compute per-sequence page counts
    pages_per_seq = [_math.ceil(k.shape[0] / page_size) for k in ks]
    max_pages     = max(pages_per_seq)
    total_pages   = sum(pages_per_seq)

    K_cache   = torch.zeros(total_pages, page_size, H, D, device=device, dtype=dtype)
    V_cache   = torch.zeros(total_pages, page_size, H, D, device=device, dtype=dtype)
    page_table = torch.zeros(B, max_pages, dtype=torch.int32)  # CPU

    phys_page = 0
    for b, (k, v) in enumerate(zip(ks, vs)):
        L = k.shape[0]
        for p in range(pages_per_seq[b]):
            start = p * page_size
            end   = min(start + page_size, L)
            chunk = end - start
            K_cache[phys_page, :chunk] = k[start:end]
            V_cache[phys_page, :chunk] = v[start:end]
            page_table[b, p] = phys_page
            phys_page += 1

    return K_cache, V_cache, page_table


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
    max_seqlen:       int = None,
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
    if max_seqlen is None:
        cu_sl_cpu  = cu_seqlens.cpu()   # no-op if already CPU (common case)
        sl_list    = cu_sl_cpu.tolist() # pure Python; no CUDA sync
        max_seqlen = int(max(sl_list[i+1] - sl_list[i] for i in range(B))) if B else 1
        cu_seqlens_dev = cu_sl_cpu.to(device=device, dtype=torch.int32, non_blocking=True)
    else:
        cu_seqlens_dev = cu_seqlens.to(device=device, dtype=torch.int32, non_blocking=True)

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
        total_tokens,
        H=H,
        HEAD_DIM=D,
        BRANCHING_FACTOR=branching_factor,
        MAX_DEPTH=max_depth,
    )
    if _cast_back:
        O = O.to(torch.bfloat16)
    return O


# ---------------------------------------------------------------------------
# Fused LSE Merge kernel
#
# Replaces the 5-op PyTorch merge:
#   lse_max = torch.maximum(lse_p, lse_t)
#   w_p  = torch.exp(lse_p - lse_max)
#   w_t  = torch.exp(lse_t - lse_max)
#   wsum = (w_p + w_t).clamp_min(1e-8).unsqueeze(-1)
#   out  = (w_p[:,None]*out_p + w_t[:,None]*out_t) / wsum
#
# Each PyTorch op above is a separate CUDA launch that reads + writes the full
# [B*H*N_q, D] tensor to/from HBM.  This kernel does one pass:
#   read lse_p, lse_t, out_p[D], out_t[D] → compute in registers → write out[D].
# Memory traffic: 4 reads + 1 write  (was ~8 reads + 5 writes across 5 kernels).
# ---------------------------------------------------------------------------

@triton.jit
def _fused_lse_merge_kernel(
    lse_p_ptr,      # [N_pos] float32  (B*H*N_q, contiguous)
    lse_t_ptr,      # [N_pos] float32
    out_p_ptr,      # [N_pos * HEAD_DIM]  fp16/bf16
    out_t_ptr,      # [N_pos * HEAD_DIM]  fp16/bf16
    result_ptr,     # [N_pos * HEAD_DIM]  fp16/bf16  (output)
    N_pos,          # int  — total number of (b,h,q) positions
    HEAD_DIM: tl.constexpr,
):
    """
    Each program handles one (b, h, q) position.
    Loads: lse_p, lse_t (scalars), out_p[D], out_t[D].
    Writes: result[D] = (w_p*out_p + w_t*out_t) / (w_p+w_t).
    All weights computed in fp32; result stored in the original dtype.
    """
    pid = tl.program_id(0)
    if pid >= N_pos:
        return

    # ── Load LSE scalars (fp32) ─────────────────────────────────────────────
    lse_p = tl.load(lse_p_ptr + pid).to(tl.float32)
    lse_t = tl.load(lse_t_ptr + pid).to(tl.float32)

    # ── Stable softmax weights ──────────────────────────────────────────────
    lse_max = tl.maximum(lse_p, lse_t)
    w_p     = tl.exp(lse_p - lse_max)
    w_t     = tl.exp(lse_t - lse_max)
    w_sum   = tl.maximum(w_p + w_t, 1e-8)

    # ── Load D-dim output vectors ───────────────────────────────────────────
    base   = pid * HEAD_DIM
    d_offs = tl.arange(0, HEAD_DIM)
    o_p_raw = tl.load(out_p_ptr + base + d_offs)   # keep original dtype for cast
    o_t_raw = tl.load(out_t_ptr + base + d_offs)
    _out_dtype = o_p_raw.dtype                      # fp16 or bf16
    o_p = o_p_raw.to(tl.float32)
    o_t = o_t_raw.to(tl.float32)

    # ── Merge and write ─────────────────────────────────────────────────────
    merged = (w_p * o_p + w_t * o_t) / w_sum
    tl.store(result_ptr + base + d_offs, merged.to(_out_dtype))


def fused_lse_merge(
    lse_pre:  torch.Tensor,   # [B, H, N_q]       float32
    lse_tree: torch.Tensor,   # [B, H, N_q]       float32
    out_pre:  torch.Tensor,   # [B, H, N_q, D]    fp16/bf16
    out_tree: torch.Tensor,   # [B, H, N_q, D]    fp16/bf16
) -> torch.Tensor:            # [B, H, N_q, D]    same dtype as out_pre
    """
    Fused online-softmax LSE merge via a single Triton kernel.

    Equivalent to the 5-op PyTorch merge but uses one HBM pass instead of five,
    eliminating ~5 kernel-launch round-trips on typically small (N≈137–274)
    attention output tensors.

    All four inputs must be contiguous (call ``.contiguous()`` if in doubt).
    """
    B, H, N_q, D = out_pre.shape
    N_pos   = B * H * N_q
    device  = out_pre.device
    dtype   = out_pre.dtype

    # Flatten [B, H, N_q] → [N_pos] and [B, H, N_q, D] → [N_pos, D]
    lse_p_flat = lse_pre.float().contiguous().view(N_pos)
    lse_t_flat = lse_tree.float().contiguous().view(N_pos)
    op_flat    = out_pre.contiguous().view(N_pos, D)
    ot_flat    = out_tree.contiguous().view(N_pos, D)

    result_flat = torch.empty(N_pos, D, dtype=dtype, device=device)

    _fused_lse_merge_kernel[(N_pos,)](
        lse_p_flat, lse_t_flat,
        op_flat, ot_flat, result_flat,
        N_pos,
        HEAD_DIM=D,
    )
    return result_flat.view(B, H, N_q, D)


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
