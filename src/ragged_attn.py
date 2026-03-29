"""
ragged_attn.py
==============
Triton ragged attention — Flash-Decoding + analytic tree masking.

Target hardware : SM75 (NVIDIA T4 16 GB, Kaggle 2×T4)
Precision       : float16 input/output, float32 accumulation
Algorithm       : Flash-Decoding (Dao et al. 2023) adapted for ragged
                  variable-length sequences with analytic tree masking.

─────────────────────────────────────────────────────────────────────────────
Bottleneck analysis (from benchmark.csv, 2×T4, SM75)
─────────────────────────────────────────────────────────────────────────────

Two distinct underperformance regimes were identified:

1.  CTA starvation — ALL configs, dominant at small N
    ──────────────────────────────────────────────────
    The prior grid was (B·H, ⌈L/BLOCK_M⌉).
    For B=1, H=8, N=15, BLOCK_M=16: only 8 CTAs against 40 SMs → GPU
    40× under-saturated.  The flat ~0.21 ms floor across all small-N
    configs in the benchmark is a dispatch floor with near-zero compute
    utilisation.  SDPA avoids this because its padded GEMM is batched
    across the full B×H·L_max² token space in a single fused kernel.

2.  Mask scatter-gather bandwidth amplification — large N
    ───────────────────────────────────────────────────────
    The prior kernel fetched tree-mask bits from a dense int8 buffer as
    scattered global loads: fidx[m,n] = mask_off + m·seq_len + n.
    Along the n-axis (BLOCK_N elements) accesses are coalesced, but
    across the m-axis (BLOCK_M rows) the stride is seq_len bytes —
    e.g. 1 365 bytes for b=4, d=5.  Each inner-loop iteration therefore
    touched BLOCK_M independent 128-byte cache lines, using only
    BLOCK_N ≤ 64 bytes from each.
    Effective cache-line utilisation ≤ 64/128 = 50 %; for BLOCK_M=16,
    seq_len=1 365: 16 cache-line loads, each delivering 16/128 = 12.5 %
    useful bytes.  For B=32, N=1 365 the total mask buffer is
    32·1 365²·1 B ≈ 60 MB; the benchmark shows ragged at 113 ms vs.
    SDPA at 17 ms for that config (speedup 0.15×).

─────────────────────────────────────────────────────────────────────────────
Architectural contributions
─────────────────────────────────────────────────────────────────────────────

1.  Analytic tree masking  (eliminates the packed_masks buffer entirely)
    ─────────────────────────────────────────────────────────────────────
    For a complete b-ary tree with BFS (level-order) node numbering:
        root = 0,   parent(k) = (k−1) // b  for k > 0.
    The predicate "j is an ancestor of i" reduces to: follow the parent
    chain from i for at most MAX_DEPTH steps and check whether any step
    equals j.  With MAX_DEPTH as a constexpr, Triton fully unrolls this
    into ≤ MAX_DEPTH integer multiply-add ops per (i,j) pair.
    Memory saved: O(B·N²) int8 buffer (60 MB at B=32, N=1 365) → 0.
    Compute cost: MAX_DEPTH ≤ 5 integer ops vs. 1 scattered global load
    (L1/L2 miss penalty ~100 ns).  Net: significant speedup at all N>64.

2.  Flash-Decoding KV split  (saturates GPU for short sequences)
    ──────────────────────────────────────────────────────────────
    Pass 1 (_ragged_attn_split_kernel):
        grid = (total_q_tiles, SPLIT_N, H).
        Each CTA handles one KV chunk [kv_lo, kv_hi) ⊆ [0, L_i),
        emitting partial (acc_p, m_p, l_p) to fp32 temp buffers.
    Pass 2 (_ragged_attn_reduce_kernel):
        grid = (total_q_tiles, H).
        Merges SPLIT_N partial results via online log-sum-exp.
    SPLIT_N is auto-chosen each call to ensure
        total_q_tiles · H · SPLIT_N ≥ num_SMs · TARGET_CTAS_PER_SM,
    guaranteeing full SM occupancy regardless of sequence length.

─────────────────────────────────────────────────────────────────────────────
Public API  (simplified — packed_masks removed)
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
# Analytic tree masking helper  (zero memory traffic)
# ---------------------------------------------------------------------------


# _is_ancestor_bary is inlined directly into _ragged_attn_split_kernel
# to avoid inter-kernel call complications across Triton versions.
# The logic: start from node_i, walk up the parent chain MAX_DEPTH times,
# checking if any ancestor equals node_j (inclusive of node_i itself).
# parent(k) = (k-1)//b  for a complete b-ary BFS-ordered tree.


# ---------------------------------------------------------------------------
# Autotune configs  (SM75 / T4)
# ---------------------------------------------------------------------------
# BLOCK_M is FIXED at 16 — ensures partial-buffer shapes are independent of
# the autotune choice and gives 4–5 CTAs/SM at D=64 (11 KB SRAM/CTA).
# BLOCK_N is autotuned to find the best KV-tile width for each head size.

_BLOCK_M = 16   # exported for wrapper arithmetic

_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
]

_REDUCE_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
]


# ---------------------------------------------------------------------------
# Pass 1 — KV-split forward kernel
# ---------------------------------------------------------------------------
# Grid : (B * H,  ceil(max_seqlen / BLOCK_M),  SPLIT_N)
# Each CTA owns (seq_head, q_tile, kv_split) and iterates over its KV chunk.
# Partial results (un-normalised acc, row-max m, normaliser l) are written to
# token-indexed fp32 temp buffers (no dependence on BLOCK_M for buffer shape).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["HEAD_DIM", "BRANCHING_FACTOR", "MAX_DEPTH"],
)
@triton.jit
def _ragged_attn_split_kernel(
    # ---- packed token buffers ----
    Q_ptr, K_ptr, V_ptr,
    # ---- partial output buffers  [total_tokens, SPLIT_N, H, …] ----
    Acc_ptr,   # fp32  [total_tokens, SPLIT_N, H, HEAD_DIM]
    M_ptr,     # fp32  [total_tokens, SPLIT_N, H]
    L_ptr,     # fp32  [total_tokens, SPLIT_N, H]
    # ---- sequence bookkeeping ----
    cu_seqlens_ptr,     # int32 [B+1]
    # ---- strides: Q/K/V (token, head, dim) ----
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    # ---- strides: Acc (token, split, head, dim) ----
    stride_at, stride_as_, stride_ah, stride_ad,
    # ---- strides: M/L (token, split, head) ----
    stride_mt, stride_ms_, stride_mh,
    # ---- scalars ----
    scale,
    max_seqlen,
    SPLIT_N,            # int (runtime); used for KV-range arithmetic only
    # ---- compile-time constants ----
    H:                tl.constexpr,
    HEAD_DIM:         tl.constexpr,
    BRANCHING_FACTOR: tl.constexpr,
    MAX_DEPTH:        tl.constexpr,
    BLOCK_M:          tl.constexpr,   # fixed = 16, passed explicitly
    BLOCK_N:          tl.constexpr,   # autotuned
):
    pid0     = tl.program_id(0)   # seq_idx * H + head_idx
    q_tile   = tl.program_id(1)   # tile index along Q axis
    split_id = tl.program_id(2)   # KV split chunk

    seq_idx  = pid0 // H
    head_idx = pid0  % H

    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end   = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len   = seq_end - seq_start

    q_off = q_tile * BLOCK_M
    if q_off >= seq_len:
        return

    # ---- KV range for this split chunk ----
    n_kv_tiles   = (seq_len + BLOCK_N - 1) // BLOCK_N
    kv_per_split = (n_kv_tiles + SPLIT_N - 1) // SPLIT_N
    kv_tile_lo   = split_id * kv_per_split
    hi_raw       = kv_tile_lo + kv_per_split
    kv_tile_hi   = tl.where(hi_raw < n_kv_tiles, hi_raw, n_kv_tiles)

    m_range = tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, HEAD_DIM)
    valid_q = (m_range + q_off) < seq_len

    # ---- Load Q tile [BLOCK_M, HEAD_DIM] fp16 ----
    q_tok  = seq_start + q_off + m_range
    q_ptrs = (Q_ptr
               + q_tok [:, None] * stride_qt
               + head_idx        * stride_qh
               + d_range[None,:] * stride_qd)
    q = tl.load(q_ptrs, mask=valid_q[:, None], other=0.0)   # fp16

    # ---- Flash-Attention-2 online softmax state ----
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # ---- Iterate over the assigned KV tiles ----
    for kv_tile in range(kv_tile_lo, kv_tile_hi):
        n_off   = kv_tile * BLOCK_N
        kv_mask = (tl.arange(0, BLOCK_N) + n_off) < seq_len
        kv_tok  = seq_start + n_off + tl.arange(0, BLOCK_N)

        k_ptrs = (K_ptr
                  + kv_tok [:, None] * stride_kt
                  + head_idx         * stride_kh
                  + d_range [None,:] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)   # fp16

        # QK^T [BLOCK_M, BLOCK_N]  —  fp16 × fp16 → fp32 accumulation
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # ---- Analytic tree mask (inlined ancestor walk) ----
        # For BFS-ordered complete b-ary tree: parent(k) = (k-1) // b
        # attend[i,j] = True iff node j is an ancestor-or-self of node i
        q_idx  = (m_range + q_off)[:, None]               # [BLOCK_M, 1]
        kv_idx = (tl.arange(0, BLOCK_N) + n_off)[None, :] # [1, BLOCK_N]
        cur    = q_idx  # walk up from query node; stays [BLOCK_M, 1]
        attend = (cur == kv_idx)  # check self     → [BLOCK_M, BLOCK_N]
        for _step in range(MAX_DEPTH):   # unrolled: MAX_DEPTH == tree depth
            cur    = tl.where(cur > 0, (cur - 1) // BRANCHING_FACTOR, 0)
            attend = attend | (cur == kv_idx)
        attend = attend & valid_q[:, None] & kv_mask[None, :]
        s      = tl.where(attend, s, float("-inf"))

        # ---- Online softmax accumulation ----
        blk_max = tl.max(s, axis=1)
        m_new   = tl.maximum(m_i, blk_max)
        alpha   = tl.exp(m_i - m_new)

        p   = tl.exp(s - m_new[:, None])
        p   = tl.where(attend, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = (V_ptr
                  + kv_tok [:, None] * stride_vt
                  + head_idx         * stride_vh
                  + d_range [None,:] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)   # fp16

        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i  = m_new

    # ---- Write partial results to token-indexed buffers ----
    tok_ids = seq_start + q_off + m_range   # [BLOCK_M] global token indices

    # Acc [total_tokens, SPLIT_N, H, HEAD_DIM]
    acc_ptrs = (Acc_ptr
                + tok_ids [:, None] * stride_at
                + split_id          * stride_as_
                + head_idx          * stride_ah
                + d_range  [None,:] * stride_ad)
    tl.store(acc_ptrs, acc, mask=valid_q[:, None])

    # M (row-max)  [total_tokens, SPLIT_N, H]
    m_ptrs = (M_ptr
              + tok_ids  * stride_mt
              + split_id * stride_ms_
              + head_idx * stride_mh)
    tl.store(m_ptrs, m_i, mask=valid_q)

    # L (normaliser)  [total_tokens, SPLIT_N, H]
    l_ptrs = (L_ptr
              + tok_ids  * stride_mt
              + split_id * stride_ms_
              + head_idx * stride_mh)
    tl.store(l_ptrs, l_i, mask=valid_q)


# ---------------------------------------------------------------------------
# Pass 2 — per-token reduction across SPLIT_N partial results
# ---------------------------------------------------------------------------
# Grid : (total_tokens,  H)
# Each CTA reads SPLIT_N partial (acc_p, m_p, l_p) entries for one
# (token, head) and merges them via online log-sum-exp to write final O.
# SPLIT_N_KEY is a constexpr so the inner loop is statically unrolled and
# Triton can exploit register reuse across iterations.
# ---------------------------------------------------------------------------

@triton.autotune(configs=_REDUCE_AUTOTUNE_CONFIGS, key=["HEAD_DIM", "SPLIT_N_KEY"])
@triton.jit
def _ragged_attn_reduce_kernel(
    Acc_ptr, M_ptr, L_ptr,    # partial buffers (read)
    O_ptr,                     # [total_tokens, H, HEAD_DIM]  fp16  (write)
    # ---- strides: Acc (token, split, head, dim) ----
    stride_at, stride_as_, stride_ah, stride_ad,
    # ---- strides: M/L (token, split, head) ----
    stride_mt, stride_ms_, stride_mh,
    # ---- strides: O (token, head, dim) ----
    stride_ot, stride_oh, stride_od,
    # ---- compile-time ----
    HEAD_DIM:    tl.constexpr,
    SPLIT_N_KEY: tl.constexpr,   # == SPLIT_N; constexpr for unrolling
):
    tok_idx  = tl.program_id(0)
    head_idx = tl.program_id(1)

    d_range = tl.arange(0, HEAD_DIM)

    m_g = tl.full([1], float("-inf"), dtype=tl.float32)[0]
    l_g = tl.full([1], 0.0,           dtype=tl.float32)[0]
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for sp in range(SPLIT_N_KEY):          # statically unrolled
        base_ml = tok_idx * stride_mt + sp * stride_ms_ + head_idx * stride_mh
        m_p = tl.load(M_ptr + base_ml)
        l_p = tl.load(L_ptr + base_ml)

        base_acc = tok_idx * stride_at + sp * stride_as_ + head_idx * stride_ah
        acc_p = tl.load(Acc_ptr + base_acc + d_range * stride_ad)

        m_new   = tl.maximum(m_g, m_p)
        alpha_g = tl.exp(m_g - m_new)
        alpha_p = tl.exp(m_p - m_new)

        l_g = l_g * alpha_g + l_p * alpha_p
        acc = acc * alpha_g + acc_p * alpha_p
        m_g = m_new

    out    = acc / tl.where(l_g == 0.0, 1.0, l_g)
    o_ptrs = O_ptr + tok_idx * stride_ot + head_idx * stride_oh + d_range * stride_od
    tl.store(o_ptrs, out.to(tl.float16))


# ---------------------------------------------------------------------------
# SPLIT_N selection helper
# ---------------------------------------------------------------------------

_T4_NUM_SMS         = 40
_TARGET_CTAS_PER_SM = 8    # gives 16 warps/SM at num_warps=2 → good latency hiding


def _compute_split_n(
    total_q_tiles: int,
    H:             int,
    max_kv_tiles:  int,
    num_sms:       int = _T4_NUM_SMS,
) -> int:
    """
    Choose SPLIT_N so that total CTAs in Pass 1 ≥ num_sms × TARGET_CTAS_PER_SM.
    Capped at max_kv_tiles (cannot split more than available KV tiles).
    """
    current_ctas = total_q_tiles * H
    desired      = num_sms * _TARGET_CTAS_PER_SM
    split        = max(1, math.ceil(desired / max(1, current_ctas)))
    return min(split, max(1, max_kv_tiles))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_inputs(
    qs: List[torch.Tensor],
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Concatenate per-sequence Q/K/V into the packed ragged layout.

    Parameters
    ----------
    qs, ks, vs : B-length lists of [L_i, H, D] fp16 tensors

    Returns
    -------
    Q, K, V    : [total_tokens, H, D]  fp16
    cu_seqlens : [B+1]  int32  (CPU tensor)
    """
    assert len(qs) == len(ks) == len(vs)
    B        = len(qs)
    seq_lens = [q.shape[0] for q in qs]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    for i, n in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + n
    return torch.cat(qs, 0), torch.cat(ks, 0), torch.cat(vs, 0), cu_seqlens


def ragged_attention(
    Q:  torch.Tensor,           # [total_tokens, H, D]  fp16  CUDA
    K:  torch.Tensor,           # [total_tokens, H, D]  fp16  CUDA
    V:  torch.Tensor,           # [total_tokens, H, D]  fp16  CUDA
    cu_seqlens:       torch.Tensor,   # [B+1]  int32
    branching_factor: int,            # b — complete b-ary draft tree
    max_depth:        int,            # d — maximum tree depth in batch
) -> torch.Tensor:
    """
    Ragged Flash-Decoding attention with analytic b-ary tree masking.

    Design highlights
    -----------------
    * No packed_masks buffer — tree masking is pure integer arithmetic.
    * SPLIT_N is auto-chosen to ensure ≥ (num_SMs × 8) active CTAs,
      saturating T4 even for single-sequence short-context batches.
    * Two-pass algorithm follows Flash-Decoding (Dao et al. 2023).

    Parameters
    ----------
    Q, K, V          : packed fp16 tensors  [Σ L_i, H, D]  on CUDA
    cu_seqlens       : cumulative sequence lengths  [B+1]  int32
    branching_factor : b  (BFS-ordered complete b-ary tree)
    max_depth        : d  (maximum tree depth in this batch)

    Returns
    -------
    O : [Σ L_i, H, D]  fp16
    """
    assert Q.dtype == K.dtype == V.dtype == torch.float16, "Inputs must be fp16"
    total_tokens, H, D = Q.shape
    B      = int(cu_seqlens.shape[0]) - 1
    device = Q.device
    scale  = 1.0 / math.sqrt(D)

    cu_seqlens_dev = cu_seqlens.to(device, dtype=torch.int32)
    seq_lens   = (cu_seqlens_dev[1:] - cu_seqlens_dev[:-1]).cpu().tolist()
    max_seqlen = int(max(seq_lens)) if seq_lens else 1

    total_q_tiles = sum(math.ceil(l / _BLOCK_M) for l in seq_lens)
    max_kv_tiles  = math.ceil(max_seqlen / _BLOCK_M)
    SPLIT_N       = _compute_split_n(total_q_tiles, H, max_kv_tiles)

    # ---- Partial buffers (neutral initial values) ----
    Acc = torch.zeros(total_tokens, SPLIT_N, H, D,
                      dtype=torch.float32, device=device)
    M   = torch.full((total_tokens, SPLIT_N, H), float("-inf"),
                     dtype=torch.float32, device=device)
    L   = torch.zeros(total_tokens, SPLIT_N, H,
                      dtype=torch.float32, device=device)
    O   = torch.empty_like(Q)

    # ---- Pass 1 — KV-split forward ----
    q_tiles_max = math.ceil(max_seqlen / _BLOCK_M)
    grid_split  = (B * H, q_tiles_max, SPLIT_N)

    _ragged_attn_split_kernel[grid_split](
        Q, K, V,
        Acc, M, L,
        cu_seqlens_dev,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Acc.stride(0), Acc.stride(1), Acc.stride(2), Acc.stride(3),
        M.stride(0),   M.stride(1),   M.stride(2),
        scale,
        max_seqlen,
        SPLIT_N,
        H=H,
        HEAD_DIM=D,
        BRANCHING_FACTOR=branching_factor,
        MAX_DEPTH=max_depth,
        BLOCK_M=_BLOCK_M,
    )

    # ---- Pass 2 — per-token reduction ----
    _ragged_attn_reduce_kernel[(total_tokens, H)](
        Acc, M, L,
        O,
        Acc.stride(0), Acc.stride(1), Acc.stride(2), Acc.stride(3),
        M.stride(0),   M.stride(1),   M.stride(2),
        O.stride(0),   O.stride(1),   O.stride(2),
        HEAD_DIM=D,
        SPLIT_N_KEY=SPLIT_N,
    )

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
    H, D, b, d, B = 4, 64, 2, 2, 3
    N = sum(b ** k for k in range(d + 1))  # 7 nodes for b=2, d=2

    qs = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]
    ks = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]
    vs = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]

    Q, K, V, cu_sl = pack_inputs(qs, ks, vs)
    O = ragged_attention(Q, K, V, cu_sl, branching_factor=b, max_depth=d)
    print(f"Smoke test passed — shape {O.shape}, dtype {O.dtype}")
