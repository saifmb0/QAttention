"""
ragged_attn.py
==============
Triton ragged (variable-length, no-padding) attention kernel with
tree-structured causal mask support.

Target hardware : SM75 (NVIDIA T4 16 GB, Kaggle 2×T4)
Precision       : float16 (accumulate in float32, write fp16)
Algorithm       : Flash-Attention-2 online softmax, tiled QK^T + OV

Data layout
-----------
All sequences in the batch are concatenated along the token axis:

  Q  : [total_tokens,  H, D]   fp16
  K  : [total_tokens,  H, D]   fp16
  V  : [total_tokens,  H, D]   fp16
  O  : [total_tokens,  H, D]   fp16

  cu_seqlens         : [B+1]   int32  – cumulative token counts
  packed_masks       : [Σ L_i²] int8  – per-sequence attention masks, row-major
  cu_mask_offsets    : [B+1]   int32  – byte (element) offsets into packed_masks

cu_seqlens[0]  = 0
cu_seqlens[i]  = sum(L_0 .. L_{i-1})
cu_seqlens[B]  = total_tokens

cu_mask_offsets[i] = sum(L_0² + L_1² + … + L_{i-1}²)

Grid
----
  axis 0 : flat index  seq_idx * H + head_idx
  axis 1 : M-tile index within the *global* padded range
            → exit early if tile_start >= L_i  (sparse, SM75-safe)

BLOCK_M and BLOCK_N are 64 for SM75 (fits inside 64 KB SRAM per SM).

Public API
----------
  pack_inputs(qs, ks, vs, masks)
      qs, ks, vs : list[Tensor[L_i, H, D], …]
      masks      : list[np.ndarray[L_i, L_i, bool]]
      returns    : (Q, K, V, cu_seqlens, packed_masks, cu_mask_offsets)

  ragged_attention(Q, K, V, cu_seqlens, packed_masks, cu_mask_offsets) -> O
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _ragged_attn_fwd_kernel(
    # ---- packed token buffers ----
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # ---- sequence bookkeeping ----
    cu_seqlens_ptr,          # int32 [B+1]
    # ---- tree mask bookkeeping ----
    packed_masks_ptr,        # int8  [Σ L_i²]
    cu_mask_offsets_ptr,     # int32 [B+1]
    # ---- tensor strides (token, head, dim) ----
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    # ---- scalar args ----
    scale,
    max_seqlen,              # upper bound for axis-1 grid sizing
    # ---- compile-time constants ----
    H:        tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
):
    # ------------------------------------------------------------------ #
    # Identify this CTA
    # ------------------------------------------------------------------ #
    pid0     = tl.program_id(0)   # seq_idx * H + head_idx
    m_tile   = tl.program_id(1)   # tile index along query axis

    seq_idx  = pid0 // H
    head_idx = pid0  % H

    # Sequence boundaries in packed buffer
    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end   = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len   = seq_end - seq_start          # dynamic (not constexpr)

    # Early exit: tile is beyond this sequence's length
    q_off = m_tile * BLOCK_M
    if q_off >= seq_len:
        return

    # ------------------------------------------------------------------ #
    # Load Q tile  [BLOCK_M, HEAD_DIM]
    # ------------------------------------------------------------------ #
    m_range  = tl.arange(0, BLOCK_M)
    d_range  = tl.arange(0, HEAD_DIM)
    q_m_mask = (m_range + q_off) < seq_len          # [BLOCK_M]

    q_tok    = seq_start + q_off + m_range           # absolute token indices
    q_ptrs   = (Q_ptr
                + q_tok [:, None] * stride_qt
                + head_idx        * stride_qh
                + d_range[None,:] * stride_qd)
    # Load as fp16 so tl.dot can use tensor cores; out_dtype=fp32 for accumulation
    q = tl.load(q_ptrs, mask=q_m_mask[:, None], other=0.0)

    # ------------------------------------------------------------------ #
    # Mask bookkeeping for this sequence
    # ------------------------------------------------------------------ #
    mask_off = tl.load(cu_mask_offsets_ptr + seq_idx)   # int32 element offset

    # ------------------------------------------------------------------ #
    # Flash-Attention-2 online softmax state
    # ------------------------------------------------------------------ #
    m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    # ------------------------------------------------------------------ #
    # Iterate over K/V tiles for this sequence
    # ------------------------------------------------------------------ #
    n_blocks = (seq_len + BLOCK_N - 1) // BLOCK_N

    for n in range(n_blocks):
        n_off    = n * BLOCK_N
        kv_tok   = seq_start + n_off + tl.arange(0, BLOCK_N)
        kv_mask  = (tl.arange(0, BLOCK_N) + n_off) < seq_len   # [BLOCK_N]

        # Load K  [BLOCK_N, HEAD_DIM]
        k_ptrs   = (K_ptr
                    + kv_tok [:, None] * stride_kt
                    + head_idx         * stride_kh
                    + d_range [None,:] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # QK^T  [BLOCK_M, BLOCK_N]  — both operands fp16, output fp32
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # ---- tree-mask lookup ----
        # packed_masks for seq i is row-major with stride = seq_len
        # element [m_local, n_local] lives at mask_off + m_local * seq_len + n_local
        m_local   = (m_range + q_off)[:, None]          # [BLOCK_M, 1]
        n_local   = (tl.arange(0, BLOCK_N) + n_off)[None, :]  # [1, BLOCK_N]
        fidx      = mask_off + m_local * seq_len + n_local     # [BLOCK_M, BLOCK_N]
        valid_mn  = q_m_mask[:, None] & kv_mask[None, :]       # [BLOCK_M, BLOCK_N]

        tree_flag = tl.load(packed_masks_ptr + fidx,
                            mask=valid_mn,
                            other=0).to(tl.int1)                # 1=attend,0=mask

        # Apply mask: set logits to -inf where not attending
        neg_inf = float("-inf")
        s = tl.where(tree_flag, s, neg_inf)
        # Also mask out padding key positions
        s = tl.where(kv_mask[None, :], s, neg_inf)

        # ---- online softmax update ----
        blk_max = tl.max(s, axis=1)                     # [BLOCK_M]
        m_new   = tl.maximum(m_i, blk_max)
        alpha   = tl.exp(m_i - m_new)                   # rescale factor

        p       = tl.exp(s - m_new[:, None])            # [BLOCK_M, BLOCK_N]
        p       = tl.where(kv_mask[None, :], p, 0.0)   # zero padding

        l_i     = l_i * alpha + tl.sum(p, axis=1)
        acc     = acc * alpha[:, None]

        # Load V  [BLOCK_N, HEAD_DIM]
        v_ptrs  = (V_ptr
                   + kv_tok [:, None] * stride_vt
                   + head_idx         * stride_vh
                   + d_range [None,:] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # P·V: both operands fp16, accumulate into fp32 acc
        acc    += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i     = m_new

    # ------------------------------------------------------------------ #
    # Normalize and write output
    # ------------------------------------------------------------------ #
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc    = acc / l_safe[:, None]

    o_tok   = seq_start + q_off + m_range
    o_ptrs  = (O_ptr
               + o_tok [:, None] * stride_ot
               + head_idx        * stride_oh
               + d_range[None,:] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=q_m_mask[:, None])


# ---------------------------------------------------------------------------
# Block-size selection for SM75 (T4)
# ---------------------------------------------------------------------------
# T4 SM75: 64 KB shared memory per SM, peak 65536 bytes
# Per-CTA SRAM budget for BLOCK_M=64, BLOCK_N=64, HEAD_DIM=64 (fp16):
#   Q tile : 64 × 64 × 2 = 8 KB
#   K tile : 8 KB  (loaded in registers, but counts toward occupancy)
#   V tile : 8 KB
#   S/P    : 64 × 64 × 4 = 16 KB  (fp32)
#   → ~40 KB → 1 CTA/SM with HEAD_DIM=64, or 32-wide N blocks for D=128
_SM75_BLOCK_M  = 64
_SM75_BLOCK_N  = 64


# ---------------------------------------------------------------------------
# Input packing helpers
# ---------------------------------------------------------------------------

def pack_inputs(
    qs: List[torch.Tensor],
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
    masks: List[np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack per-sequence Q/K/V and attention masks into the ragged layout.

    Parameters
    ----------
    qs, ks, vs : each a list of B tensors of shape [L_i, H, D]  fp16
    masks      : list of B boolean arrays of shape [L_i, L_i]
                 True = token may attend (1), False = masked (0)

    Returns
    -------
    Q, K, V         : [total_tokens, H, D]  fp16  (cat along dim 0)
    cu_seqlens      : [B+1]  int32  on CPU
    packed_masks    : [Σ L_i²]  int8  on CPU
    cu_mask_offsets : [B+1]  int32  on CPU
    """
    assert len(qs) == len(ks) == len(vs) == len(masks), \
        "All lists must have the same length (= batch size)"

    seq_lens = [q.shape[0] for q in qs]
    B = len(seq_lens)

    # cu_seqlens
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    for i, l in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + l

    # Packed Q, K, V
    Q = torch.cat(qs, dim=0)   # [Σ L_i, H, D]
    K = torch.cat(ks, dim=0)
    V = torch.cat(vs, dim=0)

    # Packed masks
    mask_sizes = [m.shape[0] * m.shape[1] for m in masks]  # L_i²
    total_mask  = sum(mask_sizes)
    packed_masks = np.empty(total_mask, dtype=np.int8)
    cu_mask_offsets = np.zeros(B + 1, dtype=np.int32)
    offset = 0
    for i, m in enumerate(masks):
        flat = m.astype(np.int8).ravel()
        packed_masks[offset : offset + len(flat)] = flat
        offset += len(flat)
        cu_mask_offsets[i + 1] = offset

    packed_masks    = torch.from_numpy(packed_masks)
    cu_mask_offsets = torch.from_numpy(cu_mask_offsets)

    return Q, K, V, cu_seqlens, packed_masks, cu_mask_offsets


# ---------------------------------------------------------------------------
# Main Python wrapper
# ---------------------------------------------------------------------------

def ragged_attention(
    Q: torch.Tensor,             # [total_tokens, H, D]  fp16  CUDA
    K: torch.Tensor,             # [total_tokens, H, D]  fp16  CUDA
    V: torch.Tensor,             # [total_tokens, H, D]  fp16  CUDA
    cu_seqlens: torch.Tensor,    # [B+1]  int32          CPU or CUDA
    packed_masks: torch.Tensor,  # [Σ L²] int8           CPU or CUDA
    cu_mask_offsets: torch.Tensor, # [B+1] int32         CPU or CUDA
    BLOCK_M: int = _SM75_BLOCK_M,
    BLOCK_N: int = _SM75_BLOCK_N,
) -> torch.Tensor:
    """
    Run ragged attention and return output O with same shape as Q.

    All tensors must live on the same CUDA device.
    fp16 input tensors are expected; output is fp16.
    """
    assert Q.dtype == torch.float16, "Q must be fp16"
    assert K.dtype == torch.float16, "K must be fp16"
    assert V.dtype == torch.float16, "V must be fp16"

    total_tokens, H, D = Q.shape
    B = cu_seqlens.shape[0] - 1
    device = Q.device

    # Move bookkeeping tensors to GPU
    cu_seqlens_dev      = cu_seqlens     .to(device, dtype=torch.int32)
    packed_masks_dev    = packed_masks   .to(device, dtype=torch.int8)
    cu_mask_offsets_dev = cu_mask_offsets.to(device, dtype=torch.int32)

    # Allocate output
    O = torch.empty_like(Q)

    # Softmax scale
    scale = 1.0 / math.sqrt(D)

    # Compute max sequence length for grid axis-1
    seq_lens   = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    max_seqlen = int(max(seq_lens)) if seq_lens else 1
    n_m_tiles  = triton.cdiv(max_seqlen, BLOCK_M)

    grid = (B * H, n_m_tiles)

    _ragged_attn_fwd_kernel[grid](
        Q, K, V, O,
        cu_seqlens_dev,
        packed_masks_dev,
        cu_mask_offsets_dev,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        scale,
        max_seqlen,
        H=H,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return O


# ---------------------------------------------------------------------------
# Convenience: build ragged inputs from per-sequence tensors + tree masks
# ---------------------------------------------------------------------------

def build_ragged_inputs_from_sequences(
    qs: List[torch.Tensor],
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
    masks: List[np.ndarray],
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper that calls pack_inputs then moves GPU tensors to *device*.
    """
    Q, K, V, cu_seqlens, packed_masks, cu_mask_offsets = pack_inputs(
        qs, ks, vs, masks
    )
    if device is not None:
        Q  = Q .to(device)
        K  = K .to(device)
        V  = V .to(device)
    return Q, K, V, cu_seqlens, packed_masks, cu_mask_offsets


# ---------------------------------------------------------------------------
# Smoke test (run with: python -m src.ragged_attn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from .tree_mask import tree_attention_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available – skipping smoke test.")
        sys.exit(0)

    torch.manual_seed(0)
    H, D = 4, 64
    b, d = 2, 2
    B    = 3

    masks_np = [tree_attention_mask(b, d) for _ in range(B)]
    N        = masks_np[0].shape[0]

    qs = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]
    ks = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]
    vs = [torch.randn(N, H, D, device=device, dtype=torch.float16) for _ in range(B)]

    Q_r, K_r, V_r, csl, pm, cmo = pack_inputs(qs, ks, vs, masks_np)
    Q_r, K_r, V_r = Q_r.to(device), K_r.to(device), V_r.to(device)

    O = ragged_attention(Q_r, K_r, V_r, csl, pm, cmo)
    print(f"Smoke test passed. Output shape: {O.shape}, dtype: {O.dtype}")
