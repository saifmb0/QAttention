"""
test_correctness.py
===================
Verify that the Triton ragged attention kernel produces output identical
(within float16 tolerance) to PyTorch scaled-dot-product attention (SDPA)
with the tree-structured causal mask correctly applied.

Baseline
--------
  torch.nn.functional.scaled_dot_product_attention with an explicit
  additive attention_bias (0 for attend, -inf for masked).
  This is the same computation path vLLM uses in its verification pass.
  The padded sequence is trimmed to valid tokens after SDPA.

Comparison tolerance
--------------------
  atol = 1e-2  (fp16 accumulates ~0.001 per term × O(D) terms ≈ 0.06 for D=64)
  rtol = 1e-2

Run with:
  pytest tests/test_correctness.py -v

Or standalone:
  python tests/test_correctness.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import itertools
from typing import List

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.tree_mask import tree_attention_mask, num_tree_nodes
from src.ragged_attn import pack_inputs, ragged_attention


# ---------------------------------------------------------------------------
# Reference: padded SDPA with tree mask (current SOTA baseline = vLLM path)
# ---------------------------------------------------------------------------

def _build_sdpa_bias(
    seq_lens: List[int],
    masks_np: List[np.ndarray],
    L_max: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build additive attention bias [B, 1, L_max, L_max] for padded SDPA.

    For each sequence i:
      - positions [0 .. L_i-1] × [0 .. L_i-1] : use tree/full mask (0 or -inf)
      - positions involving padding              : -inf
    """
    B = len(seq_lens)
    NEG_INF = torch.finfo(torch.float32).min / 2

    bias = torch.full((B, 1, L_max, L_max), NEG_INF, device=device)
    for i, (Li, mask) in enumerate(zip(seq_lens, masks_np)):
        # Fill the valid L_i × L_i block from the tree mask
        tree_t = torch.from_numpy(mask.astype(np.float32)).to(device)   # [Li, Li]
        tree_bias = torch.where(tree_t.bool(),
                                torch.zeros_like(tree_t),
                                torch.full_like(tree_t, NEG_INF))
        bias[i, 0, :Li, :Li] = tree_bias
    return bias


def padded_sdpa_reference(
    qs: List[torch.Tensor],     # fp16, each [L_i, H, D]
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
    masks_np: List[np.ndarray],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Run PyTorch SDPA on a padded batch and return valid (non-padding) tokens.

    The function uses math backend (always available on SM75) to guarantee
    bit-exact reference outputs regardless of flash-attn availability.

    Returns list of [L_i, H, D] fp16 tensors (one per sequence).
    """
    B        = len(qs)
    H, D     = qs[0].shape[1], qs[0].shape[2]
    seq_lens = [q.shape[0] for q in qs]
    L_max    = max(seq_lens)

    # Pad Q, K, V → [B, L_max, H, D] then permute → [B, H, L_max, D]
    def pad_and_stack(tensors):
        out = torch.zeros(B, L_max, H, D, device=device, dtype=torch.float32)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t.float()
        return out.permute(0, 2, 1, 3)  # [B, H, L_max, D]

    Q_pad = pad_and_stack(qs)
    K_pad = pad_and_stack(ks)
    V_pad = pad_and_stack(vs)

    # Attention bias [B, 1, L_max, L_max]
    attn_bias = _build_sdpa_bias(seq_lens, masks_np, L_max, device)

    # Force math backend so SM75 always works without flash-attn
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        out_pad = F.scaled_dot_product_attention(
            Q_pad, K_pad, V_pad,
            attn_mask=attn_bias,
            scale=1.0 / math.sqrt(D),
        )   # [B, H, L_max, D]

    # Permute back → [B, L_max, H, D] and extract valid tokens
    out_pad = out_pad.permute(0, 2, 1, 3)   # [B, L_max, H, D]
    results = []
    for i, Li in enumerate(seq_lens):
        results.append(out_pad[i, :Li].to(torch.float16))   # [Li, H, D]
    return results


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

ATOL = 1e-2   # fp16-safe absolute tolerance
RTOL = 1e-2   # fp16-safe relative tolerance


# ---------------------------------------------------------------------------
# Core comparison function (used by both pytest and standalone main)
# ---------------------------------------------------------------------------

def run_correctness_check(
    batch_size: int,
    branching_factor: int,
    depth: int,
    num_heads: int = 4,
    head_dim: int = 64,
    device_str: str = "cuda",
    verbose: bool = False,
) -> bool:
    """
    Returns True if ragged kernel output matches padded SDPA reference.
    Raises AssertionError on mismatch when called from pytest.
    """
    device = torch.device(device_str)
    torch.manual_seed(batch_size * 100 + branching_factor * 10 + depth)

    N = num_tree_nodes(branching_factor, depth)
    masks_np = [tree_attention_mask(branching_factor, depth)
                for _ in range(batch_size)]

    # Random fp16 inputs, one per sequence (all same length N for simplicity)
    qs = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    ks = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    vs = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]

    # ---- Ragged kernel ----
    Q_r, K_r, V_r, cu_seqlens = pack_inputs(qs, ks, vs)
    Q_r = Q_r.to(device)
    K_r = K_r.to(device)
    V_r = V_r.to(device)

    O_ragged = ragged_attention(
        Q_r, K_r, V_r, cu_seqlens,
        branching_factor=branching_factor, max_depth=depth,
    )
    # Split back into per-sequence tensors
    ragged_outs = []
    start = 0
    for i in range(batch_size):
        end = start + N
        ragged_outs.append(O_ragged[start:end])   # [N, H, D]
        start = end

    # ---- Padded SDPA reference ----
    ref_outs = padded_sdpa_reference(qs, ks, vs, masks_np, device)

    # ---- Compare ----
    all_ok = True
    for i in range(batch_size):
        r   = ragged_outs[i].float()
        ref = ref_outs   [i].float()
        if not torch.allclose(r, ref, atol=ATOL, rtol=RTOL):
            max_err = (r - ref).abs().max().item()
            rel_err = ((r - ref).abs() / (ref.abs() + 1e-6)).max().item()
            if verbose:
                print(
                    f"  MISMATCH seq {i}: max_abs={max_err:.4f}  max_rel={rel_err:.4f}"
                )
            all_ok = False
        else:
            if verbose:
                max_err = (r - ref).abs().max().item()
                print(
                    f"  OK seq {i}: max_abs={max_err:.5f}"
                )

    return all_ok


# ---------------------------------------------------------------------------
# Pytest parametrized tests
# ---------------------------------------------------------------------------

_CONFIGURATIONS = list(itertools.product(
    [1, 2, 4, 8],   # batch_size
    [1, 2, 3],       # branching_factor
    [1, 2, 3],       # depth
))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size,branching_factor,depth", _CONFIGURATIONS)
def test_ragged_vs_sdpa(batch_size: int, branching_factor: int, depth: int):
    ok = run_correctness_check(
        batch_size=batch_size,
        branching_factor=branching_factor,
        depth=depth,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert ok, (
        f"Ragged attention mismatch for "
        f"B={batch_size} b={branching_factor} d={depth}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_head_dims(head_dim: int):
    """Verify kernel works for different head dimensions (32/64/128)."""
    ok = run_correctness_check(
        batch_size=4,
        branching_factor=2,
        depth=3,
        num_heads=4,
        head_dim=head_dim,
        verbose=True,
    )
    assert ok, f"Ragged attention mismatch for head_dim={head_dim}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_token():
    """Edge case: tree of depth 0 = single root token only."""
    ok = run_correctness_check(
        batch_size=4,
        branching_factor=1,
        depth=0,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert ok, "Single-token edge case failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_chain():
    """Chain tree (b=1): degenerates to standard causal attention."""
    ok = run_correctness_check(
        batch_size=4,
        branching_factor=1,
        depth=8,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert ok, "Linear chain (b=1, d=8) failed"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available – skipping correctness tests.")
        sys.exit(0)

    configs = list(itertools.product(
        [1, 2, 4, 8],   # batch
        [1, 2, 3],       # b
        [1, 2, 3],       # d
    ))
    n_pass = n_fail = 0
    for B, b, d in configs:
        ok = run_correctness_check(
            batch_size=B, branching_factor=b, depth=d,
            num_heads=4, head_dim=64, verbose=False,
        )
        status = "PASS" if ok else "FAIL"
        print(f"B={B:2d}  b={b}  d={d}  →  {status}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\nResults: {n_pass} passed, {n_fail} failed out of {len(configs)} configs.")
    sys.exit(0 if n_fail == 0 else 1)
