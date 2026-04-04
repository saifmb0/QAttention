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
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
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
# Error statistics dataclass (for paper reporting)
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class ErrorStats:
    """Numerical error statistics for a single (B, b, d) configuration."""
    batch_size: int
    branching_factor: int
    depth: int
    num_tokens: int
    passed: bool
    peak_abs_err: float     # max |ragged - ref| across all elements
    avg_abs_err: float      # mean |ragged - ref|
    peak_rel_err: float     # max |ragged - ref| / (|ref| + eps)
    avg_rel_err: float      # mean of same


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
) -> ErrorStats:
    """
    Compare ragged kernel output to padded SDPA reference.
    Returns an ErrorStats dataclass with peak/avg absolute and relative errors.
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

    # ---- Compute error statistics across all sequences ----
    all_abs_errs = []
    all_rel_errs = []
    all_ok = True
    for i in range(batch_size):
        r   = ragged_outs[i].float()
        ref = ref_outs   [i].float()
        abs_err = (r - ref).abs()
        rel_err = abs_err / (ref.abs() + 1e-6)
        all_abs_errs.append(abs_err)
        all_rel_errs.append(rel_err)
        if not torch.allclose(r, ref, atol=ATOL, rtol=RTOL):
            if verbose:
                print(
                    f"  MISMATCH seq {i}: peak_abs={abs_err.max().item():.6f}  "
                    f"peak_rel={rel_err.max().item():.6f}"
                )
            all_ok = False
        else:
            if verbose:
                print(
                    f"  OK seq {i}: peak_abs={abs_err.max().item():.6f}  "
                    f"avg_abs={abs_err.mean().item():.6f}"
                )

    # Aggregate across all sequences
    cat_abs = torch.cat([e.flatten() for e in all_abs_errs])
    cat_rel = torch.cat([e.flatten() for e in all_rel_errs])

    return ErrorStats(
        batch_size=batch_size,
        branching_factor=branching_factor,
        depth=depth,
        num_tokens=N,
        passed=all_ok,
        peak_abs_err=cat_abs.max().item(),
        avg_abs_err=cat_abs.mean().item(),
        peak_rel_err=cat_rel.max().item(),
        avg_rel_err=cat_rel.mean().item(),
    )


# ---------------------------------------------------------------------------
# Pytest parametrized tests
# ---------------------------------------------------------------------------

_CONFIGURATIONS = list(itertools.product(
    [1, 2, 4, 8],   # batch_size
    [1, 2, 3],       # branching_factor
    [1, 2, 3, 4, 5, 6],  # depth  (extended to cover typical EAGLE-2 trees)
))

# Exact benchmark configuration exercised in benchmark_sota.py
_BENCHMARK_CONFIGS = [
    (4, 2, 5),   # B=4, b=2, d=5
    (4, 3, 5),   # B=4, b=3, d=5
    (8, 2, 5),   # B=8, b=2, d=5  (default bench config)
    (8, 3, 5),   # B=8, b=3, d=5
    (4, 2, 6),   # b=2, d=6 – bigger tree for stress
    (4, 3, 6),   # b=3, d=6
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size,branching_factor,depth", _CONFIGURATIONS)
def test_ragged_vs_sdpa(batch_size: int, branching_factor: int, depth: int):
    stats = run_correctness_check(
        batch_size=batch_size,
        branching_factor=branching_factor,
        depth=depth,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert stats.passed, (
        f"Ragged attention mismatch for "
        f"B={batch_size} b={branching_factor} d={depth}  "
        f"peak_abs={stats.peak_abs_err:.6f}  peak_rel={stats.peak_rel_err:.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size,branching_factor,depth", _BENCHMARK_CONFIGS)
def test_benchmark_configs(batch_size: int, branching_factor: int, depth: int):
    """Exact shapes used in benchmark_sota.py – ensures no silent correctness regressions."""
    stats = run_correctness_check(
        batch_size=batch_size,
        branching_factor=branching_factor,
        depth=depth,
        num_heads=8,
        head_dim=128,   # LLaMA-style head dim
        verbose=True,
    )
    assert stats.passed, (
        f"Benchmark-config mismatch: B={batch_size} b={branching_factor} d={depth}  "
        f"peak_abs={stats.peak_abs_err:.6f}  peak_rel={stats.peak_rel_err:.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_head_dims(head_dim: int):
    """Verify kernel works for different head dimensions (32/64/128)."""
    stats = run_correctness_check(
        batch_size=4,
        branching_factor=2,
        depth=3,
        num_heads=4,
        head_dim=head_dim,
        verbose=True,
    )
    assert stats.passed, (
        f"Ragged attention mismatch for head_dim={head_dim}  "
        f"peak_abs={stats.peak_abs_err:.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_token():
    """Edge case: tree of depth 0 = single root token only."""
    stats = run_correctness_check(
        batch_size=4,
        branching_factor=1,
        depth=0,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert stats.passed, f"Single-token edge case failed  peak_abs={stats.peak_abs_err:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_chain():
    """Chain tree (b=1): degenerates to standard causal attention."""
    stats = run_correctness_check(
        batch_size=4,
        branching_factor=1,
        depth=8,
        num_heads=4,
        head_dim=64,
        verbose=True,
    )
    assert stats.passed, f"Linear chain (b=1, d=8) failed  peak_abs={stats.peak_abs_err:.6f}"


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
        [1, 2, 3, 4, 5, 6],  # d
    )) + _BENCHMARK_CONFIGS
    # Deduplicate
    configs = list(dict.fromkeys(configs))

    all_stats: list[ErrorStats] = []
    for B, b, d in configs:
        stats = run_correctness_check(
            batch_size=B, branching_factor=b, depth=d,
            num_heads=4, head_dim=64, verbose=False,
        )
        all_stats.append(stats)

    # ---- Formatted summary table ----
    hdr = (
        f"{'B':>3s}  {'b':>2s}  {'d':>2s}  {'N':>6s}  {'status':>6s}  "
        f"{'peak_abs':>10s}  {'avg_abs':>10s}  {'peak_rel':>10s}  {'avg_rel':>10s}"
    )
    print("\n" + "=" * len(hdr))
    print("Numerical Correctness Summary  (ragged vs. padded SDPA reference)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for s in all_stats:
        tag = " PASS" if s.passed else "*FAIL"
        print(
            f"{s.batch_size:3d}  {s.branching_factor:2d}  {s.depth:2d}  "
            f"{s.num_tokens:6d}  {tag:>6s}  "
            f"{s.peak_abs_err:10.2e}  {s.avg_abs_err:10.2e}  "
            f"{s.peak_rel_err:10.2e}  {s.avg_rel_err:10.2e}"
        )
    print("-" * len(hdr))

    n_pass = sum(1 for s in all_stats if s.passed)
    n_fail = len(all_stats) - n_pass

    # Global aggregates (across ALL configs)
    if all_stats:
        global_peak_abs = max(s.peak_abs_err for s in all_stats)
        global_avg_abs  = sum(s.avg_abs_err for s in all_stats) / len(all_stats)
        global_peak_rel = max(s.peak_rel_err for s in all_stats)
        global_avg_rel  = sum(s.avg_rel_err for s in all_stats) / len(all_stats)
        print(
            f"\nGlobal  |  peak_abs = {global_peak_abs:.2e}  "
            f"avg_abs = {global_avg_abs:.2e}  "
            f"peak_rel = {global_peak_rel:.2e}  "
            f"avg_rel = {global_avg_rel:.2e}"
        )

    print(f"Result  |  {n_pass} passed, {n_fail} failed out of {len(all_stats)} configs.\n")
    sys.exit(0 if n_fail == 0 else 1)
