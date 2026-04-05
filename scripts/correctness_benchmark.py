#!/usr/bin/env python3
"""
correctness_benchmark.py — Numerical Correctness of Ragged Attention Kernel
=============================================================================

Verifies that the ragged ancestor-sparse Triton kernel produces numerically
identical output to PyTorch SDPA with the correct tree-ancestor mask applied.

This script is the canonical source of correctness numbers for the paper.
It covers a wide range of configurations and saves all error statistics to CSV
so that reviewers can inspect the numbers directly.

WHAT IS TESTED
--------------
For each (B, b, d, H, D):
  • ragged kernel output  [B*N, H, D]
  • SDPA reference output [B*N, H, D]
    Using force-math backend (SDPBackend.MATH) for bit-exact reference,
    independent of flash-attn version or SM capability.

ERROR STATISTICS PER CONFIG
-----------------------------
  peak_abs_err  — max |ragged[i] − ref[i]|  across all output elements
  mean_abs_err  — mean of the above
  peak_rel_err  — max |ragged[i] − ref[i]| / (|ref[i]| + 1e-6)
  mean_rel_err  — mean of the above
  passed        — True if torch.allclose(ragged, ref, atol=0.01, rtol=0.01)

CONFIGURATIONS
--------------
  Standard (matching test_correctness.py pytest suite):
    B ∈ {1, 2, 4, 8},  b ∈ {1, 2, 3},  d ∈ {1, 2, 3, 4, 5, 6}
    H=4, D=64

  LLaMA-matched (LLaMA-3.1-8B / Vicuna-7B real deployment dimensions):
    B ∈ {1, 2, 4},  b ∈ {2, 3, 4},  d ∈ {3, 5, 7}
    H=32, D=128

  Edge-case:
    Linear chain (b=1, d=8) = standard causal attention
    Single token  (b=1, d=0) = trivial softmax(q·k/√D) * v

OUTPUT
------
  results/correctness_benchmark.csv  — one row per (B, b, d, H, D) config
  Console summary table with pass/fail and error values

Usage:
  python scripts/correctness_benchmark.py

  # Subset for fast CI
  python scripts/correctness_benchmark.py --fast

  # Custom output path
  python scripts/correctness_benchmark.py --out-dir /tmp/results

  # With verbose per-sequence output
  python scripts/correctness_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from dataclasses import dataclass, fields, asdict
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import torch
import torch.nn.functional as F

from src.tree_mask import tree_attention_mask, num_tree_nodes
from src.ragged_attn import pack_inputs, ragged_attention


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance
# ─────────────────────────────────────────────────────────────────────────────
ATOL = 1e-2   # fp16-safe: ~0.001 per term × D=64 → ~0.064 accumulated error
RTOL = 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation — padded SDPA, math backend (always available)
# ─────────────────────────────────────────────────────────────────────────────

def _padded_sdpa_reference(
    qs: List[torch.Tensor],
    ks: List[torch.Tensor],
    vs: List[torch.Tensor],
    masks_np,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Compute ground-truth attention using PyTorch MATH backend.

    Uses SDPBackend.MATH (not flash, not efficient) so the result is
    numerically reproducible regardless of GPU generation or flash-attn
    installation status.  This is the unambiguous reference.

    Returns list of [L_i, H, D] fp16 tensors (one per sequence).
    """
    B        = len(qs)
    H, D     = qs[0].shape[1], qs[0].shape[2]
    seq_lens = [q.shape[0] for q in qs]
    L_max    = max(seq_lens)
    NEG_INF  = torch.finfo(torch.float32).min / 2

    def _pad(tensors):
        o = torch.zeros(B, L_max, H, D, device=device, dtype=torch.float32)
        for i, t in enumerate(tensors):
            o[i, :t.shape[0]] = t.float()
        return o.permute(0, 2, 1, 3)   # [B, H, L_max, D]

    Q_pad = _pad(qs)
    K_pad = _pad(ks)
    V_pad = _pad(vs)

    # Build additive bias [B, 1, L_max, L_max]
    bias = torch.full((B, 1, L_max, L_max), NEG_INF, device=device)
    for i, (Li, mask) in enumerate(zip(seq_lens, masks_np)):
        tree_t   = torch.from_numpy(mask.astype("float32")).to(device)
        tree_bias = torch.where(tree_t.bool(),
                                torch.zeros_like(tree_t),
                                torch.full_like(tree_t, NEG_INF))
        bias[i, 0, :Li, :Li] = tree_bias

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            Q_pad, K_pad, V_pad,
            attn_mask=bias,
            scale=1.0 / math.sqrt(D),
        )                               # [B, H, L_max, D]

    out = out.permute(0, 2, 1, 3)      # [B, L_max, H, D]
    return [out[i, :seq_lens[i]].to(torch.float16) for i in range(B)]


# ─────────────────────────────────────────────────────────────────────────────
# Error stats dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectnessRow:
    """One row of the correctness benchmark CSV."""
    batch_size:       int
    branching_factor: int
    depth:            int
    num_tokens:       int     # tree nodes N = (b^(d+1) − 1) / (b − 1)
    num_heads:        int
    head_dim:         int
    passed:           bool
    peak_abs_err:     float
    mean_abs_err:     float
    peak_rel_err:     float
    mean_rel_err:     float
    atol:             float
    rtol:             float


# ─────────────────────────────────────────────────────────────────────────────
# Core comparison function
# ─────────────────────────────────────────────────────────────────────────────

def check_one(
    batch_size:       int,
    branching_factor: int,
    depth:            int,
    num_heads:        int = 4,
    head_dim:         int = 64,
    device_str:       str = "cuda",
    verbose:          bool = False,
) -> CorrectnessRow:
    """
    Run ragged kernel and SDPA reference on random inputs; return error stats.
    """
    device = torch.device(device_str)
    torch.manual_seed(batch_size * 10000 + branching_factor * 1000 + depth * 100
                      + num_heads * 10 + head_dim)

    N       = num_tree_nodes(branching_factor, depth)
    masks   = [tree_attention_mask(branching_factor, depth) for _ in range(batch_size)]

    qs = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    ks = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    vs = [torch.randn(N, num_heads, head_dim, device=device, dtype=torch.float16)
          for _ in range(batch_size)]

    # ── Ragged kernel ────────────────────────────────────────────────────────
    Q_packed, K_packed, V_packed, cu = pack_inputs(qs, ks, vs)
    Q_packed = Q_packed.to(device)
    K_packed = K_packed.to(device)
    V_packed = V_packed.to(device)

    O_packed = ragged_attention(
        Q_packed, K_packed, V_packed, cu,
        branching_factor=branching_factor,
        max_depth=depth,
    )
    ragged_outs = [O_packed[i * N:(i + 1) * N] for i in range(batch_size)]

    # ── SDPA reference ────────────────────────────────────────────────────────
    ref_outs = _padded_sdpa_reference(qs, ks, vs, masks, device)

    # ── Error statistics ──────────────────────────────────────────────────────
    all_abs, all_rel = [], []
    passed = True

    for i in range(batch_size):
        r   = ragged_outs[i].float()
        ref = ref_outs[i].float()
        abs_e = (r - ref).abs()
        rel_e = abs_e / (ref.abs() + 1e-6)
        all_abs.append(abs_e)
        all_rel.append(rel_e)
        ok = torch.allclose(r, ref, atol=ATOL, rtol=RTOL)
        if not ok:
            passed = False
        if verbose:
            tag = "OK  " if ok else "FAIL"
            print(
                f"    seq {i:2d}: {tag}  "
                f"peak_abs={abs_e.max().item():.3e}  "
                f"mean_abs={abs_e.mean().item():.3e}  "
                f"peak_rel={rel_e.max().item():.3e}"
            )

    cat_abs = torch.cat([e.flatten() for e in all_abs])
    cat_rel = torch.cat([e.flatten() for e in all_rel])

    return CorrectnessRow(
        batch_size=batch_size,
        branching_factor=branching_factor,
        depth=depth,
        num_tokens=N,
        num_heads=num_heads,
        head_dim=head_dim,
        passed=passed,
        peak_abs_err=float(cat_abs.max()),
        mean_abs_err=float(cat_abs.mean()),
        peak_rel_err=float(cat_rel.max()),
        mean_rel_err=float(cat_rel.mean()),
        atol=ATOL,
        rtol=RTOL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration sets
# ─────────────────────────────────────────────────────────────────────────────

# Standard sweep (matches test_correctness.py pytest suite)
_STANDARD_CONFIGS = [
    (B, b, d, 4, 64)
    for B, b, d in itertools.product(
        [1, 2, 4, 8],           # batch
        [1, 2, 3],              # branching factor
        [1, 2, 3, 4, 5, 6],    # depth
    )
]

# LLaMA-matched dimensions (H=32, D=128 — LLaMA-3.1-8B / Vicuna-7B)
_LLAMA_CONFIGS = [
    (B, b, d, 32, 128)
    for B, b, d in itertools.product(
        [1, 2, 4],     # batch
        [2, 3, 4],     # branching factor (typical EAGLE-2/3 values)
        [3, 5, 7],     # depth
    )
]

# Edge cases
_EDGE_CONFIGS = [
    (4, 1, 0,  4,  64),   # single-token tree
    (4, 1, 8,  4,  64),   # linear chain → causal attention
    (4, 2, 8,  4,  64),   # large binary tree
    (2, 4, 5, 32, 128),   # EAGLE-2 canonical (B=2, matches benchmark_sota)
    (8, 4, 5, 32, 128),   # EAGLE-2 canonical (B=8)
    (4, 2, 3,  4,  32),   # narrow head dim
]

_FAST_CONFIGS = [
    (B, b, d, H, D)
    for B, b, d in [(1, 2, 3), (4, 3, 5), (2, 4, 5)]
    for H, D  in [(4, 64), (32, 128)]
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical correctness benchmark for the ragged attention kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir",  default="results",
                        help="Output directory for CSV (default: results)")
    parser.add_argument("--csv-name", default="correctness_benchmark.csv")
    parser.add_argument("--fast",    action="store_true",
                        help="Run a small subset of configs for quick CI")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sequence error details")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for correctness benchmark.")
        sys.exit(1)

    if args.fast:
        configs = _FAST_CONFIGS
        print("Running FAST subset for CI …")
    else:
        # Deduplicate while preserving order
        seen = set()
        configs = []
        for c in _STANDARD_CONFIGS + _LLAMA_CONFIGS + _EDGE_CONFIGS:
            if c not in seen:
                seen.add(c)
                configs.append(c)

    print(f"Correctness benchmark: {len(configs)} configurations")
    print(f"Reference: SDPBackend.MATH (bit-exact, FA-independent)")
    print(f"Tolerance: atol={ATOL}, rtol={RTOL}\n")

    rows: List[CorrectnessRow] = []
    n_pass = 0
    n_fail = 0

    # column widths for aligned output
    W = {
        "B": 3, "b": 2, "d": 2, "N": 6, "H": 4, "D": 5,
        "status": 6, "peak_abs": 10, "mean_abs": 10, "peak_rel": 10,
    }
    hdr = (
        f"{'B':>3s}  {'b':>2s}  {'d':>2s}  {'N':>6s}  "
        f"{'H':>4s}  {'D':>5s}  {'status':>6s}  "
        f"{'peak_abs':>10s}  {'mean_abs':>10s}  {'peak_rel':>10s}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for ci, (B, b, d, H, D) in enumerate(configs):
        row = check_one(
            batch_size=B,
            branching_factor=b,
            depth=d,
            num_heads=H,
            head_dim=D,
            verbose=args.verbose,
        )
        rows.append(row)

        tag = " PASS" if row.passed else "*FAIL"
        if row.passed:
            n_pass += 1
        else:
            n_fail += 1

        print(
            f"{B:3d}  {b:2d}  {d:2d}  {row.num_tokens:6d}  "
            f"{H:4d}  {D:5d}  {tag:>6s}  "
            f"{row.peak_abs_err:10.3e}  "
            f"{row.mean_abs_err:10.3e}  "
            f"{row.peak_rel_err:10.3e}"
        )

    print(sep)

    # Global aggregates
    if rows:
        g_peak_abs = max(r.peak_abs_err for r in rows)
        g_mean_abs = sum(r.mean_abs_err for r in rows) / len(rows)
        g_peak_rel = max(r.peak_rel_err for r in rows)
        g_mean_rel = sum(r.mean_rel_err for r in rows) / len(rows)
        print(
            f"\nGlobal  |  peak_abs={g_peak_abs:.3e}  "
            f"mean_abs={g_mean_abs:.3e}  "
            f"peak_rel={g_peak_rel:.3e}  "
            f"mean_rel={g_mean_rel:.3e}"
        )

    print(
        f"\nResult  |  {n_pass} PASSED  {n_fail} FAILED  "
        f"out of {len(rows)} configurations\n"
    )

    # ── CSV output ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    field_names = [f.name for f in fields(CorrectnessRow)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"Saved: {csv_path}  ({len(rows)} rows)")
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
