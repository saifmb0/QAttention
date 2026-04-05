#!/usr/bin/env python3
"""
benchmark_micro.py — Focused Kernel Micro-benchmark for Paper Figures
======================================================================

Compares the ragged ancestor-sparse attention kernel against PyTorch SDPA
with correct tree-ancestor masking across a targeted grid designed for the
paper's main result figure:

    BS ∈ {8, 16}     batch sizes (typical serving / continuous-batching)
    b  ∈ {2, 3, 4}   branching factors
    d  ∈ {1 … 8}     tree depths

Both kernels compute IDENTICAL attention over the same b-ary tree topology:
  • sdpa_tree:  PyTorch SDPA + float additive bias [B, 1, N, N] from
                tree_attention_mask(), forced to SDPBackend.MATH (the only
                backend accepting non-null additive masks — same kernel path
                as EAGLE-3's modeling_llama_kv.py).
  • ragged:     Our Triton kernel with O(d+1) ancestor-walk per query,
                ragged packed layout, no mask tensor.

Model dimensions match LLaMA-3.1-8B: H=32, D=128  (configurable via CLI).

Output
------
  results/micro_benchmark.csv          — full per-config latency data
  results/micro_benchmark_pivot.csv    — pivot: depth × (batch, branching)

Usage
-----
  # Default grid
  python scripts/benchmark_micro.py

  # Custom grid
  python scripts/benchmark_micro.py --batch-sizes 1,8,16,32 --depths 1,2,3,4,5,6,7,8

  # More iterations for tighter CIs
  python scripts/benchmark_micro.py --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import ragged_attention
from src.tree_mask import num_tree_nodes, tree_attention_mask

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES       = [8, 16]
DEFAULT_BRANCHING_FACTORS = [2, 3, 4]
DEFAULT_DEPTHS            = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_NUM_HEADS         = 32     # LLaMA-3.1-8B
DEFAULT_HEAD_DIM          = 128    # LLaMA-3.1-8B
WARMUP_ITERS              = 10
BENCH_ITERS               = 50

# Hard ceiling on total tokens per batch (B × N) to avoid OOM.
# At H=32, D=128, fp16: one Q/K/V set ≈ 3 × B×N × 32 × 128 × 2 bytes.
MAX_BATCH_TOKENS = 2_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_median_ms(fn, warmup: int, iters: int) -> float:
    """Median CUDA-event latency over `iters` calls (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MicroRow:
    batch_size:       int
    branching_factor: int
    depth:            int
    num_tree_nodes:   int
    num_heads:        int
    head_dim:         int
    sdpa_tree_ms:     float     # SDPA MATH backend + correct tree mask
    ragged_ms:        float     # our ragged kernel
    speedup:          float     # sdpa_tree_ms / ragged_ms  (>1 → ragged wins)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    B: int, b: int, d: int,
    H: int, D: int,
    warmup: int, iters: int,
    device: torch.device,
) -> MicroRow:
    """Benchmark one (B, b, d) configuration.  Returns a MicroRow."""
    N   = num_tree_nodes(b, d)
    tot = B * N
    nan = float("nan")

    if tot > MAX_BATCH_TOKENS:
        return MicroRow(B, b, d, N, H, D, nan, nan, nan)

    torch.manual_seed(B * 10000 + b * 100 + d)

    # ── Ragged tensors: [B*N, H, D] — always allocated ──────────────────────
    Q_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    K_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    V_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    cu  = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)

    t_sdpa   = nan
    t_ragged = nan

    # ── SDPA MATH + bias [B, 1, N, N] — O(N²) allocation, may OOM ──────────
    # All SDPA-related tensors are allocated and timed inside this block so
    # that a large-N OOM falls through gracefully.  The ragged kernel still
    # runs independently even if SDPA is skipped.
    try:
        # SDPA: [B, H, N, D]
        Q_s = Q_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        K_s = K_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        V_s = V_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

        # Tree-ancestor additive bias mask [B, 1, N, N]  (O(N²) — the bottleneck)
        mask_np = tree_attention_mask(b, d)   # [N, N] bool
        mask_t  = torch.from_numpy(mask_np).to(device)
        bias    = torch.where(
            mask_t,
            torch.zeros(1, device=device, dtype=torch.float16),
            torch.full((1,), float("-inf"), device=device, dtype=torch.float16),
        )
        bias4d = bias.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N).contiguous()
        del mask_t, bias

        def sdpa_fn():
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH
            ):
                return F.scaled_dot_product_attention(Q_s, K_s, V_s, attn_mask=bias4d)

        t_sdpa = _cuda_median_ms(sdpa_fn, warmup, iters)
        del Q_s, K_s, V_s, bias4d
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"    [OOM] SDPA skipped for B={B} b={b} d={d} N={N} "
              f"(bias4d would be {B * N * N * 2 / 1e9:.2f} GB)")
    finally:
        torch.cuda.empty_cache()

    # ── Ragged kernel ─────────────────────────────────────────────────────────
    def ragged_fn():
        return ragged_attention(Q_r, K_r, V_r, cu, b, d)

    try:
        t_ragged = _cuda_median_ms(ragged_fn, warmup, iters)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()

    speedup = nan
    if not (math.isnan(t_sdpa) or math.isnan(t_ragged) or t_ragged <= 0):
        speedup = t_sdpa / t_ragged

    # Cleanup
    del Q_r, K_r, V_r, cu
    torch.cuda.empty_cache()

    return MicroRow(B, b, d, N, H, D,
                    round(t_sdpa, 4)   if not math.isnan(t_sdpa)   else nan,
                    round(t_ragged, 4) if not math.isnan(t_ragged) else nan,
                    round(speedup, 4)  if not math.isnan(speedup)  else nan)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: SDPA tree-mask vs ragged kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)))
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--num-heads",  type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim",   type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--warmup",     type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",      type=int, default=BENCH_ITERS)
    parser.add_argument("--out-dir",    default="results")
    parser.add_argument("--csv-name",   default="micro_benchmark.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR]  CUDA required.")
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    bfs         = [int(x) for x in args.branching_factors.split(",")]
    depths      = [int(x) for x in args.depths.split(",")]
    H, D        = args.num_heads, args.head_dim
    device      = torch.device("cuda:0")

    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("  sd-ragged  ·  Micro Benchmark  ·  SDPA tree-mask vs Ragged")
    print("=" * 72)
    print(f"  GPU:      {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
    print(f"  Dims:     H={H}, D={D}  (fp16)")
    print(f"  Grid:     BS∈{batch_sizes}  b∈{bfs}  d∈{depths}")
    print(f"  Timing:   {args.warmup} warmup + {args.iters} iters (median)")
    print("=" * 72)

    configs = [(B, b, d) for B in batch_sizes for b in bfs for d in depths]
    rows: List[MicroRow] = []

    for ci, (B, b, d) in enumerate(configs):
        N = num_tree_nodes(b, d)
        row = benchmark_one(B, b, d, H, D, args.warmup, args.iters, device)
        rows.append(row)

        def _f(v):
            return f"{v:.3f}" if not math.isnan(v) else " n/a "

        spd_s = f"{row.speedup:.2f}×" if not math.isnan(row.speedup) else "  n/a"
        print(f"  [{ci+1:3d}/{len(configs)}]  "
              f"B={B:3d} b={b} d={d} N={N:6d}  "
              f"sdpa={_f(row.sdpa_tree_ms):>8s} ms  "
              f"ragged={_f(row.ragged_ms):>8s} ms  "
              f"speedup={spd_s:>7s}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    fieldnames = list(asdict(rows[0]).keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\n  Saved: {csv_path}")

    # ── Pivot: speedup table ─────────────────────────────────────────────────
    # Group by (B, b) and show speedup at each depth.
    print()
    print("  Speedup table  (sdpa_tree / ragged — >1 means ragged is faster)")
    print("  " + "─" * 68)

    for B in batch_sizes:
        for b in bfs:
            subset = [r for r in rows if r.batch_size == B and r.branching_factor == b]
            if not subset:
                continue
            d_vals = sorted(set(r.depth for r in subset))
            hdr = f"  B={B:>3d} b={b}  │ " + "  ".join(f"d={d:>2d}" for d in d_vals)
            print(hdr)
            vals = []
            for d in d_vals:
                match = [r for r in subset if r.depth == d]
                if match and not math.isnan(match[0].speedup):
                    vals.append(f"{match[0].speedup:5.2f}×")
                else:
                    vals.append("  n/a ")
            print(f"{'':>15s} │ " + "  ".join(f"{v:>5s}" for v in vals))
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r.speedup)]
    if valid:
        wins  = [r for r in valid if r.speedup > 1.0]
        loses = [r for r in valid if r.speedup <= 1.0]
        best  = max(valid, key=lambda r: r.speedup)
        worst = min(valid, key=lambda r: r.speedup)
        med   = float(np.median([r.speedup for r in valid]))

        print("=" * 72)
        print("  MICRO BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"  Configs tested:   {len(configs)}")
        print(f"  Valid results:    {len(valid)}")
        print(f"  Ragged wins:      {len(wins)} / {len(valid)}  "
              f"({100*len(wins)/len(valid):.0f}%)")
        print(f"  Median speedup:   {med:.2f}×")
        print(f"  Best speedup:     {best.speedup:.2f}×  "
              f"(B={best.batch_size}, b={best.branching_factor}, d={best.depth})")
        print(f"  Worst speedup:    {worst.speedup:.2f}×  "
              f"(B={worst.batch_size}, b={worst.branching_factor}, d={worst.depth})")

        # Identify crossover per (B, b) group
        print()
        print("  Crossover analysis  (min depth where ragged wins, per B×b):")
        for B in batch_sizes:
            for b in bfs:
                group = [r for r in valid if r.batch_size == B and r.branching_factor == b]
                group.sort(key=lambda r: r.depth)
                winning_depths = [r.depth for r in group if r.speedup > 1.0]
                if winning_depths:
                    print(f"    B={B:>3d}, b={b}:  wins at d ≥ {min(winning_depths)}  "
                          f"(max {max(r.speedup for r in group if r.speedup > 1.0):.2f}×)")
                else:
                    print(f"    B={B:>3d}, b={b}:  ragged does not win at any tested depth")
        print("=" * 72)
    print()


if __name__ == "__main__":
    main()
