#!/usr/bin/env python3
"""
benchmark_micro.py — Focused Kernel Micro-benchmark for Paper Figures
======================================================================

Compares the ragged ancestor-sparse attention kernel against Eagle-3's
actual tree-attention implementation across a targeted grid designed for the
paper's main result figure:

    BS ∈ {8, 16, 32, 64}   batch sizes (typical serving up to continuous-batching)
    b  ∈ {8, 10, 12}       branching factors  (bracket Eagle-3 default top_k=10)
    d  ∈ {5 … 20}          tree depths — pushed far past the E2E sweep to locate
                           the kernel crossover and show the scaling trend clearly

Eagle-3's default is (b=10, d=7, N≈60).  The kernel dispatch overhead dominates
at small N; this sweep intentionally goes beyond realistic EAGLE configs to
establish the crossover point and the asymptotic speedup ceiling.

Both kernels compute IDENTICAL attention over the same b-ary tree topology:
  • eagle_tree:  manual matmul + additive bias mask + softmax + matmul,
                 exactly matching modeling_llama_kv.py line-by-line:
                   attn_weights = matmul(Q,[B,H,N,D], K^T,[B,H,D,N]) / sqrt(D)
                   attn_weights = attn_weights + bias_mask   # -inf for non-ancestors
                   attn_weights = softmax(attn_weights, dim=-1).to(fp16)
                   attn_output  = matmul(attn_weights, V,[B,H,N,D])
  • ragged:      Our Triton kernel with O(d+1) ancestor-walk per query,
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import ragged_attention
from src.tree_mask import tree_attention_mask_n


def _tree_n(b: int, d: int, token_cap: int = 0) -> int:
    """N for a given (b, d) point in the sweep.

    Formula:  round(6 * b * d / 7), minimum 30.
    ``token_cap`` clamps the result from above when > 0; set to 0 for no cap.
    The E2E benchmark uses cap=120 (Eagle-3 runtime max).  The micro-benchmark
    uses cap=0 so we can probe kernel scaling at larger N.
    """
    n = max(30, round(6 * b * d / 7))
    return min(n, token_cap) if token_cap > 0 else n

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
# Defaults designed to show the kernel crossover clearly.
# Eagle-3 operates at b∈{8-12}, d=7 (N~48-72).  We push d up to 20 so N
# reaches 200+ at b=12, revealing the asymptotic speedup ceiling.
DEFAULT_BATCH_SIZES       = [8, 16, 32, 64, 128]
DEFAULT_BRANCHING_FACTORS = [8, 10, 12, 14, 16, 18, 20]
DEFAULT_DEPTHS            = [5, 7, 9, 12, 14, 16, 20, 24, 28, 32]
DEFAULT_TOKEN_CAP         = 0     # 0 = no cap; E2E benchmark uses 120
DEFAULT_NUM_HEADS         = 32    # LLaMA-3.1-8B
DEFAULT_HEAD_DIM          = 128   # LLaMA-3.1-8B
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
    eagle_tree_ms:    float     # Eagle manual matmul+mask+softmax (modeling_llama_kv.py)
    ragged_ms:        float     # our ragged kernel
    speedup:          float     # eagle_tree_ms / ragged_ms  (>1 → ragged wins)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    B: int, b: int, d: int,
    H: int, D: int,
    warmup: int, iters: int,
    device: torch.device,
    token_cap: int = 0,
) -> MicroRow:
    """Benchmark one (B, b, d) configuration.  Returns a MicroRow.

    N = _tree_n(b, d, token_cap) — uncapped by default in the micro-bench so
    we can probe larger trees than Eagle-3 uses at runtime, locating the exact
    crossover depth where the ragged kernel overtakes manual matmul.
    """
    N   = _tree_n(b, d, token_cap)
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

    t_eagle  = nan
    t_ragged = nan

    # ── Eagle manual matmul attention (the actual baseline) ──────────────────
    # Replicates modeling_llama_kv.py exactly:
    #   attn_weights = matmul(Q, K^T) / sqrt(D) + bias_mask
    #   attn_weights = softmax(attn_weights, dim=-1, dtype=float32).to(fp16)
    #   attn_output  = matmul(attn_weights, V)
    # This is what we replace with the ragged kernel at tree_decoding time.
    try:
        # [B, H, N, D] layout expected by modeling_llama_kv.py
        Q_s = Q_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        K_s = K_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        V_s = V_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

        # Build additive bias mask [1, 1, N, N]: 0 for ancestors, -inf otherwise
        mask_np = tree_attention_mask_n(b, N)   # [N, N] bool
        mask_t  = torch.from_numpy(mask_np).to(device)
        bias4d  = torch.where(
            mask_t,
            torch.zeros(1, device=device, dtype=torch.float16),
            torch.full((1,), float("-inf"), device=device, dtype=torch.float16),
        ).unsqueeze(0).unsqueeze(0)   # [1, 1, N, N] — broadcast over B and H
        del mask_t

        scale = 1.0 / math.sqrt(D)

        def eagle_fn():
            # Exact replica of EAGLE's LlamaAttention.forward() tree-verify path
            w = torch.matmul(Q_s, K_s.transpose(-2, -1)) * scale  # [B,H,N,N]
            w = w + bias4d
            w = torch.softmax(w, dim=-1, dtype=torch.float32).to(Q_s.dtype)
            return torch.matmul(w, V_s)                            # [B,H,N,D]

        t_eagle = _cuda_median_ms(eagle_fn, warmup, iters)
        del Q_s, K_s, V_s, bias4d
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"    [OOM] Eagle matmul skipped for B={B} b={b} d={d} N={N}")
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
    if not (math.isnan(t_eagle) or math.isnan(t_ragged) or t_ragged <= 0):
        speedup = t_eagle / t_ragged

    # Cleanup
    del Q_r, K_r, V_r, cu
    torch.cuda.empty_cache()

    return MicroRow(B, b, d, N, H, D,
                    round(t_eagle,  4) if not math.isnan(t_eagle)  else nan,
                    round(t_ragged, 4) if not math.isnan(t_ragged) else nan,
                    round(speedup,  4) if not math.isnan(speedup)  else nan)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: Eagle manual-matmul tree-attn vs ragged kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)))
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--token-cap",  type=int, default=DEFAULT_TOKEN_CAP,
                        help="Cap N from above (0 = no cap, default). "
                             "Set to 120 to match Eagle-3 E2E configs.")
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
    token_cap   = args.token_cap

    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("  sd-ragged  ·  Micro Benchmark  ·  Eagle tree-attn vs Ragged")
    print("=" * 72)
    print(f"  GPU:      {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
    print(f"  Dims:     H={H}, D={D}  (fp16)")
    print(f"  Grid:     BS∈{batch_sizes}  b∈{bfs}  d∈{depths}"
          + (f"  N_cap={token_cap}" if token_cap else "  N_cap=none"))
    print(f"  N range:  "
          + "  ".join(f"b={b},d={depths[0]}→{_tree_n(b,depths[0],token_cap)}"
                       f"/d={depths[-1]}→{_tree_n(b,depths[-1],token_cap)}"
                       for b in bfs))
    print(f"  Timing:   {args.warmup} warmup + {args.iters} iters (median)")
    print("=" * 72)

    configs = [(B, b, d) for B in batch_sizes for b in bfs for d in depths]
    rows: List[MicroRow] = []

    for ci, (B, b, d) in enumerate(configs):
        N = _tree_n(b, d, token_cap)
        row = benchmark_one(B, b, d, H, D, args.warmup, args.iters, device, token_cap)
        rows.append(row)

        def _f(v):
            return f"{v:.3f}" if not math.isnan(v) else " n/a "

        spd_s = f"{row.speedup:.2f}×" if not math.isnan(row.speedup) else "  n/a"
        print(f"  [{ci+1:3d}/{len(configs)}]  "
              f"B={B:3d} b={b} d={d} N={N:3d}  "
              f"eagle={_f(row.eagle_tree_ms):>8s} ms  "
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
    print("  Speedup table  (eagle_tree / ragged — >1 means ragged is faster)")
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
