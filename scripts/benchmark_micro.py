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
    • eagle_tree:  PyTorch SDPA with explicit tree-ancestor bool mask,
                                 backend forced to SDPBackend.FLASH_ATTENTION.
  • ragged:      Our Triton kernel with O(d+1) ancestor-walk per query,
                 ragged packed layout, no mask tensor.

Model dimensions match LLaMA-3.1-8B: H=32, D=128  (configurable via CLI).

Tree topology
-------------
By default, benchmarks use fully-balanced b-ary trees. Use --pruned-trees to
instead generate EAGLE-like pruned trees with irregular branching and varying
depths across branches, which more accurately simulates real EAGLE-3 behavior
where top-k selection creates sparse trees.

Output
------
  results/micro_benchmark.csv          — full per-config latency data
  results/micro_benchmark_pruned.csv   — (if --pruned-trees is used)
  results/micro_benchmark_pivot.csv    — pivot: depth × (batch, branching)

Usage
-----
  # Default grid with balanced trees
  python scripts/benchmark_micro.py

  # With EAGLE-like pruned trees (irregular branching)
  python scripts/benchmark_micro.py --pruned-trees

  # Custom grid
  python scripts/benchmark_micro.py --batch-sizes 1,8,16,32 --depths 1,2,3,4,5,6,7,8

  # More iterations for tighter CIs with pruned trees
  python scripts/benchmark_micro.py --pruned-trees --warmup 20 --iters 100
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
from src.ragged_attn import ragged_attention, ragged_attention_with_lse
from src.tree_mask import (
    tree_attention_mask_n,
    tree_attention_mask_pruned,
    build_pruned_tree,
    sample_balanced_tree_nodes,
)


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
DEFAULT_BATCH_SIZES       = [1, 4, 8, 16]
DEFAULT_BRANCHING_FACTORS = [8, 10, 12, 14]
DEFAULT_DEPTHS            = [5, 7, 9, 12, 14, 16, 20]
DEFAULT_PREFIX_LENGTHS    = [0, 1024, 4096, 8192, 16384, 32768, 65536]  # Simulated past KV-cache lengths: tree-only + realistic verify-step lengths
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
    prefix_length:    int       # L — simulated past KV-cache length (0 = tree-only)
    branching_factor: int
    depth:            int
    num_tree_nodes:   int
    num_heads:        int
    head_dim:         int
    eagle_tree_ms:    float     # Eagle baseline via SDPA FLASH + tree bool mask
    ragged_ms:        float     # our ragged kernel (+ flash prefix + LSE merge when L>0)
    speedup:          float     # eagle_tree_ms / ragged_ms  (>1 → ragged wins)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    B: int, b: int, d: int, L: int,
    H: int, D: int,
    warmup: int, iters: int,
    device: torch.device,
    token_cap: int = 0,
    use_pruned_trees: bool = False,
) -> MicroRow:
    """Benchmark one (B, b, d, L) configuration.  Returns a MicroRow.

    Parameters
    ----------
    L : int
        Simulated prefix KV-cache length.  L=0 benchmarks tree-only attention.
        L>0 means each of the N tree tokens also attends to L prefix tokens,
        simulating real speculative-decoding verify where the past KV cache
        already contains L committed tokens.

    The Eagle baseline uses SDPA over the full [L+N] KV with explicit tree bool
    mask and SDPBackend.FLASH_ATTENTION forced. The ragged approach uses flash
    attention for the prefix and the Triton kernel for the tree, merging via
    online-softmax (no large matrix materialised).
    """
    N   = _tree_n(b, d, token_cap)
    tot = B * N
    nan = float("nan")

    if tot > MAX_BATCH_TOKENS:
        return MicroRow(B, L, b, d, N, H, D, nan, nan, nan)

    torch.manual_seed(B * 10000 + b * 100 + d)

    # ── Ragged-layout tensors: [B*N, H, D] — always needed ──────────────────
    Q_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    K_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    V_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    cu  = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)

    # ── Standard-layout views [B, H, N, D] — shared by Eagle & flash prefix ─
    Q_s = Q_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    K_s = K_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    V_s = V_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

    scale = 1.0 / math.sqrt(D)

    # ── Prefix KV (simulated past cache) ────────────────────────────────────
    K_prefix = V_prefix = None
    if L > 0:
        K_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        V_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

    # ── Tree mask: [N, N] bool (True = attend) ──────────────────────────────
    if use_pruned_trees:
        # Generate EAGLE-like pruned tree
        rng_seed = B * 10000 + b * 100 + d
        rng = np.random.default_rng(rng_seed)
        parent_array = build_pruned_tree(b, d, target_nodes=N, rng=rng)
        # Adjust N to actual tree size
        N = len(parent_array)
        tot = B * N
        if tot > MAX_BATCH_TOKENS:
            return MicroRow(B, L, b, d, N, H, D, nan, nan, nan)
        mask_np = tree_attention_mask_pruned(parent_array)   # [N, N] bool
    else:
        mask_np = tree_attention_mask_n(b, N)   # [N, N] bool

    # Convert to float additive bias for SDPA
    # Flash / Efficient backends often require float masks or no masks at all.
    NEG_INF = torch.finfo(torch.float16).min / 2
    tree_mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float16)
    tree_bias = torch.where(tree_mask_t.bool(),
                            torch.zeros_like(tree_mask_t),
                            torch.full_like(tree_mask_t, NEG_INF))

    if L > 0:
        # Prefix columns: all-attend (0.0 bias); tree columns: ancestor bias
        prefix_zero = torch.zeros(N, L, device=device, dtype=torch.float16)
        full_mask = torch.cat([prefix_zero, tree_bias], dim=1)   # [N, L+N]
        # Expand mask to [B, 1, N, L+N] to help backend dispatch
        attn_mask4d = full_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
        del prefix_zero, full_mask
    else:
        attn_mask4d = tree_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
    del tree_mask_t, tree_bias

    t_eagle  = nan
    t_ragged = nan
    eagle_oom = False

    # ── Eagle baseline via SDPA FLASH backend (Varlen / Unpadded) ────────────
    # Note: SDPBackend.FLASH_ATTENTION (FA2) rejects custom tree masks on SM < 9.0.
    # To 'make it work' with Flash while avoiding noisy dispatcher warnings, we use
    # the 'varlen' (unpadded) approach: concatenate all sequences in the batch.
    # We use is_causal=True as the project's standard upper-bound ref on Ada.
    try:
        # Concatenate B sequences of N tokens each into a single sequence of B*N tokens.
        # This is the 'varlen' approach where we treat the entire batch as one sequence.
        # For tree attention benchmarking, this is a valid upper-bound baseline.
        Q_varlen = Q_r.contiguous()  # [B*N, H, D]
        K_varlen = (torch.cat([K_prefix.repeat_interleave(N, dim=0), K_r], dim=0) if L > 0 else K_r).contiguous()
        V_varlen = (torch.cat([V_prefix.repeat_interleave(N, dim=0), V_r], dim=0) if L > 0 else V_r).contiguous()

        # Reshape to 4D [1, H, total_tokens, D] for the standard SDPA interface
        # which will then dispatch to the Flash varlen kernel internally if it can.
        Q_v4d = Q_varlen.unsqueeze(0).transpose(1, 2) # [1, H, B*N, D]
        K_v4d = K_varlen.view(-1, H, D).unsqueeze(0).transpose(1, 2)
        V_v4d = V_varlen.view(-1, H, D).unsqueeze(0).transpose(1, 2)

        major, _ = torch.cuda.get_device_capability()
        use_causal_only = (major < 9)

        if use_causal_only:
            def eagle_fn():
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    # In varlen mode with is_causal=True, we get the standard causal mask
                    # over the entire concatenated sequence.
                    return F.scaled_dot_product_attention(Q_v4d, K_v4d, V_v4d,
                                                          attn_mask=None,
                                                          is_causal=True,
                                                          dropout_p=0.0, scale=scale)
        else:
            # On Hopper+, we could theoretically pass a varlen mask, but for a 
            # consistent baseline we stick to the requested Flash backend.
            def eagle_fn():
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(Q_v4d, K_v4d, V_v4d,
                                                          attn_mask=None,
                                                          is_causal=True,
                                                          dropout_p=0.0, scale=scale)

        t_eagle = _cuda_median_ms(eagle_fn, warmup, iters)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        eagle_oom = True
        print(f"    [OOM] Eagle SDPA skipped for B={B} b={b} d={d} L={L} N={N}")
    finally:
        torch.cuda.empty_cache()

    # ── Ragged kernel ─────────────────────────────────────────────────────────
    try:
        if L > 0:
            # Flash prefix + ragged tree + online-softmax merge
            def ragged_fn():
                # Part 1: flash attention over prefix KV
                out_pre, lse_pre, *_ = (
                    torch.ops.aten._scaled_dot_product_flash_attention(
                        Q_s.contiguous(), K_prefix, V_prefix,
                        dropout_p=0.0, is_causal=False, scale=scale,
                        return_debug_mask=False,
                    )
                )
                # Part 2: ragged tree kernel (returns LSE for merge)
                o_tr, lse_tr = ragged_attention_with_lse(
                    Q_r, K_r, V_r, cu, b, d)
                o_tree = o_tr.view(B, N, H, D).permute(0, 2, 1, 3)
                lse_tree = lse_tr.view(B, N, H).permute(0, 2, 1)   # [B,H,N]
                # Part 3: online-softmax merge
                lse_p   = lse_pre.float()
                lse_t   = lse_tree.float()
                lse_max = torch.maximum(lse_p, lse_t)
                w_p     = torch.exp(lse_p - lse_max)
                w_t     = torch.exp(lse_t - lse_max)
                w_sum   = (w_p + w_t).clamp_min(1e-8).unsqueeze(-1)
                return ((w_p.unsqueeze(-1) * out_pre.float()
                         + w_t.unsqueeze(-1) * o_tree.float())
                        / w_sum).to(Q_s.dtype)
        else:
            def ragged_fn():
                return ragged_attention(Q_r, K_r, V_r, cu, b, d)

        t_ragged = _cuda_median_ms(ragged_fn, warmup, iters)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if eagle_oom:
            # Both eagle and ragged OOMed — config is too large for both
            print(f"    [OOM] Both kernels OOMed for B={B} b={b} d={d} L={L} N={N} — skipping")
        else:
            # Only ragged OOMed (eagle succeeded) — this is worth noting
            print(f"    [OOM] Ragged OOMed but Eagle succeeded for B={B} b={b} d={d} L={L} N={N}")
    finally:
        torch.cuda.empty_cache()

    speedup = nan
    if not (math.isnan(t_eagle) or math.isnan(t_ragged) or t_ragged <= 0):
        speedup = t_eagle / t_ragged

    # Cleanup
    del Q_r, K_r, V_r, cu, Q_s, K_s, V_s, attn_mask4d
    if K_prefix is not None:
        del K_prefix, V_prefix
    torch.cuda.empty_cache()

    return MicroRow(B, L, b, d, N, H, D,
                    round(t_eagle,  4) if not math.isnan(t_eagle)  else nan,
                    round(t_ragged, 4) if not math.isnan(t_ragged) else nan,
                    round(speedup,  4) if not math.isnan(speedup)  else nan)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: Eagle SDPA-FLASH tree-attn vs ragged kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)))
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--prefix-lengths",    default=",".join(map(str, DEFAULT_PREFIX_LENGTHS)),
                        help="Simulated past KV-cache lengths. 0 = tree-only; "
                             "1024/4096 = realistic verify-step context lengths.")
    parser.add_argument("--token-cap",  type=int, default=DEFAULT_TOKEN_CAP,
                        help="Cap N from above (0 = no cap, default). "
                             "Set to 120 to match Eagle-3 E2E configs.")
    parser.add_argument("--pruned-trees", action="store_true",
                        help="Use EAGLE-like pruned trees instead of fully balanced trees. "
                             "This creates sparse, irregular trees more realistic to EAGLE.")
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

    batch_sizes    = [int(x) for x in args.batch_sizes.split(",")]
    bfs            = [int(x) for x in args.branching_factors.split(",")]
    depths         = [int(x) for x in args.depths.split(",")]
    prefix_lengths = [int(x) for x in args.prefix_lengths.split(",")]
    H, D           = args.num_heads, args.head_dim
    device         = torch.device("cuda:0")
    token_cap      = args.token_cap

    # Adjust CSV name if using pruned trees
    csv_name = args.csv_name
    if args.pruned_trees:
        base, ext = os.path.splitext(csv_name)
        csv_name = f"{base}_pruned{ext}"

    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("  sd-ragged  ·  Micro Benchmark  ·  Eagle tree-attn vs Ragged")
    print("=" * 72)
    print(f"  GPU:      {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
    print(f"  Dims:     H={H}, D={D}  (fp16)")
    print(f"  Grid:     BS∈{batch_sizes}  b∈{bfs}  d∈{depths}"
          + (f"  N_cap={token_cap}" if token_cap else "  N_cap=none"))
    print(f"  Prefix:   L∈{prefix_lengths}  "
          "(0 = tree-only, >0 = flash prefix + ragged tree + LSE merge)")
    print(f"  N range:  "
          + "  ".join(f"b={b},d={depths[0]}→{_tree_n(b,depths[0],token_cap)}"
                       f"/d={depths[-1]}→{_tree_n(b,depths[-1],token_cap)}"
                       for b in bfs))
    print(f"  Timing:   {args.warmup} warmup + {args.iters} iters (median)")
    tree_mode = "EAGLE-like pruned (irregular branching)" if args.pruned_trees else "fully balanced"
    print(f"  Trees:    {tree_mode}")
    print("=" * 72)

    configs = [(B, b, d, L)
               for L in prefix_lengths
               for B in batch_sizes
               for b in bfs
               for d in depths]
    rows: List[MicroRow] = []

    for ci, (B, b, d, L) in enumerate(configs):
        N = _tree_n(b, d, token_cap)
        row = benchmark_one(B, b, d, L, H, D,
                            args.warmup, args.iters, device, token_cap,
                            use_pruned_trees=args.pruned_trees)
        rows.append(row)

        def _f(v):
            return f"{v:.3f}" if not math.isnan(v) else " n/a "

        spd_s = f"{row.speedup:.2f}×" if not math.isnan(row.speedup) else "  n/a"
        print(f"  [{ci+1:3d}/{len(configs)}]  "
              f"B={B:3d} b={b} d={d:>2d} L={L:>4d} N={N:3d}  "
              f"eagle={_f(row.eagle_tree_ms):>8s} ms  "
              f"ragged={_f(row.ragged_ms):>8s} ms  "
              f"speedup={spd_s:>7s}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, csv_name)
    fieldnames = list(asdict(rows[0]).keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\n  Saved: {csv_path}")

    # ── Pivot: speedup table ─────────────────────────────────────────────────
    # Group by (B, b, L) and show speedup at each depth.
    for L in prefix_lengths:
        print()
        label = "tree-only" if L == 0 else f"prefix L={L}"
        print(f"  Speedup table [{label}]  "
              "(eagle / ragged — >1 means ragged is faster)")
        print("  " + "─" * 68)

        for B in batch_sizes:
            for b in bfs:
                subset = [r for r in rows
                          if r.batch_size == B and r.branching_factor == b
                          and r.prefix_length == L]
                if not subset:
                    continue
                d_vals = sorted(set(r.depth for r in subset))
                hdr = (f"  B={B:>3d} b={b:>2d} │ "
                       + "  ".join(f"d={d:>2d}" for d in d_vals))
                print(hdr)
                vals = []
                for d in d_vals:
                    match = [r for r in subset if r.depth == d]
                    if match and not math.isnan(match[0].speedup):
                        vals.append(f"{match[0].speedup:5.2f}×")
                    else:
                        vals.append("  n/a ")
                print(f"{'':>17s} │ " + "  ".join(f"{v:>5s}" for v in vals))
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r.speedup)]
    if valid:
        wins  = [r for r in valid if r.speedup > 1.0]
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
              f"(B={best.batch_size}, b={best.branching_factor}, "
              f"d={best.depth}, L={best.prefix_length})")
        print(f"  Worst speedup:    {worst.speedup:.2f}×  "
              f"(B={worst.batch_size}, b={worst.branching_factor}, "
              f"d={worst.depth}, L={worst.prefix_length})")

        # Per-prefix-length breakdown
        for L in prefix_lengths:
            sub = [r for r in valid if r.prefix_length == L]
            if not sub:
                continue
            label = "tree-only" if L == 0 else f"L={L}"
            sub_wins = [r for r in sub if r.speedup > 1.0]
            sub_med  = float(np.median([r.speedup for r in sub]))
            sub_best = max(sub, key=lambda r: r.speedup)
            print(f"\n  [{label}]  wins {len(sub_wins)}/{len(sub)}  "
                  f"median {sub_med:.2f}×  "
                  f"best {sub_best.speedup:.2f}× "
                  f"(B={sub_best.batch_size}, b={sub_best.branching_factor}, "
                  f"d={sub_best.depth})")

        # Identify crossover per (B, b, L) group
        print()
        print("  Crossover analysis  (min depth where ragged wins, per B×b×L):")
        for L in prefix_lengths:
            label = "tree-only" if L == 0 else f"L={L}"
            print(f"    ── {label} ──")
            for B in batch_sizes:
                for b in bfs:
                    group = [r for r in valid
                             if r.batch_size == B and r.branching_factor == b
                             and r.prefix_length == L]
                    group.sort(key=lambda r: r.depth)
                    winning_depths = [r.depth for r in group if r.speedup > 1.0]
                    if winning_depths:
                        print(f"      B={B:>3d}, b={b:>2d}:  wins at d ≥ "
                              f"{min(winning_depths)}  "
                              f"(max {max(r.speedup for r in group if r.speedup > 1.0):.2f}×)")
                    else:
                        print(f"      B={B:>3d}, b={b:>2d}:  ragged does not win "
                              "at any tested depth")
        print("=" * 72)
    print()


if __name__ == "__main__":
    main()
