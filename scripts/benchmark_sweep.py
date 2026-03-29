"""
benchmark_sweep.py
==================
End-to-end benchmark comparing:

  ① Triton ragged attention  (this work)
  ② PyTorch SDPA math-backend, padded  (SOTA baseline / current vLLM path)

across the full (batch_size, γ/tree_depth, branching_factor) sweep on
two T4 GPUs (SM75, 15 GB each).  Uses CUDA Events for accurate timing.

Metrics recorded
----------------
  ragged_ms      : median kernel latency in ms
  sdpa_ms        : median SDPA latency in ms
  speedup        : sdpa_ms / ragged_ms
  ragged_tflops  : effective TFLOPS of ragged kernel (based on valid tokens only)
  sdpa_tflops    : effective TFLOPS of SDPA        (based on padded computation)

Outputs
-------
  results/benchmark.csv
  results/speedup_heatmap_b<b>.png  (one per branching factor)
  results/tflops_comparison.png
  results/speedup_3d.png            (speedup surface over B × depth)

Usage
-----
  python scripts/benchmark_sweep.py [--out-dir results] [--no-plot]
  python scripts/benchmark_sweep.py --warmup 5 --iters 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.tree_mask import tree_attention_mask, num_tree_nodes
from src.ragged_attn import pack_inputs, ragged_attention

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

BATCH_SIZES       = [1, 2, 4, 8, 16, 32]
DEPTHS            = [1, 2, 3, 4, 5]
BRANCHING_FACTORS = [2, 3, 4]
CTX_LEN           = 128
NUM_HEADS         = 8
HEAD_DIM          = 64
WARMUP_ITERS      = 10
BENCH_ITERS       = 50


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _cuda_median_ms(fn, warmup: int, iters: int) -> float:
    """Time *fn()* using CUDA events and return median latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_e.record()
        fn()
        end_e.record()
        torch.cuda.synchronize()
        times.append(start_e.elapsed_time(end_e))   # ms
    return float(np.median(times))


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def ragged_flops(seq_lens: list[int], H: int, D: int) -> float:
    """FLOPs for ragged attention (only valid tokens, no padding waste)."""
    # QK^T: 2 * L_i^2 * D per head; AV: same → 4 * sum(L_i^2) * D * H
    return 4.0 * sum(l * l for l in seq_lens) * D * H


def padded_flops(seq_lens: list[int], H: int, D: int) -> float:
    """FLOPs for padded attention (includes padding waste)."""
    L_max = max(seq_lens)
    B     = len(seq_lens)
    return 4.0 * B * L_max * L_max * D * H


# ---------------------------------------------------------------------------
# SDPA reference runner (padded, tree mask, math backend)
# ---------------------------------------------------------------------------

def _build_sdpa_attn_bias(
    seq_lens: list[int],
    masks_np: list[np.ndarray],
    L_max: int,
    device: torch.device,
) -> torch.Tensor:
    NEG_INF = torch.finfo(torch.float16).min / 2
    B = len(seq_lens)
    bias = torch.full((B, 1, L_max, L_max), NEG_INF,
                      dtype=torch.float16, device=device)
    for i, (Li, m) in enumerate(zip(seq_lens, masks_np)):
        tree_t = torch.from_numpy(m).to(device=device, dtype=torch.float16)
        merged = torch.where(tree_t.bool(),
                             torch.zeros_like(tree_t),
                             torch.full_like(tree_t, NEG_INF))
        bias[i, 0, :Li, :Li] = merged
    return bias


def build_sdpa_inputs(
    qs: list[torch.Tensor],
    ks: list[torch.Tensor],
    vs: list[torch.Tensor],
    masks_np: list[np.ndarray],
    device: torch.device,
):
    """Return (Q_pad, K_pad, V_pad, attn_bias) in SDPA convention [B,H,L,D]."""
    B, H, D = len(qs), qs[0].shape[1], qs[0].shape[2]
    L_max   = max(q.shape[0] for q in qs)
    seq_lens = [q.shape[0] for q in qs]

    def _pad_stack(ts):
        out = torch.zeros(B, L_max, H, D, dtype=torch.float16, device=device)
        for i, t in enumerate(ts):
            out[i, :t.shape[0]] = t
        return out.permute(0, 2, 1, 3)   # [B, H, L_max, D]

    Q_p = _pad_stack(qs)
    K_p = _pad_stack(ks)
    V_p = _pad_stack(vs)
    bias = _build_sdpa_attn_bias(seq_lens, masks_np, L_max, device)
    return Q_p, K_p, V_p, bias


# ---------------------------------------------------------------------------
# Single benchmark point
# ---------------------------------------------------------------------------

def benchmark_one(
    batch_size: int,
    branching_factor: int,
    depth: int,
    num_heads: int      = NUM_HEADS,
    head_dim: int       = HEAD_DIM,
    ctx_len: int        = CTX_LEN,
    warmup: int         = WARMUP_ITERS,
    iters: int          = BENCH_ITERS,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda")

    torch.manual_seed(batch_size * 1000 + branching_factor * 100 + depth)

    N        = num_tree_nodes(branching_factor, depth)
    L_total  = ctx_len + N          # context + draft tree

    # For this benchmark we use the draft-tree portion only (the hot path)
    masks_np = [tree_attention_mask(branching_factor, depth)
                for _ in range(batch_size)]
    seq_lens = [N] * batch_size

    qs = [torch.randn(N, num_heads, head_dim,
                      device=device, dtype=torch.float16) for _ in range(batch_size)]
    ks = [torch.randn(N, num_heads, head_dim,
                      device=device, dtype=torch.float16) for _ in range(batch_size)]
    vs = [torch.randn(N, num_heads, head_dim,
                      device=device, dtype=torch.float16) for _ in range(batch_size)]

    # ---- Ragged inputs (pre-built, outside timed region) ----
    Q_r, K_r, V_r, cu_sl, pm, cmo = pack_inputs(qs, ks, vs, masks_np)
    Q_r = Q_r.to(device); K_r = K_r.to(device); V_r = V_r.to(device)

    def ragged_fn():
        ragged_attention(Q_r, K_r, V_r, cu_sl, pm, cmo)

    # ---- SDPA padded inputs (pre-built, outside timed region) ----
    Q_p, K_p, V_p, attn_bias = build_sdpa_inputs(qs, ks, vs, masks_np, device)
    scale = 1.0 / math.sqrt(head_dim)

    def sdpa_fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        ):
            F.scaled_dot_product_attention(
                Q_p, K_p, V_p, attn_mask=attn_bias, scale=scale
            )

    # ---- Time both ----
    try:
        ragged_ms = _cuda_median_ms(ragged_fn, warmup, iters)
    except Exception as exc:
        warnings.warn(f"Ragged kernel failed: {exc}")
        ragged_ms = float("nan")

    try:
        sdpa_ms = _cuda_median_ms(sdpa_fn, warmup, iters)
    except Exception as exc:
        warnings.warn(f"SDPA failed: {exc}")
        sdpa_ms = float("nan")

    # ---- FLOPs / TFLOPS ----
    r_flops   = ragged_flops(seq_lens, num_heads, head_dim)
    p_flops   = padded_flops(seq_lens, num_heads, head_dim)
    r_tflops  = r_flops / (ragged_ms * 1e-3) / 1e12 if not math.isnan(ragged_ms) else float("nan")
    p_tflops  = p_flops / (sdpa_ms   * 1e-3) / 1e12 if not math.isnan(sdpa_ms)   else float("nan")
    speedup   = sdpa_ms / ragged_ms if (not math.isnan(ragged_ms) and ragged_ms > 0) else float("nan")

    L_max = N
    attn_pad = 1.0 - sum(l * l for l in seq_lens) / (batch_size * L_max * L_max)

    return {
        "batch_size":        batch_size,
        "branching_factor":  branching_factor,
        "tree_depth":        depth,
        "num_tree_nodes":    N,
        "L_max":             L_max,
        "attn_padding_ratio": round(attn_pad, 4),
        "ragged_ms":         round(ragged_ms, 4),
        "sdpa_ms":           round(sdpa_ms, 4),
        "speedup":           round(speedup, 3),
        "ragged_tflops":     round(r_tflops, 3),
        "sdpa_tflops":       round(p_tflops, 3),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _speedup_heatmap(df: pd.DataFrame, b: int, out_path: str) -> None:
    sub = df[df["branching_factor"] == b]
    pivot = sub.pivot_table(
        index="tree_depth", columns="batch_size", values="speedup", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn",
        vmin=0.5, vmax=max(2.5, pivot.values.max()),  # type: ignore
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Speedup  (SDPA / Ragged)")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index,   fontsize=9)
    ax.set_xlabel("Batch size  B")
    ax.set_ylabel("Tree depth  d")
    ax.set_title(f"Speedup: Ragged vs SDPA  (b={b}, H={NUM_HEADS}, D={HEAD_DIM})")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            color = "white" if val < 1.2 else "black"
            ax.text(c, r, f"{val:.2f}×", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def _tflops_plot(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label in [
        (axes[0], "ragged_tflops", "Ragged Attention"),
        (axes[1], "sdpa_tflops",   "SDPA (padded)"),
    ]:
        for b in sorted(df["branching_factor"].unique()):
            sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)]
            sub = sub.sort_values("tree_depth")
            ax.plot(sub["tree_depth"], sub[col], marker="o", label=f"b={b}")
        ax.set_xlabel("Tree depth  d")
        ax.set_ylabel("Effective TFLOPS")
        ax.set_title(f"{label}  (B=8)")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def _speedup_lines(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10  # type: ignore
    markers = ["o", "s", "D", "^", "v"]
    for bi, b in enumerate(sorted(df["branching_factor"].unique())):
        for Bi, B in enumerate(sorted(df["batch_size"].unique())):
            sub = df[(df["branching_factor"] == b) & (df["batch_size"] == B)]
            sub = sub.sort_values("tree_depth")
            ax.plot(sub["tree_depth"], sub["speedup"],
                    color=cmap(bi / 10),
                    marker=markers[Bi % len(markers)],
                    linestyle=["-", "--", ":", "-."][Bi % 4],
                    label=f"b={b} B={B}",
                    linewidth=1.4, markersize=4, alpha=0.85)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Tree depth  d")
    ax.set_ylabel("Speedup  (SDPA / Ragged)")
    ax.set_title(f"Ragged Speedup over PyTorch SDPA  (H={NUM_HEADS}, D={HEAD_DIM})")
    ax.legend(fontsize=6, ncol=4, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: str = "results", no_plot: bool = False,
         warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available – benchmark aborted.")
        sys.exit(1)

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    print(
        f"Device: {props.name}  SM{props.major}{props.minor}  "
        f"{props.total_memory // 1024**3} GB"
    )

    os.makedirs(out_dir, exist_ok=True)

    configs = [
        (B, b, d)
        for B in BATCH_SIZES
        for b in BRANCHING_FACTORS
        for d in DEPTHS
    ]
    total = len(configs)
    rows  = []

    print(f"\nBenchmarking {total} configurations …")
    for idx, (B, b, d) in enumerate(configs):
        row = benchmark_one(B, b, d, warmup=warmup, iters=iters, device=device)
        rows.append(row)
        print(
            f"  [{idx+1:3d}/{total}]  B={B:2d} b={b} d={d}  "
            f"ragged={row['ragged_ms']:.3f}ms  sdpa={row['sdpa_ms']:.3f}ms  "
            f"speedup={row['speedup']:.2f}×  "
            f"ragged_tflops={row['ragged_tflops']:.2f}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary
    print("\n── Speedup summary (mean over B) ──")
    print(
        df.groupby(["tree_depth", "branching_factor"])["speedup"]
        .mean().round(3).unstack(level=1).to_string()
    )

    if no_plot:
        return

    for b in BRANCHING_FACTORS:
        _speedup_heatmap(
            df, b,
            out_path=os.path.join(out_dir, f"speedup_heatmap_b{b}.png")
        )

    _tflops_plot(df, out_path=os.path.join(out_dir, "tflops_comparison.png"))
    _speedup_lines(df, out_path=os.path.join(out_dir, "speedup_lines.png"))

    print("\nBenchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ragged vs SDPA benchmark sweep")
    parser.add_argument("--out-dir",  default="results")
    parser.add_argument("--no-plot",  action="store_true")
    parser.add_argument("--warmup",   type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",    type=int, default=BENCH_ITERS)
    args = parser.parse_args()
    main(out_dir=args.out_dir, no_plot=args.no_plot,
         warmup=args.warmup, iters=args.iters)
