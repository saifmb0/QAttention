#!/usr/bin/env python3
"""
generate_figures.py
===================
Produces all figures for the MLSys paper from raw benchmark data.

Usage:
    python paper/generate_figures.py

Reads:
    results.txt              – Blackwell RTX PRO 6000 raw console output
    results/sota_benchmark.csv – Blackwell structured CSV (earlier run, B≤32)
    results/benchmark.csv    – T4 dense-kernel baseline (master branch)
    results/padding_waste.csv– Padding waste sweep

Writes to paper/figures/:
    fig1_latency_vs_depth.pdf
    fig2_speedup_heatmap.pdf
    fig3_scaling_batch.pdf
    fig4_padding_waste.pdf
    fig5_cross_gpu.pdf
    fig6_bf16_parity.pdf
"""

from __future__ import annotations

import re
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_TXT = ROOT / "results.txt"
SOTA_CSV = ROOT / "results" / "sota_benchmark.csv"
T4_CSV = ROOT / "results" / "benchmark.csv"
PADDING_CSV = ROOT / "results" / "padding_waste.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Color palette (colorblind-safe, publication quality)
C_RAGGED = "#0077B6"    # Strong blue
C_RAGGED_BF16 = "#00B4D8"  # Light blue
C_SDPA_MATH = "#D62828"  # Red
C_SDPA_FLASH = "#F77F00"  # Orange
C_T4 = "#6C757D"         # Gray


# ── Parse results.txt ───────────────────────────────────────────────────────
def parse_results_txt(path: Path) -> pd.DataFrame:
    """Parse the non-standard console output into a DataFrame."""
    pattern = re.compile(
        r"B=\s*(\d+)\s+b=(\d+)\s+d=(\d+)\s*│\s*"
        r"ragged=([\d.]+)ms\s+bf16=([\d.]+)ms\s*│\s*"
        r"sdpa_math=([\d.]+|n/a)\s*ms?\s+"
        r"sdpa_flash=([\d.]+|n/a)\s*ms?"
    )
    rows = []
    with open(path) as f:
        for line in f:
            # Strip warning lines
            if "UserWarning" in line or "warnings.warn" in line or "CUDA out of memory" in line:
                continue
            m = pattern.search(line)
            if m:
                B, b, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                ragged = float(m.group(4))
                bf16 = float(m.group(5))
                sm = float(m.group(6)) if m.group(6) != "n/a" else float("nan")
                sf = float(m.group(7)) if m.group(7) != "n/a" else float("nan")
                # Compute N
                if b == 1:
                    N = d + 1
                else:
                    N = (b ** (d + 1) - 1) // (b - 1)
                rows.append({
                    "batch_size": B, "branching_factor": b, "tree_depth": d,
                    "num_tree_nodes": N,
                    "ragged_fp16_ms": ragged, "ragged_bf16_ms": bf16,
                    "sdpa_math_ms": sm, "sdpa_flash_ms": sf,
                })
    return pd.DataFrame(rows)


def num_tree_nodes(b, d):
    if b == 1:
        return d + 1
    return (b ** (d + 1) - 1) // (b - 1)


# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
df_bw = parse_results_txt(RESULTS_TXT)
print(f"  Blackwell results.txt: {len(df_bw)} rows")

# Also load the structured CSV for cross-check / B=1 data
if SOTA_CSV.exists():
    df_sota = pd.read_csv(SOTA_CSV)
    print(f"  Blackwell sota CSV:    {len(df_sota)} rows")
else:
    df_sota = pd.DataFrame()

if T4_CSV.exists():
    df_t4 = pd.read_csv(T4_CSV)
    print(f"  T4 benchmark CSV:      {len(df_t4)} rows")
else:
    df_t4 = pd.DataFrame()

if PADDING_CSV.exists():
    df_pad = pd.read_csv(PADDING_CSV)
    print(f"  Padding waste CSV:     {len(df_pad)} rows")
else:
    df_pad = pd.DataFrame()

# Merge B=1 from SOTA CSV into Blackwell if not in results.txt
if not df_sota.empty:
    b1_sota = df_sota[df_sota["batch_size"] == 1][
        ["batch_size", "branching_factor", "tree_depth", "num_tree_nodes",
         "ragged_fp16_ms", "ragged_bf16_ms", "sdpa_math_ms", "sdpa_flash_ms"]
    ].copy()
    # Only add rows not already in df_bw
    existing = set(zip(df_bw["batch_size"], df_bw["branching_factor"], df_bw["tree_depth"]))
    mask = ~b1_sota.apply(lambda r: (r["batch_size"], r["branching_factor"], r["tree_depth"]) in existing, axis=1)
    if mask.any():
        df_bw = pd.concat([b1_sota[mask], df_bw], ignore_index=True)
        df_bw.sort_values(["batch_size", "branching_factor", "tree_depth"], inplace=True)
        df_bw.reset_index(drop=True, inplace=True)
        print(f"  After merging B=1 from SOTA CSV: {len(df_bw)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Latency vs tree depth (log-scale, one panel per branching factor)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_latency_vs_depth():
    """3-panel subplot: b=2,3,4; lines for ragged/sdpa_math/sdpa_flash; B=8."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)

    B_show = 8  # representative batch size
    for ax, b in zip(axes, [2, 3, 4]):
        sub = df_bw[(df_bw["batch_size"] == B_show) & (df_bw["branching_factor"] == b)]
        sub = sub.sort_values("tree_depth")

        ax.plot(sub["tree_depth"], sub["ragged_fp16_ms"],
                "o-", color=C_RAGGED, markersize=4, linewidth=1.5,
                label="Ragged (ours)", zorder=5)
        ax.plot(sub["tree_depth"], sub["sdpa_math_ms"],
                "s--", color=C_SDPA_MATH, markersize=3.5, linewidth=1.2,
                label="SDPA math")
        ax.plot(sub["tree_depth"], sub["sdpa_flash_ms"],
                "^--", color=C_SDPA_FLASH, markersize=3.5, linewidth=1.2,
                label="SDPA flash")

        ax.set_yscale("log")
        ax.set_xlabel("Tree depth $d$")
        ax.set_title(f"$b = {b}$", fontweight="bold")
        ax.set_xticks(range(1, 9))

        # Annotate max speedup
        valid = sub.dropna(subset=["sdpa_math_ms"])
        if not valid.empty:
            idx = valid["sdpa_math_ms"].idxmax()
            spd = valid.loc[idx, "sdpa_math_ms"] / valid.loc[idx, "ragged_fp16_ms"]
            d_ann = valid.loc[idx, "tree_depth"]
            ax.annotate(f"{spd:.0f}×",
                        xy=(d_ann, valid.loc[idx, "ragged_fp16_ms"]),
                        xytext=(d_ann - 0.8, valid.loc[idx, "ragged_fp16_ms"] * 0.3),
                        fontsize=7, color=C_RAGGED, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C_RAGGED, lw=0.8))

    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[0].legend(loc="upper left", framealpha=0.9)

    fig.suptitle(f"Kernel latency vs. tree depth  ($B = {B_show}$, Blackwell RTX PRO 6000)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_latency_vs_depth.pdf")
    fig.savefig(FIG_DIR / "fig1_latency_vs_depth.png")
    plt.close(fig)
    print("  ✓ fig1_latency_vs_depth")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Speedup heatmaps (ragged vs sdpa_math AND ragged vs sdpa_flash)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_speedup_heatmap():
    """2×3 heatmap grid: top = vs sdpa_math, bottom = vs sdpa_flash."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0))

    for col, b in enumerate([2, 3, 4]):
        sub = df_bw[df_bw["branching_factor"] == b].copy()

        for row, (ref_col, title_suffix) in enumerate([
            ("sdpa_math_ms", "vs. SDPA math"),
            ("sdpa_flash_ms", "vs. SDPA flash"),
        ]):
            ax = axes[row, col]
            sub_valid = sub.dropna(subset=[ref_col])
            sub_valid = sub_valid[sub_valid[ref_col] > 0]
            sub_valid["speedup"] = sub_valid[ref_col] / sub_valid["ragged_fp16_ms"]

            # Pivot: rows=batch_size, cols=tree_depth
            pivot = sub_valid.pivot_table(
                index="batch_size", columns="tree_depth",
                values="speedup", aggfunc="mean"
            )

            if pivot.empty:
                ax.set_visible(False)
                continue

            if ref_col == "sdpa_math_ms":
                vmin, vmax = 1, 500
                norm = LogNorm(vmin=max(1, pivot.min().min()), vmax=max(2, pivot.max().max()))
                cmap = "YlOrRd"
            else:
                vmin, vmax = 0.3, 10
                norm = LogNorm(vmin=max(0.3, pivot.min().min()), vmax=max(2, pivot.max().max()))
                cmap = "RdYlGn"

            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm,
                           origin="lower")

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(c)) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(int(i)) for i in pivot.index])

            # Annotate cells
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        txt = f"{val:.0f}×" if val >= 10 else f"{val:.1f}×"
                        color = "white" if (ref_col == "sdpa_math_ms" and val > 20) else "black"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=5.5, color=color, fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"Batch size $B$\n({title_suffix})")
            if row == 1:
                ax.set_xlabel("Tree depth $d$")
            if row == 0:
                ax.set_title(f"$b = {b}$", fontweight="bold")

            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle("Speedup heatmaps — Ragged kernel (Blackwell RTX PRO 6000)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_speedup_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig2_speedup_heatmap.png")
    plt.close(fig)
    print("  ✓ fig2_speedup_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Scaling with batch size (fixed depths)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_scaling_batch():
    """Line plot: latency vs B for b=4, select depths."""
    b = 4
    # Pick 3 depths that actually have data
    avail_depths = sorted(df_bw[df_bw["branching_factor"] == b]["tree_depth"].unique())
    # Prefer [3, 5, 8] but fall back to whatever is available
    desired = [3, 5, 8]
    depths = [d for d in desired if d in avail_depths]
    if len(depths) < 3:
        depths = avail_depths[-3:] if len(avail_depths) >= 3 else avail_depths

    fig, axes = plt.subplots(1, len(depths), figsize=(7.0, 2.3), sharey=False)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        sub = df_bw[(df_bw["branching_factor"] == b) & (df_bw["tree_depth"] == d)]
        sub = sub.sort_values("batch_size")
        N = num_tree_nodes(b, d)

        if sub.empty:
            ax.set_visible(False)
            continue

        ax.plot(sub["batch_size"], sub["ragged_fp16_ms"],
                "o-", color=C_RAGGED, markersize=4, linewidth=1.5,
                label="Ragged fp16")
        ax.plot(sub["batch_size"], sub["ragged_bf16_ms"],
                "D-", color=C_RAGGED_BF16, markersize=3.5, linewidth=1.2,
                label="Ragged bf16")

        valid_math = sub.dropna(subset=["sdpa_math_ms"])
        if not valid_math.empty:
            ax.plot(valid_math["batch_size"], valid_math["sdpa_math_ms"],
                    "s--", color=C_SDPA_MATH, markersize=3.5, linewidth=1.2,
                    label="SDPA math")
        valid_flash = sub.dropna(subset=["sdpa_flash_ms"])
        if not valid_flash.empty:
            ax.plot(valid_flash["batch_size"], valid_flash["sdpa_flash_ms"],
                    "^--", color=C_SDPA_FLASH, markersize=3.5, linewidth=1.2,
                    label="SDPA flash")

        # Only use log scale if there's positive data
        batch_vals = sub["batch_size"].values
        if len(batch_vals) > 1 and all(v > 0 for v in batch_vals):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticks([v for v in [2, 4, 8, 16, 32, 64, 128] if v >= batch_vals.min() and v <= batch_vals.max()])
        ax.set_yscale("log")
        ax.set_xlabel("Batch size $B$")
        ax.set_title(f"$d = {d}$  ($N = {N:,}$)", fontweight="bold")

    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[-1].legend(loc="upper left", framealpha=0.9, fontsize=6.5)

    fig.suptitle(f"Latency scaling with batch size  ($b = {b}$, Blackwell RTX PRO 6000)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_scaling_batch.pdf")
    fig.savefig(FIG_DIR / "fig3_scaling_batch.png")
    plt.close(fig)
    print("  ✓ fig3_scaling_batch")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Padding waste characterization
# ─────────────────────────────────────────────────────────────────────────────
def fig4_padding_waste():
    """Bar chart: attention padding ratio for different tree configs."""
    if df_pad.empty:
        print("  ⚠ Skipping fig4 (no padding data)")
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    # Show B=8, ctx=128 subset
    sub = df_pad[(df_pad["batch_size"] == 8) & (df_pad["ctx_len"] == 128)]
    if sub.empty:
        sub = df_pad[df_pad["batch_size"] == df_pad["batch_size"].max()]

    sub = sub.sort_values(["branching_factor", "max_tree_depth"])
    labels = [f"b={int(r['branching_factor'])},d={int(r['max_tree_depth'])}"
              for _, r in sub.iterrows()]
    vals = sub["attn_padding_ratio"].values * 100  # percent

    colors = [C_RAGGED if v < 10 else C_SDPA_MATH for v in vals]
    bars = ax.bar(range(len(vals)), vals, color=C_SDPA_MATH, alpha=0.8, width=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Attention compute waste (%)")
    ax.set_title("Quadratic padding waste (B=8, ctx=128)", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)

    for bar, v in zip(bars, vals):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_padding_waste.pdf")
    fig.savefig(FIG_DIR / "fig4_padding_waste.png")
    plt.close(fig)
    print("  ✓ fig4_padding_waste")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Cross-GPU comparison (T4 dense-kernel vs Blackwell sparse-kernel)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_cross_gpu():
    """Grouped bar chart: T4 (dense, old kernel) vs Blackwell (sparse, new kernel)."""
    if df_t4.empty:
        print("  ⚠ Skipping fig5 (no T4 data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False)

    for ax, b in zip(axes, [2, 3, 4]):
        # T4: B=8, various d
        t4_sub = df_t4[(df_t4["batch_size"] == 8) & (df_t4["branching_factor"] == b)]
        t4_sub = t4_sub.sort_values("tree_depth")

        # Blackwell: B=8
        bw_sub = df_bw[(df_bw["batch_size"] == 8) & (df_bw["branching_factor"] == b)]
        bw_sub = bw_sub.sort_values("tree_depth")

        # Align on common depths
        common_d = sorted(set(t4_sub["tree_depth"]) & set(bw_sub["tree_depth"]))
        if not common_d:
            continue

        t4_lat = [t4_sub[t4_sub["tree_depth"] == d]["ragged_ms"].values[0] for d in common_d]
        bw_lat = [bw_sub[bw_sub["tree_depth"] == d]["ragged_fp16_ms"].values[0] for d in common_d]

        x = np.arange(len(common_d))
        w = 0.35
        ax.bar(x - w/2, t4_lat, w, color=C_T4, label="T4 (dense kernel)", alpha=0.85)
        ax.bar(x + w/2, bw_lat, w, color=C_RAGGED, label="Blackwell (sparse)", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in common_d])
        ax.set_xlabel("Tree depth $d$")
        ax.set_yscale("log")
        ax.set_title(f"$b = {b}$", fontweight="bold")

        # Speedup annotations
        for i, (t, bwl) in enumerate(zip(t4_lat, bw_lat)):
            spd = t / bwl
            ax.text(i, max(t, bwl) * 1.3, f"{spd:.1f}×", ha="center",
                    fontsize=5.5, color=C_RAGGED, fontweight="bold")

    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=6.5)

    fig.suptitle("Cross-GPU: T4 dense kernel vs. Blackwell sparse kernel  ($B = 8$)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_cross_gpu.pdf")
    fig.savefig(FIG_DIR / "fig5_cross_gpu.png")
    plt.close(fig)
    print("  ✓ fig5_cross_gpu")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: BF16 vs FP16 parity
# ─────────────────────────────────────────────────────────────────────────────
def fig6_bf16_parity():
    """Scatter: fp16 vs bf16 latency — should be on the y=x line."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    valid = df_bw.dropna(subset=["ragged_fp16_ms", "ragged_bf16_ms"])
    ax.scatter(valid["ragged_fp16_ms"], valid["ragged_bf16_ms"],
               s=12, alpha=0.6, color=C_RAGGED, edgecolors="none", zorder=3)

    lim_max = max(valid["ragged_fp16_ms"].max(), valid["ragged_bf16_ms"].max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5, label="$y = x$")

    ax.set_xlabel("FP16 latency (ms)")
    ax.set_ylabel("BF16 latency (ms)")
    ax.set_title("FP16 / BF16 parity (Blackwell)", fontweight="bold")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)

    # Report max deviation
    ratio = valid["ragged_bf16_ms"] / valid["ragged_fp16_ms"]
    max_dev = (ratio.max() - 1) * 100
    ax.text(0.05, 0.92, f"Max deviation: {max_dev:.1f}%",
            transform=ax.transAxes, fontsize=7, verticalalignment="top")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_bf16_parity.pdf")
    fig.savefig(FIG_DIR / "fig6_bf16_parity.png")
    plt.close(fig)
    print("  ✓ fig6_bf16_parity")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Complexity — O(N·d) vs O(N²) theoretical + measured
# ─────────────────────────────────────────────────────────────────────────────
def fig7_complexity():
    """Log-log plot of latency vs N (total tokens per sequence) for B=1, b=4."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    b = 4
    # Merge B=1 from sota CSV and results.txt
    all_b1 = df_bw[(df_bw["batch_size"] <= 2) & (df_bw["branching_factor"] == b)]
    # Prefer B=1
    b1 = all_b1[all_b1["batch_size"] == 1]
    if b1.empty:
        b1 = all_b1[all_b1["batch_size"] == 2]
    b1 = b1.sort_values("num_tree_nodes")

    ax.plot(b1["num_tree_nodes"], b1["ragged_fp16_ms"],
            "o-", color=C_RAGGED, markersize=5, linewidth=1.5,
            label="Ragged $O(N \\cdot d)$")

    valid_math = b1.dropna(subset=["sdpa_math_ms"])
    if not valid_math.empty:
        ax.plot(valid_math["num_tree_nodes"], valid_math["sdpa_math_ms"],
                "s--", color=C_SDPA_MATH, markersize=4, linewidth=1.2,
                label="SDPA math $O(N^2)$")

    valid_flash = b1.dropna(subset=["sdpa_flash_ms"])
    if not valid_flash.empty:
        ax.plot(valid_flash["num_tree_nodes"], valid_flash["sdpa_flash_ms"],
                "^--", color=C_SDPA_FLASH, markersize=4, linewidth=1.2,
                label="SDPA flash $O(N^2)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$ (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Complexity scaling ($B=1, b={b}$)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_complexity.pdf")
    fig.savefig(FIG_DIR / "fig7_complexity.png")
    plt.close(fig)
    print("  ✓ fig7_complexity")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table data exported as CSV for LaTeX
# ─────────────────────────────────────────────────────────────────────────────
def export_summary_table():
    """Export key numbers for the paper's Table 1."""
    rows = []
    for b in [2, 3, 4]:
        for B in [1, 4, 8, 32, 128]:
            for d in [1, 3, 5, 7, 8]:
                sub = df_bw[
                    (df_bw["batch_size"] == B) &
                    (df_bw["branching_factor"] == b) &
                    (df_bw["tree_depth"] == d)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                N = int(r["num_tree_nodes"])
                spd_math = r["sdpa_math_ms"] / r["ragged_fp16_ms"] if not np.isnan(r["sdpa_math_ms"]) else float("nan")
                spd_flash = r["sdpa_flash_ms"] / r["ragged_fp16_ms"] if not np.isnan(r["sdpa_flash_ms"]) else float("nan")
                rows.append({
                    "B": B, "b": b, "d": d, "N": N,
                    "ragged_ms": round(r["ragged_fp16_ms"], 3),
                    "sdpa_math_ms": round(r["sdpa_math_ms"], 3) if not np.isnan(r["sdpa_math_ms"]) else "OOM",
                    "sdpa_flash_ms": round(r["sdpa_flash_ms"], 3) if not np.isnan(r["sdpa_flash_ms"]) else "OOM",
                    "speedup_vs_math": f"{spd_math:.1f}x" if not np.isnan(spd_math) else "---",
                    "speedup_vs_flash": f"{spd_flash:.1f}x" if not np.isnan(spd_flash) else "---",
                })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(FIG_DIR / "table1_summary.csv", index=False)
    print("  ✓ table1_summary.csv")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating figures...")
    fig1_latency_vs_depth()
    fig2_speedup_heatmap()
    fig3_scaling_batch()
    fig4_padding_waste()
    fig5_cross_gpu()
    fig6_bf16_parity()
    fig7_complexity()
    export_summary_table()
    print(f"\nAll figures saved to {FIG_DIR}/")
