#!/usr/bin/env python3
"""
generate_figures.py
===================
Produces all figures for the MLSys paper from raw benchmark data.

Usage:
    python paper/generate_figures.py

Reads:
    results/sota_benchmark_20260404_041001.csv  – H100 SXM structured CSV (all baselines)
    results/padding_waste.csv                   – Padding waste sweep (optional)

Writes to paper/figures/:
    fig1_latency_vs_depth.pdf
    fig2_speedup_heatmap.pdf
    fig3_scaling_batch.pdf
    fig4_padding_waste.pdf
    fig5_complexity.pdf
    table1_summary.csv
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SOTA_CSV = ROOT / "results" / "sota_benchmark_20260404_041001.csv"
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
C_RAGGED = "#0077B6"     # Strong blue
C_FA2    = "#F77F00"     # Orange  — FlashAttention-2 (via SDPA flash)
C_FI     = "#D62828"     # Red     — FlashInfer
C_DEFT   = "#7209B7"     # Purple  — DeFT
C_GRAY   = "#6C757D"     # Gray    — auxiliary


def num_tree_nodes(b, d):
    if b == 1:
        return d + 1
    return (b ** (d + 1) - 1) // (b - 1)


# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
if not SOTA_CSV.exists():
    print(f"  ERROR: {SOTA_CSV} not found.")
    sys.exit(1)
df = pd.read_csv(SOTA_CSV)
print(f"  H100 SOTA CSV: {len(df)} rows, {len(df.columns)} columns")

if PADDING_CSV.exists():
    df_pad = pd.read_csv(PADDING_CSV)
    print(f"  Padding waste CSV: {len(df_pad)} rows")
else:
    df_pad = pd.DataFrame()
    print("  Padding waste CSV: not found (fig4 will be skipped)")

GPU_LABEL = "H100 SXM (Hopper)"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Latency vs tree depth (log-scale, one panel per branching factor)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_latency_vs_depth():
    """3-panel subplot: b=2,3,4; lines for ragged/FA-2/FlashInfer/DeFT; B=8."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    B_show = 8

    for ax, b in zip(axes, [2, 3, 4]):
        sub = df[(df["batch_size"] == B_show) & (df["branching_factor"] == b)]
        sub = sub.sort_values("tree_depth")

        ax.plot(sub["tree_depth"], sub["ragged_fp16_ms"],
                "o-", color=C_RAGGED, markersize=4, linewidth=1.5,
                label="Ragged (ours)", zorder=5)

        valid_fa2 = sub.dropna(subset=["sdpa_flash_tree_ms"])
        if not valid_fa2.empty:
            ax.plot(valid_fa2["tree_depth"], valid_fa2["sdpa_flash_tree_ms"],
                    "^--", color=C_FA2, markersize=3.5, linewidth=1.2,
                    label="FA-2 (SDPA)")

        valid_fi = sub.dropna(subset=["flashinfer_tree_ms"])
        if not valid_fi.empty:
            ax.plot(valid_fi["tree_depth"], valid_fi["flashinfer_tree_ms"],
                    "s--", color=C_FI, markersize=3.5, linewidth=1.2,
                    label="FlashInfer")

        valid_deft = sub.dropna(subset=["deft_ms"])
        if not valid_deft.empty:
            ax.plot(valid_deft["tree_depth"], valid_deft["deft_ms"],
                    "D--", color=C_DEFT, markersize=3.5, linewidth=1.2,
                    label="DeFT")

        ax.set_yscale("log")
        ax.set_xlabel("Tree depth $d$")
        ax.set_title(f"$b = {b}$", fontweight="bold")
        ax.set_xticks(range(1, 9))

        # Annotate max speedup vs FA-2
        valid = sub.dropna(subset=["sdpa_flash_tree_ms"])
        if not valid.empty:
            idx = valid["speedup_vs_sdpa_flash_tree"].idxmax()
            spd = valid.loc[idx, "speedup_vs_sdpa_flash_tree"]
            d_ann = valid.loc[idx, "tree_depth"]
            ax.annotate(f"{spd:.0f}× vs FA-2",
                        xy=(d_ann, valid.loc[idx, "ragged_fp16_ms"]),
                        xytext=(d_ann - 0.8, valid.loc[idx, "ragged_fp16_ms"] * 0.25),
                        fontsize=6.5, color=C_RAGGED, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C_RAGGED, lw=0.8))

    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[0].legend(loc="upper left", framealpha=0.9)

    fig.suptitle(f"Kernel latency vs. tree depth  ($B = {B_show}$, {GPU_LABEL})",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_latency_vs_depth.pdf")
    fig.savefig(FIG_DIR / "fig1_latency_vs_depth.png")
    plt.close(fig)
    print("  ✓ fig1_latency_vs_depth")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Speedup heatmaps (ragged vs FA-2, FlashInfer, DeFT)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_speedup_heatmap():
    """3×3 heatmap grid: rows = vs FA-2 / FlashInfer / DeFT; cols = b=2,3,4."""
    baselines = [
        ("speedup_vs_sdpa_flash_tree", "vs. FA-2", "RdYlGn"),
        ("speedup_vs_flashinfer_tree", "vs. FlashInfer", "RdYlGn"),
        ("speedup_vs_deft",            "vs. DeFT", "RdYlGn"),
    ]

    fig, axes = plt.subplots(len(baselines), 3, figsize=(7.5, 6.0))

    for row, (spd_col, title_suffix, cmap) in enumerate(baselines):
        for col, b in enumerate([2, 3, 4]):
            ax = axes[row, col]
            sub = df[df["branching_factor"] == b].copy()
            sub_valid = sub.dropna(subset=[spd_col])
            sub_valid = sub_valid[sub_valid[spd_col] > 0]

            pivot = sub_valid.pivot_table(
                index="batch_size", columns="tree_depth",
                values=spd_col, aggfunc="mean"
            )

            if pivot.empty:
                ax.set_visible(False)
                continue

            vmin = max(0.3, pivot.min().min())
            vmax = max(2, pivot.max().max())
            norm = LogNorm(vmin=vmin, vmax=vmax)

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
                        if val >= 100:
                            txt = f"{val:.0f}×"
                        elif val >= 10:
                            txt = f"{val:.0f}×"
                        else:
                            txt = f"{val:.1f}×"
                        color = "white" if val > 20 else "black"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=4.5, color=color, fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"Batch $B$\n({title_suffix})")
            if row == len(baselines) - 1:
                ax.set_xlabel("Tree depth $d$")
            if row == 0:
                ax.set_title(f"$b = {b}$", fontweight="bold")

            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(f"Speedup heatmaps — Ragged kernel ({GPU_LABEL})",
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
    avail_depths = sorted(df[df["branching_factor"] == b]["tree_depth"].unique())
    desired = [3, 5, 7]
    depths = [d for d in desired if d in avail_depths]
    if len(depths) < 3:
        depths = avail_depths[-3:] if len(avail_depths) >= 3 else avail_depths

    fig, axes = plt.subplots(1, len(depths), figsize=(7.0, 2.5), sharey=False)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        sub = df[(df["branching_factor"] == b) & (df["tree_depth"] == d)]
        sub = sub.sort_values("batch_size")
        N = num_tree_nodes(b, d)

        if sub.empty:
            ax.set_visible(False)
            continue

        ax.plot(sub["batch_size"], sub["ragged_fp16_ms"],
                "o-", color=C_RAGGED, markersize=4, linewidth=1.5,
                label="Ragged (ours)")

        for col, color, marker, label in [
            ("sdpa_flash_tree_ms", C_FA2, "^", "FA-2"),
            ("flashinfer_tree_ms", C_FI,  "s", "FlashInfer"),
            ("deft_ms",           C_DEFT, "D", "DeFT"),
        ]:
            valid = sub.dropna(subset=[col])
            if not valid.empty:
                ax.plot(valid["batch_size"], valid[col],
                        f"{marker}--", color=color, markersize=3.5, linewidth=1.2,
                        label=label)

        batch_vals = sub["batch_size"].values
        if len(batch_vals) > 1 and all(v > 0 for v in batch_vals):
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ticks = [v for v in [1, 2, 4, 8, 16, 32, 64, 128]
                     if v >= batch_vals.min() and v <= batch_vals.max()]
            ax.set_xticks(ticks)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size $B$")
        ax.set_title(f"$d = {d}$  ($N = {N:,}$)", fontweight="bold")

    axes[0].set_ylabel("Latency (ms, log scale)")
    axes[-1].legend(loc="upper left", framealpha=0.9, fontsize=6.5)

    fig.suptitle(f"Latency scaling with batch size  ($b = {b}$, {GPU_LABEL})",
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

    sub = df_pad[(df_pad["batch_size"] == 8) & (df_pad["ctx_len"] == 128)]
    if sub.empty:
        sub = df_pad[df_pad["batch_size"] == df_pad["batch_size"].max()]

    sub = sub.sort_values(["branching_factor", "max_tree_depth"])
    labels = [f"b={int(r['branching_factor'])},d={int(r['max_tree_depth'])}"
              for _, r in sub.iterrows()]
    vals = sub["attn_padding_ratio"].values * 100  # percent

    bars = ax.bar(range(len(vals)), vals, color=C_FI, alpha=0.8, width=0.7)
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
# Figure 5: Complexity — O(N·d) vs O(N²) theoretical + measured
# ─────────────────────────────────────────────────────────────────────────────
def fig5_complexity():
    """Log-log plot of latency vs N for B=1, b=4 — all four methods."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    b = 4
    b1 = df[(df["batch_size"] == 1) & (df["branching_factor"] == b)]
    if b1.empty:
        b1 = df[(df["batch_size"] <= 2) & (df["branching_factor"] == b)]
    b1 = b1.sort_values("num_tree_nodes")

    ax.plot(b1["num_tree_nodes"], b1["ragged_fp16_ms"],
            "o-", color=C_RAGGED, markersize=5, linewidth=1.5,
            label="Ragged $O(N \\cdot d)$", zorder=5)

    for col, color, marker, label in [
        ("sdpa_flash_tree_ms", C_FA2,  "^", "FA-2 $O(N^2)$"),
        ("flashinfer_tree_ms", C_FI,   "s", "FlashInfer $O(N^2)$"),
        ("deft_ms",            C_DEFT, "D", "DeFT $O(N^2)$"),
    ]:
        valid = b1.dropna(subset=[col])
        if not valid.empty:
            ax.plot(valid["num_tree_nodes"], valid[col],
                    f"{marker}--", color=color, markersize=4, linewidth=1.2,
                    label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$ (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Complexity scaling ($B=1, b={b}$)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_complexity.pdf")
    fig.savefig(FIG_DIR / "fig5_complexity.png")
    plt.close(fig)
    print("  ✓ fig5_complexity")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table data exported as CSV for LaTeX
# ─────────────────────────────────────────────────────────────────────────────
def export_summary_table():
    """Export key numbers for the paper's Table 1."""
    rows = []
    for b in [2, 3, 4]:
        for B in [1, 8, 32, 128]:
            for d in [3, 5, 7]:
                sub = df[
                    (df["batch_size"] == B) &
                    (df["branching_factor"] == b) &
                    (df["tree_depth"] == d)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                N = int(r["num_tree_nodes"])

                def fmt_ms(col):
                    v = r.get(col, float("nan"))
                    return round(v, 3) if not np.isnan(v) else "OOM"

                def fmt_spd(col):
                    v = r.get(col, float("nan"))
                    return f"{v:.1f}x" if not np.isnan(v) else "---"

                rows.append({
                    "B": B, "b": b, "d": d, "N": N,
                    "ragged_ms": round(r["ragged_fp16_ms"], 3),
                    "fa2_ms":    fmt_ms("sdpa_flash_tree_ms"),
                    "fi_ms":     fmt_ms("flashinfer_tree_ms"),
                    "deft_ms":   fmt_ms("deft_ms"),
                    "speedup_vs_fa2":  fmt_spd("speedup_vs_sdpa_flash_tree"),
                    "speedup_vs_fi":   fmt_spd("speedup_vs_flashinfer_tree"),
                    "speedup_vs_deft": fmt_spd("speedup_vs_deft"),
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
    fig5_complexity()
    export_summary_table()
    print(f"\nAll figures saved to {FIG_DIR}/")

