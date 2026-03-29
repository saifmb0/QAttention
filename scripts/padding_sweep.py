"""
padding_sweep.py
================
Padding waste characterisation sweep for speculative-decoding verification.

Sweeps over (batch_size, γ / tree_depth, branching_factor) and records:
  - token_padding_ratio  : fraction of padded token positions
  - attn_padding_ratio   : fraction of wasted QK^T operations (quadratic waste)

Outputs
-------
  results/padding_waste.csv
  results/padding_waste_token.png
  results/padding_waste_attn.png

Usage
-----
  python scripts/padding_sweep.py [--out-dir results] [--no-plot]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")   # headless backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.padding_waste import sweep


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

BATCH_SIZES         = [1, 2, 4, 8, 16, 32]
CTX_LENS            = [128]           # fixed context – focus on tree shape
DEPTHS              = [1, 2, 3, 4, 5]
BRANCHING_FACTORS   = [2, 3, 4]
CTX_VARIANCE        = 0.0             # uniform batch (worst-case for padding)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_CMAP = "viridis"

def _heatmap(
    df_pivot: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    fmt: str = ".2f",
    vmin: float = 0.0,
    vmax: float = 1.0,
    out_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        df_pivot.values,
        aspect="auto",
        cmap=_CMAP,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Ratio")

    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, fontsize=9)
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_yticklabels(df_pivot.index, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate cells
    for r in range(df_pivot.shape[0]):
        for c in range(df_pivot.shape[1]):
            val = df_pivot.values[r, c]
            color = "white" if val > 0.55 else "black"
            ax.text(c, r, format(val, fmt),
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    plt.close(fig)


def _line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    out_path: str | None = None,
) -> None:
    """Line plot, grouped by hue_col (and optional style linestyle by style_col)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    hue_vals   = sorted(df[hue_col].unique())
    style_vals = sorted(df[style_col].unique()) if style_col else [None]
    markers    = ["o", "s", "D", "^", "v", "x"]
    linestyles = ["-", "--", ":", "-."]
    colors     = plt.cm.tab10(np.linspace(0, 0.9, len(hue_vals)))  # type: ignore

    for hi, hv in enumerate(hue_vals):
        sub_h = df[df[hue_col] == hv]
        for si, sv in enumerate(style_vals):
            if sv is not None:
                sub = sub_h[sub_h[style_col] == sv]
                label = f"b={hv}, B={sv}"
            else:
                sub = sub_h
                label = f"b={hv}"
            sub_sorted = sub.sort_values(x_col)
            ax.plot(
                sub_sorted[x_col],
                sub_sorted[y_col],
                marker=markers[si % len(markers)],
                linestyle=linestyles[si % len(linestyles)],
                color=colors[hi],
                label=label,
                linewidth=1.6,
                markersize=5,
            )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_dir: str = "results", no_plot: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Running padding waste sweep …")
    df = sweep(
        batch_sizes=BATCH_SIZES,
        ctx_lens=CTX_LENS,
        gammas=DEPTHS,
        branching_factors=BRANCHING_FACTORS,
        ctx_variance=CTX_VARIANCE,
    )

    csv_path = os.path.join(out_dir, "padding_waste.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(df)} rows)")

    # Summary statistics
    print("\n── Summary ──")
    print(df.groupby(["tree_depth", "branching_factor"])[
        ["token_padding_ratio", "attn_padding_ratio"]
    ].mean().round(3).to_string())

    if no_plot:
        return

    # ------------------------------------------------------------------
    # Plot 1: Heatmap – attention padding ratio vs (tree_depth, batch_size)
    #         for each branching factor separately
    # ------------------------------------------------------------------
    for b in BRANCHING_FACTORS:
        sub = df[(df["branching_factor"] == b) & (df["ctx_len"] == CTX_LENS[0])]
        pivot = sub.pivot_table(
            index="tree_depth",
            columns="batch_size",
            values="attn_padding_ratio",
            aggfunc="mean",
        )
        _heatmap(
            pivot,
            title=f"Attention FLOPs Padding Ratio  (branching factor b={b})",
            xlabel="Batch size B",
            ylabel="Tree depth d",
            vmin=0.0,
            vmax=1.0,
            out_path=os.path.join(out_dir, f"padding_attn_heatmap_b{b}.png"),
        )

    # ------------------------------------------------------------------
    # Plot 2: Line plot – attn padding ratio vs tree_depth for each B
    #         for the most illustrative branching factor (b=2)
    # ------------------------------------------------------------------
    sub2 = df[(df["ctx_len"] == CTX_LENS[0])].copy()
    _line_plot(
        sub2,
        x_col="tree_depth",
        y_col="attn_padding_ratio",
        hue_col="branching_factor",
        style_col="batch_size",
        title="Attention FLOPs Padding Ratio vs Tree Depth",
        xlabel="Tree depth  d",
        ylabel="Attention padding ratio",
        out_path=os.path.join(out_dir, "padding_attn_lines.png"),
    )

    # ------------------------------------------------------------------
    # Plot 3: Token-level padding ratio (for completeness)
    # ------------------------------------------------------------------
    _line_plot(
        sub2,
        x_col="tree_depth",
        y_col="token_padding_ratio",
        hue_col="branching_factor",
        style_col="batch_size",
        title="Token-Level Padding Ratio vs Tree Depth",
        xlabel="Tree depth  d",
        ylabel="Token padding ratio",
        out_path=os.path.join(out_dir, "padding_token_lines.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Padding waste characterisation sweep")
    parser.add_argument("--out-dir",  default="results",
                        help="Output directory for CSV and plots (default: results)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()
    main(out_dir=args.out_dir, no_plot=args.no_plot)
