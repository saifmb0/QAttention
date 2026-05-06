#!/usr/bin/env python3
"""Figure generation for the paper.

Generates:
1. micro speedup heatmap
2. micro kernel latency vs tree size
3. sequoia end-to-end speedup
4. eagle accepted tokens vs cost
5. eagle acceptance rate vs cost
"""

import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, ScalarFormatter


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PALETTE = {
    1: "#332288",
    4: "#117733",
}

TOL = ["#332288", "#117733", "#DDCC77", "#CC6677"]
OURS_COLOR = "#117733"
REF_COLOR = "#999999"


def _save(name):
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close()


def load_micro():
    candidates = [
        os.path.join(REPO_ROOT, "results", "micro", "aggregate.csv"),
        os.path.join(REPO_ROOT, "results", "micro", "micro_benchmark_pruned_aggregate.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            micro = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("Could not find a micro aggregate CSV in results/micro")

    if "speedup_vs_tree" not in micro.columns or micro["speedup_vs_tree"].isnull().all():
        micro["speedup_vs_tree"] = micro["flashinfer_tree_ms"] / micro["ragged_ms"]
    return micro


def load_sequoia():
    path = os.path.join(REPO_ROOT, "results", "sequoia", "aggregate.csv")
    return pd.read_csv(path)


def load_eagle():
    path = os.path.join(REPO_ROOT, "results", "eagle_e2e", "aggregate.csv")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    headers = [
        "context_length,depth,total_token,top_k,label,vanilla_tok_s,vanilla_acc,vanilla_verify,ragged_tok_s,ragged_acc,ragged_verify,e2e_speedup",
        "context_length,depth,total_token,top_k,config_label,mode,model,eagle_model,prompt,num_tokens,num_steps,wall_ms,tok_per_sec,mean_accepted_per_step,acceptance_rate,mean_verify_ms,verify_fraction",
    ]

    sections = {}
    current_header = None
    current_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in headers:
            if current_header is not None and current_lines:
                sections[current_header] = pd.read_csv(StringIO("\n".join(current_lines)))
            current_header = stripped
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_header is not None and current_lines:
        sections[current_header] = pd.read_csv(StringIO("\n".join(current_lines)))

    if headers[1] in sections:
        eagle = sections[headers[1]]
        eagle = eagle.copy()
        for col in ["total_token", "mean_accepted_per_step", "acceptance_rate", "wall_ms", "tok_per_sec", "mean_verify_ms", "verify_fraction"]:
            if col in eagle.columns:
                eagle[col] = pd.to_numeric(eagle[col], errors="coerce")
        return eagle

    raise ValueError("Could not parse eagle aggregate.csv")


def plot_speedup_heatmap(micro):
    df = micro[(micro["batch_size"] == 1) & (micro["prefix_length"] == 0)]
    pivot_df = df.pivot_table(index="depth", columns="branching_factor", values="speedup_vs_tree")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_df.values, cmap="YlGnBu", aspect="auto", origin="lower")
    cbar = fig.colorbar(im)
    cbar.set_label("Speedup (FlashInfer / Ours)")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 2 else "white")

    ax.set_title("Tree-Attention Speedup: Depth vs Branching Factor\n(Batch Size=1, Prefix=0)")
    ax.set_xlabel("Branching Factor (b)")
    ax.set_ylabel("Tree Depth (d)")
    _save("speedup_heatmap_d_vs_b")


def plot_speedup_vs_nodes(micro):
    df = micro[micro["prefix_length"] == 0].copy()
    selected_batch_sizes = [1, 4]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 5.5),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
    )

    lower_ylim = (0.0, 0.30)
    upper_ylim = (0.34, 0.95)

    for bs in selected_batch_sizes:
        sub = df[df["batch_size"] == bs]
        if sub.empty:
            continue

        grouped = (
            sub.groupby("num_tree_nodes", as_index=False)
            .agg(
                flashinfer_tree_ms=("flashinfer_tree_ms", "mean"),
                ragged_ms=("ragged_ms", "mean"),
            )
            .sort_values("num_tree_nodes")
        )
        grouped = grouped[grouped["num_tree_nodes"] >= 100]
        if grouped.empty:
            continue

        color = PALETTE.get(bs)

        for ax in (ax_top, ax_bottom):
            ax.plot(
                grouped["num_tree_nodes"],
                grouped["ragged_ms"],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                markersize=4,
                color=color,
            )
            ax.plot(
                grouped["num_tree_nodes"],
                grouped["flashinfer_tree_ms"],
                marker="s",
                linestyle="--",
                linewidth=1.6,
                markersize=4,
                color=color,
                alpha=0.9,
            )

    ax_bottom.set_ylim(*lower_ylim)
    ax_top.set_ylim(*upper_ylim)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_top.xaxis.tick_top()
    ax_bottom.xaxis.tick_bottom()

    kwargs = dict(color="black", clip_on=False, linewidth=1.0)
    ax_top.plot((-0.015, 0.015), (-0.015, 0.015), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((0.985, 1.015), (-0.015, 0.015), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-0.015, 0.015), (0.985, 1.015), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((0.985, 1.015), (0.985, 1.015), transform=ax_bottom.transAxes, **kwargs)

    ax_bottom.set_xlim(left=100)
    ax_bottom.xaxis.set_major_locator(MultipleLocator(100))
    ax_bottom.xaxis.set_major_formatter(ScalarFormatter())
    ax_bottom.ticklabel_format(axis="x", style="plain")

    ax_bottom.set_xlabel("Tree Size (Number of Nodes)")
    ax_bottom.set_ylabel("Latency (ms)")
    ax_top.set_ylabel("Latency (ms)")
    ax_top.set_title("Kernel Latency vs Tree Size\n(Prefix Length = 0)")

    legend_handles = [
        Line2D([0], [0], color=PALETTE[1], marker="o", linestyle="-", linewidth=1.8, markersize=4, label="Ours, BS=1"),
        Line2D([0], [0], color=PALETTE[1], marker="s", linestyle="--", linewidth=1.6, markersize=4, label="FlashInfer, BS=1"),
        Line2D([0], [0], color=PALETTE[4], marker="o", linestyle="-", linewidth=1.8, markersize=4, label="Ours, BS=4"),
        Line2D([0], [0], color=PALETTE[4], marker="s", linestyle="--", linewidth=1.6, markersize=4, label="FlashInfer, BS=4"),
    ]
    ax_top.legend(
        handles=legend_handles,
        title="Kernel / Batch Size",
        loc="upper left",
        ncol=2,
        frameon=False,
    )
    ax_bottom.grid(True, which="both", ls="-", alpha=0.3)
    ax_top.grid(True, which="both", ls="-", alpha=0.3)
    _save("speedup_vs_nodes")


def plot_sequoia_e2e(sequoia):
    df = sequoia.sort_values("tree_size")
    fig, ax1 = plt.subplots(figsize=(3.6, 2.5))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    l1, = ax1.plot(
        df["tree_size"],
        df["speedup"],
        linestyle="-",
        marker="o",
        markersize=3.2,
        linewidth=1.2,
        color=OURS_COLOR,
        label="QAtten / Sequoia speedup",
    )
    l2, = ax2.plot(
        df["tree_size"],
        df["v_acc_per_step"] / df["r_acc_per_step"],
        linestyle="--",
        marker="s",
        markersize=2.8,
        linewidth=1.0,
        color=TOL[0],
        label="Sequoia/QAtten accepted tokens/step",
    )

    ax1.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax2.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sequoia tree size (nodes)")
    ax1.set_ylabel(r"QAtten / Sequoia speedup ($\times$)", color=OURS_COLOR)
    ax2.set_ylabel("Sequoia/QAtten accepted tokens/step", color=TOL[0])
    ax1.tick_params(axis="y", labelcolor=OURS_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOL[0])
    ax2.yaxis.set_major_locator(MultipleLocator(0.04))
    ax2.set_ylim([0, 1.08])
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8, handlelength=2.0)
    plt.tight_layout()
    _save("fig5_e2e_sequoia")


def plot_eagle_accepted_step(eagle):
    df = (
        eagle.groupby("total_token", as_index=False)
        .agg(
            mean_accepted_per_step=("mean_accepted_per_step", "mean"),
            wall_ms=("wall_ms", "mean"),
        )
        .sort_values("total_token")
    )

    baseline = df.iloc[0]
    df = df.copy()
    df["rel_accepted_per_step"] = df["mean_accepted_per_step"] / baseline["mean_accepted_per_step"]
    df["rel_wall_ms"] = df["wall_ms"] / baseline["wall_ms"]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(
        df["total_token"],
        df["rel_accepted_per_step"],
        linestyle="-",
        marker="o",
        color=OURS_COLOR,
        markersize=2.8,
        linewidth=1.2,
        label="Accepted tokens / step",
    )
    ax.plot(
        df["total_token"],
        df["rel_wall_ms"],
        linestyle="--",
        marker="s",
        color=TOL[0],
        markersize=2.8,
        linewidth=1.1,
        label="Wall time",
    )

    ax.axhline(1.0, color=REF_COLOR, linestyle="--", linewidth=0.7, zorder=0)
    ax.axvline(80, color=REF_COLOR, linestyle=":", linewidth=0.8)
    ax.text(80, 1.82, "~80", ha="center", va="top", fontsize=8, color=REF_COLOR)

    ax.set_xlabel("Total tokens per tree")
    ax.set_ylabel("Relative to first config")
    ax.set_ylim(0.0, 1.9)
    ax.set_ylim(1, 1.9)
    ax.set_yticks(np.arange(1.0, 2.0, 0.2))
    ax.set_title("EAGLE Scaling: Accepted Tokens vs Cost")
    ax.legend(loc="upper left", handlelength=2.0, handletextpad=0.5)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save("fig4_eagle_accepted_step")


def plot_eagle_acceptance_rate(eagle):
    df = (
        eagle.groupby("total_token", as_index=False)
        .agg(
            acceptance_rate=("acceptance_rate", "mean"),
            wall_ms=("wall_ms", "mean"),
        )
        .sort_values("total_token")
    )

    fig, ax1 = plt.subplots(figsize=(3.6, 2.6))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    l1, = ax1.plot(
        df["total_token"],
        df["acceptance_rate"],
        linestyle="-",
        marker="o",
        color=OURS_COLOR,
        markersize=2.8,
        linewidth=1.2,
        label="Acceptance rate",
    )
    l2, = ax2.plot(
        df["total_token"],
        df["wall_ms"],
        linestyle="--",
        marker="s",
        color=TOL[0],
        markersize=2.8,
        linewidth=1.1,
        label="Wall time (ms)",
    )

    ax1.axvline(80, color=REF_COLOR, linestyle=":", linewidth=0.8)
    ax1.text(80, ax1.get_ylim()[1] * 0.95, "~80", ha="center", va="top", fontsize=8, color=REF_COLOR)

    ax1.set_xlabel("Total tokens per tree")
    ax1.set_ylabel("Acceptance rate", color=OURS_COLOR)
    ax2.set_ylabel("Wall time (ms)", color=TOL[0])
    ax1.tick_params(axis="y", labelcolor=OURS_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOL[0])
    ax1.legend(handles=[l1, l2], loc="upper left", handlelength=2.0, handletextpad=0.5)
    ax1.set_title("EAGLE Scaling: Acceptance Rate vs Cost")
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save("fig5_eagle_acceptance_rate")


if __name__ == "__main__":
    print("Loading data...")
    micro = load_micro()
    sequoia = load_sequoia()
    eagle = load_eagle()

    print("Generating figures...")
    plot_speedup_heatmap(micro)
    plot_speedup_vs_nodes(micro)
    plot_sequoia_e2e(sequoia)
    plot_eagle_accepted_step(eagle)
    plot_eagle_acceptance_rate(eagle)

    print(f"\nAll experimental figures generated in {FIG_DIR}")
