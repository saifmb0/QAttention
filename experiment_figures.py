#!/usr/bin/env python3
"""Figure generation for the paper.

Generates:
1. micro speedup heatmap
2. balanced-tree speedup heatmap
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

FIG_DIR = os.path.join(os.path.dirname(__file__), "experiment_figures")
os.makedirs(FIG_DIR, exist_ok=True)

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

PALETTE = {
    1: "#332288",
    4: "#117733",
}

TOL = ["#332288", "#117733", "#DDCC77", "#CC6677"]
OURS_COLOR = "#117733"
REF_COLOR = "#999999"
EAGLE_MIN_TOKENS = 200


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


def load_balanced_micro():
    path = os.path.join(REPO_ROOT, "results", "balanced_micro", "aggregate.csv")
    micro = pd.read_csv(path)

    if "speedup_vs_deft" not in micro.columns or micro["speedup_vs_deft"].isnull().all():
        micro["speedup_vs_deft"] = micro["deft_ms"] / micro["ragged_ms"]
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
        for col in ["total_token", "num_tokens", "mean_accepted_per_step", "acceptance_rate", "wall_ms", "tok_per_sec", "mean_verify_ms", "verify_fraction"]:
            if col in eagle.columns:
                eagle[col] = pd.to_numeric(eagle[col], errors="coerce")

        if "num_tokens" in eagle.columns:
            eagle = eagle[eagle["num_tokens"] > EAGLE_MIN_TOKENS].copy()
        return eagle

    raise ValueError("Could not parse eagle aggregate.csv")


def load_eagle_e2e():
    detail = load_eagle().copy()

    keys = ["context_length", "depth", "total_token", "top_k", "config_label", "prompt"]
    paired = detail.pivot_table(
        index=keys,
        columns="mode",
        values=["wall_ms", "verify_fraction"],
        aggfunc="mean",
    )

    paired.columns = [f"{metric}_{mode}" for metric, mode in paired.columns]
    paired = paired.reset_index()

    required = [
        "wall_ms_vanilla",
        "wall_ms_ragged",
        "verify_fraction_vanilla",
        "verify_fraction_ragged",
    ]
    paired = paired.dropna(subset=required)

    paired["e2e_speedup"] = paired["wall_ms_vanilla"] / paired["wall_ms_ragged"]
    paired["verify_drop"] = (
        (paired["verify_fraction_vanilla"] - paired["verify_fraction_ragged"])
        / paired["verify_fraction_vanilla"]
        * 100.0
    )

    return (
        paired.groupby("total_token", as_index=False)
        .agg(
            e2e_speedup=("e2e_speedup", "mean"),
            verify_drop=("verify_drop", "mean"),
        )
        .sort_values("total_token")
    )


def _load_multi_section_detail_csv(path, detail_header):
    """Load the detail section from a multi-section aggregate CSV."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    sections = {}
    current_header = None
    current_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == detail_header:
            if current_header is not None and current_lines:
                sections[current_header] = pd.read_csv(StringIO("\n".join(current_lines)))
            current_header = stripped
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_header is not None and current_lines:
        sections[current_header] = pd.read_csv(StringIO("\n".join(current_lines)))

    if detail_header not in sections:
        raise ValueError(f"Could not parse {path}")

    return sections[detail_header]


def _summarize_sweep(detail, x_col):
    """Compute throughput speedup and verification-cost drop for a sweep."""
    paired = detail.pivot_table(
        index=[x_col],
        columns="mode",
        values=["tok_per_sec", "verify_fraction"],
        aggfunc="mean",
    )
    paired.columns = [f"{metric}_{mode}" for metric, mode in paired.columns]
    paired = paired.reset_index()

    required = [
        "tok_per_sec_vanilla",
        "tok_per_sec_ragged",
        "verify_fraction_vanilla",
        "verify_fraction_ragged",
    ]
    paired = paired.dropna(subset=required)

    paired["throughput_speedup"] = paired["tok_per_sec_ragged"] / paired["tok_per_sec_vanilla"]
    paired["verify_drop_pct"] = (
        (paired["verify_fraction_vanilla"] - paired["verify_fraction_ragged"])
        / paired["verify_fraction_vanilla"]
        * 100.0
    )

    return paired.sort_values(x_col)


def load_a100_ctx_sweep():
    """Load the A100 context-length sweep.

    The intended slice is the fixed d=20, b=20 run family with varying
    context lengths (roughly 2k–10k tokens).
    """
    path = os.path.join(REPO_ROOT, "results", "A100-ctx-sweep", "aggregate.csv")
    detail_header = (
        "context_length,depth,total_token,top_k,config_label,mode,model,"
        "eagle_model,prompt,num_tokens,num_steps,wall_ms,tok_per_sec,"
        "mean_accepted_per_step,acceptance_rate,mean_verify_ms,verify_fraction"
    )
    detail = _load_multi_section_detail_csv(path, detail_header).copy()

    numeric_cols = [
        "context_length",
        "depth",
        "total_token",
        "top_k",
        "num_tokens",
        "num_steps",
        "wall_ms",
        "tok_per_sec",
        "mean_accepted_per_step",
        "acceptance_rate",
        "mean_verify_ms",
        "verify_fraction",
    ]
    for col in numeric_cols:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")

    # Keep the intended fixed-tree slice only.
    detail = detail[(detail["depth"] == 20) & (detail["top_k"] == 20)].copy()
    detail = detail[(detail["context_length"] >= 2048) & (detail["context_length"] <= 10240)].copy()
    return _summarize_sweep(detail, "context_length")


def load_a100_n_sweep():
    """Load the A100 tree-size sweep."""
    path = os.path.join(REPO_ROOT, "results", "A100-n-sweep", "aggregate.csv")
    detail_header = (
        "context_length,depth,total_token,top_k,config_label,mode,model,"
        "eagle_model,prompt,num_tokens,num_steps,wall_ms,tok_per_sec,"
        "mean_accepted_per_step,acceptance_rate,mean_verify_ms,verify_fraction"
    )
    detail = _load_multi_section_detail_csv(path, detail_header).copy()

    numeric_cols = [
        "context_length",
        "depth",
        "total_token",
        "top_k",
        "num_tokens",
        "num_steps",
        "wall_ms",
        "tok_per_sec",
        "mean_accepted_per_step",
        "acceptance_rate",
        "mean_verify_ms",
        "verify_fraction",
    ]
    for col in numeric_cols:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")

    return _summarize_sweep(detail, "total_token")


def plot_a100_sweep(df, x_col, x_label, title, out_name):
    fig, ax1 = plt.subplots(figsize=(3.7, 2.6))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    ax1.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax2.axhline(y=0.0, color=REF_COLOR, linestyle=":", linewidth=0.6, zorder=0)

    l1, = ax1.plot(
        df[x_col],
        df["throughput_speedup"],
        linestyle="-",
        marker="o",
        markersize=3.3,
        linewidth=1.25,
        color=OURS_COLOR,
        label="Throughput",
    )
    l2, = ax2.plot(
        df[x_col],
        df["verify_drop_pct"],
        linestyle="--",
        marker="s",
        markersize=3.0,
        linewidth=1.1,
        color=TOL[0],
        label=r"$\Delta$ Cost (%)",
    )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"Throughput ($\times$)", color=OURS_COLOR)
    ax2.set_ylabel(r"$\Delta$ Relative verify cost (%)", color=TOL[0])
    ax1.tick_params(axis="y", labelcolor=OURS_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOL[0])
    ax1.set_title(title)
    ax1.legend(handles=[l1, l2], fontsize=8, loc="upper left", frameon=False, handlelength=0.5, handletextpad=0.5)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save(out_name)


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
    _save("fig1_heatmap_d_vs_b")


def plot_balanced_speedup_heatmap(micro):
    df = micro[(micro["batch_size"] == 1) & (micro["prefix_length"] == 0)]
    pivot_df = df.pivot_table(index="depth", columns="branching_factor", values="speedup_vs_deft")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_df.values, cmap="YlGnBu", aspect="auto", origin="lower")
    cbar = fig.colorbar(im)
    cbar.set_label("Speedup (DeFT / Ours)")

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 2 else "white")

    ax.set_title("Balanced Tree-Attention Speedup over DeFT: Depth vs Branching Factor\n(Batch Size=1, Prefix=0)")
    ax.set_xlabel("Branching Factor (b)")
    ax.set_ylabel("Tree Depth (d)")
    _save("fig2_heatmap_DeFT")


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
    _save("fig3_kernel_latency_vs_tree_size")


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
        label=r"QAtten / Sequoia ($\times$)",
    )
    l2, = ax2.plot(
        df["tree_size"],
        df["v_acc_per_step"] / df["r_acc_per_step"],
        linestyle="--",
        marker="s",
        markersize=2.8,
        linewidth=1.0,
        color=TOL[0],
        label=r"Sequoia/QAtten ($\times$)",
    )

    ax1.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax2.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sequoia tree size")
    ax1.set_ylabel(r"Throughput ($\times$)", color=OURS_COLOR)
    ax2.set_ylabel(r"Divergence acc/step ($\times$)", color=TOL[0])
    ax1.tick_params(axis="y", labelcolor=OURS_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOL[0])
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.set_ylim([0.95, 1.1])
    
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=6, handlelength=0.35)
    plt.tight_layout()
    _save("fig9_Sequoia_speedup")


def plot_eagle_e2e(eagle_e2e):
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)

    df = eagle_e2e.sort_values("total_token").copy()

    ax.axhline(y=1.0, color=REF_COLOR, linestyle="--", linewidth=0.6, zorder=0)
    ax2.axhline(y=0.0, color=REF_COLOR, linestyle=":", linewidth=0.6, zorder=0)

    l1, = ax.plot(
        df["total_token"],
        df["e2e_speedup"],
        linestyle="-",
        marker="o",
        markersize=3.4,
        linewidth=1.3,
        color=OURS_COLOR,
        label=r"Throughput ($\times$)",
    )
    l2, = ax2.plot(
        df["total_token"],
        df["verify_drop"],
        linestyle="--",
        marker="s",
        markersize=3.0,
        linewidth=1.1,
        color=TOL[0],
        label="$\Delta$ Cost (%)",
    )

    ax.set_xlabel(r"Tree size (total tokens)")
    ax.set_ylabel(r"Throughput ($\times$)")
    ax2.set_ylabel(r"$\Delta$ Relative verify cost (%)")
    ax.set_title("EAGLE-3 Tree Size Sweep (L=0)")
    ax.legend(handles=[l1, l2], loc="upper left", frameon=False,
              fontsize=8,
                handlelength=0.5,
                handletextpad=0.5
              )
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save("fig6_eagle-3_speedup_vs_tree_size")


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
        markersize=2.5,
        linewidth=1.2,
        label="Accepted tokens / step",
    )
    ax.plot(
        df["total_token"],
        df["rel_wall_ms"],
        linestyle="--",
        marker="s",
        color=TOL[0],
        markersize=2.5,
        linewidth=1.1,
        label="Wall time",
    )

    # ax.axhline(1.0, color=REF_COLOR, linestyle="--", linewidth=0.7, zorder=0)
    # ax.axvline(80, color=REF_COLOR, linestyle=":", linewidth=0.8)
    # ax.text(80, 1.82, "~80", ha="center", va="top", fontsize=8, color=REF_COLOR)

    ax.set_xlabel("Total tokens per tree")
    ax.set_ylabel("Relative to first config")
    ax.set_ylim(1, 1.9)
    ax.set_yticks(np.arange(1.0, 2.0, 0.2))
    ax.set_title("EAGLE Scaling: Accepted Tokens vs Cost")
    ax.legend(loc="upper left", handlelength=0.5, fontsize=8, handletextpad=0.5)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save("fig5_accepted_tokens_vs_cost")


def plot_eagle_acceptance_rate(eagle):
    df = (
        eagle.groupby("total_token", as_index=False)
        .agg(
            acceptance_rate=("acceptance_rate", "mean"),
            wall_ms=("wall_ms", "mean"),
        )
        .sort_values("total_token")
    )

    fig, ax1 = plt.subplots(figsize=(3.6, 2.7))

    fig, ax1 = plt.subplots(figsize=(3.6, 2.6))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    l1, = ax1.plot(
        df["total_token"],
        df["acceptance_rate"],
        linestyle="-",
        marker="o",
        color=OURS_COLOR,
        markersize=2.6,
        linewidth=1.2,
        label="Acceptance rate",
    )
    l2, = ax2.plot(
        df["total_token"],
        df["wall_ms"],
        linestyle="--",
        marker="s",
        color=TOL[0],
        markersize=2.6,
        linewidth=1.1,
        label="Wall time (ms)",
    )

    ax1.axvline(80, color=REF_COLOR, linestyle=":", linewidth=0.8)
    ax1.text(80, ax1.get_ylim()[1] * 0.95, "~80", ha="center", va="top", fontsize=8, color=REF_COLOR)

    ax1.set_xlabel("Tokens per tree")
    ax1.set_ylabel("Acceptance rate", color=OURS_COLOR)
    ax2.set_ylabel("Time (ms)", color=TOL[0])
    ax1.tick_params(axis="y", labelcolor=OURS_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOL[0])
    ax1.legend(handles=[l1, l2], loc="upper left", handlelength=0.5, fontsize=8, handletextpad=0.5)
    ax1.set_title("Acceptance Rate vs Cost")
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    _save("fig4_acceptance_rate_vs_cost")


if __name__ == "__main__":
    print("Loading data...")
    micro = load_micro()
    balanced_micro = load_balanced_micro()
    sequoia = load_sequoia()
    eagle = load_eagle()
    eagle_e2e = load_eagle_e2e()
    a100_ctx = load_a100_ctx_sweep()
    a100_n = load_a100_n_sweep()

    print("Generating figures...")
    plot_speedup_heatmap(micro)
    plot_balanced_speedup_heatmap(balanced_micro)
    plot_speedup_vs_nodes(micro)
    plot_sequoia_e2e(sequoia)
    plot_eagle_accepted_step(eagle)
    plot_eagle_acceptance_rate(eagle)
    plot_eagle_e2e(eagle_e2e)
    plot_a100_sweep(
        a100_ctx,
        "context_length",
        "Prefix (tokens)",
        "A100 Prefix Sweep (d=20, b=20)",
        "fig7_a100_ctx_sweep",
    )
    plot_a100_sweep(
        a100_n,
        "total_token",
        "Tree size (total tokens)",
        "A100 Tree Size Sweep (L=4096)",
        "fig8_a100_n_sweep",
    )
    
    print(f"\nAll experimental figures generated in {FIG_DIR}")
