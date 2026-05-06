#!/usr/bin/env python3
"""Publication-quality figure generation for the tree-attention NeurIPS paper.

Data-selection logic and column mappings are unchanged from the previous
revision; only the styling layer has been rewritten.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "regular",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": "#999999",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Paul Tol "muted" — colorblind-safe, four consistent series colors
TOL = ["#332288", "#117733", "#DDCC77", "#CC6677"]  # indigo, green, sand, rose
OURS_COLOR = "#117733"
FI_COLOR   = "#CC6677"
REF_COLOR  = "#999999"

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save(name):
    """Save current figure as both PDF (for LaTeX) and PNG at 300 dpi."""
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close()


def _markevery(n, target=8):
    return max(1, n // target)


def load_data():
    micro   = pd.read_csv(os.path.join(repo_root, 'results', 'micro',     'micro_benchmark_pruned_aggregate.csv'))
    amdahl  = pd.read_csv(os.path.join(repo_root, 'results', 'amdahl',    'amdahl_aggregate.csv'))
    sequoia = pd.read_csv(os.path.join(repo_root, 'results', 'sequoia',   'sequoia_size_aggregate.csv'))
    eagle   = pd.read_csv(os.path.join(repo_root, 'results', 'eagle_e2e', 'e2e_summary_aggregate.csv'))
    return micro, amdahl, sequoia, eagle


# ---------------------------------------------------------------------------
# Fig 1 — Latency vs depth at fixed branching factor (double column)
# ---------------------------------------------------------------------------
def plot_micro_fixed_b(micro):
    df = micro[(micro['prefix_length'] == 0) & (micro['batch_size'] == 1)]
    bs = sorted(df['branching_factor'].unique())
    selected_bs = [b for b in [4, 8, 12, 16] if b in bs]
    if len(selected_bs) < 4:
        selected_bs = bs[:4]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, b in enumerate(selected_bs):
        sub = df[df['branching_factor'] == b].sort_values('depth')
        n = len(sub)
        me = _markevery(n)
        ax.plot(sub['depth'], sub['flashinfer_tree_ms'],
                linestyle='--', marker='s', markevery=me,
                color=TOL[i], label=f'FlashInfer ($b{{=}}{b}$)')
        ax.plot(sub['depth'], sub['ragged_ms'],
                linestyle='-', marker='o', markevery=me,
                color=TOL[i], label=f'Ours ($b{{=}}{b}$)')

    ax.set_xlabel(r'Tree depth $d$')
    ax.set_ylabel('Latency (ms)')
    ax.set_yscale('log')
    ax.legend(ncol=4, loc='upper left', columnspacing=1.0,
              handlelength=2.0, handletextpad=0.5)
    plt.tight_layout()
    _save('fig1_micro_fixed_b')


# ---------------------------------------------------------------------------
# Fig 2 — Latency vs branching factor at fixed depth (double column)
# ---------------------------------------------------------------------------
def plot_micro_fixed_depth(micro):
    df = micro[(micro['prefix_length'] == 0) & (micro['batch_size'] == 1)]
    ds = sorted(df['depth'].unique())
    selected_ds = [d for d in [3, 5, 7, 10] if d in ds]
    if len(selected_ds) < 4:
        selected_ds = ds[:4]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, d in enumerate(selected_ds):
        sub = df[df['depth'] == d].sort_values('branching_factor')
        n = len(sub)
        me = _markevery(n)
        ax.plot(sub['branching_factor'], sub['flashinfer_tree_ms'],
                linestyle='--', marker='s', markevery=me,
                color=TOL[i], label=f'FlashInfer ($d{{=}}{d}$)')
        ax.plot(sub['branching_factor'], sub['ragged_ms'],
                linestyle='-', marker='o', markevery=me,
                color=TOL[i], label=f'Ours ($d{{=}}{d}$)')

    ax.set_xlabel(r'Branching factor $b$')
    ax.set_ylabel('Latency (ms)')
    ax.set_yscale('log')
    ax.legend(ncol=4, loc='upper left', columnspacing=1.0,
              handlelength=2.0, handletextpad=0.5)
    plt.tight_layout()
    _save('fig2_micro_fixed_depth')


# ---------------------------------------------------------------------------
# Fig 3 — Sweep tree size N at varying prefix length L (single column)
# ---------------------------------------------------------------------------
def plot_micro_sweep_n(micro):
    df = micro[(micro['batch_size'] == 1) & (micro['branching_factor'] == 4)]
    pls = sorted(df['prefix_length'].unique())
    selected_pls = [pl for pl in [0, 4096, 16384, 32768] if pl in pls]
    if len(selected_pls) < 4:
        selected_pls = pls[:4]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, pl in enumerate(selected_pls):
        sub = df[df['prefix_length'] == pl].sort_values('num_tree_nodes')
        n = len(sub)
        me = _markevery(n)
        ax.plot(sub['num_tree_nodes'], sub['ragged_ms'],
                linestyle='-', marker='o', markevery=me,
                color=TOL[i], label=f'$L{{=}}{pl}$')

    ax.set_xscale('log')
    ax.set_xlabel(r'Tree size $N$ (nodes)')
    ax.set_ylabel('Latency (ms)')
    ax.legend(loc='upper left', title='Prefix length',
              title_fontsize=8, ncol=2, columnspacing=1.0)
    plt.tight_layout()
    _save('fig3_micro_sweep_n')


# ---------------------------------------------------------------------------
# Fig 4 — Amdahl: acceptance rate & verify fraction vs depth (single column)
# ---------------------------------------------------------------------------
def plot_amdahl(amdahl):
    df = amdahl.groupby('depth').mean(numeric_only=True).reset_index().sort_values('depth')

    fig, ax1 = plt.subplots(figsize=(3.5, 2.6))
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)   # twin axes need explicit reset

    l1, = ax1.plot(df['depth'], df['mean_accepted_per_step'],
                   linestyle='-', marker='o', color=OURS_COLOR,
                   label='Accepted tokens / step')
    l2, = ax2.plot(df['depth'], df['verify_fraction'],
                   linestyle='--', marker='s', color=TOL[0],
                   label='Verify time fraction')

    ax1.set_xlabel(r'Tree depth $d$')
    ax1.set_ylabel('Accepted tokens / step', color=OURS_COLOR)
    ax2.set_ylabel('Verify time fraction', color=TOL[0])
    ax1.tick_params(axis='y', labelcolor=OURS_COLOR)
    ax2.tick_params(axis='y', labelcolor=TOL[0])

    ax1.legend(handles=[l1, l2], loc='center right',
               handlelength=2.0, handletextpad=0.5)
    plt.tight_layout()
    _save('fig4_amdahl')


# ---------------------------------------------------------------------------
# Fig 5 — Sequoia E2E speedup vs tree size (single column)
# ---------------------------------------------------------------------------
def plot_sequoia(sequoia):
    df = sequoia.sort_values('tree_size')

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.axhline(y=1.0, color=REF_COLOR, linestyle='--',
               linewidth=0.6, zorder=0)
    ax.plot(df['tree_size'], df['speedup'],
            linestyle='-', marker='o', color=OURS_COLOR, label='Ours vs FlashInfer')

    ax.set_xscale('log')
    ax.set_xlabel('Sequoia tree size (nodes)')
    ax.set_ylabel(r'End-to-end speedup ($\times$)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    _save('fig5_e2e_sequoia')


# ---------------------------------------------------------------------------
# Fig 6 — EAGLE-3 E2E speedup vs total tokens (single column)
# ---------------------------------------------------------------------------
def plot_eagle(eagle):
    eagle = eagle.copy()
    eagle['b'] = eagle['label'].str.extract(r'b=(\d+)').astype(float)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axhline(y=1.0, color=REF_COLOR, linestyle='--',
               linewidth=0.6, zorder=0)

    bs = sorted(eagle['b'].dropna().unique())
    for i, b in enumerate(bs):
        sub = eagle[eagle['b'] == b].sort_values('total_token')
        n = len(sub)
        me = _markevery(n, target=6)
        ax.plot(sub['total_token'], sub['e2e_speedup'],
                linestyle='-', marker='o', markevery=me,
                color=TOL[i % len(TOL)], label=f'$b{{=}}{int(b)}$')

    ax.set_xlabel(r'Total tokens per tree $tt$')
    ax.set_ylabel(r'End-to-end speedup ($\times$)')
    ax.legend(loc='upper left', ncol=2, columnspacing=1.0)
    plt.tight_layout()
    _save('fig6_e2e_eagle')


if __name__ == "__main__":
    print("Loading data...")
    micro, amdahl, sequoia, eagle = load_data()

    print("Generating Figure 1...");  plot_micro_fixed_b(micro)
    print("Generating Figure 2...");  plot_micro_fixed_depth(micro)
    print("Generating Figure 3...");  plot_micro_sweep_n(micro)
    print("Generating Figure 4...");  plot_amdahl(amdahl)
    print("Generating Figure 5...");  plot_sequoia(sequoia)
    print("Generating Figure 6...");  plot_eagle(eagle)

    print(f"All figures generated successfully in {FIG_DIR}")
