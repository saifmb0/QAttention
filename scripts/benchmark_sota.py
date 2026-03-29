"""
benchmark_sota.py
=================
SOTA comparison benchmark — RTX 6000 ADA PRO 96 GB (SM 8.9 Ada Lovelace)
"blackwell" branch

Compares ragged ancestor-sparse attention against every relevant baseline
for tree-structured speculative decoding:

┌─────────────────────────────────────────────────────────────────────────┐
│ Method                      │ Library          │ Type        │ Mask    │
├─────────────────────────────┼──────────────────┼─────────────┼─────────┤
│ Ours (ragged, fp16)         │ Triton (this)    │ ragged      │ sparse  │
│ Ours (ragged, bf16)         │ Triton (this)    │ ragged      │ sparse  │
│ PyTorch SDPA — math         │ torch            │ padded      │ dense   │
│ PyTorch SDPA — flash (FA2)  │ torch            │ padded      │ dense   │
│ PyTorch SDPA — mem-eff.     │ torch            │ padded      │ dense   │
│ FlashAttention-2            │ flash_attn       │ padded/var  │ causal* │
│ FlashInfer — batch prefill  │ flashinfer       │ ragged      │ sparse* │
│ xformers — mem-eff          │ xformers         │ padded      │ dense   │
└─────────────────────────────┴──────────────────┴─────────────┴─────────┘

  * FlashAttention-2 and FlashInfer use standard causal masking, NOT the
    tree-ancestor mask.  They are included as compute-ceiling references:
    the best any padded/ragged method could possibly do on this input shape.

Notes on research context
--------------------------
  EAGLE-2 (Li et al. 2024, arXiv:2406.16858) drives the demand for
  tree-structured attention.  vLLM's current verification path uses
  PyTorch SDPA with an explicit tree mask bias tensor (math backend for
  correctness; flash backend for simple causal masks).  FlashInfer
  (Ye et al. 2024, arXiv:2312.11508) is the closest published ragged
  attention library; it targets standard causal attention and does not
  explicitly support the ancestor-sparse tree pattern.

  This benchmark measures the latency of the *attention kernel only*,
  which is the bottlenecked component during the verification pass.

Prerequisites (auto-checked at startup)
-----------------------------------------
  Required : torch >= 2.1,  triton >= 2.3
  Optional : flash_attn >= 2.5   (pip install flash-attn --no-build-isolation)
             flashinfer           (pip install flashinfer)
             xformers             (pip install xformers)

Usage
------
  # Quickstart — all configs, saves CSVs + plots under results/
  python scripts/benchmark_sota.py

  # Selected sweep (faster)
  python scripts/benchmark_sota.py --batch-sizes 1,8,32 --depths 1,3,5

  # Skip optional SOTA libs, only compare SDPA backends
  python scripts/benchmark_sota.py --skip-flashattn --skip-flashinfer --skip-xformers

  # Disable plots (CI / headless mode)
  python scripts/benchmark_sota.py --no-plot

  # Run with bfloat16 (Ada natively accelerated)
  python scripts/benchmark_sota.py --dtype bf16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
import importlib.util
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

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

# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES       = [1, 2, 4, 8, 16, 32]
DEFAULT_DEPTHS            = [1, 2, 3, 4, 5]
DEFAULT_BRANCHING_FACTORS = [2, 3, 4]
CTX_LEN                   = 128          # context prefix length (not benchmarked)
NUM_HEADS                 = 8
HEAD_DIM                  = 64
WARMUP_ITERS              = 10
BENCH_ITERS               = 50

# ─────────────────────────────────────────────────────────────────────────────
# Optional library probes
# ─────────────────────────────────────────────────────────────────────────────

def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


HAS_FLASH_ATTN  = _has("flash_attn")
HAS_FLASHINFER  = _has("flashinfer")
HAS_XFORMERS    = _has("xformers")


# ─────────────────────────────────────────────────────────────────────────────
# GPU capability check
# ─────────────────────────────────────────────────────────────────────────────

def device_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "CPU", "sm": "N/A", "vram_gb": 0, "hbm_bw": 0}
    p = torch.cuda.get_device_properties(0)
    return {
        "name":    p.name,
        "sm":      f"{p.major}{p.minor}",
        "vram_gb": round(p.total_memory / 1024**3, 1),
        "sm_count": p.multi_processor_count,
        # Ada Lovelace (SM 8.9) HBM BW: 820 GB/s (RTX 6000 ADA PRO GDDR7)
        "is_ada":  (p.major, p.minor) >= (8, 9),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_ms(fn: Callable, warmup: int, iters: int) -> float:
    """CUDA-event median timing, ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end   = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        ev_start.record()
        fn()
        ev_end.record()
        torch.cuda.synchronize()
        times.append(ev_start.elapsed_time(ev_end))
    return float(np.median(times))


def _try_time(fn: Callable, warmup: int, iters: int, label: str) -> float:
    try:
        return _time_ms(fn, warmup, iters)
    except Exception as exc:
        warnings.warn(f"[{label}] failed: {exc}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# FLOP counting
# ─────────────────────────────────────────────────────────────────────────────

def _ragged_flops(seq_lens: list, H: int, D: int) -> float:
    """Valid (non-padding) FLOPs for the ancestor-sparse kernel."""
    # For each token, only MAX_DEPTH+1 KV positions are attended.
    # We report quadratic-equivalent FLOPs (QK^T + AV) over valid tokens.
    return 4.0 * sum(l * l for l in seq_lens) * D * H


def _padded_flops(seq_lens: list, H: int, D: int) -> float:
    L_max = max(seq_lens)
    B     = len(seq_lens)
    return 4.0 * B * L_max * L_max * D * H


def _to_tflops(flops: float, ms: float) -> float:
    if math.isnan(ms) or ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


# ─────────────────────────────────────────────────────────────────────────────
# Input construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ragged(B, N, H, D, device, dtype):
    qs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    ks = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    vs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    Q, K, V, cu_sl = pack_inputs(qs, ks, vs)
    return Q.to(device), K.to(device), V.to(device), cu_sl, qs, ks, vs


def _make_padded(qs, ks, vs, masks_np, L_max, B, H, D, device, dtype):
    """Return (Q_pad, K_pad, V_pad, attn_bias) in [B, H, L, D]."""
    NEG_INF = torch.finfo(torch.float32).min / 2

    def _pad(ts):
        out = torch.zeros(B, L_max, H, D, device=device, dtype=dtype)
        for i, t in enumerate(ts):
            out[i, :t.shape[0]] = t
        return out.permute(0, 2, 1, 3)   # [B, H, L, D]

    Q_p = _pad(qs)
    K_p = _pad(ks)
    V_p = _pad(vs)

    bias = torch.full((B, 1, L_max, L_max), NEG_INF, device=device)
    for i, m in enumerate(masks_np):
        Li = m.shape[0]
        tb = torch.from_numpy(m.astype(np.float32)).to(device)
        bias[i, 0, :Li, :Li] = torch.where(tb.bool(),
                                            torch.zeros_like(tb),
                                            torch.full_like(tb, NEG_INF))
    return Q_p, K_p, V_p, bias


# ─────────────────────────────────────────────────────────────────────────────
# Per-method runner functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_runner_ragged(Q, K, V, cu_sl, b, d):
    def fn():
        ragged_attention(Q, K, V, cu_sl, branching_factor=b, max_depth=d)
    return fn


def _make_runner_sdpa_math(Q_p, K_p, V_p, bias):
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           attn_mask=bias, scale=scale)
    return fn


def _make_runner_sdpa_flash(Q_p, K_p, V_p):
    """Flash backend — no custom bias, uses causal=True (upper bound)."""
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           is_causal=True, scale=scale)
    return fn


def _make_runner_sdpa_memeff(Q_p, K_p, V_p, bias):
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           attn_mask=bias, scale=scale)
    return fn


def _make_runner_flash_attn(Q_p, K_p, V_p):
    """FlashAttention-2 via flash_attn library (causal, no tree bias)."""
    if not HAS_FLASH_ATTN:
        return None
    try:
        from flash_attn import flash_attn_func  # type: ignore
        # flash_attn_func expects [B, L, H, D]
        q = Q_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        k = K_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        v = V_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        def fn():
            flash_attn_func(q, k, v, causal=True)
        return fn
    except Exception as exc:
        warnings.warn(f"[flash_attn] setup failed: {exc}")
        return None


def _make_runner_flashinfer(Q, K, V, cu_sl, B, L_max, H, D, device):
    """
    FlashInfer batch prefill with causal mask (no tree bias).
    Uses BatchPrefillWithRaggedKVCacheWrapper for variable-length sequences.
    """
    if not HAS_FLASHINFER:
        return None
    try:
        import flashinfer  # type: ignore
        q_indptr = cu_sl.to(device)
        kv_indptr = cu_sl.to(device)
        Q_fi = Q.to(torch.float16)    # flashinfer requires fp16
        K_fi = K.to(torch.float16)
        V_fi = V.to(torch.float16)

        workspace_buf = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buf, kv_layout="NHD"
        )
        wrapper.plan(
            q_indptr, kv_indptr, kv_indptr,
            num_qo_heads=H, num_kv_heads=H, head_dim=D,
            causal=True
        )
        def fn():
            wrapper.run(Q_fi, K_fi, V_fi)
        # Warmup once to JIT compile
        try:
            fn()
            torch.cuda.synchronize()
        except Exception:
            pass
        return fn
    except Exception as exc:
        warnings.warn(f"[flashinfer] setup failed: {exc}")
        return None


def _make_runner_xformers(Q_p, K_p, V_p):
    """xformers memory-efficient attention (causal, no tree bias)."""
    if not HAS_XFORMERS:
        return None
    try:
        import xformers.ops as xops  # type: ignore
        q = Q_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)  # [B, L, H, D]
        k = K_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        v = V_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        def fn():
            xops.memory_efficient_attention(q, k, v,
                                            attn_bias=xops.LowerTriangularMask())
        return fn
    except Exception as exc:
        warnings.warn(f"[xformers] setup failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Single benchmark point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchRow:
    batch_size:        int
    branching_factor:  int
    tree_depth:        int
    num_tree_nodes:    int
    attn_padding_ratio: float
    # latencies (ms) — NaN = skipped / unavailable
    ragged_fp16_ms:    float
    ragged_bf16_ms:    float
    sdpa_math_ms:      float
    sdpa_flash_ms:     float
    sdpa_memeff_ms:    float
    flash_attn2_ms:    float   # FlashAttention-2 lib (causal, upper-bound ref)
    flashinfer_ms:     float   # FlashInfer ragged prefill (causal, upper-bound ref)
    xformers_ms:       float   # xformers mem-eff (causal, upper-bound ref)
    # TFLOPS
    ragged_tflops:     float
    sdpa_math_tflops:  float
    # speedups vs SDPA-math (primary production baseline)
    speedup_vs_sdpa_math:  float
    speedup_vs_sdpa_flash: float
    speedup_vs_fa2:        float
    speedup_vs_flashinfer: float


def benchmark_one(
    batch_size:        int,
    branching_factor:  int,
    depth:             int,
    dtype_str:         str  = "fp16",
    num_heads:         int  = NUM_HEADS,
    head_dim:          int  = HEAD_DIM,
    ctx_len:           int  = CTX_LEN,
    warmup:            int  = WARMUP_ITERS,
    iters:             int  = BENCH_ITERS,
    device:            torch.device | None = None,
    skip_flashattn:    bool = False,
    skip_flashinfer:   bool = False,
    skip_xformers:     bool = False,
) -> BenchRow:
    if device is None:
        device = torch.device("cuda")

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    torch.manual_seed(batch_size * 1000 + branching_factor * 100 + depth)

    N    = num_tree_nodes(branching_factor, depth)
    B, H, D = batch_size, num_heads, head_dim

    masks_np = [tree_attention_mask(branching_factor, depth) for _ in range(B)]
    seq_lens = [N] * B

    # ── Build inputs ────────────────────────────────────────────────────────
    Q_r16, K_r16, V_r16, cu_sl, qs_fp16, ks_fp16, vs_fp16 = _make_ragged(
        B, N, H, D, device, torch.float16
    )
    Q_rbf, K_rbf, V_rbf, cu_sl_bf, _, _, _ = _make_ragged(
        B, N, H, D, device, torch.bfloat16
    )
    Q_p, K_p, V_p, attn_bias = _make_padded(
        qs_fp16, ks_fp16, vs_fp16, masks_np,
        N, B, H, D, device, torch.float16
    )

    # ── Build runners ────────────────────────────────────────────────────────
    run_ragged_fp16 = _make_runner_ragged(Q_r16, K_r16, V_r16, cu_sl,
                                          branching_factor, depth)
    run_ragged_bf16 = _make_runner_ragged(Q_rbf, K_rbf, V_rbf, cu_sl_bf,
                                          branching_factor, depth)
    run_sdpa_math   = _make_runner_sdpa_math(Q_p, K_p, V_p, attn_bias)
    run_sdpa_flash  = _make_runner_sdpa_flash(Q_p, K_p, V_p)
    run_sdpa_meff   = _make_runner_sdpa_memeff(Q_p, K_p, V_p, attn_bias)

    run_fa2         = None if skip_flashattn  else _make_runner_flash_attn(Q_p, K_p, V_p)
    run_fi          = None if skip_flashinfer else _make_runner_flashinfer(
        Q_r16, K_r16, V_r16, cu_sl, B, N, H, D, device
    )
    run_xf          = None if skip_xformers   else _make_runner_xformers(Q_p, K_p, V_p)

    # ── Time everything ──────────────────────────────────────────────────────
    t_r16  = _try_time(run_ragged_fp16, warmup, iters, "ragged_fp16")
    t_rbf  = _try_time(run_ragged_bf16, warmup, iters, "ragged_bf16")
    t_sm   = _try_time(run_sdpa_math,   warmup, iters, "sdpa_math")
    t_sf   = _try_time(run_sdpa_flash,  warmup, iters, "sdpa_flash")
    t_me   = _try_time(run_sdpa_meff,   warmup, iters, "sdpa_memeff")
    t_fa2  = _try_time(run_fa2,         warmup, iters, "flash_attn2") if run_fa2 else float("nan")
    t_fi   = _try_time(run_fi,          warmup, iters, "flashinfer")  if run_fi  else float("nan")
    t_xf   = _try_time(run_xf,          warmup, iters, "xformers")   if run_xf  else float("nan")

    # ── Metrics ──────────────────────────────────────────────────────────────
    r_flops = _ragged_flops(seq_lens, H, D)
    p_flops = _padded_flops(seq_lens, H, D)
    pad_rat = 1.0 - sum(l * l for l in seq_lens) / (B * N * N)

    def _spdup(t_ref, t_our):
        if math.isnan(t_ref) or math.isnan(t_our) or t_our <= 0:
            return float("nan")
        return round(t_ref / t_our, 3)

    # Use fp16 ragged as the "our" method for speedup denominators
    t_ours = t_r16

    return BenchRow(
        batch_size=B,
        branching_factor=branching_factor,
        tree_depth=depth,
        num_tree_nodes=N,
        attn_padding_ratio=round(pad_rat, 4),
        ragged_fp16_ms=round(t_r16, 4),
        ragged_bf16_ms=round(t_rbf, 4),
        sdpa_math_ms  =round(t_sm,  4),
        sdpa_flash_ms =round(t_sf,  4),
        sdpa_memeff_ms=round(t_me,  4),
        flash_attn2_ms=round(t_fa2, 4),
        flashinfer_ms =round(t_fi,  4),
        xformers_ms   =round(t_xf,  4),
        ragged_tflops =round(_to_tflops(r_flops, t_r16), 3),
        sdpa_math_tflops=round(_to_tflops(p_flops, t_sm), 3),
        speedup_vs_sdpa_math =_spdup(t_sm,  t_ours),
        speedup_vs_sdpa_flash=_spdup(t_sf,  t_ours),
        speedup_vs_fa2       =_spdup(t_fa2, t_ours),
        speedup_vs_flashinfer=_spdup(t_fi,  t_ours),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_ADA_PALETTE = {
    "ragged_fp16":   "#00b4d8",
    "ragged_bf16":   "#0096c7",
    "sdpa_math":     "#ef233c",
    "sdpa_flash":    "#fb8500",
    "sdpa_memeff":   "#ffb703",
    "flash_attn2":   "#8338ec",
    "flashinfer":    "#3a86ff",
    "xformers":      "#06d6a0",
}

_METHOD_COLS = {
    "Ragged fp16 (ours)":   "ragged_fp16_ms",
    "Ragged bf16 (ours)":   "ragged_bf16_ms",
    "SDPA math":            "sdpa_math_ms",
    "SDPA flash":           "sdpa_flash_ms",
    "SDPA mem-eff":         "sdpa_memeff_ms",
    "FlashAttention-2":     "flash_attn2_ms",
    "FlashInfer":           "flashinfer_ms",
    "xformers":             "xformers_ms",
}


def plot_latency_vs_depth(df: pd.DataFrame, out_dir: str) -> None:
    """One figure per branching_factor: latency (ms) vs tree depth, B=8."""
    for b in sorted(df["branching_factor"].unique()):
        sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)].sort_values("tree_depth")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = list(_ADA_PALETTE.values())
        for ci, (label, col) in enumerate(_METHOD_COLS.items()):
            vals = sub[col].values
            if np.all(np.isnan(vals)):
                continue
            ax.plot(sub["tree_depth"], vals, marker="o",
                    label=label, color=colors[ci % len(colors)],
                    linewidth=1.8, markersize=5)
        ax.set_xlabel("Tree depth  d", fontsize=11)
        ax.set_ylabel("Latency  (ms)", fontsize=11)
        ax.set_title(
            f"Attention Kernel Latency — b={b}, B=8, H={NUM_HEADS}, D={HEAD_DIM}\n"
            f"RTX 6000 ADA PRO  (SM 8.9 Ada Lovelace)",
            fontsize=11
        )
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sota_latency_b{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_speedup_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Speedup of our ragged fp16 vs SDPA-math across B × depth."""
    for b in sorted(df["branching_factor"].unique()):
        sub = df[df["branching_factor"] == b]
        pivot = sub.pivot_table(
            index="tree_depth", columns="batch_size",
            values="speedup_vs_sdpa_math", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        vmax = max(3.0, float(np.nanmax(pivot.values)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=0.5, vmax=vmax, origin="lower")
        fig.colorbar(im, ax=ax, label="Speedup vs SDPA-math")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Batch size  B", fontsize=10)
        ax.set_ylabel("Tree depth  d", fontsize=10)
        ax.set_title(
            f"Ragged fp16 vs SDPA-math  (b={b})  —  RTX 6000 ADA PRO",
            fontsize=10
        )
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not math.isnan(val):
                    ax.text(c, r, f"{val:.2f}×", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if val < 1.2 else "black")
        plt.tight_layout()
        path = os.path.join(out_dir, f"sota_speedup_heatmap_b{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_bar_chart(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: median latency across all configs for each method."""
    medians = {}
    for label, col in _METHOD_COLS.items():
        vals = df[col].dropna().values
        if len(vals):
            medians[label] = float(np.median(vals))

    if not medians:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(medians.keys())
    vals   = [medians[l] for l in labels]
    colors = [list(_ADA_PALETTE.values())[i % len(_ADA_PALETTE)] for i in range(len(labels))]
    bars   = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="{:.3f} ms", fontsize=8, padding=2)
    ax.set_ylabel("Median latency across all configs  (ms)", fontsize=10)
    ax.set_title(
        f"SOTA Attention Methods — RTX 6000 ADA PRO (SM 8.9)\n"
        f"B ∈ {DEFAULT_BATCH_SIZES}, d ∈ {DEFAULT_DEPTHS}, b ∈ {DEFAULT_BRANCHING_FACTORS}",
        fontsize=10
    )
    ax.tick_params(axis="x", labelrotation=25, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "sota_median_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tflops(df: pd.DataFrame, out_dir: str) -> None:
    """TFLOPS comparison: ragged (sparse) vs SDPA-math (padded)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for b in sorted(df["branching_factor"].unique()):
        sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)].sort_values("tree_depth")
        axes[0].plot(sub["tree_depth"], sub["ragged_tflops"],
                     marker="o", label=f"Ragged b={b}")
        axes[1].plot(sub["tree_depth"], sub["sdpa_math_tflops"],
                     marker="s", linestyle="--", label=f"SDPA-math b={b}")
    for ax, title in zip(axes, ["Ragged fp16 (sparse)", "SDPA math (padded)"]):
        ax.set_xlabel("Tree depth  d", fontsize=10)
        ax.set_ylabel("Effective TFLOPS", fontsize=10)
        ax.set_title(f"{title}  B=8  —  RTX 6000 ADA PRO", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        # Ada Ada peak reference lines
        ax.axhline(364.2, color="gray", linestyle=":", linewidth=0.9,
                   label="Ada FP16 peak (364 TFLOPS)")
    plt.tight_layout()
    path = os.path.join(out_dir, "sota_tflops.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(info: dict, args) -> None:
    print("=" * 70)
    print("  sd-ragged  ·  SOTA Benchmark  ·  blackwell branch")
    print("=" * 70)
    print(f"  Device : {info['name']}")
    print(f"  SM     : {info['sm']}  ({info['sm_count']} SMs)  {'[Ada Lovelace ✓]' if info.get('is_ada') else '[Non-Ada — configs may differ]'}")
    print(f"  VRAM   : {info['vram_gb']} GB")
    print(f"  dtype  : {args.dtype}")
    print()
    print("  Optional SOTA backends:")
    print(f"    FlashAttention-2 : {'available' if HAS_FLASH_ATTN else 'NOT INSTALLED (pip install flash-attn)'}")
    print(f"    FlashInfer       : {'available' if HAS_FLASHINFER else 'NOT INSTALLED (pip install flashinfer)'}")
    print(f"    xformers         : {'available' if HAS_XFORMERS  else 'NOT INSTALLED (pip install xformers)'}")
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sd-ragged SOTA benchmark — RTX 6000 ADA PRO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir",          default="results",
                        help="Output directory for CSV and plots (default: results)")
    parser.add_argument("--dtype",             default="fp16", choices=["fp16", "bf16"],
                        help="Primary dtype (default: fp16)")
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help="Comma-separated batch sizes")
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)),
                        help="Comma-separated tree depths")
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)),
                        help="Comma-separated branching factors")
    parser.add_argument("--warmup",            type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",             type=int, default=BENCH_ITERS)
    parser.add_argument("--skip-flashattn",    action="store_true")
    parser.add_argument("--skip-flashinfer",   action="store_true")
    parser.add_argument("--skip-xformers",     action="store_true")
    parser.add_argument("--no-plot",           action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available — benchmark requires a CUDA GPU.")
        sys.exit(1)

    info = device_info()
    _print_banner(info, args)

    batch_sizes       = list(map(int, args.batch_sizes.split(",")))
    depths            = list(map(int, args.depths.split(",")))
    branching_factors = list(map(int, args.branching_factors.split(",")))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    configs = [
        (B, b, d)
        for B in batch_sizes
        for b in branching_factors
        for d in depths
    ]
    total = len(configs)
    print(f"Running {total} configurations  [{args.warmup} warmup + {args.iters} timed iters each]\n")

    rows = []
    for idx, (B, b, d) in enumerate(configs):
        row = benchmark_one(
            batch_size=B,
            branching_factor=b,
            depth=d,
            dtype_str=args.dtype,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            skip_flashattn  =args.skip_flashattn,
            skip_flashinfer =args.skip_flashinfer,
            skip_xformers   =args.skip_xformers,
        )
        rows.append(asdict(row))

        # ── Per-row progress ─────────────────────────────────────────────────
        def _fmt(ms):
            return f"{ms:.3f}ms" if not math.isnan(ms) else "  n/a "
        extras = ""
        if not math.isnan(row.flash_attn2_ms):
            extras += f"  fa2={_fmt(row.flash_attn2_ms)}"
        if not math.isnan(row.flashinfer_ms):
            extras += f"  fi={_fmt(row.flashinfer_ms)}"
        print(
            f"  [{idx+1:3d}/{total}]  B={B:2d} b={b} d={d} │ "
            f"ragged={_fmt(row.ragged_fp16_ms)}  bf16={_fmt(row.ragged_bf16_ms)} │ "
            f"sdpa_math={_fmt(row.sdpa_math_ms)}  sdpa_flash={_fmt(row.sdpa_flash_ms)} │ "
            f"spdup_vs_math={row.speedup_vs_sdpa_math:.2f}×{extras}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "sota_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Speedup vs SDPA-math (mean over batch sizes) ─────────────────────")
    if "speedup_vs_sdpa_math" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_sdpa_math"]
            .mean().round(2).unstack().to_string()
        )
    print("\n── Speedup vs SDPA-flash (mean over batch sizes) ────────────────────")
    if "speedup_vs_sdpa_flash" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_sdpa_flash"]
            .mean().round(2).unstack().to_string()
        )

    if not args.no_plot:
        print("\nGenerating plots …")
        plot_latency_vs_depth(df, args.out_dir)
        plot_speedup_heatmap(df, args.out_dir)
        plot_bar_chart(df, args.out_dir)
        plot_tflops(df, args.out_dir)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
