#!/usr/bin/env python3
"""
profile_kernels.py — NCU Hardware-Counter Profiling: QAttention vs FlashInfer vs DeFT
=====================================================================================

Collects NVIDIA Nsight Compute (ncu) hardware counters for a side-by-side
profiling breakdown that NeurIPS Systems Track reviewers expect to see:

  Metric                       Why It Matters
  ───────────────────────────── ───────────────────────────────────────────────
  DRAM (HBM) Read Volume (GB)  Proves QAttention's O(N·d) vs FlashInfer's O(N²)
  DRAM (HBM) Write Volume (GB) Confirms output write cost is identical
  L2 Cache Hit Rate (%)        Validates implicit sibling coalescing theory
  L1 Cache Hit Rate (%)        Secondary cache hierarchy utilisation
  Achieved Occupancy (%)       Shows register pressure isn't killing parallelism
  Registers Per Thread          Quantifies depth-unrolling cost
  SM Throughput (%)            Memory-bound vs compute-bound classification
  HBM Read Amplification       measured_reads / theoretical_minimum_reads

Architecture
────────────
The script has two execution modes:

  Worker mode  (--run-kernel <name>):
    Invoked by the orchestrator under `ncu --profile-from-start off`.
    Sets up tensors, warms up (JIT compile), then brackets exactly one
    kernel invocation with cudaProfilerStart/cudaProfilerStop.

  Orchestrator mode  (default):
    Iterates over a config grid × {qattention, flashinfer, deft}.
    For each combination, spawns an ncu subprocess in worker mode,
    parses the CSV output, aggregates multi-kernel operations, computes
    theoretical bounds, and writes the comparison to CSV + terminal.

Output
──────
  results/profile_breakdown.csv   — full per-kernel per-config hardware counters
  Terminal comparison table       — side-by-side HBM volume, L2 hit rate, occupancy

Prerequisites
─────────────
  • ncu (NVIDIA Nsight Compute) on PATH — ships with CUDA Toolkit ≥ 11.0
  • May require `sudo` or admin profiling permissions (see --sudo flag)
  • On cloud instances: set /proc/driver/nvidia/params NVreg_RestrictProfilingToAdminUsers=0

Usage
─────
  # Full profiling (3 kernels × 6 configs ≈ 5-15 min depending on GPU):
  python scripts/profile_kernels.py --sudo

  # Specific configs (faster):
  python scripts/profile_kernels.py --sudo --batch-sizes 1 --depths 7

  # Skip kernels not installed on target:
  python scripts/profile_kernels.py --sudo --skip-deft

  # Dry run — print ncu commands without executing:
  python scripts/profile_kernels.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# NCU metric list — these are the hardware counters we collect
# ─────────────────────────────────────────────────────────────────────────────
NCU_METRICS = [
    "dram__bytes_read.sum",                                      # HBM read (bytes)
    "dram__bytes_write.sum",                                     # HBM write (bytes)
    "lts__t_sector_hit_rate.pct",                                # L2 cache hit rate (%)
    "l1tex__t_sector_hit_rate.pct",                              # L1 cache hit rate (%)
    "sm__warps_active.avg.pct_of_peak_sustained_active",         # achieved occupancy (%)
    "gpu__time_duration.sum",                                    # kernel duration (ns)
    "launch__registers_per_thread",                              # registers / thread
    "launch__occupancy_limit_registers",                         # occupancy limited by regs (%)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",          # SM throughput (%)
]

# ─────────────────────────────────────────────────────────────────────────────
# Defaults — small grid because ncu is slow (~10-30s per kernel invocation)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES       = [1, 8]
DEFAULT_BRANCHING_FACTORS = [3, 7, 14, 30, 60]
DEFAULT_DEPTHS            = [3, 7, 14, 30, 60]
DEFAULT_NUM_HEADS         = 32
DEFAULT_HEAD_DIM          = 128
WARMUP_ITERS              = 5  # enough for Triton JIT + FlashInfer JIT


def _tree_n_budgeted(b: int, d: int) -> int:
    """EAGLE-like budgeted node count: N ≈ 6·b·d/7, floor 30."""
    return max(30, round(6 * b * d / 7))


def _compute_actual_depth(N: int, b: int) -> int:
    """Compute actual tree depth for N nodes in a balanced b-ary tree."""
    actual_d = 0
    k = N - 1
    while k > 0:
        k = (k - 1) // b
        actual_d += 1
    return actual_d


# ═════════════════════════════════════════════════════════════════════════════
#  WORKER MODE — executed under ncu, profiles a single kernel invocation
# ═════════════════════════════════════════════════════════════════════════════

def _worker_setup_tensors(B, b, d, H, D):
    """Create Q, K, V, cu_seqlens, tree mask for a balanced b-ary tree.

    Returns (Q, K, V, cu, mask_np, N, actual_d, scale, device).
    """
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.tree_mask import tree_attention_mask_n

    N = _tree_n_budgeted(b, d)
    actual_d = _compute_actual_depth(N, b)
    tot = B * N
    device = torch.device("cuda:0")

    torch.manual_seed(B * 10000 + b * 100 + d)
    Q = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    K = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    V = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    cu = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)

    mask_np = tree_attention_mask_n(b, N)
    scale = 1.0 / math.sqrt(D)

    return Q, K, V, cu, mask_np, N, actual_d, scale, device


def _profiler_start():
    """Start CUDA profiling (ncu captures kernels after this call)."""
    import torch
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        # Fallback: direct ctypes call
        import ctypes
        try:
            _cudart = ctypes.CDLL("libcudart.so")
            _cudart.cudaProfilerStart()
        except Exception:
            pass  # ncu will still capture if --profile-from-start is not off


def _profiler_stop():
    """Stop CUDA profiling."""
    import torch
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        import ctypes
        try:
            _cudart = ctypes.CDLL("libcudart.so")
            _cudart.cudaProfilerStop()
        except Exception:
            pass


def _worker_qattention(B, b, d, H, D, timing_only=False):
    """Worker: profile QAttention ragged kernel."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.ragged_attn import ragged_attention

    Q, K, V, cu, mask_np, N, actual_d, scale, device = _worker_setup_tensors(B, b, d, H, D)

    def run_fn():
        ragged_attention(Q, K, V, cu, b, actual_d, max_seqlen=N)

    # Warmup — JIT compile Triton kernel + stabilise caches
    for _ in range(WARMUP_ITERS):
        run_fn()
    torch.cuda.synchronize()

    if timing_only:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iters = 50
        start_event.record()
        for _ in range(iters):
            run_fn()
        end_event.record()
        torch.cuda.synchronize()
        duration_ns = (start_event.elapsed_time(end_event) / iters) * 1_000_000
        print(f"TIMING: duration_ns={duration_ns:.3f}")
    else:
        # Profiled region — exactly one kernel launch
        _profiler_start()
        run_fn()
        torch.cuda.synchronize()
        _profiler_stop()


def _worker_flashinfer(B, b, d, H, D, timing_only=False):
    """Worker: profile FlashInfer BatchPrefillWithRaggedKVCache (tree mask)."""
    import torch
    import flashinfer
    import flashinfer.prefill

    Q, K, V, cu, mask_np, N, actual_d, scale, device = _worker_setup_tensors(B, b, d, H, D)

    # Build FlashInfer wrapper with flattened tree mask
    m_tree = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
    flattened_mask = m_tree.repeat(B, 1).flatten()

    kv_indptr = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout="NHD"
    )
    wrapper.plan(
        qo_indptr, kv_indptr, H, H, D,
        custom_mask=flattened_mask, causal=False,
        sm_scale=scale, q_data_type=Q.dtype,
    )

    def run_fn():
        wrapper.run(Q, K, V)

    # Warmup
    for _ in range(WARMUP_ITERS):
        run_fn()
    torch.cuda.synchronize()

    if timing_only:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iters = 50
        start_event.record()
        for _ in range(iters):
            run_fn()
        end_event.record()
        torch.cuda.synchronize()
        duration_ns = (start_event.elapsed_time(end_event) / iters) * 1_000_000
        print(f"TIMING: duration_ns={duration_ns:.3f}")
    else:
        # Profiled region
        _profiler_start()
        run_fn()
        torch.cuda.synchronize()
        _profiler_stop()


def _worker_deft(B, b, d, H, D, timing_only=False):
    """Worker: profile DeFT Triton kernel."""
    import torch

    _DEFT_KERNEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "FastTree", "kernel_bench",
    )
    if not os.path.isfile(os.path.join(_DEFT_KERNEL_DIR, "DeFT.py")):
        print("[ERROR] DeFT not found. Run setup.sh to clone FastTree-Artifact.", file=sys.stderr)
        sys.exit(1)
    if _DEFT_KERNEL_DIR not in sys.path:
        sys.path.insert(0, _DEFT_KERNEL_DIR)

    from kv_tree_simple import KVTreeNode   # type: ignore
    from DeFT import DeFT_preparation, DeFT_decode  # type: ignore
    import DeFT as _deft_mod                # type: ignore

    Q, K, V, cu, mask_np, N, actual_d, scale, device = _worker_setup_tensors(B, b, d, H, D)

    # ── Build DeFT tree structures (mirrored from benchmark_micro.py) ────────
    parent_arr = [-1] + [(i - 1) // b for i in range(1, N)]
    children_d: list = [[] for _ in range(N)]
    for i in range(1, N):
        children_d[parent_arr[i]].append(i)

    def _bfs_sub(root):
        res, q, qi = [root], [root], 0
        while qi < len(q):
            for c in children_d[q[qi]]:
                res.append(c); q.append(c)
            qi += 1
        return res

    all_sub = [_bfs_sub(j) for j in range(N)]
    max_pl = actual_d + 1

    anc_chains = []
    for i in range(N):
        ch, cur = [], i
        while cur != -1:
            ch.append(cur); cur = parent_arr[cur]
        ch.reverse(); anc_chains.append(ch)

    idx_d = torch.zeros(N, max_pl, dtype=torch.long, device=device)
    for i in range(N):
        ch = anc_chains[i]
        for pos, anc in enumerate(ch):
            idx_d[i, pos] = anc
        for pos in range(len(ch), max_pl):
            idx_d[i, pos] = ch[-1]

    tree_info_d = []
    for j in range(N):
        node = KVTreeNode()
        node.parent = parent_arr[j]; node.id = j
        node.seqlen = 1; node.num_children = len(children_d[j])
        node.requests = all_sub[j]
        tree_info_d.append(node)

    K_flat_b0 = K[0:N].contiguous()
    K_cache_b0 = K_flat_b0[idx_d.view(-1)].view(N, max_pl, H, D).contiguous()

    _deft_mod.cur_length = 0
    DeFT_aux = DeFT_preparation(tree_info_d, K_cache_b0, 128, 64, H, D)
    sm_d = 1.0 / math.sqrt(D)
    Out_d = torch.empty(N, H, D, device=device, dtype=torch.float16)

    Qs_d, Kc_d, Vc_d = [], [], []
    for bi in range(B):
        Kb = K[bi * N:(bi + 1) * N].contiguous()
        Vb = V[bi * N:(bi + 1) * N].contiguous()
        Qs_d.append(Q[bi * N:(bi + 1) * N].contiguous())
        Kc_d.append(Kb[idx_d.view(-1)].view(N, max_pl, H, D).contiguous().view(-1, H, D))
        Vc_d.append(Vb[idx_d.view(-1)].view(N, max_pl, H, D).contiguous().view(-1, H, D))

    def run_fn():
        for bi in range(B):
            DeFT_decode(Qs_d[bi], Kc_d[bi], Vc_d[bi], Out_d, *DeFT_aux,
                        Q_TILE_SIZE=16, KV_TILE_SIZE=32,
                        sm_scale=sm_d, mask_len=64)

    # Warmup
    for _ in range(WARMUP_ITERS):
        run_fn()
    torch.cuda.synchronize()

    if timing_only:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        iters = 50
        start_event.record()
        for _ in range(iters):
            run_fn()
        end_event.record()
        torch.cuda.synchronize()
        duration_ns = (start_event.elapsed_time(end_event) / iters) * 1_000_000
        print(f"TIMING: duration_ns={duration_ns:.3f}")
    else:
        # Profiled region
        _profiler_start()
        run_fn()
        torch.cuda.synchronize()
        _profiler_stop()


_WORKER_DISPATCH = {
    "qattention": _worker_qattention,
    "flashinfer": _worker_flashinfer,
    "deft":       _worker_deft,
}


def worker_main(args):
    """Worker entry point — invoked under ncu or directly for timing."""
    runner = _WORKER_DISPATCH.get(args.run_kernel)
    if runner is None:
        print(f"[ERROR] Unknown kernel: {args.run_kernel}", file=sys.stderr)
        sys.exit(1)
    runner(args.B, args.b, args.d, args.H, args.D, timing_only=args.timing_only)


# ═════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR MODE — spawns ncu, parses CSV, builds comparison table
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ProfileResult:
    """One row of the profiling comparison table."""
    kernel:                    str
    batch_size:                int
    branching_factor:          int
    depth:                     int
    num_nodes:                 int
    actual_depth:              int   = 0
    hbm_read_bytes:            float = 0.0
    hbm_write_bytes:           float = 0.0
    hbm_read_gb:               float = 0.0
    hbm_write_gb:              float = 0.0
    l2_hit_rate_pct:           float = float("nan")
    l1_hit_rate_pct:           float = float("nan")
    occupancy_pct:             float = float("nan")
    occupancy_limit_regs_pct:  float = float("nan")
    duration_ns:               float = 0.0
    duration_us:               float = 0.0
    registers_per_thread:      float = float("nan")
    sm_throughput_pct:         float = float("nan")
    theoretical_read_bytes:    float = 0.0
    theoretical_read_gb:       float = 0.0
    hbm_read_amplification:    float = float("nan")
    status:                    str   = "OK"


def _find_ncu() -> Optional[str]:
    """Locate ncu binary on the system."""
    ncu = shutil.which("ncu")
    if ncu:
        return ncu
    # Common CUDA Toolkit locations
    for base in [
        "/usr/local/cuda", "/usr/local/cuda-12.1", "/usr/local/cuda-12.4",
        "/usr/local/cuda-12.6", "/usr/local/cuda-12.8",
        "/opt/nvidia/nsight-compute",
    ]:
        candidate = os.path.join(base, "bin", "ncu")
        if os.path.isfile(candidate):
            return candidate
    return None


def _parse_ncu_csv(csv_text: str) -> List[Dict[str, str]]:
    """Parse ncu --csv output into a list of per-kernel metric dicts.

    ncu CSV format (one row per metric per kernel):
      "ID","Process ID","Process Name","Host Name","Kernel Name",
      "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

    Returns a list of dicts, one per unique kernel ID, each mapping
    metric_name → metric_value.
    """
    # Filter to just the CSV lines (ncu may print progress/info before CSV)
    lines = []
    capturing = False
    for line in csv_text.split("\n"):
        if not capturing and '"ID"' in line:
            capturing = True
        if capturing:
            lines.append(line)
    if not lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    kernels: Dict[str, Dict[str, str]] = {}
    for row in reader:
        kid = row.get("ID", "0")
        if kid not in kernels:
            kernels[kid] = {"_kernel_name": row.get("Kernel Name", "?")}
        metric = row.get("Metric Name", "")
        value = row.get("Metric Value", "")
        if metric and value:
            # ncu sometimes formats large numbers with commas — strip them
            kernels[kid][metric] = value.replace(",", "")
    return list(kernels.values())


def _safe_float(d: dict, key: str, default: float = 0.0) -> float:
    """Extract a float from a parsed ncu dict, returning default on failure."""
    v = d.get(key, "")
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _run_ncu_for_kernel(
    ncu_path: Optional[str],
    kernel_name: str,
    B: int, b: int, d: int, H: int, D: int,
    script_path: str,
    use_sudo: bool = False,
    timing_only: bool = False,
) -> Optional[List[Dict[str, str]]]:
    """Invoke ncu for one (kernel, config), return parsed per-CUDA-kernel metrics."""
    if timing_only:
        cmd = [
            sys.executable,
            script_path,
            "--run-kernel", kernel_name,
            "--B", str(B),
            "--b", str(b),
            "--d", str(d),
            "--H", str(H),
            "--D", str(D),
            "--timing-only",
        ]
    else:
        cmd = []
        if use_sudo:
            cmd.append("sudo")
        cmd += [
            ncu_path or "ncu",
            "--profile-from-start", "off",
            "--csv",
            "--print-units", "base",
            "--metrics", ",".join(NCU_METRICS),
            sys.executable,
            script_path,
            "--run-kernel", kernel_name,
            "--B", str(B),
            "--b", str(b),
            "--d", str(d),
            "--H", str(H),
            "--D", str(D),
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per profile
        )
        if result.returncode != 0:
            print(f"FAIL (exit {result.returncode})")
            if result.stderr and result.stderr.strip():
                print("    [stderr]")
                for line in result.stderr.strip().split("\n")[-15:]:
                    print(f"      {line}")
            if result.stdout and result.stdout.strip():
                print("    [stdout]")
                for line in result.stdout.strip().split("\n")[-15:]:
                    print(f"      {line}")
            return None

        if timing_only:
            duration_ns = None
            for line in result.stdout.split("\n"):
                if line.startswith("TIMING: duration_ns="):
                    try:
                        duration_ns = float(line.split("=")[1])
                        break
                    except (ValueError, IndexError):
                        pass
            if duration_ns is None:
                print("FAIL (could not parse duration_ns)")
                if result.stderr and result.stderr.strip():
                    print("    [stderr]")
                    for line in result.stderr.strip().split("\n")[-10:]:
                        print(f"      {line}")
                if result.stdout and result.stdout.strip():
                    print("    [stdout]")
                    for line in result.stdout.strip().split("\n")[-10:]:
                        print(f"      {line}")
                return None
            return [{"gpu__time_duration.sum": f"{duration_ns:.3f}"}]

        parsed = _parse_ncu_csv(result.stdout)
        if not parsed:
            err_msg = (result.stderr or "") + (result.stdout or "")
            if "permission" in err_msg.lower() or "err_nvgpuctrperm" in err_msg.lower():
                print("FAIL (permission denied — check GPU profiling permissions or run container with --privileged)")
            else:
                print("FAIL (no ncu CSV output)")
                if result.stderr and result.stderr.strip():
                    print("    [stderr]")
                    for line in result.stderr.strip().split("\n")[-10:]:
                        print(f"      {line}")
                if result.stdout and result.stdout.strip():
                    print("    [stdout]")
                    for line in result.stdout.strip().split("\n")[-10:]:
                        print(f"      {line}")
            return None
        return parsed

    except subprocess.TimeoutExpired:
        print("TIMEOUT (600s)")
        return None
    except Exception as e:
        print(f"ERROR ({e})")
        return None


def _theoretical_min_read_bytes(kernel: str, B: int, N: int, actual_d: int, H: int, D: int) -> float:
    """Theoretical minimum HBM read bytes for tree-only attention (L=0).

    For ANY correct tree attention kernel, the absolute minimum reads are:
      Q:         B × N × H × D × 2  (each query loaded once)
      K ancestors: B × N × (d+1) × H × D × 2  (each query's ancestor chain)
      V ancestors: B × N × (d+1) × H × D × 2  (same)

    QAttention achieves close to this minimum.
    FlashInfer/DeFT read significantly more due to block-granularity overhead.
    """
    elem = 2  # fp16 = 2 bytes per element
    q_bytes = B * N * H * D * elem                     # Q reads
    kv_bytes = 2 * B * N * (actual_d + 1) * H * D * elem  # K + V ancestor reads
    return float(q_bytes + kv_bytes)


def _aggregate_kernel_metrics(parsed: List[Dict[str, str]]) -> Dict[str, float]:
    """Aggregate metrics across multiple CUDA kernels in one logical operation.

    Bytes and durations are summed.
    Rates (hit rates, occupancy, throughput) are duration-weighted averages.
    Registers are taken as the max across kernels.
    """
    total_read   = sum(_safe_float(k, "dram__bytes_read.sum") for k in parsed)
    total_write  = sum(_safe_float(k, "dram__bytes_write.sum") for k in parsed)
    total_dur    = sum(_safe_float(k, "gpu__time_duration.sum") for k in parsed)

    def _weighted_avg(metric: str) -> float:
        pairs = []
        for km in parsed:
            val = _safe_float(km, metric, default=float("nan"))
            dur = _safe_float(km, "gpu__time_duration.sum", default=0.0)
            if not math.isnan(val) and dur > 0:
                pairs.append((val, dur))
        if not pairs:
            return float("nan")
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return float("nan")
        return sum(v * w for v, w in pairs) / total_w

    max_regs = float("nan")
    for km in parsed:
        r = _safe_float(km, "launch__registers_per_thread", default=float("nan"))
        if not math.isnan(r):
            max_regs = r if math.isnan(max_regs) else max(max_regs, r)

    max_occ_limit = float("nan")
    for km in parsed:
        r = _safe_float(km, "launch__occupancy_limit_registers", default=float("nan"))
        if not math.isnan(r):
            max_occ_limit = r if math.isnan(max_occ_limit) else min(max_occ_limit, r)

    return {
        "hbm_read_bytes":    total_read,
        "hbm_write_bytes":   total_write,
        "duration_ns":       total_dur,
        "l2_hit_rate_pct":   _weighted_avg("lts__t_sector_hit_rate.pct"),
        "l1_hit_rate_pct":   _weighted_avg("l1tex__t_sector_hit_rate.pct"),
        "occupancy_pct":     _weighted_avg("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "sm_throughput_pct": _weighted_avg("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "registers_per_thread": max_regs,
        "occupancy_limit_regs_pct": max_occ_limit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(results: List[ProfileResult], csv_path: str):
    """Write full profiling results to CSV."""
    fieldnames = [
        "kernel", "batch_size", "branching_factor", "depth", "num_nodes",
        "actual_depth", "status",
        "hbm_read_bytes", "hbm_write_bytes", "hbm_read_gb", "hbm_write_gb",
        "l2_hit_rate_pct", "l1_hit_rate_pct",
        "occupancy_pct", "occupancy_limit_regs_pct",
        "duration_ns", "duration_us",
        "registers_per_thread", "sm_throughput_pct",
        "theoretical_read_bytes", "theoretical_read_gb",
        "hbm_read_amplification",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for fn in fieldnames:
                v = getattr(r, fn)
                if isinstance(v, float):
                    row[fn] = f"{v:.6g}" if not math.isnan(v) else "nan"
                else:
                    row[fn] = v
            writer.writerow(row)


def _fv(val: float, width: int = 8, decimals: int = 4) -> str:
    """Format a float for table display, handling nan."""
    if math.isnan(val):
        return "n/a".center(width)
    if val == 0.0:
        return "0".center(width)
    return f"{val:{width}.{decimals}f}"


def _fv1(val: float, width: int = 5) -> str:
    """Format a percentage (1 decimal)."""
    if math.isnan(val):
        return "n/a".center(width)
    return f"{val:{width}.1f}"


def _print_comparison(results: List[ProfileResult]):
    """Print a side-by-side comparison table to terminal."""
    # Group by config
    configs: Dict[tuple, Dict[str, ProfileResult]] = {}
    for r in results:
        key = (r.batch_size, r.branching_factor, r.depth)
        if key not in configs:
            configs[key] = {}
        configs[key][r.kernel] = r

    kernels = ["qattention", "flashinfer", "deft"]
    kernel_labels = {"qattention": "QAttention", "flashinfer": "FlashInfer", "deft": "DeFT"}

    # ── Table 1: HBM Read Volume ─────────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  TABLE 1: HBM Read Volume (GB) — measures actual memory traffic")
    print("═" * 90)
    print(f"  {'Config':<20s}  {'QAttention':>12s}  {'FlashInfer':>12s}  {'DeFT':>12s}  {'QA Amplif.':>12s}")
    print("  " + "─" * 86)

    for key in sorted(configs.keys()):
        B, b, d = key
        N = _tree_n_budgeted(b, d)
        cfg = f"B={B:2d} b={b:2d} d={d:2d} N={N:3d}"
        kd = configs[key]
        vals = []
        for kn in kernels:
            r = kd.get(kn)
            if r and r.status == "OK" and r.hbm_read_bytes > 0:
                vals.append(f"{r.hbm_read_gb:12.4f}")
            else:
                vals.append("         n/a")
        # amplification for QAttention
        qa = kd.get("qattention")
        amp = f"{qa.hbm_read_amplification:12.2f}×" if (qa and not math.isnan(qa.hbm_read_amplification)) else "         n/a"
        print(f"  {cfg:<20s}  {vals[0]}  {vals[1]}  {vals[2]}  {amp}")

    # ── Table 2: L2 Cache Hit Rate ───────────────────────────────────────────
    print("\n" + "═" * 90)
    print("  TABLE 2: L2 Cache Hit Rate (%) — validates sibling coalescing")
    print("═" * 90)
    print(f"  {'Config':<20s}  {'QAttention':>12s}  {'FlashInfer':>12s}  {'DeFT':>12s}")
    print("  " + "─" * 62)

    for key in sorted(configs.keys()):
        B, b, d = key
        N = _tree_n_budgeted(b, d)
        cfg = f"B={B:2d} b={b:2d} d={d:2d} N={N:3d}"
        kd = configs[key]
        vals = []
        for kn in kernels:
            r = kd.get(kn)
            if r and r.status == "OK" and not math.isnan(r.l2_hit_rate_pct):
                vals.append(f"{r.l2_hit_rate_pct:11.1f}%")
            else:
                vals.append("         n/a")
        print(f"  {cfg:<20s}  {vals[0]}  {vals[1]}  {vals[2]}")

    # ── Table 3: Occupancy & Register Pressure ───────────────────────────────
    print("\n" + "═" * 90)
    print("  TABLE 3: Achieved Occupancy (%) & Registers/Thread")
    print("═" * 90)
    print(f"  {'Config':<20s}  {'QA Occ%':>8s} {'QA Regs':>8s}  {'FI Occ%':>8s} {'FI Regs':>8s}  {'DeFT Occ%':>9s} {'DeFT Regs':>9s}")
    print("  " + "─" * 82)

    for key in sorted(configs.keys()):
        B, b, d = key
        N = _tree_n_budgeted(b, d)
        cfg = f"B={B:2d} b={b:2d} d={d:2d} N={N:3d}"
        kd = configs[key]

        def _occ_regs(kn, wid=8):
            r = kd.get(kn)
            if not r or r.status != "OK":
                return "n/a".rjust(wid), "n/a".rjust(wid)
            occ = f"{r.occupancy_pct:.1f}" if not math.isnan(r.occupancy_pct) else "n/a"
            regs = f"{r.registers_per_thread:.0f}" if not math.isnan(r.registers_per_thread) else "n/a"
            return occ.rjust(wid), regs.rjust(wid)

        qa_o, qa_r = _occ_regs("qattention")
        fi_o, fi_r = _occ_regs("flashinfer")
        de_o, de_r = _occ_regs("deft", 9)
        print(f"  {cfg:<20s}  {qa_o} {qa_r}  {fi_o} {fi_r}  {de_o} {de_r}")

    # ── Table 4: Duration & SM Throughput ────────────────────────────────────
    print("\n" + "═" * 90)
    print("  TABLE 4: Kernel Duration (µs) & SM Throughput (%)")
    print("═" * 90)
    print(f"  {'Config':<20s}  {'QA µs':>10s} {'QA SM%':>7s}  {'FI µs':>10s} {'FI SM%':>7s}  {'DeFT µs':>10s} {'DeFT SM%':>8s}")
    print("  " + "─" * 82)

    for key in sorted(configs.keys()):
        B, b, d = key
        N = _tree_n_budgeted(b, d)
        cfg = f"B={B:2d} b={b:2d} d={d:2d} N={N:3d}"
        kd = configs[key]

        def _dur_sm(kn):
            r = kd.get(kn)
            if not r or r.status != "OK" or r.duration_ns == 0:
                return "n/a".rjust(10), "n/a".rjust(7)
            dur = f"{r.duration_us:.2f}".rjust(10)
            sm = f"{r.sm_throughput_pct:.1f}" if not math.isnan(r.sm_throughput_pct) else "n/a"
            return dur, sm.rjust(7)

        qa_d, qa_s = _dur_sm("qattention")
        fi_d, fi_s = _dur_sm("flashinfer")
        de_d, de_s = _dur_sm("deft")
        print(f"  {cfg:<20s}  {qa_d} {qa_s}  {fi_d} {fi_s}  {de_d} {de_s}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator main
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(args):
    """Orchestrator: loop over configs × kernels, invoke ncu, aggregate."""
    fallback_timing_only = args.timing_only
    ncu_path = None
    if not fallback_timing_only:
        ncu_path = args.ncu_path or _find_ncu()
        if ncu_path is None:
            print("=" * 70)
            print("ERROR: ncu (NVIDIA Nsight Compute) not found.")
            print()
            print("Install options:")
            print("  Ubuntu/Debian : sudo apt install nsight-compute")
            print("  CUDA Toolkit  : ncu ships with cuda-toolkit-12.x")
            print("  Standalone    : https://developer.nvidia.com/nsight-compute")
            print("=" * 70)
            sys.exit(1)

    if fallback_timing_only:
        print("ncu binary : n/a (timing-only mode)")
    else:
        print(f"ncu binary : {ncu_path}")
    script_path = os.path.abspath(__file__)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    bfs         = [int(x) for x in args.branching_factors.split(",")]
    depths      = [int(x) for x in args.depths.split(",")]
    H, D        = args.num_heads, args.head_dim

    kernels_to_profile = []
    if not args.skip_qattention:
        kernels_to_profile.append("qattention")
    if not args.skip_flashinfer:
        kernels_to_profile.append("flashinfer")
    if not args.skip_deft:
        kernels_to_profile.append("deft")

    if not kernels_to_profile:
        print("ERROR: All kernels skipped. Nothing to profile.")
        sys.exit(1)

    # Build (b, d) pairs
    use_pairwise = args.pairwise or (len(bfs) == len(depths) and not args.cartesian)
    if use_pairwise:
        if len(bfs) != len(depths):
            print(f"ERROR: Pairwise sweep requires same number of branching factors and depths (got {len(bfs)} and {len(depths)}).")
            sys.exit(1)
        b_d_pairs = list(zip(bfs, depths))
    else:
        b_d_pairs = [(b, d) for b in bfs for d in depths]

    total_runs = len(batch_sizes) * len(b_d_pairs) * len(kernels_to_profile)

    print(f"Profiling  : {total_runs} ncu invocations")
    if use_pairwise:
        print(f"             ({len(kernels_to_profile)} kernels x {len(batch_sizes)} batch x {len(b_d_pairs)} b-d pairs)")
    else:
        print(f"             ({len(kernels_to_profile)} kernels x "
              f"{len(batch_sizes)} batch x {len(bfs)} branching x {len(depths)} depths)")
    print(f"Model dims : H={H}, D={D} (LLaMA-3.1-8B)")
    print(f"sudo       : {'yes' if args.sudo else 'no'}")
    if args.dry_run:
        print(f"\n  DRY RUN — printing ncu commands only:\n")

    print()
    results: List[ProfileResult] = []
    idx = 0

    for B in batch_sizes:
        for b, d in b_d_pairs:
            N = _tree_n_budgeted(b, d)
            actual_d = _compute_actual_depth(N, b)

            for kernel in kernels_to_profile:
                idx += 1
                tag = f"[{idx:3d}/{total_runs}]"
                print(f"  {tag} {kernel:12s}  B={B:<3d}  b={b:<2d}  d={d:<2d}  N={N:<4d} ... ",
                      end="", flush=True)

                # Dry run — just show the command
                if args.dry_run:
                    cmd_parts = []
                    if args.sudo:
                        cmd_parts.append("sudo")
                    cmd_parts += [
                        ncu_path or "ncu",
                        "--profile-from-start off",
                        "--csv --print-units base",
                        f"--metrics {','.join(NCU_METRICS)}",
                        sys.executable,
                        script_path,
                        f"--run-kernel {kernel}",
                        f"--B {B} --b {b} --d {d} --H {H} --D {D}",
                    ]
                    print("\n    " + " \\\n      ".join(cmd_parts))
                    continue

                # Real invocation
                parsed = _run_ncu_for_kernel(
                    ncu_path, kernel, B, b, d, H, D,
                    script_path, use_sudo=args.sudo,
                    timing_only=fallback_timing_only,
                )

                theo_read = _theoretical_min_read_bytes(kernel, B, N, actual_d, H, D)

                if parsed is None or len(parsed) == 0:
                    if not fallback_timing_only:
                        print(" -> falling back to timing-only mode")
                        fallback_timing_only = True
                        # Retry this config with timing-only
                        parsed = _run_ncu_for_kernel(
                            ncu_path, kernel, B, b, d, H, D,
                            script_path, use_sudo=args.sudo,
                            timing_only=True,
                        )

                    if parsed is None or len(parsed) == 0:
                        results.append(ProfileResult(
                            kernel=kernel, batch_size=B,
                            branching_factor=b, depth=d,
                            num_nodes=N, actual_depth=actual_d,
                            theoretical_read_bytes=theo_read,
                            theoretical_read_gb=theo_read / 1e9,
                            status="SKIP",
                        ))
                        continue

                agg = _aggregate_kernel_metrics(parsed)

                read_amp = (agg["hbm_read_bytes"] / theo_read) if (theo_read > 0 and agg["hbm_read_bytes"] > 0) else float("nan")

                pr = ProfileResult(
                    kernel=kernel,
                    batch_size=B,
                    branching_factor=b,
                    depth=d,
                    num_nodes=N,
                    actual_depth=actual_d,
                    hbm_read_bytes=agg["hbm_read_bytes"],
                    hbm_write_bytes=agg["hbm_write_bytes"],
                    hbm_read_gb=agg["hbm_read_bytes"] / 1e9,
                    hbm_write_gb=agg["hbm_write_bytes"] / 1e9,
                    l2_hit_rate_pct=agg["l2_hit_rate_pct"],
                    l1_hit_rate_pct=agg["l1_hit_rate_pct"],
                    occupancy_pct=agg["occupancy_pct"],
                    occupancy_limit_regs_pct=agg["occupancy_limit_regs_pct"],
                    duration_ns=agg["duration_ns"],
                    duration_us=agg["duration_ns"] / 1000.0,
                    registers_per_thread=agg["registers_per_thread"],
                    sm_throughput_pct=agg["sm_throughput_pct"],
                    theoretical_read_bytes=theo_read,
                    theoretical_read_gb=theo_read / 1e9,
                    hbm_read_amplification=read_amp,
                    status="OK",
                )
                results.append(pr)

                # One-line summary
                hbm_r_str = f"{pr.hbm_read_gb:.4f}GB" if pr.hbm_read_bytes > 0 else "n/a"
                amp_str = f"{read_amp:.2f}×" if not math.isnan(read_amp) else "n/a"
                print(f"OK  "
                      f"HBM_R={hbm_r_str:<8s}  "
                      f"L2={_fv1(pr.l2_hit_rate_pct, 4)}%  "
                      f"Occ={_fv1(pr.occupancy_pct, 4)}%  "
                      f"Regs={_fv1(pr.registers_per_thread, 3)}  "
                      f"{pr.duration_us:.1f}µs  "
                      f"Amp={amp_str}")

    if args.dry_run:
        print("\n  Dry run complete. No profiling performed.")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    _write_csv(results, csv_path)

    print(f"\n{'═' * 70}")
    print(f"  Results CSV: {csv_path}")
    print(f"  {len(results)} rows ({sum(1 for r in results if r.status == 'OK')} OK, "
          f"{sum(1 for r in results if r.status != 'OK')} skipped)")
    print(f"{'═' * 70}")

    # ── Print comparison tables ───────────────────────────────────────────────
    ok_results = [r for r in results if r.status == "OK"]
    if ok_results:
        _print_comparison(results)
    else:
        print("\n  No successful profiles — cannot generate comparison tables.")
        print("  Check ncu permissions (try --sudo) and kernel availability.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NCU hardware-counter profiling: QAttention vs FlashInfer vs DeFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full profiling suite (may need sudo for hardware counters):
              python scripts/profile_kernels.py --sudo

              # Dry run — see ncu commands without executing:
              python scripts/profile_kernels.py --dry-run

              # Specific configs:
              python scripts/profile_kernels.py --sudo --batch-sizes 1,8 --depths 3,7,14

              # Skip kernels not installed:
              python scripts/profile_kernels.py --sudo --skip-deft --skip-flashinfer

            Notes:
              • ncu requires elevated permissions for hardware counter collection.
                Use --sudo, or set kernel.perf_event_paranoid / NVreg_RestrictProfilingToAdminUsers.
              • Each ncu invocation takes ~10-30s. The default grid runs ~18 invocations.
              • Results are written to results/profile_breakdown.csv.
        """),
    )

    # Worker mode (invoked by orchestrator under ncu)
    parser.add_argument("--run-kernel", type=str, default=None,
                        choices=["qattention", "flashinfer", "deft"],
                        help=argparse.SUPPRESS)  # internal — not user-facing
    parser.add_argument("--B", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--b", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--d", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--H", type=int, default=DEFAULT_NUM_HEADS, help=argparse.SUPPRESS)
    parser.add_argument("--D", type=int, default=DEFAULT_HEAD_DIM, help=argparse.SUPPRESS)

    # Orchestrator mode (user-facing)
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help=f"Comma-separated batch sizes (default: {DEFAULT_BATCH_SIZES})")
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)),
                        help=f"Comma-separated branching factors (default: {DEFAULT_BRANCHING_FACTORS})")
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)),
                        help=f"Comma-separated tree depths (default: {DEFAULT_DEPTHS})")
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS,
                        help=f"Number of attention heads (default: {DEFAULT_NUM_HEADS})")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM,
                        help=f"Head dimension (default: {DEFAULT_HEAD_DIM})")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--csv-name", default="profile_breakdown.csv",
                        help="Output CSV filename (default: profile_breakdown.csv)")
    parser.add_argument("--sudo", action="store_true",
                        help="Run ncu with sudo (needed for hw counters on most systems)")
    parser.add_argument("--skip-qattention", action="store_true",
                        help="Skip QAttention kernel")
    parser.add_argument("--skip-flashinfer", action="store_true",
                        help="Skip FlashInfer kernel")
    parser.add_argument("--skip-deft", action="store_true",
                        help="Skip DeFT kernel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ncu commands without executing")
    parser.add_argument("--ncu-path", type=str, default=None,
                        help="Path to ncu executable (overrides auto-detection)")
    parser.add_argument("--timing-only", action="store_true",
                        help="Only measure GPU execution latency and skip NCU hardware counters")
    parser.add_argument("--pairwise", action="store_true",
                        help="Sweep branching-factors and depths pairwise (default if lists have same length)")
    parser.add_argument("--cartesian", action="store_true",
                        help="Force Cartesian product sweep of branching-factors and depths")

    args = parser.parse_args()

    if args.run_kernel:
        worker_main(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
