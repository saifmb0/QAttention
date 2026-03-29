"""
profile_kernel.py
=================
Data-driven profiling of the ragged attention kernel against SDPA.

For each configuration this script measures:
  - Kernel latency (triton.testing.do_bench, 200 iterations)
  - Achieved HBM bandwidth (GB/s)
  - Achieved TFLOPS
  - Roofline ceiling (memory-bound vs compute-bound)
  - CTA count vs SM count → saturation ratio

Roofline constants for T4 (SM75):
  HBM bandwidth : 300 GB/s
  FP16 Tensor   :  65 TFLOPS

Usage
-----
  python scripts/profile_kernel.py
  python scripts/profile_kernel.py --csv results/profile.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

try:
    import triton
    import triton.testing
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from src.tree_mask  import tree_attention_mask, num_tree_nodes
from src.ragged_attn import pack_inputs, ragged_attention

# ---------------------------------------------------------------------------
# T4 hardware constants
# ---------------------------------------------------------------------------
T4_HBM_BW_GBs  = 300.0   # GB/s  (spec: 320, de-rate 6 % for ECC overhead)
T4_TFLOPS_FP16  = 65.0    # TFLOPS dense tensor-core FP16
T4_NUM_SMS      = 40
BLOCK_M_DEFAULT = 16      # smallest autotune block — used for CTA count estimate


# ---------------------------------------------------------------------------
# Analytic cost models
# ---------------------------------------------------------------------------

def _hbm_bytes(seq_lens: list[int], H: int, D: int) -> float:
    """
    Lower-bound HBM traffic for a correct Flash-Attention kernel.

    Each sequence of length L loads:
        Q  : L * D * 2 bytes
        K  : L * D * 2  (read once per Q tile)
        V  : L * D * 2
        O  : L * D * 2  (write)
    Total per sequence = 8 * L * D bytes.
    """
    return sum(8.0 * L * D * H for L in seq_lens)


def _flops(seq_lens: list[int], H: int, D: int) -> float:
    """
    Total FLOPs: QK^T + row-softmax + PV  (both fused softmax + matmuls).
    Each pair (L_q × L_k): 2 MACs → 4 FLOPs.
    Two matmuls: QK^T and PV → 8 FLOPs total per (q, k) pair.
    """
    return sum(8.0 * L * L * D * H for L in seq_lens)


def _roofline_ms(hbm_bytes: float, flops: float) -> tuple[float, float, str]:
    """
    Return (roofline_ms, arithmetic_intensity, bottleneck_label).
    Arithmetic intensity I = FLOPs / bytes.
    Ridge point at I* = T4_TFLOPS / T4_HBM_BW.
    I < I*  → HBM-bound,   time = bytes / BW
    I >= I* → compute-bound, time = FLOPs / TFLOPS
    """
    I      = flops / max(hbm_bytes, 1.0)
    I_star = T4_TFLOPS_FP16 * 1e12 / (T4_HBM_BW_GBs * 1e9)
    if I < I_star:
        roof_ms = hbm_bytes / (T4_HBM_BW_GBs * 1e9) * 1e3
        label   = f"HBM-bound  I={I:.1f} FLOPs/B < I*={I_star:.0f}"
    else:
        roof_ms = flops / (T4_TFLOPS_FP16 * 1e12) * 1e3
        label   = f"Compute-bound  I={I:.1f} FLOPs/B >= I*={I_star:.0f}"
    return roof_ms, I, label


# ---------------------------------------------------------------------------
# Single profiling point
# ---------------------------------------------------------------------------

def profile_one(
    batch_size: int,
    branching_factor: int,
    depth: int,
    num_heads: int = 8,
    head_dim: int  = 64,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda")

    torch.manual_seed(0)
    N        = num_tree_nodes(branching_factor, depth)
    seq_lens = [N] * batch_size
    H, D     = num_heads, head_dim

    qs = [torch.randn(N, H, D, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    ks = [torch.randn(N, H, D, device=device, dtype=torch.float16)
          for _ in range(batch_size)]
    vs = [torch.randn(N, H, D, device=device, dtype=torch.float16)
          for _ in range(batch_size)]

    Q, K, V, cu_sl = pack_inputs(qs, ks, vs)

    fn = lambda: ragged_attention(Q, K, V, cu_sl,        # noqa: E731
                                  branching_factor=branching_factor,
                                  max_depth=depth)

    if not HAS_TRITON:
        # fallback: use CUDA events
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        ragged_ms = (time.perf_counter() - t0) / 50 * 1e3
    else:
        # triton.testing.do_bench returns median ms
        ragged_ms = triton.testing.do_bench(fn, warmup=25, rep=200,
                                            fast_flush=True)

    hbm    = _hbm_bytes(seq_lens, H, D)
    flops  = _flops(seq_lens, H, D)
    roof_ms, intensity, bottleneck = _roofline_ms(hbm, flops)

    achieved_tflops  = flops  / (ragged_ms * 1e-3) / 1e12
    achieved_bw_gbs  = hbm   / (ragged_ms * 1e-3) / 1e9
    util_pct         = (ragged_ms / roof_ms) * 100.0 if roof_ms > 0 else float("nan")

    # CTA count estimate (uses minimum block size; actual depends on autotune)
    q_tiles  = sum(math.ceil(L / BLOCK_M_DEFAULT) for L in seq_lens)
    cta_count = q_tiles * batch_size * H
    saturation = cta_count / T4_NUM_SMS

    return {
        "B":               batch_size,
        "b":               branching_factor,
        "d":               depth,
        "N":               N,
        "ragged_ms":       round(ragged_ms,       4),
        "roofline_ms":     round(roof_ms,          4),
        "util_pct":        round(util_pct,         1),
        "achieved_tflops": round(achieved_tflops,  3),
        "achieved_bw_gbs": round(achieved_bw_gbs,  1),
        "intensity":       round(intensity,         2),
        "cta_count":       cta_count,
        "sm_saturation":   round(saturation,        2),
        "bottleneck":      bottleneck,
    }


# ---------------------------------------------------------------------------
# Sweep and report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/profile.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}  SM{props.major}{props.minor}  "
          f"{props.total_memory // (1<<30)} GB")
    print()

    BATCH_SIZES       = [1, 4, 16, 32]
    BRANCHING_FACTORS = [2, 3, 4]
    DEPTHS            = [1, 3, 5]

    rows = []
    for B in BATCH_SIZES:
        for b in BRANCHING_FACTORS:
            for d in DEPTHS:
                row = profile_one(B, b, d, device=device)
                rows.append(row)
                print(
                    f"  B={B:2d} b={b} d={d}  N={row['N']:4d}  "
                    f"ragged={row['ragged_ms']:.3f}ms  "
                    f"roof={row['roofline_ms']:.3f}ms  "
                    f"util={row['util_pct']:.0f}%  "
                    f"I={row['intensity']:.1f} FLOPs/B  "
                    f"CTAs/SM={row['sm_saturation']:.1f}×  "
                    f"{row['bottleneck']}"
                )

    # Summary
    print("\n── Roofline efficiency by regime ──")
    print(f"  {'Config':<18}  {'util%':>6}  {'achieved_TFLOPS':>16}  {'CTAs/SM':>7}")
    for r in rows:
        label = f"B={r['B']} b={r['b']} d={r['d']}"
        print(f"  {label:<18}  {r['util_pct']:>6.1f}  "
              f"{r['achieved_tflops']:>16.3f}  {r['sm_saturation']:>7.1f}")

    # Save CSV
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    import csv
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.csv}")

    # Key takeaways
    hbm_bound = [r for r in rows if "HBM-bound" in r["bottleneck"]]
    avg_util  = sum(r["util_pct"] for r in hbm_bound) / max(len(hbm_bound), 1)
    starved   = [r for r in rows if r["sm_saturation"] < 1.0]
    print("\n── Bottleneck summary ──")
    print(f"  HBM-bound configs:          {len(hbm_bound)}/{len(rows)}")
    print(f"  Average roofline util (HBM): {avg_util:.0f}%")
    print(f"  CTA-starved configs (<1 CTA/SM): {len(starved)}")
    if avg_util < 30:
        print("\n  ACTION: HBM utilisation is low."
              " Increase BLOCK_M×BLOCK_N to raise arithmetic intensity.")
        print(f"  Current ridge point I* = {T4_TFLOPS_FP16*1e3/T4_HBM_BW_GBs:.0f} FLOPs/B.")
        print("  Target BLOCK_M=BLOCK_N=64 → I = 32 FLOPs/B → ~9 TFLOPS.")


if __name__ == "__main__":
    main()
