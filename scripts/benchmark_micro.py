#!/usr/bin/env python3
"""
benchmark_micro.py — Focused Kernel Micro-benchmark for Paper Figures
======================================================================

Compares the ragged ancestor-sparse attention kernel against Eagle-3's
actual tree-attention implementation across a targeted grid designed for the
paper's main result figure:

    BS ∈ {8, 16, 32, 64}   batch sizes (typical serving up to continuous-batching)
    b  ∈ {8, 10, 12}       branching factors  (bracket Eagle-3 default top_k=10)
    d  ∈ {5 … 20}          tree depths — pushed far past the E2E sweep to locate
                           the kernel crossover and show the scaling trend clearly

Eagle-3's default is (b=10, d=7, N≈60).  The kernel dispatch overhead dominates
at small N; this sweep intentionally goes beyond realistic EAGLE configs to
establish the crossover point and the asymptotic speedup ceiling.

Both kernels compute IDENTICAL attention over the same b-ary tree topology:
    • eagle_tree:  FlashInfer single_prefill_with_kv_cache with explicit 
                 tree-ancestor bool mask (SGLang production path).
  • ragged:      Our Triton kernel with O(d+1) ancestor-walk per query,
                 ragged packed layout, no mask tensor.

Model dimensions match LLaMA-3.1-8B: H=32, D=128  (configurable via CLI).

Tree topology
-------------
By default, benchmarks use fully-balanced b-ary trees. Use --pruned-trees to
instead generate EAGLE-like pruned trees with irregular branching and varying
depths across branches, which more accurately simulates real EAGLE-3 behavior
where top-k selection creates sparse trees.

Output
------
  results/micro_benchmark.csv          — full per-config latency data
  results/micro_benchmark_pruned.csv   — (if --pruned-trees is used)
  results/micro_benchmark_pivot.csv    — pivot: depth × (batch, branching)

Usage
-----
  # Default grid with balanced trees
  python scripts/benchmark_micro.py

  # With EAGLE-like pruned trees (irregular branching)
  python scripts/benchmark_micro.py --pruned-trees

  # Custom grid
  python scripts/benchmark_micro.py --batch-sizes 1,8,16,32 --depths 1,2,3,4,5,6,7,8

  # More iterations for tighter CIs with pruned trees
  python scripts/benchmark_micro.py --pruned-trees --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import (
    ragged_attention,
    ragged_attention_with_lse,
    ragged_attention_sibling,
    fused_lse_merge,
    ragged_attention_with_parents,
)
from src.tree_mask import (
    num_tree_nodes,
    tree_attention_mask_n,
    tree_attention_mask_pruned,
    build_pruned_tree,
    sample_balanced_tree_nodes,
)


def _tree_n(
    b: int,
    d: int,
    token_cap: int = 0,
    node_count_mode: str = "exact",
) -> int:
    """N for a given (b, d) point in the sweep.

    Modes
    -----
    exact:
        Complete b-ary tree node count for the requested depth:
            N = (b^(d+1)-1)/(b-1)   (or d+1 when b=1).
        This preserves true tree-depth semantics and is the integrity-first mode.

    budgeted:
        Legacy budgeted proxy used by older micro-benchmarks:
            N = max(30, round(6*b*d/7)).
        Keeps N in an EAGLE-like token budget for stress tests at large nominal d.

    ``token_cap`` clamps N from above when > 0.
    """
    if node_count_mode == "exact":
        n = num_tree_nodes(b, d)
    elif node_count_mode == "budgeted":
        n = max(30, round(6 * b * d / 7))
    else:
        raise ValueError(f"Unknown node_count_mode={node_count_mode!r}")
    return min(n, token_cap) if token_cap > 0 else n

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
# Defaults designed to show the kernel crossover clearly.
# Eagle-3 operates at b∈{8-12}, d=7 (N~48-72).  We push d up to 20 so N
# reaches 200+ at b=12, revealing the asymptotic speedup ceiling.
DEFAULT_BATCH_SIZES       = [1]
DEFAULT_BRANCHING_FACTORS = [8, 10, 12, 14, 16, 24, 32]
DEFAULT_DEPTHS            = [5, 7, 9, 12, 14, 16, 24, 32]
DEFAULT_PREFIX_LENGTHS    = [0]  # Simulated past KV-cache lengths: tree-only + realistic verify-step lengths
DEFAULT_TOKEN_CAP         = 0     # 0 = no cap; E2E benchmark uses 120
DEFAULT_NUM_HEADS         = 32    # LLaMA-3.1-8B
DEFAULT_HEAD_DIM          = 128   # LLaMA-3.1-8B
WARMUP_ITERS              = 10
BENCH_ITERS               = 50

# Hard ceiling on total tokens per batch (B × N) to avoid OOM.
# At H=32, D=128, fp16: one Q/K/V set ≈ 3 × B×N × 32 × 128 × 2 bytes.
MAX_BATCH_TOKENS = 2_000_000
# Dense Eagle baseline needs an explicit [N, N] tree mask. Guard pathological
# cases so we skip invalid comparisons rather than silently capping N.
MAX_TREE_MASK_ELEMENTS = 100_000_000
MAX_FLATTENED_MASK_ELEMENTS = 250_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_median_ms(fn, warmup: int, iters: int) -> float:
    """Median CUDA-event latency over `iters` calls (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def _wrap_cuda_graph(fn):
    """Capture fn in a CUDA Graph for overhead-free replay, if supported.

    Both kernels MUST use the same timing methodology for a fair comparison.
    CUDA Graphs eliminate Python dispatch and inter-kernel scheduling bubbles.
    If graph capture fails (dynamic allocations, FlashInfer backend quirks, etc.)
    we fall back to direct timing so neither kernel is silently disadvantaged.

    Returns (callable, graphed: bool).
    """
    try:
        torch.cuda.synchronize()
        fn()  # pre-capture warmup — flushes lazy init and JIT compilation
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g.replay, True
    except Exception:
        return fn, False


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MicroRow:
    batch_size:       int
    prefix_length:    int       # L — simulated past KV-cache length (0 = tree-only)
    branching_factor: int
    depth:            int
    effective_depth:  int       # actual max depth represented by the benchmarked tree
    num_tree_nodes:   int
    num_heads:        int
    head_dim:         int
    eagle_tree_ms:    float     # Eagle baseline via FlashInfer (SGLang path) + tree bool mask
    ragged_ms:        float     # our ragged kernel (+ flash prefix + LSE merge when L>0)
    ragged_sibling_ms: float    # ours — sibling-coalesced single-pass variant; L=0 only (no merge wired)
    speedup:          float     # eagle_tree_ms / ragged_ms  (>1 → ragged wins)
    speedup_sibling_vs_base: float  # ragged_ms / ragged_sibling_ms  (>1 → sibling wins)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    B: int, b: int, d: int, L: int,
    H: int, D: int,
    warmup: int, iters: int,
    device: torch.device,
    token_cap: int = 0,
    use_pruned_trees: bool = False,
    node_count_mode: str = "budgeted",
) -> MicroRow:
    """Benchmark one (B, b, d, L) configuration.  Returns a MicroRow.

    Parameters
    ----------
    L : int
        Simulated prefix KV-cache length.  L=0 benchmarks tree-only attention.
        L>0 means each of the N tree tokens also attends to L prefix tokens,
        simulating real speculative-decoding verify where the past KV cache
        already contains L committed tokens.

    The Eagle baseline uses FlashInfer over the full [L+N] KV with explicit tree 
    bool mask. The ragged approach uses flash attention for the prefix and the 
    Triton kernel for the tree, merging via online-softmax (no large matrix 
    materialised).
    """
    N   = _tree_n(b, d, token_cap, node_count_mode=node_count_mode)
    nan = float("nan")

    # ── Tree mask: [N, N] bool (True = attend) ──────────────────────────────
    if use_pruned_trees:
        # Generate EAGLE-like pruned tree
        # Keep topology invariant across batch sizes so B-scaling is comparable.
        rng_seed = b * 10000 + d * 100 + N
        rng = np.random.default_rng(rng_seed)
        parent_array = build_pruned_tree(b, d, target_nodes=N, rng=rng)
        # Adjust N to actual tree size
        N = len(parent_array)
        if N == 0:
            return MicroRow(B, L, b, d, 0, 0, H, D, nan, nan, nan, nan, nan)
        tot = B * N
        if tot > MAX_BATCH_TOKENS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
        if N * N > MAX_TREE_MASK_ELEMENTS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
        if B * N * (L + N) > MAX_FLATTENED_MASK_ELEMENTS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
        mask_np = tree_attention_mask_pruned(parent_array)   # [N, N] bool
        
        depths = [0] * len(parent_array)
        for i in range(1, len(parent_array)):
            depths[i] = depths[parent_array[i]] + 1
        actual_d = max(depths) if depths else 0
        
        # Kernel requires parents[root] = root (0 instead of -1)
        p_array = [max(0, p) for p in parent_array]
        parents_tensor = torch.tensor(p_array * B, dtype=torch.int32, device=device)
    else:
        tot = B * N
        if tot > MAX_BATCH_TOKENS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
        if N * N > MAX_TREE_MASK_ELEMENTS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
        if B * N * (L + N) > MAX_FLATTENED_MASK_ELEMENTS:
            return MicroRow(B, L, b, d, 0, N, H, D, nan, nan, nan, nan, nan)
            
        mask_np = tree_attention_mask_n(b, N)   # [N, N] bool
        
        actual_d = 0
        k = N - 1
        while k > 0:
            k = (k - 1) // b
            actual_d += 1

    torch.manual_seed(B * 10000 + b * 100 + d)

    # ── Ragged-layout tensors: [B*N, H, D] — always needed ──────────────────
    Q_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    K_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    V_r = torch.randn(tot, H, D, device=device, dtype=torch.float16)
    cu  = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)

    # ── Standard-layout views [B, H, N, D] — shared by Eagle & flash prefix ─
    Q_s = Q_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    K_s = K_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    V_s = V_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

    scale = 1.0 / math.sqrt(D)

    # ── Prefix KV (simulated past cache) ────────────────────────────────────
    K_prefix = V_prefix = None
    if L > 0:
        K_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        V_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

    # Convert to float additive bias for SDPA
    # Flash / Efficient backends often require float masks or no masks at all.
    NEG_INF = torch.finfo(torch.float16).min / 2
    tree_mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float16)
    tree_bias = torch.where(tree_mask_t.bool(),
                            torch.zeros_like(tree_mask_t),
                            torch.full_like(tree_mask_t, NEG_INF))

    if L > 0:
        # Prefix columns: all-attend (0.0 bias); tree columns: ancestor bias
        prefix_zero = torch.zeros(N, L, device=device, dtype=torch.float16)
        full_mask = torch.cat([prefix_zero, tree_bias], dim=1)   # [N, L+N]
        # Expand mask to [B, 1, N, L+N] to help backend dispatch
        attn_mask4d = full_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
        del prefix_zero, full_mask
    else:
        attn_mask4d = tree_bias.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).contiguous()
    del tree_mask_t, tree_bias

    t_eagle        = nan
    t_ragged       = nan
    t_sibling      = nan
    eagle_oom      = False
    _eagle_graphed = None  # set by eagle block; checked by ragged block for symmetry

    # ── Eagle baseline via FlashInfer (SGLang production path) ────────────
    # SGLang and other high-performance frameworks use FlashInfer for tree-structured
    # attention. We use BatchPrefillWithRaggedKVCacheWrapper for efficiency.
    try:
        import flashinfer
        import flashinfer.prefill
        
        # Prepare packed KV cache if there is a prefix
        if L > 0:
            # K_prefix: [B, H, L, D] -> [B, L, H, D]
            K_p = K_prefix.permute(0, 2, 1, 3).reshape(B * L, H, D)
            V_p = V_prefix.permute(0, 2, 1, 3).reshape(B * L, H, D)
            
            # Interleave prefix and tree tokens for each sequence
            K_full_list = []
            V_full_list = []
            starts = cu[:-1].cpu().tolist()
            ends   = cu[1:].cpu().tolist()
            for i, (s, e) in enumerate(zip(starts, ends)):
                K_full_list.append(K_p[i*L : (i+1)*L])
                K_full_list.append(K_r[s:e])
                V_full_list.append(V_p[i*L : (i+1)*L])
                V_full_list.append(V_r[s:e])
            
            K_full = torch.cat(K_full_list, dim=0).contiguous()
            V_full = torch.cat(V_full_list, dim=0).contiguous()
            kv_lens = torch.tensor([L + N] * B, dtype=torch.int32, device=device)
        else:
            K_full = K_r.contiguous()
            V_full = V_r.contiguous()
            kv_lens = torch.tensor([N] * B, dtype=torch.int32, device=device)
            
        kv_indptr = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), 
                               torch.cumsum(kv_lens, dim=0, dtype=torch.int32)])
        
        # Query is always just the tree nodes [B*N, H, D]
        qo_indptr = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)
        
        # Flattened masks: each sequence has [N, L+N] mask
        m_tree = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
        if L > 0:
            m_pre = torch.ones(N, L, device=device, dtype=torch.bool)
            m_full = torch.cat([m_pre, m_tree], dim=1)
            flattened_mask = m_full.repeat(B, 1).flatten()
        else:
            flattened_mask = m_tree.repeat(B, 1).flatten()

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )
        
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            H, H, D,
            custom_mask=flattened_mask,
            causal=False,
            sm_scale=scale,
            q_data_type=Q_r.dtype,
        )

        def eagle_fn():
            return wrapper.run(Q_r, K_full, V_full)

        # Apply CUDA Graph (same methodology as ragged — fair apples-to-apples)
        eagle_g_fn, _eagle_graphed = _wrap_cuda_graph(eagle_fn)
        t_eagle = _cuda_median_ms(eagle_g_fn, warmup, iters)
    except (ImportError, RuntimeError) as exc:
        # Fallback to NaN if FlashInfer is not installed or fails
        t_eagle = nan
        if "out of memory" in str(exc).lower():
            eagle_oom = True
            print(f"    [OOM] FlashInfer skipped for B={B} b={b} d={d} L={L} N={N}")
        else:
            print(f"    [SKIP] FlashInfer baseline unavailable: {exc}")
    finally:
        torch.cuda.empty_cache()

    # ── Ragged kernel ─────────────────────────────────────────────────────────
    try:
        if L > 0:
            # Flash prefix + ragged tree + online-softmax merge
            def ragged_fn():
                # Part 1: flash attention over prefix KV
                out_pre, lse_pre, *_ = (
                    torch.ops.aten._scaled_dot_product_flash_attention(
                        Q_s.contiguous(), K_prefix, V_prefix,
                        dropout_p=0.0, is_causal=False, scale=scale,
                        return_debug_mask=False,
                    )
                )
                # Part 2: ragged tree kernel (returns LSE for merge)
                if use_pruned_trees:
                    o_tr, lse_tr = ragged_attention_with_parents(
                        Q_r, K_r, V_r, cu, parents_tensor, actual_d, max_seqlen=N)
                else:
                    o_tr, lse_tr = ragged_attention_with_lse(
                        Q_r, K_r, V_r, cu, b, actual_d, max_seqlen=N)
                o_tree = o_tr.view(B, N, H, D).permute(0, 2, 1, 3)
                lse_tree = lse_tr.view(B, N, H).permute(0, 2, 1)   # [B,H,N]
                # Part 3: online-softmax merge
                return fused_lse_merge(lse_pre.float(), lse_tree.float(), out_pre, o_tree)
        else:
            def ragged_fn():
                if use_pruned_trees:
                    return ragged_attention_with_parents(
                        Q_r, K_r, V_r, cu, parents_tensor, actual_d, max_seqlen=N)[0]
                else:
                    return ragged_attention(Q_r, K_r, V_r, cu, b, actual_d, max_seqlen=N)

        # Apply CUDA Graph (same methodology as eagle baseline — fair comparison)
        ragged_g_fn, _ragged_graphed = _wrap_cuda_graph(ragged_fn)
        if _ragged_graphed != _eagle_graphed:
            print(
                f"    [WARN] Timing asymmetry: eagle_graph={_eagle_graphed} "
                f"ragged_graph={_ragged_graphed} — rerun with investigation"
            )
        t_ragged = _cuda_median_ms(ragged_g_fn, warmup, iters)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if eagle_oom:
            # Both eagle and ragged OOMed — config is too large for both
            print(f"    [OOM] Both kernels OOMed for B={B} b={b} d={d} L={L} N={N} — skipping")
        else:
            # Only ragged OOMed (eagle succeeded) — this is worth noting
            print(f"    [OOM] Ragged OOMed but Eagle succeeded for B={B} b={b} d={d} L={L} N={N}")
    finally:
        torch.cuda.empty_cache()

    # ── Sibling-coalesced kernel ─────────────────────────────────────────────
    # Numerically equivalent to ragged_attention; structurally distinct
    # (single-pass gather + softmax).  Timed only at L=0 because the LSE-merge
    # path with prefix isn't wired to the sibling variant in this script —
    # the comparison of interest is sibling vs base ragged for tree-only work.
    if L == 0 and not use_pruned_trees:
        try:
            def sibling_fn():
                return ragged_attention_sibling(
                    Q_r, K_r, V_r, cu, b, actual_d, max_seqlen=N
                )
            sibling_g_fn, _ = _wrap_cuda_graph(sibling_fn)
            t_sibling = _cuda_median_ms(sibling_g_fn, warmup, iters)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower():
                print(f"    [SKIP] Sibling kernel failed: {exc}")
            else:
                print(f"    [OOM] Sibling kernel OOMed for B={B} b={b} d={d} L={L} N={N}")
        finally:
            torch.cuda.empty_cache()

    speedup = nan
    if not (math.isnan(t_eagle) or math.isnan(t_ragged) or t_ragged <= 0):
        speedup = t_eagle / t_ragged

    speedup_sib_vs_base = nan
    if not (math.isnan(t_ragged) or math.isnan(t_sibling) or t_sibling <= 0):
        speedup_sib_vs_base = t_ragged / t_sibling

    # Cleanup
    del Q_r, K_r, V_r, cu, Q_s, K_s, V_s, attn_mask4d
    if K_prefix is not None:
        del K_prefix, V_prefix
    torch.cuda.empty_cache()

    return MicroRow(B, L, b, d, actual_d, N, H, D,
                    round(t_eagle,    4) if not math.isnan(t_eagle)    else nan,
                    round(t_ragged,   4) if not math.isnan(t_ragged)   else nan,
                    round(t_sibling,  4) if not math.isnan(t_sibling)  else nan,
                    round(speedup,    4) if not math.isnan(speedup)    else nan,
                    round(speedup_sib_vs_base, 4)
                        if not math.isnan(speedup_sib_vs_base) else nan)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: Eagle SDPA-FLASH tree-attn vs ragged kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)))
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--prefix-lengths",    default=",".join(map(str, DEFAULT_PREFIX_LENGTHS)),
                        help="Simulated past KV-cache lengths. 0 = tree-only; "
                             "1024/4096 = realistic verify-step context lengths.")
    parser.add_argument("--token-cap",  type=int, default=DEFAULT_TOKEN_CAP,
                        help="Cap N from above (0 = no cap, default). "
                             "Set to 120 to match Eagle-3 E2E configs.")
    parser.add_argument(
        "--node-count-mode",
        choices=["exact", "budgeted"],
        default="budgeted",
        help=(
            "How N is derived from (b,d): "
            "'exact' = complete b-ary tree node count (integrity-first); "
            "'budgeted' = EAGLE-like proxy N≈6*b*d/7 with floor 30 (production-style)."
        ),
    )
    parser.add_argument("--balanced-trees", action="store_true",
                        help="Use complete balanced trees instead of EAGLE-like pruned trees.")
    parser.add_argument("--num-heads",  type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim",   type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--warmup",     type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",      type=int, default=BENCH_ITERS)
    parser.add_argument("--out-dir",    default="results")
    parser.add_argument("--csv-name",   default="micro_benchmark.csv")
    args = parser.parse_args()

    args.pruned_trees = not args.balanced_trees
    if args.pruned_trees and args.node_count_mode == "exact" and args.token_cap == 0:
        print(
            "[ERROR] --node-count-mode exact with pruned trees and no --token-cap "
            "explodes N and is not a production-like EAGLE setup.\n"
            "Use default budgeted mode, or specify --token-cap, or add --balanced-trees."
        )
        sys.exit(2)

    if not torch.cuda.is_available():
        print("[ERROR]  CUDA required.")
        sys.exit(1)

    batch_sizes    = [int(x) for x in args.batch_sizes.split(",")]
    bfs            = [int(x) for x in args.branching_factors.split(",")]
    depths         = [int(x) for x in args.depths.split(",")]
    prefix_lengths = [int(x) for x in args.prefix_lengths.split(",")]
    H, D           = args.num_heads, args.head_dim
    device         = torch.device("cuda:0")
    token_cap      = args.token_cap

    # Adjust CSV name if using pruned trees
    csv_name = args.csv_name
    if args.pruned_trees:
        base, ext = os.path.splitext(csv_name)
        csv_name = f"{base}_pruned{ext}"

    p = torch.cuda.get_device_properties(0)
    print("=" * 72)
    print("  sd-ragged  ·  Micro Benchmark  ·  Eagle tree-attn vs Ragged")
    print("=" * 72)
    print(f"  GPU:      {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
    print(f"  Dims:     H={H}, D={D}  (fp16)")
    print(f"  Grid:     BS∈{batch_sizes}  b∈{bfs}  d∈{depths}"
          + (f"  N_cap={token_cap}" if token_cap else "  N_cap=none"))
    print(f"  Prefix:   L∈{prefix_lengths}  "
          "(0 = tree-only, >0 = flash prefix + ragged tree + LSE merge)")
    print(f"  N range:  "
          + "  ".join(f"b={b},d={depths[0]}→{_tree_n(b,depths[0],token_cap,args.node_count_mode)}"
                       f"/d={depths[-1]}→{_tree_n(b,depths[-1],token_cap,args.node_count_mode)}"
                       for b in bfs))
    print(f"  Timing:   {args.warmup} warmup + {args.iters} iters (median)")
    tree_mode = "EAGLE-like pruned (irregular branching)" if args.pruned_trees else "fully balanced"
    print(f"  Trees:    {tree_mode}")
    print(f"  N mode:   {args.node_count_mode}")
    print("=" * 72)

    configs = [(B, b, d, L)
               for L in prefix_lengths
               for B in batch_sizes
               for b in bfs
               for d in depths]
    rows: List[MicroRow] = []

    for ci, (B, b, d, L) in enumerate(configs):
        N = _tree_n(b, d, token_cap, args.node_count_mode)
        row = benchmark_one(B, b, d, L, H, D,
                            args.warmup, args.iters, device, token_cap,
                            use_pruned_trees=args.pruned_trees,
                            node_count_mode=args.node_count_mode)
        rows.append(row)

        def _f(v):
            return f"{v:.3f}" if not math.isnan(v) else " n/a "

        spd_s = f"{row.speedup:.2f}×" if not math.isnan(row.speedup) else "  n/a"
        d_eff_s = f"{row.effective_depth:>2d}" if row.effective_depth >= 0 else "na"
        print(f"  [{ci+1:3d}/{len(configs)}]  "
              f"B={B:3d} b={b} d={d:>2d} d_eff={d_eff_s} L={L:>4d} N={N:3d}  "
              f"eagle={_f(row.eagle_tree_ms):>8s} ms  "
              f"ragged={_f(row.ragged_ms):>8s} ms  "
              f"speedup={spd_s:>7s}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    import datetime
    os.makedirs(args.out_dir, exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(csv_name)
    csv_path = os.path.join(args.out_dir, f"{base}_{_ts}{ext}")
    csv_latest = os.path.join(args.out_dir, csv_name)
    
    fieldnames = list(asdict(rows[0]).keys())
    for _p in (csv_path, csv_latest):
        with open(_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {csv_latest} (latest)")

    # ── Pivot: speedup table ─────────────────────────────────────────────────
    # Group by (B, b, L) and show speedup at each depth.
    for L in prefix_lengths:
        print()
        label = "tree-only" if L == 0 else f"prefix L={L}"
        print(f"  Speedup table [{label}]  "
              "(eagle / ragged — >1 means ragged is faster)")
        print("  " + "─" * 68)

        for B in batch_sizes:
            for b in bfs:
                subset = [r for r in rows
                          if r.batch_size == B and r.branching_factor == b
                          and r.prefix_length == L]
                if not subset:
                    continue
                d_vals = sorted(set(r.depth for r in subset))
                hdr = (f"  B={B:>3d} b={b:>2d} │ "
                       + "  ".join(f"d={d:>2d}" for d in d_vals))
                print(hdr)
                vals = []
                for d in d_vals:
                    match = [r for r in subset if r.depth == d]
                    if match and not math.isnan(match[0].speedup):
                        vals.append(f"{match[0].speedup:5.2f}×")
                    else:
                        vals.append("  n/a ")
                print(f"{'':>17s} │ " + "  ".join(f"{v:>5s}" for v in vals))
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r.speedup)]
    if valid:
        wins  = [r for r in valid if r.speedup > 1.0]
        best  = max(valid, key=lambda r: r.speedup)
        worst = min(valid, key=lambda r: r.speedup)
        med   = float(np.median([r.speedup for r in valid]))

        print("=" * 72)
        print("  MICRO BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"  Configs tested:   {len(configs)}")
        print(f"  Valid results:    {len(valid)}")
        print(f"  Ragged wins:      {len(wins)} / {len(valid)}  "
              f"({100*len(wins)/len(valid):.0f}%)")
        print(f"  Median speedup:   {med:.2f}×")
        print(f"  Best speedup:     {best.speedup:.2f}×  "
              f"(B={best.batch_size}, b={best.branching_factor}, "
              f"d={best.depth}, L={best.prefix_length})")
        print(f"  Worst speedup:    {worst.speedup:.2f}×  "
              f"(B={worst.batch_size}, b={worst.branching_factor}, "
              f"d={worst.depth}, L={worst.prefix_length})")

        # Per-prefix-length breakdown
        for L in prefix_lengths:
            sub = [r for r in valid if r.prefix_length == L]
            if not sub:
                continue
            label = "tree-only" if L == 0 else f"L={L}"
            sub_wins = [r for r in sub if r.speedup > 1.0]
            sub_med  = float(np.median([r.speedup for r in sub]))
            sub_best = max(sub, key=lambda r: r.speedup)
            print(f"\n  [{label}]  wins {len(sub_wins)}/{len(sub)}  "
                  f"median {sub_med:.2f}×  "
                  f"best {sub_best.speedup:.2f}× "
                  f"(B={sub_best.batch_size}, b={sub_best.branching_factor}, "
                  f"d={sub_best.depth})")

        # Identify crossover per (B, b, L) group
        print()
        print("  Crossover analysis  (min depth where ragged wins, per B×b×L):")
        for L in prefix_lengths:
            label = "tree-only" if L == 0 else f"L={L}"
            print(f"    ── {label} ──")
            for B in batch_sizes:
                for b in bfs:
                    group = [r for r in valid
                             if r.batch_size == B and r.branching_factor == b
                             and r.prefix_length == L]
                    group.sort(key=lambda r: r.depth)
                    winning_depths = [r.depth for r in group if r.speedup > 1.0]
                    if winning_depths:
                        print(f"      B={B:>3d}, b={b:>2d}:  wins at d ≥ "
                              f"{min(winning_depths)}  "
                              f"(max {max(r.speedup for r in group if r.speedup > 1.0):.2f}×)")
                    else:
                        print(f"      B={B:>3d}, b={b:>2d}:  ragged does not win "
                              "at any tested depth")
        print("=" * 72)
    print()


if __name__ == "__main__":
    main()
