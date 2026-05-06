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

# DeFT (arXiv:2404.00242, ICLR'25) — optional, cloned by setup_blackwell.sh
_DEFT_KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "FastTree", "kernel_bench",
)
HAS_DEFT = os.path.isfile(os.path.join(_DEFT_KERNEL_DIR, "DeFT.py"))


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
DEFAULT_BATCH_SIZES       = [1, 2, 4, 8, 16]   # N_branches (independent completions from shared prefix)
DEFAULT_BRANCHING_FACTORS = [4, 8, 12, 16, 20, 24, 28]              # Eagle-3 default; fixed for Phase-1 sweep
DEFAULT_DEPTHS            = [3, 5, 7, 10, 14, 20, 28]               # Eagle-3 default; fixed for Phase-1 sweep
DEFAULT_PREFIX_LENGTHS    = [0, 1024, 4096, 8192, 16384, 32768]  # context_len — shared prefix lengths
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
    batch_size:              int    # n_sequences — number of independent parallel trees (NOT branching factor b)
    prefix_length:           int    # L — shared KV-cache prefix length (context_len)
    branching_factor:        int
    depth:                   int
    effective_depth:         int    # actual max depth in tree
    num_tree_nodes:          int
    num_heads:               int
    head_dim:                int
    flashinfer_tree_ms:      float  # FlashInfer + flattened tree mask (monolithic, SGLang path)
    flashinfer_cascade_ms:   float  # FlashInfer cascade: shared-prefix pass + tree pass + LSE merge
    deft_ms:                 float  # DeFT Triton kernel (arXiv:2404.00242); NaN if not installed
    ragged_ms:               float  # ours: flash prefix + ragged tree + LSE merge
    ragged_sibling_ms:       float  # ours — sibling-coalesced variant; L=0 only
    speedup_vs_tree:         float  # flashinfer_tree_ms / ragged_ms
    speedup_vs_cascade:      float  # flashinfer_cascade_ms / ragged_ms
    speedup_vs_deft:         float  # deft_ms / ragged_ms
    speedup_sibling_vs_base: float  # ragged_ms / ragged_sibling_ms  (>1 → sibling wins)
    bw_gb_s:                 float  # ragged kernel tree-portion bandwidth (GB/s)
    bw_util_pct:             float  # bw_gb_s / peak_bw_gb_s × 100


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
    skip_cascade: bool = False,
    skip_deft: bool = False,
    peak_bw_gb_s: float = 0.0,
) -> MicroRow:
    """Benchmark one (B, b, d, L) configuration.  Returns a MicroRow.

    Parameters
    ----------
    L : int
        Simulated prefix KV-cache length.  L=0 benchmarks tree-only attention.
        L>0 means each of the N tree tokens also attends to L prefix tokens,
        simulating real speculative-decoding verify where the past KV cache
        already contains L committed tokens.

    Baselines:
      flashinfer_tree: FlashInfer over full [L+N] KV with flattened bool mask.
      flashinfer_cascade: FlashInfer shared-prefix pass (1 copy of L prefix KV)
        + FlashInfer tree pass + LSE merge.  Reflects the cascade saving.
      deft: DeFT Triton kernel (tree-only, no prefix support).
    Ours: flash prefix (shared) + ragged tree kernel + LSE merge.
    """
    N   = _tree_n(b, d, token_cap, node_count_mode=node_count_mode)
    nan = float("nan")

    def _nan_row(N_val=None):
        n = N_val if N_val is not None else N
        return MicroRow(B, L, b, d, 0, n, H, D,
                        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)

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
            return _nan_row(0)
        tot = B * N
        if tot > MAX_BATCH_TOKENS:
            return _nan_row()
        if N * N > MAX_TREE_MASK_ELEMENTS:
            return _nan_row()
        if B * N * (L + N) > MAX_FLATTENED_MASK_ELEMENTS:
            return _nan_row()
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
            return _nan_row()
        if N * N > MAX_TREE_MASK_ELEMENTS:
            return _nan_row()
        if B * N * (L + N) > MAX_FLATTENED_MASK_ELEMENTS:
            return _nan_row()
            
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
    # K_prefix / V_prefix: [B, H, L, D] — one copy per branch (for flash prefix + ragged)
    # K_shared / V_shared: [H, L, D]   — single shared copy (for cascade baseline)
    K_prefix = V_prefix = None
    K_shared = V_shared = None
    if L > 0:
        K_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        V_prefix = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
        # Shared prefix: use branch-0 as the common prefix (broadcast to all B)
        K_shared = K_prefix[0].contiguous()   # [H, L, D]
        V_shared = V_prefix[0].contiguous()
        # NHD layout required by FlashInfer single_prefill
        K_sh_nhd = K_shared.permute(1, 0, 2).contiguous()  # [L, H, D]
        V_sh_nhd = V_shared.permute(1, 0, 2).contiguous()

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

    t_fi_tree     = nan   # FlashInfer monolithic (prefix+tree concatenated, flattened mask)
    t_fi_cascade  = nan   # FlashInfer cascade (shared prefix pass + tree pass + merge)
    t_deft        = nan   # DeFT Triton kernel
    t_ragged      = nan   # ours
    t_sibling     = nan   # sibling-coalesced variant (L=0 only)
    fi_tree_oom   = False
    _fi_graphed   = None  # set by FlashInfer tree block; checked for timing symmetry

    # ── FlashInfer tree baseline (SGLang / monolithic path) ──────────────────
    # Uses BatchPrefillWithRaggedKVCacheWrapper with flattened custom_mask covering
    # the full [L+N] KV per sequence — the production path in SGLang.
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

        def fi_tree_fn():
            return wrapper.run(Q_r, K_full, V_full)

        fi_tree_g_fn, _fi_graphed = _wrap_cuda_graph(fi_tree_fn)
        t_fi_tree = _cuda_median_ms(fi_tree_g_fn, warmup, iters)
    except (ImportError, RuntimeError) as exc:
        t_fi_tree = nan
        if "out of memory" in str(exc).lower():
            fi_tree_oom = True
            print(f"    [OOM] FlashInfer tree skipped for B={B} b={b} d={d} L={L} N={N}")
        else:
            print(f"    [SKIP] FlashInfer tree baseline unavailable: {exc}")
    finally:
        torch.cuda.empty_cache()

    # ── FlashInfer cascade baseline ───────────────────────────────────────────
    # Separates the shared prefix from the per-branch tree attention:
    #   1. single_prefill_with_kv_cache_return_lse: all B*N queries → shared prefix KV
    #   2. BatchPrefillWithRaggedKVCacheWrapper: tree-only KV with tree mask
    #   3. flashinfer.merge_state: online-softmax merge of prefix + tree outputs
    # This reflects the cascade saving: prefix KV is loaded ONCE for all B branches.
    if not skip_cascade and L > 0:
        try:
            import flashinfer
            import flashinfer.prefill

            # Tree-only wrapper (same plan as fi_tree but KV = tree tokens only)
            kv_indptr_tree = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)
            qo_indptr_c    = torch.arange(0, B * N + 1, N, dtype=torch.int32, device=device)
            m_tree_c = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
            flat_tree_mask = m_tree_c.repeat(B, 1).flatten()

            ws_c = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper_c = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(ws_c, kv_layout="NHD")
            wrapper_c.plan(
                qo_indptr_c, kv_indptr_tree, H, H, D,
                custom_mask=flat_tree_mask, causal=False,
                sm_scale=scale, q_data_type=Q_r.dtype,
            )

            def cascade_fn():
                # Step 1: all B*N queries attend to 1 shared prefix (L tokens)
                o_pre, lse_pre = flashinfer.single_prefill_with_kv_cache_return_lse(
                    Q_r, K_sh_nhd, V_sh_nhd,
                    causal=False, sm_scale=scale,
                )
                # lse_pre: [B*N, H]; o_pre: [B*N, H, D]
                # Step 2: tree-only attention per branch
                o_tree, lse_tree = wrapper_c.run_return_lse(Q_r, K_r, V_r)
                # lse_tree: [B*N, H]; o_tree: [B*N, H, D]
                # Step 3: merge
                flashinfer.merge_state(o_pre, lse_pre, o_tree, lse_tree)

            casc_g_fn, _ = _wrap_cuda_graph(cascade_fn)
            t_fi_cascade = _cuda_median_ms(casc_g_fn, warmup, iters)
        except Exception as exc:
            t_fi_cascade = nan
            if "out of memory" not in str(exc).lower():
                print(f"    [SKIP] FlashInfer cascade unavailable: {exc}")
        finally:
            torch.cuda.empty_cache()

    # ── DeFT baseline (arXiv:2404.00242) ─────────────────────────────────────
    # Tree-only Triton kernel; skipped if third_party/FastTree not cloned or L>0.
    if not skip_deft and HAS_DEFT and L == 0 and not use_pruned_trees:
        try:
            import sys as _sys
            if _DEFT_KERNEL_DIR not in _sys.path:
                _sys.path.insert(0, _DEFT_KERNEL_DIR)
            from kv_tree_simple import KVTreeNode   # type: ignore
            from DeFT import DeFT_preparation, DeFT_decode  # type: ignore
            import DeFT as _deft_mod                # type: ignore

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

            # ancestor index table
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

            K_flat_b0 = K_r[0:N].contiguous()
            V_flat_b0 = V_r[0:N].contiguous()
            K_cache_b0 = K_flat_b0[idx_d.view(-1)].view(N, max_pl, H, D).contiguous()

            _deft_mod.cur_length = 0
            DeFT_aux = DeFT_preparation(tree_info_d, K_cache_b0, 128, 64, H, D)
            sm_d = 1.0 / math.sqrt(D)
            Out_d = torch.empty(N, H, D, device=device, dtype=torch.float16)

            Qs_d, Kc_d, Vc_d = [], [], []
            for bi in range(B):
                Kb = K_r[bi * N:(bi + 1) * N].contiguous()
                Vb = V_r[bi * N:(bi + 1) * N].contiguous()
                Qs_d.append(Q_r[bi * N:(bi + 1) * N].contiguous())
                Kc_d.append(Kb[idx_d.view(-1)].view(N, max_pl, H, D).contiguous().view(-1, H, D))
                Vc_d.append(Vb[idx_d.view(-1)].view(N, max_pl, H, D).contiguous().view(-1, H, D))

            def deft_fn():
                for bi in range(B):
                    DeFT_decode(Qs_d[bi], Kc_d[bi], Vc_d[bi], Out_d, *DeFT_aux,
                                Q_TILE_SIZE=16, KV_TILE_SIZE=32,
                                sm_scale=sm_d, mask_len=64)

            deft_g_fn, _ = _wrap_cuda_graph(deft_fn)
            t_deft = _cuda_median_ms(deft_g_fn, warmup, iters)
        except Exception as exc:
            t_deft = nan
            print(f"    [SKIP] DeFT: {exc}")
        finally:
            torch.cuda.empty_cache()

    # ── Ragged kernel ─────────────────────────────────────────────────────────
    try:
        if L > 0:
            # Shared prefix (FlashInfer single_prefill, reads prefix KV once for all B
            # branches) + ragged tree + fused LSE merge.
            # Prefix step is now identical to cascade — only tree step differs.
            import flashinfer as _fi
            def ragged_fn():
                # Part 1: shared prefix attention — Q_r[B*N,H,D] × K_sh_nhd[L,H,D]
                # Reads prefix KV ONCE for all B branches (cascade decomposition).
                o_pre_flat, lse_pre_flat = _fi.single_prefill_with_kv_cache_return_lse(
                    Q_r, K_sh_nhd, V_sh_nhd,
                    causal=False, sm_scale=scale,
                )
                # Reshape flat [B*N,...] → [B,H,N,...] for fused_lse_merge
                out_pre  = o_pre_flat.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
                lse_pre  = lse_pre_flat.view(B, N, H).permute(0, 2, 1).contiguous()
                # Part 2: ragged tree kernel O(N·d) — where we beat cascade's O(N²)
                if use_pruned_trees:
                    o_tr, lse_tr = ragged_attention_with_parents(
                        Q_r, K_r, V_r, cu, parents_tensor, actual_d, max_seqlen=N)
                else:
                    o_tr, lse_tr = ragged_attention_with_lse(
                        Q_r, K_r, V_r, cu, b, actual_d, max_seqlen=N)
                o_tree   = o_tr.view(B, N, H, D).permute(0, 2, 1, 3)
                lse_tree = lse_tr.view(B, N, H).permute(0, 2, 1)
                # Part 3: fused LSE merge
                return fused_lse_merge(lse_pre.float(), lse_tree.float(), out_pre, o_tree)
        else:
            def ragged_fn():
                if use_pruned_trees:
                    return ragged_attention_with_parents(
                        Q_r, K_r, V_r, cu, parents_tensor, actual_d, max_seqlen=N)[0]
                else:
                    return ragged_attention(Q_r, K_r, V_r, cu, b, actual_d, max_seqlen=N)

        # Apply CUDA Graph (same methodology as FlashInfer — fair comparison)
        ragged_g_fn, _ragged_graphed = _wrap_cuda_graph(ragged_fn)
        if _ragged_graphed != _fi_graphed:
            print(
                f"    [WARN] Timing asymmetry: fi_graph={_fi_graphed} "
                f"ragged_graph={_ragged_graphed} — rerun with investigation"
            )
        t_ragged = _cuda_median_ms(ragged_g_fn, warmup, iters)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if fi_tree_oom:
            print(f"    [OOM] Both kernels OOMed for B={B} b={b} d={d} L={L} N={N} — skipping")
        else:
            print(f"    [OOM] Ragged OOMed but FlashInfer succeeded for B={B} b={b} d={d} L={L} N={N}")
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

    # ── Bandwidth utilization ────────────────────────────────────────────────
    # Ragged kernel tree-portion: each of B*N queries reads (actual_d+1) K+V entries.
    # bytes = Q_read + K_ancestors + V_ancestors + O_write
    bw_gb_s = nan
    bw_util_pct = nan
    if not math.isnan(t_ragged) and t_ragged > 0 and actual_d >= 0:
        elem = 2  # fp16 bytes per element
        n_queries = B * N
        bytes_tree = elem * H * D * n_queries * (2 + 2 * (actual_d + 1))  # Q+O + K+V ancestors
        bw_gb_s = bytes_tree / (t_ragged * 1e-3) / 1e9
        if peak_bw_gb_s > 0:
            bw_util_pct = bw_gb_s / peak_bw_gb_s * 100.0

    def _sp(num, den):
        if math.isnan(num) or math.isnan(den) or den <= 0:
            return nan
        return num / den

    speedup_sib_vs_base = nan
    if not (math.isnan(t_ragged) or math.isnan(t_sibling) or t_sibling <= 0):
        speedup_sib_vs_base = t_ragged / t_sibling

    # Cleanup
    del Q_r, K_r, V_r, cu, Q_s, K_s, V_s, attn_mask4d
    if K_prefix is not None:
        del K_prefix, V_prefix
    if K_shared is not None:
        del K_shared, V_shared
    torch.cuda.empty_cache()

    def _r(v):
        return round(v, 4) if not math.isnan(v) else nan

    return MicroRow(
        batch_size               = B,
        prefix_length            = L,
        branching_factor         = b,
        depth                    = d,
        effective_depth          = actual_d,
        num_tree_nodes           = N,
        num_heads                = H,
        head_dim                 = D,
        flashinfer_tree_ms       = _r(t_fi_tree),
        flashinfer_cascade_ms    = _r(t_fi_cascade),
        deft_ms                  = _r(t_deft),
        ragged_ms                = _r(t_ragged),
        ragged_sibling_ms        = _r(t_sibling),
        speedup_vs_tree          = _r(_sp(t_fi_tree,    t_ragged)),
        speedup_vs_cascade       = _r(_sp(t_fi_cascade, t_ragged)),
        speedup_vs_deft          = _r(_sp(t_deft,       t_ragged)),
        speedup_sibling_vs_base  = _r(_sp(t_ragged,     t_sibling)),
        bw_gb_s                  = _r(bw_gb_s),
        bw_util_pct              = _r(bw_util_pct),
    )


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
    parser.add_argument("--skip-cascade", action="store_true",
                        help="Skip FlashInfer cascade baseline.")
    parser.add_argument("--skip-deft",    action="store_true",
                        help="Skip DeFT baseline (auto-skipped if third_party/FastTree not cloned).")
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
    # Peak memory bandwidth in GB/s.
    # torch device properties lack clock/bus fields on some builds; query nvidia-smi instead.
    _KNOWN_BW = {                         # GB/s for common GPUs
        "H100 SXM5": 3350, "H100 SXM": 3350,
        "H100 PCIe": 2000, "H100":      3350,
        "A100-SXM4": 2000, "A100":      2000,
        "RTX 4090":   1008, "RTX 4080":  717,
        "RTX 4000 Ada": 288,
        "RTX 3090":    936, "RTX 3080":  760,
        "T4":          300,
    }
    peak_bw_gb_s = 0.0
    gpu_name = p.name
    for key, bw in _KNOWN_BW.items():
        if key in gpu_name:
            peak_bw_gb_s = float(bw)
            break
    if peak_bw_gb_s == 0.0:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=clocks.max.memory,memory.total",
                 "--format=csv,noheader"],
                timeout=5, text=True,
            ).strip().split("\n")[0]
            parts = [s.strip() for s in out.split(",")]
            clk_mhz = float(parts[0].split()[0])
            mem_mb  = float(parts[1].split()[0])
            # Estimate bus width from total VRAM (rough heuristic)
            if mem_mb >= 78000: bus = 5120
            elif mem_mb >= 38000: bus = 3072
            elif mem_mb >= 20000: bus = 320
            elif mem_mb >= 12000: bus = 192
            else:                 bus = 128
            peak_bw_gb_s = clk_mhz * 1e6 * bus / 8 * 2 / 1e9
        except Exception:
            peak_bw_gb_s = 0.0

    print("=" * 72)
    print("  sd-ragged  ·  Micro Benchmark  ·  FlashInfer / DeFT / Cascade vs Ragged")
    print("=" * 72)
    print(f"  GPU:      {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB  peak BW {peak_bw_gb_s:.0f} GB/s")
    print(f"  Dims:     H={H}, D={D}  (fp16)")
    print(f"  Grid:     n_seq (parallel trees, NOT branching)∈{batch_sizes}  b (branching)∈{bfs}  d (depth)∈{depths}"
          + (f"  N_cap={token_cap}" if token_cap else "  N_cap=none"))
    print(f"  context_len:  L∈{prefix_lengths}  (shared prefix; cascade enabled when L>0)")
    print(f"  N range:  "
          + "  ".join(f"b={b},d={depths[0]}→{_tree_n(b,depths[0],token_cap,args.node_count_mode)}"
                       f"/d={depths[-1]}→{_tree_n(b,depths[-1],token_cap,args.node_count_mode)}"
                       for b in bfs))
    print(f"  Timing:   {args.warmup} warmup + {args.iters} iters (median)")
    tree_mode = "EAGLE-like pruned (irregular branching)" if args.pruned_trees else "fully balanced"
    print(f"  Trees:    {tree_mode}")
    print(f"  N mode:   {args.node_count_mode}")
    print(f"  Baselines: FlashInfer-tree  FlashInfer-cascade  "
          + (f"DeFT (L=0 only)" if HAS_DEFT and not args.skip_deft else
             "DeFT [SKIP — third_party/FastTree not found]" if not HAS_DEFT else "DeFT [SKIP]"))
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
                            node_count_mode=args.node_count_mode,
                            skip_cascade=args.skip_cascade,
                            skip_deft=args.skip_deft,
                            peak_bw_gb_s=peak_bw_gb_s)
        rows.append(row)

        def _f(v):
            return f"{v:.3f}" if not math.isnan(v) else "  n/a "

        def _spd(v):
            return f"{v:.2f}×" if not math.isnan(v) else "  n/a"

        d_eff_s = f"{row.effective_depth:>2d}" if row.effective_depth >= 0 else "na"
        print(f"  [{ci+1:3d}/{len(configs)}]  "
              f"B={B:3d} b={b} d={d:>2d} d_eff={d_eff_s} L={L:>4d} N={N:3d}  "
              f"fi_tree={_f(row.flashinfer_tree_ms):>8s}ms  "
              f"cascade={_f(row.flashinfer_cascade_ms):>8s}ms  "
              f"deft={_f(row.deft_ms):>8s}ms  "
              f"ragged={_f(row.ragged_ms):>8s}ms  "
              f"spd_tree={_spd(row.speedup_vs_tree):>7s}  "
              f"bw={_f(row.bw_gb_s):>7s}GB/s  "
              f"bw_util={_f(row.bw_util_pct):>5s}%")

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

    # ── Pivot: speedup table (ragged vs FlashInfer-tree) ────────────────────
    for L in prefix_lengths:
        print()
        label = "tree-only" if L == 0 else f"context_len={L}"
        print(f"  Speedup table [{label}]  (fi_tree / ragged — >1 means ragged wins)")
        print("  " + "─" * 72)

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
                for label_col, attr in [("vs_tree", "speedup_vs_tree"),
                                         ("vs_casc", "speedup_vs_cascade"),
                                         ("vs_deft", "speedup_vs_deft"),
                                         ("bw GB/s", "bw_gb_s")]:
                    vals = []
                    for d in d_vals:
                        match = [r for r in subset if r.depth == d]
                        v = getattr(match[0], attr) if match else float("nan")
                        vals.append(f"{v:5.2f}" if not math.isnan(v) else "  n/a")
                    print(f"  {label_col:>8s}       │ " + "  ".join(f"{v:>5s}" for v in vals))
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r.speedup_vs_tree)]
    if valid:
        wins  = [r for r in valid if r.speedup_vs_tree > 1.0]
        best  = max(valid, key=lambda r: r.speedup_vs_tree)
        worst = min(valid, key=lambda r: r.speedup_vs_tree)
        med   = float(np.median([r.speedup_vs_tree for r in valid]))

        bw_valid = [r for r in rows if not math.isnan(r.bw_gb_s)]
        _nan = float("nan")
        med_bw   = float(np.median([r.bw_gb_s for r in bw_valid])) if bw_valid else _nan
        med_util = float(np.median([r.bw_util_pct for r in bw_valid])) if bw_valid else _nan

        print("=" * 72)
        print("  MICRO BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"  Configs tested:   {len(configs)}")
        print(f"  Valid results:    {len(valid)}")
        print(f"  Ragged wins (vs fi_tree):  {len(wins)} / {len(valid)}  "
              f"({100*len(wins)/len(valid):.0f}%)")
        print(f"  Median speedup (vs fi_tree): {med:.2f}×")
        print(f"  Best speedup:     {best.speedup_vs_tree:.2f}×  "
              f"(B={best.batch_size}, b={best.branching_factor}, "
              f"d={best.depth}, L={best.prefix_length})")
        print(f"  Worst speedup:    {worst.speedup_vs_tree:.2f}×  "
              f"(B={worst.batch_size}, b={worst.branching_factor}, "
              f"d={worst.depth}, L={worst.prefix_length})")
        print(f"  Median BW:        {med_bw:.1f} GB/s  ({med_util:.1f}% of peak {peak_bw_gb_s:.0f} GB/s)")

        # Per-context-len breakdown
        for L in prefix_lengths:
            sub = [r for r in valid if r.prefix_length == L]
            if not sub:
                continue
            label = "tree-only" if L == 0 else f"L={L}"
            sub_wins = [r for r in sub if r.speedup_vs_tree > 1.0]
            sub_med  = float(np.median([r.speedup_vs_tree for r in sub]))
            sub_best = max(sub, key=lambda r: r.speedup_vs_tree)
            print(f"\n  [{label}]  wins {len(sub_wins)}/{len(sub)}  "
                  f"median {sub_med:.2f}×  "
                  f"best {sub_best.speedup_vs_tree:.2f}× "
                  f"(B={sub_best.batch_size}, b={sub_best.branching_factor}, "
                  f"d={sub_best.depth})")

        # N_branches (batch) scaling — fixed (b, d), vary B and L
        print()
        print("  N_branches scaling  (speedup vs fi_tree, per context_len):")
        for b in bfs:
            for d in depths:
                print(f"    b={b} d={d}:")
                for L in prefix_lengths:
                    vals = []
                    for B in batch_sizes:
                        match = [r for r in valid
                                 if r.batch_size == B and r.branching_factor == b
                                 and r.depth == d and r.prefix_length == L]
                        spd = match[0].speedup_vs_tree if match else float("nan")
                        vals.append(f"B={B}:{spd:.2f}×" if not math.isnan(spd) else f"B={B}:n/a")
                    print(f"      L={L:>4d}: " + "  ".join(vals))
        print("=" * 72)
    print()


if __name__ == "__main__":
    main()
