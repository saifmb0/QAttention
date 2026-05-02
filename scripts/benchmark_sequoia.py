"""
benchmark_sequoia.py — RaggedAttention vs Sequoia vanilla attention

Two sections:
  --section micro  Time attention kernel only at Sequoia tree sizes.
                   No models needed; uses random Q/K/V.
  --section e2e    Full E2E tok/s with Llama-2-7B + TinyLlama-68m.
                   Requires --draft, --target, --growmap.
"""
import argparse
import contextlib
import math
import os
import sys
import time
import types as _types
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# ── repo root on path ──────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_SEQ = os.path.join(_REPO, "sequoia")
if not os.path.isdir(_SEQ):
    raise RuntimeError("sequoia/ not found — run: git clone https://github.com/Infini-AI-Lab/Sequoia sequoia/")
sys.path.insert(0, _SEQ)

from src.ragged_attn import (
    ragged_attention_with_parents,
    fused_lse_merge,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--section",    choices=["micro", "tree-sweep", "e2e", "e2e-sweep"], default="micro")
parser.add_argument("--growmap",    default=os.path.join(_SEQ, "A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt"))
parser.add_argument("--draft",      default="JackFram/llama-68m",          help="HF draft model (e2e only)")
parser.add_argument("--target",     default="meta-llama/Llama-2-7b-hf",    help="HF target model (e2e only)")
parser.add_argument("--n-prompts",  type=int, default=50)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--warmup",     type=int, default=20)
parser.add_argument("--iters",      type=int, default=100)
parser.add_argument("--prefix-lengths", default="0,512,1024",
                    help="Comma-separated prefix KV lengths to sweep in micro section")
parser.add_argument("--tree-sweep-dir",
                    default=os.path.join(_SEQ, "A100_growmaps/68m_13b/growmaps"),
                    help="Directory of growmaps for tree-size sweep (tree-sweep section)")
parser.add_argument("--tree-sweep-prefix", type=int, default=1024,
                    help="Fixed N_prefix for tree-size sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_growmap(path: str):
    gm = torch.load(path, map_location="cpu", weights_only=False)
    return gm


def growmap_parents(gm) -> torch.Tensor:
    """Convert grow_map Successors list to a parent array (int32, CPU).

    Returns parents[i] = parent of node i (local tree index).
    For root-children (parent = node 0, which is in prefix) we use a
    self-loop so ragged_attention_with_parents treats them as local roots.
    """
    successors: List[List[int]] = gm["Successors"]
    n = gm["size"]
    raw = torch.zeros(n, dtype=torch.int32)
    for parent_idx, children in enumerate(successors):
        for c in children:
            raw[c] = parent_idx
    # raw[0] = 0  already (root self-loop)
    return raw


def _vanilla_attention(Q, K, V, scale):
    """Dense O(N_q × N_kv) attention — Sequoia's baseline."""
    w = torch.matmul(Q, K.transpose(-2, -1)) * scale
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(Q.dtype)
    return torch.matmul(w, V)


def _ragged_attention(Q, K, V, scale, parents_tree, max_depth):
    """
    Prefix (FA2) + tree (ragged kernel) + LSE merge.

    Q / K / V : [1, H, N_total, D]  where N_total = N_prefix + N_tree
    parents_tree : [N_tree] int32 LOCAL indices within tree portion.
    """
    import torch.ops  # noqa
    B, H, N_tree, D = Q.shape          # Q is tree tokens only
    N_kv             = K.shape[2]      # full KV cache length
    N_prefix         = N_kv - N_tree

    # ── 1. Prefix attention (FA2 / SDPA) ─────────────────────────────────────
    # Query: tree tokens only ([B, H, N_tree, D]) attending to prefix KV.
    if N_prefix > 0:
        Q_pre = Q.contiguous()                       # [B, H, N_tree, D] — tree queries
        K_pre = K[:, :, :N_prefix, :].contiguous()  # [B, H, N_prefix, D]
        V_pre = V[:, :, :N_prefix, :].contiguous()
        try:
            out_pre, lse_pre, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                Q_pre, K_pre, V_pre,
                dropout_p=0.0, is_causal=False, scale=scale,
                return_debug_mask=False,
            )
            # out_pre: [B, H, N_tree, D],  lse_pre: [B, H, N_tree]
        except Exception:
            out_pre = _vanilla_attention(Q_pre, K_pre, V_pre, scale)
            w = torch.matmul(Q_pre, K_pre.transpose(-2, -1)) * scale
            lse_pre = torch.logsumexp(w.float(), dim=-1)
    else:
        out_pre = None
        lse_pre = None

    # ── 2. Tree attention (ragged kernel) ─────────────────────────────────────
    # Q is already tree-only; K/V need to be sliced to tree portion
    Q_tree = Q.contiguous()
    K_tree = K[:, :, N_prefix:, :].contiguous()
    V_tree = V[:, :, N_prefix:, :].contiguous()
    Q_r = Q_tree.permute(0, 2, 1, 3).contiguous().view(B * N_tree, H, D)
    K_r = K_tree.permute(0, 2, 1, 3).contiguous().view(B * N_tree, H, D)
    V_r = V_tree.permute(0, 2, 1, 3).contiguous().view(B * N_tree, H, D)
    cu  = torch.arange(0, (B + 1) * N_tree, N_tree, dtype=torch.int32, device=Q.device)
    p   = parents_tree.to(Q.device)
    if B > 1:
        p = p.repeat(B)

    out_tree_r, lse_tree_r = ragged_attention_with_parents(
        Q_r, K_r, V_r, cu, p, max_depth, max_seqlen=N_tree,
    )

    out_tree  = out_tree_r.view(B, N_tree, H, D).permute(0, 2, 1, 3)
    lse_tree  = lse_tree_r.view(B, N_tree, H).permute(0, 2, 1)

    # ── 3. LSE merge → [B, H, N_tree, D] ─────────────────────────────────────
    if out_pre is not None:
        return fused_lse_merge(lse_pre.float(), lse_tree.float(), out_pre, out_tree)
    else:
        return out_tree.to(Q.dtype)
    # Note: returned tensor covers only tree tokens [N_tree], not N_total


# ═══════════════════════════════════════════════════════════════════════════════
# Section A — Attention micro-benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_micro(args):
    gm = load_growmap(args.growmap)
    tree_size: int = gm["size"]
    parents_full = growmap_parents(gm)  # [tree_size]

    # Tree portion: nodes 1..tree_size-1 (node 0 is root / last accepted token)
    # Their local parent indices within [0, tree_size-2].
    raw = parents_full[1:]            # parent grow_map indices for nodes 1..tree_size-1
    parents_tree = raw.clone()
    # nodes whose grow_map parent is 0 (the root, now in prefix) → self-loop
    is_root_child = raw == 0
    parents_tree[is_root_child] = torch.where(
        is_root_child, torch.arange(tree_size - 1, dtype=torch.int32), raw
    )[is_root_child]
    # nodes whose grow_map parent > 0 → shift by -1
    mask = raw > 0
    parents_tree[mask] = raw[mask] - 1

    N_tree = tree_size - 1
    max_depth = int(gm["depth"].max().item())

    # LLaMA-2-7B dims
    H, D = 32, 128
    device = torch.device("cuda:0")
    dtype  = torch.float16
    parents_tree = parents_tree.to(device)

    prefix_lengths = [int(x) for x in args.prefix_lengths.split(",")]

    print(f"\n{'─'*72}")
    print(f"  Sequoia Attention Micro-Benchmark")
    print(f"  Growmap: {os.path.basename(args.growmap)}")
    print(f"  Tree size N={tree_size}  (N_tree={N_tree}, max_depth={max_depth})")
    print(f"  Dims: H={H}, D={D},  warmup={args.warmup}, iters={args.iters}")
    print(f"{'─'*72}")
    print(f"  {'N_prefix':>10}  {'N_total':>8}  {'vanilla_ms':>12}  {'ragged_ms':>10}  {'speedup':>8}")
    print(f"{'─'*72}")

    scale = 1.0 / math.sqrt(D)

    for L in prefix_lengths:
        N_total = L + N_tree

        # Q = tree tokens only; K/V = full KV cache (prefix + tree)
        Q = torch.randn(1, H, N_tree,  D, device=device, dtype=dtype)
        K = torch.randn(1, H, N_total, D, device=device, dtype=dtype)
        V = torch.randn(1, H, N_total, D, device=device, dtype=dtype)

        # ── warmup ──────────────────────────────────────────────────────────
        # vanilla: Q[N_tree] × K[N_total] (Sequoia's actual computation)
        for _ in range(args.warmup):
            _vanilla_attention(Q, K, V, scale)
        for _ in range(args.warmup):
            _ragged_attention(Q, K, V, scale, parents_tree, max_depth)
        torch.cuda.synchronize()

        # ── vanilla timing ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _vanilla_attention(Q, K, V, scale)
        torch.cuda.synchronize()
        vanilla_ms = (time.perf_counter() - t0) / args.iters * 1000

        # ── ragged timing ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ragged_attention(Q, K, V, scale, parents_tree, max_depth)
        torch.cuda.synchronize()
        ragged_ms = (time.perf_counter() - t0) / args.iters * 1000

        speedup = vanilla_ms / ragged_ms
        print(f"  {L:>10}  {N_total:>8}  {vanilla_ms:>12.3f}  {ragged_ms:>10.3f}  {speedup:>7.2f}×")

    print(f"{'─'*72}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section A2 — Tree-size sweep at fixed N_prefix
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parents_tree(gm):
    """Return (parents_tree, N_tree, max_depth) for a growmap."""
    tree_size: int = gm["size"]
    parents_full = growmap_parents(gm)
    raw = parents_full[1:]
    parents_tree = raw.clone()
    is_root_child = raw == 0
    parents_tree[is_root_child] = torch.where(
        is_root_child, torch.arange(tree_size - 1, dtype=torch.int32), raw
    )[is_root_child]
    mask = raw > 0
    parents_tree[mask] = raw[mask] - 1
    N_tree = tree_size - 1
    max_depth = int(gm["depth"].max().item())
    return parents_tree, N_tree, max_depth


def run_tree_sweep(args):
    """Sweep tree sizes at a fixed N_prefix using all *.pt files in --tree-sweep-dir."""
    import glob

    sweep_dir = args.tree_sweep_dir
    pt_files = sorted(glob.glob(os.path.join(sweep_dir, "*.pt")))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {sweep_dir}")

    N_prefix = args.tree_sweep_prefix
    H, D = 32, 128
    device = torch.device("cuda:0")
    dtype  = torch.float16
    scale  = 1.0 / math.sqrt(D)

    # Load all growmaps, deduplicate by tree size (keep one per size), sort ascending
    seen_sizes = {}
    for pt in pt_files:
        try:
            gm = load_growmap(pt)
            sz = int(gm["size"])
            if sz not in seen_sizes:
                seen_sizes[sz] = (pt, gm)
        except Exception:
            continue

    entries = sorted(seen_sizes.items())  # [(size, (pt, gm)), ...]

    print(f"\n{'─'*80}")
    print(f"  Sequoia Attention — Tree-Size Sweep at N_prefix={N_prefix}")
    print(f"  Dir: {os.path.basename(sweep_dir)}")
    print(f"  Dims: H={H}, D={D},  warmup={args.warmup}, iters={args.iters}")
    print(f"{'─'*80}")
    print(f"  {'N_tree':>8}  {'N_total':>8}  {'depth':>6}  {'vanilla_ms':>12}  {'ragged_ms':>10}  {'speedup':>8}")
    print(f"{'─'*80}")

    for sz, (pt, gm) in entries:
        parents_tree, N_tree, max_depth = _build_parents_tree(gm)
        parents_tree = parents_tree.to(device)
        N_total = N_prefix + N_tree

        Q = torch.randn(1, H, N_tree,  D, device=device, dtype=dtype)
        K = torch.randn(1, H, N_total, D, device=device, dtype=dtype)
        V = torch.randn(1, H, N_total, D, device=device, dtype=dtype)

        for _ in range(args.warmup):
            _vanilla_attention(Q, K, V, scale)
        for _ in range(args.warmup):
            _ragged_attention(Q, K, V, scale, parents_tree, max_depth)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _vanilla_attention(Q, K, V, scale)
        torch.cuda.synchronize()
        vanilla_ms = (time.perf_counter() - t0) / args.iters * 1000

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ragged_attention(Q, K, V, scale, parents_tree, max_depth)
        torch.cuda.synchronize()
        ragged_ms = (time.perf_counter() - t0) / args.iters * 1000

        speedup = vanilla_ms / ragged_ms
        print(f"  {N_tree:>8}  {N_total:>8}  {max_depth:>6}  {vanilla_ms:>12.3f}  {ragged_ms:>10.3f}  {speedup:>7.2f}×")

    print(f"{'─'*80}")


# ═══════════════════════════════════════════════════════════════════════════════
# Context manager — patch LlamaAttention_TG for ragged verify
# ═══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def sequoia_ragged_context(target_engine, gm, kernel_fires: List):
    """
    Patches LlamaAttention_TG.forward on every layer of the target model
    to use our ragged kernel for tree-verify steps (q_len == tree_size).
    The first verify call (prefill + tree, q_len > tree_size) falls back to vanilla.
    kernel_fires is a single-element list used as a mutable counter.
    """
    from Engine.Llama_modules import LlamaAttention_TG
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    tree_size: int  = gm["size"]
    # During steady-state verify, Sequoia queries tree_size tokens:
    # tokens[target_kv_len : target_kv_len + tree_size]
    #   = [residual, draft_1, ..., draft_{tree_size-1}]
    # The residual is the new tree root (query[0]), so growmap_parents
    # maps directly: parents[i] = parent of query[i] in [0, tree_size-1].
    N_tree    = tree_size
    max_depth = int(gm["depth"].max().item())

    parents_full = growmap_parents(gm)   # [tree_size], parents[0]=0 (self-loop for residual)
    _pt_device = parents_full.to("cuda:0")

    _AttnClass = LlamaAttention_TG
    _orig_fwd  = _AttnClass.forward

    def _ragged_fwd(
        self_attn,
        hidden_states: torch.Tensor,
        max_length: int,
        storage_ids: torch.LongTensor,
        attention_mask=None,
        position_ids=None,
        kv_cache=None,
        debug: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Fallback to vanilla for initial prefill (q_len includes prefix + tree)
        if q_len != N_tree:
            return _orig_fwd(self_attn, hidden_states, max_length, storage_ids,
                             attention_mask, position_ids, kv_cache, debug)

        query_states = self_attn.q_proj(hidden_states)
        key_states   = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self_attn.num_heads,           self_attn.head_dim).transpose(1, 2)
        key_states   = key_states  .view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

        cos, sin = self_attn.rotary_emb(value_states.dtype, seq_len=max_length)
        cos, sin = cos[position_ids], sin[position_ids]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states = kv_cache.update_kv_cache(
            key_states, value_states, self_attn.layer_idx,
            storage_ids=storage_ids, debug=debug,
        )
        kv_len = kv_cache.get_usable_length(layer_idx=self_attn.layer_idx,
                                             input_length=len(storage_ids))
        key_states   = key_states  [..., :kv_len, :]
        value_states = value_states[..., :kv_len, :]

        key_states   = repeat_kv(key_states,   self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

        _scale = 1.0 / math.sqrt(self_attn.head_dim)
        attn_output = _ragged_attention(
            query_states, key_states, value_states, _scale, _pt_device, max_depth
        )
        kernel_fires[0] += 1

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)
        attn_output = self_attn.o_proj(attn_output)
        return attn_output

    _AttnClass.forward = _ragged_fwd
    try:
        yield
    finally:
        _AttnClass.forward = _orig_fwd


# ═══════════════════════════════════════════════════════════════════════════════
# Section B — Full E2E
# ═══════════════════════════════════════════════════════════════════════════════

def _load_engines(args, M, T, vocab_size_override=None):
    """Load draft+target engines. Returns (draft_engine, target_engine, tokenizer, vocab_size)."""
    from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
    from transformers import AutoTokenizer
    device = "cuda:0"
    dtype  = torch.float16
    draft_engine = GraphInferenceEngine(
        max_length=M, model_name_or_path=args.draft, dtype=dtype, device=device,
    )
    target_engine = GraphInferenceEngineTG(
        max_length=M, model_name_or_path=args.target, dtype=dtype, device=device,
    )
    tokenizer  = AutoTokenizer.from_pretrained(args.target)
    vocab_size = vocab_size_override or draft_engine.engine.model.config.vocab_size
    return draft_engine, target_engine, tokenizer, vocab_size


def _run_one_growmap(draft_engine, target_engine, tokenizer, vocab_size,
                     gm, growmap_path, prompts, M, T, top_p, device, dtype):
    """Run vanilla + ragged E2E on one growmap. Returns (v_tok, r_tok, speedup)."""
    from Tree.SpecTree import SpecTree
    from utils import (
        cuda_graph_for_residual,
        cuda_graph_for_sampling_without_replacement,
    )

    tree_size: int = gm["size"]
    idx_lists      = gm["roots"]
    branch_lists   = gm["branches"]
    draft_step     = len(idx_lists)

    # Sampling callables must be built BEFORE initialize_cuda_graph (see comment in run_e2e)
    residual_graph        = cuda_graph_for_residual(dim=vocab_size)
    sampling_callables    = {}
    sample_gather_indices = {}
    for i in range(draft_step - 1):
        idx_len     = len(idx_lists[i])
        num_samples = max(branch_lists[i])
        sampling_callables[i] = cuda_graph_for_sampling_without_replacement(
            max_length=M, idx_len=idx_len, num_samples=num_samples,
            temperature=T, tree_size=tree_size, dim=vocab_size,
        )
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device=device, dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        sample_gather_indices[i] = torch.cat(ith_gather_list)

    graph_capture_list = [sum(x) for x in branch_lists]
    graph_capture_list.append(1)
    draft_engine.initialize_cuda_graph(graph_capture_list)

    attn_mask         = torch.full((M, M), torch.finfo(dtype).min, dtype=dtype, device=device)
    sequence          = torch.arange(M, device=device).long().unsqueeze(-1)
    position_ids      = torch.zeros(M, dtype=torch.long, device=device)
    new_tokens_buffer = torch.zeros(M, dtype=torch.long, device=device)
    parents_buffer    = torch.zeros(M, dtype=torch.long, device=device)

    def _run_simulation(use_ragged: bool, label: str):
        kernel_fires = [0]
        total_tok    = 0
        total_steps  = 0
        ctx = (
            sequoia_ragged_context(target_engine, gm, kernel_fires)
            if use_ragged else contextlib.nullcontext()
        )
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with ctx:
            for prompt in prompts:
                ids = tokenizer("[INST]" + prompt + "[/INST]",
                                return_tensors="pt").input_ids.to(device)
                if ids.shape[1] > 200:
                    continue
                attn_mask.fill_(torch.finfo(dtype).min)
                draft_engine.clear_kv()
                target_engine.clear_kv()
                spectree = SpecTree(
                    prefix=ids.squeeze(0), device=device,
                    temperature=T, top_p=top_p,
                    draft_kv_len=0, target_kv_len=0,
                    draft_model_engine=draft_engine,
                    target_model_engine=target_engine,
                    max_length=M, grow_map=gm,
                    attn_mask=attn_mask, sequence=sequence,
                    new_tokens_buffer=new_tokens_buffer,
                    parents_buffer=parents_buffer,
                    position_ids=position_ids,
                    residual_graph=residual_graph,
                    sampling_callables=sampling_callables,
                    sample_gather_indices=sample_gather_indices,
                    vocab_size=vocab_size,
                )
                input_len = ids.shape[1]
                terminate = False
                n_tok = n_steps = 0
                while n_tok + input_len < 256 and not terminate:
                    spectree.construct_grow_map()
                    valid_tokens, _, _, terminate = spectree.verify()
                    n_tok     += valid_tokens.shape[0] - input_len
                    input_len  = valid_tokens.shape[0]
                    n_steps   += 1
                    if valid_tokens[-1] in (0, 2):
                        terminate = True
                total_tok   += n_tok
                total_steps += n_steps
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        tok_s       = total_tok / max(elapsed, 1e-9)
        acc_per_step = total_tok / max(total_steps, 1)
        print(f"  [{label:7s}]  {tok_s:7.1f} tok/s   "
              f"{total_tok} tok / {total_steps} steps   "
              f"acc/step={acc_per_step:.2f}   "
              + (f"kernel_fires={kernel_fires[0]}" if use_ragged else ""))
        return tok_s, acc_per_step

    v_tok, v_acc   = _run_simulation(use_ragged=False, label="vanilla")
    r_tok, r_acc   = _run_simulation(use_ragged=True,  label="ragged")
    speedup = r_tok / v_tok
    print(f"\n  ▸ Speedup: {speedup:.3f}×  ({v_tok:.1f} → {r_tok:.1f} tok/s)"
          f"  acc/step: vanilla={v_acc:.2f}  ragged={r_acc:.2f}")

    return v_tok, r_tok, speedup, v_acc, r_acc


def run_e2e(args):
    device = "cuda:0"
    dtype  = torch.float16
    M      = args.max_length
    T      = 0.6
    top_p  = 0.9

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    print(f"\nLoading models…")
    draft_engine, target_engine, tokenizer, vocab_size = _load_engines(args, M, T)
    print(f"  draft={args.draft}  target={args.target}  M={M}  vocab={vocab_size}")

    gm = load_growmap(args.growmap)
    print(f"  growmap: {os.path.basename(args.growmap)}  tree_size={gm['size']}")

    base_prompts = [
        "The history of artificial intelligence begins in",
        "Quantum computing promises to revolutionize",
        "The human brain contains approximately",
        "Large language models are trained on",
        "Climate change is driven primarily by",
    ]
    prompts = (base_prompts * max(1, args.n_prompts // len(base_prompts) + 1))[:args.n_prompts]

    print(f"\n{'─'*72}")
    print(f"  Sequoia E2E  ({len(prompts)} prompts, M={M}, T={T})")
    print(f"{'─'*72}")
    _run_one_growmap(draft_engine, target_engine, tokenizer, vocab_size,
                     gm, args.growmap, prompts, M, T, top_p, device, dtype)


def run_e2e_sweep(args):
    """Sweep through multiple tree-size growmaps in one invocation."""
    import re, glob
    device = "cuda:0"
    dtype  = torch.float16
    M      = args.max_length
    T      = 0.6
    top_p  = 0.9

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    sweep_dir = args.tree_sweep_dir
    pattern   = os.path.join(sweep_dir, "*.pt")
    all_maps  = sorted(glob.glob(pattern))
    # Filter to S-prefixed size maps and sort by tree size numerically
    def _size(p):
        m = re.search(r"-S(\d+)\.pt$", p)
        return int(m.group(1)) if m else 0
    all_maps = [p for p in all_maps if _size(p) > 0]
    all_maps.sort(key=_size)

    if not all_maps:
        raise RuntimeError(f"No growmaps found in {sweep_dir}")

    print(f"  draft={args.draft}  target={args.target}  M={M}")

    base_prompts = [
        "The history of artificial intelligence begins in",
        "Quantum computing promises to revolutionize",
        "The human brain contains approximately",
        "Large language models are trained on",
        "Climate change is driven primarily by",
    ]
    prompts = (base_prompts * max(1, args.n_prompts // len(base_prompts) + 1))[:args.n_prompts]

    draft_engine = target_engine = tokenizer = vocab_size = None
    rows = []
    for gmap_path in all_maps:
        sz = _size(gmap_path)
        gm = load_growmap(gmap_path)
        print(f"\n{'═'*72}")
        print(f"  growmap: {os.path.basename(gmap_path)}  tree_size={gm['size']}")
        print(f"{'─'*72}")
        # Reload fresh engines per growmap: different tree sizes require
        # different CUDA graph capture lists; sharing engines across sizes
        # is not safe once initialize_cuda_graph has run.
        del draft_engine, target_engine
        torch.cuda.empty_cache()
        print(f"\n  Loading models…")
        draft_engine, target_engine, tokenizer, vocab_size = _load_engines(args, M, T)
        v_tok, r_tok, speedup, v_acc, r_acc = _run_one_growmap(
            draft_engine, target_engine, tokenizer, vocab_size,
            gm, gmap_path, prompts, M, T, top_p, device, dtype,
        )
        rows.append((sz, v_tok, r_tok, speedup, v_acc, r_acc))

    print(f"\n{'═'*72}")
    print(f"  E2E sweep summary  ({len(prompts)} prompts, M={M})")
    print(f"{'─'*72}")
    print(f"  {'tree_size':>10}  {'vanilla tok/s':>14}  {'ragged tok/s':>13}  {'speedup':>9}  {'v acc/step':>11}  {'r acc/step':>11}")
    for sz, v, r, sp, va, ra in rows:
        print(f"  {sz:>10}  {v:>14.2f}  {r:>13.2f}  {sp:>8.3f}×  {va:>11.2f}  {ra:>11.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parser.parse_args()
    if args.section == "micro":
        run_micro(args)
    elif args.section == "tree-sweep":
        run_tree_sweep(args)
    elif args.section == "e2e-sweep":
        run_e2e_sweep(args)
    else:
        run_e2e(args)
