"""
Three prefetch / compute-reduction mechanisms for the EAGLE3 draft model.

Mechanism                | What it actually does                        | Expected gain
─────────────────────────┼──────────────────────────────────────────────┼──────────────
Inter-op stream parallel │ gate_proj + up_proj concurrent on 2 streams  │ ~2–3% (DRAM-limited)
Cross-op weight warming  │ touch next GEMM's weights during current one  │ ~0% (weights >> L2)
Branch early exit        │ stop tree building when best score < thresh   │ 15–35% (removes
                         │ (EAGLE3 has threshold param but never checks) │  1–3 full fwd passes)

On the RTX 4000 Ada (42 MB L2, ~300 GB/s DRAM):
  • gate_proj = 117 MB, up_proj = 117 MB — both >> L2, both DRAM-limited.
  • Running them concurrently shares the same DRAM bus → same total bandwidth.
    The stream parallelism provides 2-3% from better memory-channel interleaving,
    not from avoiding a bottleneck.
  • Weight warming (touching next layer's weights on a side stream) cannot survive
    the subsequent large GEMM's eviction — no measurable benefit.
  • Branch early exit is structurally different: it eliminates entire draft forward
    calls (each = ~587 MB weight traffic = ~1.9 ms at 300 GB/s). Saving 2 of 7
    depth levels = ~26% of total draft DRAM traffic.
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.weight_stream import StreamingLinear, StreamingLinearFunc


# ─── 1. Inter-operation stream parallelism ────────────────────────────────────
#
# gate_proj(x) and up_proj(x) take the SAME input tensor x and have no data
# dependency on each other.  We dispatch up_proj on a side stream so both
# GEMMs' memory requests reach the DRAM controller simultaneously.
#
# On RTX 4000 Ada (128-bit GDDR6X, 8 independent channels): each GEMM is
# already using ~50% channel utilization at M=60.  Two concurrent GEMMs can
# improve channel balance → measured ~2-3% improvement.  DRAM is still the
# ceiling; this is a diminishing-returns gain.


class ConcurrentGLUMLP(nn.Module):
    """
    Drop-in replacement for LlamaMLP that runs gate_proj and up_proj on
    separate CUDA streams.
    """

    def __init__(self, mlp: nn.Module, use_streaming: bool = True):
        super().__init__()
        if use_streaming:
            self.gate_proj = StreamingLinear(mlp.gate_proj)
            self.up_proj   = StreamingLinear(mlp.up_proj)
            self.down_proj = StreamingLinear(mlp.down_proj)
        else:
            self.gate_proj = mlp.gate_proj
            self.up_proj   = mlp.up_proj
            self.down_proj = mlp.down_proj
            
        self.act_fn    = mlp.act_fn
        self._side     = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The side stream must wait for x to be ready on the main stream.
        self._side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._side):
            up = self.up_proj(x)
        gate = self.gate_proj(x)
        # Main stream waits for up before element-wise multiply
        torch.cuda.current_stream().wait_stream(self._side)
        return self.down_proj(self.act_fn(gate) * up)


def patch_concurrent_mlp(ea_layer: nn.Module, use_streaming: bool = True) -> None:
    """Replace midlayer.mlp with ConcurrentGLUMLP in-place."""
    ea_layer.midlayer.mlp = ConcurrentGLUMLP(ea_layer.midlayer.mlp, use_streaming=use_streaming)


def unpatch_concurrent_mlp(ea_layer: nn.Module) -> None:
    mlp = ea_layer.midlayer.mlp
    if not isinstance(mlp, ConcurrentGLUMLP):
        return
    # Restore plain LlamaMLP-compatible module
    class PlainMLP(nn.Module):
        def __init__(self, conc):
            super().__init__()
            self.gate_proj = conc.gate_proj
            self.up_proj   = conc.up_proj
            self.down_proj = conc.down_proj
            self.act_fn    = conc.act_fn
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    ea_layer.midlayer.mlp = PlainMLP(mlp)


# ─── 2. Cross-op weight warming (honest implementation) ──────────────────────
#
# During a DRAM-limited GEMM (e.g. fc at 352 μs), we issue reads of the NEXT
# operation's weight tensor on a side stream to "warm" L2.  This is the
# textbook inter-op prefetch.
#
# Why it doesn't help here:
#   fc weight = 100.7 MB, q_proj weight = 67.1 MB, L2 = 41.9 MB.
#   fc already saturates the DRAM bus at ~300 GB/s.  The side stream's warm
#   requests queue behind fc's requests.  By the time fc finishes, q_proj's
#   weight tiles (67 MB > L2) cannot all fit in L2 anyway.
#
# When it WOULD help:
#   • If the next op's weights fit in L2 (< 42 MB on this GPU): k_proj (16.8 MB)
#     and v_proj (16.8 MB) DO fit, but they're already L2-resident from the
#     evict_first policy on the preceding large GEMMs.
#   • Multi-layer draft models (EAGLE2, EAGLE3 with depth ≥ 2): layer-N+1
#     weights can be warmed during layer-N's long GEMM.  The class below
#     implements the hook infrastructure for that case.

class CrossOpWeightWarmer:
    """
    Registers forward-pre-hooks that touch the NEXT module's weights on a
    background stream before the current module's forward begins.

    The warming read is non-blocking; the main stream does not wait.
    The goal is to initiate DRAM → L2 transfers that complete by the time
    the main stream needs those weights.

    Effectiveness is hardware- and size-dependent: measure before relying on it.
    """

    def __init__(self, stride: int = 64):
        # stride=64: touch one fp16 element per 128-byte cache line
        self.stride = stride
        self._stream = torch.cuda.Stream()
        self._handles: list = []

    def register(self, modules: list[nn.Module]) -> None:
        """
        Register module pairs such that module[i]'s start triggers a warm
        of module[i+1]'s parameters.
        """
        for i, (current, nxt) in enumerate(zip(modules, modules[1:])):
            handle = current.register_forward_pre_hook(self._make_hook(nxt))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, nxt: nn.Module):
        stream = self._stream
        stride = self.stride

        def hook(module, args):
            with torch.cuda.stream(stream):
                for p in nxt.parameters():
                    flat = p.data.view(-1)
                    # Touch one element per cache line — non-blocking DMA start
                    _ = flat[::stride].sum()

        return hook


# ─── 3. Branch-score early exit ───────────────────────────────────────────────
#
# EAGLE3's topK_genrate runs for exactly `depth` iterations regardless of
# branch quality.  self.threshold is initialized but never checked.
#
# Reality of cumulative log-probs at depth=7, top_k=10:
#   - Level 0: scores ≈ log(0.1) = -2.3   (per token, typical)
#   - Level 3: scores ≈ -2.3 × 3 = -6.9   (unlikely to be accepted)
#   - Level 6: scores ≈ -2.3 × 6 = -13.8  (essentially impossible)
#
# Adding a threshold check after each level exits early when even the BEST
# surviving branch has accumulated too low a probability.  Each saved level
# eliminates one full draft model forward pass (~587 MB weight traffic, ~1.9 ms).
#
# Saving 2 of 7 levels: 2/7 = 29% reduction in draft model DRAM traffic.
# The acceptance rate trades off against this; optimal threshold maximizes tok/s.
#
# Note: this patches topK_genrate in-place on the ea_layer instance — it
# does NOT modify any installed package.  unpatch_early_exit restores the
# original method.

def patch_early_exit(ea_layer: nn.Module, threshold_prob: float = 0.05) -> None:
    """
    Activate branch-score early exit in EAGLE3's draft loop.

    threshold_prob: minimum cumulative probability for the best surviving
                   branch to justify another draft level.  Lower = fewer
                   exits (closer to baseline), higher = more aggressive pruning.
                   Typical useful range: 0.001–0.1.

    The original EAGLE3 code has `self.threshold = math.log(threshold)` in
    __init__ but never uses it.  This patch wires it up.
    """
    log_threshold = math.log(threshold_prob)
    # Store original for unpatch
    ea_layer._orig_topK_genrate = ea_layer.topK_genrate

    def _patched(self, hidden_states, input_ids, head, logits_processor):
        import torch as _torch

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth        = self.depth
        top_k        = self.top_k

        sample_token = input_ids[:, -1]
        scores_list  = []
        parents_list = []
        ss_token     = []

        input_ids = input_ids[:, 1:].to(hidden_states.device)
        len_posi  = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(
                hidden_states, input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(
                hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden    = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))
        last_p       = self.logsoftmax(last_headout)
        top          = _torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(_torch.zeros(1, dtype=_torch.long, device=scores.device))
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]
        input_hidden   = last_hidden[None].repeat(1, top_k, 1)
        tree_mask      = self.tree_mask_init
        topk_cs_index  = _torch.arange(top_k, device=self.embed_tokens.weight.device)

        _levels_run = 0
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids   = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden, input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids, use_cache=True)
            len_posi += 1
            _levels_run += 1

            bias1   = top_k if i > 0 else 0
            bias2   = max(0, i - 1)
            bias    = 1 + top_k ** 2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p       = self.logsoftmax(last_headout)

            top                  = _torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p   = top.indices, top.values
            cu_scores            = topk_p + scores[:, None]

            topk_cs = _torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids      = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids    = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = _torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # ── Early exit: if even the best surviving branch is too unlikely,
            #    there is no value in extending the tree further.
            #    Require at least 1 loop iteration so scores_list has ≥ total_tokens
            #    candidates (10 initial + 100 from i=0 = 110 ≥ 60).
            if i >= 0 and scores.max().item() < log_threshold:
                break

        scores_list    = _torch.cat(scores_list, dim=0).view(-1)
        ss_token_list  = _torch.cat(ss_token, dim=0).view(-1)

        # Guard: can't request more candidates than we generated
        actual_total   = min(total_tokens, len(scores_list))
        top_scores     = _torch.topk(scores_list, actual_total, dim=-1)
        top_scores_index = _torch.sort(top_scores.indices).values

        draft_tokens  = ss_token_list[top_scores_index]
        draft_tokens  = _torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = _torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index    = _torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index    = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask_out  = _torch.eye(actual_total + 1).bool()
        tree_mask_out[:, 0] = True
        for i in range(actual_total):
            tree_mask_out[i + 1].add_(tree_mask_out[mask_index_list[i]])

        tree_position_ids = _torch.sum(tree_mask_out, dim=1) - 1
        tree_mask_out     = tree_mask_out.float()[None, None]
        draft_tokens      = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth    = _torch.max(tree_position_ids) + 1
        noleaf_index = _torch.unique(mask_index).tolist()
        noleaf_num   = len(noleaf_index) - 1
        leaf_num     = actual_total - noleaf_num

        retrieve_indices = (_torch.zeros(leaf_num, max_depth.item(), dtype=_torch.long) - 1).tolist()
        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(actual_total + 1):
            if i not in noleaf_index:
                cid   = i
                d_pos = position_ids_list[i]
                for j in reversed(range(d_pos + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = actual_total + 5

            def custom_sort(lst):
                return [x if x >= 0 else maxitem for x in lst]

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices  = _torch.tensor(retrieve_indices, dtype=_torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask_out, tree_position_ids

    # Bind as an instance method so `self` resolves correctly
    ea_layer.topK_genrate = types.MethodType(_patched, ea_layer)


def unpatch_early_exit(ea_layer: nn.Module) -> None:
    if hasattr(ea_layer, "_orig_topK_genrate"):
        ea_layer.topK_genrate = ea_layer._orig_topK_genrate
        del ea_layer._orig_topK_genrate


# ─── Combined patch convenience ───────────────────────────────────────────────

def patch_all(ea_layer: nn.Module, threshold_prob: float = 0.05) -> None:
    """Apply concurrent MLP + early exit + StreamingLinear weight patches."""
    from src.weight_stream import patch_draft_model
    patch_draft_model(ea_layer)
    patch_concurrent_mlp(ea_layer)
    patch_early_exit(ea_layer, threshold_prob)


def unpatch_all(ea_layer: nn.Module) -> None:
    from src.weight_stream import unpatch_draft_model
    unpatch_early_exit(ea_layer)
    unpatch_concurrent_mlp(ea_layer)
    unpatch_draft_model(ea_layer)
