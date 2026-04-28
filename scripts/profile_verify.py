#!/usr/bin/env python3
"""
profile_verify.py
=================
Fine-grained CUDA-event profiler for the Eagle-3 verify step.

Instruments each LlamaDecoderLayer to split verify_ms into:
  draft_ms      — EAGLE cnets forward (not quantized in any mode)
  attn_ms       — QKV proj + attention core + O proj  (32 layers summed)
  mlp_ms        — gate + up + down proj              (32 layers summed)
  norm_ms       — input/post layernorm               (32 layers summed)
  other_ms      — residuals, misc

Runs both vanilla-SDPA and ragged paths on the same inputs so the
attn delta is directly visible.

Usage
-----
  # FP16 baseline
  python scripts/profile_verify.py

  # BitsAndBytes NF4 4-bit (same base model, quantized weights)
  python scripts/profile_verify.py --load-in-4bit

  # Different tree config
  python scripts/profile_verify.py --branching 8 --depth 8

  # More warmup/rep steps for stable averages
  python scripts/profile_verify.py --warmup 20 --reps 100

Reads model / EAGLE defaults from w4a16_e2e.py so configs stay in sync.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

# ── Silence HF noise ─────────────────────────────────────────────────────────
try:
    import transformers as _tr
    _tr.logging.set_verbosity_error()
except Exception:
    pass

# ── Eagle + ragged imports ────────────────────────────────────────────────────
try:
    import importlib.util
    if importlib.util.find_spec("eagle") is not None:
        # Apply same RoPE / transformer compat shims as w4a16_e2e.py
        try:
            import transformers.utils as _tu
            from typing import TypedDict
            if not hasattr(_tu, "LossKwargs"):
                class _LK(TypedDict, total=False): pass
                _tu.LossKwargs = _LK
            if not hasattr(_tu, "auto_docstring"):
                _tu.auto_docstring = lambda *a, **kw: (a[0] if a and callable(a[0]) else lambda f: f)
            if not hasattr(_tu, "can_return_tuple"):
                _tu.can_return_tuple = lambda f: f
        except Exception:
            pass
        from eagle.model.ea_model import EaModel
        import eagle.model.ea_model as _eagle_ea
        HAS_EAGLE = True
    else:
        HAS_EAGLE = False
except Exception:
    HAS_EAGLE = False

from src.ragged_attn import (
    ragged_attention_with_lse,
    ragged_attention_with_parents,
    fused_lse_merge,
)
from src.tree_mask import tree_attention_mask, num_tree_nodes

# ─────────────────────────────────────────────────────────────────────────────
# Layer-level CUDA event hooks
# ─────────────────────────────────────────────────────────────────────────────

class _LayerProfiler:
    """
    Hooks into each LlamaDecoderLayer to record 5 CUDA events per forward:
        [0] layer start
        [1] attn start  (after input_norm)
        [2] attn end    (after o_proj + residual add)
        [3] mlp start   (after post_norm)
        [4] mlp end / layer end

    Timing is collected across all steps and all layers, then averaged.
    """

    def __init__(self):
        self.reset()
        self._hooks: list = []

    def reset(self):
        # Raw elapsed times (ms) per category, summed across all (step × layer) calls
        self.norm_ms  = 0.0  # input_layernorm + post_layernorm
        self.attn_ms  = 0.0  # q/k/v_proj + core attention + o_proj
        self.mlp_ms   = 0.0  # gate + up + down proj
        self.total_ms = 0.0  # full layer
        self.n_calls  = 0    # (num_steps × num_layers)

    def _make_hooks(self, layer: nn.Module):
        """Install pre/mid/post hooks on a single LlamaDecoderLayer."""
        evts: List[torch.cuda.Event] = []

        def _pre_layer(mod, inp):
            nonlocal evts
            evts = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            evts[0].record()   # layer start

        def _pre_attn(mod, inp):
            evts[1].record()   # attn start (input_norm already done by caller)

        def _post_attn(mod, inp, out):
            evts[2].record()   # attn + residual done

        def _pre_mlp(mod, inp):
            evts[3].record()   # mlp start (post_norm done by caller)

        def _post_layer(mod, inp, out):
            evts[4].record()   # layer end
            torch.cuda.synchronize()
            # Compute intervals
            norm_t  = evts[0].elapsed_time(evts[1]) + evts[2].elapsed_time(evts[3])
            attn_t  = evts[1].elapsed_time(evts[2])
            mlp_t   = evts[3].elapsed_time(evts[4])
            total_t = evts[0].elapsed_time(evts[4])
            self.norm_ms  += norm_t
            self.attn_ms  += attn_t
            self.mlp_ms   += mlp_t
            self.total_ms += total_t
            self.n_calls  += 1

        h0 = layer.register_forward_pre_hook(_pre_layer)
        h1 = layer.self_attn.register_forward_pre_hook(_pre_attn)
        h2 = layer.self_attn.register_forward_hook(_post_attn)
        h3 = layer.mlp.register_forward_pre_hook(_pre_mlp)
        h4 = layer.register_forward_hook(_post_layer)
        self._hooks += [h0, h1, h2, h3, h4]

    def attach(self, base_model):
        """Attach hooks to all decoder layers of the base model."""
        layers = base_model.model.layers
        for layer in layers:
            self._make_hooks(layer)
        print(f"  [profiler] hooked {len(layers)} decoder layers")

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self, n_steps: int, n_layers: int, label: str) -> str:
        if self.n_calls == 0:
            return f"  [{label}] no data"
        # Per-step totals (summed over all layers per step)
        layers_per_step = max(self.n_calls // max(n_steps, 1), 1)
        norm_step  = self.norm_ms  / n_steps
        attn_step  = self.attn_ms  / n_steps
        mlp_step   = self.mlp_ms   / n_steps
        total_step = self.total_ms / n_steps
        other_step = total_step - norm_step - attn_step - mlp_step

        def pct(v): return v / total_step * 100 if total_step else 0.0

        lines = [
            f"  ┌── [{label}]  n_steps={n_steps}  layers={layers_per_step}  "
            f"calls={self.n_calls} {'─' * 20}",
            f"  │ {'Component':<14}  {'ms/step':>9}  {'%':>6}  {'ms/call':>9}",
            f"  │ {'─' * 14}  {'─' * 9}  {'─' * 6}  {'─' * 9}",
            f"  │ {'input+post norm':<14}  {norm_step:>9.3f}  {pct(norm_step):>5.1f}%"
            f"  {self.norm_ms/self.n_calls:>9.4f}",
            f"  │ {'attention':<14}  {attn_step:>9.3f}  {pct(attn_step):>5.1f}%"
            f"  {self.attn_ms/self.n_calls:>9.4f}",
            f"  │ {'mlp':<14}  {mlp_step:>9.3f}  {pct(mlp_step):>5.1f}%"
            f"  {self.mlp_ms/self.n_calls:>9.4f}",
            f"  │ {'other':<14}  {other_step:>9.3f}  {pct(other_step):>5.1f}%"
            f"  {(self.total_ms - self.norm_ms - self.attn_ms - self.mlp_ms) / self.n_calls:>9.4f}",
            f"  │ {'─' * 14}  {'─' * 9}  {'─' * 6}  {'─' * 9}",
            f"  │ {'TOTAL (verify)':<14}  {total_step:>9.3f}  {'100.0':>5s}%"
            f"  {self.total_ms/self.n_calls:>9.4f}",
            f"  └{'─' * 60}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic tree generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_tree_inputs(
    branching: int, depth: int, prefix_len: int,
    model, device: torch.device,
):
    """
    Build the (tree_candidates, past_key_values, tree_position_ids, input_ids,
    retrieve_indices) tuple that EAGLE's tree_decoding() expects.

    We fabricate random inputs so the profiler runs without a real prompt;
    the weight-traffic pattern is identical to a real verify step.
    """
    from eagle.model.utils import initialize_tree
    cfg = model.base_model.config
    H   = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", H)
    D   = cfg.hidden_size // H
    L   = cfg.num_hidden_layers

    N = num_tree_nodes(branching, depth)

    # Build a dummy past_key_values (StaticCache-style or tuple-of-tuple)
    kv_len = prefix_len + 1   # prefix + BOS
    past_key_values = tuple(
        (
            torch.randn(1, Hkv, kv_len + N, D, device=device, dtype=torch.float16),
            torch.randn(1, Hkv, kv_len + N, D, device=device, dtype=torch.float16),
        )
        for _ in range(L)
    )

    # tree_candidates: [1, N] token ids
    tree_candidates   = torch.randint(0, 32000, (1, N), device=device)
    tree_position_ids = torch.arange(N, device=device).unsqueeze(0)
    input_ids         = torch.randint(0, 32000, (1, kv_len), device=device)
    # retrieve_indices: [N, depth+1] — dummy
    retrieve_indices  = torch.zeros(N, depth + 1, dtype=torch.long, device=device)

    return tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices


# ─────────────────────────────────────────────────────────────────────────────
# Ragged attention context manager (simplified from w4a16_e2e.py)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _ragged_ctx(model, branching: int, depth: int, n_prefix: int):
    """Patch LlamaAttention.forward to use ragged kernel during verify."""
    try:
        from eagle.model.modeling_llama_kv import LlamaAttention
        from eagle.model.modeling_llama_kv import (
            LlamaRotaryEmbedding_L31, apply_rotary_pos_emb,
            apply_rotary_pos_emb_L31, repeat_kv,
        )
    except ImportError:
        yield
        return

    _orig_fwd = LlamaAttention.forward

    scale = 1.0 / math.sqrt(model.base_model.config.hidden_size //
                             model.base_model.config.num_attention_heads)

    def _ragged_fwd(self, hidden_states, attention_mask=None,
                    position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, **kw):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        num_heads    = self.num_heads
        num_kv_heads = self.num_key_value_heads
        head_dim     = self.head_dim

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        try:
            cos, sin = self.rotary_emb(value_states,
                                        seq_len=past_key_value[0].shape[-2] if past_key_value else q_len)
            query_states, key_states = apply_rotary_pos_emb_L31(
                query_states, key_states, cos, sin, position_ids)
        except Exception:
            pass

        # Append to KV cache
        if past_key_value is not None:
            key_states   = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)

        key_states   = repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = repeat_kv(value_states, num_heads // num_kv_heads)

        N_kv     = key_states.shape[2]
        N_pre    = N_kv - q_len

        if N_pre > 0:
            K_pre = key_states[:, :, :N_pre, :].contiguous()
            V_pre = value_states[:, :, :N_pre, :].contiguous()
            try:
                out_pre, lse_pre, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    query_states.contiguous(), K_pre, V_pre,
                    dropout_p=0.0, is_causal=False, scale=scale,
                    return_debug_mask=False,
                )
            except Exception:
                sc      = torch.matmul(query_states, K_pre.transpose(-2, -1)) * scale
                lse_pre = torch.logsumexp(sc.float(), dim=-1)
                out_pre = torch.softmax(sc, dim=-1) @ V_pre

        if bsz == 1:
            Q_r = query_states.squeeze(0).transpose(0, 1)
            K_r = key_states[:, :, N_pre:, :].squeeze(0).transpose(0, 1)
            V_r = value_states[:, :, N_pre:, :].squeeze(0).transpose(0, 1)
        else:
            Q_r = query_states.permute(0, 2, 1, 3).contiguous().view(bsz * q_len, num_heads, head_dim)
            K_r = key_states[:, :, N_pre:, :].permute(0, 2, 1, 3).contiguous().view(bsz * q_len, num_heads, head_dim)
            V_r = value_states[:, :, N_pre:, :].permute(0, 2, 1, 3).contiguous().view(bsz * q_len, num_heads, head_dim)

        cu = torch.arange(0, (bsz + 1) * q_len, q_len, dtype=torch.int32, device=hidden_states.device)
        out_tree_r, lse_tree_r = ragged_attention_with_lse(
            Q_r, K_r, V_r, cu, branching, depth, max_seqlen=q_len)

        if bsz == 1:
            out_tree = out_tree_r.transpose(0, 1).unsqueeze(0)
            lse_tree = lse_tree_r.transpose(0, 1).unsqueeze(0)
        else:
            out_tree = out_tree_r.view(bsz, q_len, num_heads, head_dim).permute(0, 2, 1, 3)
            lse_tree = lse_tree_r.view(bsz, q_len, num_heads).permute(0, 2, 1)

        if N_pre == 0:
            attn_output = out_tree.to(hidden_states.dtype)
        else:
            attn_output = fused_lse_merge(lse_pre, lse_tree, out_pre, out_tree)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    LlamaAttention.forward = _ragged_fwd
    try:
        yield
    finally:
        LlamaAttention.forward = _orig_fwd


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic verify-step microbench (no real EAGLE loop needed)
# ─────────────────────────────────────────────────────────────────────────────

def _run_verify_steps(model, branching: int, depth: int, prefix_len: int,
                      n_reps: int, use_ragged: bool, device: torch.device,
                      n_tokens: int = 30,
                      profiler_to_reset: "_LayerProfiler | None" = None) -> float:
    """
    Run n_reps synthetic tree_decoding calls and return total wall ms.

    Uses random tensors so we bypass the EAGLE generation loop entirely;
    the GPU kernel pattern is the same as a real verify step.
    n_tokens: actual tree budget (NOT the full b^d count — use _default_total_token).
    """
    cfg  = model.base_model.config
    H    = cfg.num_attention_heads
    Hkv  = getattr(cfg, "num_key_value_heads", H)
    D    = cfg.hidden_size // H
    L    = cfg.num_hidden_layers
    N    = n_tokens
    hdim = cfg.hidden_size

    # Fabricate hidden states that tree_decoding would receive
    hidden = torch.randn(1, N, hdim, device=device, dtype=torch.float16)

    # Fabricate KV cache: prefix-only (L>0), or None (L=0).
    # The model will compute new K,V from hidden_states and append to past_key_values
    # during attention, giving total context = prefix_len + N (correct pattern).
    if prefix_len > 0:
        pkv = tuple(
            (
                torch.randn(1, Hkv, prefix_len, D, device=device, dtype=torch.float16),
                torch.randn(1, Hkv, prefix_len, D, device=device, dtype=torch.float16),
            )
            for _ in range(L)
        )
    else:
        pkv = None

    pos_ids = torch.arange(N, device=device).unsqueeze(0)

    def _one_forward():
        with torch.no_grad():
            model.base_model(
                inputs_embeds=hidden,
                position_ids=pos_ids,
                past_key_values=pkv,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

    # Warmup (5 iters) — run before timed loop; reset profiler afterward so warmup
    # events don't contaminate per-step averages.
    for _ in range(5):
        _one_forward()
    torch.cuda.synchronize()
    if profiler_to_reset is not None:
        profiler_to_reset.reset()

    ctx = _ragged_ctx(model, branching, depth, prefix_len) if use_ragged else contextlib.nullcontext()
    with ctx:
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _one_forward()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000   # ms

    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Draft-model timing (EAGLE cnets forward)
# ─────────────────────────────────────────────────────────────────────────────

def _run_verify_steps_standalone(base_model, prefix_len: int,
                                  n_reps: int, device: torch.device,
                                  n_tokens: int = 30, hdim: int = 4096,
                                  profiler_to_reset=None) -> float:
    """Like _run_verify_steps but uses a plain AutoModelForCausalLM (no EaModel).

    Used for GPTQ / GGUF models loaded outside of EAGLE.
    """
    cfg  = base_model.config
    Hkv  = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    D    = cfg.hidden_size // cfg.num_attention_heads
    L    = cfg.num_hidden_layers
    N    = n_tokens

    hidden  = torch.randn(1, N, hdim, device=device, dtype=torch.float16)
    pkv     = None
    if prefix_len > 0:
        pkv = tuple(
            (
                torch.randn(1, Hkv, prefix_len, D, device=device, dtype=torch.float16),
                torch.randn(1, Hkv, prefix_len, D, device=device, dtype=torch.float16),
            )
            for _ in range(L)
        )
    pos_ids = torch.arange(N, device=device).unsqueeze(0)

    def _fwd():
        with torch.no_grad():
            base_model(inputs_embeds=hidden, position_ids=pos_ids,
                       past_key_values=pkv, use_cache=False,
                       output_attentions=False, output_hidden_states=False)

    for _ in range(5):
        _fwd()
    torch.cuda.synchronize()
    if profiler_to_reset is not None:
        profiler_to_reset.reset()

    t0 = time.perf_counter()
    for _ in range(n_reps):
        _fwd()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def _time_draft_forward(model, branching: int, depth: int, n_reps: int,
                         device: torch.device, n_tokens: int = 30) -> float:
    """Time the EAGLE draft model (cnets) forward pass for one depth level."""
    N    = n_tokens
    hdim = model.base_model.config.hidden_size

    hidden = torch.randn(1, N, hdim, device=device, dtype=torch.float16)
    inputs = torch.randint(0, 32000, (1, N), device=device)

    ea = model.ea_layer
    for _ in range(5):
        with torch.no_grad():
            _ = ea(hidden, inputs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_reps):
        with torch.no_grad():
            _ = ea(hidden, inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model",   default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--eagle-model",  default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    ap.add_argument("--branching",    type=int, default=4,
                    help="Tree branching factor / top-k (default: 4)")
    ap.add_argument("--depth",        type=int, default=8,
                    help="Tree depth (default: 8)")
    ap.add_argument("--total-tokens", type=int, default=None,
                    help="Actual token budget (overrides formula max(30,round(6*b*d/7))). "
                         "Must match the tree size used in w4a16_e2e.py.")
    ap.add_argument("--prefix-len",   type=int, default=0,
                    help="Simulated KV-cache prefix length L (default: 0 = no prefix)")
    ap.add_argument("--warmup",       type=int, default=10)
    ap.add_argument("--reps",         type=int, default=50,
                    help="Timed verify steps per run (default: 50)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load base model with BitsAndBytes NF4 4-bit quantization")
    ap.add_argument("--no-eagle3",    action="store_true")
    ap.add_argument("--no-eagle",     action="store_true",
                    help="Load base model standalone via AutoModelForCausalLM (no EAGLE needed). "
                         "Required for GPTQ models. Draft timing is skipped.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)
    if not args.no_eagle and not HAS_EAGLE:
        print("EAGLE not installed (use --no-eagle for GPTQ model profiling)."); sys.exit(1)

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    print(f"\n  GPU: {props.name}  SM{props.major}.{props.minor}  "
          f"{props.total_memory // (1 << 30)} GB VRAM")

    quant_tag = "BnB-NF4-4bit" if args.load_in_4bit else "FP16"
    # Use pruned token budget (same formula as w4a16_e2e.py _default_total_token)
    # NOT the full b^d tree which is huge and doesn't match the real benchmark.
    N = args.total_tokens if args.total_tokens is not None else max(30, round(6 * args.branching * args.depth / 7))
    print(f"\n  Config: b={args.branching}  d={args.depth}  N={N} tree-tokens (budget)  "
          f"L={args.prefix_len} prefix  quant={quant_tag}")
    print(f"  Reps: {args.warmup} warmup + {args.reps} timed")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading {args.base_model} ({quant_tag}) ...")
    t0 = time.perf_counter()

    base_model_standalone = None  # set when --no-eagle is used

    if args.no_eagle:
        # Standalone path: load via AutoModelForCausalLM so we can profile GPTQ
        # models without EaModel.  Ragged comparison is skipped (EAGLE not present).
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            _cfg = AutoConfig.from_pretrained(args.base_model)
            quant_cfg = getattr(_cfg, "quantization_config", {})
            if isinstance(quant_cfg, dict) and quant_cfg.get("quant_method") in ("gptq", "awq"):
                quant_tag = f"GPTQ-W{quant_cfg.get('bits',4)}A16-g{quant_cfg.get('group_size',128)}"
            base_model_standalone = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )
            base_model_standalone.eval()
        except Exception as e:
            print(f"  ERROR loading base model: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
        cfg      = base_model_standalone.config
        n_layers = cfg.num_hidden_layers
        H        = cfg.num_attention_heads
        D        = cfg.hidden_size // H
        hdim     = cfg.hidden_size
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
        print(f"  LLM: {n_layers}L  H={H}  D={D}  hidden={hdim}  quant={quant_tag}")
    else:
        try:
            from transformers import BitsAndBytesConfig
            from eagle.model.ea_model import EaModel

            try:
                from eagle.model import cnets as _cnets
                _orig_init_rope = _cnets.LlamaAttention._init_rope
                def _p_rope(self):
                    rs = getattr(self.config, "rope_scaling", None)
                    if isinstance(rs, dict) and rs.get("type","") not in {"linear","dynamic"}:
                        import copy
                        self.config = copy.deepcopy(self.config)
                        self.config.rope_scaling = None
                    _orig_init_rope(self)
                _cnets.LlamaAttention._init_rope = _p_rope
            except Exception:
                pass

            model = EaModel.from_pretrained(
                use_eagle3=not args.no_eagle3,
                base_model_path=args.base_model,
                ea_model_path=args.eagle_model,
                total_token=N - 1,
                depth=args.depth,
                top_k=args.branching,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                load_in_4bit=args.load_in_4bit,
            )
            model.eval()
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            sys.exit(1)

        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
        cfg  = model.base_model.config
        H    = cfg.num_attention_heads
        D    = cfg.hidden_size // H
        hdim = cfg.hidden_size
        n_layers = cfg.num_hidden_layers
        print(f"  LLM: {n_layers}L  H={H}  D={D}  hidden={hdim}")

    # Resolve which base model object to use for layer hooking / forward calls
    _base = base_model_standalone if args.no_eagle else model.base_model

    # ── Triton warmup (skip in --no-eagle mode, ragged is not patched) ─────────
    if not args.no_eagle:
        print(f"\n  [warmup] Compiling Triton ragged kernel for b={args.branching} d={args.depth} N={N} ...")
        _ = _run_verify_steps(model, args.branching, args.depth, args.prefix_len,
                              n_reps=args.warmup, use_ragged=True, device=device, n_tokens=N)
        _ = _run_verify_steps(model, args.branching, args.depth, args.prefix_len,
                              n_reps=args.warmup, use_ragged=False, device=device, n_tokens=N)
    else:
        # Vanilla-only warmup so the GPTQ kernels are JIT-compiled/cached
        print(f"\n  [warmup] {args.warmup} warmup steps (GPTQ model, no ragged) ...")
        _ = _run_verify_steps_standalone(_base, args.prefix_len,
                                         n_reps=args.warmup, device=device, n_tokens=N, hdim=hdim)
    torch.cuda.empty_cache()

    # ── Attach layer profiler ────────────────────────────────────────────────
    prof_vanilla = _LayerProfiler()
    prof_ragged  = _LayerProfiler()

    # ── Vanilla timed run ─────────────────────────────────────────────────────
    print(f"\n  [vanilla] {args.reps} timed steps (SDPA attention) ...")
    prof_vanilla.attach(_base)
    if args.no_eagle:
        vanilla_ms = _run_verify_steps_standalone(_base, args.prefix_len,
                                                  n_reps=args.reps, device=device, n_tokens=N,
                                                  hdim=hdim, profiler_to_reset=prof_vanilla)
    else:
        vanilla_ms = _run_verify_steps(model, args.branching, args.depth, args.prefix_len,
                                       n_reps=args.reps, use_ragged=False, device=device, n_tokens=N,
                                       profiler_to_reset=prof_vanilla)
    prof_vanilla.detach()
    vanilla_step = vanilla_ms / args.reps

    # ── Ragged timed run (skip for --no-eagle) ────────────────────────────────
    ragged_step = None
    if not args.no_eagle:
        print(f"  [ragged]  {args.reps} timed steps (Triton ragged kernel) ...")
        prof_ragged.attach(_base)
        ragged_ms = _run_verify_steps(model, args.branching, args.depth, args.prefix_len,
                                      n_reps=args.reps, use_ragged=True, device=device, n_tokens=N,
                                      profiler_to_reset=prof_ragged)
        prof_ragged.detach()
        ragged_step = ragged_ms / args.reps

    # ── Draft model timing (skip for --no-eagle) ──────────────────────────────
    draft_step = None
    if not args.no_eagle:
        print(f"  [draft]   {args.reps} timed steps (EAGLE cnets) ...")
        draft_ms   = _time_draft_forward(model, args.branching, args.depth,
                                          n_reps=args.reps, device=device, n_tokens=N)
        draft_step = draft_ms / args.reps

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  Verify Step Breakdown — {quant_tag}  "
          f"b={args.branching}  d={args.depth}  N={N}  L={args.prefix_len}")
    print(f"{'═' * 70}")

    print(prof_vanilla.summary(args.reps, n_layers, "VANILLA"))
    if ragged_step is not None:
        print()
        print(prof_ragged.summary(args.reps, n_layers, "RAGGED"))

    # ── Summary table ─────────────────────────────────────────────────────────
    def _ms(v): return f"{v:.3f} ms" if v is not None else "    n/a"
    def _pct(v, tot): return f"{v / tot * 100:.1f}%" if tot else "n/a"

    attn_v  = prof_vanilla.attn_ms  / args.reps
    mlp_v   = prof_vanilla.mlp_ms   / args.reps
    total_v = prof_vanilla.total_ms / args.reps
    attn_r  = (prof_ragged.attn_ms  / args.reps) if ragged_step is not None else None
    mlp_r   = (prof_ragged.mlp_ms   / args.reps) if ragged_step is not None else None

    print(f"\n  {'─' * 60}")
    print(f"  {'':30s}  {'vanilla':>10}  {'ragged':>10}  {'delta':>10}")
    print(f"  {'─' * 30}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    print(f"  {'verify_ms (wall)':<30}  {_ms(vanilla_step):>10}  {_ms(ragged_step):>10}"
          + (f"  {ragged_step/vanilla_step:>9.3f}×" if ragged_step else ""))
    print(f"  {'verify_ms (profiler sum)':<30}  {_ms(total_v):>10}")
    print(f"  {'attn_ms (all layers)':<30}  {_ms(attn_v):>10}  {_ms(attn_r):>10}"
          f"  {_pct(attn_v, total_v):>10}")
    print(f"  {'mlp_ms  (all layers)':<30}  {_ms(mlp_v):>10}  {_ms(mlp_r):>10}"
          f"  {_pct(mlp_v, total_v):>10}")
    if draft_step is not None:
        print(f"  {'draft_ms (EAGLE cnets fwd)':<30}  {_ms(draft_step):>10}  {'(same)':>10}")
    print(f"  {'─' * 30}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    if ragged_step is not None and attn_r:
        print(f"  {'attn speedup (ragged/vanilla)':<30}  {'':10}  {'':10}"
              f"  {attn_v/attn_r:>9.3f}×")
    if ragged_step is not None and draft_step is not None:
        total_v_wall = vanilla_step + draft_step
        total_r_wall = ragged_step  + draft_step
        print(f"  {'E2E speedup estimate':<30}")
        print(f"    vanilla_total = verify({vanilla_step:.2f}) + draft({draft_step:.2f})"
              f" = {total_v_wall:.2f} ms/step")
        print(f"    ragged_total  = verify({ragged_step:.2f}) + draft({draft_step:.2f})"
              f" = {total_r_wall:.2f} ms/step")
        if total_v_wall > 0:
            print(f"    E2E speedup   = {total_v_wall / total_r_wall:.3f}×  "
                  f"({'win' if total_v_wall > total_r_wall else 'LOSS'})")

    print(f"\n  {'─' * 60}")
    print(f"  Amdahl analysis ({quant_tag}):")
    attn_frac_v = attn_v / total_v if total_v else 0
    mlp_frac_v  = mlp_v  / total_v if total_v else 0
    print(f"    verify breakdown:  attn={attn_frac_v:.1%}  mlp={mlp_frac_v:.1%}")

    # Project W4A16 impact if currently measuring FP16
    if not args.load_in_4bit and not args.no_eagle:
        mlp_speedup_est = 1.6
        new_mlp = mlp_v / mlp_speedup_est
        new_total = attn_v + new_mlp + (total_v - attn_v - mlp_v)
        new_attn_frac = attn_v / new_total
        print(f"    If W4A16 Marlin applied (est {mlp_speedup_est:.1f}× MLP speedup):")
        print(f"      attn fraction: {attn_frac_v:.1%} → {new_attn_frac:.1%}")
        if draft_step and ragged_step:
            verify_saving = vanilla_step - ragged_step
            w4a16_vanilla_step = total_v - mlp_v + new_mlp
            w4a16_total_v = w4a16_vanilla_step + draft_step
            w4a16_ragged_step  = w4a16_vanilla_step - verify_saving
            w4a16_total_r = max(w4a16_ragged_step, 0.01) + draft_step
            if w4a16_total_v > 0:
                print(f"      Projected E2E speedup with W4A16+ragged: "
                      f"{w4a16_total_v / w4a16_total_r:.3f}×")
    elif args.no_eagle:
        # Actually measured with GPTQ — show ragged opportunity
        ragged_saving_est = attn_v * 0.24   # 24% from FP16 profiler
        new_verify = total_v - ragged_saving_est
        print(f"    Ragged kernel 24% attn saving → saves {ragged_saving_est:.2f}ms/step")
        print(f"    Projected verify: {total_v:.2f}ms → {new_verify:.2f}ms  "
              f"({new_verify/total_v:.3f}× speedup within verify)")
        print(f"    NOTE: draft model (EAGLE cnets) also runs in FP16 → additional floor")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
