#!/usr/bin/env python3
"""
benchmark_70b.py
================
Profiles ragged vs. vanilla attention for Llama-3-70B, in both BF16 and
W4A16-GPTQ-Marlin modes, across a sweep of prefix lengths.

For each (mode, prefix_len) we measure per-verify-step:
  attn_ms   — QKV proj + flash attention + O proj   (summed over all layers)
  mlp_ms    — gate + up + down proj                 (summed over all layers)
  ragged_ms — same attn block with ragged ancestor kernel

Then we project to E2E step time using a draft-time estimate (from either
a measured EAGLE 70B draft or the --draft-ms flag) and report Amdahl speedup.

Modes
-----
  --mode bf16
      Loads meta-llama/Meta-Llama-3-70B-Instruct (BF16) with device_map="auto".
      Requires an EAGLE 70B draft model (--eagle-model).
      Needs ~140 GB VRAM (2× A100 80 GB recommended).

  --mode w4a16
      Loads a W4A16 GPTQ-Marlin quantized 70B model via llamacu.
      Requires --base-model and --eagle-model (W4A16-Rot variants).
      Fits on a single A100 80 GB.

  --mode synthetic
      No model download required.  Creates a single LlamaDecoderLayer with
      70B config (or custom --hidden-size etc.) and times it with random weights.
      Per-layer result is scaled by --num-layers (default 80).
      Useful for kernel-timing studies when the full model is unavailable.

Usage examples
--------------
  # Synthetic kernel study (no download needed)
  python scripts/benchmark_70b.py --mode synthetic

  # BF16 full model (2× A100 80 GB)
  python scripts/benchmark_70b.py --mode bf16 \\
      --base-model  meta-llama/Meta-Llama-3-70B-Instruct \\
      --eagle-model yuhuili/EAGLE3-LLaMA3.1-Instruct-70B

  # W4A16 GPTQ-Marlin (single A100 80 GB)
  python scripts/benchmark_70b.py --mode w4a16 \\
      --base-model  /path/to/Meta-Llama-3-70B-Instruct-W4A16-g128-Rot \\
      --eagle-model /path/to/EAGLE-LLaMA3-Instruct-70B-on-W4A16-Rot \\
      --llamacu-path /path/to/SpecMQuant
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragged_attn import ragged_attention_with_lse, fused_lse_merge

# ─────────────────────────────────────────────────────────────────────────────
# Defaults for 70B Llama-3
# ─────────────────────────────────────────────────────────────────────────────
LLAMA_70B = dict(
    hidden_size=8192,
    intermediate_size=28672,
    num_attention_heads=64,
    num_key_value_heads=8,
    num_hidden_layers=80,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    max_position_embeddings=131072,
    vocab_size=128256,
)

PREFIX_SWEEP = [128, 256, 512, 1024, 2048, 4096]

# ─────────────────────────────────────────────────────────────────────────────
# CUDA-event layer profiler (same design as profile_verify.py)
# ─────────────────────────────────────────────────────────────────────────────
class _LayerProfiler:
    def __init__(self):
        self.reset()
        self._hooks: list = []

    def reset(self):
        self.norm_ms = self.attn_ms = self.mlp_ms = self.total_ms = 0.0
        self.n_calls = 0

    def _make_hooks(self, layer: nn.Module):
        evts: List[torch.cuda.Event] = []

        def _pre_layer(mod, inp):
            nonlocal evts
            evts = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            evts[0].record()

        def _pre_attn(mod, inp):
            evts[1].record()

        def _post_attn(mod, inp, out):
            evts[2].record()

        def _pre_mlp(mod, inp):
            evts[3].record()

        def _post_layer(mod, inp, out):
            evts[4].record()
            torch.cuda.synchronize()
            self.norm_ms  += evts[0].elapsed_time(evts[1]) + evts[2].elapsed_time(evts[3])
            self.attn_ms  += evts[1].elapsed_time(evts[2])
            self.mlp_ms   += evts[3].elapsed_time(evts[4])
            self.total_ms += evts[0].elapsed_time(evts[4])
            self.n_calls  += 1

        self._hooks += [
            layer.register_forward_pre_hook(_pre_layer),
            layer.self_attn.register_forward_pre_hook(_pre_attn),
            layer.self_attn.register_forward_hook(_post_attn),
            layer.mlp.register_forward_pre_hook(_pre_mlp),
            layer.register_forward_hook(_post_layer),
        ]

    def attach(self, base_model):
        layers = base_model.model.layers
        for layer in layers:
            self._make_hooks(layer)

    def attach_single(self, layer: nn.Module):
        self._make_hooks(layer)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def per_step(self, n_steps: int):
        """Return (norm, attn, mlp, total) ms per verify step."""
        s = max(n_steps, 1)
        return (self.norm_ms / s, self.attn_ms / s,
                self.mlp_ms / s, self.total_ms / s)


# ─────────────────────────────────────────────────────────────────────────────
# Ragged context manager — patches a HF LlamaAttention for the tree portion
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _ragged_ctx(layer_or_model, branching: int, depth: int, n_prefix: int,
                dtype: torch.dtype = torch.bfloat16):
    """Monkey-patch self_attn.forward to use the ragged Triton kernel."""
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except ImportError:
        yield; return

    _orig = LlamaAttention.forward
    scale = 1.0 / math.sqrt(LLAMA_70B["hidden_size"] //
                             LLAMA_70B["num_attention_heads"])

    def _ragged_fwd(self, hidden_states, attention_mask=None,
                    position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, **kw):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        nh  = self.num_heads
        nkv = self.num_key_value_heads
        d   = self.head_dim

        q = q.view(bsz, q_len, nh,  d).transpose(1, 2)
        k = k.view(bsz, q_len, nkv, d).transpose(1, 2)
        v = v.view(bsz, q_len, nkv, d).transpose(1, 2)

        # RoPE
        try:
            cos, sin = self.rotary_emb(v, position_ids=position_ids)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        except Exception:
            pass

        # Append prefix KV
        if past_key_value is not None:
            pk, pv = past_key_value
            if isinstance(pk, torch.Tensor):
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
        past_key_value = (k[:, :, :n_prefix, :], v[:, :, :n_prefix, :])

        # Expand GQA
        r = nh // nkv
        k_exp = k.repeat_interleave(r, dim=1)
        v_exp = v.repeat_interleave(r, dim=1)

        N_kv = k_exp.shape[2]
        N_pre = N_kv - q_len

        # Prefix attention
        if N_pre > 0:
            K_pre = k_exp[:, :, :N_pre, :].contiguous()
            V_pre = v_exp[:, :, :N_pre, :].contiguous()
            try:
                out_pre, lse_pre, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q.contiguous(), K_pre, V_pre,
                    dropout_p=0.0, is_causal=False, scale=scale,
                    return_debug_mask=False)
            except Exception:
                sc = (q @ K_pre.transpose(-2, -1)) * scale
                lse_pre = torch.logsumexp(sc.float(), dim=-1)
                out_pre = torch.softmax(sc, dim=-1) @ V_pre

        # Ragged tree attention
        Q_r = q.squeeze(0).transpose(0, 1).contiguous()
        K_r = k_exp[:, :, N_pre:, :].squeeze(0).transpose(0, 1).contiguous()
        V_r = v_exp[:, :, N_pre:, :].squeeze(0).transpose(0, 1).contiguous()
        cu  = torch.arange(0, q_len + 1, dtype=torch.int32, device=q.device)
        out_tree_r, lse_tree_r = ragged_attention_with_lse(
            Q_r, K_r, V_r, cu, branching, depth, max_seqlen=q_len)
        out_tree = out_tree_r.transpose(0, 1).unsqueeze(0)
        lse_tree = lse_tree_r.transpose(0, 1).unsqueeze(0)

        if N_pre == 0:
            attn_out = out_tree.to(dtype)
        else:
            attn_out = fused_lse_merge(lse_pre, lse_tree, out_pre, out_tree)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, nh * d)
        attn_out = self.o_proj(attn_out)
        return attn_out, None, past_key_value

    LlamaAttention.forward = _ragged_fwd
    try:
        yield
    finally:
        LlamaAttention.forward = _orig


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic single-layer benchmark
# ─────────────────────────────────────────────────────────────────────────────
def _build_synthetic_layer(cfg_dict: dict, dtype: torch.dtype,
                            device: torch.device) -> nn.Module:
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    cfg = LlamaConfig(**cfg_dict)
    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(dtype=dtype, device=device)
    layer.eval()
    return layer


def _run_synthetic(
    layer: nn.Module,
    n_tokens: int,
    prefix_len: int,
    n_reps: int,
    branching: int,
    depth: int,
    dtype: torch.dtype,
    device: torch.device,
    use_ragged: bool,
    prof: _LayerProfiler,
) -> float:
    H   = LLAMA_70B["hidden_size"]
    Hkv = LLAMA_70B["num_key_value_heads"]
    D   = H // LLAMA_70B["num_attention_heads"]

    hidden = torch.randn(1, n_tokens, H, dtype=dtype, device=device)
    pos_ids = torch.arange(prefix_len, prefix_len + n_tokens,
                           device=device, dtype=torch.long).unsqueeze(0)
    pk = torch.randn(1, Hkv, prefix_len, D, dtype=dtype, device=device)
    pv = torch.randn_like(pk)
    past_kv = (pk, pv)

    def _fwd():
        with torch.no_grad():
            layer(hidden_states=hidden,
                  position_ids=pos_ids,
                  past_key_value=past_kv,
                  use_cache=False)

    # Warmup
    for _ in range(5):
        _fwd()
    torch.cuda.synchronize()
    prof.reset()

    ctx = (_ragged_ctx(layer, branching, depth, prefix_len, dtype)
           if use_ragged else contextlib.nullcontext())
    with ctx:
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _fwd()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3  # ms


# ─────────────────────────────────────────────────────────────────────────────
# Full HF model benchmark (BF16, multi-GPU via device_map="auto")
# ─────────────────────────────────────────────────────────────────────────────
def _run_hf_model(
    base_model_path: str,
    eagle_model_path: str,
    n_tokens: int,
    prefix_sweep: List[int],
    n_reps: int,
    branching: int,
    depth: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Load a HF EAGLE+LLaMA model and profile for each prefix in sweep."""
    print(f"  Loading {base_model_path} ...")
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
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=eagle_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    cfg = model.base_model.config
    H   = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", H)
    D   = cfg.hidden_size // H
    L   = cfg.num_hidden_layers

    results = {}
    for prefix_len in prefix_sweep:
        print(f"  prefix={prefix_len} ...", end="", flush=True)
        hidden = torch.randn(1, n_tokens, cfg.hidden_size, dtype=dtype, device=device)
        pos_ids = torch.arange(n_tokens, device=device).unsqueeze(0)
        pkv = tuple(
            (torch.randn(1, Hkv, prefix_len, D, dtype=dtype, device=device),
             torch.randn(1, Hkv, prefix_len, D, dtype=dtype, device=device))
            for _ in range(L)
        )

        def _fwd():
            with torch.no_grad():
                model.base_model(inputs_embeds=hidden, position_ids=pos_ids,
                                 past_key_values=pkv, use_cache=False,
                                 output_attentions=False, output_hidden_states=False)

        prof_v = _LayerProfiler(); prof_r = _LayerProfiler()
        prof_v.attach(model.base_model)

        # Warmup
        for _ in range(5): _fwd()
        torch.cuda.synchronize()
        prof_v.reset()

        t0 = time.perf_counter()
        for _ in range(n_reps): _fwd()
        torch.cuda.synchronize()
        vanilla_ms = (time.perf_counter() - t0) * 1e3
        prof_v.detach()

        prof_r.attach(model.base_model)
        torch.cuda.synchronize(); prof_r.reset()
        with _ragged_ctx(model.base_model, branching, depth, prefix_len, dtype):
            t0 = time.perf_counter()
            for _ in range(n_reps): _fwd()
            torch.cuda.synchronize()
            ragged_ms = (time.perf_counter() - t0) * 1e3
        prof_r.detach()

        _, attn_v, mlp_v, total_v = prof_v.per_step(n_reps)
        _, attn_r, mlp_r, total_r = prof_r.per_step(n_reps)
        results[prefix_len] = dict(vanilla_ms=vanilla_ms / n_reps,
                                   ragged_ms=ragged_ms / n_reps,
                                   attn_v=attn_v, mlp_v=mlp_v, total_v=total_v,
                                   attn_r=attn_r, mlp_r=mlp_r, total_r=total_r)
        print(f" vanilla={vanilla_ms/n_reps:.1f}ms  ragged={ragged_ms/n_reps:.1f}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# W4A16 llamacu benchmark
# ─────────────────────────────────────────────────────────────────────────────
def _run_w4a16(args, n_tokens: int, prefix_sweep: List[int],
               n_reps: int, branching: int, depth: int) -> dict:
    import sys as _sys
    if args.llamacu_path:
        _sys.path.insert(0, args.llamacu_path)

    import os as _os
    _lib_dir  = _os.path.join(_sys.prefix, "lib")
    _torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
    _ld = f"{_lib_dir}:{_torch_lib}"
    _os.environ["LD_LIBRARY_PATH"] = _ld + ":" + _os.environ.get("LD_LIBRARY_PATH", "")

    from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import (
        W4A16GPTQMarlinLLM_with_eagle,
    )
    from llamacu import C

    print(f"  Loading W4A16 model ...")
    model = W4A16GPTQMarlinLLM_with_eagle(
        eagle_path=args.eagle_model,
        base_path=args.base_model,
        memory_limit=args.memory_limit,
        rotation=True,
    )
    model.init_storage()
    model.load_from_hf()
    print(f"  Model loaded.")

    device = torch.device("cuda:0")
    n_tok  = n_tokens

    # Bootstrap KV cache
    dummy_prompt = torch.arange(64, dtype=torch.int32, device=device)
    pos = torch.arange(64, dtype=torch.int32, device=device)
    logits = model.prefill(dummy_prompt, pos)
    model.tree_draft_ids[:1].copy_(logits[0].argmax(-1))
    model.cache_length[0] = 64

    results = {}
    for prefix_len in prefix_sweep:
        print(f"  prefix={prefix_len} ...", end="", flush=True)
        model.cache_length[0] = prefix_len

        e_draft = torch.cuda.Event(enable_timing=True)
        e_decode = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        draft_ms_acc = decode_ms_acc = 0.0
        for i in range(5 + n_reps):
            e_draft.record()
            C.draft(model.tree_draft_ids.data_ptr(),
                    model.tree_position_ids.data_ptr(),
                    model.cache_length.data_ptr(),
                    model.tree_attn_mask.data_ptr(),
                    model.tree_parent.data_ptr())
            e_decode.record()
            model.decode(model.tree_draft_ids, model.tree_position_ids,
                         model.cache_length, mask_2d=model.tree_attn_mask)
            e_end.record()
            torch.cuda.synchronize()
            if i >= 5:
                draft_ms_acc  += e_draft.elapsed_time(e_decode)
                decode_ms_acc += e_decode.elapsed_time(e_end)

        results[prefix_len] = dict(
            draft_ms=draft_ms_acc / n_reps,
            decode_ms=decode_ms_acc / n_reps,
        )
        print(f" draft={results[prefix_len]['draft_ms']:.1f}ms "
              f"decode={results[prefix_len]['decode_ms']:.1f}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def _print_header(mode: str, n_tokens: int, branching: int, depth: int,
                  n_layers: int):
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  Llama-3-70B  mode={mode.upper()}  N={n_tokens} tokens  "
          f"b={branching} d={depth}  layers={n_layers}")
    print(sep)


def _print_sweep_table(results: dict, n_layers: int, draft_ms: Optional[float],
                       label: str = "VANILLA vs RAGGED"):
    print(f"\n  {label}")
    hdr = (f"  {'prefix':>6}  {'attn_v':>7}  {'mlp_v':>7}  "
           f"{'attn%':>6}  {'attn_r':>7}  {'saving%':>7}  "
           f"{'verify_v':>8}  {'verify_r':>8}  {'E2E_spd':>7}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for prefix, r in sorted(results.items()):
        av   = r["attn_v"]
        mlv  = r["mlp_v"]
        tv   = r["total_v"]
        ar   = r["attn_r"]
        tr   = r["total_r"]

        # Scale to full model
        av80  = av  * n_layers
        mlv80 = mlv * n_layers
        tv80  = tv  * n_layers
        ar80  = ar  * n_layers
        tr80  = tr  * n_layers

        attn_pct   = av80 / tv80 * 100 if tv80 else 0
        saving_pct = (av80 - ar80) / av80 * 100 if av80 else 0

        if draft_ms is not None:
            step_v = draft_ms + tv80
            step_r = draft_ms + tr80
            spd = step_v / step_r if step_r else 1.0
            spd_str = f"{spd:.4f}×"
        else:
            spd_str = "    n/a"

        print(f"  {prefix:>6}  {av80:>7.2f}  {mlv80:>7.2f}  "
              f"{attn_pct:>5.1f}%  {ar80:>7.2f}  {saving_pct:>6.1f}%  "
              f"{tv80:>8.2f}  {tr80:>8.2f}  {spd_str:>7}")


def _print_w4a16_table(results: dict):
    print(f"\n  W4A16 step breakdown")
    hdr = f"  {'prefix':>6}  {'draft_ms':>9}  {'decode_ms':>10}  {'total_ms':>9}  {'draft%':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for prefix, r in sorted(results.items()):
        d = r["draft_ms"]; dec = r["decode_ms"]; tot = d + dec
        print(f"  {prefix:>6}  {d:>9.2f}  {dec:>10.2f}  {tot:>9.2f}  "
              f"{d/tot*100:>6.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--mode", choices=["bf16", "w4a16", "synthetic"],
                    default="synthetic")
    ap.add_argument("--base-model",  type=str, default=None,
                    help="HF repo or local path to the base LLaMA model")
    ap.add_argument("--eagle-model", type=str, default=None,
                    help="HF repo or local path to the EAGLE draft model")
    ap.add_argument("--llamacu-path", type=str, default=None,
                    help="Path to SpecMQuant repo root (for W4A16 mode)")
    ap.add_argument("--memory-limit", type=float, default=0.85,
                    help="GPU memory fraction for llamacu pool (W4A16 mode)")
    ap.add_argument("--prefix-sweep", type=int, nargs="+", default=PREFIX_SWEEP,
                    help="Prefix lengths to benchmark")
    ap.add_argument("--n-tokens",  type=int, default=60,
                    help="Tree token budget (default: 60)")
    ap.add_argument("--branching", type=int, default=10)
    ap.add_argument("--depth",     type=int, default=7)
    ap.add_argument("--reps",      type=int, default=100,
                    help="Timed repetitions per prefix (default: 100)")
    ap.add_argument("--draft-ms",  type=float, default=None,
                    help="Override draft step time in ms for Amdahl projection. "
                         "If omitted: synthetic=None, bf16 measured, w4a16 measured.")
    ap.add_argument("--num-layers", type=int, default=80,
                    help="Number of transformer layers (synthetic mode only)")
    # Synthetic-mode overrides
    ap.add_argument("--hidden-size",       type=int, default=LLAMA_70B["hidden_size"])
    ap.add_argument("--intermediate-size", type=int, default=LLAMA_70B["intermediate_size"])
    ap.add_argument("--num-heads",         type=int, default=LLAMA_70B["num_attention_heads"])
    ap.add_argument("--num-kv-heads",      type=int, default=LLAMA_70B["num_key_value_heads"])

    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    print(f"\n  GPU: {props.name}  SM{props.major}.{props.minor}  "
          f"{props.total_memory >> 30} GB VRAM")
    print(f"  mode={args.mode}  N={args.n_tokens}  "
          f"b={args.branching}  d={args.depth}  reps={args.reps}")

    n_tokens = args.n_tokens
    branching = args.branching
    depth = args.depth

    # ── Synthetic mode ────────────────────────────────────────────────────────
    if args.mode == "synthetic":
        dtype  = torch.bfloat16
        n_lay  = args.num_layers
        cfg_dict = dict(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_heads,
            num_key_value_heads=args.num_kv_heads,
            num_hidden_layers=1,
            rms_norm_eps=LLAMA_70B["rms_norm_eps"],
            rope_theta=LLAMA_70B["rope_theta"],
            max_position_embeddings=LLAMA_70B["max_position_embeddings"],
            vocab_size=LLAMA_70B["vocab_size"],
        )
        print(f"\n  [Synthetic] single LlamaDecoderLayer, results ×{n_lay}")
        print(f"  Config: hidden={args.hidden_size} inter={args.intermediate_size} "
              f"heads={args.num_heads} kv_heads={args.num_kv_heads}")

        layer = _build_synthetic_layer(cfg_dict, dtype, device)
        weight_mb = sum(p.numel() * p.element_size()
                        for p in layer.parameters()) / 1e6
        print(f"  Single layer: {weight_mb:.0f} MB weights on GPU")

        prof_v = _LayerProfiler(); prof_r = _LayerProfiler()
        prof_v.attach_single(layer); prof_r.attach_single(layer)

        results = {}
        for prefix_len in args.prefix_sweep:
            print(f"  prefix={prefix_len} ... vanilla", end="", flush=True)
            _run_synthetic(layer, n_tokens, prefix_len, args.reps,
                           branching, depth, dtype, device,
                           use_ragged=False, prof=prof_v)
            print(f"  ragged", end="", flush=True)
            _run_synthetic(layer, n_tokens, prefix_len, args.reps,
                           branching, depth, dtype, device,
                           use_ragged=True, prof=prof_r)

            _, attn_v, mlp_v, total_v = prof_v.per_step(args.reps)
            _, attn_r, mlp_r, total_r = prof_r.per_step(args.reps)
            results[prefix_len] = dict(attn_v=attn_v, mlp_v=mlp_v, total_v=total_v,
                                       attn_r=attn_r, mlp_r=mlp_r, total_r=total_r)
            prof_v.reset(); prof_r.reset()
            print(f"  attn={attn_v*n_lay:.1f}ms  mlp={mlp_v*n_lay:.1f}ms  "
                  f"ragged_saving={(attn_v-attn_r)/attn_v*100:.1f}%")

        _print_header("synthetic-bf16", n_tokens, branching, depth, n_lay)
        _print_sweep_table(results, n_lay, args.draft_ms,
                           label="VANILLA vs RAGGED  (per-layer ×N_layers projected)")
        _note_draft(args.draft_ms)

    # ── BF16 full-model mode ──────────────────────────────────────────────────
    elif args.mode == "bf16":
        if not args.base_model or not args.eagle_model:
            print("--base-model and --eagle-model required for bf16 mode")
            sys.exit(1)
        dtype = torch.bfloat16
        results = _run_hf_model(args.base_model, args.eagle_model,
                                 n_tokens, args.prefix_sweep, args.reps,
                                 branching, depth, dtype, device)
        cfg = next(iter(results.values()))
        n_lay = LLAMA_70B["num_hidden_layers"]
        _print_header("bf16", n_tokens, branching, depth, n_lay)
        _print_sweep_table(results, 1, args.draft_ms,
                           label="VANILLA vs RAGGED  (full model, all layers)")
        _note_draft(args.draft_ms)

    # ── W4A16 mode ────────────────────────────────────────────────────────────
    elif args.mode == "w4a16":
        if not args.base_model or not args.eagle_model:
            print("--base-model and --eagle-model required for w4a16 mode")
            sys.exit(1)
        results = _run_w4a16(args, n_tokens, args.prefix_sweep,
                             args.reps, branching, depth)
        _print_header("w4a16", n_tokens, branching, depth, LLAMA_70B["num_hidden_layers"])
        _print_w4a16_table(results)

        # Also run BF16 single-layer for the attn/MLP split since llamacu
        # doesn't expose per-phase timing natively.
        print("\n  [Supplement] BF16 single-layer attn/MLP split for Amdahl estimate:")
        dtype  = torch.bfloat16
        n_lay  = LLAMA_70B["num_hidden_layers"]
        layer  = _build_synthetic_layer(LLAMA_70B | {"num_hidden_layers": 1},
                                         dtype, device)
        prof_v = _LayerProfiler(); prof_v.attach_single(layer)
        synth = {}
        for prefix_len in args.prefix_sweep:
            print(f"  prefix={prefix_len} ...", end="", flush=True)
            _run_synthetic(layer, n_tokens, prefix_len, args.reps,
                           branching, depth, dtype, device,
                           use_ragged=False, prof=prof_v)
            _, attn_v, mlp_v, total_v = prof_v.per_step(args.reps)
            synth[prefix_len] = dict(attn_frac=attn_v / total_v if total_v else 0,
                                     mlp_frac=mlp_v / total_v if total_v else 0)
            prof_v.reset()
            print(f" attn={attn_v*n_lay:.1f}ms ({attn_v/total_v*100:.0f}%)  "
                  f"mlp={mlp_v*n_lay:.1f}ms ({mlp_v/total_v*100:.0f}%)")

        print(f"\n  W4A16 Amdahl projection (ragged 24% attn saving assumed):")
        hdr2 = f"  {'prefix':>6}  {'draft_ms':>9}  {'decode_ms':>10}  " \
               f"{'attn%':>6}  {'saving_ms':>9}  {'E2E_spd':>8}"
        print(hdr2)
        print("  " + "─" * (len(hdr2) - 2))
        for prefix, r in sorted(results.items()):
            draft = r["draft_ms"]; dec = r["decode_ms"]
            afrac = synth.get(prefix, {}).get("attn_frac", 0.26)
            saving = dec * afrac * 0.24
            step_v = draft + dec
            step_r = draft + dec - saving
            spd = step_v / step_r if step_r else 1.0
            print(f"  {prefix:>6}  {draft:>9.2f}  {dec:>10.2f}  "
                  f"{afrac*100:>5.0f}%  {saving:>9.2f}  {spd:>8.4f}×")


def _note_draft(draft_ms: Optional[float]):
    print()
    if draft_ms is not None:
        print(f"  Draft time used for Amdahl: {draft_ms:.1f} ms (--draft-ms override)")
    else:
        print("  Draft time not provided — pass --draft-ms <value> for E2E projection.")
        print("  Tip: for 70B EAGLE (BF16), ~120 ms on A100 is a reasonable estimate.")
        print("       for W4A16 mode, draft is measured directly.")
    print()


if __name__ == "__main__":
    main()
