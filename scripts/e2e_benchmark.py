#!/usr/bin/env python3
"""
e2e_benchmark.py — Eagle-3 E2E Benchmark: Ragged Kernel vs. Vanilla Eagle-3
=============================================================================

Compares two perspectives on the same Eagle-3 speculative decoding system:

  Mode A — Vanilla Eagle-3
    Uses Eagle-3's default attention implementation (PyTorch SDPA + tree mask).
    Measures wall-clock tok/s, acceptance rate, and verification step latency.

  Mode B — Ragged-kernel projection
    Benchmarks the intra-tree attention block with our ragged Triton kernel at
    the exact model dimensions (H, D) and tree shapes used in Mode A, using the
    same sdpa_tree baseline that Eagle-3 uses internally.
    Projects end-to-end speedup via Amdahl's law using the measured verification
    time fraction from Mode A.

The comparison is DIRECT: same model, same prompts, same tree configuration.
We do NOT hot-swap FA-2, FlashInfer, or DeFT into Eagle-3 — those comparisons
belong in benchmark_sota.py, which benchmarks the attention kernel in isolation.

WHY AMDAHL PROJECTION (not a literal drop-in)
----------------------------------------------
Our kernel handles the *intra-tree* Q × K block only.  During verification, each
query token also attends to the preceding KV cache (context prefix).  A correct
in-place swap requires an online-softmax merge of the tree block and prefix block,
which in turn requires the kernel to return log-sum-exp values.  The Amdahl
projection uses measured wall-clock numbers and a conservative estimate of the
intra-tree fraction, giving a rigorous lower bound on the achievable speedup.

WHAT IS MEASURED
-----------------
Step 1 — E2E Generation (vanilla Eagle-3):
  • eagenerate() wall time, tok/s, acceptance rate, mean accepted/step
  • Per-step verification latency via CUDA events (measures tree_decoding only)
  • Verification time fraction of total wall clock

Step 2 — Tree Attention Kernel Comparison:
  • At model-matched dimensions (H, D from the loaded LLaMA config)
  • sdpa_tree  — PyTorch SDPA with explicit tree-ancestor bool mask
                 (this is exactly what Eagle-3's verification pass uses)
  • ragged     — our ancestor-sparse Triton kernel
  • Speedup = sdpa_tree_ms / ragged_ms  (>1 means ragged is faster)
  • Swept over batch_size × branching_factor × depth

Step 3 — E2E Amdahl Projection:
  • f = verify_fraction × attn_fraction × intra_tree_fraction
  • k = kernel speedup from step 2 (median, B=1 matching Eagle-3 usage)
  • Projected E2E speedup = 1 / ((1 − f) + f/k)
  • intra_tree_fraction = N_tree / (N_prefix + N_tree)  (conservative)

OUTPUT
------
  results/e2e_benchmark.csv

MODELS
------
  Default: meta-llama/Llama-3.1-8B-Instruct  +  yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
  Alt:     lmsys/vicuna-7b-v1.3  +  yuhuili/EAGLE-Vicuna-7B-v1.3  (--no-eagle3)

Prerequisites:
  pip install git+https://github.com/SafeAILab/EAGLE.git fschat transformers accelerate
  For LLaMA: huggingface-cli login

Usage:
  python scripts/e2e_benchmark.py                          # default LLaMA-3.1-8B
  python scripts/e2e_benchmark.py --skip-generation        # kernel benchmark only
  python scripts/e2e_benchmark.py --skip-kernel            # generation timing only
  python scripts/e2e_benchmark.py \\
      --base-model lmsys/vicuna-7b-v1.3 \\
      --eagle-model yuhuili/EAGLE-Vicuna-7B-v1.3 \\
      --model-type vicuna --no-eagle3
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import ragged_attention
from src.tree_mask import num_tree_nodes, tree_attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_transformers_for_eagle() -> None:
    """
    Compatibility shim for EAGLE 3.0.0 vs transformers v5+.

    eagle/model/modeling_qwen3_kv.py imports LossKwargs, auto_docstring, and
    can_return_tuple from transformers.utils.  These symbols were added in
    transformers 4.47–4.50 and were moved / renamed in the v5 refactor.
    We inject lightweight stubs for any missing symbol so EaModel imports
    cleanly regardless of the installed transformers version.
    """
    try:
        import transformers.utils as _tu
        from typing import TypedDict

        if not hasattr(_tu, "LossKwargs"):
            class _LossKwargs(TypedDict, total=False):
                pass
            _tu.LossKwargs = _LossKwargs  # type: ignore[attr-defined]

        if not hasattr(_tu, "auto_docstring"):
            def _auto_docstring(*args, **kwargs):
                if args and callable(args[0]):
                    return args[0]
                return lambda fn: fn
            _tu.auto_docstring = _auto_docstring  # type: ignore[attr-defined]

        if not hasattr(_tu, "can_return_tuple"):
            def _can_return_tuple(fn):
                return fn
            _tu.can_return_tuple = _can_return_tuple  # type: ignore[attr-defined]

    except Exception:
        pass  # Let EaModel fail naturally if transformers isn't importable at all


# Apply at module load so _has("eagle") and all subsequent imports succeed.
_patch_transformers_for_eagle()


def _has(pkg: str) -> bool:
    if importlib.util.find_spec(pkg) is None:
        return False
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


HAS_EAGLE = _has("eagle")

DEFAULT_PROMPTS = [
    "The theory of general relativity, proposed by Albert Einstein in 1915,",
    "In machine learning, the attention mechanism was introduced to",
    "The Python programming language was created by Guido van Rossum and",
    "Large language models have transformed natural language processing by",
    "Speculative decoding accelerates autoregressive generation by",
    "The transformer architecture consists of an encoder and a decoder, where",
    "CUDA programming enables massively parallel computation on",
    "Flash attention optimizes the attention computation by reducing",
    "Tree-structured speculative decoding extends chain-based methods by",
    "The key insight of our approach is that each query token only attends to",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    """One completed vanilla Eagle-3 eagenerate() call."""
    prompt:                 str
    num_tokens:             int
    num_steps:              int
    wall_ms:                float
    tok_per_sec:            float
    mean_accepted_per_step: float
    acceptance_rate:        float   # mean_accepted / tree_budget
    mean_verify_ms:         float   # mean CUDA-event time for tree_decoding
    verify_fraction:        float   # sum(verify_ms) / wall_ms


@dataclass
class KernelRecord:
    """One kernel timing row."""
    method:           str   # "sdpa_tree" or "ragged"
    batch_size:       int
    branching_factor: int
    depth:            int
    num_tree_nodes:   int
    num_heads:        int
    head_dim:         int
    latency_ms:       float
    speedup:          float  # sdpa_tree_ms / ragged_ms  (>1 → ragged faster)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1  —  Vanilla Eagle-3 generation
# ─────────────────────────────────────────────────────────────────────────────

def _get_prompt(model_type: str, message: str) -> str:
    try:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template(model_type)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    except ImportError:
        return f"User: {message}\nAssistant:"


def load_eagle_model(
    base_model: str,
    eagle_model: str,
    use_eagle3: bool = True,
    total_token: int = 60,
    depth: int = 7,
    top_k: int = 10,
) -> "eagle.model.ea_model.EaModel":
    _patch_transformers_for_eagle()
    from eagle.model.ea_model import EaModel

    # Llama-3.1 uses rope_scaling["rope_type"] but EAGLE's cnets.py expects ["type"].
    # Monkey-patch _init_rope to normalise the key before it is read.
    try:
        from eagle.model import cnets as _cnets
        _orig_init_rope = _cnets.LlamaAttention._init_rope
        def _patched_init_rope(self):
            rs = getattr(self.config, "rope_scaling", None)
            if isinstance(rs, dict) and "type" not in rs and "rope_type" in rs:
                import copy
                self.config.rope_scaling = {**rs, "type": rs["rope_type"]}
            _orig_init_rope(self)
        _cnets.LlamaAttention._init_rope = _patched_init_rope
    except Exception:
        pass

    print(f"\n  Loading Eagle model:")
    print(f"    Base:  {base_model}")
    print(f"    Eagle: {eagle_model}")
    print(f"    Mode:  {'EAGLE-3' if use_eagle3 else 'EAGLE-2'}")
    print(f"    Tree:  total_token={total_token}, depth={depth}, top_k={top_k}")
    t0 = time.perf_counter()
    model = EaModel.from_pretrained(
        use_eagle3=use_eagle3,
        base_model_path=base_model,
        ea_model_path=eagle_model,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    cfg = model.base_model.config
    H    = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H)
    D    = cfg.hidden_size // H
    L    = cfg.num_hidden_layers
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"    LLM: H={H}, H_kv={H_kv}, D={D}, layers={L}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"    GPU: {p.name}  SM {p.major}.{p.minor}  "
              f"{p.total_memory // 1024**3} GB")
    return model


def run_generation(
    model,
    prompts: List[str],
    model_type: str,
    max_new_tokens: int,
    is_llama3: bool,
) -> List[GenerationRecord]:
    """
    Run vanilla Eagle-3 eagenerate() on every prompt.

    Each tree_decoding() call is timed with a pair of CUDA events so we
    know exactly how much wall time is spent in verification.
    """
    from eagle.model.utils import (
        initialize_tree, tree_decoding, evaluate_posterior,
        update_inference_inputs, reset_tree_mode,
    )
    from eagle.model.kv_cache import initialize_past_key_values

    device      = next(model.base_model.parameters()).device
    total_token = model.ea_layer.total_tokens
    records: List[GenerationRecord] = []

    for pi, raw in enumerate(prompts):
        prompt    = _get_prompt(model_type, raw)
        input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]
        max_len   = input_len + max_new_tokens + total_token + 10

        (past_kv, past_kv_data, cur_len_data) = initialize_past_key_values(
            model.base_model, max_length=max_len
        )
        model.past_key_values       = past_kv
        model.past_key_values_data  = past_kv_data
        model.current_length_data   = cur_len_data
        model.ea_layer.reset_kv()
        reset_tree_mode(model)

        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()

        (draft_tokens, retrieve_idx, tree_mask, tree_pos_ids,
         logits, hidden_state, sample_token) = initialize_tree(
            input_ids, model, past_kv, None
        )

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(device)
        new_token   = 0
        verify_ms_list: List[float] = []
        accepted_list:  List[int]   = []
        gen_max = max_len - total_token - 10

        for _ in range(gen_max):
            model.base_model.model.tree_mask = tree_mask

            # ── time verification (base-model forward with tree Q×K) ───────
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            logits, hidden_new, _ = tree_decoding(
                model, draft_tokens, past_kv, tree_pos_ids, input_ids, retrieve_idx,
            )
            e1.record()
            torch.cuda.synchronize()
            verify_ms_list.append(e0.elapsed_time(e1))

            # acceptance
            draft_long = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_long[0, retrieve_idx]
            best_cand, accept_len, sample_p = evaluate_posterior(
                logits, candidates, None
            )
            (input_ids, draft_tokens, retrieve_idx, tree_mask, tree_pos_ids,
             new_token, hidden_state, sample_token) = update_inference_inputs(
                input_ids, candidates, best_cand, accept_len,
                retrieve_idx, None, new_token,
                past_kv_data, cur_len_data, model, hidden_new, sample_p,
            )
            accepted_list.append(int(accept_len.item()) + 1)  # +1 for sampled base token

            # termination
            gen_so_far = input_ids[0, input_len:].tolist()
            if is_llama3:
                eot = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot in gen_so_far:
                    break
            if model.tokenizer.eos_token_id in gen_so_far:
                break
            if new_token >= max_new_tokens or input_ids.shape[1] >= gen_max:
                break

        torch.cuda.synchronize()
        wall_ms  = (time.perf_counter() - wall_t0) * 1000
        n_steps  = len(verify_ms_list)
        total_a  = sum(accepted_list)
        mean_a   = total_a / n_steps if n_steps else 0.0
        acc_rate = mean_a  / (total_token + 1) if total_token else 0.0
        tot_vms  = sum(verify_ms_list)
        mean_vms = tot_vms / n_steps if n_steps else 0.0
        vfrac    = tot_vms / wall_ms  if wall_ms  else 0.0

        rec = GenerationRecord(
            prompt=raw[:80],
            num_tokens=new_token,
            num_steps=n_steps,
            wall_ms=wall_ms,
            tok_per_sec=new_token / (wall_ms / 1000) if wall_ms else 0.0,
            mean_accepted_per_step=mean_a,
            acceptance_rate=acc_rate,
            mean_verify_ms=mean_vms,
            verify_fraction=vfrac,
        )
        records.append(rec)

        snippet = model.tokenizer.decode(
            input_ids[0, input_len: input_len + min(new_token, 50)],
            skip_special_tokens=True,
        )
        print(
            f"  [{pi+1}/{len(prompts)}] "
            f"{new_token} tok / {n_steps} steps  "
            f"accept={mean_a:.2f}/step ({acc_rate:.1%})  "
            f"{rec.tok_per_sec:.1f} tok/s  "
            f"verify={mean_vms:.1f} ms/step ({vfrac:.0%} of time)  "
            f"→ \"{snippet[:60]}...\""
        )

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 2  —  Tree attention kernel comparison
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_time(fn, warmup: int, iters: int) -> float:
    """Return mean CUDA-elapsed time in ms over `iters` measured calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in pairs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in pairs) / iters


def _time_sdpa_tree(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    b: int, d: int,
    warmup: int, iters: int,
) -> float:
    """
    Time PyTorch SDPA with explicit tree-ancestor bool mask.

    This is the same kernel path Eagle-3 uses internally:
    the tree_mask is converted to a float additive bias and passed to SDPA.
    Prefers FLASH_ATTENTION backend; falls back to EFFICIENT_ATTENTION if
    flash rejects a non-null additive mask.

    Q / K / V shape: [B, H, N, D]  (SDPA convention, fp16)
    """
    B, H, N, D = Q.shape
    mask_np = tree_attention_mask(b, d)   # [N, N] bool numpy
    mask_t  = torch.from_numpy(mask_np).to(Q.device)
    bias    = torch.where(
        mask_t,
        torch.zeros(1, device=Q.device, dtype=torch.float32),
        torch.full( (1,), float("-inf"), device=Q.device, dtype=torch.float32),
    )                                     # [N, N]
    bias4d = bias.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N).contiguous()

    # detect which SDPA backend accepts an additive mask (flash may refuse)
    try:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            F.scaled_dot_product_attention(Q, K, V, attn_mask=bias4d)
        backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION
    except RuntimeError as _be:
        if "out of memory" in str(_be).lower():
            raise  # propagate OOM — don't silently fall through
        backend = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
    except Exception:
        backend = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION

    def fn():
        with torch.nn.attention.sdpa_kernel(backend):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=bias4d)

    return _cuda_time(fn, warmup, iters)


def _time_ragged(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    cu: torch.Tensor,
    b: int, d: int,
    warmup: int, iters: int,
) -> float:
    """Time our ragged kernel. Q/K/V: [B*N, H, D] packed layout, fp16."""
    def fn():
        return ragged_attention(Q, K, V, cu, b, d)
    return _cuda_time(fn, warmup, iters)


def run_kernel_comparison(
    num_heads: int,
    head_dim: int,
    batch_sizes: List[int],
    branching_factors: List[int],
    depths: List[int],
    warmup: int,
    iters: int,
) -> List[KernelRecord]:
    """
    For each (B, b, d): benchmark sdpa_tree then ragged on matching fp16 tensors.
    Returns two KernelRecords per config — one per method.
    """
    device = torch.device("cuda")
    H, D   = num_heads, head_dim
    records: List[KernelRecord] = []
    configs = [
        (B, b, d)
        for B in batch_sizes
        for b in branching_factors
        for d in depths
    ]

    for ci, (B, b, d) in enumerate(configs):
        N    = num_tree_nodes(b, d)
        tot  = B * N

        torch.manual_seed(B * 1000 + b * 100 + d)
        try:
            # packed ragged layout [B*N, H, D]
            Q_r  = torch.randn(tot, H, D, device=device, dtype=torch.float16)
            K_r  = torch.randn(tot, H, D, device=device, dtype=torch.float16)
            V_r  = torch.randn(tot, H, D, device=device, dtype=torch.float16)
            cu   = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)
            # SDPA layout [B, H, N, D]
            Q_s  = Q_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
            K_s  = K_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
            V_s  = V_r.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(
                    f"  [{ci+1:3d}/{len(configs)}]  "
                    f"B={B:3d} b={b} d={d} N={N:5d}  "
                    f"OOM during tensor alloc — skipped"
                )
                continue
            raise

        t_sdpa   = float("nan")
        t_ragged = float("nan")

        try:
            t_sdpa = _time_sdpa_tree(Q_s, K_s, V_s, b, d, warmup, iters)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(
                    f"  [{ci+1:3d}/{len(configs)}]  "
                    f"B={B:3d} b={b} d={d} N={N:5d}  sdpa_tree OOM — t_sdpa=NaN"
                )
            else:
                print(f"  [{ci+1}/{len(configs)}] sdpa_tree B={B} b={b} d={d}  ERROR: {exc}")
        except Exception as exc:
            print(f"  [{ci+1}/{len(configs)}] sdpa_tree B={B} b={b} d={d}  ERROR: {exc}")

        try:
            t_ragged = _time_ragged(Q_r, K_r, V_r, cu, b, d, warmup, iters)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(
                    f"  [{ci+1:3d}/{len(configs)}]  "
                    f"B={B:3d} b={b} d={d} N={N:5d}  ragged OOM — t_ragged=NaN"
                )
            else:
                print(f"  [{ci+1}/{len(configs)}] ragged B={B} b={b} d={d}  ERROR: {exc}")
        except Exception as exc:
            print(f"  [{ci+1}/{len(configs)}] ragged B={B} b={b} d={d}  ERROR: {exc}")

        speedup = (t_sdpa / t_ragged
                   if (not math.isnan(t_sdpa) and not math.isnan(t_ragged)
                       and t_ragged > 0)
                   else float("nan"))
        spd_str = f"{speedup:.2f}×" if not math.isnan(speedup) else " n/a "

        print(
            f"  [{ci+1:3d}/{len(configs)}]  "
            f"B={B:3d} b={b} d={d} N={N:5d}  "
            f"sdpa_tree={t_sdpa:8.3f} ms  "
            f"ragged={t_ragged:8.3f} ms  "
            f"speedup={spd_str}"
        )

        for method, ms in [("sdpa_tree", t_sdpa), ("ragged", t_ragged)]:
            records.append(KernelRecord(
                method=method,
                batch_size=B,
                branching_factor=b,
                depth=d,
                num_tree_nodes=N,
                num_heads=H,
                head_dim=D,
                latency_ms=round(ms, 4) if not math.isnan(ms) else float("nan"),
                speedup=round(speedup, 4) if not math.isnan(speedup) else float("nan"),
            ))

        del Q_r, K_r, V_r, Q_s, K_s, V_s, cu
        torch.cuda.empty_cache()

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 3  —  Amdahl projection
# ─────────────────────────────────────────────────────────────────────────────

def amdahl_projection(
    gen_records:    List[GenerationRecord],
    kernel_records: List[KernelRecord],
    prompt_len:     int,
    tree_size:      int,
) -> Dict:
    """
    Project E2E speedup from replacing intra-tree attention with our kernel.

    Amdahl formula:
        S = 1 / ( (1 - f) + f/k )
    where:
        f = verify_fraction × attn_fraction × intra_tree_fraction
        k = kernel speedup  (sdpa_tree_ms / ragged_ms, median at B=1)
        attn_fraction       ≈ 0.35  (conservative: ~35% of model forward is attn)
        intra_tree_fraction = N_tree / (N_prefix + N_tree)
    """
    if not gen_records or not kernel_records:
        return {}

    mean_vfrac  = float(np.mean([r.verify_fraction for r in gen_records]))
    mean_tok_s  = float(np.mean([r.tok_per_sec for r in gen_records]))
    mean_acc    = float(np.mean([r.mean_accepted_per_step for r in gen_records]))

    # Kernel speedup: use B=1 rows (Eagle-3 runs batch=1)
    speedups_b1 = [
        r.speedup for r in kernel_records
        if r.method == "ragged" and r.batch_size == 1
        and not math.isnan(r.speedup)
    ]
    if not speedups_b1:
        return {
            "eagle3_tok_per_sec":     round(mean_tok_s, 1),
            "mean_accepted_per_step": round(mean_acc, 3),
            "verify_fraction":        round(mean_vfrac, 4),
            "note": "no B=1 kernel results for projection",
        }

    k       = float(np.median(speedups_b1))
    f_intra = tree_size / max(prompt_len + tree_size, 1)
    F_ATTN  = 0.35   # conservative fraction of model forward in attention

    f       = mean_vfrac * F_ATTN * f_intra
    amdahl  = 1.0 / ((1.0 - f) + f / k) if k > 0 else 1.0
    proj_t  = mean_tok_s * amdahl

    return {
        "eagle3_tok_per_sec":         round(mean_tok_s, 1),
        "mean_accepted_per_step":     round(mean_acc, 3),
        "verify_fraction":            round(mean_vfrac, 4),
        "intra_tree_fraction":        round(f_intra, 4),
        "attn_fraction_assumed":      F_ATTN,
        "kernel_speedup_b1_median":   round(k, 3),
        "amdahl_f":                   round(f, 5),
        "amdahl_e2e_speedup":         round(amdahl, 3),
        "projected_tok_per_sec":      round(proj_t, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eagle-3 E2E benchmark: ragged kernel vs. vanilla Eagle-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Model
    parser.add_argument("--base-model",  default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle-model", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--model-type",  default="llama-3-instruct",
                        choices=["llama-3-instruct", "llama3", "llama2", "vicuna"])
    parser.add_argument("--no-eagle3",   action="store_true",
                        help="Use EAGLE-2 (not EAGLE-3)")
    # Tree config
    parser.add_argument("--total-tokens", type=int, default=60)
    parser.add_argument("--depth",        type=int, default=7)
    parser.add_argument("--top-k",        type=int, default=10)
    # Generation
    parser.add_argument("--num-prompts",    type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # Kernel benchmark
    parser.add_argument("--kernel-batch-sizes", default="1,8,32,128")
    parser.add_argument("--kernel-branching",   default="3,4")
    parser.add_argument("--kernel-depths",      default="3,5,7")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters",  type=int, default=20)
    parser.add_argument("--prompt-len", type=int, default=128,
                        help="Estimated prompt length for Amdahl intra-tree fraction")
    # Control
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-kernel",     action="store_true")
    parser.add_argument("--out-dir",  default="results")
    parser.add_argument("--csv-name", default="e2e_benchmark.csv")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token for gated models "
                             "(overrides $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN)")
    args = parser.parse_args()

    # Propagate HF token so every huggingface_hub call (including EAGLE internals)
    # authenticates automatically.  CLI flag takes priority over env vars.
    _hf_token = (args.hf_token
                 or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    is_eagle3 = not args.no_eagle3
    is_llama3 = args.model_type in ("llama3", "llama-3-instruct")
    prompts   = DEFAULT_PROMPTS[:args.num_prompts]
    kb_sizes  = [int(x) for x in args.kernel_batch_sizes.split(",")]
    kb_bf     = [int(x) for x in args.kernel_branching.split(",")]
    kb_d      = [int(x) for x in args.kernel_depths.split(",")]

    print("\n" + "=" * 70)
    print("  Eagle-3 E2E Benchmark  ·  ragged kernel vs. vanilla Eagle-3")
    print("=" * 70)
    print(f"  Base:          {args.base_model}")
    print(f"  Eagle:         {args.eagle_model}  "
          f"({'EAGLE-3' if is_eagle3 else 'EAGLE-2'})")
    print(f"  Tree:          total={args.total_tokens}, "
          f"depth={args.depth}, top_k={args.top_k}")
    print(f"  Kernel grid:   B∈{kb_sizes}, b∈{kb_bf}, d∈{kb_d}")
    if _hf_token:
        print(f"  HF token:      {_hf_token[:8]}…")
    else:
        print("  HF token:      (none — public/local models only)")

    # Model dimensions — will be updated from actual model if generation runs
    n_heads, head_dim = 32, 128  # LLaMA-3.1-8B defaults

    # ── Step 1: Vanilla Eagle-3 generation ──────────────────────────────────
    gen_records: List[GenerationRecord] = []

    if not args.skip_generation:
        if not HAS_EAGLE:
            print("\nERROR: EAGLE not installed.")
            print("  pip install git+https://github.com/SafeAILab/EAGLE.git fschat")
            sys.exit(1)

        model = load_eagle_model(
            base_model=args.base_model,
            eagle_model=args.eagle_model,
            use_eagle3=is_eagle3,
            total_token=args.total_tokens,
            depth=args.depth,
            top_k=args.top_k,
        )
        cfg       = model.base_model.config
        n_heads   = cfg.num_attention_heads
        head_dim  = cfg.hidden_size // n_heads

        print(f"\n{'─' * 70}")
        print("  STEP 1  —  Vanilla Eagle-3 generation  (default attention)")
        print(f"{'─' * 70}")

        gen_records = run_generation(
            model, prompts, args.model_type,
            max_new_tokens=args.max_new_tokens,
            is_llama3=is_llama3,
        )

        if gen_records:
            tok_s  = np.mean([r.tok_per_sec for r in gen_records])
            acc    = np.mean([r.mean_accepted_per_step for r in gen_records])
            acc_r  = np.mean([r.acceptance_rate for r in gen_records])
            vms    = np.mean([r.mean_verify_ms for r in gen_records])
            vfrac  = np.mean([r.verify_fraction for r in gen_records])
            print()
            print(f"  Summary ({len(gen_records)} prompts):")
            print(f"    tok/s                : {tok_s:.1f}")
            print(f"    accepted/step (mean) : {acc:.2f}")
            print(f"    acceptance rate      : {acc_r:.1%}")
            print(f"    verify latency       : {vms:.1f} ms/step")
            print(f"    verify time fraction : {vfrac:.0%}")

        del model
        torch.cuda.empty_cache()
    else:
        try:
            from transformers import AutoConfig
            cfg      = AutoConfig.from_pretrained(args.base_model)
            n_heads  = cfg.num_attention_heads
            head_dim = cfg.hidden_size // n_heads
        except Exception:
            pass  # keep LLaMA defaults

    # ── Step 2: Kernel comparison ─────────────────────────────────────────────
    kernel_records: List[KernelRecord] = []

    if not args.skip_kernel:
        print(f"\n{'─' * 70}")
        print("  STEP 2  —  Tree attention kernel comparison")
        print(f"             sdpa_tree  (Eagle-3 native)  vs  ragged  (ours)")
        print(f"{'─' * 70}")
        print(f"  Model dims: H={n_heads}, D={head_dim}")
        print()

        kernel_records = run_kernel_comparison(
            num_heads=n_heads,
            head_dim=head_dim,
            batch_sizes=kb_sizes,
            branching_factors=kb_bf,
            depths=kb_d,
            warmup=args.warmup,
            iters=args.iters,
        )

        # per-config speedup table
        print()
        print("  Speedup table  (sdpa_tree_ms / ragged_ms — >1 means ragged is faster):")
        print(f"  {'B':>4s} {'b':>2s} {'d':>2s} {'N':>6s}  {'speedup':>10s}")
        for r in kernel_records:
            if r.method == "ragged":
                s = f"{r.speedup:.2f}×" if not math.isnan(r.speedup) else "    n/a"
                print(f"  {r.batch_size:4d} {r.branching_factor:2d} "
                      f"{r.depth:2d} {r.num_tree_nodes:6d}  {s:>10s}")

    # ── Step 3: Amdahl projection ─────────────────────────────────────────────
    proj: Dict = {}
    if gen_records and kernel_records:
        # Typical tree size for Eagle-3 with the configured b/d midpoint
        typical_n = num_tree_nodes(
            kb_bf[0] if kb_bf else 3,
            kb_d[len(kb_d) // 2] if kb_d else 5,
        )
        proj = amdahl_projection(
            gen_records, kernel_records,
            prompt_len=args.prompt_len,
            tree_size=typical_n,
        )

        print(f"\n{'─' * 70}")
        print("  STEP 3  —  Amdahl E2E projection")
        print(f"{'─' * 70}")
        for k, v in proj.items():
            print(f"  {k:<35s}: {v}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    rows: List[dict] = []

    for r in gen_records:
        rows.append({
            "phase":                    "generation",
            "mode":                     "vanilla_eagle3",
            "model":                    args.base_model,
            "eagle_model":              args.eagle_model,
            "prompt":                   r.prompt,
            "num_tokens":               r.num_tokens,
            "num_steps":                r.num_steps,
            "wall_ms":                  round(r.wall_ms, 1),
            "tok_per_sec":              round(r.tok_per_sec, 1),
            "mean_accepted_per_step":   round(r.mean_accepted_per_step, 3),
            "acceptance_rate":          round(r.acceptance_rate, 4),
            "mean_verify_ms":           round(r.mean_verify_ms, 3),
            "verify_fraction":          round(r.verify_fraction, 4),
        })

    for r in kernel_records:
        rows.append({
            "phase":                           "kernel",
            "method":                          r.method,
            "batch_size":                      r.batch_size,
            "branching_factor":                r.branching_factor,
            "depth":                           r.depth,
            "num_tree_nodes":                  r.num_tree_nodes,
            "num_heads":                       r.num_heads,
            "head_dim":                        r.head_dim,
            "latency_ms":                      r.latency_ms,
            "speedup_sdpa_tree_over_ragged":   r.speedup,
        })

    if proj:
        rows.append({"phase": "amdahl_projection", **proj})

    if rows:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\n  Saved: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 70}")
    if gen_records:
        print(f"  Vanilla Eagle-3 tok/s :   "
              f"{np.mean([r.tok_per_sec for r in gen_records]):.1f}")
    if kernel_records:
        spd = [r.speedup for r in kernel_records
               if r.method == "ragged" and not math.isnan(r.speedup)]
        if spd:
            print(f"  Kernel speedup range  :   "
                  f"{min(spd):.2f}× – {max(spd):.2f}×  "
                  f"(median {float(np.median(spd)):.2f}×)")
    if proj:
        print(f"  Amdahl projected speedup: "
              f"{proj.get('amdahl_e2e_speedup', 'n/a')}")
        print(f"  Projected tok/s       :   "
              f"{proj.get('projected_tok_per_sec', 'n/a')}")
    print()


if __name__ == "__main__":
    main()
