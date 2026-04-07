#!/usr/bin/env python3
"""
e2e_benchmark.py — Eagle-3 E2E Benchmark: Tree Depth Sweep
=============================================================================

Compares vanilla Eagle-3 vs. our ragged-kernel Eagle-3 across a GRID of
tree-depth configurations to establish the paper's core claim:

    *Deeper draft trees → more E2E speedup from the ragged attention kernel.*

Narrative (validated by micro-benchmark on H100 SXM):
------------------------------------------------------
The ragged kernel's advantage is driven by the number of tree tokens N,
which scales with both depth d and branching factor b.

Micro-benchmark crossover thresholds (L=0, no prefix, LLaMA-3.1-8B dims):
  B=1  (single-user serving): ~N > 200  (d≥16 at b=10)
  B=8  (light batching):      ~N ≥ 103  (d≥12 at b=10)
  B=32 (moderate batching):   ~N ≥  51  (d≥7  at b=10)
  B=128(full batching):       ~N ≥  34  (d≥5  at b=10)

At L=1024 (realistic prefix KV cache): ragged wins 100% of configurations.
At L=4096: ragged wins 100%, median 4.3×, worst case 1.83×.

This E2E sweep therefore spans from the loss regime (d=5,7 — N≈43,60) to
the clear win regime (d=24,28,32 — N≈206,240,274) at EAGLE-3's default
branching factor b=10.  The crossover shifts left (lower d) with:
  (a) larger batch sizes (continuous-batching serving scenario), or
  (b) larger prefix KV caches (common in mid-to-late generation).

Design
------
For each tree config (depth d, total_token tt, top_k):
  1. Set EAGLE model tree parameters in-place (no reload).
  2. Run vanilla Eagle-3 (default SDPA attention) on N shared prompts.
  3. Run ragged-kernel Eagle-3 (patched matmul → ragged + flash + LSE merge)
     on the same prompts.
  4. Record wall-clock tok/s, acceptance rate, verify latency for both.
  5. Report actual E2E speedup  = ragged_tok_s / vanilla_tok_s.

The model is loaded ONCE at the maximum-depth config to allocate a
sufficiently large KV cache.  Between configs, only ``ea_layer.total_tokens``,
``ea_layer.depth``, and ``ea_layer.top_k`` are updated — these are read
dynamically during ``eagenerate()`` (EAGLE-3 builds its tree at runtime).

Kernel-level microbenchmarks (SDPA vs FlashInfer vs DeFT) are NOT duplicated
here — they belong in ``scripts/benchmark_sota.py``.  A focused paper-figure
micro-bench is in ``scripts/benchmark_micro.py``.

Output
------
  results/e2e_benchmark.csv   — per-prompt records (all configs × all prompts)
  results/e2e_summary.csv     — one row per config with aggregated metrics

Usage
-----
  # Default 2D sweep: depths=[5,7,9,12,16,20,24,28,32] × branching=[8,10,12]
  python scripts/e2e_benchmark.py

  # Faster run (fewer prompts, narrower sweep)
  python scripts/e2e_benchmark.py --num-prompts 5 --depths 5,7,12,20,32

  # Custom grid
  python scripts/e2e_benchmark.py --depths 7,12,20,28 --branching-factors 10,12

  # Override token budget for specific (b, d) cells
  python scripts/e2e_benchmark.py --total-tokens-map 'b10d7:60,b12d9:100'

  # Only vanilla (no ragged kernel, useful for baseline timing)
  python scripts/e2e_benchmark.py --skip-ragged

Prerequisites
-------------
  pip install git+https://github.com/SafeAILab/EAGLE.git fschat
  pip install 'transformers==4.53.1' 'accelerate>=0.26.0,<1.0'
  huggingface-cli login   # for LLaMA gated models
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import random

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import ragged_attention_with_lse, ragged_attention_with_parents


# ─────────────────────────────────────────────────────────────────────────────
# Per-component profiler for _ragged_tree_attn
# ─────────────────────────────────────────────────────────────────────────────

class _RaggedProfiler:
    """
    Zero-overhead (CUDA-event based) profiler for _ragged_tree_attn.

    Records 7 CUDA events per call, bracketing 6 phases:
        [0]→[1] prefix_split   : K_pre/V_pre slice + .contiguous()
        [1]→[2] flash_prefix   : flash-attention over prefix KV cache
        [2]→[3] tree_reshape   : permute/reshape/contiguous + cu_seqlens
        [3]→[4] ragged_kernel  : Triton ragged ancestor-sparse kernel
        [4]→[5] out_reshape    : view + permute of kernel output
        [5]→[6] lse_merge      : online-softmax merge of prefix + tree

    Events are recorded asynchronously (negligible overhead).
    Timing is resolved ONCE after torch.cuda.synchronize() in summary().
    """

    PHASES = ("prefix_split", "flash_prefix", "tree_reshape",
              "ragged_kernel", "out_reshape", "lse_merge")

    def __init__(self, max_shapes: int = 10):
        self.max_shapes = max_shapes
        self.entries: List[List[torch.cuda.Event]] = []   # 7 events each
        self.shapes: List[str] = []
        # hook dispatch counters
        self.hook_total       = 0   # all _patched_matmul invocations
        self.hook_intercept   = 0   # matched QK or AV  (ragged path)
        self.hook_skip_verify = 0   # skipped: not in tree_verify window
        self.hook_skip_inside = 0   # skipped: re-entrancy guard
        self.hook_skip_shape  = 0   # skipped: dim/shape didn't match

    def reset(self):
        """Clear for next prompt."""
        self.entries.clear()
        self.shapes.clear()
        self.hook_total = 0
        self.hook_intercept = 0
        self.hook_skip_verify = 0
        self.hook_skip_inside = 0
        self.hook_skip_shape = 0

    def record_shapes(self, Q: torch.Tensor, K: torch.Tensor,
                      N_prefix: int, N_tree: int):
        if len(self.shapes) < self.max_shapes:
            self.shapes.append(
                f"Q={list(Q.shape)} K={list(K.shape)} "
                f"N_prefix={N_prefix} N_tree={N_tree} "
                f"prefix:tree=1:{N_prefix / max(N_tree, 1):.1f}"
            )

    def summary(self) -> str:
        """
        Format a timing breakdown table.
        MUST be called after torch.cuda.synchronize().
        """
        n = len(self.entries)
        if n == 0:
            return "  [profile] no ragged attention calls recorded"

        # compute per-phase timings from events
        phase_sums = {p: 0.0 for p in self.PHASES}
        total_sum = 0.0
        for evts in self.entries:
            for i, p in enumerate(self.PHASES):
                phase_sums[p] += evts[i].elapsed_time(evts[i + 1])
            total_sum += evts[0].elapsed_time(evts[6])

        n_steps = max(n // 32, 1)  # assume 32 layers (LLaMA-3.1-8B)

        lines = [
            f"  ┌── RAGGED PROFILE  ({n} calls ≈ {n_steps} steps × 32 layers) {'─' * 24}",
            f"  │ {'Component':<18s}  {'Mean/call':>10s}  {'Total':>10s}  "
            f"{'%total':>7s}  {'Per-step':>10s}",
            f"  │ {'─' * 18}  {'─' * 10}  {'─' * 10}  {'─' * 7}  {'─' * 10}",
        ]
        for p in self.PHASES:
            s = phase_sums[p]
            m = s / n
            pct = s / total_sum * 100 if total_sum else 0
            per_step = s / n_steps
            lines.append(
                f"  │ {p:<18s}  {m:>8.4f}ms  {s:>8.1f}ms  "
                f"{pct:>6.1f}%  {per_step:>8.3f}ms"
            )
        lines.append(f"  │ {'─' * 18}  {'─' * 10}  {'─' * 10}  {'─' * 7}  {'─' * 10}")
        per_step_total = total_sum / n_steps
        lines.append(
            f"  │ {'TOTAL':<18s}  {total_sum / n:>8.4f}ms  {total_sum:>8.1f}ms  "
            f"{'100.0':>6s}%  {per_step_total:>8.3f}ms"
        )

        # hook dispatch stats
        lines.append(f"  │")
        lines.append(f"  │ Hook dispatch:  {self.hook_total} total matmul calls")
        lines.append(f"  │   intercepted (→ ragged):  {self.hook_intercept}  "
                     f"({self.hook_intercept // 2} layers × 2 matmuls)")
        lines.append(f"  │   skip (not in verify):    {self.hook_skip_verify}")
        lines.append(f"  │   skip (re-entrancy):      {self.hook_skip_inside}")
        lines.append(f"  │   skip (dim/shape):        {self.hook_skip_shape}")

        # shapes
        if self.shapes:
            lines.append(f"  │")
            lines.append(f"  │ Shapes (first {len(self.shapes)} calls):")
            for s in self.shapes:
                lines.append(f"  │   {s}")

        lines.append(f"  └{'─' * 70}")
        return "\n".join(lines)


# Module-level profiler — set by run_generation, read by _ragged_tree_attn
_ACTIVE_PROFILER: Optional[_RaggedProfiler] = None


# ── Version gate ─────────────────────────────────────────────────────────────
# EAGLE 3.0.x was authored + tested against transformers 4.53.1 and
# accelerate 0.26.0.  Silent numerical corruption (degenerate "Destination
# Destination..." output, 0% acceptance rate) has been observed with:
#   • transformers > 4.53.x  (changed RoPE / attn-mask internals)
#   • accelerate >= 1.0       (changed device-map hook semantics)
# If you see garbage output, run:
#   pip install "transformers==4.53.1" "accelerate>=0.26.0,<1.0"
def _check_env_versions() -> None:
    import importlib.metadata as _im
    def _ver(pkg):
        try: return tuple(int(x) for x in _im.version(pkg).split(".")[:3])
        except Exception: return (0,)

    tx = _ver("transformers")
    ac = _ver("accelerate")

    problems = []
    if tx < (4, 53, 1):
        problems.append(
            f"transformers {'.'.join(str(x) for x in tx)} < 4.53.1  "
            "(EAGLE requires >=4.53.1)"
        )
    if tx >= (5, 0, 0):
        problems.append(
            f"transformers {'.'.join(str(x) for x in tx)} is a major-version "
            "release that EAGLE has not been tested with"
        )
    if ac >= (1, 0, 0):
        problems.append(
            f"accelerate {'.'.join(str(x) for x in ac)} >= 1.0  "
            "(EAGLE requires <1.0 for stable device_map='auto' behaviour)"
        )

    tx_s = ".".join(str(x) for x in tx)
    ac_s = ".".join(str(x) for x in ac)
    print(f"  [env] transformers={tx_s}  accelerate={ac_s}", end="")
    if problems:
        print("  ← WARNING")
        for p in problems:
            print(f"    [env:warn] {p}")
        print("    Run:  pip install 'transformers==4.53.1' 'accelerate>=0.26.0,<1.0'")
    else:
        print("  OK")


_check_env_versions()


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

# ── ShareGPT prompt loader ──────────────────────────────────────────────────
# anon8231489123/ShareGPT_Vicuna_unfiltered  (53 k conversations, Apache-2.0)
# Schema: {"id": "...", "conversations": [{"from": "human"|"gpt", "value": "..."}]}
_SHAREGPT_REPO   = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_SHAREGPT_FILE   = "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"

# Small fallback used only when huggingface_hub is unavailable.
_FALLBACK_PROMPTS: List[str] = [
    "Can you explain how transformers work in natural language processing?",
    "Write a Python function to merge two sorted lists.",
    "What are the main differences between supervised and unsupervised learning?",
    "Explain the concept of attention mechanism in deep learning.",
    "How does speculative decoding improve autoregressive generation speed?",
    "What is the difference between CUDA cores and Tensor Cores on an NVIDIA GPU?",
    "Can you help me debug this JavaScript code that handles API requests?",
    "Describe the key challenges in deploying large language models at scale.",
    "How does FlashAttention reduce memory usage for the attention computation?",
    "Write a SQL query to find the top 5 customers by total order value.",
]


def _load_sharegpt_prompts(
    n: int,
    seed: int = 42,
    hf_token: Optional[str] = None,
    min_len: int = 40,
    max_len: int = 512,
) -> List[str]:
    """Return `n` first-human-turn prompts from ShareGPT_Vicuna_unfiltered.

    Downloads ``ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`` from
    ``anon8231489123/ShareGPT_Vicuna_unfiltered`` via huggingface_hub (cached
    after first download).  Falls back to _FALLBACK_PROMPTS if the hub is
    unreachable.
    """
    import json as _json

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print("  [prompts] huggingface_hub not installed — using fallback prompts.")
        rng = random.Random(seed)
        return (rng.sample(_FALLBACK_PROMPTS, min(n, len(_FALLBACK_PROMPTS))) * math.ceil(n / len(_FALLBACK_PROMPTS)))[:n]

    print(f"  [prompts] loading {_SHAREGPT_FILE} from {_SHAREGPT_REPO} …")
    try:
        local_path = hf_hub_download(
            repo_id=_SHAREGPT_REPO,
            filename=_SHAREGPT_FILE,
            repo_type="dataset",
            token=hf_token,
        )
    except Exception as e:
        print(f"  [prompts] download failed ({e}) — using fallback prompts.")
        rng = random.Random(seed)
        return (rng.sample(_FALLBACK_PROMPTS, min(n, len(_FALLBACK_PROMPTS))) * math.ceil(n / len(_FALLBACK_PROMPTS)))[:n]

    with open(local_path, "r", encoding="utf-8") as fh:
        data = _json.load(fh)

    # Extract first human turn from each conversation.
    candidates: List[str] = []
    for row in data:
        convs = row.get("conversations") or []
        for turn in convs:
            # Schema: {"from": "human"|"gpt", "value": "..."}
            if turn.get("from") == "human":
                text = (turn.get("value") or "").strip()
                if min_len <= len(text) <= max_len:
                    candidates.append(text)
                break  # one prompt per conversation

    if not candidates:
        print("  [prompts] no usable prompts found — using fallback.")
        candidates = _FALLBACK_PROMPTS

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if len(candidates) < n:
        print(f"  [prompts] only {len(candidates)} candidates, need {n} — repeating.")
        candidates = (candidates * math.ceil(n / len(candidates)))[:n]

    print(f"  [prompts] loaded {n} prompts from ShareGPT_Vicuna_unfiltered (seed={seed})")
    return candidates[:n]


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
class TreeConfig:
    """One tree configuration to benchmark."""
    depth:        int
    total_token:  int
    top_k:        int
    label:        str


# Depth sweep spanning the full story arc validated by micro-benchmark:
#
#   LOSS REGIME (N too small, kernel overhead dominates):
#     d=5  → N≈43  (b=10)   |  E2E: ragged < vanilla at B≤8
#     d=7  → N≈60  (b=10)   |  E2E: ragged ≈ vanilla at B=8, +21% at B=32
#
#   CROSSOVER (B-dependent; shown here for single-user B=1 serving):
#     d=9  → N≈77            |  approaching crossover
#     d=12 → N≈103           |  B=8 crossover from micro-benchmark
#     d=16 → N≈137           |  clear win at B≥8
#
#   WIN REGIME (sparse ancestor walk dominates):
#     d=20 → N≈171  |  d=24 → N≈206  |  d=28 → N≈240  |  d=32 → N≈274
#
# The paper shows this crossover: EAGLE-3 default (d=7) is in or near the
# loss regime, but deeper trees — as targeted by future SD systems — win.
#
# Focus on WIN regime for initial E2E confirmation — once we beat vanilla
# at any point we can expand the grid back to the full sweep.
# Previous runs showed d≤12 was LOSS at all b (0.53–0.57×) due to:
#   1. Overhead in prefix flash + reshape + LSE merge dominates small trees
#   2. Correctness gap (acceptance rate drop) inflates step count
#
# d≥16 is where the micro-benchmark shows clear kernel wins (N≥137).
DEFAULT_DEPTH_SWEEP       = [16, 20, 24, 28, 32]
DEFAULT_BRANCHING_FACTORS = [10]            # EAGLE-3 default ★ only for now


def _default_total_token(b: int, d: int) -> int:
    """Token budget for a given (branching, depth) config.

    Anchored at EAGLE-3's default: (b=10, d=7) → total_token=60.
    Formula: round(6 * b * d / 7), with floor of 30 and NO upper cap.

    The upper cap (previously 120) has been REMOVED so that large-depth
    configs produce proportionally larger trees.  This is essential for
    the paper's story: the ragged kernel's win regime requires N>100,
    which only manifests at d≥12 (b=10) if the token budget is allowed
    to scale with depth.

    Token budget grid (selected b×d shown here):
      b=8 : d=5→34   d=7→48   d=9→62   d=12→82   d=16→110  d=20→137
            d=24→165  d=28→192 d=32→219
      b=10: d=5→43   d=7→60★  d=9→77   d=12→103  d=16→137  d=20→171
            d=24→206  d=28→240 d=32→274
      b=12: d=5→51   d=7→72   d=9→93   d=12→123  d=16→165  d=20→206
            d=24→247  d=28→288 d=32→329

    Crossover from micro-benchmark (L=0, B=1 serving):
      N ≈ 200+ needed for clear E2E win at single-user B=1.
      N ≈ 103  for B=8  crossover  → d≥12 at b=10.
      N ≈  51  for B=32 crossover  → d≥7  at b=10 (already wins!).
    """
    return max(30, round(6 * b * d / 7))


def set_tree_config(model, cfg: TreeConfig) -> None:
    """Modify EAGLE tree parameters in-place between generation runs.

    EAGLE-3's cnets.Model stores total_tokens (= total_token − 1), depth,
    and top_k as plain attributes read at each eagenerate() call.  The tree
    is built dynamically — no stale pre-computed indices.
    init_tree() re-creates the [top_k, top_k] initial mask buffer.
    """
    ea = model.ea_layer
    ea.total_tokens = cfg.total_token - 1   # cnets convention
    ea.depth        = cfg.depth
    if ea.top_k != cfg.top_k:
        ea.top_k = cfg.top_k
        ea.init_tree()  # re-create mask buffer sized to new top_k


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def _get_prompt(model_type: str, message: str) -> str:
    """Format a raw user message as a prompt string for non-LLaMA-3 models.

    For LLaMA-3 / LLaMA-3.1 Instruct, use ``_llama3_input_ids`` instead —
    the tokenizer's Jinja chat template must be applied *with tokenization*
    so that special tokens like <|begin_of_text|> are encoded as their token
    IDs, not as literal sub-word text.
    """
    try:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template(model_type)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    except ImportError:
        pass

    return f"User: {message}\nAssistant:"


def _llama3_input_ids(tokenizer, message: str, device) -> torch.Tensor:
    """Tokenize a user message for LLaMA-3/3.1 Instruct via the tokenizer's
    built-in Jinja chat template.

    Handles two transformers-version behaviours of apply_chat_template:
      - Returns list[int]  (transformers <5): wrap directly.
      - Returns str        (some v5 builds):  encode with add_special_tokens=False.
        The LLaMA-3 tokenizer registers <|begin_of_text|> etc. as special
        tokens, so tokenizer.encode() maps them to their correct token IDs
        rather than splitting them into sub-word pieces.

    Never go through tokenizer(text_with_markers) via __call__ — that path
    can treat the marker strings as ordinary sub-words, producing garbage.
    """
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Normalise every return type across transformers versions:
    #   list[int]          — transformers <5, most common
    #   str                — some v5 builds return rendered text even with tokenize=True
    #   tokenizers.Encoding — HF fast tokenizer backend; has .ids: list[int]
    #   BatchEncoding      — has .input_ids: list[int] or tensor
    if isinstance(result, str):
        # Rendered text: encode via the fast tokenizer so that special tokens
        # (e.g. <|begin_of_text|>) are mapped to their correct token IDs.
        ids: List[int] = tokenizer.encode(result, add_special_tokens=False)
    elif isinstance(result, (list, tuple)) and result and isinstance(result[0], int):
        ids = list(result)
    elif hasattr(result, "ids"):
        # tokenizers.Encoding (HF tokenizers backend)
        ids = list(result.ids)
    elif hasattr(result, "input_ids"):
        # BatchEncoding or similar wrapper. `result.input_ids` can be:
        #  - a Tensor of shape [1, L]
        #  - a list[list[int]] (batch of sequences)
        #  - a list[int] (single sequence)
        raw_ids = result.input_ids
        if hasattr(raw_ids, "tolist"):
            arr = raw_ids.tolist()
            if arr and isinstance(arr[0], list):
                ids = arr[0]
            else:
                ids = arr
        elif isinstance(raw_ids, list):
            if raw_ids and isinstance(raw_ids[0], list):
                ids = raw_ids[0]
            else:
                ids = raw_ids
        else:
            raise TypeError(
                f"_llama3_input_ids: unrecognised input_ids type {type(raw_ids).__name__!r}"
            )
    else:
        raise TypeError(
            f"_llama3_input_ids: unrecognised apply_chat_template return type "
            f"{type(result).__name__!r}  (value={repr(result)[:120]})"
        )
    return torch.tensor([ids], dtype=torch.long, device=device)


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

    # EAGLE's cnets.py _init_rope only understands rope_scaling types "linear" and
    # "dynamic" and requires a "factor" key.  Llama-3.1 uses type "llama3" with a
    # slightly different schema.  For the EAGLE *draft* model the RoPE precision
    # doesn't affect correctness testing, so we simply fall back to standard
    # (unscaled) RoPE for any type that cnets.py doesn't natively handle.
    try:
        from eagle.model import cnets as _cnets
        _EAGLE_ROPE_TYPES = {"linear", "dynamic"}
        _orig_init_rope = _cnets.LlamaAttention._init_rope
        def _patched_init_rope(self):
            rs = getattr(self.config, "rope_scaling", None)
            if isinstance(rs, dict):
                rope_type = rs.get("type") or rs.get("rope_type", "")
                if rope_type not in _EAGLE_ROPE_TYPES:
                    # Unsupported type — disable scaling so the fallback branch runs
                    # IMPORTANT: Deepcopy so we don't mutate the base model's shared config!
                    import copy
                    self.config = copy.deepcopy(self.config)
                    self.config.rope_scaling = None
                elif "type" not in rs:
                    # Has rope_type but not type — add alias
                    import copy
                    self.config = copy.deepcopy(self.config)
                    self.config.rope_scaling = {**rs, "type": rope_type}
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
    # EAGLE was authored and tested at float16.  EAGLE's own README notes
    # bf16 as a special case only for Qwen2 (numerical overflow).  For LLaMA
    # models the default is fp16 regardless of GPU capability.  On SM < 8.0
    # (Turing / T4) bf16 compute falls back to fp16 anyway, so fp16 is always
    # the right choice here.
    _dtype = torch.float16
    print(f"    dtype: float16 (EAGLE tested config)")

    model = EaModel.from_pretrained(
        use_eagle3=use_eagle3,
        base_model_path=base_model,
        ea_model_path=eagle_model,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=_dtype,
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
    use_ragged: bool = False,
    branching_factor: int = 4,
    max_depth: int = 7,
    profile: bool = False,
) -> List[GenerationRecord]:
    """
    Run Eagle-3 generation on every prompt by calling model.eagenerate() directly
    (no reimplemented loop) and monkey-patching tree_decoding for per-step timing.

    When use_ragged=True, SDPA is additionally patched inside each tree_decoding
    call to route the intra-tree Q×K block through the ragged Triton kernel.
    """
    import eagle.model.utils as _eagle_utils
    import eagle.model.ea_model as _eagle_ea

    prefill_device = next(model.base_model.parameters()).device
    total_token    = model.ea_layer.total_tokens
    records: List[GenerationRecord] = []

    # ── profiler setup ──────────────────────────────────────────────────────
    global _ACTIVE_PROFILER
    _profiler: Optional[_RaggedProfiler] = None
    if profile and use_ragged:
        _profiler = _RaggedProfiler()
        _ACTIVE_PROFILER = _profiler

    # ── per-call timing state (shared across the monkey-patch closure) ───────
    _state: Dict[str, object] = {
        "verify_ms":      [],    # List[float]
        "accepted":       [],    # List[int]  (accept_length + 1 per step)
        "prev_new_tok":   0,     # int – new_token value before this step
        "in_tree_verify": False, # True only while inside _orig_tree_decoding
        # ^^^ guards the matmul hook against firing during:
        #   • initialize_tree backbone prefill (all tokens, q_len=63)
        #   • initialize_tree incremental decode of accepted tokens (q_len=2-4)
        #   • draft-model cnets forward (unrelated matmuls)
        # The hook must ONLY run during tree_decoding(), where Q is the full
        # draft tree and the attention really is the sparse ancestor attention.
    }

    # ea_model.py uses `from .utils import *` so eagenerate() calls the names
    # bound in ea_model's own module namespace.  Patch there, not in utils.
    _orig_tree_decoding = _eagle_ea.tree_decoding
    _orig_update        = _eagle_ea.update_inference_inputs

    def _timed_tree_decoding(mdl, tree_candidates, past_key_values,
                              tree_position_ids, input_ids, retrieve_indices):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()

        # ── Extract parent array from EAGLE-3's tree_mask ────────────────────
        # EAGLE sets mdl.model.tree_mask = [1, 1, N, N] float before calling
        # tree_decoding.  N = number of tree tokens (root + selected candidates).
        # tree_position_ids[i] = depth of token i.
        # We reconstruct the parent array:  parents[i] = j  where
        #   tree_mask[i,j]=True  and  depth[j] = depth[i]-1.
        # Root (depth=0): parents[0] = 0  (self-loop).
        if use_ragged and hasattr(mdl, 'model') and hasattr(mdl.model, 'tree_mask'):
            tm = mdl.model.tree_mask  # [1, 1, N, N] float
            if tm is not None and tm.dim() == 4:
                tm_bool = tm[0, 0].bool()         # [N, N]
                N = tm_bool.shape[0]
                depths = tree_position_ids          # [N]
                if depths.dim() > 1:
                    depths = depths.squeeze(0)
                depths = depths[:N]                 # ensure matching length

                # Vectorised parent extraction:
                # For each node i, find col j where mask[i,j]=True and depth[j]=depth[i]-1
                target_depths = (depths - 1).unsqueeze(1)   # [N, 1]
                col_depths    = depths.unsqueeze(0)          # [1, N]
                depth_match   = (col_depths == target_depths)  # [N, N]
                valid         = tm_bool & depth_match          # [N, N]
                parents       = valid.int().argmax(dim=1).to(torch.int32)  # [N]
                parents[0]    = 0  # root self-loop
                _state["tree_parents"] = parents
            else:
                _state["tree_parents"] = None
        else:
            _state["tree_parents"] = None

        # Raise the flag so the matmul hook knows we are inside tree_decoding.
        # This is the ONLY window where the hook should intercept matmuls.
        # Without this guard the hook fires during initialize_tree's backbone
        # call on the newly-accepted tokens (q_len=2–4, kv_len >> q_len),
        # routing non-tree Q/K/V through the ragged kernel and crashing when
        # seq_len is not a power of 2.
        _state["in_tree_verify"] = True
        try:
            result = _orig_tree_decoding(
                mdl, tree_candidates, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )
        finally:
            _state["in_tree_verify"] = False
        e1.record()
        torch.cuda.synchronize()
        _state["verify_ms"].append(e0.elapsed_time(e1))  # type: ignore[attr-defined]
        return result

    def _tracked_update(input_ids, candidates, best_candidate, accept_length,
                        retrieve_indices, logits_processor, new_token,
                        past_key_values_data_list, current_length_data,
                        mdl, hidden_state_new, sample_p):
        result = _orig_update(
            input_ids, candidates, best_candidate, accept_length,
            retrieve_indices, logits_processor, new_token,
            past_key_values_data_list, current_length_data,
            mdl, hidden_state_new, sample_p,
        )
        # result[5] is new_token after the update step.
        new_tok_after = int(result[5])
        delta = new_tok_after - int(_state["prev_new_tok"])  # type: ignore[arg-type]
        _state["accepted"].append(delta)  # type: ignore[attr-defined]
        _state["prev_new_tok"] = new_tok_after
        return result

    _eagle_ea.tree_decoding           = _timed_tree_decoding
    _eagle_ea.update_inference_inputs = _tracked_update

    try:
        for pi, raw in enumerate(prompts):
            # ── tokenise ─────────────────────────────────────────────────────
            if is_llama3 and hasattr(model.tokenizer, "apply_chat_template"):
                input_ids = _llama3_input_ids(model.tokenizer, raw, prefill_device)
            else:
                prompt    = _get_prompt(model_type, raw)
                input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids.to(prefill_device)

            # Diagnostic on first prompt.
            if pi == 0:
                decoded_head = model.tokenizer.decode(input_ids[0, :16], skip_special_tokens=False)
                print(f"  [diag] input_ids shape={tuple(input_ids.shape)} "
                      f"device={input_ids.device}  "
                      f"head tokens: {decoded_head!r}")

            input_len = input_ids.shape[1]

            # ── reset per-prompt state ────────────────────────────────────────
            _state["verify_ms"]    = []
            _state["accepted"]     = []
            _state["prev_new_tok"] = 0
            if _profiler is not None:
                _profiler.reset()

            torch.cuda.synchronize()
            wall_t0 = time.perf_counter()

            # ── run generation via EAGLE's own loop (no reimplementation) ─────
            # When use_ragged=True, wrap the ENTIRE eagenerate() call with
            # ragged_eagle_context so torch.matmul is patched once for all
            # verify steps in this generation, not re-patched per step.
            # The q_len>1 guard in the state machine ensures the hook only
            # fires during tree verification (q_len=N_tree > 1), never during
            # the autoregressive backbone forward pass (q_len=1 always).
            if use_ragged:
                with ragged_eagle_context(branching_factor, max_depth,
                                          silent=True,
                                          verify_flag=_state) as _hook_state:
                    out_ids, new_token, n_steps_idx = model.eagenerate(
                        input_ids,
                        temperature=0.0,
                        max_new_tokens=max_new_tokens,
                        is_llama3=is_llama3,
                        log=True,
                    )
                _hook_fires = int(_hook_state["n"])  # type: ignore[arg-type]
            else:
                out_ids, new_token, n_steps_idx = model.eagenerate(
                    input_ids,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    is_llama3=is_llama3,
                    log=True,
                )
                _hook_fires = 0

            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_t0) * 1000

            verify_ms_list: List[float] = list(_state["verify_ms"])   # type: ignore[arg-type]
            accepted_list:  List[int]   = list(_state["accepted"])     # type: ignore[arg-type]
            n_steps  = len(verify_ms_list)
            total_a  = sum(accepted_list)
            mean_a   = total_a / n_steps if n_steps else 0.0
            acc_rate = mean_a  / (total_token + 1) if total_token else 0.0
            tot_vms  = sum(verify_ms_list)
            mean_vms = tot_vms / n_steps if n_steps else 0.0
            vfrac    = tot_vms / wall_ms  if wall_ms  else 0.0
            n_new    = int(new_token)

            rec = GenerationRecord(
                prompt=raw[:80],
                num_tokens=n_new,
                num_steps=n_steps,
                wall_ms=wall_ms,
                tok_per_sec=n_new / (wall_ms / 1000) if wall_ms else 0.0,
                mean_accepted_per_step=mean_a,
                acceptance_rate=acc_rate,
                mean_verify_ms=mean_vms,
                verify_fraction=vfrac,
            )
            records.append(rec)

            snippet = model.tokenizer.decode(
                out_ids[0, input_len: input_len + min(n_new, 50)],
                skip_special_tokens=True,
            )
            print(
                f"  [{pi+1}/{len(prompts)}] "
                f"{n_new} tok / {n_steps} steps  "
                f"accept={mean_a:.2f}/step ({acc_rate:.1%})  "
                f"{rec.tok_per_sec:.1f} tok/s  "
                f"verify={mean_vms:.1f} ms/step ({vfrac:.0%} of time)  "
                + (f"[hook={_hook_fires}×]  " if use_ragged and _hook_fires else "")
                + f"→ \"{snippet[:60]}...\""
            )

            # ── profiler summary per prompt ──────────────────────────────────
            if _profiler is not None and _profiler.entries:
                # Events already synced from wall_ms measurement
                print(_profiler.summary())

    finally:
        # Always restore originals even if an exception occurs.
        _eagle_ea.tree_decoding           = _orig_tree_decoding
        _eagle_ea.update_inference_inputs = _orig_update
        # Clear module-level profiler reference
        if _profiler is not None:
            _ACTIVE_PROFILER = None

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Ragged-kernel attention replacement + context manager
# ─────────────────────────────────────────────────────────────────────────────

def _ragged_tree_attn(
    Q:  torch.Tensor,   # [B, H, N_q, D]  — post-RoPE query
    K:  torch.Tensor,   # [B, H, N_kv, D] — full KV (prefix + tree)
    V:  torch.Tensor,   # [B, H, N_kv, D]
    branching_factor: int,
    max_depth: int,
    tree_parents: Optional[torch.Tensor] = None,
) -> torch.Tensor:      # [B, H, N_q, D]
    """
    Ragged ancestor-sparse attention for Eagle-3 tree-verification.

    Splits K/V into prefix (dense causal) and tree (sparse ancestor) parts,
    runs flash attention over the prefix and the ragged Triton kernel over
    the tree, then merges via online-softmax LSE combination.

    When ``tree_parents`` is provided (int32 [N_tree] — seq-local parent index),
    uses the explicit-parent kernel variant for EAGLE-3's dynamic/pruned trees.
    Otherwise falls back to the formula-based kernel (complete b-ary trees only).

    When ``_ACTIVE_PROFILER`` is set, records 7 CUDA events per call
    (one at each phase boundary) for zero-overhead timing.
    """
    prof = _ACTIVE_PROFILER  # read module-level profiler

    B, H, N_q, D = Q.shape
    N_kv         = K.shape[2]
    N_prefix     = N_kv - N_q
    scale_v      = 1.0 / math.sqrt(D)
    dtype        = Q.dtype

    if prof is not None:
        evts = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        evts[0].record()                                        # ── [0] START
        prof.record_shapes(Q, K, N_prefix, N_q)

    # ── Part 1: dense prefix attention ──────────────────────────────────────
    if N_prefix > 0:
        K_pre = K[:, :, :N_prefix, :].contiguous()   # [B, H, N_prefix, D]
        V_pre = V[:, :, :N_prefix, :].contiguous()

    if prof is not None:
        evts[1].record()                                        # ── [1] prefix_split done

    if N_prefix > 0:
        try:
            out_pre, lse_pre, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                Q.contiguous(), K_pre, V_pre,
                dropout_p=0.0, is_causal=False, scale=scale_v,
                return_debug_mask=False,
            )
        except Exception:
            sc       = torch.ops.aten.matmul(Q, K_pre.transpose(-2, -1)) * scale_v
            lse_pre  = torch.logsumexp(sc.float(), dim=-1)     # [B, H, N_q]
            out_pre  = torch.softmax(sc, dim=-1) @ V_pre       # [B, H, N_q, D]

    if prof is not None:
        evts[2].record()                                        # ── [2] flash_prefix done

    # ── Part 2: ragged intra-tree attention ─────────────────────────────────
    K_tree = K[:, :, N_prefix:, :].contiguous()   # [B, H, N_q, D]
    V_tree = V[:, :, N_prefix:, :].contiguous()

    # Pack [B, H, N_q, D] → [B*N_q, H, D]
    Q_r = Q.permute(0, 2, 1, 3).reshape(B * N_q, H, D).contiguous()
    K_r = K_tree.permute(0, 2, 1, 3).reshape(B * N_q, H, D).contiguous()
    V_r = V_tree.permute(0, 2, 1, 3).reshape(B * N_q, H, D).contiguous()
    cu  = torch.arange(0, (B + 1) * N_q, N_q, dtype=torch.int32, device=Q.device)

    if prof is not None:
        evts[3].record()                                        # ── [3] tree_reshape done

    # Use explicit-parent kernel when EAGLE-3 parent array is available.
    if tree_parents is not None and tree_parents.shape[0] == N_q:
        # Pack parent array for batch: replicate per batch element
        # tree_parents is [N_q] in local space; for packed [B*N_q] we tile it.
        parents_packed = tree_parents.repeat(B)  # [B*N_q]
        out_tree_r, lse_tree_r = ragged_attention_with_parents(
            Q_r, K_r, V_r, cu, parents_packed, max_depth)
    else:
        out_tree_r, lse_tree_r = ragged_attention_with_lse(
            Q_r, K_r, V_r, cu, branching_factor, max_depth)

    if prof is not None:
        evts[4].record()                                        # ── [4] ragged_kernel done

    out_tree = out_tree_r.view(B, N_q, H, D).permute(0, 2, 1, 3)   # [B…]
    lse_tree = lse_tree_r.view(B, N_q, H).permute(0, 2, 1)          # [B, H, N_q]

    if prof is not None:
        evts[5].record()                                        # ── [5] out_reshape done

    # ── Part 3: online-softmax merge ────────────────────────────────────────
    if N_prefix == 0:
        result = out_tree.to(dtype)
    else:
        lse_p   = lse_pre.float()
        lse_t   = lse_tree.float()
        lse_max = torch.maximum(lse_p, lse_t)
        w_p     = torch.exp(lse_p - lse_max)
        w_t     = torch.exp(lse_t - lse_max)
        w_sum   = (w_p + w_t).clamp_min(1e-8).unsqueeze(-1)
        result  = ((w_p.unsqueeze(-1) * out_pre.float()
                    + w_t.unsqueeze(-1) * out_tree.float()) / w_sum).to(dtype)

    if prof is not None:
        evts[6].record()                                        # ── [6] lse_merge done
        prof.entries.append(evts)

    return result


@contextlib.contextmanager
def ragged_eagle_context(branching_factor: int, max_depth: int,
                         silent: bool = False,
                         verify_flag: Optional[Dict[str, object]] = None):
    """
    Context manager that replaces Eagle-3's tree-verification attention
    (the dense QK-softmax-AV triple) with the ragged Triton kernel.

    IMPORTANT — apply this ONCE per eagenerate() call, NOT per verify step.
    Entering/exiting this context patches and unpatches ``torch.matmul``
    globally.  Wrapping individual tree_decoding steps instead would add
    Python patch/unpatch + print overhead on every step (~75× per prompt),
    easily negating any kernel speedup.

    Root cause of previous SDPA-patch failure:
      EAGLE's modeling_llama_kv.py and cnets.py both implement attention via
      manual  ``torch.matmul(Q, K^T) / scale``  + ``softmax``  + ``matmul(A, V)``.
      They NEVER call ``F.scaled_dot_product_attention``, so patching SDPA
      had zero effect — the hook fired 0 times.

    This implementation patches ``torch.matmul`` with a shape-aware state
    machine that identifies the two attention matmuls by their 4-D shapes:

      QK step:  matmul([B,H,q,D], [B,H,D,kv])  where q>1 and kv>q
                → save Q and K; return zeros placeholder (discarded by softmax)

      AV step:  matmul([B,H,q,kv], [B,H,kv,D])  (immediately after QK)
                → ignore the garbage softmax weights in A;
                  use saved Q, K and current V to run _ragged_tree_attn;
                  return the correct ragged output.

    Safety:
      • ``verify_flag`` is a dict from ``run_generation``'s ``_state`` containing
        ``in_tree_verify: bool``.  ``_timed_tree_decoding`` sets it True only
        for the duration of the actual verify call, and the hook is a no-op
        outside that window.  This prevents false positives during:
          - initialize_tree backbone prefill (input tokens, q_len=63)
          - initialize_tree incremental decode (accepted tokens, q_len=2–4
            with kv_len >> q_len — would otherwise satisfy the shape guard)
          - draft-model cnets forward pass
      • Re-entrancy guard (``_inside`` flag) prevents the prefix-flash fallback
        inside ``_ragged_tree_attn`` from re-triggering the state machine.

    Yields the internal state dict so callers can read ``state["n"]`` (total
    hook fires = layers × steps) after the context exits.

    Args:
        branching_factor: tree branching factor (= EAGLE top_k).
        max_depth:        maximum tree depth.
        silent:           if True, suppress the summary print on exit.
        verify_flag:      dict with key ``"in_tree_verify"`` (bool) shared
                          with ``_timed_tree_decoding``.  When provided, the
                          hook is a no-op unless the flag is True.
    """
    _orig_mm = torch.matmul

    # State: q/kt set while waiting for the AV matmul; cleared after.
    _s: Dict[str, object] = {"q": None, "kt": None, "n": 0, "inside": False}

    def _patched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        _prof = _ACTIVE_PROFILER  # read once

        # Gate 1: only intercept when we're inside tree_decoding().
        # Without this, the hook fires during initialize_tree's backbone call
        # on newly-accepted tokens (q_len=2–4, kv_len>>q_len), routing
        # non-tree Q/K/V through the ragged kernel and crashing when seq_len
        # is not a power of 2 for Triton's tl.arange.
        if verify_flag is not None and not verify_flag["in_tree_verify"]:
            if _prof is not None:
                _prof.hook_total += 1
                _prof.hook_skip_verify += 1
            return _orig_mm(A, B)

        # Gate 2: re-entrancy guard — calls from inside _ragged_tree_attn bypass us.
        if _s["inside"]:
            if _prof is not None:
                _prof.hook_total += 1
                _prof.hook_skip_inside += 1
            return _orig_mm(A, B)

        if _prof is not None:
            _prof.hook_total += 1

        if A.dim() == 4 and B.dim() == 4:
            q_len  = A.shape[2]
            kv_len = B.shape[3]   # B is already transposed: [B,H,D,kv]

            if _s["q"] is None and q_len > 1 and kv_len > q_len:
                # ── QK matmul: save Q=[B,H,q,D] and K^T=[B,H,D,kv] ──────────
                # Return zeros; the downstream softmax produces garbage weights
                # that we discard at the AV step.
                _s["q"]  = A
                _s["kt"] = B
                if _prof is not None:
                    _prof.hook_intercept += 1
                return torch.zeros(A.shape[0], A.shape[1], q_len, kv_len,
                                   device=A.device, dtype=A.dtype)

            elif _s["q"] is not None:
                # ── AV matmul: ignore A (garbage softmax), run ragged ────────
                Q  = _s["q"]                    # [B, H, q_len, D]
                K  = _s["kt"].transpose(-2, -1) # [B, H, kv_len, D]
                V  = B                          # [B, H, kv_len, D]
                _s["q"] = _s["kt"] = None
                _s["n"] = _s["n"] + 1           # type: ignore[assignment]
                if _prof is not None:
                    _prof.hook_intercept += 1
                # Read parent array extracted in _timed_tree_decoding
                _tree_parents = (verify_flag.get("tree_parents")
                                 if verify_flag is not None else None)
                _s["inside"] = True
                try:
                    return _ragged_tree_attn(Q, K, V, branching_factor, max_depth,
                                             tree_parents=_tree_parents)
                finally:
                    _s["inside"] = False

        if _prof is not None:
            _prof.hook_skip_shape += 1
        return _orig_mm(A, B)

    torch.matmul = _patched_matmul          # type: ignore[assignment]
    try:
        yield _s                            # caller can inspect _s["n"]
    finally:
        torch.matmul = _orig_mm             # type: ignore[assignment]
        if not silent:
            n_fires = int(_s["n"])          # type: ignore[arg-type]
            n_layers = 32                   # LLaMA-3.1-8B
            n_steps = n_fires // max(1, n_layers)
            print(f"  [ragged] hook fired {n_fires}× total  "
                  f"({n_layers} layers × {n_steps} verify steps)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eagle-3 E2E benchmark: ragged kernel vs. vanilla Eagle-3\n"
                    "Sweeps over tree-depth configs to show where our ragged\n"
                    "kernel wins and where it doesn't.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument("--base-model",  default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle-model", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--model-type",  default="llama-3-instruct",
                        choices=["llama-3-instruct", "llama3", "llama2", "vicuna"])
    parser.add_argument("--no-eagle3",   action="store_true",
                        help="Use EAGLE-2 (not EAGLE-3)")
    # Tree sweep — 2D grid: depths × branching factors
    parser.add_argument("--depths",
                        default=",".join(map(str, DEFAULT_DEPTH_SWEEP)),
                        help="Comma-separated tree depths — spans loss→crossover→win "
                             f"(default: {','.join(map(str, DEFAULT_DEPTH_SWEEP))})")
    parser.add_argument("--branching-factors",
                        default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)),
                        help="Comma-separated top-k / branching factors "
                             f"(default: {','.join(map(str, DEFAULT_BRANCHING_FACTORS))}). "
                             "b=10 is the EAGLE-3 default.")
    parser.add_argument("--total-tokens-map", default=None,
                        help="Override total_token for specific (b,d) pairs, "
                             "e.g. 'b10d7:60,b12d9:100'. "
                             "Unspecified pairs use the built-in formula.")
    # Generation
    parser.add_argument("--num-prompts",    type=int, default=10,
                        help="Number of ShareGPT prompts per config (default: 10)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # Control
    parser.add_argument("--skip-vanilla",  action="store_true",
                        help="Skip vanilla Eagle-3 runs (only run ragged)")
    parser.add_argument("--skip-ragged",   action="store_true",
                        help="Skip ragged-kernel runs (only run vanilla)")
    parser.add_argument("--out-dir",  default="results")
    parser.add_argument("--csv-name", default="e2e_benchmark.csv")
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token (overrides $HF_TOKEN)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-component profiling in _ragged_tree_attn. "
                             "Prints a breakdown of prefix_split / flash_prefix / "
                             "tree_reshape / ragged_kernel / out_reshape / lse_merge "
                             "timing per prompt, plus hook dispatch stats.")
    args = parser.parse_args()

    # ── HF token propagation ─────────────────────────────────────────────────
    _hf_token = (args.hf_token
                 or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)
    if not HAS_EAGLE:
        print("ERROR: EAGLE not installed.")
        print("  pip install git+https://github.com/SafeAILab/EAGLE.git fschat")
        sys.exit(1)

    is_eagle3 = not args.no_eagle3
    is_llama3 = args.model_type in ("llama3", "llama-3-instruct")

    # ── Build 2D tree-config sweep (depths × branching factors) ────────────
    depths  = [int(x) for x in args.depths.split(",")]
    bfacs   = [int(x) for x in args.branching_factors.split(",")]

    # Parse optional per-cell total-token overrides: 'b10d7:60,b12d9:100'
    tt_overrides: dict = {}
    if args.total_tokens_map:
        for pair in args.total_tokens_map.split(","):
            key, val = pair.split(":")
            # key format: b{B}d{D}
            import re as _re
            m = _re.match(r"b(\d+)d(\d+)", key.strip())
            if m:
                tt_overrides[(int(m.group(1)), int(m.group(2)))] = int(val)

    configs: List[TreeConfig] = []
    for b in bfacs:
        for d in depths:
            tt = tt_overrides.get((b, d), _default_total_token(b, d))
            is_default = (b == 10 and d == 7)
            label = f"b={b} d={d} tt={tt}" + (" [Eagle-3 default]" if is_default else "")
            configs.append(TreeConfig(depth=d, total_token=tt, top_k=b, label=label))

    prompts = _load_sharegpt_prompts(
        args.num_prompts, seed=args.prompt_seed, hf_token=_hf_token
    )

    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Eagle-3 E2E Benchmark  ·  2D Sweep: depth × branching factor")
    print("  Narrative (micro-benchmark validated on H100 SXM):")
    print("    d=5–7  : LOSS regime   (N≈43–60, kernel overhead > savings)")
    print("    d=9–12 : CROSSOVER     (N≈77–103, B-dependent win/loss)")
    print("    d≥16   : WIN  regime   (N≥137, ancestor walk dominates)")
    print("  L=0 crossovers: B=8→N≥103 | B=32→N≥51 | L≥1024→always wins")
    print("=" * 72)
    print(f"  Base:             {args.base_model}")
    print(f"  Eagle:            {args.eagle_model}  "
          f"({'EAGLE-3' if is_eagle3 else 'EAGLE-2'})")
    print(f"  Prompts:          {args.num_prompts} per config (seed={args.prompt_seed})")
    print(f"  Depths:           {depths}")
    print(f"  Branching (top_k):{bfacs}")
    print(f"  Total configs:    {len(configs)}  ({len(depths)} depths × {len(bfacs)} b-factors)")
    # Print token budget grid with crossover markers
    print(f"  Token budget grid (total_token per (b, d))  — N≥103 → B=8 wins:")
    hdr_b = "  " + " " * 10 + "".join(f"  d={d:2d}" for d in depths)
    print(hdr_b)
    for b in bfacs:
        row_tt = "".join(
            f"  {_default_total_token(b,d):4d}" + (
                "★" if b==10 and d==7 else
                "↑" if _default_total_token(b,d) >= 103 else " "
            )
            for d in depths
        )
        mark = " ← EAGLE-3 default" if b == 10 else ""
        print(f"  b={b:>2d} (top_k)  {row_tt}{mark}")
    print(f"  (★ = EAGLE-3 default  |  ↑ = above B=8 kernel crossover N≥103)")
    if _hf_token:
        print(f"  HF token:         {_hf_token[:8]}…")
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU:              {p.name}  SM {p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
    if args.profile:
        print(f"  PROFILE MODE:     ON  (per-component timing in ragged path)")
    print("=" * 72)

    # ── Load model once at the deepest config ────────────────────────────────
    max_cfg = max(configs, key=lambda c: c.total_token)
    model = load_eagle_model(
        base_model=args.base_model,
        eagle_model=args.eagle_model,
        use_eagle3=is_eagle3,
        total_token=max_cfg.total_token,
        depth=max_cfg.depth,
        top_k=max_cfg.top_k,
    )

    # ── Warmup: trigger Triton JIT compilation before timed runs ────────────
    # The first time the ragged kernel sees a new (HEAD_DIM, MAX_DEPTH) combo,
    # Triton compiles + autotunes, adding 10-15× overhead to that call.
    # We run one short generation to pay that cost outside the measurement loop.
    if not args.skip_ragged:
        print("\n  [warmup]  Compiling Triton kernels (1 prompt, short) …")
        set_tree_config(model, configs[0])
        warmup_prompts = [prompts[0][:200]]  # first prompt, truncated
        _warmup = run_generation(
            model, warmup_prompts, args.model_type,
            max_new_tokens=32,
            is_llama3=is_llama3,
            use_ragged=True,
            branching_factor=configs[0].top_k,
            max_depth=configs[0].depth,
            profile=False,
        )
        if _warmup:
            print(f"  [warmup]  done ({_warmup[0].tok_per_sec:.1f} tok/s, "
                  f"accept={_warmup[0].mean_accepted_per_step:.2f}/step)")
        else:
            print("  [warmup]  completed (no output)")
        del _warmup
        torch.cuda.empty_cache()

    # ── Sweep ────────────────────────────────────────────────────────────────
    # Per-config aggregated results:
    #   config_results[i] = { "vanilla": List[GenRecord], "ragged": List[GenRecord] }
    all_rows: List[dict] = []
    summary_rows: List[dict] = []   # one row per config for the final table

    for ci, cfg in enumerate(configs):
        print(f"\n{'━' * 72}")
        print(f"  CONFIG {ci+1}/{len(configs)}  —  {cfg.label}")
        print(f"    depth={cfg.depth}  total_token={cfg.total_token}  top_k={cfg.top_k}")
        print(f"{'━' * 72}")

        set_tree_config(model, cfg)

        v_records: List[GenerationRecord] = []
        r_records: List[GenerationRecord] = []

        # ── Vanilla Eagle-3 ──────────────────────────────────────────────────
        if not args.skip_vanilla:
            print(f"\n  [vanilla]  depth={cfg.depth}  total_token={cfg.total_token}")
            v_records = run_generation(
                model, prompts, args.model_type,
                max_new_tokens=args.max_new_tokens,
                is_llama3=is_llama3,
                use_ragged=False,
                profile=args.profile,
            )
            if v_records:
                v_tok = np.mean([r.tok_per_sec for r in v_records])
                v_acc = np.mean([r.mean_accepted_per_step for r in v_records])
                v_vms = np.mean([r.mean_verify_ms for r in v_records])
                v_vfr = np.mean([r.verify_fraction for r in v_records])
                print(f"    → {v_tok:.1f} tok/s   {v_acc:.2f} acc/step   "
                      f"verify={v_vms:.1f} ms ({v_vfr:.0%})")

        # ── Ragged-kernel Eagle-3 ────────────────────────────────────────────
        if not args.skip_ragged:
            print(f"\n  [ragged]   depth={cfg.depth}  total_token={cfg.total_token}")
            r_records = run_generation(
                model, prompts, args.model_type,
                max_new_tokens=args.max_new_tokens,
                is_llama3=is_llama3,
                use_ragged=True,
                branching_factor=cfg.top_k,
                max_depth=cfg.depth,
                profile=args.profile,
            )
            if r_records:
                r_tok = np.mean([r.tok_per_sec for r in r_records])
                r_acc = np.mean([r.mean_accepted_per_step for r in r_records])
                r_vms = np.mean([r.mean_verify_ms for r in r_records])
                r_vfr = np.mean([r.verify_fraction for r in r_records])
                print(f"    → {r_tok:.1f} tok/s   {r_acc:.2f} acc/step   "
                      f"verify={r_vms:.1f} ms ({r_vfr:.0%})")

        # ── Per-config speedup ───────────────────────────────────────────────
        if v_records and r_records:
            v_mean = np.mean([r.tok_per_sec for r in v_records])
            r_mean = np.mean([r.tok_per_sec for r in r_records])
            speedup = r_mean / v_mean if v_mean > 0 else float("nan")
            print(f"\n  ▸ E2E speedup at d={cfg.depth}:  {speedup:.3f}×  "
                  f"({v_mean:.1f} → {r_mean:.1f} tok/s)")

        # ── Collect CSV rows ─────────────────────────────────────────────────
        for mode_label, recs in [("vanilla", v_records), ("ragged", r_records)]:
            for r in recs:
                all_rows.append({
                    "depth":                    cfg.depth,
                    "total_token":              cfg.total_token,
                    "top_k":                    cfg.top_k,
                    "config_label":             cfg.label,
                    "mode":                     mode_label,
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

        # ── Summary row ──────────────────────────────────────────────────────
        s = {"depth": cfg.depth, "total_token": cfg.total_token, "top_k": cfg.top_k,
             "label": cfg.label}
        if v_records:
            s["vanilla_tok_s"]   = round(float(np.mean([r.tok_per_sec for r in v_records])), 1)
            s["vanilla_acc"]     = round(float(np.mean([r.mean_accepted_per_step for r in v_records])), 3)
            s["vanilla_verify"]  = round(float(np.mean([r.verify_fraction for r in v_records])), 4)
        if r_records:
            s["ragged_tok_s"]    = round(float(np.mean([r.tok_per_sec for r in r_records])), 1)
            s["ragged_acc"]      = round(float(np.mean([r.mean_accepted_per_step for r in r_records])), 3)
            s["ragged_verify"]   = round(float(np.mean([r.verify_fraction for r in r_records])), 4)
        if v_records and r_records:
            s["e2e_speedup"] = round(s["ragged_tok_s"] / s["vanilla_tok_s"], 4) \
                if s.get("vanilla_tok_s", 0) > 0 else float("nan")
        summary_rows.append(s)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    # ── Write CSV ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Saved: {csv_path}")

    # Summary CSV — one row per (b, d) config.
    summary_csv = os.path.join(args.out_dir, "e2e_summary.csv")
    if summary_rows:
        # Collect all keys across rows (some may be absent if a mode was skipped)
        s_fields: List[str] = []
        for row in summary_rows:
            for k in row.keys():
                if k not in s_fields:
                    s_fields.append(k)
        for row in summary_rows:
            for k in s_fields:
                row.setdefault(k, "")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=s_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved: {summary_csv}")

    # ── Final summary: 2D pivot tables ───────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  E2E BENCHMARK COMPLETE  —  depth × branching factor grid")
    print(f"{'=' * 72}")

    # Helper: look up summary value for (b, d)
    def _s(b: int, d: int, key: str):
        for row in summary_rows:
            if row["top_k"] == b and row["depth"] == d:
                v = row.get(key, "")
                return v if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and math.isnan(v)
                ) else None
        return None

    def _pivot(metric_key: str, fmt: str, title: str) -> None:
        print(f"\n  {title}")
        col_w = 8
        # header row
        hline = "  " + f"{'b \\ d':>6s}" + "".join(f"{d:>{col_w}d}" for d in depths)
        print(hline)
        print("  " + "─" * (len(hline) - 2))
        for b in bfacs:
            cells = []
            for d in depths:
                v = _s(b, d, metric_key)
                if v is None:
                    cells.append(f"{'—':>{col_w}s}")
                else:
                    mark = "★" if b == 10 and d == 7 else " "
                    cells.append(f"{fmt.format(v)+mark:>{col_w}s}")
            b_mark = " ← default" if b == 10 else ""
            print(f"  {f'b={b}':>6s}" + "".join(cells) + b_mark)

    _pivot("e2e_speedup",    "{:.3f}×", "E2E Speedup  (ragged tok/s / vanilla tok/s — >1 means ragged wins)")
    _pivot("vanilla_tok_s",  "{:.1f}",  "Vanilla Eagle-3  tok/s")
    _pivot("ragged_tok_s",   "{:.1f}",  "Ragged-kernel Eagle-3  tok/s")
    _pivot("vanilla_acc",    "{:.2f}",  "Mean accepted tokens/step  (vanilla)")

    # ── Per-b crossover analysis ──────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print("  Crossover analysis  (min depth where ragged first beats vanilla, per b)")
    print(f"  {'─' * 68}")
    all_speedups = [
        (s["top_k"], s["depth"], s["e2e_speedup"])
        for s in summary_rows
        if isinstance(s.get("e2e_speedup"), (int, float))
        and not math.isnan(float(s["e2e_speedup"]))
    ]
    if all_speedups:
        best_overall = max(all_speedups, key=lambda x: x[2])
        print(f"  Overall best: {best_overall[2]:.3f}× at b={best_overall[0]}, d={best_overall[1]}")
        print()
        for b in bfacs:
            group = sorted(
                [(d, sp) for bv, d, sp in all_speedups if bv == b],
                key=lambda x: x[0],
            )
            wins  = [d for d, sp in group if sp > 1.0]
            b_mark = " (EAGLE-3 default top_k ★)" if b == 10 else ""
            if wins:
                best_d, best_sp = max(group, key=lambda x: x[1])
                xover = min(wins)
                print(f"  b={b}{b_mark}:  ragged wins at d ≥ {xover}  "
                      f"(best {best_sp:.3f}× at d={best_d})")
            else:
                print(f"  b={b}{b_mark}:  ragged does not win at any tested depth")
                print(f"    → expected: extend depths beyond {max(d for d,_ in group)} "
                      f"or run with larger batch (use continuous-batching scenario)")
    # ── Micro-benchmark reference ─────────────────────────────────────────────
    print()
    print("  Reference: micro-benchmark crossover thresholds (H100 SXM, L=0):")
    for b in bfacs:
        for d in depths:
            tt = _s(b, d, 'total_token')
            if tt is None:
                tt = _default_total_token(b, d)
            n = tt
            regime = ("WIN ↑ " if n >= 103 else
                      "CROSS" if n >= 51 else
                      "LOSS ↓")
            print(f"    b={b:>2d} d={d:>2d}  N≈{n:>3d}  [{regime}]" +
                  ("  ← EAGLE-3 default" if b == 10 and d == 7 else ""))
        print()
    print("  Kernel microbenchmarks (SDPA vs FlashInfer vs DeFT): benchmark_sota.py")
    print("  Paper-figure micro-bench (BS×b×d×L):                 benchmark_micro.py")
    print(f"{'=' * 72}")
    print()


if __name__ == "__main__":
    main()
