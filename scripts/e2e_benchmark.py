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

# ── Dependency shims ─────────────────────────────────────────────────────────
# Must run before any transformers / EAGLE import.
#
# Problem: transformers 4.53.1 imports `kernels`, but kernels 0.13 uses a
# huggingface_hub 1.x dataclass API that is broken / incompatible in this env.
# Solution: inject a minimal `kernels` stub into sys.modules so transformers
# gets the `get_kernel` symbol it needs without loading the real broken package.
import sys as _sys, types as _types, importlib.machinery as _ilm
if "kernels" not in _sys.modules:
    _km = _types.ModuleType("kernels")
    _km.get_kernel = lambda *a, **kw: None
    _km.__spec__ = _ilm.ModuleSpec("kernels", None)   # required by importlib.util.find_spec
    _km.__version__ = "0.0.0-stub"
    # Pre-register sub-modules so any further `from kernels.x import y` is a no-op.
    for _sub in ("kernels.layer", "kernels.utils", "kernels.deps"):
        _sm = _types.ModuleType(_sub)
        _sm.__spec__ = _ilm.ModuleSpec(_sub, None)
        _sys.modules.setdefault(_sub, _sm)
    _sys.modules["kernels"] = _km
del _sys, _types, _ilm
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import contextlib
import csv
import datetime
import importlib.util
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Suppress verbose HuggingFace/transformers warnings that clutter output
warnings.filterwarnings(
    "ignore",
    message=r".*has generative capabilities.*doesn.t directly inherit from.*GenerationMixin.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*GenerationMixin.*",
    category=UserWarning,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ragged_attn import ragged_attention_with_lse, ragged_attention_with_parents, fused_lse_merge

# EAGLE internals — needed for module-level attention forward patch
from eagle.model.modeling_llama_kv import (
    LlamaRotaryEmbedding_L31,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_L31,
    repeat_kv,
)


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

# ── Vanilla attention timer (Exp 1 — Amdahl table) ───────────────────────────
# When --attn-profile is set, we patch F.scaled_dot_product_attention once at
# module load to record per-call CUDA events.  The _IN_ATTN_PROFILE flag ensures
# timing only fires during tree_verify (never during prefill or draft-model).

_IN_ATTN_PROFILE: bool = False
_ATTN_EVENTS: List[Tuple["torch.cuda.Event", "torch.cuda.Event"]] = []

_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_timed(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, **kwargs):
    if not _IN_ATTN_PROFILE:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, **kwargs)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    out = _orig_sdpa(query, key, value, attn_mask=attn_mask,
                     dropout_p=dropout_p, is_causal=is_causal,
                     scale=scale, **kwargs)
    e1.record()
    _ATTN_EVENTS.append((e0, e1))
    return out


torch.nn.functional.scaled_dot_product_attention = _sdpa_timed


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
    if tx < (4, 46, 2):
        problems.append(
            f"transformers {'.'.join(str(x) for x in tx)} < 4.46.2  "
            "(EAGLE requires >=4.46.2)"
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
        print("    Run:  pip install 'transformers==4.46.2' 'accelerate>=0.26.0,<1.0'")
    else:
        print("  OK")


_check_env_versions()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch_transformers_for_eagle() -> None:
    """
    Compatibility shim so EAGLE 3.0.0 imports cleanly across transformers versions.

    EAGLE's modeling_qwen3_kv.py references symbols that were added or moved
    across transformers 4.46–5.x.  We inject no-op stubs for any missing symbol
    rather than hard-requiring a specific version.

    Patched modules / symbols:
      transformers.utils        — LossKwargs, auto_docstring, can_return_tuple
      transformers.integrations — use_kernel_forward_from_hub  (added ~4.53)
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
        pass

    try:
        import transformers.integrations as _ti

        if not hasattr(_ti, "use_kernel_forward_from_hub"):
            def _use_kernel_forward_from_hub(*args, **kwargs):
                # No-op passthrough decorator — kernel hub not available.
                if args and callable(args[0]):
                    return args[0]
                return lambda fn: fn
            _ti.use_kernel_forward_from_hub = _use_kernel_forward_from_hub  # type: ignore[attr-defined]

    except Exception:
        pass


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
# Context-length padding (growing-context simulation)
# ─────────────────────────────────────────────────────────────────────────────

# arXiv paper used as the realistic long-context document.
# FlashAttention-2 (Dao 2023) — on-topic and ~12k tokens of clean prose.
_ARXIV_DOC_ID    = "1706.03762"   # "Attention Is All You Need" — confirmed HTML available
_ARXIV_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
_ARXIV_CACHE_PATH = os.path.join(_ARXIV_CACHE_DIR, f"arxiv_{_ARXIV_DOC_ID}.txt")

# Fallback text used only when the download fails.
_FALLBACK_FILLER = (
    "Large language models have demonstrated remarkable capabilities across a "
    "wide range of natural language processing tasks. These models are trained "
    "on large corpora of text using the next-token prediction objective. During "
    "inference the model generates text autoregressively, one token at a time, "
    "using the key-value cache to avoid recomputing attention over previously "
    "processed tokens. Speculative decoding accelerates this process by drafting "
    "multiple candidate tokens ahead of the target model and verifying them in "
    "parallel, achieving super-linear speedups when the draft model has a high "
    "acceptance rate. "
)


def _fetch_arxiv_text(paper_id: str, cache_path: str) -> str:
    """Download an arXiv paper's HTML export and return stripped plain text.

    Result is cached to *cache_path* so subsequent runs don't hit the network.
    Falls back to _FALLBACK_FILLER on any network/parse error.
    """
    import urllib.request
    import re as _re

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if text:
            print(f"  [prefix] Loaded arXiv {paper_id} from cache ({len(text):,} chars)")
            return text

    url = f"https://arxiv.org/html/{paper_id}"
    print(f"  [prefix] Downloading arXiv {paper_id} from {url} …")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [prefix] Download failed ({exc}); using fallback filler.")
        return ""

    # Strip HTML tags, collapse whitespace.
    text = _re.sub(r"<[^>]+>", " ", html)
    text = _re.sub(r"\s+", " ", text).strip()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  [prefix] Downloaded and cached ({len(text):,} chars)")
    return text


_DOC_TEXT_CACHE: Optional[str] = None


def _get_doc_text() -> str:
    global _DOC_TEXT_CACHE
    if _DOC_TEXT_CACHE is None:
        _DOC_TEXT_CACHE = _fetch_arxiv_text(_ARXIV_DOC_ID, _ARXIV_CACHE_PATH)
        if not _DOC_TEXT_CACHE:
            _DOC_TEXT_CACHE = _FALLBACK_FILLER
    return _DOC_TEXT_CACHE


def _apply_chat_template_ids(tokenizer, messages: list, add_generation_prompt: bool) -> List[int]:
    """Wrapper around apply_chat_template that always returns List[int]."""
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(result, str):
        return tokenizer.encode(result, add_special_tokens=False)
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], int):
        return list(result)
    if hasattr(result, "ids"):
        return list(result.ids)
    if hasattr(result, "input_ids"):
        raw = result.input_ids
        arr = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return arr[0] if arr and isinstance(arr[0], list) else arr
    return list(result)


def _build_prefix_ids(
    target_len: int,
    tokenizer,
    device: torch.device,
    is_llama3: bool = False,
) -> torch.Tensor:
    """Build a [1, target_len] KV-cache prefix from a real arXiv document.

    The document (FlashAttention-2 paper by default) is downloaded once and
    cached to .cache/.  For LLaMA-3/Instruct models it is wrapped in a chat
    template as a user-turn share + brief assistant acknowledgment so the
    sequence structure is valid.  The caller must strip BOS from the actual
    prompt before concatenating (prefix already starts with BOS).

    For non-LLaMA-3 models the raw document tokens are used (no template).
    """
    doc_text = _get_doc_text()
    doc_ids: List[int] = tokenizer.encode(doc_text, add_special_tokens=False)
    if not doc_ids:
        doc_ids = tokenizer.encode(_FALLBACK_FILLER, add_special_tokens=False) or [13] * 50

    if not is_llama3 or not hasattr(tokenizer, "apply_chat_template"):
        # Raw token tiling — no chat structure.
        reps = math.ceil(target_len / len(doc_ids)) + 1
        return torch.tensor([(doc_ids * reps)[:target_len]], dtype=torch.long, device=device)

    # Wrap as: [BOS][user]{document…}[eot][assistant]{ack}[eot]
    # The user turn is filled with the document (tiled if needed).
    # Caller appends the actual prompt after stripping its BOS.
    ack = (
        "I've read the document carefully and I'm ready to answer your questions about it."
    )
    ack_ids = tokenizer.encode(ack, add_special_tokens=False)

    # Measure template overhead (header tokens, eot tokens) from a tiny sample.
    sample_ids = _apply_chat_template_ids(
        tokenizer,
        [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
        add_generation_prompt=False,
    )
    overhead = max(len(sample_ids) - 2, 20)  # tokens consumed by template structure

    user_budget = max(target_len - overhead - len(ack_ids), len(doc_ids))
    reps = math.ceil(user_budget / len(doc_ids)) + 1
    user_ids = (doc_ids * reps)[:user_budget]
    user_content = tokenizer.decode(user_ids, skip_special_tokens=True)

    ids = _apply_chat_template_ids(
        tokenizer,
        [{"role": "user", "content": user_content}, {"role": "assistant", "content": ack}],
        add_generation_prompt=False,
    )

    # Pad with raw doc tokens if still short (template overhead estimates can be off).
    if len(ids) < target_len:
        ids = ids + (doc_ids * math.ceil((target_len - len(ids)) / len(doc_ids)))

    return torch.tensor([ids[:target_len]], dtype=torch.long, device=device)


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
DEFAULT_DEPTH_SWEEP       = [4, 8, 16, 32]
DEFAULT_BRANCHING_FACTORS = [4, 8, 16, 32]            # EAGLE-3 default ★ only for now


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
    max_length: int = 2048,
    use_fp8: bool = False,
    load_in_4bit: bool = False,
) -> "eagle.model.ea_model.EaModel":
    _patch_transformers_for_eagle()
    from eagle.model.ea_model import EaModel
    from transformers import BitsAndBytesConfig

    # ... (RoPE patching code) ...

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
    print(f"    KV cache max_length: {max_length} tokens")

    # Suppress transformers' verbose "doesn't inherit GenerationMixin" warnings
    try:
        import transformers as _tr
        _tr.logging.set_verbosity_error()
    except Exception:
        pass

    # Build quantization config explicitly so transformers 5.x honours it
    # (passing load_in_4bit=True alongside torch_dtype is silently ignored in newer transformers)
    _quant_cfg = None
    _load_kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        _quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _load_kwargs["quantization_config"] = _quant_cfg
        # torch_dtype must not be set when using bitsandbytes quantization
    elif use_fp8:
        _load_kwargs["load_in_8bit"] = True
        _load_kwargs["torch_dtype"] = _dtype
    else:
        _load_kwargs["torch_dtype"] = _dtype

    # bitsandbytes quantization requires device_map="auto"; explicit dict silently skips quant
    _device_map = "auto" if load_in_4bit else {"": "cuda:0"}
    model = EaModel.from_pretrained(
        use_eagle3=use_eagle3,
        base_model_path=base_model,
        ea_model_path=eagle_model,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        device_map=_device_map,
        low_cpu_mem_usage=True,
        **_load_kwargs,
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


def run_ar_baseline(
    args,
    prompts: List[str],
    is_llama3: bool,
    max_new_tokens: int,
    extra_prefix_ids: Optional[torch.Tensor] = None,
) -> float:
    """Greedy autoregressive generation using a clean Transformers model.
    
    EAGLE's internal base_model (modeling_llama_kv.py) has custom mask logic
    that breaks native .forward() and .generate(). To get a fair baseline,
    we load the original Llama model in isolation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"    [AR] Loading clean base model: {args.base_model} ...")

    # Use bfloat16 for the baseline if supported, else float16
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    _ar_kwargs: dict = {}
    if args.load_in_4bit:
        _ar_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # bitsandbytes requires device_map="auto" and no torch_dtype
        _ar_kwargs["device_map"] = "auto"
    elif args.fp8:
        _ar_kwargs["load_in_8bit"] = True
        _ar_kwargs["torch_dtype"] = dtype
        _ar_kwargs["device_map"] = {"": "cuda:0"}
    else:
        _ar_kwargs["torch_dtype"] = dtype
        _ar_kwargs["device_map"] = {"": "cuda:0"}

    temp_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True,
        **_ar_kwargs,
    )
    temp_model.eval()
    prefill_device = next(temp_model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    tps_list: List[float] = []

    for pi, raw in enumerate(prompts):
        if is_llama3 and hasattr(tokenizer, "apply_chat_template"):
            input_ids = _llama3_input_ids(tokenizer, raw, prefill_device)
        else:
            input_ids = tokenizer([raw], return_tensors="pt").input_ids.to(prefill_device)
        if extra_prefix_ids is not None and extra_prefix_ids.shape[1] > 0:
            # Prefix starts with BOS; strip BOS from prompt to avoid a duplicate mid-sequence.
            bos_id = getattr(tokenizer, "bos_token_id", None)
            if bos_id is not None and input_ids[0, 0].item() == bos_id:
                input_ids = input_ids[:, 1:]
            input_ids = torch.cat([extra_prefix_ids.to(prefill_device), input_ids], dim=1)

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = temp_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_new = out.shape[1] - input_ids.shape[1]
            if n_new > 0:
                tps_list.append(n_new / elapsed)
        except torch.cuda.OutOfMemoryError as _oom:
            print(f"      [AR OOM prompt {pi}] {_oom}")
            torch.cuda.empty_cache()
            break
            
    # Cleanup to save VRAM
    del temp_model
    torch.cuda.empty_cache()
    
    return float(np.mean(tps_list)) if tps_list else 0.0


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
    attn_profile: bool = False,
    extra_prefix_ids: Optional[torch.Tensor] = None,
    max_length: int = 2048,
    use_cuda_graph: bool = False,
) -> List[GenerationRecord]:
    """
    extra_prefix_ids: optional [1, L] int64 tensor prepended to each prompt's
      input_ids before eagenerate().  Set by the context-length sweep to
      simulate a mid-conversation KV cache of exactly L tokens.  The prefix is
      real filler text so attention entropy is representative of actual usage.
    """
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

    # Evict stale KV cache if it was allocated at a smaller max_length than what
    # this run requires (happens after warmup or after a shorter-context run).
    # eagenerate() caches past_key_values on the model object and reuses them
    # on the next call WITHOUT checking whether the tensor is large enough.
    # Deleting them here forces a fresh allocation at the correct size.
    _existing_kv = getattr(model, 'past_key_values', None)
    if _existing_kv is not None:
        _existing_kv_size = _existing_kv[0][0].data.shape[2]  # [H_kv, max_len, D]
        if _existing_kv_size < max_length:
            for _attr in ('past_key_values', 'past_key_values_data', 'current_length_data'):
                if hasattr(model, _attr):
                    delattr(model, _attr)

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

        # ── Extract parent array from EAGLE-3's tree_mask ────────────────────
        # EAGLE sets mdl.base_model.model.tree_mask = [1, 1, N, N] float before
        # calling tree_decoding.  N = number of tree tokens.
        # tree_position_ids[i] = depth of token i.
        # We reconstruct the parent array:  parents[i] = j  where
        #   tree_mask[i,j]=True  and  depth[j] = depth[i]-1.
        # Root (depth=0): parents[0] = 0  (self-loop).
        _base_model = getattr(mdl, 'base_model', None)
        _llama_model = getattr(_base_model, 'model', None) if _base_model is not None else None
        if use_ragged and _llama_model is not None and hasattr(_llama_model, 'tree_mask'):
            tm = _llama_model.tree_mask  # [1, 1, N, N] float
            if tm is not None and tm.dim() == 4:
                tm_bool = tm[0, 0].bool()         # [N, N]
                N = tm_bool.shape[0]
                dev = tm_bool.device
                depths = tree_position_ids          # [N]
                if depths.dim() > 1:
                    depths = depths.squeeze(0)
                depths = depths[:N].to(dev)         # ensure matching device + length

                # Vectorised parent extraction:
                # For each node i, find col j where mask[i,j]=True and depth[j]=depth[i]-1
                target_depths = (depths - 1).unsqueeze(1)   # [N, 1]
                col_depths    = depths.unsqueeze(0)          # [1, N]
                depth_match   = (col_depths == target_depths)  # [N, N]  (on dev)
                valid         = tm_bool & depth_match          # [N, N]
                parents       = valid.int().argmax(dim=1).to(torch.int32)  # [N]
                parents[0]    = 0  # root self-loop

                _state["tree_parents"] = parents
            else:
                _state["tree_parents"] = None
        else:
            _state["tree_parents"] = None

        # Start timing after parent extraction so verify_ms measures tree_decoding
        # only — parent extraction is ragged-path overhead not present in vanilla.
        e0.record()

        # Raise the flag so the matmul hook knows we are inside tree_decoding.
        # This is the ONLY window where the hook should intercept matmuls.
        # Without this guard the hook fires during initialize_tree's backbone
        # call on the newly-accepted tokens (q_len=2–4, kv_len >> q_len),
        # routing non-tree Q/K/V through the ragged kernel and crashing when
        # seq_len is not a power of 2.
        global _IN_ATTN_PROFILE
        _state["in_tree_verify"] = True
        if attn_profile and not use_ragged:
            _IN_ATTN_PROFILE = True
        try:
            result = _orig_tree_decoding(
                mdl, tree_candidates, past_key_values,
                tree_position_ids, input_ids, retrieve_indices,
            )
        finally:
            _state["in_tree_verify"] = False
            _IN_ATTN_PROFILE = False
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

            # ── Prepend chat-history prefix ───────────────────────────────────
            # extra_prefix_ids is a [1, L] tensor of properly formatted prior
            # conversation turns (built by _build_prefix_ids with is_llama3=True).
            # It already starts with BOS, so we strip BOS from the current
            # prompt before concatenating to avoid a duplicate mid-sequence.
            if extra_prefix_ids is not None and extra_prefix_ids.shape[1] > 0:
                bos_id = getattr(model.tokenizer, "bos_token_id", None)
                if bos_id is not None and input_ids[0, 0].item() == bos_id:
                    input_ids = input_ids[:, 1:]
                input_ids = torch.cat(
                    [extra_prefix_ids.to(prefill_device), input_ids], dim=1
                )

            # Diagnostic on first prompt.
            if pi == 0:
                ctx_len = extra_prefix_ids.shape[1] if extra_prefix_ids is not None else 0
                decoded_head = model.tokenizer.decode(input_ids[0, :16], skip_special_tokens=False)
                print(f"  [diag] input_ids shape={tuple(input_ids.shape)} "
                      f"device={input_ids.device}  "
                      + (f"prefix={ctx_len} + " if ctx_len else "")
                      + f"head tokens: {decoded_head!r}")

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
            # ragged_eagle_context which patches LlamaAttention.forward on
            # the base model so tree verification uses the ragged kernel.
            # The in_tree_verify flag ensures the patch only fires during
            # tree_decoding(), never during backbone or draft-model calls.
            if use_ragged:
                with ragged_eagle_context(branching_factor, max_depth,
                                          silent=True,
                                          verify_flag=_state,
                                          model=model,
                                          use_cuda_graph=use_cuda_graph) as _hook_state:
                    out_ids, new_token, n_steps_idx = model.eagenerate(
                        input_ids,
                        temperature=0.0,
                        max_new_tokens=max_new_tokens,
                        max_length=max_length,
                        is_llama3=is_llama3,
                        log=True,
                    )
                _hook_fires = int(_hook_state["n"])  # type: ignore[arg-type]
            else:
                out_ids, new_token, n_steps_idx = model.eagenerate(
                    input_ids,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
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

    # ── Amdahl table (--attn-profile, vanilla path only) ─────────────────────
    if attn_profile and not use_ragged and _ATTN_EVENTS:
        torch.cuda.synchronize()
        total_attn_ms = sum(e0.elapsed_time(e1) for e0, e1 in _ATTN_EVENTS)
        _ATTN_EVENTS.clear()
        n_steps_total = sum(len(list(_state["verify_ms"])) for _ in [None])  # type: ignore
        n_steps_total = sum(r.num_steps for r in records)
        total_verify_ms = sum(r.mean_verify_ms * r.num_steps for r in records)
        total_wall_ms   = sum(r.wall_ms for r in records)
        attn_pct_verify = total_attn_ms / total_verify_ms * 100 if total_verify_ms else 0
        attn_pct_wall   = total_attn_ms / total_wall_ms   * 100 if total_wall_ms   else 0
        non_attn_ms     = total_verify_ms - total_attn_ms
        n_calls         = n_steps_total * 32  # 32 layers
        print(f"\n  ┌── AMDAHL PROFILE  ({n_steps_total} steps × 32 layers = {n_calls} SDPA calls) {'─'*18}")
        print(f"  │ {'Component':<20s}  {'Total ms':>10s}  {'Per-step ms':>12s}  {'% verify':>9s}  {'% wall':>7s}")
        print(f"  │ {'─'*20}  {'─'*10}  {'─'*12}  {'─'*9}  {'─'*7}")
        print(f"  │ {'attention (SDPA)':<20s}  {total_attn_ms:>10.1f}  "
              f"{total_attn_ms/n_steps_total:>12.3f}  {attn_pct_verify:>8.1f}%  {attn_pct_wall:>6.1f}%")
        print(f"  │ {'non-attention':<20s}  {non_attn_ms:>10.1f}  "
              f"{non_attn_ms/n_steps_total:>12.3f}  {100-attn_pct_verify:>8.1f}%  "
              f"{non_attn_ms/total_wall_ms*100:>6.1f}%")
        print(f"  │ {'verify total':<20s}  {total_verify_ms:>10.1f}  "
              f"{total_verify_ms/n_steps_total:>12.3f}  {'100.0':>8s}%  "
              f"{total_verify_ms/total_wall_ms*100:>6.1f}%")
        print(f"  └{'─'*70}")

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

    Optimizations vs. the original matmul-hook approach:
     - No wasted softmax/mask/scale/cast runs on garbage (those were eliminated
       by the module-level forward patch in ragged_eagle_context).
     - K_tree and V_tree are packed to [B*N_q,H,D] in a single
       slice+permute+contiguous() instead of two separate copies.
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
    # Avoid .contiguous() copies completely for B=1 by using transpose.
    # PyTorch SDPA outputs contiguous [B, N, H, D], Llama wrapper transposes to [B, H, N, D].
    # For B=1, we can just squeeze B and transpose H/N to get [N, H, D] with correct strides.
    if B == 1:
        Q_r = Q.squeeze(0).transpose(0, 1)  # [N_q, H, D]
        K_r = K[:, :, N_prefix:, :].squeeze(0).transpose(0, 1)
        V_r = V[:, :, N_prefix:, :].squeeze(0).transpose(0, 1)
    else:
        Q_r = Q.permute(0, 2, 1, 3).contiguous().view(B * N_q, H, D)
        K_r = K[:, :, N_prefix:, :].permute(0, 2, 1, 3).contiguous().view(B * N_q, H, D)
        V_r = V[:, :, N_prefix:, :].permute(0, 2, 1, 3).contiguous().view(B * N_q, H, D)
    
    cu  = torch.arange(0, (B + 1) * N_q, N_q, dtype=torch.int32, device=Q.device)

    if prof is not None:
        evts[3].record()                                        # ── [3] tree_reshape done

    # Use explicit-parent kernel when EAGLE-3 parent array is available.
    if tree_parents is not None and tree_parents.shape[0] == N_q:
        # B=1: tree_parents is already [N_q], skip .repeat(1)
        parents_packed = tree_parents if B == 1 else tree_parents.repeat(B)
        out_tree_r, lse_tree_r = ragged_attention_with_parents(
            Q_r, K_r, V_r, cu, parents_packed, max_depth, max_seqlen=N_q)
    else:
        out_tree_r, lse_tree_r = ragged_attention_with_lse(
            Q_r, K_r, V_r, cu, branching_factor, max_depth, max_seqlen=N_q)

    if prof is not None:
        evts[4].record()                                        # ── [4] ragged_kernel done

    if B == 1:
        out_tree = out_tree_r.transpose(0, 1).unsqueeze(0)   # [1, H, N_q, D]
        lse_tree = lse_tree_r.transpose(0, 1).unsqueeze(0)   # [1, H, N_q]
    else:
        out_tree = out_tree_r.view(B, N_q, H, D).permute(0, 2, 1, 3)   # [B…]
        lse_tree = lse_tree_r.view(B, N_q, H).permute(0, 2, 1)          # [B, H, N_q]

    if prof is not None:
        evts[5].record()                                        # ── [5] out_reshape done

    # ── Part 3: fused online-softmax merge (single Triton kernel) ───────────
    if N_prefix == 0:
        result = out_tree.to(dtype)
    else:
        result = fused_lse_merge(lse_pre, lse_tree, out_pre, out_tree)

    if prof is not None:
        evts[6].record()                                        # ── [6] lse_merge done
        prof.entries.append(evts)

    return result


# ── Shared buffers + Graph for CUDA Graphs (to save VRAM) ───────────────
# We only need ONE set of buffers AND one graph for the whole model 
# because layers execute sequentially and perform identical work on the 
# buffers. This reduces graph overhead from ~1.1GB to ~35MB.
# Move to module level to allow reuse across prompts and avoid re-capture.
_GLOBAL_GRAPH_CACHE: Dict[tuple, Dict[str, object]] = {}

@contextlib.contextmanager
def ragged_eagle_context(branching_factor: int, max_depth: int,
                         silent: bool = False,
                         verify_flag: Optional[Dict[str, object]] = None,
                         model=None,
                         use_cuda_graph: bool = False):
    """
    Module-level attention patch for EAGLE-3 tree verification.
    ... (rest of docstring omitted for brevity in system log) ...
    """
    if model is None:
        raise ValueError("ragged_eagle_context requires model= parameter "
                         "for module-level attention patching")

    _s: Dict[str, object] = {"n": 0}

    # ── Resolve the LlamaAttention class to patch ───────────────────────────
    _AttnClass = type(model.base_model.model.layers[0].self_attn)
    _orig_forward = _AttnClass.forward

    def _ragged_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """Patched LlamaAttention.forward that uses ragged kernel during verify."""
        # Gate: use original forward when NOT in tree verification
        if verify_flag is None or not verify_flag.get("in_tree_verify", False):
            return _orig_forward(self, hidden_states, attention_mask,
                                 position_ids, past_key_value,
                                 output_attentions, use_cache)

        # ==================================================================
        # PRE-ATTENTION — replicated from EAGLE's LlamaAttention.forward
        # ==================================================================
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if isinstance(self.rotary_emb, LlamaRotaryEmbedding_L31):
            cos, sin = self.rotary_emb(query_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb_L31(
                query_states, key_states, cos, sin)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states   = past_key_value[0].cat(key_states, dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)
        past_key_value = None

        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ==================================================================
        # RAGGED ATTENTION — Split Prefix and Tree for CUDA Graphs
        # ==================================================================
        _tree_parents = (verify_flag.get("tree_parents")
                         if verify_flag is not None else None)
        
        if use_cuda_graph:
            N_q = query_states.shape[-2]
            N_kv = key_states.shape[-2]
            N_prefix = N_kv - N_q
            has_prefix = N_prefix > 0
            
            GRAPH_N_THRESHOLD = 512
            
            if N_q <= GRAPH_N_THRESHOLD:
                # 1. Dynamic Prefix Attention (NOT graphed due to dynamic N_prefix)
                scale_v = 1.0 / math.sqrt(self.head_dim)
                if has_prefix:
                    K_pre = key_states[:, :, :N_prefix, :].contiguous()
                    V_pre = value_states[:, :, :N_prefix, :].contiguous()
                    try:
                        out_pre, lse_pre, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                            query_states.contiguous(), K_pre, V_pre,
                            dropout_p=0.0, is_causal=False, scale=scale_v,
                            return_debug_mask=False,
                        )
                    except Exception:
                        sc = torch.ops.aten.matmul(query_states, K_pre.transpose(-2, -1)) * scale_v
                        lse_pre = torch.logsumexp(sc.float(), dim=-1)
                        out_pre = torch.softmax(sc, dim=-1) @ V_pre
                else:
                    out_pre = None
                    lse_pre = None

                # 2. Graphed Tree Attention and LSE Merge (Static N_q)
                graph_key = (N_q, has_prefix)
                if graph_key not in _GLOBAL_GRAPH_CACHE:
                    s = {
                        "q": torch.empty_like(query_states),
                        "k_tree": torch.empty_like(query_states),
                        "v_tree": torch.empty_like(query_states),
                        "p": torch.zeros(bsz * N_q, dtype=torch.int32, device=query_states.device),
                        "cu": torch.arange(0, (bsz + 1) * N_q, N_q, dtype=torch.int32, device=query_states.device),
                        "o": torch.empty_like(query_states),
                        "graph": torch.cuda.CUDAGraph(),
                    }
                    if has_prefix:
                        s["lse_pre"] = torch.empty((bsz, self.num_heads, N_q), dtype=torch.float32, device=query_states.device)
                        s["out_pre"] = torch.empty_like(query_states)
                    
                    # Warmup call
                    if bsz == 1:
                        Q_r = s["q"].squeeze(0).transpose(0, 1)
                        K_r = s["k_tree"].squeeze(0).transpose(0, 1)
                        V_r = s["v_tree"].squeeze(0).transpose(0, 1)
                    else:
                        Q_r = s["q"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                        K_r = s["k_tree"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                        V_r = s["v_tree"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                    
                    if _tree_parents is not None:
                        out_tree_r, lse_tree_r = ragged_attention_with_parents(
                            Q_r, K_r, V_r, s["cu"], s["p"], max_depth, max_seqlen=N_q)
                    else:
                        out_tree_r, lse_tree_r = ragged_attention_with_lse(
                            Q_r, K_r, V_r, s["cu"], branching_factor, max_depth, max_seqlen=N_q)
                        
                    if bsz == 1:
                        out_tree = out_tree_r.transpose(0, 1).unsqueeze(0)
                    else:
                        out_tree = out_tree_r.view(bsz, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                    
                    if has_prefix:
                        if bsz == 1:
                            lse_tree = lse_tree_r.transpose(0, 1).unsqueeze(0)
                        else:
                            lse_tree = lse_tree_r.view(bsz, N_q, self.num_heads).permute(0, 2, 1)
                        _ = fused_lse_merge(s["lse_pre"], lse_tree, s["out_pre"], out_tree)

                    # Capture
                    with torch.cuda.graph(s["graph"]):
                        if bsz == 1:
                            Q_r_g = s["q"].squeeze(0).transpose(0, 1)
                            K_r_g = s["k_tree"].squeeze(0).transpose(0, 1)
                            V_r_g = s["v_tree"].squeeze(0).transpose(0, 1)
                        else:
                            Q_r_g = s["q"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                            K_r_g = s["k_tree"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                            V_r_g = s["v_tree"].permute(0, 2, 1, 3).contiguous().view(bsz * N_q, self.num_heads, self.head_dim)
                        
                        if _tree_parents is not None:
                            out_tree_r_g, lse_tree_r_g = ragged_attention_with_parents(
                                Q_r_g, K_r_g, V_r_g, s["cu"], s["p"], max_depth, max_seqlen=N_q)
                        else:
                            out_tree_r_g, lse_tree_r_g = ragged_attention_with_lse(
                                Q_r_g, K_r_g, V_r_g, s["cu"], branching_factor, max_depth, max_seqlen=N_q)
                            
                        if bsz == 1:
                            out_tree_g = out_tree_r_g.transpose(0, 1).unsqueeze(0)
                        else:
                            out_tree_g = out_tree_r_g.view(bsz, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                        
                        if has_prefix:
                            if bsz == 1:
                                lse_tree_g = lse_tree_r_g.transpose(0, 1).unsqueeze(0)
                            else:
                                lse_tree_g = lse_tree_r_g.view(bsz, N_q, self.num_heads).permute(0, 2, 1)
                            s["o"] = fused_lse_merge(s["lse_pre"], lse_tree_g, s["out_pre"], out_tree_g)
                        else:
                            s["o"] = out_tree_g.to(query_states.dtype)

                    _GLOBAL_GRAPH_CACHE[graph_key] = s

                s = _GLOBAL_GRAPH_CACHE[graph_key]
                s["q"].copy_(query_states)
                s["k_tree"].copy_(key_states[:, :, N_prefix:, :])
                s["v_tree"].copy_(value_states[:, :, N_prefix:, :])
                if has_prefix:
                    s["lse_pre"].copy_(lse_pre)
                    s["out_pre"].copy_(out_pre)
                
                if _tree_parents is not None:
                    parents_p = _tree_parents if bsz == 1 else _tree_parents.repeat(bsz)
                    s["p"].copy_(parents_p)
                
                s["graph"].replay()
                attn_output = s["o"]
            else:
                attn_output = _ragged_tree_attn(
                    query_states, key_states, value_states,
                    branching_factor, max_depth,
                    tree_parents=_tree_parents)
        else:
            attn_output = _ragged_tree_attn(
                query_states, key_states, value_states,
                branching_factor, max_depth,
                tree_parents=_tree_parents)

        _s["n"] = int(_s["n"]) + 1

        # ==================================================================
        # POST-ATTENTION
        # ==================================================================
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    # Accelerate's add_hook_to_module() stores the original forward as
    # module._old_forward and replaces module.forward (instance-level) with a
    # wrapper that calls _old_forward.  The instance-level attr shadows any
    # class-level patch, so we must patch _old_forward on every attention
    # instance.  Fall back to class-level patch for modules without the hook.
    import types as _types
    _layers = model.base_model.model.layers
    _saved: List[tuple] = []
    for _layer in _layers:
        _attn = _layer.self_attn
        if hasattr(_attn, "_old_forward"):
            _saved.append((_attn, "old", _attn._old_forward))
            _attn._old_forward = _types.MethodType(_ragged_attn_forward, _attn)
        else:
            _saved.append((_attn, "class", None))
    # Always do class-level patch as belt-and-suspenders
    _AttnClass.forward = _ragged_attn_forward
    try:
        yield _s
    finally:
        _AttnClass.forward = _orig_forward
        for _attn, _kind, _orig_fw in _saved:
            if _kind == "old":
                _attn._old_forward = _orig_fw
        
        if not silent:
            n_fires = int(_s["n"])               # type: ignore[arg-type]
            n_layers = 32                        # LLaMA-3.1-8B
            n_steps = n_fires // max(1, n_layers)
            print(f"  [ragged] module patch fired {n_fires}× total  "
                  f"({n_layers} layers × {n_steps} verify steps)")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 2: Depth sweep — acceptance rate vs depth (vanilla only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_eagle_model(args, max_length: int):
    """Load EAGLE-3 model + tokenizer. Returns (model, model_type, is_llama3)."""
    _patch_transformers_for_eagle()
    import eagle.model.ea_model as _eagle_ea
    from eagle.model.ea_model import EaModel

    is_eagle3  = not args.no_eagle3
    model_type = args.model_type
    is_llama3  = model_type in ("llama3", "llama-3-instruct")

    model = EaModel.from_pretrained(
        base_model_path=args.base_model,
        ea_model_path=args.eagle_model,
        total_token=-1,
        depth=-1,
        top_k=-1,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda",
        is_eagle3=is_eagle3,
        max_length=max_length,
    )
    model.eval()
    return model, model_type, is_llama3


def run_depth_sweep(args) -> None:
    """
    Exp 2: Acceptance rate vs tree depth (vanilla EAGLE-3 only).

    Runs at fixed b=16, L=0 over depths from --depths, records
    mean_accepted_per_step at each depth to reveal the Model Wall plateau.
    Saves results/depth_sweep.csv.
    """
    import csv as _csv

    is_eagle3  = not args.no_eagle3
    model_type = args.model_type
    is_llama3  = model_type in ("llama3", "llama-3-instruct")
    depths     = [int(x) for x in args.depths.split(",")]
    b          = int(args.branching_factors.split(",")[0])  # first value (default 16)

    _hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    prompts = _load_sharegpt_prompts(args.num_prompts, seed=args.prompt_seed,
                                     hf_token=_hf_token)

    print("\n" + "=" * 72)
    print("  EAGLE-3 Depth Sweep  ·  Acceptance Rate vs Depth  (vanilla only)")
    print("=" * 72)
    print(f"  Branching b={b}, L=0, {len(prompts)} prompts")
    print(f"  Depths: {depths}")
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  {p.total_memory // 1024**3} GB")
    print("=" * 72)

    max_length = 2048
    model, model_type, is_llama3 = _load_eagle_model(args, max_length)

    rows = []
    for d in depths:
        tt = _default_total_token(b, d)
        model.ea_layer.total_tokens = tt
        model.ea_layer.depth        = d
        model.ea_layer.top_k        = b

        print(f"\n  ── depth={d:2d}  N={tt:4d}  b={b} {'─'*50}")
        recs = run_generation(
            model, prompts, model_type,
            max_new_tokens=args.max_new_tokens,
            is_llama3=is_llama3,
            use_ragged=False,
            branching_factor=b,
            max_depth=d,
            max_length=max_length,
        )
        acc   = sum(r.mean_accepted_per_step for r in recs) / len(recs)
        toks  = sum(r.tok_per_sec            for r in recs) / len(recs)
        vms   = sum(r.mean_verify_ms         for r in recs) / len(recs)
        vfrac = sum(r.verify_fraction        for r in recs) / len(recs)
        print(f"    → depth={d:2d} N={tt:4d}  acc/step={acc:.3f}  "
              f"{toks:.1f} tok/s  verify={vms:.1f}ms ({vfrac:.0%})")
        rows.append({"depth": d, "N_tree": tt, "branching": b,
                     "acc_per_step": acc, "tok_per_sec": toks,
                     "verify_ms": vms, "verify_fraction": vfrac})

    # Save
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "depth_sweep.csv")
    fields = ["depth", "N_tree", "branching", "acc_per_step",
              "tok_per_sec", "verify_ms", "verify_fraction"]
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    # Print summary table
    print(f"\n  {'depth':>6}  {'N_tree':>7}  {'acc/step':>9}  {'tok/s':>7}  {'verify_ms':>10}  {'verify_frac':>12}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*10}  {'─'*12}")
    for r in rows:
        print(f"  {r['depth']:>6d}  {r['N_tree']:>7d}  {r['acc_per_step']:>9.3f}  "
              f"{r['tok_per_sec']:>7.1f}  {r['verify_ms']:>10.1f}  {r['verify_fraction']:>11.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Exp 4: Cross-system — EAGLE-3 at Sequoia-matching N values
# ─────────────────────────────────────────────────────────────────────────────

# Target (b, d) configs that produce N values matching Sequoia tree sizes
_CROSS_SYSTEM_CONFIGS = [
    # target_N, b, d
    (62,  12,  6),   # ≈ S=64
    (123, 16,  9),   # ≈ S=128  (also in existing sweep)
    (247, 24, 19),   # ≈ S=256
    (576, 24, 28),   # ≈ S=512
    (988, 32, 36),   # ≈ S=1024 (OOM risk on 20GB)
]


def run_cross_system(args) -> None:
    """
    Exp 4: EAGLE-3 vanilla vs ragged at N values matching Sequoia tree sizes.

    Produces results/cross_system_eagle.csv with columns:
      N_tree | vanilla_tok_s | ragged_tok_s | speedup | v_acc | r_acc |
      v_verify_ms | r_verify_ms | verify_fraction

    Combine with Sequoia e2e-sweep results (from benchmark_sequoia.py) to
    produce the unified cross-system Table 4.
    """
    import csv as _csv

    model_type = args.model_type
    is_llama3  = model_type in ("llama3", "llama-3-instruct")

    _hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    prompts = _load_sharegpt_prompts(args.num_prompts, seed=args.prompt_seed,
                                     hf_token=_hf_token)

    print("\n" + "=" * 72)
    print("  EAGLE-3 Cross-System Configs  (Sequoia-matching N values)")
    print("=" * 72)
    for tgt, b, d in _CROSS_SYSTEM_CONFIGS:
        print(f"  target_N={tgt:4d}  b={b}  d={d}  actual_N={_default_total_token(b, d)}")
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  {p.total_memory // 1024**3} GB")
    print("=" * 72)

    rows = []
    for tgt_n, b, d in _CROSS_SYSTEM_CONFIGS:
        tt = _default_total_token(b, d)
        max_length = max(2048, tt + 512 + args.max_new_tokens)
        print(f"\n  ── target_N={tgt_n}  (b={b} d={d} N={tt}) {'─'*40}")

        try:
            model, model_type2, is_llama32 = _load_eagle_model(args, max_length)
            model.ea_layer.total_tokens = tt
            model.ea_layer.depth        = d
            model.ea_layer.top_k        = b

            v_recs = run_generation(model, prompts, model_type,
                                    max_new_tokens=args.max_new_tokens,
                                    is_llama3=is_llama3,
                                    use_ragged=False,
                                    branching_factor=b, max_depth=d,
                                    max_length=max_length)
            r_recs = run_generation(model, prompts, model_type,
                                    max_new_tokens=args.max_new_tokens,
                                    is_llama3=is_llama3,
                                    use_ragged=True,
                                    branching_factor=b, max_depth=d,
                                    max_length=max_length)

            def _mean(recs, key):
                return sum(getattr(r, key) for r in recs) / len(recs)

            v_toks = _mean(v_recs, "tok_per_sec")
            r_toks = _mean(r_recs, "tok_per_sec")
            v_acc  = _mean(v_recs, "mean_accepted_per_step")
            r_acc  = _mean(r_recs, "mean_accepted_per_step")
            v_vms  = _mean(v_recs, "mean_verify_ms")
            r_vms  = _mean(r_recs, "mean_verify_ms")
            vfrac  = _mean(v_recs, "verify_fraction")
            speedup = r_toks / v_toks if v_toks else float("nan")

            print(f"  target_N={tgt_n:4d}  N={tt}  vanilla={v_toks:.1f} tok/s  "
                  f"ragged={r_toks:.1f} tok/s  speedup={speedup:.3f}×  "
                  f"acc_v={v_acc:.2f}  acc_r={r_acc:.2f}")
            rows.append({"target_N": tgt_n, "actual_N": tt, "b": b, "d": d,
                         "vanilla_tok_s": v_toks, "ragged_tok_s": r_toks,
                         "speedup": speedup, "v_acc_per_step": v_acc,
                         "r_acc_per_step": r_acc, "v_verify_ms": v_vms,
                         "r_verify_ms": r_vms, "verify_fraction": vfrac})

            del model
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as oom:
            print(f"  [OOM at target_N={tgt_n}] {oom}")
            rows.append({"target_N": tgt_n, "actual_N": tt, "b": b, "d": d,
                         **{k: float("nan") for k in [
                             "vanilla_tok_s", "ragged_tok_s", "speedup",
                             "v_acc_per_step", "r_acc_per_step",
                             "v_verify_ms", "r_verify_ms", "verify_fraction"]}})

    # Save
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cross_system_eagle.csv")
    fields = ["target_N", "actual_N", "b", "d", "vanilla_tok_s", "ragged_tok_s",
              "speedup", "v_acc_per_step", "r_acc_per_step",
              "v_verify_ms", "r_verify_ms", "verify_fraction"]
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    print(f"\n  {'target_N':>9}  {'N':>5}  {'vanilla':>8}  {'ragged':>8}  "
          f"{'speedup':>8}  {'v_acc':>6}  {'r_acc':>6}")
    print(f"  {'─'*9}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for r in rows:
        v, rg = r["vanilla_tok_s"], r["ragged_tok_s"]
        sp    = r["speedup"]
        va, ra = r["v_acc_per_step"], r["r_acc_per_step"]
        if not (v != v):  # not NaN
            print(f"  {r['target_N']:>9d}  {r['actual_N']:>5d}  {v:>8.1f}  "
                  f"{rg:>8.1f}  {sp:>7.3f}×  {va:>6.2f}  {ra:>6.2f}")
        else:
            print(f"  {r['target_N']:>9d}  {r['actual_N']:>5d}  {'OOM':>8}  "
                  f"{'OOM':>8}  {'—':>8}  {'—':>6}  {'—':>6}")


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
    parser.add_argument("--section", default="e2e",
                        choices=["e2e", "depth-sweep", "cross-system"],
                        help="Which benchmark section to run. "
                             "e2e: full 2D sweep (default). "
                             "depth-sweep: acceptance rate vs depth (Exp 2, vanilla only). "
                             "cross-system: EAGLE-3 at Sequoia-matching N values (Exp 4).")
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
    parser.add_argument("--include-ar-baseline", action="store_true",
                        help="Also run plain autoregressive generation (no speculative "
                             "decoding) at each context length using base_model.generate(). "
                             "Reports SD speedup-over-AR alongside ragged-over-vanilla.")
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
    parser.add_argument("--attn-profile", action="store_true",
                        help="Measure attention fraction of verify time in the vanilla path "
                             "(Amdahl table). Patches F.scaled_dot_product_attention with "
                             "CUDA event timing, gated to fire only during tree_verify. "
                             "Prints: attn_ms | non_attn_ms | attn_pct_verify | attn_pct_wall.")
    parser.add_argument(
        "--context-lengths",
        default="0",
        help="Comma-separated list of context-prefix lengths L (tokens) to sweep over. "
             "For each L>0, exactly L filler tokens are prepended to every prompt before "
             "eagenerate(), simulating a mid-conversation KV cache of that size. "
             "Micro-benchmarks show ragged wins 100%% of configs at L≥1024; this flag "
             "validates that result end-to-end. "
             "Example: --context-lengths 0,1024,4096  (default: 0)",
    )
    parser.add_argument("--use-cuda-graph", action="store_true",
                        help="Use CUDA Graphs to eliminate Python launch overhead in ragged path.")
    parser.add_argument("--fp8", action="store_true",
                        help="Use native FP8 precision (supported on Ada/Hopper) to save VRAM and accelerate compute.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use 4-bit quantization (NF4) to save VRAM and fit larger trees/contexts on 20GB cards.")
    args = parser.parse_args()

    # ── HF token propagation ─────────────────────────────────────────────────
    _hf_token = (args.hf_token
                 or os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

    # Enable expandable_segments to reduce fragmentation on tight VRAM (20GB card)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    depths         = [int(x) for x in args.depths.split(",")]
    bfacs          = [int(x) for x in args.branching_factors.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

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
    print(f"  Context lengths:  {context_lengths}")
    print(f"  Total configs:    {len(configs) * len(context_lengths)}  "
          f"({len(depths)} depths × {len(bfacs)} b-factors × {len(context_lengths)} ctx-lengths)")
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

    # ── Pre-warm doc download so it's cached before GPU work starts ──────────
    _get_doc_text()

    _prompt_budget = 512   # conservative upper bound for ShareGPT prompts
    _tree_budget   = max(c.total_token for c in configs)
    max_cfg        = max(configs, key=lambda c: c.total_token)

    # ── Sweep — model is reloaded per context-length ─────────────────────────
    # Loading once at the global max (66k) pre-allocates ~8.5 GB of KV cache,
    # leaving insufficient VRAM for prefill at large contexts on a 20 GB card.
    # Instead we size the KV cache to just fit the current context group, then
    # free the model before advancing to the next group.
    all_rows: List[dict] = []
    summary_rows: List[dict] = []

    import gc

    for ctx_len in context_lengths:

        # ── Per-context KV cache budget ───────────────────────────────────────
        _raw  = ctx_len + _prompt_budget + args.max_new_tokens + _tree_budget
        _kv_max_length = max(2048, math.ceil(_raw / 256) * 256)
        print(f"\n{'═' * 72}")
        print(f"  CONTEXT LENGTH  L={ctx_len}  —  KV cache max_length={_kv_max_length}"
              f"  ({ctx_len} + {_prompt_budget} + {args.max_new_tokens} + {_tree_budget})")
        print(f"{'═' * 72}")

        # ── Load model sized for this context ─────────────────────────────────
        try:
            model = load_eagle_model(
                base_model=args.base_model,
                eagle_model=args.eagle_model,
                use_eagle3=is_eagle3,
                total_token=max_cfg.total_token,
                depth=max_cfg.depth,
                top_k=max_cfg.top_k,
                max_length=_kv_max_length,
                use_fp8=args.fp8,
                load_in_4bit=args.load_in_4bit,
            )
        except torch.cuda.OutOfMemoryError as _oom:
            print(f"  [OOM] Cannot load model at L={ctx_len}: {_oom}")
            torch.cuda.empty_cache()
            continue

        # ── Warmup: trigger Triton JIT compilation ────────────────────────────
        if not args.skip_ragged:
            unique_depths = sorted(set(c.depth for c in configs))
            print(f"\n  [warmup]  Compiling Triton kernels: depths={unique_depths} …")
            warmup_prompts = [prompts[0][:200]]
            for _wd in unique_depths:
                _wcfg = next(c for c in configs if c.depth == _wd)
                set_tree_config(model, _wcfg)
                try:
                    _warmup = run_generation(
                        model, warmup_prompts, args.model_type,
                        max_new_tokens=32, is_llama3=is_llama3,
                        use_ragged=True, branching_factor=_wcfg.top_k,
                        max_depth=_wcfg.depth, profile=False,
                        max_length=_kv_max_length, use_cuda_graph=args.use_cuda_graph,
                    )
                except torch.cuda.OutOfMemoryError as _oom:
                    print(f"  [OOM] warmup d={_wd} skipped — {_oom}")
                    torch.cuda.empty_cache()
                    continue
                if _warmup:
                    print(f"  [warmup d={_wd}]  done ({_warmup[0].tok_per_sec:.1f} tok/s, "
                          f"accept={_warmup[0].mean_accepted_per_step:.2f}/step)")
                del _warmup
            torch.cuda.empty_cache()

        # ── Build prefix tensor for this context length ───────────────────────
        prefill_dev = next(model.base_model.parameters()).device
        if ctx_len > 0:
            extra_prefix = _build_prefix_ids(
                ctx_len, model.tokenizer, prefill_dev, is_llama3=is_llama3
            )
            print(f"  [ctx] Built prefix tensor: L={ctx_len} tokens  "
                  f"(actual={extra_prefix.shape[1]}, device={prefill_dev})")
        else:
            extra_prefix = None

        ctx_tag  = f"  L={ctx_len}" if ctx_len > 0 else ""
        ar_tok_s = 0.0

        for ci, cfg in enumerate(configs):
            print(f"\n{'━' * 72}")
            print(f"  CONFIG {ci+1}/{len(configs)}  —  {cfg.label}{ctx_tag}")
            print(f"    depth={cfg.depth}  total_token={cfg.total_token}  "
                  f"top_k={cfg.top_k}  context_length={ctx_len}")
            print(f"{'━' * 72}")

            set_tree_config(model, cfg)

            v_records: List[GenerationRecord] = []
            r_records: List[GenerationRecord] = []

            # ── Vanilla Eagle-3 ──────────────────────────────────────────────────
            if not args.skip_vanilla:
                print(f"\n  [vanilla]  depth={cfg.depth}  total_token={cfg.total_token}"
                      + (f"  L={ctx_len}" if ctx_len > 0 else ""))
                try:
                    v_records = run_generation(
                        model, prompts, args.model_type,
                        max_new_tokens=args.max_new_tokens,
                        is_llama3=is_llama3,
                        use_ragged=False,
                        profile=args.profile,
                        extra_prefix_ids=extra_prefix,
                        max_length=_kv_max_length,
                    )
                except torch.cuda.OutOfMemoryError as _oom:
                    print(f"  [OOM] vanilla skipped — {_oom}")
                    torch.cuda.empty_cache()
                if v_records:
                    v_tok = np.mean([r.tok_per_sec for r in v_records])
                    v_acc = np.mean([r.mean_accepted_per_step for r in v_records])
                    v_vms = np.mean([r.mean_verify_ms for r in v_records])
                    v_vfr = np.mean([r.verify_fraction for r in v_records])
                    print(f"    → {v_tok:.1f} tok/s   {v_acc:.2f} acc/step   "
                          f"verify={v_vms:.1f} ms ({v_vfr:.0%})")

            # ── Ragged-kernel Eagle-3 ────────────────────────────────────────────
            if not args.skip_ragged:
                print(f"\n  [ragged]   depth={cfg.depth}  total_token={cfg.total_token}"
                      + (f"  L={ctx_len}" if ctx_len > 0 else ""))
                try:
                    r_records = run_generation(
                        model, prompts, args.model_type,
                        max_new_tokens=args.max_new_tokens,
                        is_llama3=is_llama3,
                        use_ragged=True,
                        branching_factor=cfg.top_k,
                        max_depth=cfg.depth,
                        profile=args.profile,
                        extra_prefix_ids=extra_prefix,
                        max_length=_kv_max_length,
                        use_cuda_graph=args.use_cuda_graph,
                    )
                except torch.cuda.OutOfMemoryError as _oom:
                    print(f"  [OOM] ragged skipped — {_oom}")
                    torch.cuda.empty_cache()
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
                ctx_note = f" L={ctx_len}" if ctx_len > 0 else ""
                print(f"\n  ▸ E2E speedup at d={cfg.depth}{ctx_note}:  {speedup:.3f}×  "
                      f"({v_mean:.1f} → {r_mean:.1f} tok/s)")

            # ── Collect CSV rows ─────────────────────────────────────────────────
            for mode_label, recs in [("vanilla", v_records), ("ragged", r_records)]:
                for r in recs:
                    all_rows.append({
                        "context_length":            ctx_len,
                        "depth":                     cfg.depth,
                        "total_token":               cfg.total_token,
                        "top_k":                     cfg.top_k,
                        "config_label":              cfg.label,
                        "mode":                      mode_label,
                        "model":                     args.base_model,
                        "eagle_model":               args.eagle_model,
                        "prompt":                    r.prompt,
                        "num_tokens":                r.num_tokens,
                        "num_steps":                 r.num_steps,
                        "wall_ms":                   round(r.wall_ms, 1),
                        "tok_per_sec":               round(r.tok_per_sec, 1),
                        "mean_accepted_per_step":    round(r.mean_accepted_per_step, 3),
                        "acceptance_rate":           round(r.acceptance_rate, 4),
                        "mean_verify_ms":            round(r.mean_verify_ms, 3),
                        "verify_fraction":           round(r.verify_fraction, 4),
                    })

            # ── Summary row ──────────────────────────────────────────────────────
            s = {"context_length": ctx_len, "depth": cfg.depth,
                 "total_token": cfg.total_token, "top_k": cfg.top_k,
                 "label": cfg.label}
            if ar_tok_s > 0:
                s["ar_tok_s"] = round(ar_tok_s, 1)
            if v_records:
                s["vanilla_tok_s"]   = round(float(np.mean([r.tok_per_sec for r in v_records])), 1)
                s["vanilla_acc"]     = round(float(np.mean([r.mean_accepted_per_step for r in v_records])), 3)
                s["vanilla_verify"]  = round(float(np.mean([r.verify_fraction for r in v_records])), 4)
                if ar_tok_s > 0:
                    s["eagle_over_ar"] = round(s["vanilla_tok_s"] / ar_tok_s, 4)
            if r_records:
                s["ragged_tok_s"]    = round(float(np.mean([r.tok_per_sec for r in r_records])), 1)
                s["ragged_acc"]      = round(float(np.mean([r.mean_accepted_per_step for r in r_records])), 3)
                s["ragged_verify"]   = round(float(np.mean([r.verify_fraction for r in r_records])), 4)
                if ar_tok_s > 0:
                    s["ragged_over_ar"] = round(s["ragged_tok_s"] / ar_tok_s, 4)
            if v_records and r_records:
                s["e2e_speedup"] = round(s["ragged_tok_s"] / s["vanilla_tok_s"], 4) \
                    if s.get("vanilla_tok_s", 0) > 0 else float("nan")
            summary_rows.append(s)

        # ── Free model before next context length ─────────────────────────────
        del model, extra_prefix
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  [ctx={ctx_len}] Model freed, VRAM released for next group.")

    # ── Write CSV (datetime-stamped to avoid overwriting) ────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _base_name = args.csv_name.removesuffix(".csv")
    csv_path   = os.path.join(args.out_dir, f"{_base_name}_{_ts}.csv")
    csv_latest = os.path.join(args.out_dir, args.csv_name)
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        for _p in (csv_path, csv_latest):
            with open(_p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {csv_latest}  (latest)")

    # Summary CSV — one row per (b, d) config.
    summary_csv_ts = os.path.join(args.out_dir, f"e2e_summary_{_ts}.csv")
    summary_csv    = os.path.join(args.out_dir, "e2e_summary.csv")
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
        for _p in (summary_csv_ts, summary_csv):
            with open(_p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=s_fields)
                writer.writeheader()
                writer.writerows(summary_rows)
        print(f"  Saved: {summary_csv_ts}")
        print(f"  Saved: {summary_csv}  (latest)")

    # ── Final summary: 2D pivot tables ───────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  E2E BENCHMARK COMPLETE  —  depth × branching factor × context-length")
    print(f"{'=' * 72}")

    # Helper: look up summary value for (ctx_len, b, d)
    def _s(ctx_len: int, b: int, d: int, key: str):
        for row in summary_rows:
            if row["context_length"] == ctx_len and row["top_k"] == b and row["depth"] == d:
                v = row.get(key, "")
                return v if isinstance(v, (int, float)) and not (
                    isinstance(v, float) and math.isnan(v)
                ) else None
        return None

    def _pivot(ctx_len: int, metric_key: str, fmt: str, title: str) -> None:
        print(f"\n  {title}  [L={ctx_len}]")
        col_w = 8
        hline = "  " + f"{'b \\ d':>6s}" + "".join(f"{d:>{col_w}d}" for d in depths)
        print(hline)
        print("  " + "─" * (len(hline) - 2))
        for b in bfacs:
            cells = []
            for d in depths:
                v = _s(ctx_len, b, d, metric_key)
                if v is None:
                    cells.append(f"{'—':>{col_w}s}")
                else:
                    mark = "★" if b == 10 and d == 7 else " "
                    cells.append(f"{fmt.format(v)+mark:>{col_w}s}")
            b_mark = " ← default" if b == 10 else ""
            print(f"  {f'b={b}':>6s}" + "".join(cells) + b_mark)

    for ctx_len in context_lengths:
        print(f"\n{'─' * 72}")
        print(f"  CONTEXT LENGTH L={ctx_len}" + (" (baseline: no extra prefix)" if ctx_len == 0 else ""))
        print(f"{'─' * 72}")

        _pivot(ctx_len, "e2e_speedup",   "{:.3f}×", "E2E Speedup  (ragged / vanilla — >1 means ragged wins)")
        _pivot(ctx_len, "vanilla_tok_s", "{:.1f}",  "Vanilla Eagle-3  tok/s")
        _pivot(ctx_len, "ragged_tok_s",  "{:.1f}",  "Ragged-kernel Eagle-3  tok/s")
        _pivot(ctx_len, "vanilla_acc",   "{:.2f}",  "Mean accepted tokens/step  (vanilla)")

        # ── Per-b crossover analysis ──────────────────────────────────────────
        print(f"\n  {'─' * 66}")
        print(f"  Crossover analysis  [L={ctx_len}]  (min depth where ragged beats vanilla)")
        print(f"  {'─' * 66}")
        all_speedups = [
            (s["top_k"], s["depth"], s["e2e_speedup"])
            for s in summary_rows
            if s["context_length"] == ctx_len
            and isinstance(s.get("e2e_speedup"), (int, float))
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
            tt = _s(0, b, d, 'total_token')
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
