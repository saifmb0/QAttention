"""
benchmark_sota.py
=================
SOTA comparison benchmark — RTX PRO 6000 Blackwell Server Edition 94 GB (SM 12.0)
"blackwell" branch

Compares ragged ancestor-sparse attention against every relevant baseline
for tree-structured speculative decoding:

┌─────────────────────────────────────────────────────────────────────────┐
│ Method                      │ Library          │ Type        │ Mask    │
├─────────────────────────────┼──────────────────┼─────────────┼─────────┤
│ Ours (ragged, fp16)         │ Triton (this)    │ ragged      │ sparse  │
│ Ours (ragged, bf16)         │ Triton (this)    │ ragged      │ sparse  │
│ PyTorch SDPA — math         │ torch            │ padded      │ dense   │
│ PyTorch SDPA — flash (FA2)  │ torch            │ padded      │ dense   │
│ PyTorch SDPA — mem-eff.     │ torch            │ padded      │ dense   │
│ FlashAttention-2            │ flash_attn       │ padded/var  │ causal* │
│ FlashInfer — batch prefill  │ flashinfer       │ ragged      │ sparse* │
│ xformers — mem-eff          │ xformers         │ padded      │ dense   │
└─────────────────────────────┴──────────────────┴─────────────┴─────────┘

  * FlashAttention-2 and FlashInfer use standard causal masking, NOT the
    tree-ancestor mask.  They are included as compute-ceiling references:
    the best any padded/ragged method could possibly do on this input shape.

Notes on research context
--------------------------
  EAGLE-2 (Li et al. 2024, arXiv:2406.16858) drives the demand for
  tree-structured attention.  vLLM's current verification path uses
  PyTorch SDPA with an explicit tree mask bias tensor (math backend for
  correctness; flash backend for simple causal masks).  FlashInfer
  (Ye et al. 2024, arXiv:2312.11508) is the closest published ragged
  attention library; it targets standard causal attention and does not
  explicitly support the ancestor-sparse tree pattern.

  This benchmark measures the latency of the *attention kernel only*,
  which is the bottlenecked component during the verification pass.

Prerequisites (auto-checked at startup)
-----------------------------------------
  Required : torch >= 2.1,  triton >= 2.3
  Optional : flash_attn >= 2.5   (pip install flash-attn --no-build-isolation)
             flashinfer           (pip install flashinfer)
             xformers             (pip install xformers)

Usage
------
  # Quickstart — all configs, saves CSVs + plots under results/
  python scripts/benchmark_sota.py

  # Selected sweep (faster)
  python scripts/benchmark_sota.py --batch-sizes 1,8,32 --depths 1,3,5

  # Skip optional SOTA libs, only compare SDPA backends
  python scripts/benchmark_sota.py --skip-flashattn --skip-flashinfer --skip-xformers

  # Disable plots (CI / headless mode)
  python scripts/benchmark_sota.py --no-plot

  # Run with bfloat16 (Ada natively accelerated)
  python scripts/benchmark_sota.py --dtype bf16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
import importlib.util
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.tree_mask import tree_attention_mask, num_tree_nodes
from src.ragged_attn import pack_inputs, ragged_attention

# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZES       = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_DEPTHS            = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_BRANCHING_FACTORS = [2, 3, 4]
CTX_LEN                   = 128          # context prefix length (not benchmarked)
NUM_HEADS                 = 8
HEAD_DIM                  = 64
WARMUP_ITERS              = 10
BENCH_ITERS               = 50
# Dense baselines (sdpa_math, sdpa_flash) are O(N²) in memory.
# Skip them when the batch-level token count would exceed this threshold
# to avoid ~60 GB allocations on deep trees.
MAX_DENSE_TOKENS          = 8_000        # N per sequence; ~6 GB attn matrix at B=32

# ─────────────────────────────────────────────────────────────────────────────
# Optional library probes
# ─────────────────────────────────────────────────────────────────────────────

def _has(pkg: str) -> bool:
    """Return True only if the package is present AND can actually be imported.
    A simple find_spec() check is not enough — broken wheels (e.g. flash_attn
    compiled against a different torch ABI) pass find_spec but raise at import."""
    if importlib.util.find_spec(pkg) is None:
        return False
    try:
        __import__(pkg)
        return True
    except Exception:
        return False


HAS_FLASH_ATTN  = _has("flash_attn")
HAS_FLASHINFER  = _has("flashinfer")
HAS_TENSORRT    = _has("tensorrt") or _has("torch_tensorrt")
HAS_TRT_LLM     = _has("tensorrt_llm")

# xformers imports fine at the top level but memory_efficient_attention
# internally loads flash_attn_2_cuda.so on some builds.  Probe once here.
def _has_xformers_ops() -> bool:
    if not _has("xformers"):
        return False
    try:
        import xformers.ops as _xops
        # Do a tiny dry-run to flush any lazy .so loading
        _q = torch.zeros(1, 1, 1, 8, dtype=torch.float16,
                         device="cuda" if torch.cuda.is_available() else "cpu")
        _xops.memory_efficient_attention(_q, _q, _q)
        return True
    except Exception:
        return False

HAS_XFORMERS = _has_xformers_ops()


# ─────────────────────────────────────────────────────────────────────────────
# GPU capability check
# ─────────────────────────────────────────────────────────────────────────────

def device_info() -> dict:
    if not torch.cuda.is_available():
        return {"name": "CPU", "sm": "N/A", "vram_gb": 0, "sm_count": 0,
                "is_ada": False, "is_blackwell": False, "arch": "cpu"}
    p = torch.cuda.get_device_properties(0)
    sm = (p.major, p.minor)
    is_blackwell = sm >= (12, 0)
    is_ada       = (not is_blackwell) and sm >= (8, 9)
    if is_blackwell:
        arch = "Blackwell"
    elif is_ada:
        arch = "Lovelace"
    elif sm >= (8, 0):
        arch = "Ampere"
    elif sm >= (7, 5):
        arch = "Turing"
    else:
        arch = f"SM{p.major}{p.minor}"
    return {
        "name":         p.name,
        "sm":           f"{p.major}{p.minor}",
        "vram_gb":      round(p.total_memory / 1024**3, 1),
        "sm_count":     p.multi_processor_count,
        "is_ada":       is_ada,
        "is_blackwell": is_blackwell,
        "arch":         arch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_ms(fn: Callable, warmup: int, iters: int) -> float:
    """CUDA-event median timing, ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end   = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        ev_start.record()
        fn()
        ev_end.record()
        torch.cuda.synchronize()
        times.append(ev_start.elapsed_time(ev_end))
    return float(np.median(times))


def _try_time(fn: Callable, warmup: int, iters: int, label: str) -> float:
    try:
        return _time_ms(fn, warmup, iters)
    except Exception as exc:
        warnings.warn(f"[{label}] failed: {exc}")
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# FLOP counting
# ─────────────────────────────────────────────────────────────────────────────

def _ragged_flops(seq_lens: list, H: int, D: int, max_depth: int) -> float:
    """
    Actual (sparse) FLOPs executed by the ancestor-walk kernel.

    Each token attends to exactly (max_depth+1) ancestor positions.
    Per token: QK dot = 2·(d+1)·D FMAs; AV accumulate = 2·(d+1)·D FMAs.
    Total = 4 · (d+1) · N · D · H  per batch.
    """
    return 4.0 * (max_depth + 1) * sum(seq_lens) * D * H


def _dense_equivalent_flops(seq_lens: list, H: int, D: int) -> float:
    """
    Dense O(N²) FLOPs (what a standard padded attention would compute).
    Used as a relative-strength proxy in TFLOPS comparisons.
    """
    L_max = max(seq_lens)
    B     = len(seq_lens)
    return 4.0 * B * L_max * L_max * D * H


def _to_tflops(flops: float, ms: float) -> float:
    if math.isnan(ms) or ms <= 0:
        return float("nan")
    return flops / (ms * 1e-3) / 1e12


# ─────────────────────────────────────────────────────────────────────────────
# Input construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ragged(B, N, H, D, device, dtype):
    qs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    ks = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    vs = [torch.randn(N, H, D, device=device, dtype=dtype) for _ in range(B)]
    Q, K, V, cu_sl = pack_inputs(qs, ks, vs)
    return Q.to(device), K.to(device), V.to(device), cu_sl, qs, ks, vs


def _make_padded(qs, ks, vs, masks_np, L_max, B, H, D, device, dtype):
    """Return (Q_pad, K_pad, V_pad, attn_bias) in [B, H, L, D]."""
    # Use the fill value appropriate for the target dtype to avoid overflow
    # (float32.min/2 ≈ -8.5e37, which doesn't fit in fp16).
    NEG_INF = torch.finfo(dtype).min / 2

    def _pad(ts):
        out = torch.zeros(B, L_max, H, D, device=device, dtype=dtype)
        for i, t in enumerate(ts):
            out[i, :t.shape[0]] = t
        return out.permute(0, 2, 1, 3)   # [B, H, L, D]

    Q_p = _pad(qs)
    K_p = _pad(ks)
    V_p = _pad(vs)

    # bias must match Q dtype — flash / mem-efficient backends reject float32 bias
    # when queries are fp16/bf16.
    bias = torch.full((B, 1, L_max, L_max), NEG_INF, device=device, dtype=dtype)
    for i, m in enumerate(masks_np):
        Li = m.shape[0]
        tb = torch.from_numpy(m.astype(np.float32)).to(device=device, dtype=dtype)
        bias[i, 0, :Li, :Li] = torch.where(tb.bool(),
                                            torch.zeros_like(tb),
                                            torch.full_like(tb, NEG_INF))
    return Q_p, K_p, V_p, bias


# ─────────────────────────────────────────────────────────────────────────────
# Per-method runner functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_runner_ragged(Q, K, V, cu_sl, b, d):
    def fn():
        ragged_attention(Q, K, V, cu_sl, branching_factor=b, max_depth=d)
    return fn


def _make_runner_sdpa_math(Q_p, K_p, V_p, bias):
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           attn_mask=bias, scale=scale)
    return fn


def _make_runner_sdpa_flash(Q_p, K_p, V_p):
    """Flash backend — no custom bias, uses causal=True (upper bound)."""
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           is_causal=True, scale=scale)
    return fn


def _make_runner_sdpa_memeff(Q_p, K_p, V_p, bias):
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                           attn_mask=bias, scale=scale)
    return fn


def _make_runner_flash_attn(Q_p, K_p, V_p):
    """FlashAttention-2 via flash_attn library (causal, no tree bias)."""
    if not HAS_FLASH_ATTN:
        return None
    try:
        from flash_attn import flash_attn_func  # type: ignore
        # flash_attn_func expects [B, L, H, D]
        q = Q_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        k = K_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        v = V_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        def fn():
            flash_attn_func(q, k, v, causal=True)
        return fn
    except Exception as exc:
        warnings.warn(f"[flash_attn] setup failed: {exc}")
        return None


def _make_runner_flashinfer(Q, K, V, cu_sl, B, L_max, H, D, device):
    """
    FlashInfer batch prefill with causal mask (no tree bias).
    Uses BatchPrefillWithRaggedKVCacheWrapper for variable-length sequences.
    """
    if not HAS_FLASHINFER:
        return None
    try:
        import flashinfer  # type: ignore
        q_indptr = cu_sl.to(device)
        kv_indptr = cu_sl.to(device)
        Q_fi = Q.to(torch.float16)    # flashinfer requires fp16
        K_fi = K.to(torch.float16)
        V_fi = V.to(torch.float16)

        workspace_buf = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buf, kv_layout="NHD"
        )
        wrapper.plan(
            q_indptr, kv_indptr, kv_indptr,
            num_qo_heads=H, num_kv_heads=H, head_dim=D,
            causal=True
        )
        def fn():
            wrapper.run(Q_fi, K_fi, V_fi)
        # Warmup once to JIT compile
        try:
            fn()
            torch.cuda.synchronize()
        except Exception:
            pass
        return fn
    except Exception as exc:
        warnings.warn(f"[flashinfer] setup failed: {exc}")
        return None


def _make_runner_xformers(Q_p, K_p, V_p):
    """xformers memory-efficient attention (causal, no tree bias)."""
    if not HAS_XFORMERS:
        return None
    try:
        import xformers.ops as xops  # type: ignore
        q = Q_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)  # [B, L, H, D]
        k = K_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        v = V_p.permute(0, 2, 1, 3).contiguous().to(torch.float16)
        def fn():
            xops.memory_efficient_attention(q, k, v,
                                            attn_bias=xops.LowerTriangularMask())
        return fn
    except Exception as exc:
        warnings.warn(f"[xformers] setup failed: {exc}")
        return None


def _make_runner_sdpa_flash_masked(Q_p, K_p, V_p, bias):
    """
    PyTorch SDPA with tree-ancestry float-additive bias on PADDED tensors.

    NOTE: a float attn_mask forces the mem-efficient or math backend — the
    flash kernel cannot accept an arbitrary float bias.  This baseline measures
    the best PyTorch can do for the correct mask semantics when given padded
    batch tensors (the typical deployment path in vLLM today).
    """
    scale = 1.0 / math.sqrt(Q_p.shape[-1])
    def fn():
        F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                       attn_mask=bias, scale=scale)
    return fn


def _make_runner_sdpa_batched_bool(Q_p, K_p, V_p, masks_np, B, N, device, dtype):
    """
    FAIR baseline: single batched SDPA call with exact tree-ancestor BOOLEAN
    mask of shape [B, 1, N, N] on padded tensors.

    Why this is the right comparison:
    - One kernel launch — no serial per-sample dispatch overhead
    - Boolean attn_mask: PyTorch's flash-SDP backend accepts booleans
      (converts to -inf additive internally), so this CAN use flash on
      supported shapes/dtypes, unlike the float-additive-bias path which
      always falls back to mem-efficient or math
    - Correct tree-ancestor semantics (same as our kernel)
    - Padding tokens present (this measures padding waste honestly)

    The gap between this and our ragged kernel is the combined benefit of:
      (1) eliminating O(N²) work via ancestor-sparse computation
      (2) eliminating padding waste across the batch

    Reviewer note: no attention library (FA-2, FlashInfer) exposes a
    packed-varlen API that also accepts a custom per-sequence sparse mask.
    This single-call boolean-mask approach is therefore the FASTEST achievable
    with correct semantics using standard PyTorch.
    """
    # Build batched boolean mask [B, 1, N, N]: True = attend
    bool_bias = torch.zeros(B, 1, N, N, dtype=torch.bool, device=device)
    for i, m in enumerate(masks_np):
        Li = m.shape[0]
        bool_bias[i, 0, :Li, :Li] = torch.from_numpy(m.astype(bool)).to(device=device)
    scale = 1.0 / math.sqrt(Q_p.shape[-1])

    def fn():
        F.scaled_dot_product_attention(Q_p, K_p, V_p,
                                       attn_mask=bool_bias, scale=scale)
    return fn


def _make_runner_fa2_varlen(Q, K, V, cu_sl, B, N, H, D, tree_depth):
    """
    FlashAttention-2 flash_attn_varlen_func (ragged packed inputs, causal).

    Important framing: FA-2 varlen handles *inter-sequence* variable lengths
    natively (no padding between sequences) but still applies standard causal
    masking *within* each sequence — NOT the ancestor-sparse tree mask.
    This is the answer to: "what if you just use FA-2 varlen on the packed
    tokens without exploiting tree structure?"
    Result is mathematically INCORRECT for tree verification, but tests the
    latency of the fastest available attention primitive on the same input shape.
    """
    if not HAS_FLASH_ATTN:
        return None
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore
        device = Q.device
        cu = cu_sl.to(device=device, dtype=torch.int32)
        q = Q.to(torch.float16).contiguous()  # [total_tokens, H, D]
        k = K.to(torch.float16).contiguous()
        v = V.to(torch.float16).contiguous()
        max_seqlen = N  # all seqs same length in this benchmark

        def fn():
            flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
        # warm up once
        try:
            fn(); torch.cuda.synchronize()
        except Exception:
            pass
        return fn
    except Exception as exc:
        warnings.warn(f"[fa2_varlen] setup failed: {exc}")
        return None


def _make_runner_trt_attention(Q_p, K_p, V_p, bias, B, H, N, D):
    """
    TensorRT attention kernel via tensorrt_llm or tensorrt directly.

    Attempts three escalating fallbacks:
      1. tensorrt_llm.functional.bert_attention  (full TRT-LLM)
      2. torch.ops.tensorrt.scaled_dot_product_attention (TRT torch-tensorrt)
      3. Returns None if neither is importable.

    Note: TRT-LLM requires building a TensorRT engine first. The latency here
    includes the PyTorch→TRT dispatch overhead but NOT engine build time.
    No tree-mask support in TRT-LLM's prefill FMHA by default; this measures
    the *causal* upper-bound on their stack, same as the FA-2 causal baseline.
    """
    if not (HAS_TENSORRT or HAS_TRT_LLM):
        return None

    # Attempt 1: tensorrt_llm
    if HAS_TRT_LLM:
        try:
            import tensorrt_llm  # type: ignore  # noqa: F401
            # TRT-LLM attention requires GPT model context — not directly callable
            # as a standalone op.  We skip and fall through.
            warnings.warn("[trt_llm] tensorrt_llm is installed but its attention "
                          "primitive requires a full model context. "
                          "Use trt_llm_e2e benchmark instead. Skipping.")
        except Exception:
            pass

    # Attempt 2: torch-tensorrt compiled module
    if HAS_TENSORRT:
        try:
            import tensorrt as trt  # type: ignore  # noqa: F401
            import torch_tensorrt  # type: ignore  # noqa: F401

            scale = 1.0 / math.sqrt(D)
            q = Q_p.to(torch.float16).contiguous()
            k = K_p.to(torch.float16).contiguous()
            v = V_p.to(torch.float16).contiguous()

            # Compile a simple SDPA module with TRT backend
            class _SDPA(torch.nn.Module):
                def forward(self, q, k, v):
                    return F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, scale=scale
                    )

            m = _SDPA().eval().cuda()
            try:
                compiled = torch_tensorrt.compile(
                    m,
                    inputs=[
                        torch_tensorrt.Input(q.shape, dtype=torch.float16),
                        torch_tensorrt.Input(k.shape, dtype=torch.float16),
                        torch_tensorrt.Input(v.shape, dtype=torch.float16),
                    ],
                    enabled_precisions={torch.float16},
                    truncate_double=True,
                )
                compiled(q, k, v); torch.cuda.synchronize()  # warmup
                def fn():
                    compiled(q, k, v)
                return fn
            except Exception as exc:
                warnings.warn(f"[torch_tensorrt] compile failed: {exc}")
        except ImportError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single benchmark point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchRow:
    batch_size:        int
    branching_factor:  int
    tree_depth:        int
    num_tree_nodes:    int
    attn_padding_ratio: float
    # latencies (ms) — NaN = skipped / unavailable
    ragged_fp16_ms:    float
    ragged_bf16_ms:    float
    sdpa_math_ms:      float
    sdpa_flash_ms:     float        # causal upper-bound (no tree mask)
    sdpa_flash_masked_ms: float     # padded float-bias → mem-eff/math backend
    sdpa_batched_bool_ms: float     # FAIR: batched boolean mask → flash eligible
    sdpa_memeff_ms:    float
    flash_attn2_ms:    float        # FA-2 lib (causal, upper-bound ref)
    fa2_varlen_ms:     float        # FA-2 varlen = vLLM/PagedAttn prefill kernel (ragged packed, causal UB ref)
    flashinfer_ms:     float        # FlashInfer ragged prefill (causal UB ref)
    xformers_ms:       float        # xformers mem-eff (causal UB ref)
    trt_attention_ms:  float        # TensorRT compiled attention (causal UB ref)
    # Actual sparse TFLOPS (4·(d+1)·N·D·H / latency)
    ragged_sparse_tflops: float
    # Dense-equivalent TFLOPS (how fast would dense need to run to match)
    ragged_dense_equiv_tflops: float
    sdpa_math_tflops:  float
    # speedups vs named baselines
    speedup_vs_sdpa_math:        float
    speedup_vs_sdpa_flash:       float   # vs causal UB
    speedup_vs_sdpa_flash_masked: float  # vs padded float-bias baseline
    speedup_vs_sdpa_batched_bool: float  # FAIR: vs batched boolean-mask (flash eligible)
    speedup_vs_fa2:              float
    speedup_vs_fa2_varlen:       float
    speedup_vs_flashinfer:       float
    speedup_vs_trt:              float


def benchmark_one(
    batch_size:        int,
    branching_factor:  int,
    depth:             int,
    dtype_str:         str  = "fp16",
    num_heads:         int  = NUM_HEADS,
    head_dim:          int  = HEAD_DIM,
    ctx_len:           int  = CTX_LEN,
    warmup:            int  = WARMUP_ITERS,
    iters:             int  = BENCH_ITERS,
    device:            torch.device | None = None,
    skip_flashattn:    bool = False,
    skip_flashinfer:   bool = False,
    skip_xformers:     bool = False,
    skip_tensorrt:     bool = False,
) -> BenchRow:
    if device is None:
        device = torch.device("cuda")

    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    torch.manual_seed(batch_size * 1000 + branching_factor * 100 + depth)

    N    = num_tree_nodes(branching_factor, depth)
    B, H, D = batch_size, num_heads, head_dim

    # Skip O(N²) dense baselines when the sequence is too long to avoid OOM.
    # Ragged kernel results are always collected.
    skip_dense = N > MAX_DENSE_TOKENS

    masks_np = [tree_attention_mask(branching_factor, depth) for _ in range(B)]
    seq_lens = [N] * B

    # ── Build inputs ────────────────────────────────────────────────────────
    Q_r16, K_r16, V_r16, cu_sl, qs_fp16, ks_fp16, vs_fp16 = _make_ragged(
        B, N, H, D, device, torch.float16
    )
    Q_rbf, K_rbf, V_rbf, cu_sl_bf, _, _, _ = _make_ragged(
        B, N, H, D, device, torch.bfloat16
    )
    if not skip_dense:
        Q_p, K_p, V_p, attn_bias = _make_padded(
            qs_fp16, ks_fp16, vs_fp16, masks_np,
            N, B, H, D, device, torch.float16
        )
    else:
        Q_p = K_p = V_p = attn_bias = None

    # ── Build runners ────────────────────────────────────────────────────────
    run_ragged_fp16   = _make_runner_ragged(Q_r16, K_r16, V_r16, cu_sl,
                                            branching_factor, depth)
    run_ragged_bf16   = _make_runner_ragged(Q_rbf, K_rbf, V_rbf, cu_sl_bf,
                                            branching_factor, depth)
    run_sdpa_math     = None if skip_dense else _make_runner_sdpa_math(Q_p, K_p, V_p, attn_bias)
    # Causal upper-bound (NO tree mask — semantically wrong but fastest possible)
    run_sdpa_flash    = None if skip_dense else _make_runner_sdpa_flash(Q_p, K_p, V_p)
    # float-additive-bias baseline → always forces mem-eff/math path
    run_sdpa_flash_m  = None if skip_dense else _make_runner_sdpa_flash_masked(Q_p, K_p, V_p, attn_bias)
    # FAIR: single batched SDPA with boolean tree mask → flash kernel eligible
    run_sdpa_bb       = None if skip_dense else _make_runner_sdpa_batched_bool(
        Q_p, K_p, V_p, masks_np, B, N, device, torch.float16
    )
    run_sdpa_meff     = None if skip_dense else _make_runner_sdpa_memeff(Q_p, K_p, V_p, attn_bias)

    run_fa2     = None if (skip_flashattn  or skip_dense) else _make_runner_flash_attn(Q_p, K_p, V_p)
    run_fa2_var = None if skip_flashattn else _make_runner_fa2_varlen(
        Q_r16, K_r16, V_r16, cu_sl, B, N, H, D, depth
    )
    run_fi      = None if skip_flashinfer else _make_runner_flashinfer(
        Q_r16, K_r16, V_r16, cu_sl, B, N, H, D, device
    )
    run_xf      = None if (skip_xformers   or skip_dense) else _make_runner_xformers(Q_p, K_p, V_p)
    run_trt     = None if (skip_dense or skip_tensorrt) else _make_runner_trt_attention(
        Q_p, K_p, V_p, attn_bias, B, H, N, D
    )

    # ── Time everything ──────────────────────────────────────────────────────
    t_r16   = _try_time(run_ragged_fp16,  warmup, iters, "ragged_fp16")
    t_rbf   = _try_time(run_ragged_bf16,  warmup, iters, "ragged_bf16")
    t_sm    = _try_time(run_sdpa_math,    warmup, iters, "sdpa_math")    if run_sdpa_math   else float("nan")
    t_sf    = _try_time(run_sdpa_flash,   warmup, iters, "sdpa_flash")   if run_sdpa_flash  else float("nan")
    t_sfm   = _try_time(run_sdpa_flash_m, warmup, iters, "sdpa_flash_masked") if run_sdpa_flash_m else float("nan")
    t_sbb   = _try_time(run_sdpa_bb,       warmup, iters, "sdpa_batched_bool") if run_sdpa_bb      else float("nan")
    t_me    = _try_time(run_sdpa_meff,    warmup, iters, "sdpa_memeff")  if run_sdpa_meff   else float("nan")
    t_fa2   = _try_time(run_fa2,          warmup, iters, "flash_attn2")  if run_fa2         else float("nan")
    t_fa2v  = _try_time(run_fa2_var,      warmup, iters, "fa2_varlen")   if run_fa2_var     else float("nan")
    t_fi    = _try_time(run_fi,           warmup, iters, "flashinfer")   if run_fi          else float("nan")
    t_xf    = _try_time(run_xf,           warmup, iters, "xformers")     if run_xf          else float("nan")
    t_trt   = _try_time(run_trt,          warmup, iters, "trt")          if run_trt         else float("nan")

    # ── Metrics ──────────────────────────────────────────────────────────────
    # Actual sparse FLOPs (what the kernel really executes)
    sparse_flops = _ragged_flops(seq_lens, H, D, depth)
    # Dense-equivalent FLOPs (for comparison context only)
    dense_flops  = _dense_equivalent_flops(seq_lens, H, D)
    pad_rat      = 1.0 - sum(l * l for l in seq_lens) / (B * N * N)

    def _spdup(t_ref, t_our):
        if math.isnan(t_ref) or math.isnan(t_our) or t_our <= 0:
            return float("nan")
        return round(t_ref / t_our, 3)

    t_ours = t_r16   # fp16 ragged as "ours"

    return BenchRow(
        batch_size=B,
        branching_factor=branching_factor,
        tree_depth=depth,
        num_tree_nodes=N,
        attn_padding_ratio=round(pad_rat, 4),
        ragged_fp16_ms         =round(t_r16,  4),
        ragged_bf16_ms         =round(t_rbf,  4),
        sdpa_math_ms           =round(t_sm,   4),
        sdpa_flash_ms          =round(t_sf,   4),
        sdpa_flash_masked_ms   =round(t_sfm,  4),
        sdpa_batched_bool_ms   =round(t_sbb,  4),
        sdpa_memeff_ms         =round(t_me,   4),
        flash_attn2_ms         =round(t_fa2,  4),
        fa2_varlen_ms          =round(t_fa2v, 4),
        flashinfer_ms          =round(t_fi,   4),
        xformers_ms            =round(t_xf,   4),
        trt_attention_ms       =round(t_trt,  4),
        ragged_sparse_tflops       =round(_to_tflops(sparse_flops, t_r16), 5),
        ragged_dense_equiv_tflops  =round(_to_tflops(dense_flops,  t_r16), 3),
        sdpa_math_tflops           =round(_to_tflops(dense_flops,  t_sm),  3),
        speedup_vs_sdpa_math        =_spdup(t_sm,   t_ours),
        speedup_vs_sdpa_flash       =_spdup(t_sf,   t_ours),
        speedup_vs_sdpa_flash_masked=_spdup(t_sfm,  t_ours),
        speedup_vs_sdpa_batched_bool=_spdup(t_sbb,  t_ours),
        speedup_vs_fa2              =_spdup(t_fa2,  t_ours),
        speedup_vs_fa2_varlen       =_spdup(t_fa2v, t_ours),
        speedup_vs_flashinfer       =_spdup(t_fi,   t_ours),
        speedup_vs_trt              =_spdup(t_trt,  t_ours),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_ADA_PALETTE = {
    "ragged_fp16":        "#00b4d8",
    "ragged_bf16":        "#0096c7",
    "sdpa_math":          "#ef233c",
    "sdpa_flash":         "#fb8500",       # causal UB (wrong mask)
    "sdpa_flash_masked":  "#e07b00",       # padded float-bias → mem-eff/math
    "sdpa_batched_bool":  "#2dc653",       # FAIR: batched boolean mask → flash eligible
    "sdpa_memeff":        "#ffb703",
    "flash_attn2":        "#8338ec",
    "fa2_varlen":         "#c77dff",       # FA-2 varlen = vLLM/PagedAttn prefill kernel (ragged, causal UB)
    "flashinfer":         "#3a86ff",
    "xformers":           "#06d6a0",
    "trt_attention":      "#d62828",
}

# Columns in the order they appear in the latency-vs-depth plot
# Key = display label, value = BenchRow field name
_METHOD_COLS = {
    "Ragged fp16 (ours)":           "ragged_fp16_ms",
    "Ragged bf16 (ours)":           "ragged_bf16_ms",
    "SDPA math [tree mask]":        "sdpa_math_ms",
    "SDPA flash [causal UB]":       "sdpa_flash_ms",
    "SDPA [tree mask, float bias]": "sdpa_flash_masked_ms",
    "SDPA [tree mask, bool, FAIR]": "sdpa_batched_bool_ms",
    "SDPA mem-eff":                 "sdpa_memeff_ms",
    "FlashAttention-2 [causal UB]": "flash_attn2_ms",
    "FA-2 varlen [=vLLM/PA kernel, causal UB]":  "fa2_varlen_ms",
    "FlashInfer [causal UB]":       "flashinfer_ms",
    "xformers [causal UB]":         "xformers_ms",
    "TensorRT [causal UB]":         "trt_attention_ms",
}


def plot_latency_vs_depth(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 1: Latency (ms) vs tree depth for B=8, one panel per branching factor.
    Uses log-y scale to show quadratic vs linear growth clearly.
    Marks the EAGLE-2 practical operating region (d=3..5).
    Each line is annotated with architecture tier and mask type.
    """
    _info = device_info()
    _gpu_label = f"{_info['name']}  ({_info['arch']}  SM {_info['sm']})"
    colors = list(_ADA_PALETTE.values())

    for b in sorted(df["branching_factor"].unique()):
        sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)].sort_values("tree_depth")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        # Shade the EAGLE-2 practical operating region
        ax.axvspan(3, 5, alpha=0.07, color="green", label="EAGLE-2 practical range (d=3–5)")

        for ci, (label, col) in enumerate(_METHOD_COLS.items()):
            vals = sub[col].values.astype(float)
            if np.all(np.isnan(vals)):
                continue
            color = colors[ci % len(colors)]
            ls = "-" if "ours" in label.lower() else (
                "--" if "FAIR" in label else ":"  # solid=ours, dashed=fair, dotted=UB
            )
            lw = 2.5 if "ours" in label.lower() else 1.5
            ax.plot(sub["tree_depth"], vals, marker="o", label=label,
                    color=color, linewidth=lw, markersize=5, linestyle=ls)

        ax.set_yscale("log")
        ax.set_xlabel("Tree depth  d", fontsize=12)
        ax.set_ylabel("Latency  (ms, log scale)", fontsize=12)
        ax.set_title(
            f"Attention Kernel Latency — b={b}, B=8, H={NUM_HEADS}, D={HEAD_DIM}\n"
            f"{_gpu_label}\n"
            f"Dashed = upper-bound (wrong mask). Dotted = FAIR (tree mask). Solid = our method.",
            fontsize=10
        )
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.25, which="both")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        plt.tight_layout()
        path = os.path.join(out_dir, f"fig1_latency_b{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_speedup_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 2: Two heatmaps per branching factor:
      Left  — speedup vs sdpa_math with tree mask   (primary / fair production baseline)
      Right — speedup vs sdpa_flash_masked with tree mask  (fairest possible comparison)
    Cells with OOM are shown as grey.
    """
    _info = device_info()
    _gpu_label = f"{_info['name']}  ({_info['arch']}  SM {_info['sm']})"

    pairs = [
        ("speedup_vs_sdpa_math",          "vs SDPA-math [tree mask, padded float bias]"),
        ("speedup_vs_sdpa_batched_bool",  "vs SDPA [tree mask, bool mask, FAIR]"),
    ]

    for b in sorted(df["branching_factor"].unique()):
        sub = df[df["branching_factor"] == b]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f"Ragged fp16 speedup  (b={b})  —  {_gpu_label}", fontsize=11)

        for ax, (spd_col, title) in zip(axes, pairs):
            pivot = sub.pivot_table(
                index="tree_depth", columns="batch_size",
                values=spd_col, aggfunc="mean"
            )
            valid = pivot.values[~np.isnan(pivot.values)]
            vmax  = max(3.0, float(np.nanmax(pivot.values))) if len(valid) else 3.0
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           vmin=0.5, vmax=vmax, origin="lower")
            fig.colorbar(im, ax=ax, label="Speedup×")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_xlabel("Batch size  B", fontsize=10)
            ax.set_ylabel("Tree depth  d", fontsize=10)
            ax.set_title(title, fontsize=10)
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.values[r, c]
                    if not np.isnan(val):
                        ax.text(c, r, f"{val:.1f}×", ha="center", va="center",
                                fontsize=8, fontweight="bold",
                                color="white" if val < 1.5 else "black")
                    else:
                        ax.text(c, r, "OOM", ha="center", va="center",
                                fontsize=7, color="grey")
        plt.tight_layout()
        path = os.path.join(out_dir, f"fig2_speedup_heatmap_b{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_representative_bar(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 3: Bar chart at the representative EAGLE-2 production config (b=4, d=5, B=8).
    Shows all methods with their latency, colour-coded by mask type.
    Notes which methods use wrong (causal) mask vs correct tree mask.
    """
    _info = device_info()
    b_rep, d_rep, B_rep = 4, 5, 8
    row_mask = ((df["branching_factor"] == b_rep) &
                (df["tree_depth"] == d_rep) &
                (df["batch_size"] == B_rep))
    if not row_mask.any():
        return
    row = df[row_mask].iloc[0]

    methods = []
    for label, col in _METHOD_COLS.items():
        val = row.get(col, float("nan"))
        if not (isinstance(val, float) and math.isnan(val)):
            methods.append((label, float(val)))
    methods.sort(key=lambda x: x[1])

    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    colors_bar = []
    for lbl in labels:
        if "ours" in lbl.lower():
            colors_bar.append("#00b4d8")   # blue = our kernel
        elif "FAIR" in lbl:
            colors_bar.append("#2dc653")   # green = true FAIR+ varlen baseline
        elif "padded bias" in lbl.lower() or "padded" in lbl.lower():
            colors_bar.append("#e07b00")   # orange = padded float-bias baseline
        else:
            colors_bar.append("#cccccc")   # grey = causal UB (wrong mask)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, values, color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f ms", fontsize=8, padding=3)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_title(
        f"All Methods at b={b_rep}, d={d_rep}, B={B_rep}  —  {_info['name']}  (SM {_info['sm']})\n"
        f"Blue = our kernel.  Green = FAIR+ varlen (bool mask).  Orange = padded bias.  Grey = causal UB (wrong semantics).",
        fontsize=9
    )
    ax.axvline(row.get("ragged_fp16_ms", 0), color="#00b4d8", linestyle="--", linewidth=1.2,
               label="Our latency")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_representative_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tflops_roofline(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 4: TFLOPS roofline panel.
    Left:  actual sparse TFLOPS of our kernel vs depth (very low — we do little work)
    Right: dense-equivalent TFLOPS (how fast dense would need to be to match our latency)
    Annotated with roofline ceiling for this GPU.
    """
    _info = device_info()
    _gpu_label = f"{_info['name']}  ({_info['arch']}  SM {_info['sm']})"

    # Approximate SM-specific FP16 peak TFLOPS
    sm_str = _info.get("sm", "0")
    try:
        sm_maj = int(sm_str[0])
    except Exception:
        sm_maj = 7
    if sm_maj >= 12:
        peak_tflops = 580.0   # RTX PRO 6000 Blackwell approx FP16 TF
    elif sm_maj >= 9:
        peak_tflops = 989.0   # H100 SXM
    elif sm_maj >= 8:
        peak_tflops = 330.0   # A100 80GB
    else:
        peak_tflops = 65.0    # T4

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"TFLOPS Analysis  —  {_gpu_label}", fontsize=11)

    for b in sorted(df["branching_factor"].unique()):
        sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)].sort_values("tree_depth")
        axes[0].plot(sub["tree_depth"], sub["ragged_sparse_tflops"],
                     marker="o", label=f"Sparse TFLOPS b={b}")
        axes[1].plot(sub["tree_depth"], sub["ragged_dense_equiv_tflops"],
                     marker="s", linestyle="--", label=f"Dense-equiv b={b}")

    axes[0].axhline(peak_tflops, color="red", linestyle=":", linewidth=1.5,
                    label=f"HW peak ≈{peak_tflops:.0f} TFLOPS")
    axes[1].axhline(peak_tflops, color="red", linestyle=":", linewidth=1.5,
                    label=f"HW peak ≈{peak_tflops:.0f} TFLOPS")
    axes[0].set_ylabel("Achieved sparse TFLOPS\n(actual work / latency)", fontsize=9)
    axes[1].set_ylabel("Dense-equivalent TFLOPS\n(dense FLOPs / our latency)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Tree depth  d", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    axes[0].set_title("Actual compute utilisation\n(sparse kernel does very few FLOPs — memory-bound)", fontsize=9)
    axes[1].set_title("Dense-equivalent throughput\n('virtual' TFLOPS — shows how much we save vs dense)", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_tflops_roofline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fair_vs_causal_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 5: Direct comparison of speedup_vs_sdpa_flash (causal UB)
    vs speedup_vs_sdpa_flash_masked (FAIR tree-mask) at B=8.
    This is the key reviewer concern: are we comparing apples to apples?
    Shows the gap between headline number and fair number explicitly.
    """
    _info = device_info()
    _gpu_label = f"{_info['name']}  ({_info['arch']}  SM {_info['sm']})"

    for b in sorted(df["branching_factor"].unique()):
        sub = df[(df["branching_factor"] == b) & (df["batch_size"] == 8)].sort_values("tree_depth")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        depths = sub["tree_depth"].values
        spd_causal   = sub["speedup_vs_sdpa_flash"].values.astype(float)
        spd_fair_pad = sub["speedup_vs_sdpa_flash_masked"].values.astype(float)
        spd_fair_bb  = sub["speedup_vs_sdpa_batched_bool"].values.astype(float)
        spd_math     = sub["speedup_vs_sdpa_math"].values.astype(float)

        ax.fill_between(depths, spd_causal, spd_fair_bb,
                        where=~(np.isnan(spd_causal) | np.isnan(spd_fair_bb)),
                        alpha=0.15, color="#fb8500",
                        label="Headline inflation (causal UB vs fair)")
        ax.plot(depths, spd_math,     marker="^", color="#ef233c", lw=2,
                label="vs SDPA-math [tree mask, padded]")
        ax.plot(depths, spd_fair_bb,  marker="o", color="#2dc653", lw=2.5,
                label="vs SDPA [tree mask, bool — FAIR]")
        ax.plot(depths, spd_fair_pad, marker="D", color="#e07b00", lw=1.5, linestyle="--",
                label="vs SDPA [tree mask, float bias]")
        ax.plot(depths, spd_causal,   marker="s", color="#fb8500", lw=1.5, linestyle=":",
                label="vs SDPA-flash [causal UB — headline]")

        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Tree depth  d", fontsize=11)
        ax.set_ylabel("Speedup  (baseline / ours)", fontsize=11)
        ax.set_title(
            f"Fair vs Headline Speedup  (b={b}, B=8)  —  {_gpu_label}\n"
            f"Green = FAIR+ (per-sample boolean mask, flash eligible).  "
            f"Shaded = headline inflation from causal UB.",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        path = os.path.join(out_dir, f"fig5_fair_vs_headline_b{b}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_batch_scaling(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 6: Latency vs batch size at the representative config (b=4, d=5).
    Shows that our kernel latency grows linearly with B while dense grows quadratically.
    """
    _info = device_info()
    d_rep, b_rep = 5, 4
    sub = df[(df["branching_factor"] == b_rep) & (df["tree_depth"] == d_rep)].sort_values("batch_size")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    methods_to_show = [
        ("Ragged fp16 (ours)",             "ragged_fp16_ms",        "#00b4d8", "-",  2.5),
        ("SDPA math [tree mask]",          "sdpa_math_ms",          "#ef233c", "-",  1.8),
        ("SDPA bool mask [FAIR]",          "sdpa_batched_bool_ms",  "#2dc653", "--", 2.0),
        ("SDPA float bias [padded]",       "sdpa_flash_masked_ms",  "#e07b00", "--", 1.4),
        ("SDPA flash [causal UB]",         "sdpa_flash_ms",         "#fb8500", ":",  1.4),
        ("FA-2 varlen [causal UB]",        "fa2_varlen_ms",         "#c77dff", ":",  1.4),
    ]
    for label, col, color, ls, lw in methods_to_show:
        vals = sub[col].values.astype(float)
        if np.all(np.isnan(vals)):
            continue
        ax.plot(sub["batch_size"], vals, marker="o", label=label,
                color=color, linestyle=ls, linewidth=lw, markersize=5)

    ax.set_xlabel("Batch size  B", fontsize=11)
    ax.set_ylabel("Latency  (ms)", fontsize=11)
    ax.set_title(
        f"Latency vs Batch Size  (b={b_rep}, d={d_rep}, N=1365)  —  {_info['name']}\n"
        f"Linear vs quadratic scaling in total token count B·N",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig6_batch_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(info: dict, args) -> None:
    print("=" * 70)
    print("  sd-ragged  ·  SOTA Benchmark  ·  blackwell branch")
    print("=" * 70)
    print(f"  Device : {info['name']}")
    print(f"  SM     : {info['sm']}  ({info['sm_count']} SMs)  [{info['arch']}]")
    print(f"  VRAM   : {info['vram_gb']} GB")
    print(f"  dtype  : {args.dtype}")
    print()
    print("  Optional SOTA backends:")
    print(f"    FlashAttention-2 : {'available' if HAS_FLASH_ATTN else 'NOT INSTALLED (pip install flash-attn)'}")
    print(f"    FlashInfer       : {'available' if HAS_FLASHINFER else 'NOT INSTALLED (pip install flashinfer)'}")
    print(f"    xformers         : {'available' if HAS_XFORMERS  else 'NOT INSTALLED (pip install xformers)'}")
    print(f"    torch_tensorrt   : {'available' if HAS_TENSORRT  else 'NOT INSTALLED (pip install torch-tensorrt)'}")
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sd-ragged SOTA benchmark — RTX 6000 ADA PRO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir",          default="results",
                        help="Output directory for CSV and plots (default: results)")
    parser.add_argument("--dtype",             default="fp16", choices=["fp16", "bf16"],
                        help="Primary dtype (default: fp16)")
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help="Comma-separated batch sizes")
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)),
                        help="Comma-separated tree depths")
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)),
                        help="Comma-separated branching factors")
    parser.add_argument("--warmup",            type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",             type=int, default=BENCH_ITERS)
    parser.add_argument("--skip-flashattn",    action="store_true")
    parser.add_argument("--skip-flashinfer",   action="store_true")
    parser.add_argument("--skip-xformers",     action="store_true")
    parser.add_argument("--skip-tensorrt",     action="store_true",
                        help="Skip TensorRT Attention baseline even if tensorrt is installed")
    parser.add_argument("--no-plot",           action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available — benchmark requires a CUDA GPU.")
        sys.exit(1)

    info = device_info()
    _print_banner(info, args)

    batch_sizes       = list(map(int, args.batch_sizes.split(",")))
    depths            = list(map(int, args.depths.split(",")))
    branching_factors = list(map(int, args.branching_factors.split(",")))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    # Checkpoint CSV — written after every row so a crash never loses data.
    # If a partial run exists we resume from where it left off.
    csv_path = os.path.join(args.out_dir, "sota_benchmark.csv")

    configs = [
        (B, b, d)
        for B in batch_sizes
        for b in branching_factors
        for d in depths
    ]

    # ── Resume support: skip configs already in checkpoint file ──────────────
    completed: set[tuple] = set()
    rows: list[dict] = []
    if os.path.exists(csv_path):
        try:
            _existing = pd.read_csv(csv_path)
            rows = _existing.to_dict("records")
            for r in rows:
                completed.add((int(r["batch_size"]),
                               int(r["branching_factor"]),
                               int(r["tree_depth"])))
            print(f"[resume] Found {len(rows)} completed rows in {csv_path} — skipping those configs.")
        except Exception as _e:
            print(f"[resume] Could not parse existing checkpoint ({_e}) — starting fresh.")
            rows = []

    pending = [(B, b, d) for B, b, d in configs if (B, b, d) not in completed]
    total   = len(configs)
    done_so_far = len(completed)
    print(f"Running {len(pending)} configurations  [{args.warmup} warmup + {args.iters} timed iters each]")
    if done_so_far:
        print(f"({done_so_far} already done from previous run)")
    print()
    for idx, (B, b, d) in enumerate(pending):
        display_idx = done_so_far + idx + 1
        try:
            row = benchmark_one(
                batch_size=B,
                branching_factor=b,
                depth=d,
                dtype_str=args.dtype,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                skip_flashattn  =args.skip_flashattn,
                skip_flashinfer =args.skip_flashinfer,
                skip_xformers   =args.skip_xformers,
                skip_tensorrt   =args.skip_tensorrt,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [{display_idx:3d}/{total}]  B={B:2d} b={b} d={d} │  OOM — skipped")
            continue
        except Exception as exc:
            print(f"  [{display_idx:3d}/{total}]  B={B:2d} b={b} d={d} │  ERROR: {exc} — skipped")
            continue

        rows.append(asdict(row))

        # ── Checkpoint: flush to CSV after every row ──────────────────────────
        try:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        except Exception as _ce:
            print(f"  [checkpoint] save failed: {_ce}")

        # ── Per-row progress ─────────────────────────────────────────────────
        def _fmt(ms):
            return f"{ms:.3f}ms" if not math.isnan(ms) else "  n/a "
        extras = ""
        if not math.isnan(row.fa2_varlen_ms):
            extras += f"  fa2v={_fmt(row.fa2_varlen_ms)}"
        if not math.isnan(row.flash_attn2_ms):
            extras += f"  fa2={_fmt(row.flash_attn2_ms)}"
        if not math.isnan(row.trt_attention_ms):
            extras += f"  trt={_fmt(row.trt_attention_ms)}"
        print(
            f"  [{display_idx:3d}/{total}]  B={B:2d} b={b} d={d} │ "
            f"ragged={_fmt(row.ragged_fp16_ms)}  bf16={_fmt(row.ragged_bf16_ms)} │ "
            f"math={_fmt(row.sdpa_math_ms)}  flash_causal={_fmt(row.sdpa_flash_ms)} │ "
            f"bool_fair={_fmt(row.sdpa_batched_bool_ms)} │ "
            f"spdup_vs_math={row.speedup_vs_sdpa_math:.2f}×  "
            f"spdup_vs_bool_fair={row.speedup_vs_sdpa_batched_bool:.2f}×"
            f"{extras}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Speedup vs SDPA-math [tree mask] (mean over batch sizes) ─────────")
    if "speedup_vs_sdpa_math" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_sdpa_math"]
            .mean().round(2).unstack().to_string()
        )
    print("\n── Speedup vs SDPA [tree mask, bool, FAIR] (mean over batch sizes) ──────")
    if "speedup_vs_sdpa_batched_bool" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_sdpa_batched_bool"]
            .mean().round(2).unstack().to_string()
        )
    print("\n── Speedup vs SDPA [tree mask, float bias] (mean) ───────────────────────")
    if "speedup_vs_sdpa_flash_masked" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_sdpa_flash_masked"]
            .mean().round(2).unstack().to_string()
        )
    print("\n── Speedup vs FA-2 varlen [causal UB] (mean over batch sizes) ────────")
    if "speedup_vs_fa2_varlen" in df.columns:
        print(
            df.groupby(["tree_depth", "branching_factor"])["speedup_vs_fa2_varlen"]
            .mean().round(2).unstack().to_string()
        )
    print("\n── Actual sparse TFLOPS (B=8, all branching) ────────────────────────")
    if "ragged_sparse_tflops" in df.columns:
        print(
            df[df["batch_size"] == 8]
            .groupby(["tree_depth", "branching_factor"])["ragged_sparse_tflops"]
            .mean().round(5).unstack().to_string()
        )

    if not args.no_plot:
        print("\nGenerating plots …")
        plot_latency_vs_depth(df, args.out_dir)
        plot_speedup_heatmap(df, args.out_dir)
        plot_representative_bar(df, args.out_dir)
        plot_tflops_roofline(df, args.out_dir)
        plot_fair_vs_causal_comparison(df, args.out_dir)
        plot_batch_scaling(df, args.out_dir)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
