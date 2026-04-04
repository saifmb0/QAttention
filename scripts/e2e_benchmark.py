"""
e2e_benchmark.py  —  End-to-end transformer comparison benchmark
=================================================================

PURPOSE
-------
This script answers TWO questions:

    1. "How fast is our ragged kernel in a full L-layer transformer stack
       versus FA2, FlashInfer, and DeFT baselines?"

    2. "What fraction of total transformer time is spent in attention,
       and how does that change with tree depth / branching factor?"

ATTENTION METHODS COMPARED
---------------------------
  ragged       — our ancestor-sparse Triton kernel  (O(d+1) per query)
  sdpa_flash   — FA2 via torch.nn.attention.sdpa_kernel(FLASH_ATTENTION)
                 padded to [B, H, N, D] with causal mask (upper bound)
  flashinfer   — FlashInfer BatchPrefillWithRaggedKVCacheWrapper, causal=True
                 ragged layout, no padding (upper bound)
  deft         — DeFT (arXiv:2404.00242) standalone Triton kernel
                 PanZaifeng/FastTree-Artifact/kernel_bench/
                 tree-structured KV cache, all N nodes as queries

  The SDPA flash and FlashInfer baselines use CAUSAL masking (not tree
  masking) — they are upper-bound references only.  DeFT uses correct
  tree structure.

WHAT IS MEASURED
----------------
Per (method, B, b, d) configuration:

  fwd_ms       — full L-layer transformer forward pass:
                   embed → (QKV proj → attention → O proj → RMSNorm
                            → SwiGLU FFN) × L layers
  attn_only_ms — single attention sublayer (QKV proj + attention + O proj
                 + RMSNorm), timed on one block as a proxy.
  attn_frac    ≈ (attn_only_ms × L) / fwd_ms
  tok_per_sec  — verification tokens per second (full model)
  speedup      — fwd_ms(baseline) / fwd_ms(ragged)  per config

CAVEATS
--------
  • Random fp16 weights — not real LLaMA-2 parameters.
  • No draft model, no speculative sampling, no EAGLE-2 pipeline.
  • tok_per_sec is synthetic ragged-path throughput, not real EAGLE-2 speed.
  • FlashInfer plan() and DeFT preparation called once, not timed.

Model presets (hidden, num_heads, head_dim, ffn_hidden, layers):
  synthetic …  1 024-dim,  8 heads, 128 D,  2 816 FFN,  4 layers (smoke test)
  7b        …  4 096-dim, 32 heads, 128 D, 11 008 FFN, 32 layers  (LLaMA-2 7B)
  13b       …  5 120-dim, 40 heads, 128 D, 13 824 FFN, 40 layers  (LLaMA-2 13B)

Usage
-----
  python scripts/e2e_benchmark.py \\
      --model-size 7b \\
      --batch-sizes 1,2,4,8 \\
      --depths 3,5,7 \\
      --branching-factors 2,3 \\
      --methods ragged,sdpa_flash,flashinfer,deft \\
      --out-dir results

  # Skip unavailable methods
  python scripts/e2e_benchmark.py --skip-flashinfer --skip-deft

Output
------
  results/e2e_benchmark.csv  — per-row: attn_method, model, batch, depth,
                                bfactor, tokens, layers, fwd_ms,
                                attn_only_ms, tok_per_sec, attn_frac
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local project root on sys.path so we can import src.ragged_attn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragged_attn import ragged_attention
from src.tree_mask import num_tree_nodes

# ── Optional library probes ───────────────────────────────────────────────────

def _has(pkg: str) -> bool:
    if importlib.util.find_spec(pkg) is None:
        return False
    try:
        __import__(pkg)
        return True
    except Exception:
        return False

HAS_FLASHINFER = _has("flashinfer")

_DEFT_KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "FastTree", "kernel_bench",
)
HAS_DEFT = os.path.isfile(os.path.join(_DEFT_KERNEL_DIR, "DeFT.py"))

# All available attention methods
ALL_METHODS = ["ragged", "sdpa_flash", "flashinfer", "deft"]

# ── Model presets ─────────────────────────────────────────────────────────────

MODEL_PRESETS = {
    #  label       hidden  heads  head_d  ffn     layers
    "synthetic": dict(hidden=1024,  H=8,  D=128, ffn=2816,  L=4),
    "7b":        dict(hidden=4096,  H=32, D=128, ffn=11008, L=32),
    "13b":       dict(hidden=5120,  H=40, D=128, ffn=13824, L=40),
}

# ── Benchmark defaults ────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZES       = [1, 2, 4, 8]
DEFAULT_DEPTHS            = [3, 5, 7]
DEFAULT_BRANCHING_FACTORS = [2, 3]
WARMUP_ITERS              = 3
BENCH_ITERS               = 10

# ── Minimal transformer primitives ───────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class RaggedAttnLayer(nn.Module):
    """
    Single attention sublayer that wraps ``ragged_attention``.
    Input:  x  [total_tokens, hidden]
    Output: x  [total_tokens, hidden]  (residual-added)
    """

    def __init__(self, hidden: int, H: int, D: int):
        super().__init__()
        self.H = H
        self.D = D
        self.qkv_proj = nn.Linear(hidden, 3 * H * D, bias=False)
        self.o_proj   = nn.Linear(H * D, hidden, bias=False)
        self.norm     = RMSNorm(hidden)

    def forward(
        self,
        x:             torch.Tensor,   # [total_tokens, hidden]
        cu_seqlens:    torch.Tensor,   # [B+1] int32
        branching_factor: int,
        depth:            int,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        total_tokens = x.shape[0]
        B = cu_seqlens.shape[0] - 1

        qkv = self.qkv_proj(x)                                      # [T, 3·H·D]
        Q, K, V = qkv.chunk(3, dim=-1)                              # each [T, H·D]
        Q = Q.view(total_tokens, self.H, self.D)
        K = K.view(total_tokens, self.H, self.D)
        V = V.view(total_tokens, self.H, self.D)

        out = ragged_attention(Q, K, V, cu_seqlens,
                               branching_factor=branching_factor,
                               max_depth=depth)                      # [T, H, D]
        out = out.view(total_tokens, self.H * self.D)
        out = self.o_proj(out)
        return residual + out

    def prepare(self, cu_seqlens, B, N, branching_factor, depth, device):
        """No-op — ragged kernel needs no pre-planning."""
        pass


class SdpaFlashAttnLayer(nn.Module):
    """
    FA2 baseline via torch.nn.attention.sdpa_kernel(FLASH_ATTENTION).

    Reshapes ragged [T, H, D] → padded [B, H, N, D], runs SDPA flash with
    causal mask, reshapes back.  All batch items have equal length N in
    this benchmark, so reshape is zero-copy.

    Uses causal masking (NOT tree masking) — this is an UPPER BOUND reference.
    """

    def __init__(self, hidden: int, H: int, D: int):
        super().__init__()
        self.H, self.D = H, D
        self.qkv_proj = nn.Linear(hidden, 3 * H * D, bias=False)
        self.o_proj   = nn.Linear(H * D, hidden, bias=False)
        self.norm     = RMSNorm(hidden)
        self._scale   = 1.0 / math.sqrt(D)

    def prepare(self, cu_seqlens, B, N, branching_factor, depth, device):
        pass

    def forward(self, x, cu_seqlens, branching_factor, depth):
        residual = x
        x = self.norm(x)
        T = x.shape[0]
        B = cu_seqlens.shape[0] - 1
        N = T // B

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        # [T, H·D] → [B, N, H, D] → [B, H, N, D]
        Q = Q.view(B, N, self.H, self.D).transpose(1, 2)
        K = K.view(B, N, self.H, self.D).transpose(1, 2)
        V = V.view(B, N, self.H, self.D).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, scale=self._scale,
            )
        out = out.transpose(1, 2).reshape(T, self.H * self.D)
        out = self.o_proj(out)
        return residual + out


class FlashInferAttnLayer(nn.Module):
    """
    FlashInfer baseline — BatchPrefillWithRaggedKVCacheWrapper, causal=True.

    Uses ragged (packed) layout — no padding waste.  plan() is called once
    in prepare(), run() is called each forward.

    Uses causal masking (NOT tree masking) — this is an UPPER BOUND reference.
    """

    def __init__(self, hidden: int, H: int, D: int):
        super().__init__()
        self.H, self.D = H, D
        self.qkv_proj = nn.Linear(hidden, 3 * H * D, bias=False)
        self.o_proj   = nn.Linear(H * D, hidden, bias=False)
        self.norm     = RMSNorm(hidden)
        self._wrapper = None

    def prepare(self, cu_seqlens, B, N, branching_factor, depth, device):
        import flashinfer  # type: ignore
        workspace_buf = torch.empty(
            32 * 1024 * 1024, dtype=torch.uint8, device=device,
        )
        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buf, kv_layout="NHD",
        )
        self._wrapper.plan(
            cu_seqlens.to(device), cu_seqlens.to(device),
            num_qo_heads=self.H, num_kv_heads=self.H, head_dim_qk=self.D,
            causal=True,
        )
        # Warmup JIT
        T = B * N
        _Q = torch.randn(T, self.H, self.D, device=device, dtype=torch.float16)
        _K = torch.randn(T, self.H, self.D, device=device, dtype=torch.float16)
        _V = torch.randn(T, self.H, self.D, device=device, dtype=torch.float16)
        try:
            self._wrapper.run(_Q, _K, _V)
            torch.cuda.synchronize()
        except Exception:
            pass

    def forward(self, x, cu_seqlens, branching_factor, depth):
        residual = x
        x = self.norm(x)
        T = x.shape[0]

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(T, self.H, self.D).to(torch.float16)
        K = K.view(T, self.H, self.D).to(torch.float16)
        V = V.view(T, self.H, self.D).to(torch.float16)

        out = self._wrapper.run(Q, K, V)
        out = out.view(T, self.H * self.D)
        out = self.o_proj(out)
        return residual + out


class DeFTAttnLayer(nn.Module):
    """
    DeFT baseline (arXiv:2404.00242) — standalone Triton kernel.

    prepare() builds the ancestor-chain scatter index and calls
    DeFT_preparation once.  forward() gathers K/V into DeFT's K_cache
    format and runs DeFT_decode for each batch item.

    This uses CORRECT tree masking — directly comparable to our ragged kernel.
    """

    def __init__(self, hidden: int, H: int, D: int):
        super().__init__()
        self.H, self.D = H, D
        self.qkv_proj = nn.Linear(hidden, 3 * H * D, bias=False)
        self.o_proj   = nn.Linear(H * D, hidden, bias=False)
        self.norm     = RMSNorm(hidden)
        self._sm_scale = 1.0 / math.sqrt(D)
        # Populated by prepare()
        self._scatter_idx = None   # [N, d+1] long
        self._deft_aux    = None   # tuple from DeFT_preparation
        self._N           = 0
        self._B           = 0
        self._max_path    = 0
        self._mask_len    = 64
        self._DeFT_decode = None
        self._deft_mod    = None

    def prepare(self, cu_seqlens, B, N, branching_factor, depth, device):
        import sys as _sys
        if _DEFT_KERNEL_DIR not in _sys.path:
            _sys.path.insert(0, _DEFT_KERNEL_DIR)
        from kv_tree_simple import KVTreeNode   # type: ignore
        from DeFT import DeFT_preparation, DeFT_decode  # type: ignore
        import DeFT as _deft_mod                # type: ignore

        self._DeFT_decode = DeFT_decode
        self._deft_mod    = _deft_mod
        self._N = N
        self._B = B
        self._max_path = depth + 1

        # Parent array for BFS b-ary tree
        parent_arr = [-1] * N
        for i in range(1, N):
            parent_arr[i] = (i - 1) // branching_factor

        # Ancestor chains: chain[i] = [root, ..., i]
        ancestor_chains: list[list[int]] = []
        for i in range(N):
            chain: list[int] = []
            cur = i
            while cur != -1:
                chain.append(cur)
                cur = parent_arr[cur]
            chain.reverse()
            ancestor_chains.append(chain)

        # Children lists
        children: list[list[int]] = [[] for _ in range(N)]
        for i in range(1, N):
            children[parent_arr[i]].append(i)

        def _bfs_subtree(root: int) -> list[int]:
            result = [root]
            q: list[int] = [root]
            qi = 0
            while qi < len(q):
                node = q[qi]; qi += 1
                for child in children[node]:
                    result.append(child)
                    q.append(child)
            return result

        # KVTreeNode list
        tree_info = []
        for j in range(N):
            node = KVTreeNode()
            node.parent       = parent_arr[j]
            node.id           = j
            node.seqlen       = 1
            node.num_children = len(children[j])
            node.requests     = _bfs_subtree(j)
            tree_info.append(node)

        # Scatter index: idx[i, pos] = ancestor of node i at depth pos
        idx = torch.zeros(N, self._max_path, dtype=torch.long, device=device)
        for i in range(N):
            chain = ancestor_chains[i]
            for pos, anc in enumerate(chain):
                idx[i, pos] = anc
            for pos in range(len(chain), self._max_path):
                idx[i, pos] = chain[-1]
        self._scatter_idx = idx

        # DeFT_preparation (CPU-side) — reset cur_length first
        subtree_len = 128
        _deft_mod.cur_length = 0
        # Build dummy K_cache to get aux data shapes
        K_dummy = torch.randn(N, self._max_path, self.H, self.D,
                              device=device, dtype=torch.float16)
        self._deft_aux = DeFT_preparation(
            tree_info, K_dummy, subtree_len, self._mask_len, self.H, self.D,
        )

        # Warmup DeFT Triton kernel
        Q_w = torch.randn(N, self.H, self.D, device=device, dtype=torch.float16)
        Kf  = K_dummy.reshape(-1, self.H, self.D)
        Vf  = K_dummy.reshape(-1, self.H, self.D)
        Out = torch.empty_like(Q_w)
        try:
            DeFT_decode(
                Q_w, Kf, Vf, Out, *self._deft_aux,
                Q_TILE_SIZE=16, KV_TILE_SIZE=32,
                sm_scale=self._sm_scale, mask_len=self._mask_len,
            )
            torch.cuda.synchronize()
        except Exception:
            pass

    def forward(self, x, cu_seqlens, branching_factor, depth):
        residual = x
        x = self.norm(x)
        T = x.shape[0]

        qkv = self.qkv_proj(x)
        Q_all, K_all, V_all = qkv.chunk(3, dim=-1)
        Q_all = Q_all.view(T, self.H, self.D).to(torch.float16)
        K_all = K_all.view(T, self.H, self.D).to(torch.float16)
        V_all = V_all.view(T, self.H, self.D).to(torch.float16)

        Out = torch.empty(T, self.H, self.D, device=x.device, dtype=torch.float16)

        for b_idx in range(self._B):
            s = b_idx * self._N
            e = s + self._N
            Q_b = Q_all[s:e].contiguous()
            K_b = K_all[s:e].contiguous()
            V_b = V_all[s:e].contiguous()

            # Gather into ancestor-chain format: [N, d+1, H, D]
            K_cache = K_b[self._scatter_idx.view(-1)].view(
                self._N, self._max_path, self.H, self.D
            )
            V_cache = V_b[self._scatter_idx.view(-1)].view(
                self._N, self._max_path, self.H, self.D
            )

            self._DeFT_decode(
                Q_b,
                K_cache.reshape(-1, self.H, self.D),
                V_cache.reshape(-1, self.H, self.D),
                Out[s:e],
                *self._deft_aux,
                Q_TILE_SIZE=16, KV_TILE_SIZE=32,
                sm_scale=self._sm_scale,
                mask_len=self._mask_len,
            )

        out = Out.view(T, self.H * self.D)
        out = self.o_proj(out)
        return residual + out


# Map of method name → attention layer class
ATTN_LAYER_CLS = {
    "ragged":      RaggedAttnLayer,
    "sdpa_flash":  SdpaFlashAttnLayer,
    "flashinfer":  FlashInferAttnLayer,
    "deft":        DeFTAttnLayer,
}


class RaggedFFNLayer(nn.Module):
    """
    SwiGLU FFN sublayer.
    Input / output: [total_tokens, hidden]
    """

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden, 2 * ffn, bias=False)
        self.down    = nn.Linear(ffn, hidden, bias=False)
        self.norm    = RMSNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return residual + self.down(torch.nn.functional.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, H: int, D: int, ffn: int,
                 attn_cls: type = RaggedAttnLayer):
        super().__init__()
        self.attn = attn_cls(hidden, H, D)
        self.ffn  = RaggedFFNLayer(hidden, ffn)

    def forward(self, x, cu_seqlens, branching_factor, depth):
        x = self.attn(x, cu_seqlens, branching_factor, depth)
        x = self.ffn(x)
        return x


class SyntheticModel(nn.Module):
    """
    Stack of ``L`` transformer blocks operating on ragged (packed) token sequences.
    Used purely for timing — weights are random fp16, no sampling head needed.
    The attention layer class is parameterised to allow swapping in baselines.
    """

    def __init__(self, hidden: int, H: int, D: int, ffn: int, L: int,
                 attn_cls: type = RaggedAttnLayer):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden, H, D, ffn, attn_cls) for _ in range(L)
        ])
        self.embed = nn.Embedding(32000, hidden)

    def forward(self, token_ids, cu_seqlens, branching_factor, depth):
        x = self.embed(token_ids).to(torch.float16)
        for layer in self.layers:
            x = layer(x, cu_seqlens, branching_factor, depth)
        return x

    def prepare_all_layers(self, cu_seqlens, B, N, branching_factor, depth, device):
        """Call prepare() on every attention sublayer (FlashInfer plan, DeFT prep, etc.)."""
        for layer in self.layers:
            layer.attn.prepare(cu_seqlens, B, N, branching_factor, depth, device)

# ── Timing helpers ────────────────────────────────────────────────────────────

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn, warmup: int, iters: int) -> float:
    """Returns mean wall-clock time in milliseconds over *iters* runs."""
    for _ in range(warmup):
        fn()
    _sync_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_cuda()
    return (time.perf_counter() - t0) * 1e3 / iters


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class E2ERow:
    attn_method:           str
    model_size:            str
    num_layers:            int
    batch_size:            int
    branching_factor:      int
    depth:                 int
    tokens_per_seq:        int
    total_tokens:          int
    fwd_ms:                float   # full forward pass (ms)
    attn_only_ms:          float   # attention sublayer alone, single block (ms)
    tok_per_sec:           float   # tokens / second  (full model)
    attn_frac:             float   # approx fraction of runtime in attention


# ── Core benchmark function ───────────────────────────────────────────────────

def benchmark_e2e(
    model_size:        str,
    batch_size:        int,
    branching_factor:  int,
    depth:             int,
    attn_method:       str  = "ragged",
    warmup:            int  = WARMUP_ITERS,
    iters:             int  = BENCH_ITERS,
    device:            torch.device | None = None,
) -> E2ERow:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MODEL_PRESETS[model_size]
    hidden, H, D, ffn, L = cfg["hidden"], cfg["H"], cfg["D"], cfg["ffn"], cfg["L"]

    N = num_tree_nodes(branching_factor, depth)
    B = batch_size
    total_tokens = B * N

    attn_cls = ATTN_LAYER_CLS[attn_method]

    # ── Build model (fp16, no grad) ──────────────────────────────────────────
    model = SyntheticModel(hidden, H, D, ffn, L, attn_cls=attn_cls).to(device).half().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Build ragged inputs ───────────────────────────────────────────────────
    torch.manual_seed(batch_size * 1000 + branching_factor * 100 + depth)
    token_ids = torch.randint(0, 32000, (total_tokens,), device=device)
    cu_sl = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.int32)

    # ── Prepare attention layers (plan / prep — not timed) ───────────────────
    model.prepare_all_layers(cu_sl, B, N, branching_factor, depth, device)

    # ── Time full forward ────────────────────────────────────────────────────
    fwd_ms = _time_fn(
        lambda: model(token_ids, cu_sl, branching_factor, depth),
        warmup, iters
    )

    # ── Time a single attention sublayer (proxy for attn fraction) ───────────
    attn_layer = model.layers[0].attn
    with torch.no_grad():
        x_sample = model.embed(token_ids).half()

    attn_only_ms = _time_fn(
        lambda: attn_layer(x_sample, cu_sl, branching_factor, depth),
        warmup, iters
    )

    tok_per_sec = total_tokens / (fwd_ms * 1e-3)
    attn_frac = min(1.0, (attn_only_ms * L) / max(fwd_ms, 1e-9))

    return E2ERow(
        attn_method      = attn_method,
        model_size       = model_size,
        num_layers       = L,
        batch_size       = batch_size,
        branching_factor = branching_factor,
        depth            = depth,
        tokens_per_seq   = N,
        total_tokens     = total_tokens,
        fwd_ms           = round(fwd_ms,        3),
        attn_only_ms     = round(attn_only_ms,  3),
        tok_per_sec      = round(tok_per_sec,   1),
        attn_frac        = round(attn_frac,     4),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end comparison benchmark for ragged (tree) speculative decoding"
    )
    parser.add_argument("--model-size",        default="synthetic",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model preset (default: synthetic)")
    parser.add_argument("--batch-sizes",       default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help="Comma-separated list of batch sizes")
    parser.add_argument("--depths",            default=",".join(map(str, DEFAULT_DEPTHS)),
                        help="Comma-separated list of tree depths")
    parser.add_argument("--branching-factors", default=",".join(map(str, DEFAULT_BRANCHING_FACTORS)),
                        help="Comma-separated list of branching factors")
    parser.add_argument("--methods",           default=None,
                        help="Comma-separated attention methods to run "
                             "(default: all available). Choices: "
                             + ",".join(ALL_METHODS))
    parser.add_argument("--skip-flashinfer",   action="store_true",
                        help="Skip FlashInfer baseline")
    parser.add_argument("--skip-deft",         action="store_true",
                        help="Skip DeFT baseline")
    parser.add_argument("--warmup",            type=int, default=WARMUP_ITERS)
    parser.add_argument("--iters",             type=int, default=BENCH_ITERS)
    parser.add_argument("--out-dir",           default="results",
                        help="Directory to write CSV output")
    parser.add_argument("--csv-name",          default="e2e_benchmark.csv",
                        help="Output CSV filename (relative to --out-dir)")
    args = parser.parse_args()

    batch_sizes       = [int(x) for x in args.batch_sizes.split(",")]
    depths            = [int(x) for x in args.depths.split(",")]
    branching_factors = [int(x) for x in args.branching_factors.split(",")]

    # ── Resolve methods to run ────────────────────────────────────────────────
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = list(ALL_METHODS)          # start with all
    if args.skip_flashinfer or not HAS_FLASHINFER:
        methods = [m for m in methods if m != "flashinfer"]
        if not HAS_FLASHINFER and "flashinfer" not in (args.methods or ""):
            print("  [info] FlashInfer not available — skipping.")
    if args.skip_deft or not HAS_DEFT:
        methods = [m for m in methods if m != "deft"]
        if not HAS_DEFT and "deft" not in (args.methods or ""):
            print("  [info] DeFT not available — skipping.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = MODEL_PRESETS[args.model_size]
    print(
        f"\ne2e_benchmark  —  end-to-end transformer comparison"
        f"\n  model={args.model_size}  ({cfg['L']} layers, hidden={cfg['hidden']}, "
        f"H={cfg['H']}, D={cfg['D']})"
        f"\n  device={device}   dtype=fp16   random weights"
        f"\n  methods: {', '.join(methods)}"
    )
    print()
    print("  Metrics per (method, B, d, b):")
    print("    fwd_ms       — full L-layer forward pass (embed+attn+FFN) × L")
    print("    attn_only_ms — single attention sublayer (proxy for per-layer cost)")
    print("    attn_frac    — estimated attention share of total compute")
    print("    tok/s        — verification tokens/sec (synthetic)")
    print()

    configs = [
        (B, d, b)
        for B in batch_sizes
        for d in depths
        for b in branching_factors
    ]
    total_runs = len(configs) * len(methods)
    run_idx    = 0
    rows: list[dict] = []

    for method in methods:
        print(f"\n{'─' * 70}")
        print(f"  Method: {method}")
        print(f"{'─' * 70}")
        for B, d, b in configs:
            run_idx += 1
            N = num_tree_nodes(b, d)
            print(
                f"  [{run_idx:3d}/{total_runs}] {method:12s}  "
                f"B={B}  d={d}  b={b}  N={N}  T={B*N} … ",
                end="", flush=True,
            )
            try:
                row = benchmark_e2e(
                    model_size       = args.model_size,
                    batch_size       = B,
                    branching_factor = b,
                    depth            = d,
                    attn_method      = method,
                    warmup           = args.warmup,
                    iters            = args.iters,
                    device           = device,
                )
                rows.append(asdict(row))
                print(
                    f"fwd={row.fwd_ms:.1f}ms  "
                    f"tok/s={row.tok_per_sec:.0f}  "
                    f"attn_frac={row.attn_frac:.1%}"
                )
            except Exception as exc:
                print(f"ERROR: {exc}")

    if not rows:
        print("No results collected.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, args.csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {csv_path}")

    # ── Comparison summary ────────────────────────────────────────────────────
    from collections import defaultdict

    # Index: (B, d, b, method) → fwd_ms
    fwd_index: dict[tuple, float] = {}
    for r in rows:
        key = (r["batch_size"], r["depth"], r["branching_factor"], r["attn_method"])
        fwd_index[key] = r["fwd_ms"]

    # Print comparison table
    print(f"\n{'═' * 80}")
    print("  E2E Comparison — fwd_ms and speedup vs ragged")
    print(f"{'═' * 80}")

    hdr_methods = "".join(f"  {m:>14s}" for m in methods)
    spd_methods = "".join(f"  {'spdup':>14s}" for m in methods if m != "ragged")
    print(f"  {'B':>3s}  {'d':>2s}  {'b':>2s}  {'N':>6s}{hdr_methods}{spd_methods}")
    print(f"  {'':>3s}  {'':>2s}  {'':>2s}  {'':>6s}"
          + "".join(f"  {'(ms)':>14s}" for _ in methods)
          + "".join(f"  {'(×)':>14s}" for m in methods if m != "ragged"))
    print("  " + "─" * (3+2+2+6 + 16*len(methods) + 16*(len(methods)-1) + 6))

    for B, d, b in configs:
        N = num_tree_nodes(b, d)
        parts = [f"  {B:3d}  {d:2d}  {b:2d}  {N:6d}"]

        # fwd_ms for each method
        ms_vals = []
        for m in methods:
            ms = fwd_index.get((B, d, b, m))
            if ms is not None:
                parts.append(f"  {ms:14.1f}")
                ms_vals.append((m, ms))
            else:
                parts.append(f"  {'—':>14s}")
                ms_vals.append((m, None))

        # speedup vs ragged for non-ragged methods
        ragged_ms = fwd_index.get((B, d, b, "ragged"))
        for m, ms in ms_vals:
            if m == "ragged":
                continue
            if ragged_ms and ms:
                parts.append(f"  {ms / ragged_ms:14.2f}")
            else:
                parts.append(f"  {'—':>14s}")

        print("".join(parts))

    # ── Per-method tok/s and attn_frac summary ────────────────────────────────
    for method in methods:
        method_rows = [r for r in rows if r["attn_method"] == method]
        if not method_rows:
            continue
        print(f"\n── {method} — tok/s summary (all batch sizes averaged) ──")
        groups: dict[tuple, list[float]] = defaultdict(list)
        for r in method_rows:
            groups[(r["depth"], r["branching_factor"])].append(r["tok_per_sec"])
        hdr = f"{'depth':>6}  {'b':>3}  {'mean_tok/s':>12}  {'min_tok/s':>12}"
        print(hdr)
        print("-" * len(hdr))
        for (d, b), vals in sorted(groups.items()):
            print(f"{d:>6}  {b:>3}  {sum(vals)/len(vals):>12.1f}  {min(vals):>12.1f}")

    print()
    print("─" * 70)
    print("  Notes:")
    print("  • sdpa_flash / flashinfer use CAUSAL masking (upper bound refs)")
    print("  • deft uses CORRECT tree masking (directly comparable)")
    print("  • speedup > 1.0 means BASELINE is slower than ragged")
    print("  • Random fp16 weights — scaling trends, not absolute throughput")
    print("─" * 70)


if __name__ == "__main__":
    main()
