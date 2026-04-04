"""
e2e_benchmark.py  —  Kernel overhead profiling in a full transformer stack
===========================================================================

PURPOSE
-------
This script answers ONE question:

    "What fraction of total transformer time is spent in attention,
     and how does that change with tree depth / branching factor?"

This is NOT a comparison benchmark.  Only our ragged ancestor-sparse
attention kernel is run.  For kernel-vs-kernel latency comparisons
(ours vs SDPA bool-mask vs FlashInfer vs DeFT) see benchmark_sota.py.

WHAT IS MEASURED
----------------
Two timings per configuration:

  fwd_ms       — full L-layer transformer forward pass:
                   embed → (QKV proj → ragged_attention → O proj → RMSNorm
                            → SwiGLU FFN) × L layers
  attn_only_ms — single attention sublayer (QKV proj + ragged_attention +
                 O proj + one RMSNorm), timed on a single block as a proxy.
  attn_frac    ≈ (attn_only_ms × L) / fwd_ms
               The fraction of total compute that lives in attention.
               Plug into Amdahl's law to bound the end-to-end speedup a
               faster attention kernel can deliver.
  tok_per_sec  — verification tokens processed per second on the ragged path.

WHAT IS NOT MEASURED / NOT CLAIMED
------------------------------------
  • No comparison kernel. No speedup ratio.  Use benchmark_sota.py for that.
  • Random fp16 weights — not real LLaMA-2 parameters.
  • No draft model, no speculative sampling, no EAGLE-2 pipeline.
  • tok_per_sec is synthetic ragged-path throughput, not real EAGLE-2 speed.

HOW TO READ THE OUTPUT
-----------------------
  attn_frac answers "how much room is there to improve end-to-end speed by
  improving the attention kernel?"  Example:
    attn_frac=0.05 at d=3 → a 10× attention speedup gains only ~5% overall.
    attn_frac=0.45 at d=7 → a 10× attention speedup gains ~29% overall.
  (Amdahl: 1 / (1 - attn_frac × (1 - 1/kernel_speedup)))

  Use benchmark_sota.py to see what actual attention speedups are achieved.

Model presets (hidden, num_heads, head_dim, ffn_hidden, layers):
  synthetic …  1 024-dim,  8 heads, 128 D,  2 816 FFN,  4 layers (smoke test)
  7b        …  4 096-dim, 32 heads, 128 D, 11 008 FFN, 32 layers  (LLaMA-2 7B)
  13b       …  5 120-dim, 40 heads, 128 D, 13 824 FFN, 40 layers  (LLaMA-2 13B)

  NOTE: head_dim=128 throughout.  benchmark_sota.py uses head_dim=64.
  Run pytest tests/ to verify the kernel supports head_dim=128 before
  publishing attn_frac numbers from the 7b/13b presets.

Usage
-----
  python scripts/e2e_benchmark.py \
      --model-size 7b \
      --batch-sizes 1,2,4,8 \
      --depths 3,5,7 \
      --branching-factors 2,3 \
      --out-dir results

Output
------
  results/e2e_benchmark.csv  — per-row: model, batch, depth, bfactor, tokens,
                                layers, fwd_ms, attn_only_ms, tok_per_sec,
                                attn_frac
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import List

import torch
import torch.nn as nn

# Local project root on sys.path so we can import src.ragged_attn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ragged_attn import ragged_attention
from src.tree_mask import num_tree_nodes

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


class RaggedTransformerBlock(nn.Module):
    def __init__(self, hidden: int, H: int, D: int, ffn: int):
        super().__init__()
        self.attn = RaggedAttnLayer(hidden, H, D)
        self.ffn  = RaggedFFNLayer(hidden, ffn)

    def forward(self, x, cu_seqlens, branching_factor, depth):
        x = self.attn(x, cu_seqlens, branching_factor, depth)
        x = self.ffn(x)
        return x


class SyntheticRaggedModel(nn.Module):
    """
    Stack of ``L`` transformer blocks operating on ragged (packed) token sequences.
    Used purely for timing — weights are random fp16, no sampling head needed.
    """

    def __init__(self, hidden: int, H: int, D: int, ffn: int, L: int):
        super().__init__()
        self.layers = nn.ModuleList([
            RaggedTransformerBlock(hidden, H, D, ffn) for _ in range(L)
        ])
        # Minimal embedding table (vocab=32 000) just to produce valid fp16 inputs
        self.embed = nn.Embedding(32000, hidden)

    def forward(self, token_ids, cu_seqlens, branching_factor, depth):
        x = self.embed(token_ids).to(torch.float16)
        for layer in self.layers:
            x = layer(x, cu_seqlens, branching_factor, depth)
        return x

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

    # ── Build model (fp16, no grad) ──────────────────────────────────────────
    model = SyntheticRaggedModel(hidden, H, D, ffn, L).to(device).half().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Build ragged inputs ───────────────────────────────────────────────────
    torch.manual_seed(batch_size * 1000 + branching_factor * 100 + depth)
    token_ids = torch.randint(0, 32000, (total_tokens,), device=device)
    cu_sl = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.int32)

    # ── Time full forward ────────────────────────────────────────────────────
    fwd_ms = _time_fn(
        lambda: model(token_ids, cu_sl, branching_factor, depth),
        warmup, iters
    )

    # ── Time a single attention sublayer (proxy for attn fraction) ───────────
    attn_layer = model.layers[0].attn
    # Build layer input: embed + norm
    with torch.no_grad():
        x_sample = model.embed(token_ids).half()

    attn_only_ms = _time_fn(
        lambda: attn_layer(x_sample, cu_sl, branching_factor, depth),
        warmup, iters
    )

    tok_per_sec = total_tokens / (fwd_ms * 1e-3)
    # Approximate fraction: single-block attn * L layers (serial), vs total
    attn_frac = min(1.0, (attn_only_ms * L) / max(fwd_ms, 1e-9))

    return E2ERow(
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
        description="End-to-end tok/s benchmark for ragged (tree) speculative decoding"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = MODEL_PRESETS[args.model_size]
    print(
        f"\ne2e_benchmark  —  kernel overhead profiling  (our ragged kernel only)"
        f"\n  model={args.model_size}  ({cfg['L']} layers, hidden={cfg['hidden']}, H={cfg['H']}, D={cfg['D']})"
        f"\n  device={device}   dtype=fp16   random weights"
    )
    print()
    print("  This is NOT a comparison benchmark.  For kernel-vs-kernel speedups")
    print("  (vs SDPA bool-mask, FlashInfer, DeFT) see: benchmark_sota.py")
    print()
    print("  Metrics reported:")
    print("    fwd_ms       — full L-layer forward pass (embed+attn+FFN) × L")
    print("    attn_only_ms — single attention sublayer alone (proxy for per-layer cost)")
    print("    attn_frac    — estimated attention share of total compute")
    print("    tok/s        — verification tokens/sec on ragged path (synthetic)")
    print()

    total_runs = len(batch_sizes) * len(depths) * len(branching_factors)
    run_idx    = 0
    rows: list[dict] = []

    for B in batch_sizes:
        for d in depths:
            for b in branching_factors:
                run_idx += 1
                N = num_tree_nodes(b, d)
                print(
                    f"  [{run_idx:3d}/{total_runs}] "
                    f"B={B}  depth={d}  branching={b}  N={N}  total_tokens={B*N} … ",
                    end="", flush=True
                )
                try:
                    row = benchmark_e2e(
                        model_size       = args.model_size,
                        batch_size       = B,
                        branching_factor = b,
                        depth            = d,
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

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── tok/s summary (all batch sizes averaged) ─────────────────────────")
    # Group by (depth, branching_factor), average tok/s
    from collections import defaultdict
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["depth"], r["branching_factor"])].append(r["tok_per_sec"])

    header = f"{'depth':>6}  {'branching':>9}  {'mean_tok/s':>12}  {'min_tok/s':>12}"
    print(header)
    print("-" * len(header))
    for (d, b), vals in sorted(groups.items()):
        print(f"{d:>6}  {b:>9}  {sum(vals)/len(vals):>12.1f}  {min(vals):>12.1f}")

    print("\n── attn fraction summary ────────────────────────────────────────────")
    groups2: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups2[(r["depth"], r["branching_factor"])].append(r["attn_frac"])
    for (d, b), vals in sorted(groups2.items()):
        print(f"  depth={d}  b={b}  attn_frac={sum(vals)/len(vals):.1%}")

    print()
    print("─" * 70)
    print("  attn_frac interpretation (Amdahl):")
    print("    If attn_frac=F, a K× attention speedup gives at most")
    print("    1 / (1 - F×(1 - 1/K)) end-to-end speedup.")
    print("    Use benchmark_sota.py to see the K achieved by our kernel.")
    print()
    print("  tok/s and attn_frac use random weights and synthetic batch layout.")
    print("  They show scaling trends and kernel overhead share, not absolute")
    print("  production throughput.")
    print("─" * 70)


if __name__ == "__main__":
    main()
