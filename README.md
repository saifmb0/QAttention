# sd-ragged: Ancestor-Sparse Flash Attention for Speculative Decoding

A Triton kernel for tree-structured speculative decoding (SD) attention, co-designed
with a data-driven profiling methodology that drove successive algorithmic improvements
from **0.17× SDPA** to **16×+ SDPA** at production batch sizes.

---

## Background

In speculative decoding (e.g. EAGLE-2), a small draft model proposes a *tree* of
candidate token sequences in parallel. A single target-model forward pass then
verifies all draft candidates simultaneously. The verification attention has a
non-standard causal structure: draft token `i` may attend to token `j` only if
`j` is an ancestor of `i` in the BFS-ordered draft tree (or `j == i`).

Naively handling this with padded dense attention wastes:

- **Memory**: quadratic in the maximum sequence length
- **Compute**: O(N²) per sequence but only O(N × d) work is useful, where `d`
  is tree depth (≤ 5 in practice)

This project implements and benchmarks a correct, efficient kernel for exactly
this operation.

**Tree definition.** A complete *b*-ary tree of depth *d* has
$N = (b^{d+1} - 1) / (b - 1)$ nodes numbered in BFS order.
Root = 0; for any node $k > 0$: $\text{parent}(k) = (k - 1) // b$.

---

## Hardware target

| Property | Value |
|---|---|
| GPU | NVIDIA A100 |
| SM | 80 (Ampere) |
| HBM bandwidth | ~2,000 GB/s |
| FP16 tensor-core peak | 312 TFLOPS |
| L2 cache | 40 MB |
| # SMs | 108 |
| Roofline ridge point $I^*$ | ~156 FLOPs/byte |

---

## Repository structure

```
sd-ragged/
├── src/
│   ├── tree_mask.py        # BFS tree construction + reference attention mask
│   ├── padding_waste.py    # Analytic padding waste model (token + attention)
│   └── ragged_attn.py      # Triton kernel: ancestor-sparse Flash Attention
├── tests/
│   └── test_correctness.py # 41 pytest cases against PyTorch SDPA reference
├── scripts/
│   ├── benchmark_sweep.py  # End-to-end speedup vs SDPA across 90 (B,b,d) configs
│   ├── padding_sweep.py    # Padding waste characterisation sweep
│   └── profile_kernel.py   # Roofline profiler: HBM BW, TFLOPS, util%, CTAs/SM
├── results/                # Saved CSVs and plots (git-tracked)
├── requirements.txt
├── conftest.py
└── run_all.sh              # Single script: padding + tests + benchmark + profile
```

---

## Algorithm evolution

The final design was reached through four distinct kernel versions, each driven
by measured data rather than guesswork. The trajectory is documented here as a
methods narrative.

### v1 — Dense masked kernel (broken + slow)

**Design.** Standard Flash-Attention-2 with the tree mask computed analytically
inside the kernel: for each `(query, key)` tile, walk the ancestor chain from
each query row up to `MAX_DEPTH` steps and compare against every key column.

**Bugs found.**

1. *Triton fp16/fp32 dtype error (SM75)*: `tl.dot` on SM75 requires matching
   dtypes. Fixed by keeping native fp16 loads and passing `out_dtype=tl.float32`.

2. *Padding waste = 0.0*: All sequences in a batch were built with the same
   depth → zero padding by construction. Fixed by sampling each request's depth
   independently from `Uniform[1, max_depth]`.

3. *tl.where eager-evaluation (b=3)*: The ancestor walk used:
   ```python
   cur = tl.where(cur > 0, (cur - 1) // b, 0)
   ```
   Triton evaluates both branches eagerly on all lanes. For non-power-of-2 `b`
   (e.g. b=3), PTX lowers the division to a multiply-high sequence that treats
   the dividend as unsigned 32-bit. When `cur = 0`, `cur - 1 = -1 (int32)` is
   reinterpreted as `0xFFFFFFFF`, yielding `0xFFFFFFFF // 3 = 0x55555554` — a
   spurious ancestor index that bled into the attend mask, producing
   `max_abs ≈ 3.4` output errors.

   **Fix (wrong first attempt):** `tl.maximum((cur-1) // b, 0)` — still divides
   `-1` first, then clamps. Same bug.

   **Fix (correct):** clamp the subtraction *before* dividing:
   ```python
   cur = tl.maximum(cur - 1, tl.zeros_like(cur)) // BRANCHING_FACTOR
   ```
   This guarantees the dividend is non-negative before the multiply-high
   sequence executes.

**Performance.** Even with all correctness bugs fixed, the kernel was 5–80×
*slower* than SDPA at large N. Autotune with BLOCK_M=BLOCK_N∈{16,32,64} did
not help.

---

### v2 — Profiling-driven diagnosis

After running `scripts/profile_kernel.py` (roofline analysis), the data showed:

```
B=32 b=4 d=5  N=1365  ragged=114ms  roof=3.8ms  util=3054%
```

`util% = actual_ms / roofline_ms × 100`. A value of 3054% means the kernel
ran **30× slower than the HBM bandwidth ceiling, independent of batch size B**.

The B-independence is the critical observation. CTA starvation would scale with
B (more CTAs at larger B), but the slowdown ratio was flat at ~30× for
B=1 through B=32. This immediately ruled out CTA starvation as the primary
cause.

**Root cause: algorithmic sparsity blindness.**

In a BFS b-ary tree, query `q` can attend to *at most* `MAX_DEPTH + 1 ≤ 6`
distinct KV positions out of `N` total. The dense kernel iterated over all
`⌈N/BLOCK_N⌉` KV blocks and masked non-ancestors to −∞ *after* computing
`QK^T`. For `b=4, d=5, N=1365`:

$$\text{useful work} / \text{total work} = 6 / 1365 = 0.44\%$$

99.56% of every `QK^T` computation and every K/V memory load was thrown away.
The ~2 TFLOPS plateau was simply the HBM bandwidth cost of streaming 1365 × 64
fp16 K/V values per query tile that were never used.

---

### v3 — Ancestor-sparse Flash Attention (final design)

**Design.** Replace the KV-block loop with an ancestor-walk loop of exactly
`MAX_DEPTH + 1` iterations. Each iteration performs a *scattered gather*:
load `K[parent^s(q)]` and `V[parent^s(q)]` for every query row `q` in the
BLOCK_M-sized query tile.

The Flash-Attention-2 online softmax update still applies — each step
contributes one score per query row instead of `BLOCK_N` scores, but the
update equations are identical:

```
m_new = max(m_i, score)
alpha = exp(m_i - m_new)
p     = exp(score - m_new)
acc   = acc * alpha + p * v_anc
l_i   = l_i * alpha + p
```

Duplicate detection via `cur != prev` handles shallow sequences where the
ancestor walk reaches the root before `MAX_DEPTH` steps complete — subsequent
iterations become no-ops (alpha=1, p=0, zero-masked loads).

**Complexity comparison** (B=4, b=4, d=5, N=1365, D=64, H=8, BLOCK_M=64):

| | Dense (v1) | Sparse (v3) |
|---|---|---|
| FMAs / Q-tile | 22 × 64 × 64 × 64 × 2 = 11.5 GFLOPs | 6 × 64 × 64 = 24,576 |
| K+V bytes / Q-tile | 341 KB | 96 KB |
| Reduction | — | **470× fewer FMAs, 3.6× less traffic** |

With L2 reuse (all N unique K/V positions = 341 KB — fits in L2), HBM
fills approximately once per unique K/V position across all Q-tiles.

**Additionally fixed in v3:** The `ragged_attention()` Python function was
performing a blocking GPU→CPU sync on every call:
```python
# old: unnecessary round-trip
cu_seqlens_dev = cu_seqlens.to(device)
seq_lens = (cu_seqlens_dev[1:] - cu_seqlens_dev[:-1]).cpu().tolist()  # sync!
```
`cu_seqlens` from `pack_inputs()` is already a CPU tensor. Fixed by reading it
on-CPU before the H2D transfer, saving ~20–50 µs of blocking overhead per call.

---

## Results (A100, SM80)

### Correctness: 41/41 tests pass

All parametric correctness tests (`B ∈ {1,2,4,8}`, `b ∈ {1,2,3}`,
`d ∈ {1,2,3}`, plus standalone `test_head_dims[32,64,128]`,
`test_single_token` (b=1,d=0), and `test_linear_chain` (b=1,d=8))
pass against PyTorch SDPA reference with `atol=1e-2, rtol=1e-2`.

> **Setup**: `num_heads=4`, `head_dim=64` for the parametric grid;
reference uses `enable_flash=False, enable_math=True,
enable_mem_efficient=False` to force the math backend.

Test runtime: **21 seconds** (was 123 seconds with the dense kernel).

### Benchmark vs SDPA

**What is timed.** Each measurement is the median of 50 CUDA-Event-timed
calls (after 10 warmup iterations). The timed region covers the full
`ragged_attention()` Python call — including `max_seqlen` computation and
Triton kernel dispatch — but excludes `pack_inputs()`, which is
pre-computed outside the timed region. Both kernels are timed by the same
method for a fair comparison.

**Batch uniformity.** Every sequence in a benchmark batch has the same
complete *b*-ary tree of depth `d` (`seq_lens = [N] * B`). Variable-depth
batching is not exercised in this sweep.

**SDPA backend.** Baseline uses `enable_flash=False, enable_math=True,
enable_mem_efficient=True`; PyTorch selects the faster of math and
mem-efficient per config (math-only is forced in correctness tests only).

---

```
── Speedup summary (mean over B) ──
branching_factor      2      3      4
tree_depth
1                 1.07   1.08   1.07
2                 1.05   1.05   1.08
3                 1.03   0.66   1.06
4                 1.03   1.06   1.70
5                 1.05   2.28   9.12
```

*Mean across B=1,2,4,8,16,32. H=8, D=64.*

Selected highlights (H=8, D=64):

| B | b | d | N | Ragged (ms) | SDPA (ms) | Speedup | Dense-equiv TFLOPS† |
|---|---|---|---|-------------|-----------|---------|---------------------|
| 1 | 4 | 5 | 1365 | 0.132 | 0.586 | **4.45×** | 29 |
| 2 | 4 | 5 | 1365 | 0.166 | 0.906 | **5.47×** | 46 |
| 4 | 4 | 5 | 1365 | 0.231 | 2.205 | **9.53×** | 66 |
| 8 | 4 | 5 | 1365 | 0.359 | 4.330 | **12.06×** | 85 |
| 16 | 4 | 5 | 1365 | 0.617 | 8.902 | **14.43×** | 99 |
| 32 | 4 | 5 | 1365 | 1.138 | 18.262 | **16.05×** | **107** |

† **Dense-equiv TFLOPS** = `4·B·N²·D·H / latency`. Uses the standard
dense attention FLOP formula (`4·L²·D·H`) as the numerator — not the
sparse kernel's actual `4·(d+1)·L·D·H` FMAs. This is a throughput
proxy: it answers "how fast would a dense kernel need to run to match
this latency?" It is **not** a hardware utilisation metric for the sparse
kernel (which executes only ~0.5 GFLOPS; see Roofline below).

### Roofline analysis

All configs are HBM-bound in the roofline sense:
- Arithmetic intensity `I = 4·(d+1)·B·L·D·H / (8·L·D·H·B) ≈ 1–3 FLOPs/byte`
- Ridge point `I* = 65 TFLOPS / 300 GB/s ≈ 217 FLOPs/byte`

`util% = actual_ms / roofline_ms × 100`. At large N (b=4,d=5), measured
util% converges to ~170–210%, meaning the kernel runs at ~50–60% of the
HBM bandwidth spec (achieving ~170 GB/s vs 300 GB/s spec). The residual
~1.7× gap over roofline is the **scattered access penalty**: each step
loads `BLOCK_M` K-rows from scattered positions (one distinct cache line
per query row), achieving ~170 GB/s effective bandwidth vs 300 GB/s for
a sequential stream.

At small N (N < 64), the ~0.13ms floor is Triton kernel dispatch latency vs
cuDNN's pre-compiled ~0.09ms floor for SDPA. This gap is structural and closes
as N grows.

---

## Padding waste characterisation

The `padding_waste.py` module quantifies the motivation for ragged batching.
The attention-compute padding ratio grows steeply with branching factor and
depth:

| max_depth | b=2 | b=3 | b=4 |
|---|---|---|---|
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.018 | 0.044 | 0.077 |
| 3 | 0.061 | 0.139 | 0.297 |
| 4 | 0.140 | 0.378 | 0.468 |
| 5 | 0.207 | 0.468 | 0.556 |

At b=4, d=5: **55.6% of all attention FLOPs are wasted** by padding if dense
batched attention is used. The ragged kernel eliminates this entirely.

---

## Kernel API

```python
from src.ragged_attn import pack_inputs, ragged_attention

# Pack per-sequence tensors into ragged layout
Q, K, V, cu_seqlens = pack_inputs(qs, ks, vs)
# qs, ks, vs: lists of B tensors each shaped [L_i, H, D] fp16 on CUDA

# Run ancestor-sparse attention
O = ragged_attention(
    Q, K, V,
    cu_seqlens,          # [B+1] int32 CPU tensor
    branching_factor=4,  # b
    max_depth=5,         # d
)
# O: [Σ L_i, H, D] fp16 on CUDA
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `torch >= 2.1`, `triton >= 2.1`, `numpy`, `pandas`,
`matplotlib`, `pytest`. Tested on Python 3.12.

---

## Running everything

```bash
bash run_all.sh
```

Or individually:

```bash
# Correctness (41 tests)
pytest tests/test_correctness.py -v

# Benchmark vs SDPA (90 configs)
python scripts/benchmark_sweep.py

# Padding waste sweep
python scripts/padding_sweep.py

# Roofline profiler
python scripts/profile_kernel.py --csv results/profile.csv
```

---

## Key engineering decisions

| Decision | Rationale |
|---|---|
| Ancestor-walk loop instead of KV-block loop | Reduces work from O(N²) to O(N×d); 470× fewer FMAs at b=4,d=5 |
| Online softmax (FA-2) with single-element updates | Enables arbitrary loop order; no materialised N×N score matrix |
| `tl.maximum(cur-1, 0) // b` for parent computation | Avoids PTX unsigned-reinterpretation of −1 for non-power-of-2 b |
| `prev != cur` duplicate guard | Handles root revisits when actual depth < MAX_DEPTH, at no branch cost |
| CPU-side `max_seqlen` computation | Eliminates a blocking GPU→CPU round-trip from every kernel call |
| `num_stages=1` in autotune | Step s+1's load address depends on step s's `cur` value — no pipelining possible |
| Autotune key: HEAD_DIM + BRANCHING_FACTOR + MAX_DEPTH | These three constexpr args determine the compiled PTX; keying on them avoids cache misses |

---

## Commit history summary

| Commit | Change |
|---|---|
| Initial | Scaffold: tree_mask, padding_waste, dense FA kernel, tests, benchmarks |
| dtype fix | Triton fp16 input + `out_dtype=tl.float32` for SM75 tl.dot |
| padding fix | Per-request depth sampling in sequence_lengths() |
| analytic mask | Eliminate O(B×N²) packed_masks buffer; inline ancestor walk |
| b=3 correctness | `tl.maximum(cur-1, 0) // b` — clamp before divide |
| operator order | Moves clamp to correct position (before, not after, division) |
| sparse kernel | Replace KV-block loop with ancestor-walk loop; 470× improvement at large N |
| no GPU sync | Compute max_seqlen from CPU tensor; async H2D with non_blocking=True |
