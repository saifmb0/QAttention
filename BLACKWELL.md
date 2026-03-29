# sd-ragged — **blackwell** branch

Ancestor-sparse Flash Attention for speculative-decoding verification,
optimised for the **NVIDIA RTX 6000 ADA PRO 96 GB** (SM 8.9, Ada Lovelace).

---

## Hardware target

| Property | RTX 6000 ADA PRO | Tesla T4 (main branch) |
|---|---|---|
| Architecture | Ada Lovelace (AD102) | Turing |
| SM version | **8.9** | 7.5 |
| SMs | **142** | 40 |
| FP16 tensor-core peak | **364.2 TFLOPS** | 65 TFLOPS |
| FP8 tensor-core peak | **1 457 TFLOPS** | — |
| VRAM | **96 GB GDDR7** | 15 GB |
| HBM bandwidth | **820 GB/s** | 300 GB/s |
| L2 cache | **96 MB** | 3.8 MB |
| Roofline ridge $I^*$ | **444 FLOPs/byte** | 217 FLOPs/byte |

The 96 MB L2 is large enough to hold the entire K/V working set for
typical speculative-decoding batches, making the ancestor-sparse kernel
fully L2-resident past depth ≥ 2.

---

## What changed in this branch

### 1 — Triton autotune configs (`src/ragged_attn.py`)

| Param | SM 7.5 (T4) | SM 8.9 (Ada) |
|---|---|---|
| `BLOCK_M` candidates | 16 / 32 / 64 / 128 | 32 / 64 / **128 / 256** |
| `num_warps` | 2 / 4 / 8 | 4 / 8 / **16** |
| `num_stages` | 1 | **2** (step-0 prefetch) |

`_get_autotune_configs()` selects the correct set at runtime based on
`torch.cuda.get_device_properties().major/minor`, so a single checkout
runs correctly on both SM 7.5 and SM 8.9.

### 2 — BF16 support

`ragged_attention()` now accepts `torch.bfloat16` tensors.
On SM 8.9, BF16 is natively accelerated by Ada tensor cores.
On SM 7.5 (T4), BF16 inputs are silently cast to FP16 before launch
and cast back on return, preserving backward compat.

### 3 — SOTA benchmark (`scripts/benchmark_sota.py`)

Compares against every relevant production baseline:

| Method | Library | Mask | Notes |
|---|---|---|---|
| **Ragged fp16** (ours) | Triton | ancestor-sparse | primary |
| **Ragged bf16** (ours) | Triton | ancestor-sparse | Ada-native |
| PyTorch SDPA — math | `torch` | tree mask | vLLM production path |
| PyTorch SDPA — flash | `torch` | causal | upper-bound ref |
| PyTorch SDPA — mem-eff | `torch` | tree mask | |
| FlashAttention-2 | `flash_attn` | causal | upper-bound ref |
| FlashInfer batch-prefill | `flashinfer` | causal | upper-bound ref |
| xformers mem-eff | `xformers` | causal | upper-bound ref |

The three "upper-bound ref" methods use standard causal masking, **not**
the tree-ancestor mask, so their latency is a lower bound — even they
cannot beat our kernel at deeper/wider trees because they still iterate
over all KV positions.

---

## Repository structure

```
sd-ragged/
├── src/
│   ├── ragged_attn.py          Triton kernel — SM75 + SM89 autotune
│   ├── tree_mask.py            BFS tree construction
│   └── padding_waste.py        Analytic padding waste model
├── tests/
│   └── test_correctness.py     41 pytest cases vs SDPA reference
├── scripts/
│   ├── benchmark_sota.py       ← NEW  SOTA multi-method benchmark
│   ├── benchmark_sweep.py      Original ragged vs SDPA sweep
│   ├── padding_sweep.py        Padding waste characterisation
│   └── profile_kernel.py       Roofline profiler
├── results/                    CSVs + plots (git-tracked)
├── requirements_blackwell.txt  ← NEW  pinned deps for RTX 6000 ADA PRO
├── setup_blackwell.sh          ← NEW  fresh-instance bootstrap
├── run_blackwell.sh            ← NEW  single entry-point: test + bench
├── requirements.txt            Original (T4-compatible)
└── BLACKWELL.md                This file
```

---

## Quick start

```bash
# Clone and enter the blackwell branch
git clone <repo-url> sd-ragged
cd sd-ragged
git checkout blackwell

# Bootstrap a fresh instance (installs CUDA, Python deps, optional SOTA libs)
bash setup_blackwell.sh

# Run everything: smoke test → correct tests → SOTA benchmark → plots
bash run_blackwell.sh
```

Results are written to `results/`:

| File | Description |
|---|---|
| `sota_benchmark.csv` | Full latency / speedup table |
| `sota_latency_b*.png` | Latency vs depth per branching factor |
| `sota_speedup_heatmap_b*.png` | Speedup vs SDPA-math, B × depth |
| `sota_median_bar.png` | Median latency bar chart across all methods |
| `sota_tflops.png` | Effective TFLOPS comparison |

---

## Manual commands

```bash
# Correctness tests only
pytest tests/ -v

# SOTA benchmark (full sweep, ~15 min on RTX 6000 ADA PRO)
python scripts/benchmark_sota.py

# SOTA benchmark — fast pass (fewer configs)
python scripts/benchmark_sota.py \
    --batch-sizes 1,8,32 --depths 1,3,5 \
    --warmup 5 --iters 20

# Run with BF16 (Ada native)
python scripts/benchmark_sota.py --dtype bf16

# Skip optional SOTA libs (if not installed)
python scripts/benchmark_sota.py \
    --skip-flashattn --skip-flashinfer --skip-xformers

# Original T4-style sweep (still works on Ada)
python scripts/benchmark_sweep.py
```

---

## Installing optional SOTA libraries

The benchmark auto-detects which libraries are available and silently
skips any that are missing.  To install them on the RTX 6000 ADA PRO:

```bash
# FlashAttention-2 (build from source for SM 8.9 optimisations)
pip install flash-attn --no-build-isolation

# FlashInfer — pre-built wheel for CUDA 12.1 + torch 2.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# xformers — pre-built wheel
pip install xformers
```

---

## References

- EAGLE-2: Li et al. (2024). *EAGLE-2: Faster Inference of Language Models
  with Dynamic Draft Trees.* [arXiv:2406.16858](https://arxiv.org/abs/2406.16858)
- FlashAttention-2: Dao (2023). *FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- FlashInfer: Ye et al. (2024). *FlashInfer: Efficient and Customizable Attention
  Engine for LLM Inference Serving.* [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
- vLLM: Kwon et al. (2023). *Efficient Memory Management for Large Language Model
  Serving with PagedAttention.* [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
