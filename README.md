# QAttention: Ancestor-Sparse Flash Attention for Speculative Decoding

This repository contains the code and data for the paper **"Breaking the Quadratic Wall in Tree-Based Speculative Decoding"** (under review).

It provides an `O(N · d)` ancestor-walk Triton kernel for tree-structured attention that avoids the `O(N²)` inner loop used by production baseline kernels (FlashInfer, DeFT).

---

## What does this kernel do?

In tree-based speculative decoding (e.g., EAGLE, Sequoia, Medusa), a draft model generates a tree of candidate tokens. The target model verifies this tree in a single forward pass. 

Because it's a tree, token `i` only attends to its ancestor chain up to the root. In a $b$-ary tree of depth $d$, the total number of tokens $N$ grows exponentially, but the ancestor chain is only $d+1$ tokens long.

**Production baselines** (FlashInfer, DeFT) process this by having every query iterate over all $N$ keys/values and masking out the non-ancestors. This results in $O(N^2)$ memory traffic and FLOPs.

**QAttention** explicitly walks the $d+1$ ancestor chain for each query. It executes a scattered gather inside a standard FlashAttention-2 online-softmax loop, reducing work to exactly $O(N \cdot d)$.

---

## When does this matter?

At the kernel level, the asymptotic advantage is massive. The kernel reaches **7.75× speedup over FlashInfer** and **16.99× over DeFT** at deep/wide tree configurations, while maintaining strict parity at the small trees used by current systems. 

However, end-to-end (E2E) throughput depends on Amdahl's Law. This kernel improves E2E throughput significantly when attention becomes a bottleneck:

1. **Long Context ($L \gg 1000$)**: As the prefix KV-cache grows, FlashInfer's $O(N^2)$ verification of draft tokens accumulates massive amounts of unused work. QAttention's verify time remains flat, yielding up to **1.61× E2E speedup** on EAGLE-3 at 10k tokens.
2. **Deep Trees ($N \ge 512$)**: When testing explicitly scalable trees like Sequoia, the $O(N^2)$ penalty throttles throughput. QAttention delivers **1.40× E2E speedup** on Sequoia at $N=1024$.
3. **Decoupling Tree Search**: Currently, tree-search policies are restricted to shallow trees to avoid stalling the verifier. This kernel removes the systems barrier, allowing tree policies to scale strictly on their statistical acceptance-rate merits.

*(Note: At short context sizes, attention is a very small fraction of the step time. Speeding it up yields an honest, Amdahl-limited ~1.04× E2E gain).*

---

## Repository Structure

* `src/`: Triton kernel implementation (`ragged_attn.py`) and tree masking logic.
* `tests/`: Extensive correctness suite verifying fp16 online-softmax parity against a PyTorch padded SDPA reference.
* `scripts/`: Benchmarking scripts used to generate the paper's figures.
* `results/`: The raw CSV data and statistics used in the paper.
* `neurips-paper/`: The LaTeX source code and figures for the manuscript.

---

## Usage

The API is a direct drop-in for standard variable-length sequence packing:

```python
import torch
from src.ragged_attn import pack_inputs, ragged_attention

# qs, ks, vs: lists of B tensors each shaped [L_i, H, D] fp16 on CUDA
Q, K, V, cu_seqlens = pack_inputs(qs, ks, vs)

# Run ancestor-sparse attention
O = ragged_attention(
    Q, K, V,
    cu_seqlens,          # [B+1] int32 CPU tensor
    branching_factor=4,  # b
    max_depth=5,         # d
)
# O: [Σ L_i, H, D] fp16 on CUDA
```

### Installation & Tests

Tested on Python 3.12, PyTorch $\ge$ 2.1, Triton $\ge$ 2.1. Requires an NVIDIA GPU.

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run correctness suite (72 configurations)
pytest tests/test_correctness.py -v
```
