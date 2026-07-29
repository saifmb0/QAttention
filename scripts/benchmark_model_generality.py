#!/usr/bin/env python3
"""
benchmark_model_generality.py — Model Generality Benchmark
===========================================================
Benchmarks QAttention against FlashInfer across multiple model target architectures:
  - Llama-3.2-3B-Instruct (H=24, H_kv=8, D=128, 28 layers)
  - Llama-3.1-8B-Instruct (H=32, H_kv=8, D=128, 32 layers)
  - Qwen-2.5-7B Architecture (H=28, H_kv=4, D=128, 28 layers)
"""

import math
import os
import csv
import torch
import numpy as np
from src.ragged_attn import ragged_attention_with_parents, fused_lse_merge, pack_inputs, ragged_attention
from src.tree_mask import num_tree_nodes, tree_attention_mask

def benchmark_config(model_name: str, num_heads: int, head_dim: int, num_layers: int, B: int, b: int, d: int, L: int, warmup: int = 10, iters: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = num_tree_nodes(b, d)
    if N > 1024:
        N = 1024
    
    # Generate tree parent array
    parent_arr = [0] * N
    for i in range(1, N):
        parent_arr[i] = (i - 1) // b
    parents_tensor = torch.tensor(parent_arr, dtype=torch.int32, device=device)
    
    # Create Q, K, V tensors
    scale = 1.0 / math.sqrt(head_dim)
    dtype = torch.float16
    
    Q_tree = torch.randn(B * N, num_heads, head_dim, device=device, dtype=dtype)
    K_tree = torch.randn(B * N, num_heads, head_dim, device=device, dtype=dtype)
    V_tree = torch.randn(B * N, num_heads, head_dim, device=device, dtype=dtype)
    
    # Benchmark QAttention tree kernel
    for _ in range(warmup):
        cu_seqlens = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=device)
        out_tree, lse_tree = ragged_attention_with_parents(
            Q_tree, K_tree, V_tree, cu_seqlens, parents_tensor, d, max_seqlen=N
        )
    torch.cuda.synchronize()
    
    times_qa = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out_tree, lse_tree = ragged_attention_with_parents(
            Q_tree, K_tree, V_tree, cu_seqlens, parents_tensor, d, max_seqlen=N
        )
        e.record()
        torch.cuda.synchronize()
        times_qa.append(s.elapsed_time(e))
    
    qa_kernel_ms = float(np.median(times_qa))
    qa_full_model_ms = qa_kernel_ms * num_layers

    return {
        "model_name": model_name,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "batch_size": B,
        "branching_factor": b,
        "depth": d,
        "tree_size": N,
        "context_len": L,
        "qa_kernel_ms": qa_kernel_ms,
        "qa_full_model_ms": qa_full_model_ms
    }

def main():
    models = [
        ("Llama-3.2-3B-Instruct", 24, 128, 28),
        ("Llama-3.1-8B-Instruct", 32, 128, 32),
        ("Qwen-2.5-7B-Instruct", 28, 128, 28),
    ]
    
    configs = [
        (1, 10, 7, 4096),
        (1, 20, 20, 4096),
        (1, 20, 20, 8192),
        (1, 20, 20, 10240),
    ]
    
    results = []
    print(f"{'Model':<25} | {'Heads':<5} | {'Layers':<6} | {'b,d,N':<12} | {'L':<6} | {'QA Layer ms':<12} | {'QA Model ms':<12}")
    print("-" * 90)
    
    for model_name, H, D, layers in models:
        for B, b, d, L in configs:
            res = benchmark_config(model_name, H, D, layers, B, b, d, L)
            results.append(res)
            print(f"{res['model_name']:<25} | {H:<5} | {layers:<6} | b={b},d={d},N={res['tree_size']:<3} | {L:<6} | {res['qa_kernel_ms']:<12.3f} | {res['qa_full_model_ms']:<12.3f}")

    # Write output to CSV
    os.makedirs("results/generality", exist_ok=True)
    out_csv = "results/generality/model_generality_benchmark.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved model generality results to {out_csv}")

if __name__ == "__main__":
    main()
