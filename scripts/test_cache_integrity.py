import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.weight_stream import StreamingLinear

def test_cache_integrity():
    device = "cuda"
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    # KV-cache proxy: 10 MB (Total L2 = 40 MB)
    hot_size = 10 * 1024 * 1024 // 2
    hot_buf = torch.randn(hot_size, dtype=torch.float16, device=device)

    # Draft Weight proxy: 30 MB (will stream through L2)
    # M=60, K=4096, N=3840 -> ~230k elements per N? No.
    # 30 MB = 15.7M elements.
    M, K, N = 60, 4096, 3840
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    linear_ref = nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)

    # Evict First (Our technique)
    stream_first = StreamingLinear(linear_ref, eviction_policy="evict_first")
    # Evict Last (Normal Triton / naive approach)
    stream_last  = StreamingLinear(linear_ref, eviction_policy="evict_last")

    def measure_first_access():
        # DO NOT warm up. We want to see if it's still there from BEFORE the GEMM.
        start = time.perf_counter()
        _ = hot_buf.sum()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    print(f"GPU L2 Size: {l2_size / 1024**2:.1f} MB")
    print(f"Hot Buffer: {hot_buf.element_size() * hot_buf.numel() / 1024**2:.1f} MB")
    print(f"Weight Size: {linear_ref.weight.element_size() * linear_ref.weight.numel() / 1024**2:.1f} MB")

    # 1. Base latency (resident in L2)
    hot_buf.sum()
    torch.cuda.synchronize()
    base_lat = measure_first_access()
    print(f"\nBaseline Hot Latency (L2-resident): {base_lat:.3f} ms")

    # 2. Test Evict First
    hot_buf.sum()
    torch.cuda.synchronize()
    _ = stream_first(x)
    torch.cuda.synchronize()
    first_lat = measure_first_access()
    print(f"Latency after 'evict_first' GEMM:   {first_lat:.3f} ms")

    # 3. Test Evict Last
    hot_buf.sum()
    torch.cuda.synchronize()
    _ = stream_last(x)
    torch.cuda.synchronize()
    last_lat = measure_first_access()
    print(f"Latency after 'evict_last' GEMM:    {last_lat:.3f} ms")

    # 4. Test cuBLAS
    hot_buf.sum()
    torch.cuda.synchronize()
    _ = linear_ref(x)
    torch.cuda.synchronize()
    cublas_lat = measure_first_access()
    print(f"Latency after cuBLAS GEMM:         {cublas_lat:.3f} ms")

if __name__ == "__main__":
    test_cache_integrity()
