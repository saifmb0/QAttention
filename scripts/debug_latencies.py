import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.weight_stream import StreamingLinear

def flush_l2():
    # 40MB L2 -> 160MB buffer
    _ = torch.empty(160 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda").fill_(0)
    torch.cuda.synchronize()

def debug():
    device = "cuda:0"
    M, K, N = 60, 4096, 14336
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    linear = nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
    stream = StreamingLinear(linear)

    print(f"Shape: M={M}, K={K}, N={N}")
    
    # 1. Warmup
    for _ in range(10):
        _ = linear(a)
        _ = stream(a)
    torch.cuda.synchronize()

    # 2. cuBLAS (Cold)
    flush_l2()
    start = time.perf_counter()
    _ = linear(a)
    torch.cuda.synchronize()
    print(f"cuBLAS (Cold): {(time.perf_counter() - start)*1000:.3f} ms")

    # 3. cuBLAS (Warm)
    start = time.perf_counter()
    for _ in range(100):
        _ = linear(a)
    torch.cuda.synchronize()
    print(f"cuBLAS (Warm): {(time.perf_counter() - start)*10:.3f} ms")

    # 4. Stream (Cold)
    flush_l2()
    start = time.perf_counter()
    _ = stream(a)
    torch.cuda.synchronize()
    print(f"Stream (Cold): {(time.perf_counter() - start)*1000:.3f} ms")

    # 5. Stream (Warm)
    start = time.perf_counter()
    for _ in range(100):
        _ = stream(a)
    torch.cuda.synchronize()
    print(f"Stream (Warm): {(time.perf_counter() - start)*10:.3f} ms")

if __name__ == "__main__":
    debug()
