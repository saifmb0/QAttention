import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
@triton.jit
def _raw_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k + k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0,
                         eviction_policy="evict_first")
        b = tl.trans(b_tile)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

def benchmark_configs():
    device = "cuda:0"
    M, K, N = 60, 4096, 14336
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    w = torch.randn((N, K), device=device, dtype=torch.float16)
    c = torch.empty((M, N), device=device, dtype=torch.float16)

    # Reference cuBLAS
    linear = nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
    linear.weight.data = w
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = linear(a)
    torch.cuda.synchronize()
    cublas_ms = (time.perf_counter() - start) * 10
    print(f"cuBLAS Latency: {cublas_ms:.3f} ms")

    configs = [
        # Original configs
        {'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4},
        {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4},
        {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4},
        # Aggressive configs
        {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4},
        {'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 5, 'num_warps': 4},
        {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8},
        {'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4},
        # Wide N
        {'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8},
    ]

    best_ms = float('inf')
    best_config = None

    for conf in configs:
        # Clear cache
        _ = torch.empty(1024*1024*40, device=device).fill_(0) 
        
        grid = (
            triton.cdiv(M, conf['BLOCK_M']),
            triton.cdiv(N, conf['BLOCK_N']),
        )

        # Warmup
        _raw_matmul_kernel[grid](
            a, w, c,
            M, N, K,
            a.stride(0), a.stride(1),
            w.stride(1), w.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=conf['BLOCK_M'], BLOCK_N=conf['BLOCK_N'], BLOCK_K=conf['BLOCK_K'],
            num_stages=conf['num_stages'], num_warps=conf['num_warps']
        )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _raw_matmul_kernel[grid](
                a, w, c,
                M, N, K,
                a.stride(0), a.stride(1),
                w.stride(1), w.stride(0),
                c.stride(0), c.stride(1),
                BLOCK_M=conf['BLOCK_M'], BLOCK_N=conf['BLOCK_N'], BLOCK_K=conf['BLOCK_K'],
                num_stages=conf['num_stages'], num_warps=conf['num_warps']
            )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 10
        
        print(f"Config {conf}: {ms:.3f} ms")
        if ms < best_ms:
            best_ms = ms
            best_config = conf

    print(f"\nBest Config: {best_config} at {best_ms:.3f} ms")
    print(f"Gap to cuBLAS: {best_ms - cublas_ms:+.3f} ms")

if __name__ == "__main__":
    benchmark_configs()
