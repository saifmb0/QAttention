import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.eagle_patches import ConcurrentGLUMLP

def ablate():
    device = "cuda:0"
    M, K, N = 60, 4096, 14336
    x = torch.randn(M, K, device=device, dtype=torch.float16)
    
    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
            self.up_proj   = nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
            self.down_proj = nn.Linear(N, K, bias=False, device=device, dtype=torch.float16)
            self.act_fn    = nn.SiLU()
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp_seq = MockMLP()
    mlp_con = ConcurrentGLUMLP(mlp_seq, use_streaming=False)

    # Warmup
    for _ in range(100):
        _ = mlp_seq(x)
        _ = mlp_con(x)
    torch.cuda.synchronize()

    iters = 1000
    
    # 1. Sequential
    start = time.perf_counter()
    for _ in range(iters):
        _ = mlp_seq(x)
    torch.cuda.synchronize()
    t_seq = (time.perf_counter() - start) * 1000 / iters

    # 2. Concurrent
    start = time.perf_counter()
    for _ in range(iters):
        _ = mlp_con(x)
    torch.cuda.synchronize()
    t_con = (time.perf_counter() - start) * 1000 / iters

    print(f"Sequential MLP: {t_seq:.3f} ms")
    print(f"Concurrent MLP: {t_con:.3f} ms")
    print(f"Speedup:        {(t_seq/t_con - 1)*100:+.1f}%")

if __name__ == "__main__":
    ablate()
