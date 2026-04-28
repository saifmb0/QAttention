import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.weight_stream import StreamingLinear
from src.ragged_attn import ragged_attention, pack_inputs

def validate_hypotheses():
    device = "cuda:0"
    torch.cuda.set_device(0)
    
    # ─── Configuration ────────────────────────────────────────────────────────
    B = 1
    H, D = 8, 128
    b, d = 4, 16
    N_tree = 103 
    
    # Draft Weight proxy: 112 MB (Actual Llama-3 gate_proj)
    M_mlp, K_mlp, N_mlp = 60, 4096, 14336 
    x_mlp = torch.randn(M_mlp, K_mlp, device=device, dtype=torch.float16)
    linear_ref = nn.Linear(K_mlp, N_mlp, bias=False, device=device, dtype=torch.float16)
    stream_first = StreamingLinear(linear_ref)

    # GLOBAL WARMUP to eliminate autotune/compilation from timing
    print("Warming up kernels...")
    _ = stream_first(x_mlp)
    _ = linear_ref(x_mlp)
    torch.cuda.synchronize()

    L_sweep = [4096, 16384, 65536]
    
    print(f"{'L':>8} | {'Kernel':<10} | {'MLP (ms)':>10} | {'Attn (ms)':>10} | {'Dense (ms)':>10} | {'Gaps (S/D)'}")
    print("-" * 75)

    for L in L_sweep:
        try:
            # Setup KV-cache for this L
            qs = [torch.randn(N_tree, H, D, device=device, dtype=torch.float16) for _ in range(B)]
            ks = [torch.randn(L, H, D, device=device, dtype=torch.float16) for _ in range(B)]
            vs = [torch.randn(L, H, D, device=device, dtype=torch.float16) for _ in range(B)]
            Q, K, V, cu_sl = pack_inputs(qs, ks, vs)
            
            # Warmup
            for _ in range(5):
                _ = linear_ref(x_mlp)
                _ = ragged_attention(Q, K, V, cu_sl, b, d)
            torch.cuda.synchronize()

            def measure(mlp_module):
                # 1. Force KV into L2
                _ = K.sum() + V.sum()
                torch.cuda.synchronize()
                
                # 2. Time MLP
                s1 = time.perf_counter()
                _ = mlp_module(x_mlp)
                torch.cuda.synchronize()
                t_mlp = (time.perf_counter() - s1) * 1000
                
                # 3. Time Attention (Sparse)
                s2 = time.perf_counter()
                _ = ragged_attention(Q, K, V, cu_sl, b, d)
                torch.cuda.synchronize()
                t_attn = (time.perf_counter() - s2) * 1000

                # 4. Time Attention (Dense - simulated by full sum)
                # This touches ALL L elements of K and V
                s3 = time.perf_counter()
                _ = K.sum() + V.sum()
                torch.cuda.synchronize()
                t_dense = (time.perf_counter() - s3) * 1000
                
                return t_mlp, t_attn, t_dense

            n_iters = 50
            m_c, a_c, d_c = 0, 0, 0
            for _ in range(n_iters):
                m, a, d_at = measure(linear_ref)
                m_c += m; a_c += a; d_c += d_at
            m_c /= n_iters; a_c /= n_iters; d_c /= n_iters

            m_s, a_s, d_s = 0, 0, 0
            for _ in range(n_iters):
                m, a, d_at = measure(stream_first)
                m_s += m; a_s += a; d_s += d_at
            m_s /= n_iters; a_s /= n_iters; d_s /= n_iters

            gap = a_c - a_s
            dense_gap = d_c - d_s
            print(f"{L:>8} | {'cuBLAS':<10} | {m_c:>10.3f} | {a_c:>10.3f} | {d_c:>10.3f} |")
            print(f"{L:>8} | {'Stream':<10} | {m_s:>10.3f} | {a_s:>10.3f} | {d_s:>10.3f} | {gap:+.3f} ms / {dense_gap:+.3f} ms")
            print("-" * 75)
            
            # Cleanup for next L
            del qs, ks, vs, Q, K, V, cu_sl
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{L:>8} | OOM")
                break
            else: raise e

if __name__ == "__main__":
    validate_hypotheses()
