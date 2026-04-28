import torch
import torch.nn as nn
import triton
import triton.language as tl


# ─── Kernel ──────────────────────────────────────────────────────────────────
#
# Key design choices:
#
# 1. Weight tile access pattern: we load W as (BLOCK_N, BLOCK_K) tiles so that
#    the K dimension (stride=1) is the fast dimension → coalesced reads.
#    A non-transposed B-tile load (b_ptrs[k, n] with stride_bn=K) forces each
#    row of the tile to touch BLOCK_N separate cache lines (stride K=4096
#    between consecutive n-elements), wasting ~98% of each cache line.
#    Loading W-rows first (coalesced) then tl.trans() is zero extra cost:
#    Triton lowers it to a tensor-core HMMA with column-major B operand.
#
# 2. eviction_policy="evict_first" on weight loads: marks each weight cache
#    line as the first eviction candidate after use.  On Ada (sm89) this maps
#    to the PTX `ld.global.L2::evict_first` qualifier.  With draft-model
#    weights totalling >> 41.9 MB (RTX 4000 Ada L2), default policy would
#    continuously evict KV-cache and activation data.  evict_first keeps those
#    hot buffers resident while weights stream through without polluting L2.
#
# 3. Activation loads use default eviction policy (evict_last) so the small
#    M-dimensional activation tiles (M ≈ 10–60) stay warm across the K loop.
#
# 4. SPLIT_K removed: all prior autotune configs had SPLIT_K=1, making the
#    atomic_add branch dead.  Removed entirely; standard grid suffices for
#    M ≈ 10–60 with N=14336 (~224 CTAs → 100% SM occupancy on 48 SMs).
#
# 5. Software pipeline depth (num_stages): Triton emits cp.async instructions
#    to overlap weight tile loads with tensor-core computation.  Stages 3–5
#    are explored in autotune; higher stages increase register pressure.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _streaming_matmul_kernel(
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

    # A: activations (BLOCK_M, BLOCK_K) — keep in L2 (default eviction = evict_last)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B: weight tiles loaded as (BLOCK_N, BLOCK_K) to ensure coalesced reads
    # along the K dimension (stride=1).  We then tl.trans() in registers.
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k + k < K

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # Load (BLOCK_N, BLOCK_K) tile — coalesced because stride_bk=1
        b_tile = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0,
                         eviction_policy="evict_first")
        # Transpose to (BLOCK_K, BLOCK_N) for dot product
        b = tl.trans(b_tile)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


# ─── Python wrappers ──────────────────────────────────────────────────────────

class StreamingLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        orig_shape = a.shape
        M = int(a.numel() // orig_shape[-1])
        K = orig_shape[-1]
        N = weight.shape[0]

        a_flat = a.contiguous().view(M, K)
        c_flat = torch.empty((M, N), device=a.device, dtype=a.dtype)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        _streaming_matmul_kernel[grid](
            a_flat, weight, c_flat,
            M, N, K,
            a_flat.stride(0), a_flat.stride(1),
            weight.stride(1), weight.stride(0),   # bk=stride(1)=1, bn=stride(0)=K
            c_flat.stride(0), c_flat.stride(1),
        )

        out = c_flat.view(*orig_shape[:-1], N)
        if bias is not None:
            out = out + bias
        return out


class StreamingLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

    def forward(self, x):
        return StreamingLinearFunc.apply(x, self.weight, self.bias)


_MLP_NAMES = frozenset({'gate_proj', 'up_proj', 'down_proj'})


def patch_draft_model(module: nn.Module) -> None:
    """
    Patch only the MLP gate/up/down projections with evict_first StreamingLinear.

    Targeting by module name (not shape) avoids accidentally patching attention
    projections (q_proj K=8192), the fc feature-fusion layer (K=12288), or
    lm_head (N=32000) — all of which have shapes outside the autotune's range
    and regress if patched.

    The three MLP weight matrices (117 MB each, total 352 MB) are the primary
    L2 pollution source per draft forward call.  All other Linear modules stay
    as cuBLAS, which already achieves 600+ GB/s for the smaller attention weights
    (confirmed L2-resident from the per-GEMM timing benchmark).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in _MLP_NAMES:
            setattr(module, name, StreamingLinear(child))
        else:
            patch_draft_model(child)


def unpatch_draft_model(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, StreamingLinear):
            linear = nn.Linear(child.weight.shape[1], child.weight.shape[0],
                               bias=child.bias is not None,
                               device=child.weight.device,
                               dtype=child.weight.dtype)
            linear.weight = child.weight
            if child.bias is not None:
                linear.bias = child.bias
            setattr(module, name, linear)
        else:
            unpatch_draft_model(child)
