import torch
import triton
import sys
import os

from src.ragged_attn import _ragged_attn_sparse_kernel

# Set up types
signature = {
    'Q_ptr': '*fp16', 'K_ptr': '*fp16', 'V_ptr': '*fp16', 'O_ptr': '*fp16',
    'cu_seqlens_ptr': '*i32',
    'stride_qt': 'i32', 'stride_qh': 'i32', 'stride_qd': 'i32',
    'stride_kt': 'i32', 'stride_kh': 'i32', 'stride_kd': 'i32',
    'stride_vt': 'i32', 'stride_vh': 'i32', 'stride_vd': 'i32',
    'stride_ot': 'i32', 'stride_oh': 'i32', 'stride_od': 'i32',
    'scale': 'fp32',
    'max_seqlen': 'i32',
}
constants = {
    'H': 32,
    'HEAD_DIM': 128,
    'BRANCHING_FACTOR': 8,
    'MAX_DEPTH': 5,
    'BLOCK_M': 32,
    'stride_qd': 1,
    'stride_kd': 1,
    'stride_vd': 1,
    'stride_od': 1,
}

print("Compiling Triton kernel to PTX...")

# Ensure we're compiling for SM89 (RTX 4000 Ada)
cc = 89
compiled = triton.compile(
    _ragged_attn_sparse_kernel,
    signature=signature,
    constants=constants,
    num_warps=4,
    num_stages=1,
)

asm_keys = compiled.asm.keys()
print("Available formats:", asm_keys)

if 'ptx' in asm_keys:
    with open("ragged_attn.ptx", "w") as f:
        f.write(compiled.asm['ptx'])
    print("Saved ragged_attn.ptx")

if 'cubin' in asm_keys:
    with open("ragged_attn.cubin", "wb") as f:
        f.write(compiled.asm['cubin'])
    print("Saved ragged_attn.cubin")
