import torch
import numpy as np

def _reverse_marlin_weights(qweight, K, N):
    device = qweight.device
    total_int32 = qweight.numel()
    unpacked = torch.zeros((total_int32, 8), dtype=torch.int8, device=device)
    for i in range(8):
        unpacked[:, i] = (qweight.view(-1).to(torch.int64) >> (4 * i)) & 0xF
    unpacked = unpacked.reshape(K // 16, N // 64, 16, 64)
    unpacked = unpacked.permute(0, 2, 1, 3).reshape(K, N)
    perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device)
    inv_perm = torch.argsort(perm)
    unpacked = unpacked.reshape(-1, 8)[:, inv_perm].reshape(K, N)
    unpacked = unpacked.reshape(K // 8, 8, N)
    gptq_packed = torch.zeros((K // 8, N), dtype=torch.int64, device=device)
    for i in range(8):
        gptq_packed |= unpacked[:, i, :].to(torch.int64) << (4 * i)
    return gptq_packed.to(torch.int32)

# Generate some dummy GPTQ weights
K, N = 256, 256
gptq_weights = torch.randint(0, 16, (K, N), dtype=torch.int8)

# Pack to standard GPTQ layout
gptq_packed = torch.zeros((K // 8, N), dtype=torch.int32)
for i in range(8):
    gptq_packed |= gptq_weights[i::8, :].to(torch.int32) << (4 * i)

# Use gptqmodel's repack to convert GPTQ to Marlin
from gptqmodel.utils.marlin import gptq_marlin_repack

# gptq_marlin_repack expects:
# b_q_weight: [K // 8, N]
# perm: empty (or standard perm)
# size_k, size_n
# num_bits
perm = torch.empty(0, dtype=torch.int32)
marlin_weights = gptq_marlin_repack(gptq_packed.cuda(), perm.cuda(), K, N, 4).cpu()

print(f"Marlin shape: {marlin_weights.shape}")
print(f"GPTQ shape: {gptq_packed.shape}")

# Reverse
reversed_gptq = _reverse_marlin_weights(marlin_weights, K, N)

# Compare
if torch.equal(gptq_packed, reversed_gptq):
    print("SUCCESS! _reverse_marlin_weights correctly reverses gptq_marlin_repack.")
else:
    print("FAILURE! _reverse_marlin_weights is incorrect.")
    
    # Let's see the difference in a small sub-block
    print("Original GPTQ unpacked [0, :8]:", gptq_weights[0, :8])
    
    unpacked_rev = torch.zeros((K, N), dtype=torch.int8)
    for i in range(8):
        unpacked_rev[i::8, :] = (reversed_gptq.to(torch.int64) >> (4 * i)) & 0xF
    print("Reversed GPTQ unpacked [0, :8]:", unpacked_rev[0, :8])
