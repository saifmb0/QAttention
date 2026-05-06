import torch
import os
import json
import shutil
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

base_model_path = "/home/202311016/models/Meta-Llama-3-8B-Instruct-W4A16-g128"
new_model_path = "/home/202311016/models/Meta-Llama-3-8B-Instruct-W4A16-g128-split"

if not os.path.exists(new_model_path):
    os.makedirs(new_model_path)

# Copy config and update it
config_path = os.path.join(base_model_path, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Copy quantize_config and update it to marlin
quant_config_path = os.path.join(base_model_path, "quantize_config.json")
with open(quant_config_path, "r") as f:
    qconfig = json.load(f)
qconfig["checkpoint_format"] = "marlin"

with open(os.path.join(new_model_path, "quantize_config.json"), "w") as f:
    json.dump(qconfig, f, indent=2)
with open(os.path.join(new_model_path, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Copy other required files
for f in ["special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model"]:
    if os.path.exists(os.path.join(base_model_path, f)):
        shutil.copy(os.path.join(base_model_path, f), os.path.join(new_model_path, f))

print("Loading safetensors...")
sd = load_file(os.path.join(base_model_path, "model.safetensors"), device="cpu")

new_sd = {}
layers = {}
for k, v in sd.items():
    if "model.layers." in k:
        parts = k.split(".")
        idx = parts[2]
        if idx not in layers: layers[idx] = {}
        layers[idx][".".join(parts[3:])] = v
    else:
        new_sd[k] = v

hidden_size = config["hidden_size"]
num_heads = config["num_attention_heads"]
num_kv_heads = config.get("num_key_value_heads", num_heads)
head_dim = hidden_size // num_heads

def slice_marlin(tensor, n_start, n_end):
    # For Marlin, the last dimension is N * 2 (for 4-bit)
    return tensor[..., n_start * 2 : n_end * 2]

for idx, lsd in layers.items():
    if "self_attn.qkv_proj.qweight" in lsd:
        q = lsd["self_attn.qkv_proj.qweight"]
        s = lsd["self_attn.qkv_proj.scales"]
        
        n_q = num_heads * head_dim
        n_k = num_kv_heads * head_dim
        n_v = num_kv_heads * head_dim
        
        qw = slice_marlin(q, 0, n_q).contiguous()
        kw = slice_marlin(q, n_q, n_q + n_k).contiguous()
        vw = slice_marlin(q, n_q + n_k, n_q + n_k + n_v).contiguous()
        
        qs = s[:, :n_q].contiguous()
        ks = s[:, n_q:n_q+n_k].contiguous()
        vs = s[:, n_q+n_k:n_q+n_k+n_v].contiguous()
        
        new_sd[f"model.layers.{idx}.self_attn.q_proj.qweight"] = qw
        new_sd[f"model.layers.{idx}.self_attn.k_proj.qweight"] = kw
        new_sd[f"model.layers.{idx}.self_attn.v_proj.qweight"] = vw
        new_sd[f"model.layers.{idx}.self_attn.q_proj.scales"] = qs
        new_sd[f"model.layers.{idx}.self_attn.k_proj.scales"] = ks
        new_sd[f"model.layers.{idx}.self_attn.v_proj.scales"] = vs
        
    if "mlp.gate_up_proj.qweight" in lsd:
        q = lsd["mlp.gate_up_proj.qweight"]
        s = lsd["mlp.gate_up_proj.scales"]
        n_gate = config["intermediate_size"]
        
        new_sd[f"model.layers.{idx}.mlp.gate_proj.qweight"] = slice_marlin(q, 0, n_gate).contiguous()
        new_sd[f"model.layers.{idx}.mlp.up_proj.qweight"] = slice_marlin(q, n_gate, n_gate * 2).contiguous()
        new_sd[f"model.layers.{idx}.mlp.gate_proj.scales"] = s[:, :n_gate].contiguous() # Scales are [..., N] not N*2 usually? Let's check below.
        new_sd[f"model.layers.{idx}.mlp.up_proj.scales"] = s[:, n_gate:].contiguous()
        
    if "self_attn.o_proj.qweight" in lsd:
        new_sd[f"model.layers.{idx}.self_attn.o_proj.qweight"] = lsd["self_attn.o_proj.qweight"]
        new_sd[f"model.layers.{idx}.self_attn.o_proj.scales"] = lsd["self_attn.o_proj.scales"]
        
    if "mlp.down_proj.qweight" in lsd:
        new_sd[f"model.layers.{idx}.mlp.down_proj.qweight"] = lsd["mlp.down_proj.qweight"]
        new_sd[f"model.layers.{idx}.mlp.down_proj.scales"] = lsd["mlp.down_proj.scales"]
        
    for k, v in lsd.items():
        if "norm" in k or "weight" in k and "proj" not in k:
            new_sd[f"model.layers.{idx}.{k}"] = v

# Fix scale shapes
# Actually, Marlin scales have shape [K // group_size, N] or similar. So they are just sliced along N directly.
print("Saving split model...")
save_file(new_sd, os.path.join(new_model_path, "model.safetensors"))
print("Done!")
