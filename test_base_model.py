import torch
from transformers import AutoConfig, LlamaForCausalLM
import os
from safetensors.torch import load_file

base_model_path = "/home/user/models/Meta-Llama-3-8B-Instruct-W4A16-g128-Rot"

class MarlinToGPTQConverter:
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.intermediate_size = config.intermediate_size
        self.group_size = 128
        self.device = torch.device("cpu")

    def _reverse_marlin_weights(self, qweight, K, N):
        qweight = qweight.to(self.device)
        total_int32 = qweight.numel()
        unpacked = torch.zeros((total_int32, 8), dtype=torch.int8, device=self.device)
        for i in range(8):
            unpacked[:, i] = (qweight.view(-1).to(torch.int64) >> (4 * i)) & 0xF
        unpacked = unpacked.reshape(K // 16, N // 64, 16, 64)
        unpacked = unpacked.permute(0, 2, 1, 3).reshape(K, N)
        perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=self.device)
        inv_perm = torch.argsort(perm)
        unpacked = unpacked.reshape(-1, 8)[:, inv_perm].reshape(K, N)
        unpacked = unpacked.reshape(K // 8, 8, N)
        gptq_packed = torch.zeros((K // 8, N), dtype=torch.int64, device=self.device)
        for i in range(8):
            gptq_packed |= unpacked[:, i, :].to(torch.int64) << (4 * i)
        return gptq_packed.to(torch.int32)

    def _reverse_marlin_scales(self, scales):
        scales = scales.to(self.device)
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm = torch.tensor(scale_perm, device=self.device)
        s = scales.reshape((-1, 64))[:, scale_perm]
        return s.reshape(scales.shape).contiguous()

    def convert(self, sd):
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
        
        def _get_qzeros(k_dim, n_dim):
            q = torch.zeros((k_dim // self.group_size, n_dim // 8), dtype=torch.int64)
            for i in range(8):
                q |= torch.full((k_dim // self.group_size, n_dim // 8), 8, dtype=torch.int64) << (4 * i)
            return q.to(torch.int32)

        for idx, layer_sd in layers.items():
            if "self_attn.qkv_proj.qweight" in layer_sd:
                K, N_q, N_k, N_v = self.hidden_size, self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim
                gptq_q = self._reverse_marlin_weights(layer_sd["self_attn.qkv_proj.qweight"], K, N_q + N_k + N_v)
                gptq_s = self._reverse_marlin_scales(layer_sd["self_attn.qkv_proj.scales"])
                new_sd[f"model.layers.{idx}.self_attn.q_proj.qweight"] = gptq_q[:, :N_q]
                new_sd[f"model.layers.{idx}.self_attn.k_proj.qweight"] = gptq_q[:, N_q:N_q+N_k]
                new_sd[f"model.layers.{idx}.self_attn.v_proj.qweight"] = gptq_q[:, N_q+N_k:]
                new_sd[f"model.layers.{idx}.self_attn.q_proj.scales"] = gptq_s[:, :N_q]
                new_sd[f"model.layers.{idx}.self_attn.k_proj.scales"] = gptq_s[:, N_q:N_q+N_k]
                new_sd[f"model.layers.{idx}.self_attn.v_proj.scales"] = gptq_s[:, N_q+N_k:]
                for p, n in [("q_proj", N_q), ("k_proj", N_k), ("v_proj", N_v)]:
                    new_sd[f"model.layers.{idx}.self_attn.{p}.g_idx"] = torch.tensor([i // self.group_size for i in range(K)], dtype=torch.int32)
                    new_sd[f"model.layers.{idx}.self_attn.{p}.qzeros"] = _get_qzeros(K, n)
            
            if "self_attn.o_proj.qweight" in layer_sd:
                K, N = self.num_heads * self.head_dim, self.hidden_size
                new_sd[f"model.layers.{idx}.self_attn.o_proj.qweight"] = self._reverse_marlin_weights(layer_sd["self_attn.o_proj.qweight"], K, N)
                new_sd[f"model.layers.{idx}.self_attn.o_proj.scales"] = self._reverse_marlin_scales(layer_sd["self_attn.o_proj.scales"])
                new_sd[f"model.layers.{idx}.self_attn.o_proj.g_idx"] = torch.tensor([i // self.group_size for i in range(K)], dtype=torch.int32)
                new_sd[f"model.layers.{idx}.self_attn.o_proj.qzeros"] = _get_qzeros(K, N)

            if "mlp.gate_up_proj.qweight" in layer_sd:
                K, N_gate, N_up = self.hidden_size, self.intermediate_size, self.intermediate_size
                gptq_q = self._reverse_marlin_weights(layer_sd["mlp.gate_up_proj.qweight"], K, N_gate + N_up)
                gptq_s = self._reverse_marlin_scales(layer_sd["mlp.gate_up_proj.scales"])
                new_sd[f"model.layers.{idx}.mlp.gate_proj.qweight"] = gptq_q[:, :N_gate]
                new_sd[f"model.layers.{idx}.mlp.up_proj.qweight"] = gptq_q[:, N_gate:]
                new_sd[f"model.layers.{idx}.mlp.gate_proj.scales"] = gptq_s[:, :N_gate]
                new_sd[f"model.layers.{idx}.mlp.up_proj.scales"] = gptq_s[:, N_gate:]
                for p in ["gate_proj", "up_proj"]:
                    new_sd[f"model.layers.{idx}.mlp.{p}.g_idx"] = torch.tensor([i // self.group_size for i in range(K)], dtype=torch.int32)
                    new_sd[f"model.layers.{idx}.mlp.{p}.qzeros"] = _get_qzeros(K, N_gate)
            
            if "mlp.down_proj.qweight" in layer_sd:
                K, N = self.intermediate_size, self.hidden_size
                new_sd[f"model.layers.{idx}.mlp.down_proj.qweight"] = self._reverse_marlin_weights(layer_sd["mlp.down_proj.qweight"], K, N)
                new_sd[f"model.layers.{idx}.mlp.down_proj.scales"] = self._reverse_marlin_scales(layer_sd["mlp.down_proj.scales"])
                new_sd[f"model.layers.{idx}.mlp.down_proj.g_idx"] = torch.tensor([i // self.group_size for i in range(K)], dtype=torch.int32)
                new_sd[f"model.layers.{idx}.mlp.down_proj.qzeros"] = _get_qzeros(K, N)

            for k, v in layer_sd.items():
                if "norm" in k or "weight" in k and "proj" not in k:
                    new_sd[f"model.layers.{idx}.{k}"] = v
        return new_sd

print("Loading config...")
config = AutoConfig.from_pretrained(base_model_path)
print("Loading safetensors...")
sd = load_file(os.path.join(base_model_path, "model_gptq_marlin.safetensors"), device="cpu")
print("Converting sd...")
converted_sd = MarlinToGPTQConverter(config).convert(sd)

print("Initializing model...")
model = LlamaForCausalLM.from_pretrained(None, config=config, state_dict=converted_sd, device_map={"": "cuda:0"}, torch_dtype=torch.float16)
model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
prompt = "Hello, how are you today?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

print("Generating...")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    
print(f"Prompt: {prompt}")
print(f"Output: {tokenizer.decode(output[0])}")
