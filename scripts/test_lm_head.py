import torch, sys
sys.path.insert(0, ".")
from scripts.e2e_benchmark import load_eagle_model
model = load_eagle_model("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", True, 60, 5, 10)
print("lm_head sum:", model.base_model.lm_head.weight.sum().item())
print("lm_head std:", model.base_model.lm_head.weight.std().item())
print("lm_head mean:", model.base_model.lm_head.weight.mean().item())
from transformers import AutoModelForCausalLM
clean = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda:0")
print("clean lm_head sum:", clean.lm_head.weight.sum().item())
print("clean lm_head std:", clean.lm_head.weight.std().item())
print("clean lm_head mean:", clean.lm_head.weight.mean().item())
