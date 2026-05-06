import torch
from transformers import AutoConfig, LlamaForCausalLM
import os

# Use a standard model for testing the from_pretrained(None, ...) logic
model_id = "meta-llama/Llama-2-7b-hf" # Just for config

try:
    config = AutoConfig.from_pretrained(model_id)
    # Create a dummy state dict
    sd = {} # Empty sd will fail loading but we want to see if it reaches that point
    model = LlamaForCausalLM.from_pretrained(None, config=config, state_dict=sd, low_cpu_mem_usage=True)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
