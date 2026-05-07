#!/usr/bin/env python3
"""
Quantize LLaMA-3.1-8B-Instruct to 4-bit GPTQ (standard format, desc_act=False).
desc_act=False → compatible with GPTQModel's Marlin backend directly.
Saves in standard GPTQ format (separate q/k/v projections, K//8 × N qweight shape).
"""
import os
import json
import torch

# Fix 1: GPTQModel 6.x passes `dtype` kwarg to LlamaForCausalLM.__init__
# which doesn't accept it in transformers 4.53.1.
import transformers as _tr
_orig_llama_init = _tr.models.llama.modeling_llama.LlamaForCausalLM.__init__
def _patched_llama_init(self, config, **kwargs):
    kwargs.pop('dtype', None)
    _orig_llama_init(self, config, **kwargs)
_tr.models.llama.modeling_llama.LlamaForCausalLM.__init__ = _patched_llama_init

# Fix 2: GenerationConfig.to_json_string fails when dtype is torch.dtype object.
_orig_json_dumps = json.dumps
def _patched_json_dumps(obj, **kwargs):
    def _default(o):
        if isinstance(o, torch.dtype):
            return str(o).replace("torch.", "")
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
    kwargs.setdefault('default', _default)
    return _orig_json_dumps(obj, **kwargs)
json.dumps = _patched_json_dumps

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
OUT_PATH  = "/home/user/models/Meta-Llama-3.1-8B-Instruct-W4A16-g128-gptq"
os.makedirs(OUT_PATH, exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Loading {MODEL_ID} ...")

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
)

model = GPTQModel.load(MODEL_ID, quant_config)

print("Loading calibration data ...")
data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [x["text"] for x in data if len(x["text"].strip()) > 200][:128]
print(f"  {len(texts)} calibration samples")

print("Quantizing ...")
model.quantize(texts, batch_size=4)

print(f"Saving to {OUT_PATH} ...")
model.save(OUT_PATH)
print("Done!")
