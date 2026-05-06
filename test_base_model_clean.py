import torch
import os
import sys

# Patch optimum
try:
    import optimum.gptq.quantizer as _opt_gptq
    if hasattr(_opt_gptq, "BACKEND") and not hasattr(_opt_gptq.BACKEND, "EXLLAMA_V1"):
        setattr(_opt_gptq.BACKEND, "EXLLAMA_V1", "exllama_v1")
except: pass

import gptqmodel.nn_modules.qlinear.marlin as _gptq_marlin
_orig_marlin_init = _gptq_marlin.MarlinQuantLinear.__init__
_orig_marlin_post_init = _gptq_marlin.MarlinQuantLinear.post_init

def _patched_marlin_init(self, bits, group_size, desc_act, sym, in_features, out_features, *args, **kwargs):
    _orig_marlin_init(self, bits, group_size, desc_act, sym, in_features, out_features, *args, **kwargs)
    # Override the GPTQ-shaped qweight with the Marlin shape [K//16, N*2]
    pack_dtype = kwargs.get("pack_dtype", torch.int32)
    self.qweight = torch.nn.Parameter(
        torch.zeros((in_features // 16, out_features * 2), dtype=pack_dtype),
        requires_grad=False
    )

def _patched_marlin_post_init(self):
    device = self.qweight.device
    self.is_k_full = _gptq_marlin.marlin_is_k_full(self.desc_act, is_row_parallel=False)
    self.workspace = _gptq_marlin.marlin_make_workspace_new(device)
    
    if self.desc_act:
        g_idx, g_idx_sort_indices = _gptq_marlin.marlin_sort_g_idx(getattr(self, "g_idx"))
        _gptq_marlin._transform_param(self, "g_idx", lambda _: g_idx)
        self.g_idx_sort_indices = g_idx_sort_indices
    else:
        setattr(self, "g_idx", _gptq_marlin.marlin_make_empty_g_idx(device))
        self.g_idx_sort_indices = _gptq_marlin.marlin_make_empty_g_idx(device)

    setattr(self, "qzeros", _gptq_marlin.marlin_make_empty_g_idx(device))

    if hasattr(self, "bias") and self.bias is not None:
        self.bias.data = _gptq_marlin.marlin_permute_bias(self.bias)
    
    # We DO NOT call the repacking functions because the weights are ALREADY Marlin packed!
    # However, we must call super().post_init() but we can't easily without duplicating it.
    # Actually we can just do what super().post_init() does (it's empty for GPTQQuantLinear).
    pass

_gptq_marlin.MarlinQuantLinear.__init__ = _patched_marlin_init
_gptq_marlin.MarlinQuantLinear.post_init = _patched_marlin_post_init

from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/home/202311016/models/Meta-Llama-3-8B-Instruct-W4A16-g128-split"
print(f"Loading {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
prompt = "Hello, how are you today?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Fix transformers generate patch from gptqmodel
import transformers.generation.utils as gen_utils
if hasattr(gen_utils.GenerationMixin, "_prepare_cache_for_generation_orig"):
    gen_utils.GenerationMixin._prepare_cache_for_generation = gen_utils.GenerationMixin._prepare_cache_for_generation_orig
    
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False, use_cache=True)
    
print(f"Prompt: {prompt}")
print(f"Output: {tokenizer.decode(output[0])}")
