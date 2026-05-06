import re

with open("scripts/w4a16_e2e.py", "r") as f:
    content = f.read()

# Find the start of load_eagle_model
start_idx = content.find("def load_eagle_model(")
# Find the start of run_ar_baseline
end_idx = content.find("def run_ar_baseline(")

new_func = """def load_eagle_model(
    base_model: str,
    eagle_model: str,
    use_eagle3: bool = True,
    total_token: int = 60,
    depth: int = 7,
    top_k: int = 10,
    max_length: int = 2048,
    use_fp8: bool = False,
    load_in_4bit: bool = False,
) -> "eagle.model.ea_model.EaModel":
    _patch_transformers_for_eagle()
    from eagle.model.ea_model import EaModel
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
    from transformers import BitsAndBytesConfig, AutoConfig
    import os
    import torch

    try:
        import optimum.gptq.quantizer as _opt_gptq
        if hasattr(_opt_gptq, "BACKEND") and not hasattr(_opt_gptq.BACKEND, "EXLLAMA_V1"):
            try: type(_opt_gptq.BACKEND).EXLLAMA_V1 = "exllama_v1"
            except:
                try: setattr(_opt_gptq.BACKEND, "EXLLAMA_V1", "exllama_v1")
                except: pass
    except: pass

    # Patch GPTQModel's MarlinQuantLinear to accept native Marlin shapes
    try:
        import gptqmodel.nn_modules.qlinear.marlin as _gptq_marlin
        _orig_marlin_init = _gptq_marlin.MarlinQuantLinear.__init__
        _orig_marlin_post_init = _gptq_marlin.MarlinQuantLinear.post_init

        def _patched_marlin_init(self, bits, group_size, desc_act, sym, in_features, out_features, *args, **kwargs):
            _orig_marlin_init(self, bits, group_size, desc_act, sym, in_features, out_features, *args, **kwargs)
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

        _gptq_marlin.MarlinQuantLinear.__init__ = _patched_marlin_init
        _gptq_marlin.MarlinQuantLinear.post_init = _patched_marlin_post_init
    except ImportError:
        pass

    from eagle.model.configs import EConfig
    _orig_econfig = EConfig.from_pretrained
    @classmethod
    def _patched_econfig(cls, path, **kwargs):
        c = _orig_econfig(path, **kwargs)
        if not hasattr(c, "draft_vocab_size"): c.draft_vocab_size = getattr(c, "vocab_size", 128256)
        return c
    EConfig.from_pretrained = _patched_econfig

    import eagle.model.cnets, eagle.model.cnets1, huggingface_hub
    _orig_hf = huggingface_hub.hf_hub_download
    def _patched_hf(repo, file, **kwargs):
        if isinstance(repo, str) and (os.path.isdir(repo) or os.path.sep in repo):
            loc = os.path.join(repo, file)
            if os.path.exists(loc): return loc
            raise FileNotFoundError(f"Local file {loc} not found.")
        return _orig_hf(repo, file, **kwargs)
    huggingface_hub.hf_hub_download = eagle.model.cnets.hf_hub_download = eagle.model.cnets1.hf_hub_download = _patched_hf

    def _make_safe_init(orig):
        def _safe(self, *args, **kwargs):
            try: return orig(self, *args, **kwargs)
            except:
                nk = kwargs.copy()
                if nk.get("load_emb"): nk["load_emb"] = False; return orig(self, *args, **nk)
                elif len(args) > 0 and args[0] is True: na = list(args); na[0] = False; return orig(self, *na, **nk)
                raise
        return _safe
    eagle.model.cnets.Model.__init__ = _make_safe_init(eagle.model.cnets.Model.__init__)
    if hasattr(eagle.model.cnets1, "Model"): eagle.model.cnets1.Model.__init__ = _make_safe_init(eagle.model.cnets1.Model.__init__)

    class _PatchedLinear(torch.nn.Linear):
        def __init__(self, inf, outf, *args, **kwargs):
            if inf == 12288 and outf == 4096: inf = 8192
            super().__init__(inf, outf, *args, **kwargs)
    eagle.model.cnets.nn.Linear = _PatchedLinear

    import eagle.model.ea_model
    _orig_ea = eagle.model.ea_model.EaModel.__init__
    def _patched_ea(self, *args, **kwargs):
        if len(args) > 1:
            base = args[1]
            for l in getattr(base.model, "layers", []):
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    pr = getattr(l.self_attn, p, None)
                    if pr and not hasattr(pr, "weight"): pr.weight = pr.qweight
                for p in ["gate_proj", "up_proj", "down_proj"]:
                    pr = getattr(l.mlp, p, None)
                    if pr and not hasattr(pr, "weight"): pr.weight = pr.qweight
        return _orig_ea(self, *args, **kwargs)
    eagle.model.ea_model.EaModel.__init__ = _patched_ea

    # EAGLE's cnets.py _init_rope only understands rope_scaling types "linear" and
    # "dynamic" and requires a "factor" key.  Llama-3.1 uses type "llama3" with a
    # slightly different schema.  For the EAGLE *draft* model the RoPE precision
    # doesn't affect correctness testing, so we simply fall back to standard
    # (unscaled) RoPE for any type that cnets.py doesn't natively handle.
    try:
        from eagle.model import cnets as _cnets
        _EAGLE_ROPE_TYPES = {"linear", "dynamic"}
        _orig_init_rope = _cnets.LlamaAttention._init_rope
        def _patched_init_rope(self):
            rs = getattr(self.config, "rope_scaling", None)
            if isinstance(rs, dict):
                rope_type = rs.get("type") or rs.get("rope_type", "")
                if rope_type not in _EAGLE_ROPE_TYPES:
                    import copy
                    self.config = copy.deepcopy(self.config)
                    self.config.rope_scaling = None
                elif "type" not in rs:
                    import copy
                    self.config = copy.deepcopy(self.config)
                    self.config.rope_scaling = {**rs, "type": rope_type}
            _orig_init_rope(self)
        _cnets.LlamaAttention._init_rope = _patched_init_rope
    except Exception:
        pass

    print(f"\\n  Loading Eagle model:")
    print(f"    Base:  {base_model}")
    print(f"    Eagle: {eagle_model}")
    print(f"    Mode:  {'EAGLE-3' if use_eagle3 else 'EAGLE-2'}")
    print(f"    Tree:  total_token={total_token}, depth={depth}, top_k={top_k}")
    t0 = time.perf_counter()
    _dtype = torch.float16
    print(f"    dtype: float16 (EAGLE tested config)")
    print(f"    KV cache max_length: {max_length} tokens")

    try:
        import transformers as _tr
        _tr.logging.set_verbosity_error()
    except Exception:
        pass

    model = EaModel.from_pretrained(
        use_eagle3=use_eagle3,
        base_model_path=base_model,
        ea_model_path=eagle_model,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        torch_dtype=_dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        load_in_8bit=use_fp8,
        load_in_4bit=load_in_4bit,
    )
    model.eval()
    cfg = model.base_model.config
    H    = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H)
    D    = cfg.hidden_size // H
    L    = cfg.num_hidden_layers
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")
    print(f"    LLM: H={H}, H_kv={H_kv}, D={D}, layers={L}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"    GPU: {p.name}  SM {p.major}.{p.minor}  "
              f"{p.total_memory // 1024**3} GB")
    return model


"""

new_content = content[:start_idx] + new_func + content[end_idx:]

with open("scripts/w4a16_e2e.py", "w") as f:
    f.write(new_content)

print("Patched scripts/w4a16_e2e.py successfully")
