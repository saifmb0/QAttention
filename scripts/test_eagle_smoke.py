#!/usr/bin/env python3
"""
test_eagle_smoke.py — Minimal EAGLE sanity check
=================================================
Run this BEFORE e2e_benchmark.py to verify that EAGLE + the current
transformers/accelerate stack can generate coherent text.

Usage:
    python3 scripts/test_eagle_smoke.py

Exit codes:
    0 — generation looks coherent (no obvious garbage)
    1 — generation is degenerate / environment is broken
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import transformers
import accelerate

print("=" * 60)
print("EAGLE smoke test — environment report")
print("=" * 60)
print(f"  transformers : {transformers.__version__}")
print(f"  accelerate   : {accelerate.__version__}")
print(f"  torch        : {torch.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU          : {p.name}  SM{p.major}.{p.minor}  "
          f"{p.total_memory // 1024**3} GB")
print()

# ── Load model ────────────────────────────────────────────────────────────────
# Apply the same transformer-version compat shim as e2e_benchmark.py
try:
    import transformers.utils as _tu
    from typing import TypedDict
    if not hasattr(_tu, "LossKwargs"):
        class _LK(TypedDict, total=False): pass
        _tu.LossKwargs = _LK
    if not hasattr(_tu, "auto_docstring"):
        def _ad(*a, **k): return a[0] if a and callable(a[0]) else lambda fn: fn
        _tu.auto_docstring = _ad
    if not hasattr(_tu, "can_return_tuple"):
        _tu.can_return_tuple = lambda fn: fn
except Exception:
    pass

from eagle.model.ea_model import EaModel

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

print(f"Loading {BASE_MODEL} + {EAGLE_MODEL} …")
model = EaModel.from_pretrained(
    base_model_path=BASE_MODEL,
    ea_model_path=EAGLE_MODEL,
    use_eagle3=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
print("  Model loaded.\n")

# ── Tokenise a simple prompt ──────────────────────────────────────────────────
tok = model.tokenizer
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2?"}],
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tok([prompt], add_special_tokens=False).input_ids
input_ids = torch.as_tensor(input_ids).cuda()

print(f"Prompt tokens: {input_ids.shape[1]}")

# ── Generate via EAGLE's own eagenerate (no benchmark hooks) ─────────────────
with torch.no_grad():
    out_ids, new_token, idx = model.eagenerate(
        input_ids,
        temperature=0.0,
        max_new_tokens=30,
        is_llama3=True,
        log=True,
    )

response_ids = out_ids[0][input_ids.shape[1]:]
response = tok.decode(response_ids, skip_special_tokens=True)

print(f"\nGenerated {new_token} tokens in {idx + 1} steps")
print(f"Response: {response!r}")
print()

# ── Heuristic coherence check ─────────────────────────────────────────────────
unique_toks = len(set(response_ids.tolist()))
total_toks  = len(response_ids)
repetition  = 1.0 - (unique_toks / total_toks) if total_toks > 0 else 1.0

print(f"Unique tokens: {unique_toks}/{total_toks}  repetition index: {repetition:.2f}")

if repetition > 0.8 or total_toks == 0:
    print("\n[FAIL] Output is degenerate (high repetition or empty).")
    print("       Environment is broken — check transformers / accelerate versions.")
    print("       EAGLE requires transformers >=4.53.1 and accelerate <1.0")
    sys.exit(1)
else:
    print("\n[PASS] Output looks coherent.  Benchmark should work.")
    sys.exit(0)
