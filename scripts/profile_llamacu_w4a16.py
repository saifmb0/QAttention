#!/usr/bin/env python3
"""
profile_llamacu_w4a16.py
========================
Step-level CUDA-event profiler for the llamacu W4A16-GPTQ-Marlin + EAGLE-3
generate loop.  Breaks each decode step into three phases:

    draft_ms      — C.draft() : EAGLE-3 cnets forward (FP16)
    decode_ms     — self.decode() : W4A16 GPTQ-Marlin target model forward
                     (tree_size tokens, attention + W4A16 GEMM × 32 layers + lm_head)
    fixup_ms      — C.verify_and_fix() : accept/reject token selection

Reports per-step averages, the fraction of total step time each phase takes,
and the ratio between draft and decode so you can see whether W4A16 has
moved the decode bottleneck relative to FP16.

Additionally runs torch.profiler on ONE decode call to list the top CUDA
kernels by duration — this lets you see whether attention or matmul dominates
inside the opaque C.decode().

Usage
-----
  python scripts/profile_llamacu_w4a16.py \\
      --base-model  /path/to/W4A16-GPTQ-Marlin-LLaMA-3.1-8B \\
      --eagle-model /path/to/EAGLE3-LLaMA3.1-Instruct-8B \\
      --steps 200

  # Compare with FP16 (base model loaded in float16, not quantized)
  python scripts/profile_llamacu_w4a16.py \\
      --base-model /path/to/fp16-model  --steps 200 --fp16

Dependencies
------------
  SpecMQuant repo must be on PYTHONPATH:
    export PYTHONPATH=/home/202311016/sandbox/SpecMQuant:$PYTHONPATH
  or run from that directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import torch

# ── SpecMQuant / llamacu path ─────────────────────────────────────────────────
_SPEC_ROOT = "/home/202311016/sandbox/SpecMQuant"
if _SPEC_ROOT not in sys.path:
    sys.path.insert(0, _SPEC_ROOT)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────────────────────
# CUDA-event step timer
# ─────────────────────────────────────────────────────────────────────────────

class StepTimer:
    """Records CUDA-event timing for draft / decode / fixup each step."""

    def __init__(self):
        self.draft_ms:  List[float] = []
        self.decode_ms: List[float] = []
        self.fixup_ms:  List[float] = []

    def _pair(self):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        return e0, e1

    def time_step(self, model, tree_draft_ids, tree_position_ids,
                  cache_length, tree_attn_mask, tree_parent, tree_gt_ids):
        """Run one generate step, return accept_length."""
        from llamacu import C

        e0d, e1d = self._pair()
        e0v, e1v = self._pair()
        e0f, e1f = self._pair()

        e0d.record()
        C.draft(tree_draft_ids.data_ptr(), tree_position_ids.data_ptr(),
                cache_length.data_ptr(), tree_attn_mask.data_ptr(),
                tree_parent.data_ptr())
        e1d.record()

        e0v.record()
        logits = model.decode(tree_draft_ids, tree_position_ids,
                              cache_length, mask_2d=tree_attn_mask)
        e1v.record()

        tree_gt_ids.copy_(logits.argmax(dim=-1))

        e0f.record()
        accept_length = C.verify_and_fix(
            tree_draft_ids.numel(),
            tree_draft_ids.data_ptr(), tree_gt_ids.data_ptr(),
            tree_position_ids.data_ptr(), cache_length.data_ptr(),
            tree_attn_mask.data_ptr(), tree_parent.data_ptr(),
        )
        e1f.record()

        torch.cuda.synchronize()
        self.draft_ms.append(e0d.elapsed_time(e1d))
        self.decode_ms.append(e0v.elapsed_time(e1v))
        self.fixup_ms.append(e0f.elapsed_time(e1f))
        return accept_length

    def summary(self) -> str:
        n = len(self.draft_ms)
        if n == 0:
            return "  [timer] no data"

        def _stats(vals):
            v = sorted(vals)
            return (sum(v) / n,
                    v[int(n * 0.50)],
                    v[int(n * 0.95)])

        dm, dm50, dm95 = _stats(self.draft_ms)
        vm, vm50, vm95 = _stats(self.decode_ms)
        fm, fm50, fm95 = _stats(self.fixup_ms)
        total_m = dm + vm + fm

        def pct(v): return v / total_m * 100 if total_m else 0.0

        lines = [
            f"  ┌── Step Timing  ({n} steps)  {'─' * 40}",
            f"  │ {'Phase':<18}  {'mean ms':>9}  {'p50':>9}  {'p95':>9}  {'% total':>8}",
            f"  │ {'─' * 18}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 8}",
            f"  │ {'draft (C.draft)':<18}  {dm:>9.3f}  {dm50:>9.3f}  {dm95:>9.3f}  {pct(dm):>7.1f}%",
            f"  │ {'decode (target)':<18}  {vm:>9.3f}  {vm50:>9.3f}  {vm95:>9.3f}  {pct(vm):>7.1f}%",
            f"  │ {'fixup (C.verify)':<18}  {fm:>9.3f}  {fm50:>9.3f}  {fm95:>9.3f}  {pct(fm):>7.1f}%",
            f"  │ {'─' * 18}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 8}",
            f"  │ {'TOTAL / step':<18}  {total_m:>9.3f}",
            f"  │",
            f"  │  draft/decode ratio: {dm/vm:.3f}×  "
            f"({'draft is the bottleneck' if dm > vm else 'decode is the bottleneck'})",
            f"  └{'─' * 70}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# torch.profiler kernel breakdown for one decode call
# ─────────────────────────────────────────────────────────────────────────────

def profile_one_decode(model, tree_draft_ids, tree_position_ids,
                       cache_length, tree_attn_mask, top_k: int = 20):
    """
    Run torch.profiler on a single decode call.
    Returns a formatted string of the top-K CUDA kernels by duration.
    """
    try:
        from torch.profiler import profile as tprof, ProfilerActivity, record_function
    except ImportError:
        return "  [kernel profiler] torch.profiler not available"

    with tprof(activities=[ProfilerActivity.CUDA],
               record_shapes=False,
               with_stack=False) as prof:
        with record_function("decode"):
            _ = model.decode(tree_draft_ids, tree_position_ids,
                             cache_length, mask_2d=tree_attn_mask)
        torch.cuda.synchronize()

    events = prof.key_averages()
    # Filter to CUDA events, sort by CUDA time
    cuda_events = [(e.key, e.self_device_time_total, e.count)
                   for e in events if e.self_device_time_total > 0]
    cuda_events.sort(key=lambda x: -x[1])

    total_us = sum(t for _, t, _ in cuda_events)
    lines = [
        f"  ┌── Top-{top_k} CUDA kernels in one decode call  (total={total_us/1000:.2f} ms) {'─' * 20}",
        f"  │ {'Kernel':<55}  {'ms':>8}  {'%':>6}  {'calls':>6}",
        f"  │ {'─' * 55}  {'─' * 8}  {'─' * 6}  {'─' * 6}",
    ]
    for name, us, cnt in cuda_events[:top_k]:
        ms  = us / 1000
        pct = us / total_us * 100 if total_us else 0
        lines.append(f"  │ {name[:55]:<55}  {ms:>8.3f}  {pct:>5.1f}%  {cnt:>6}")
    lines.append(f"  └{'─' * 80}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model",   required=True,
                    help="Path to W4A16 GPTQ-Marlin base model (or FP16 model with --fp16)")
    ap.add_argument("--eagle-model",  required=True,
                    help="Path to EAGLE-3 draft model weights")
    ap.add_argument("--steps",        type=int, default=200,
                    help="Number of generate steps to time (default: 200)")
    ap.add_argument("--warmup",       type=int, default=20,
                    help="Warmup steps before timing begins (default: 20)")
    ap.add_argument("--tree-size",    type=int, default=60,
                    help="Total token tree budget (default: 60, EAGLE-3 default)")
    ap.add_argument("--top-k",        type=int, default=10,
                    help="EAGLE draft topk per iter (default: 10)")
    ap.add_argument("--num-iter",     type=int, default=6,
                    help="EAGLE draft iterations (default: 6)")
    ap.add_argument("--memory-limit", type=float, default=0.8,
                    help="GPU memory fraction for llamacu pool (default: 0.8)")
    ap.add_argument("--kernel-profile", action="store_true",
                    help="Run torch.profiler on one decode call to show CUDA kernel breakdown")
    ap.add_argument("--fp16", action="store_true",
                    help="Load base model in FP16 instead of W4A16 for comparison "
                         "(uses llamacu.LLM_with_eagle3 instead of W4A16GPTQMarlinLLM_with_eagle3)")
    ap.add_argument("--chunk-length", type=int, default=4096)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    device = torch.device("cuda:0")
    props  = torch.cuda.get_device_properties(device)
    print(f"\n  GPU: {props.name}  SM{props.major}.{props.minor}  "
          f"{props.total_memory // (1 << 30)} GB VRAM")

    quant = "FP16" if args.fp16 else "W4A16-GPTQ-Marlin"
    print(f"  Mode: {quant}")
    print(f"  Base:  {args.base_model}")
    print(f"  Draft: {args.eagle_model}")
    print(f"  Tree:  size={args.tree_size}  top_k={args.top_k}  num_iter={args.num_iter}")
    print(f"  Steps: {args.warmup} warmup + {args.steps} timed")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n  Loading {quant} model ...")
    t0 = time.perf_counter()
    try:
        if args.fp16:
            from llamacu.speculative.eagle3 import LLM_with_eagle3
            model = LLM_with_eagle3(
                eagle_path=args.eagle_model,
                base_path=args.base_model,
                num_iter=args.num_iter,
                topk_per_iter=args.top_k,
                tree_size=args.tree_size,
                memory_limit=args.memory_limit,
                chunk_length=args.chunk_length,
            )
        else:
            from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import (
                W4A16GPTQMarlinLLM_with_eagle,
            )
            model = W4A16GPTQMarlinLLM_with_eagle(
                eagle_path=args.eagle_model,
                base_path=args.base_model,
                num_iter=args.num_iter,
                topk_per_iter=args.top_k,
                tree_size=args.tree_size,
                memory_limit=args.memory_limit,
                chunk_length=args.chunk_length,
                rotation=True,
            )
        model.init_storage()
        model.load_from_hf()
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    from llamacu import C

    # ── Bootstrap: prefill a short prompt ────────────────────────────────────
    print(f"\n  Prefilling short prompt to initialise KV cache ...")
    dummy_prompt = torch.arange(64, dtype=torch.int32, device="cuda")
    pos = torch.arange(64, dtype=torch.int32, device="cuda")
    logits = model.prefill(dummy_prompt, pos)
    model.tree_draft_ids[:1].copy_(logits[0].argmax(dim=-1))

    # W4A16GPTQMarlinLLM_with_eagle3 has a draft_prefill_sep path;
    # initialise the draft KV cache (one-time prefill before the loop).
    model.cache_length[0] = 64
    if hasattr(model, 'draft_prefill_sep') and model.draft_prefill_sep:
        C.draft_prefill(model.tree_draft_ids.data_ptr(),
                        model.tree_position_ids.data_ptr(),
                        model.cache_length.data_ptr())

    timer = StepTimer()
    cache_len_start = 64

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"  Warmup {args.warmup} steps ...")
    for i in range(args.warmup):
        model.cache_length[0] = cache_len_start + i
        C.draft(model.tree_draft_ids.data_ptr(), model.tree_position_ids.data_ptr(),
                model.cache_length.data_ptr(), model.tree_attn_mask.data_ptr(),
                model.tree_parent.data_ptr())
        logits = model.decode(model.tree_draft_ids, model.tree_position_ids,
                              model.cache_length, mask_2d=model.tree_attn_mask)
        model.tree_gt_ids.copy_(logits.argmax(dim=-1))
        C.verify_and_fix(
            model.tree_draft_ids.numel(),
            model.tree_draft_ids.data_ptr(), model.tree_gt_ids.data_ptr(),
            model.tree_position_ids.data_ptr(), model.cache_length.data_ptr(),
            model.tree_attn_mask.data_ptr(), model.tree_parent.data_ptr(),
        )
    torch.cuda.synchronize()

    # ── Kernel profiler on one decode ─────────────────────────────────────────
    if args.kernel_profile:
        print(f"\n  Running kernel profiler on one decode call ...")
        model.cache_length[0] = cache_len_start + args.warmup
        C.draft(model.tree_draft_ids.data_ptr(), model.tree_position_ids.data_ptr(),
                model.cache_length.data_ptr(), model.tree_attn_mask.data_ptr(),
                model.tree_parent.data_ptr())
        print(profile_one_decode(
            model, model.tree_draft_ids, model.tree_position_ids,
            model.cache_length, model.tree_attn_mask,
        ))

    # ── Timed steps ───────────────────────────────────────────────────────────
    print(f"  Timing {args.steps} steps ...")
    total_accepted = 0
    for i in range(args.steps):
        model.cache_length[0] = cache_len_start + args.warmup + i
        accept = timer.time_step(
            model,
            model.tree_draft_ids, model.tree_position_ids,
            model.cache_length, model.tree_attn_mask,
            model.tree_parent, model.tree_gt_ids,
        )
        total_accepted += accept
        model.tree_draft_ids[0] = model.tree_draft_ids[accept - 1]

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  llamacu {quant}  EAGLE-3  tree_size={args.tree_size}")
    print(f"{'═' * 72}")
    print(timer.summary())

    avg_accept = total_accepted / args.steps
    avg_step_ms = (sum(timer.draft_ms) + sum(timer.decode_ms) + sum(timer.fixup_ms)) / args.steps
    tok_s = avg_accept / (avg_step_ms / 1000) if avg_step_ms else 0
    print(f"\n  Acceptance: {avg_accept:.2f} tok/step  →  {tok_s:.1f} tok/s")

    # ── Amdahl projection ─────────────────────────────────────────────────────
    dm  = sum(timer.draft_ms)  / args.steps
    vm  = sum(timer.decode_ms) / args.steps

    print(f"\n  Amdahl: decode={vm:.2f}ms  draft={dm:.2f}ms")
    print(f"  Our ragged kernel targets attention within decode_ms.")
    # In FP16, attn ≈ 26% of transformer forward (from profile_verify.py).
    # With W4A16 MLP speedup, attn fraction rises to ~35%.
    attn_frac_fp16  = 0.26
    attn_frac_w4a16 = 0.35
    if args.fp16:
        attn_est = vm * attn_frac_fp16
        print(f"  FP16:  attn_est ≈ {attn_est:.2f}ms ({attn_frac_fp16:.0%} of decode)")
    else:
        attn_est = vm * attn_frac_w4a16
        print(f"  W4A16: attn_est ≈ {attn_est:.2f}ms ({attn_frac_w4a16:.0%} of decode)")
        print(f"  (MLP is now ~{1-attn_frac_w4a16:.0%} — less than FP16's ~74%)")

    # If ragged kernel saves 24% of attn (from profile_verify.py):
    ragged_attn_saving = attn_est * 0.24
    new_decode = vm - ragged_attn_saving
    new_total  = dm + new_decode + sum(timer.fixup_ms) / args.steps
    old_total  = dm + vm         + sum(timer.fixup_ms) / args.steps
    print(f"  Ragged 24% attn saving → saves {ragged_attn_saving:.2f}ms/step from decode")
    print(f"  E2E speedup projection: {old_total / new_total:.3f}×  "
          f"(draft floor = {dm:.2f}ms, fixup = {sum(timer.fixup_ms)/args.steps:.2f}ms)")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
