import sys
import os
import torch
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.weight_stream import patch_draft_model, unpatch_draft_model
from src.eagle_patches import patch_all, unpatch_all
from scripts.e2e_benchmark import load_eagle_model, _load_sharegpt_prompts, run_generation, TreeConfig, set_tree_config


# ─── L2 cache isolation ───────────────────────────────────────────────────────

def flush_l2_cache(device: str = 'cuda') -> None:
    """
    Evict all resident data from GPU L2 by writing and reading a buffer
    4× larger than L2.  Without this, sequential conditions share L2
    state, making weight-streaming comparisons invalid.

    RTX 4000 Ada: L2 = 41.9 MB → flush buf = 167.8 MB (fp16).
    """
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    n_elems = (l2_bytes * 4) // 2   # fp16 = 2 bytes
    buf = torch.empty(n_elems, dtype=torch.float16, device=device)
    # Write then read to guarantee eviction of prior L2 contents.
    buf.fill_(0.0)
    _ = buf.sum().item()
    del buf
    torch.cuda.synchronize(device)


# ─── Benchmark conditions ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle-model", default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--num-warmup", type=int, default=2,
                        help="Warmup prompts run before each condition (not timed).")
    args = parser.parse_args()

    print("Loading model...")
    model = load_eagle_model(
        base_model=args.base_model,
        eagle_model=args.eagle_model,
        use_eagle3=True,
        total_token=60,
        depth=7,
        top_k=10,
        max_length=2048,
        use_fp8=False,
    )

    prompts = _load_sharegpt_prompts(args.num_prompts + args.num_warmup, seed=42)
    warmup_prompts = prompts[:args.num_warmup]
    bench_prompts  = prompts[args.num_warmup:]

    config = TreeConfig(depth=16, total_token=103, top_k=10, label="d16")
    set_tree_config(model, config)

    original_draft = model.ea_layer
    draft_params_bytes = sum(p.numel() for p in original_draft.parameters()) * 2  # FP16

    results = {}

    # ── Condition 1: Vanilla baseline ─────────────────────────────────────────
    print("\n" + "="*60)
    print("=== CONDITION 1: Vanilla baseline (cuBLAS, no patch) ===")
    flush_l2_cache()
    _ = run_generation(model, warmup_prompts, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=False)
    flush_l2_cache()
    results["vanilla"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                        args.max_new_tokens, True, use_ragged=False)

    # ── Condition 2: Ragged attention only ────────────────────────────────────
    print("\n" + "="*60)
    print("=== CONDITION 2: Ragged attention only (no weight patch) ===")
    flush_l2_cache()
    _ = run_generation(model, warmup_prompts, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=True)
    flush_l2_cache()
    results["ragged"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                       args.max_new_tokens, True, use_ragged=True)

    # ── Condition 3: Early exit only ──────────────────────────────────────────
    print("\n" + "="*60)
    print("=== CONDITION 3: Early exit only (no weight patch, no ragged) ===")
    from src.eagle_patches import patch_early_exit, unpatch_early_exit
    patch_early_exit(original_draft, threshold_prob=0.05)
    flush_l2_cache()
    _ = run_generation(model, warmup_prompts, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=False)
    flush_l2_cache()
    try:
        results["early_exit"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                                args.max_new_tokens, True, use_ragged=False)
    except Exception as e:
        print(f"  ERROR: {e}")
    unpatch_early_exit(original_draft)
    model.ea_layer = original_draft

    # ── Condition 4: Weight streaming kernel only (no ragged, no compile) ─────
    # This isolates the Triton kernel effect (evict_first + coalesced access)
    print("\n" + "="*60)
    print("=== CONDITION 4: Weight stream kernel only (no compile, no ragged) ===")
    patch_draft_model(original_draft)
    flush_l2_cache()
    _ = run_generation(model, warmup_prompts, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=False)
    flush_l2_cache()
    try:
        results["weight_stream"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                                   args.max_new_tokens, True, use_ragged=False)
    except Exception as e:
        print(f"  ERROR: {e}")
    unpatch_draft_model(original_draft)
    model.ea_layer = original_draft

    # ── Condition 5: Full stack (ragged + weight stream + early exit + concurrent) ─
    print("\n" + "="*60)
    print("=== CONDITION 5: Full stack (no compile) ===")
    patch_all(original_draft, threshold_prob=0.05)
    flush_l2_cache()
    _ = run_generation(model, warmup_prompts, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=True)
    flush_l2_cache()
    try:
        results["full_stack"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                                args.max_new_tokens, True, use_ragged=True)
    except Exception as e:
        print(f"  ERROR: {e}")
    unpatch_all(original_draft)
    model.ea_layer = original_draft

    # ── Condition 6: Full stack + torch.compile ───────────────────────────────
    print("\n" + "="*60)
    print("=== CONDITION 6: Full stack + torch.compile ===")
    patch_all(original_draft, threshold_prob=0.05)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model.ea_layer = torch.compile(original_draft, mode="reduce-overhead", fullgraph=False)
    flush_l2_cache()
    compile_warmup = _load_sharegpt_prompts(args.num_warmup + 2, seed=99)
    print("  [Dynamo warmup...]")
    _ = run_generation(model, compile_warmup, "llama-3-instruct", args.max_new_tokens,
                       True, use_ragged=True)
    flush_l2_cache()
    try:
        results["full_stack_compiled"] = run_generation(model, bench_prompts, "llama-3-instruct",
                                                         args.max_new_tokens, True, use_ragged=True)
    except Exception as e:
        print(f"  ERROR: {e}")

    model.ea_layer = original_draft

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("--- Ablation Benchmark Results ---")
    print("="*60)
    print(f"{'Condition':<18s} | {'Tok/s':>6s} | {'Acc':>5s} | {'Est Draft BW':>14s} | {'vs vanilla':>10s}")
    print("-"*70)

    baseline_tok_s = None
    for name, r_list in results.items():
        if not r_list:
            print(f"{name:<18s} | {'ERROR':>6s}")
            continue
        avg_tok_s   = sum(r.tok_per_sec      for r in r_list) / len(r_list)
        avg_acc     = sum(r.acceptance_rate   for r in r_list) / len(r_list)
        avg_wall_ms = sum(r.wall_ms           for r in r_list) / len(r_list)
        avg_steps   = sum(r.num_steps         for r in r_list) / len(r_list)

        bw_gb_s = (draft_params_bytes * avg_steps) / (avg_wall_ms / 1000) / (1024**3)

        if name == "vanilla":
            baseline_tok_s = avg_tok_s

        vs = ""
        if baseline_tok_s and name != "vanilla":
            pct = (avg_tok_s / baseline_tok_s - 1) * 100
            vs = f"{pct:+.1f}%"

        print(f"{name:<18s} | {avg_tok_s:>6.1f} | {avg_acc:>5.2f} | {bw_gb_s:>12.1f} GB/s | {vs:>10s}")


if __name__ == "__main__":
    main()
