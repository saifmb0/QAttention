#!/bin/bash
# Paper-ready benchmark suite
# Four blocks, each run 3x for mean ± std reporting.
# Estimated total: ~3 hours on A10G.

set -uo pipefail

LOGDIR="logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/paper_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo "  Paper benchmark — $(date)"
echo "  Log: $LOG"
echo "======================================================"

if [[ "${CONDA_DEFAULT_ENV:-}" != "saif" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate saif
fi

cd "$(dirname "$0")"

run() {
    local label="$1"; shift
    echo ""
    echo "------------------------------------------------------"
    echo "  START: $label  [$(date +%H:%M:%S)]"
    echo "------------------------------------------------------"
    if "$@"; then
        echo "  DONE:  $label  [$(date +%H:%M:%S)]"
    else
        echo "  FAILED: $label (exit $?) — continuing"
    fi
}

# ══════════════════════════════════════════════════════════════════════
# BLOCK 1 — Micro kernel benchmark (3 independent runs)
# Covers: speedup vs b, d, B, L. Core kernel claim.
# ══════════════════════════════════════════════════════════════════════
for RUN in 1 2 3; do
run "Micro run${RUN}" \
python scripts/benchmark_micro.py \
    --batch-sizes 1,2,4,8 \
    --branching-factors 4,8,12,16,24,32 \
    --depths 7,12,16,24,32 \
    --prefix-lengths 0,1024,4096 \
    --warmup 20 \
    --iters 100 \
    --csv-name micro_r${RUN}.csv
done

# ══════════════════════════════════════════════════════════════════════
# BLOCK 2 — E2E branching sweep (3 seeds)
# d=16 fixed (safe), b=4..32. Shows kernel benefit grows with b.
# 20 prompts / 128 tokens stays within VRAM at all b values.
# ══════════════════════════════════════════════════════════════════════
for SEED in 42 123 456; do
run "E2E branch s${SEED}" \
python scripts/e2e_benchmark.py \
    --depths 16 \
    --branching-factors 4,6,8,10,12,16,20,24,32 \
    --num-prompts 20 \
    --max-new-tokens 128 \
    --context-lengths 0 \
    --prompt-seed ${SEED} \
    --csv-name e2e_branch_s${SEED}.csv
done

# ══════════════════════════════════════════════════════════════════════
# BLOCK 3 — E2E context sweep (3 seeds)
# d=16, b=10/16/24, L=0/1024. Shows L normalises gains.
# Skip L=4096 — OOMs E2E on this GPU.
# ══════════════════════════════════════════════════════════════════════
for SEED in 42 123 456; do
run "E2E context s${SEED}" \
python scripts/e2e_benchmark.py \
    --depths 16 \
    --branching-factors 10,16,24 \
    --num-prompts 15 \
    --max-new-tokens 128 \
    --context-lengths 0,1024 \
    --prompt-seed ${SEED} \
    --csv-name e2e_ctx_s${SEED}.csv
done

# ══════════════════════════════════════════════════════════════════════
# BLOCK 4 — Model Wall: acceptance rate vs depth (3 seeds)
# Minimal prompts/tokens to stay within VRAM at d=24/28.
# Both vanilla and ragged run to confirm acceptance parity.
# ══════════════════════════════════════════════════════════════════════
for SEED in 42 123 456; do
run "Model Wall s${SEED}" \
python scripts/e2e_benchmark.py \
    --depths 7,12,16,20,24,28 \
    --branching-factors 10 \
    --num-prompts 10 \
    --max-new-tokens 64 \
    --context-lengths 0 \
    --prompt-seed ${SEED} \
    --csv-name e2e_wall_s${SEED}.csv
done

echo ""
echo "======================================================"
echo "  All runs complete — $(date)"
echo "  Results in: results/"
echo "  Log:        $LOG"
echo "======================================================"
