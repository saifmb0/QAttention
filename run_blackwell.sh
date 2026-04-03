#!/usr/bin/env bash
# =============================================================================
# run_blackwell.sh
# Single entry-point for the sd-ragged "hopper" branch.
# Target: NVIDIA H100 / H200  (SM 9.0, Hopper)
#
# Stages (all on by default):
#   1 — smoke   : Python smoke test (kernel launches, shapes correct)
#   2 — test    : full pytest correctness suite (~2 min)
#   3 — bench   : SOTA benchmark sweep (all methods, ~15 min full / ~3 min fast)
#   4 — profile : roofline profiler for the ragged kernel
#   5 — e2e     : end-to-end tok/s benchmark (synthetic model)
#
# Usage:
#   bash run_blackwell.sh                     # full run (all stages)
#   bash run_blackwell.sh --fast              # small grid, fewer iters
#   bash run_blackwell.sh --skip-profile      # skip profiler
#   bash run_blackwell.sh --skip-e2e          # skip end-to-end tok/s benchmark
#   bash run_blackwell.sh --dtype bf16        # run benchmark in BF16
#   bash run_blackwell.sh --no-sota           # skip optional SOTA libs
#   bash run_blackwell.sh --out-dir /tmp/out  # custom output directory
# =============================================================================
set -euo pipefail

# ─── Colour helpers ───────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
section() { echo; echo -e "${CYAN}${BOLD}══ $* ══${NC}"; echo; }
ok()      { echo -e "${GREEN}✓  $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
fail()    { echo -e "${RED}✗  $*${NC}"; exit 1; }

# ─── Defaults ────────────────────────────────────────────────────────────────
FAST=0
SKIP_PROFILE=0
SKIP_E2E=0
SKIP_SOTA_LIBS=0
DTYPE="fp16"
OUT_DIR="results"

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)           FAST=1           ; shift ;;
        --skip-profile)   SKIP_PROFILE=1   ; shift ;;
        --skip-e2e)       SKIP_E2E=1       ; shift ;;
        --no-sota)        SKIP_SOTA_LIBS=1 ; shift ;;
        --dtype)          DTYPE="$2"       ; shift 2 ;;
        --out-dir)        OUT_DIR="$2"     ; shift 2 ;;
        *) warn "Unknown flag: $1" ; shift ;;
    esac
done

# ─── Environment ─────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

# ─── Header ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  sd-ragged · hopper branch                           ║"
echo "  ║  NVIDIA H100 / H200  (SM 9.0, Hopper)               ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

$PYTHON - <<'PYEOF'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU   : {p.name}")
    print(f"  SM    : {p.major}{p.minor}  ({p.multi_processor_count} SMs)")
    print(f"  VRAM  : {p.total_memory // 1024**3} GB")
    sm = (p.major, p.minor)
    if sm >= (9, 0) and sm < (12, 0):
        tier = 'SM90 configs active (Hopper)'
    elif sm >= (12, 0):
        tier = 'SM120 configs active (Blackwell — use blackwell branch)'
    elif sm >= (8, 9):
        tier = 'SM89 configs active (Lovelace)'
    else:
        tier = 'SM75 configs active (fallback)'
    print(f"  Config tier: {tier}")
else:
    print("  NO CUDA GPU DETECTED")
PYEOF

# ─── Stage 1 — Smoke test ────────────────────────────────────────────────────
section "Stage 1 / 5 — Smoke test"
$PYTHON -m src.ragged_attn && ok "Smoke test passed"

# ─── Stage 2 — Correctness (pytest) ─────────────────────────────────────────
section "Stage 2 / 5 — Correctness tests (pytest)"
pytest tests/ -v --timeout=120 && ok "All correctness tests passed"

# ─── Stage 3 — SOTA benchmark ────────────────────────────────────────────────
section "Stage 3 / 5 — SOTA benchmark"

SOTA_ARGS=(
    "--out-dir" "$OUT_DIR"
    "--dtype"   "$DTYPE"
)

if [[ $FAST -eq 1 ]]; then
    warn "Fast mode: reduced grid (B=1,8,32 · d=1,3,5 · warmup=5 · iters=20)"
    SOTA_ARGS+=(
        "--batch-sizes"       "1,8,32"
        "--depths"            "1,3,5"
        "--branching-factors" "2,4"
        "--warmup"            "5"
        "--iters"             "20"
    )
fi

if [[ $SKIP_SOTA_LIBS -eq 1 ]]; then
    SOTA_ARGS+=("--skip-flashinfer")
fi

$PYTHON scripts/benchmark_sota.py "${SOTA_ARGS[@]}" && ok "SOTA benchmark complete"

# ─── Stage 4 — Roofline profiler ─────────────────────────────────────────────
if [[ $SKIP_PROFILE -eq 0 ]]; then
    section "Stage 4 / 5 — Roofline profiler"
    $PYTHON scripts/profile_kernel.py --csv "$OUT_DIR/profile.csv" \
        && ok "Profiler complete" \
        || warn "Profiler failed or not available — skipping (use --skip-profile to suppress)"
else
    section "Stage 4 / 5 — Roofline profiler (SKIPPED)"
    warn "Passed --skip-profile; skipping roofline profiler."
fi

# ─── Stage 5 — End-to-end tok/s benchmark ────────────────────────────────────
if [[ $SKIP_E2E -eq 0 ]]; then
    section "Stage 5 / 5 — End-to-end tok/s benchmark"
    E2E_ARGS=("--out-dir" "$OUT_DIR")
    if [[ $FAST -eq 1 ]]; then
        E2E_ARGS+=(
            "--model-size"        "synthetic"
            "--batch-sizes"       "1,4"
            "--depths"            "3,5"
            "--branching-factors" "2"
            "--warmup"            "2"
            "--iters"             "5"
        )
    else
        E2E_ARGS+=("--model-size" "7b")
    fi
    $PYTHON scripts/e2e_benchmark.py "${E2E_ARGS[@]}" \
        && ok "E2E benchmark complete" \
        || warn "E2E benchmark failed (use --skip-e2e to suppress)"
else
    section "Stage 5 / 5 — End-to-end tok/s benchmark (SKIPPED)"
    warn "Passed --skip-e2e; skipping end-to-end benchmark."
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
section "Summary"
echo -e "  Output directory : ${BOLD}${OUT_DIR}/${NC}"
echo
ls -lh "$OUT_DIR"/ 2>/dev/null | grep -E "\.(csv|png)$" | awk '{print "  "$NF"  ("$5")"}' \
    || warn "No output files found in $OUT_DIR"
echo
ok "run_blackwell.sh complete."
