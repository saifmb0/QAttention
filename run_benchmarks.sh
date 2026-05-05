#!/usr/bin/env bash
# RaggedAttention — reproducible benchmark suite
#
# Usage:
#   bash run_benchmarks.sh [options] [exp...]
#
# Options:
#   --runs N        number of independent runs per experiment (default: 3)
#   --sleep S       seconds to sleep between runs for GPU cooling (default: 120)
#
# Experiments (default: all except correctness which runs once):
#   correctness     one-time correctness check
#   micro           kernel micro-benchmark vs FlashInfer
#   sequoia         Sequoia E2E sweep
#   eagle-l0        EAGLE-3 E2E at L=0
#   eagle-l4096     EAGLE-3 E2E at L=4096
#   amdahl          attention-fraction profiling (Amdahl table)
#
# Examples:
#   bash run_benchmarks.sh                        # 3× all
#   bash run_benchmarks.sh --runs 5 micro         # 5× micro only
#   bash run_benchmarks.sh correctness micro      # 1× correctness, 3× micro

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Prevent CUDA allocator fragmentation that causes OOM on large configs (b=24 d=24 tt=494)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Argument parsing ──────────────────────────────────────────────────────────
RUNS=3
SLEEP_S=120
EXPS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)  RUNS="$2";  shift 2 ;;
        --sleep) SLEEP_S="$2"; shift 2 ;;
        *)       EXPS+=("$1"); shift ;;
    esac
done

# Default experiment list (correctness runs once outside the loop)
if [[ ${#EXPS[@]} -eq 0 ]]; then
    EXPS=(micro sequoia eagle-l0 eagle-l4096 amdahl)
fi

want() { local e; for e in "${EXPS[@]}"; do [[ "$e" == "$1" ]] && return 0; done; return 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
log_banner() { echo ""; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }

aggregate_csvs() {
    # Merge all run CSVs into per-basename aggregates with mean ± std columns.
    # Groups run_1/foo.csv + run_2/foo.csv + run_3/foo.csv together by stable
    # basename (timestamps stripped), handles all-numeric CSVs (sequoia), and
    # multiple CSV schemas per run dir (eagle_e2e).
    local name="$1"
    local out_base="results/${name}"
    python3 - "$out_base" <<'PYEOF'
import sys, os, glob, csv, re
from collections import defaultdict
import numpy as np

base = sys.argv[1]
runs = sorted(glob.glob(os.path.join(base, "run_*")))
if not runs:
    print(f"  [aggregate] no runs found in {base}")
    sys.exit(0)

def stable_name(path):
    """Strip trailing _YYYYMMDD_HHMMSS timestamp from a CSV basename."""
    fname = os.path.basename(path)
    stem, ext = os.path.splitext(fname)
    stem = re.sub(r'_\d{8}_\d{6}$', '', stem)
    return stem + ext

def best_csv_per_basename(run_dir):
    """Return {stable_basename: path}, preferring 'pruned' over plain, and
    non-timestamped stable names over timestamped duplicates."""
    result = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.csv"))):
        sname = stable_name(path)
        if sname not in result or 'pruned' in os.path.basename(path):
            result[sname] = path
    return result

run_maps = [best_csv_per_basename(r) for r in runs]
all_basenames = sorted(set().union(*[m.keys() for m in run_maps]))

def is_num(val):
    try: float(val); return True
    except: return False

any_written = False
for basename in all_basenames:
    run_csvs = [m[basename] for m in run_maps if basename in m]
    if not run_csvs:
        continue

    all_rows = []
    for path in run_csvs:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            all_rows.append(rows)

    if not all_rows:
        continue

    fieldnames = list(all_rows[0][0].keys())

    numeric_cols = [k for k in fieldnames if all(
        is_num(row.get(k, "")) for rows in all_rows for row in rows
        if row.get(k, "") != ""
    )]
    key_cols = [k for k in fieldnames if k not in numeric_cols]

    # If every column looks numeric (e.g. sequoia label="4","8",...),
    # use the first column as a string key so rows don't all collapse.
    if not key_cols and fieldnames:
        key_cols = [fieldnames[0]]
        numeric_cols = [c for c in numeric_cols if c != fieldnames[0]]

    grouped = defaultdict(lambda: defaultdict(list))
    for rows in all_rows:
        for row in rows:
            key = tuple(row.get(k, "") for k in key_cols)
            for col in numeric_cols:
                val = row.get(col, "")
                if is_num(val):
                    grouped[key][col].append(float(val))

    stem = os.path.splitext(basename)[0]
    agg_path = os.path.join(base, f"{stem}_aggregate.csv")
    mean_std_cols = []
    for c in numeric_cols:
        mean_std_cols += [f"{c}_mean", f"{c}_std", f"{c}_cv_pct"]

    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(key_cols + mean_std_cols + ["n_runs"])
        for key, vals in grouped.items():
            row = list(key)
            for col in numeric_cols:
                v = np.array(vals[col], dtype=float)
                mean = float(np.mean(v))
                std  = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                cv   = (std / mean * 100) if mean != 0 else 0.0
                row += [f"{mean:.6g}", f"{std:.6g}", f"{cv:.2f}"]
            row.append(len(run_csvs))
            writer.writerow(row)

    print(f"  [aggregate] {basename}: {len(grouped)} configs × {len(run_csvs)} runs → {agg_path}")
    any_written = True

if not any_written:
    print(f"  [aggregate] no CSVs found under {base}")
PYEOF
}

# ── One-time correctness check ────────────────────────────────────────────────
if want correctness; then
    log_banner "correctness (1 run)"
    mkdir -p results/correctness/run_1
    python scripts/correctness_benchmark.py \
        --out-dir results/correctness/run_1 \
        --csv-name correctness_benchmark.csv \
        2>&1 | tee results/correctness/run_1/correctness.log
    echo "  → results/correctness/run_1/correctness_benchmark.csv"
fi

# ── Repeated experiments ──────────────────────────────────────────────────────
for RUN in $(seq 1 "$RUNS"); do
    log_banner "RUN $RUN / $RUNS"

    # ── micro-benchmark ───────────────────────────────────────────────────────
    if want micro; then
        echo "  [micro] run $RUN"
        mkdir -p "results/micro/run_${RUN}"
        python scripts/benchmark_micro.py \
            --warmup 5 --iters 50 \
            --out-dir "results/micro/run_${RUN}" \
            --csv-name "micro_benchmark.csv" \
            2>&1 | tee "results/micro/run_${RUN}/micro.log"
    fi

    # ── Sequoia E2E ───────────────────────────────────────────────────────────
    if want sequoia; then
        echo "  [sequoia] run $RUN"
        mkdir -p "results/sequoia/run_${RUN}"
        python scripts/benchmark_sequoia.py \
            --section e2e-sweep \
            --out-dir "results/sequoia/run_${RUN}" \
            2>&1 | tee "results/sequoia/run_${RUN}/sequoia.log"
    fi

    # ── EAGLE-3 E2E at L=0 ────────────────────────────────────────────────────
    if want eagle-l0; then
        echo "  [eagle-l0] run $RUN"
        mkdir -p "results/eagle_e2e/run_${RUN}"
        python scripts/e2e_benchmark.py \
            --section e2e \
            --branching-factors 6,16,24 \
            --depths 6,16,24 \
            --context-lengths 0 \
            --num-prompts 10 \
            --out-dir "results/eagle_e2e/run_${RUN}" \
            --csv-name "eagle_l0.csv" \
            2>&1 | tee "results/eagle_e2e/run_${RUN}/eagle_l0.log"
    fi

    # ── EAGLE-3 E2E at L=4096 ─────────────────────────────────────────────────
    if want eagle-l4096; then
        echo "  [eagle-l4096] run $RUN"
        mkdir -p "results/eagle_e2e/run_${RUN}"
        python scripts/e2e_benchmark.py \
            --section e2e \
            --branching-factors 6,16,24 \
            --depths 6,16,24 \
            --context-lengths 4096 \
            --num-prompts 10 \
            --out-dir "results/eagle_e2e/run_${RUN}" \
            --csv-name "eagle_l4096.csv" \
            2>&1 | tee "results/eagle_e2e/run_${RUN}/eagle_l4096.log"
    fi

    # ── Amdahl (attention fraction) ────────────────────────────────────────────
    if want amdahl; then
        echo "  [amdahl] run $RUN"
        mkdir -p "results/amdahl/run_${RUN}"
        python scripts/e2e_benchmark.py \
            --section e2e \
            --branching-factors 6,24 \
            --depths 6,24 \
            --context-lengths 0 \
            --num-prompts 10 \
            --attn-profile \
            --out-dir "results/amdahl/run_${RUN}" \
            --csv-name "amdahl.csv" \
            2>&1 | tee "results/amdahl/run_${RUN}/amdahl.log"
    fi

    # Sleep between runs to let GPU thermals settle (skip after last run)
    if [[ $RUN -lt $RUNS ]]; then
        echo "  sleeping ${SLEEP_S}s for GPU cooling…"
        sleep "$SLEEP_S"
    fi
done

# ── Aggregate results ─────────────────────────────────────────────────────────
log_banner "Aggregating results"
want micro      && aggregate_csvs "micro"
want sequoia    && aggregate_csvs "sequoia"
want eagle-l0   && aggregate_csvs "eagle_e2e"
want eagle-l4096 && aggregate_csvs "eagle_e2e"
want amdahl     && aggregate_csvs "amdahl"

echo ""
echo "Done. Results layout:"
echo "  results/"
echo "    micro/run_{1..N}/micro_benchmark_pruned.csv  + aggregate.csv"
echo "    sequoia/run_{1..N}/sequoia_size_*.csv        + aggregate.csv"
echo "    eagle_e2e/run_{1..N}/eagle_l{0,4096}.csv    + aggregate.csv"
echo "    amdahl/run_{1..N}/amdahl.csv                + aggregate.csv"
echo "    correctness/run_1/correctness_benchmark.csv"
