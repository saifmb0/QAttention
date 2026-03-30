#!/usr/bin/env bash
# run_all.sh – convenience wrapper used locally (see Kaggle commands in README section below)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== 1/3  Padding characterisation sweep ==="
python scripts/padding_sweep.py --out-dir results

echo ""
echo "=== 2/3  Correctness tests ==="
python -m pytest tests/test_correctness.py -v --tb=short

echo ""
echo "=== 3/4  Benchmark sweep ==="
python scripts/benchmark_sweep.py --out-dir results

echo ""
echo "=== 4/4  End-to-end tok/s benchmark (synthetic model) ==="
python scripts/e2e_benchmark.py \
    --model-size synthetic \
    --batch-sizes 1,2,4,8 \
    --depths 3,5,7 \
    --branching-factors 2,3 \
    --out-dir results

echo ""
echo "All steps complete. Results in: $ROOT/results/"
