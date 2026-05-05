#!/bin/bash

RUN=0
mkdir -p logs

while true; do
    RUN=$((RUN + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "========================================="
    echo "RUN $RUN — $TIMESTAMP"
    echo "========================================="

    echo "[RUN $RUN] Starting micro benchmark..."
    python scripts/benchmark_micro.py \
        --batch-sizes 1,2,4,8 \
        --branching-factors 6,10,16,24,32,36 \
        --depths 6,10,16,24,32,36 \
        --prefix-lengths 0,1024,4096,8192,16384 \
        --out-dir logs \
        --csv-name "micro_run${RUN}_${TIMESTAMP}.csv" \
        2>&1 | tee logs/micro_run${RUN}_${TIMESTAMP}.log

    echo "[RUN $RUN] Starting E2E benchmark..."
    python scripts/e2e_benchmark.py \
        --context-lengths 0,4096,8192 \
        --num-prompts 10 \
        --load-in-4bit \
        --branching-factors 6,16,24 \
        --depths 6,16,24 \
        2>&1 | tee logs/e2e_run${RUN}_${TIMESTAMP}.log

    echo "[RUN $RUN] Done. Sleeping 10s..."
    sleep 10
done
