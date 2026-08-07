#!/usr/bin/env bash
# Auto-queue cached-forecast SAC retrains (w_unserved: 20.0) sequentially
# (1 at a time) to avoid CPU overheating. Runs seeds 0, 1, 2 in order.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h.yaml"
STEPS=1000000
ARTIFACT_DIR="artifacts/sac/cached-1m-v3"
SEEDS=(0 1 2)

echo "state=starting" > logs/queue_state.txt
echo "=== $(date) watcher started (w_unserved=20.0) ===" >> logs/queue_run.log

mkdir -p "$ARTIFACT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "state=launching_seed_${seed}" > logs/queue_state.txt
    echo "seed_${seed}_pid=pending" >> logs/queue_state.txt
    echo "=== $(date) launching seed ${seed} (1M steps) ===" >> logs/queue_run.log
    nohup uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode cached \
        --artifact-dir "$ARTIFACT_DIR" \
        > "logs/train_cached_v3_seed${seed}.log" 2>&1 &
    TRAIN_PID=$!
    echo "seed_${seed}_pid=${TRAIN_PID}" >> logs/queue_state.txt

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
    done

    if [ -f "${ARTIFACT_DIR}/sac_microgrid.zip" ]; then
        mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
        mv "${ARTIFACT_DIR}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
        mv "${ARTIFACT_DIR}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
        echo "=== $(date) seed ${seed} finished, artifact saved ===" >> logs/queue_run.log
    else
        echo "=== $(date) seed ${seed} FAILED: no artifact ===" >> logs/queue_run.log
    fi
done

echo "state=completed" > logs/queue_state.txt
echo "all seeds done" >> logs/queue_run.log