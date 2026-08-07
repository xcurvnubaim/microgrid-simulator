#!/usr/bin/env bash
# Auto-queue hard-unserved SAC v3 retrain (3 seeds x 500k steps,
# hard_unserved_penalty=50000.0, hard_unserved_tol_mw=0.002), sequentially.
# Decision 2026-08-07: 3 x 500k (user) instead of full 3 x 1M.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h-hardunserved.yaml"
STEPS=500000
ARTIFACT_DIR="artifacts/sac/hardunserved-500k-v3"
SEEDS=(0 1 2)

echo "state=starting" > logs/queue_hardunserved_500k_v3_state.txt
echo "=== $(date) watcher started (v3 500k steps, hard_unserved_penalty=50000) ===" >> logs/queue_hardunserved_500k_v3_run.log

mkdir -p "$ARTIFACT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "state=launching_seed_${seed}" > logs/queue_hardunserved_500k_v3_state.txt
    echo "seed_${seed}_pid=pending" >> logs/queue_hardunserved_500k_v3_state.txt
    echo "=== $(date) launching seed ${seed} (500k steps) ===" >> logs/queue_hardunserved_500k_v3_run.log

    uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode cached \
        --artifact-dir "${ARTIFACT_DIR}/temp_seed_${seed}" \
        > "logs/train_hardunserved_500k_v3_seed${seed}.log" 2>&1

    if [ -f "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" ]; then
        mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
        rm -rf "${ARTIFACT_DIR}/temp_seed_${seed}"
        echo "=== $(date) seed ${seed} finished, artifact saved ===" >> logs/queue_hardunserved_500k_v3_run.log
    else
        echo "=== $(date) seed ${seed} FAILED: no artifact ===" >> logs/queue_hardunserved_500k_v3_run.log
    fi
done

echo "state=completed" > logs/queue_hardunserved_500k_v3_state.txt
echo "all seeds done" >> logs/queue_hardunserved_500k_v3_run.log