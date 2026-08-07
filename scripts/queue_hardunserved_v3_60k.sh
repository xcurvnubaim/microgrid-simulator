#!/usr/bin/env bash
# Auto-queue cached-forecast SAC training for the hard-unserved-load islanded
# scenario v3 (3 seeds x 60k steps, hard_unserved_penalty=50000.0, hard_unserved_tol_mw=0.002), sequentially.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h-hardunserved.yaml"
STEPS=60000
ARTIFACT_DIR="artifacts/sac/hardunserved-v3-60k"
SEEDS=(0 1 2)

echo "state=starting" > logs/queue_v3_60k_state.txt
echo "=== $(date) watcher started (v3 60k steps, hard_unserved_penalty=50000) ===" >> logs/queue_v3_60k_run.log

mkdir -p "$ARTIFACT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "state=launching_seed_${seed}" > logs/queue_v3_60k_state.txt
    echo "seed_${seed}_pid=pending" >> logs/queue_v3_60k_state.txt
    echo "=== $(date) launching seed ${seed} (60k steps) ===" >> logs/queue_v3_60k_run.log
    
    # Run training (uses default config or explicit args)
    uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode cached \
        --artifact-dir "${ARTIFACT_DIR}/temp_seed_${seed}" \
        > "logs/train_hardunserved_v3_60k_seed${seed}.log" 2>&1
    
    if [ -f "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" ]; then
        mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
        rm -rf "${ARTIFACT_DIR}/temp_seed_${seed}"
        echo "=== $(date) seed ${seed} finished, artifact saved ===" >> logs/queue_v3_60k_run.log
    else
        echo "=== $(date) seed ${seed} FAILED: no artifact ===" >> logs/queue_v3_60k_run.log
    fi
done

echo "state=completed" > logs/queue_v3_60k_state.txt
echo "all seeds done" >> logs/queue_v3_60k_run.log
