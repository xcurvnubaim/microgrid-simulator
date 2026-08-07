#!/usr/bin/env bash
# Auto-queue no-forecast SAC ablation (3 seeds x 1M steps, forecast vector zeroed,
# forecast_mode=none, current reward w_unserved=20.0), sequentially.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h.yaml"
STEPS=1000000
ARTIFACT_DIR="artifacts/sac/noforecast-1m"
SEEDS=(0 1 2)

echo "state=starting" > logs/queue_noforecast_1m_state.txt
echo "=== $(date) watcher started (no-forecast SAC, 1M steps) ===" >> logs/queue_noforecast_1m_run.log

mkdir -p "$ARTIFACT_DIR"

for seed in "${SEEDS[@]}"; do
    echo "state=launching_seed_${seed}" > logs/queue_noforecast_1m_state.txt
    echo "seed_${seed}_pid=pending" >> logs/queue_noforecast_1m_state.txt
    echo "=== $(date) launching seed ${seed} (1M steps) ===" >> logs/queue_noforecast_1m_run.log

    uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode none \
        --artifact-dir "${ARTIFACT_DIR}/temp_seed_${seed}" \
        > "logs/train_noforecast_1m_seed${seed}.log" 2>&1

    if [ -f "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" ]; then
        mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
        mv "${ARTIFACT_DIR}/temp_seed_${seed}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
        rm -rf "${ARTIFACT_DIR}/temp_seed_${seed}"
        echo "=== $(date) seed ${seed} finished, artifact saved ===" >> logs/queue_noforecast_1m_run.log
    else
        echo "=== $(date) seed ${seed} FAILED: no artifact ===" >> logs/queue_noforecast_1m_run.log
    fi
done

echo "state=completed" > logs/queue_noforecast_1m_state.txt
echo "all seeds done" >> logs/queue_noforecast_1m_run.log