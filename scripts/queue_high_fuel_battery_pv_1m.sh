#!/usr/bin/env bash
# Train the high-fuel battery/PV-utilization SAC sensitivity (3 seeds x 1M).
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-high-fuel-battery-pv-72h.yaml"
STEPS=1000000
ARTIFACT_DIR="artifacts/sac/high-fuel-battery-pv-1m"
SEEDS=(0 1 2)

echo "state=starting" > logs/queue_high_fuel_battery_pv_state.txt
echo "=== $(date) high-fuel sensitivity started ===" > logs/queue_high_fuel_battery_pv_run.log
mkdir -p "$ARTIFACT_DIR"

for seed in "${SEEDS[@]}"; do
    FINAL_DIR="${ARTIFACT_DIR}/seed-${seed}"
    TEMP_DIR="${ARTIFACT_DIR}/temp-seed-${seed}"
    if [ -f "${FINAL_DIR}/sac_microgrid.zip" ]; then
        echo "=== $(date) seed ${seed} already complete; skipping ===" >> logs/queue_high_fuel_battery_pv_run.log
        continue
    fi

    echo "state=training_seed_${seed}" > logs/queue_high_fuel_battery_pv_state.txt
    echo "=== $(date) launching seed ${seed} ===" >> logs/queue_high_fuel_battery_pv_run.log
    uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode cached \
        --artifact-dir "$TEMP_DIR" \
        > "logs/train_high_fuel_battery_pv_seed${seed}.log" 2>&1
    RC=$?

    if [ $RC -ne 0 ] || [ ! -f "${TEMP_DIR}/sac_microgrid.zip" ]; then
        echo "state=failed_seed_${seed}" > logs/queue_high_fuel_battery_pv_state.txt
        echo "=== $(date) seed ${seed} failed (rc=${RC}) ===" >> logs/queue_high_fuel_battery_pv_run.log
        exit 1
    fi

    mkdir -p "$FINAL_DIR"
    mv "${TEMP_DIR}/sac_microgrid.zip" "${FINAL_DIR}/sac_microgrid.zip"
    if [ -f "${TEMP_DIR}/sac_microgrid_vecnormalize.pkl" ]; then
        mv "${TEMP_DIR}/sac_microgrid_vecnormalize.pkl" "${FINAL_DIR}/sac_microgrid_vecnormalize.pkl"
    fi
    rmdir "$TEMP_DIR"
    echo "=== $(date) seed ${seed} complete ===" >> logs/queue_high_fuel_battery_pv_run.log
done

echo "state=completed" > logs/queue_high_fuel_battery_pv_state.txt
echo "=== $(date) all seeds complete ===" >> logs/queue_high_fuel_battery_pv_run.log
