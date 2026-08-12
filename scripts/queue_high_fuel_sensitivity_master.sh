#!/usr/bin/env bash
# End-to-end high-fuel sensitivity: train three SAC seeds, then validate on February.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

echo "state=training" > logs/high_fuel_sensitivity_master_state.txt
bash scripts/queue_high_fuel_battery_pv_1m.sh
RC=$?
if [ $RC -ne 0 ]; then
    echo "state=failed_training" > logs/high_fuel_sensitivity_master_state.txt
    exit $RC
fi

echo "state=evaluating" > logs/high_fuel_sensitivity_master_state.txt
uv run python scripts/evaluate_high_fuel_battery_pv.py \
    > logs/evaluate_high_fuel_battery_pv.log 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "state=failed_evaluation" > logs/high_fuel_sensitivity_master_state.txt
    exit $RC
fi

echo "state=completed" > logs/high_fuel_sensitivity_master_state.txt
