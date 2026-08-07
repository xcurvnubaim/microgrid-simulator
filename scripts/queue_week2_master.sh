#!/usr/bin/env bash
# Week 2 queue master: run the remaining trainings sequentially so only one
# SAC run trains at a time (avoids CPU overheating; GPU is shared because envs run
# on CPU and only the policy updates hit CUDA).
# Order: hardunserved-500k-v3 (~45 min) -> noforecast-1m (~90 min).
# Note 2026-08-07: oracle-1m ablation was removed from the queue/plan (user decision);
# the oracle forecast_mode implementation and its regression test remain in code.
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

echo "master=starting $(date)" > logs/queue_week2_master_state.txt

run_script() {
    local script="$1"
    echo "master=run $script $(date)" >> logs/queue_week2_master_state.txt
    bash "$script"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "master=ok $script $(date)" >> logs/queue_week2_master_state.txt
        return 0
    fi
    echo "master=failed($rc) $script $(date)" >> logs/queue_week2_master_state.txt
    return $rc
}

run_script scripts/queue_hardunserved_500k_v3.sh
HARD_RC=$?

# Diagnostics first: hard-unserved retrained to a valid 500k artifact is a stress
# variant; proceed only if it completed (or report), then continue with ablations.
run_script scripts/queue_noforecast_1m.sh

echo "state=completed" > logs/queue_week2_master_state.txt
echo "queue master done (hardunserved rc=$HARD_RC)" >> logs/queue_week2_master_state.txt