#!/usr/bin/env bash
# Auto-queue cached-forecast SAC retrains sequentially (1 at a time) to avoid
# CPU overheating. Waits for the already-running seed 0, then runs seeds 1 and 2
# one at a time. One seed at a time keeps CPU load low (4 pandapower subprocesses).
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h.yaml"
STEPS=1000000
ARTIFACT_DIR="artifacts/sac/cached-1m-v2"
SEED0_PID="${SEED0_PID:-299804}"

echo "state=waiting_seed0" > logs/queue_state.txt
echo "=== $(date) watcher started, waiting for running seed 0 (pid ${SEED0_PID}) ===" >> logs/queue_run.log

# Wait for the already-running seed 0 to finish.
if kill -0 "$SEED0_PID" 2>/dev/null; then
    while kill -0 "$SEED0_PID" 2>/dev/null; do
        sleep 60
    done
fi
echo "state=seed0_done" > logs/queue_state.txt
echo "=== $(date) seed 0 finished ===" >> logs/queue_run.log

# Seed 0's artifact was written to the artifact dir by the CLI; move it into a
# seed-specific subdir so the final artifacts are distinguishable.
if [ -f "${ARTIFACT_DIR}/sac_microgrid.zip" ]; then
    mkdir -p "${ARTIFACT_DIR}/seed-0"
    mv "${ARTIFACT_DIR}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-0/sac_microgrid.zip"
    mv "${ARTIFACT_DIR}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-0/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
fi

# Now run seeds 1 and 2 sequentially.
for seed in 1 2; do
    echo "state=launching_seed_${seed}" > logs/queue_state.txt
    echo "=== $(date) launching seed ${seed} (1M steps) ===" >> logs/queue_run.log
    nohup uv run microgrid-sim train \
        --config "$CONFIG" \
        --timesteps "$STEPS" \
        --seed "$seed" \
        --forecast-mode cached \
        --artifact-dir "$ARTIFACT_DIR" \
        > "logs/train_cached_seed${seed}.log" 2>&1 &
    TRAIN_PID=$!
    echo "seed_${seed}_pid=${TRAIN_PID}" >> logs/queue_state.txt

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 60
    done

    # Guard: if the run crashed before writing the artifact, retry that seed once.
    if [ ! -f "${ARTIFACT_DIR}/sac_microgrid.zip" ]; then
        echo "=== $(date) seed ${seed} produced no artifact, retrying once ===" >> logs/queue_run.log
        nohup uv run microgrid-sim train \
            --config "$CONFIG" \
            --timesteps "$STEPS" \
            --seed "$seed" \
            --forecast-mode cached \
            --artifact-dir "${ARTIFACT_DIR}-tmp" \
            > "logs/train_cached_seed${seed}_retry.log" 2>&1 &
        RETRY_PID=$!
        while kill -0 "$RETRY_PID" 2>/dev/null; do sleep 60; done
        if [ -f "${ARTIFACT_DIR}-tmp/sac_microgrid.zip" ]; then
            mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
            mv "${ARTIFACT_DIR}-tmp/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
            mv "${ARTIFACT_DIR}-tmp/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
        fi
    fi

    echo "=== $(date) seed ${seed} finished ===" >> logs/queue_run.log
    if [ -f "${ARTIFACT_DIR}/sac_microgrid.zip" ]; then
        mkdir -p "${ARTIFACT_DIR}/seed-${seed}"
        mv "${ARTIFACT_DIR}/sac_microgrid.zip" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid.zip"
        mv "${ARTIFACT_DIR}/sac_microgrid_vecnormalize.pkl" "${ARTIFACT_DIR}/seed-${seed}/sac_microgrid_vecnormalize.pkl" 2>/dev/null || true
    fi
done

echo "state=completed" > logs/queue_state.txt
echo "all seeds done" >> logs/queue_run.log