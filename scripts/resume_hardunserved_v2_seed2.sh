#!/usr/bin/env bash
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

CONFIG="configs/islanded-baseline-72h-hardunserved.yaml"
STEPS=1000000
ARTIFACT_DIR="artifacts/sac/hardunserved-1m-v2"
seed=2

echo "state=launching_seed_${seed}" > logs/queue_state.txt
echo "seed_${seed}_pid=pending" >> logs/queue_state.txt
echo "=== $(date) RESUMING seed ${seed} (1M steps) ===" >> logs/queue_run.log

nohup uv run microgrid-sim train \
    --config "$CONFIG" \
    --timesteps "$STEPS" \
    --seed "$seed" \
    --forecast-mode cached \
    --artifact-dir "$ARTIFACT_DIR" \
    > "logs/train_hardunserved_v2_seed${seed}.log" 2>&1 &
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

echo "state=completed" > logs/queue_state.txt
echo "all seeds done" >> logs/queue_run.log
