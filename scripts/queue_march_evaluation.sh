#!/usr/bin/env bash
# Unattended Week 3 execution pipeline: March evaluation -> metrics -> figures -> report draft
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

LOG_FILE="logs/march_evaluation_run.log"
STATE_FILE="logs/march_evaluation_state.txt"

echo "state=starting" > "$STATE_FILE"
echo "=== $(date) launching March final evaluation pipeline ===" > "$LOG_FILE"

uv run python scripts/run_march_evaluation_suite.py \
    --output-dir reports/experiments/march_evaluation_2026-08-07 \
    >> "$LOG_FILE" 2>&1

RC=$?

if [ $RC -eq 0 ]; then
    echo "state=completed" > "$STATE_FILE"
    echo "=== $(date) March final evaluation pipeline COMPLETED successfully ===" >> "$LOG_FILE"
else
    echo "state=failed($RC)" > "$STATE_FILE"
    echo "=== $(date) March final evaluation pipeline FAILED with return code $RC ===" >> "$LOG_FILE"
fi

exit $RC
