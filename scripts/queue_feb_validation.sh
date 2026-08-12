#!/usr/bin/env bash
# Run the deterministic February validation suite across all 6 February windows
# for Rule, MPC, Cached-forecast SAC (3 seeds), No-forecast SAC (3 seeds), and optional Hard-unserved SAC (3 seeds).
set -u

cd /home/xcurv/teep-taiwan/microgrid-simulator

LOG_FILE="logs/feb_validation_run.log"
STATE_FILE="logs/feb_validation_state.txt"

echo "state=starting" > "$STATE_FILE"
echo "=== $(date) launching February validation suite ===" > "$LOG_FILE"

uv run python scripts/run_feb_validation_suite.py \
    --output-dir reports/experiments/feb_validation_2026-08-07 \
    --include-hardunserved \
    >> "$LOG_FILE" 2>&1

RC=$?

if [ $RC -eq 0 ]; then
    echo "state=completed" > "$STATE_FILE"
    echo "=== $(date) February validation suite COMPLETED successfully ===" >> "$LOG_FILE"
else
    echo "state=failed($RC)" > "$STATE_FILE"
    echo "=== $(date) February validation suite FAILED with return code $RC ===" >> "$LOG_FILE"
fi

exit $RC
