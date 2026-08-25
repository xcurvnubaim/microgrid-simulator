#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

log_file="/tmp/f3-summary-pilot.log"
matrix_log="/tmp/f3-summary-pilot-matrix.log"

while true; do
    status=$(.venv/bin/python scripts/monitor_queue.py \
        --json f3_summary_pilot_jobs.json \
        --log "$log_file" \
        --format json)
    running=$(printf '%s' "$status" | .venv/bin/python -c \
        'import json, sys; print(json.load(sys.stdin)["counts"]["running"])')
    pending=$(printf '%s' "$status" | .venv/bin/python -c \
        'import json, sys; print(json.load(sys.stdin)["counts"]["pending"])')
    failed=$(printf '%s' "$status" | .venv/bin/python -c \
        'import json, sys; print(json.load(sys.stdin)["counts"]["failed"])')
    if [ "$failed" -gt 0 ]; then
        printf '%s\n' 'Summary pilot has failed jobs; matrix not launched.'
        exit 1
    fi
    if [ "$running" -eq 0 ] && [ "$pending" -eq 0 ]; then
        break
    fi
    date -Is
    printf '%s\n' "$status"
    sleep 300
done

if grep -q 'FAILED' "$log_file"; then
    printf '%s\n' 'Summary pilot stopped with failed training jobs; matrix not launched.'
    exit 1
fi

artifact_count=$(find artifacts/sac/f3-summary -name sac_microgrid.zip | wc -l)
if [ "$artifact_count" -ne 9 ]; then
    printf 'Expected 9 summary artifacts, found %s; matrix not launched.\n' "$artifact_count"
    exit 1
fi

printf '%s\n' 'All summary pilot artifacts verified; launching summary evaluation matrix.'
exec uv run python scripts/run_f3_comparison_matrix.py \
    --summary-pilot \
    --workers 9 >"$matrix_log" 2>&1
