#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$script_dir/.." && pwd -P)
config_arg="configs/docker-islanded-72h.yaml"
mode="ems"
policy="rule"
artifact_arg=""
seed=0
dry_run=false
interactive=false
mode_explicit=false
policy_explicit=false
config_explicit=false
active_run_id=""

usage() {
  cat <<'EOF'
Usage: scripts/run_ems_scenario.sh [OPTIONS]

Launch either the normal interactive simulator or an EMS-owned scenario.

Options:
  --mode <name>      normal or ems (interactive menu when attached to a terminal)
  --config <yaml>    Scenario below configs/ (default: configs/docker-islanded-72h.yaml)
  --policy <name>    rule, sac, or ppo; EMS mode only
  --artifact <zip>   Learned-policy file below artifacts/; required for SAC/PPO
  --seed <integer>   EMS run seed (default: 0)
  --interactive      Force directory-backed menus
  --dry-run          Validate and print the resolved launch without using Docker
  -h, --help         Show this help
EOF
}

fail() { printf 'error: %s\n' "$*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --mode) (($# >= 2)) || fail "--mode requires normal or ems"; mode=$2; mode_explicit=true; shift 2 ;;
    --config) (($# >= 2)) || fail "--config requires a YAML path"; config_arg=$2; config_explicit=true; shift 2 ;;
    --policy) (($# >= 2)) || fail "--policy requires rule, sac, or ppo"; policy=$2; policy_explicit=true; shift 2 ;;
    --artifact) (($# >= 2)) || fail "--artifact requires a .zip path"; artifact_arg=$2; shift 2 ;;
    --seed) (($# >= 2)) || fail "--seed requires an integer"; seed=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --interactive) interactive=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ "$seed" =~ ^-?[0-9]+$ ]] || fail "--seed must be an integer"

choose_number() {
  local prompt=$1 maximum=$2 answer
  while true; do
    printf '%s' "$prompt" >&2
    read -r answer || fail "interactive selection cancelled"
    if [[ "$answer" =~ ^[0-9]+$ ]] && ((answer >= 1 && answer <= maximum)); then
      printf '%s' "$answer"; return
    fi
    printf 'Please enter a number from 1 to %d.\n' "$maximum" >&2
  done
}

choose_mode() {
  printf 'Runtime mode:\n  1) normal — dashboard controls an integrated simulation\n  2) ems — CLI EMS controls a headless plant; dashboard is view-only\n'
  case "$(choose_number 'Select mode: ' 2)" in
    1) mode=normal ;;
    2) mode=ems ;;
  esac
}

choose_config() {
  local -a configs=()
  local item index candidate
  while IFS= read -r -d '' candidate; do
    if [[ "$mode" == normal ]] || awk '/^[[:space:]]*file:[[:space:]]*\/data\// { count += 1 } END { exit(count >= 2 ? 0 : 1) }' "$candidate"; then
      configs+=("$candidate")
    fi
  done < <(find "$repo_dir/configs" -type f \( -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z)
  ((${#configs[@]})) || fail "no scenarios found below configs/"
  printf 'Available %s scenarios (discovered from configs/):\n' "$mode"
  for index in "${!configs[@]}"; do printf '  %d) %s\n' "$((index + 1))" "${configs[$index]#"$repo_dir/"}"; done
  item=$(choose_number 'Select scenario: ' "${#configs[@]}")
  config_arg=${configs[$((item - 1))]}
}

choose_policy() {
  local -a policies=(rule) artifacts=()
  local candidate index selected count
  for candidate in sac ppo; do
    if find "$repo_dir/artifacts/$candidate" -type f -name '*.zip' -print -quit 2>/dev/null | grep -q .; then
      policies+=("$candidate")
    fi
  done
  printf 'Available EMS policies (discovered from artifacts/):\n'
  for index in "${!policies[@]}"; do
    if [[ "${policies[$index]}" == rule ]]; then
      printf '  %d) rule (no artifact)\n' "$((index + 1))"
    else
      count=$(find "$repo_dir/artifacts/${policies[$index]}" -type f -name '*.zip' | wc -l)
      printf '  %d) %s (%d artifacts)\n' "$((index + 1))" "${policies[$index]}" "$count"
    fi
  done
  selected=$(choose_number 'Select policy: ' "${#policies[@]}")
  policy=${policies[$((selected - 1))]}
  [[ "$policy" == rule ]] && { artifact_arg=""; return; }
  mapfile -d '' -t artifacts < <(find "$repo_dir/artifacts/$policy" -type f -name '*.zip' -print0 | sort -z)
  printf 'Available %s artifacts:\n' "${policy^^}"
  for index in "${!artifacts[@]}"; do printf '  %d) %s\n' "$((index + 1))" "${artifacts[$index]#"$repo_dir/"}"; done
  selected=$(choose_number 'Select artifact: ' "${#artifacts[@]}")
  artifact_arg=${artifacts[$((selected - 1))]}
}

if [[ "$interactive" == true ]] || { [[ "$mode_explicit" == false && -t 0 && -t 1 ]]; }; then
  [[ "$mode_explicit" == true ]] || choose_mode
  printf '\n'
  [[ "$config_explicit" == true ]] || choose_config
  if [[ "$mode" == ems ]]; then printf '\n'; [[ "$policy_explicit" == true ]] || choose_policy; fi
fi

case "$mode" in normal|ems) ;; *) fail "invalid mode '$mode'; choose normal or ems" ;; esac
case "$policy" in rule|sac|ppo) ;; *) fail "invalid policy '$policy'; choose rule, sac, or ppo" ;; esac

case "$config_arg" in /*) config_candidate=$config_arg ;; *) config_candidate=$repo_dir/$config_arg ;; esac
[[ -f "$config_candidate" ]] || fail "config file not found: $config_arg"
config_host=$(realpath -e -- "$config_candidate")
case "$config_host" in "$repo_dir/configs/"*) ;; *) fail "config must reside under $repo_dir/configs" ;; esac
case "$config_host" in *.yaml|*.yml) ;; *) fail "config must be YAML" ;; esac

if [[ "$mode" == ems ]] && ! awk '/^[[:space:]]*file:[[:space:]]*\/data\// { count += 1 } END { exit(count >= 2 ? 0 : 1) }' "$config_host"; then
  fail "EMS scenarios require load and PV telemetry files below /data/"
fi

artifact_host=""; artifact_container=""
if [[ "$mode" == normal ]]; then
  [[ -z "$artifact_arg" ]] || fail "--artifact is only valid in EMS mode"
else
  if [[ "$policy" == rule ]]; then
    [[ -z "$artifact_arg" ]] || fail "--artifact is only valid with SAC/PPO"
  else
    [[ -n "$artifact_arg" ]] || fail "--artifact is required with --policy $policy"
    case "$artifact_arg" in /*) artifact_candidate=$artifact_arg ;; *) artifact_candidate=$repo_dir/$artifact_arg ;; esac
    [[ -f "$artifact_candidate" ]] || fail "artifact file not found: $artifact_arg"
    artifact_host=$(realpath -e -- "$artifact_candidate")
    case "$artifact_host" in "$repo_dir/artifacts/"*) ;; *) fail "artifact must reside under $repo_dir/artifacts" ;; esac
    [[ "$artifact_host" == *.zip ]] || fail "artifact must be a .zip file"
    artifact_container=/app/artifacts/$(realpath --relative-to="$repo_dir/artifacts" -- "$artifact_host")
  fi
fi

config_container=/app/configs/$(realpath --relative-to="$repo_dir/configs" -- "$config_host")
printf 'Scenario launch\n  mode:     %s\n  config:   %s -> %s\n' "$mode" "$config_host" "$config_container"
if [[ "$mode" == ems ]]; then
  printf '  policy:   %s\n' "$policy"
  if [[ -n "$artifact_host" ]]; then
    printf '  artifact: %s -> %s\n' "$artifact_host" "$artifact_container"
  else
    printf '  artifact: none\n'
  fi
  printf '  seed:     %s\n' "$seed"
fi
printf '  profile:  %s\n' "$mode"
[[ "$dry_run" == true ]] && exit 0

export MGS_CONFIG=$config_container MGS_EMS_POLICY=$policy MGS_EMS_ARTIFACT=$artifact_container

ems_call() {
  local path=$1 body=${2:-}
  if [[ -n "$body" ]]; then
    curl -fsS -X POST "http://127.0.0.1:${EMS_CONTROL_PORT:-8004}$path" -H 'Content-Type: application/json' -d "$body"
  else
    curl -fsS -X POST "http://127.0.0.1:${EMS_CONTROL_PORT:-8004}$path"
  fi
}
cleanup() {
  docker compose --project-directory "$repo_dir" --profile normal --profile ems down --remove-orphans
}
cancel_and_exit() {
  if [[ -n "$active_run_id" ]]; then
    printf '\nRequesting safe cancellation at the next tick boundary...\n' >&2
    ems_call "/runs/$active_run_id/cancel" '{}' >/dev/null 2>&1 || true
    for _ in $(seq 1 100); do
      cancel_state=$(curl -fsS "http://127.0.0.1:${EMS_CONTROL_PORT:-8004}/runs/$active_run_id" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || true)
      [[ "$cancel_state" == cancelled || "$cancel_state" == failed ]] && break
      sleep 0.1
    done
  fi
  exit 130
}
trap cleanup EXIT
trap cancel_and_exit INT TERM

cd -- "$repo_dir"
# Remove containers created by the pre-profile launcher as well. Volumes are
# intentionally retained because neither cleanup path uses `-v`.
docker compose -p microgrid-ems --profile normal --profile ems down --remove-orphans
docker compose --profile normal --profile ems down --remove-orphans

if [[ "$mode" == normal ]]; then
  printf 'Dashboard: http://localhost:${EMS_DASHBOARD_PORT:-8501}\n'
  docker compose --profile normal up --build
  exit 0
fi

docker compose --profile ems up --build -d
printf 'Waiting for EMS control plane'
ems_ready=false
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${EMS_CONTROL_PORT:-8004}/ready" >/dev/null 2>&1; then ems_ready=true; break; fi
  printf '.'; sleep 1
done
[[ "$ems_ready" == true ]] || fail "EMS control plane did not become ready"
printf '\nDashboard (view-only): http://localhost:${EMS_DASHBOARD_PORT:-8501}\n'
start_json=$(ems_call /runs "{\"seed\":$seed}") || fail "could not start EMS run"
active_run_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])' <<<"$start_json")

while true; do
  status_json=$(curl -fsS "http://127.0.0.1:${EMS_CONTROL_PORT:-8004}/runs/$active_run_id")
  read -r state episode steps expected error_text < <(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["state"], d["episode_index"]+1, d["steps"], d["expected_steps"], repr(d.get("error")))' <<<"$status_json")
  printf '\rEMS run %s · episode %s · %s/%s ticks   ' "$state" "$episode" "$steps" "$expected"
  case "$state" in
    initializing|running|cancelling) sleep 1 ;;
    awaiting_continuation)
      printf '\nEpisode %s complete. Continue with the next shifted 72-hour window? [y/N] ' "$episode"
      read -r answer || answer=n
      if [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        ems_call "/runs/$active_run_id/continue" '{}' >/dev/null
      else
        ems_call "/runs/$active_run_id/finish" '{}' >/dev/null
        printf 'Run completed.\n'
        active_run_id=""
        break
      fi
      ;;
    failed) printf '\nEMS run failed: %s\n' "$error_text" >&2; exit 1 ;;
    cancelled) printf '\nEMS run cancelled.\n'; active_run_id=""; break ;;
    completed) printf '\nEMS run completed.\n'; active_run_id=""; break ;;
  esac
done
