#!/usr/bin/env python3
"""Thermal Watchdog & Training Progress Monitor for Module 6.

Monitors CPU/GPU temperatures. If CPU max temp exceeds THERMAL_LIMIT_C (default 85°C),
it pauses the training process (SIGSTOP), waits for temperatures to cool down to
COOL_DOWN_C (default 65°C), and then resumes training (SIGCONT).

Also tracks training timesteps and progress across seeds.
"""

from __future__ import annotations

import glob
import os
import re
import signal
import time
from pathlib import Path

THERMAL_LIMIT_C = 85.0
COOL_DOWN_C = 65.0
CHECK_INTERVAL_SEC = 10.0


def get_max_temp() -> float:
    temps = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(p) as f:
                val = float(f.read().strip()) / 1000.0
                temps.append(val)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def get_training_pids() -> list[int]:
    pids = []
    try:
        for p in Path("/proc").iterdir():
            if p.is_dir() and p.name.isdigit():
                cmd_file = p / "cmdline"
                if cmd_file.exists():
                    try:
                        cmd = cmd_file.read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
                        if "microgrid-sim train" in cmd or "train_high_fuel" in cmd:
                            pids.append(int(p.name))
                    except Exception:
                        pass
    except Exception:
        pass
    return pids


def get_training_progress() -> str:
    log_file = Path("logs/train_high_fuel_battery_pv_seed0.log")
    state_file = Path("logs/queue_high_fuel_battery_pv_state.txt")

    state = state_file.read_text().strip() if state_file.exists() else "unknown"

    for seed in (0, 1, 2):
        s_log = Path(f"logs/train_high_fuel_battery_pv_seed{seed}.log")
        if s_log.exists():
            text = s_log.read_text()
            matches = re.findall(r"total_timesteps\s*\|\s*(\d+)", text)
            if matches:
                current = int(matches[-1])
                pct = (current / 1_000_000) * 100
                return f"State: {state} | Seed {seed}: {current:,}/1,000,000 steps ({pct:.1f}%)"
    return f"State: {state} | Waiting for logs..."


def main() -> None:
    print(f"=== Thermal Watchdog Started (Limit: {THERMAL_LIMIT_C}°C, Cooldown: {COOL_DOWN_C}°C) ===")
    paused = False

    while True:
        try:
            max_t = get_max_temp()
            pids = get_training_pids()
            progress = get_training_progress()

            if max_t > THERMAL_LIMIT_C and not paused:
                print(f"\n🔥 HIGH TEMP DETECTED: {max_t:.1f}°C > {THERMAL_LIMIT_C}°C! Pausing training processes: {pids}")
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except Exception as e:
                        print(f"  Error pausing PID {pid}: {e}")
                paused = True

            elif paused and max_t <= COOL_DOWN_C:
                print(f"\n❄️ COOLED DOWN: {max_t:.1f}°C <= {COOL_DOWN_C}°C. Resuming training processes: {pids}")
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except Exception as e:
                        print(f"  Error resuming PID {pid}: {e}")
                paused = False

            status_str = f"[Temp: {max_t:.1f}°C | Status: {'PAUSED' if paused else 'RUNNING'}] {progress}"
            print(status_str, flush=True)

        except Exception as e:
            print(f"Watchdog error: {e}")

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
