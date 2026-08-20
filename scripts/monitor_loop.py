#!/usr/bin/env python3
"""Local monitor loop for remote ma012 training queue.

Queries ma012 every 10 seconds and prints status. Exits cleanly on Ctrl+C.
"""

from __future__ import annotations

import subprocess
import time
import sys


def main() -> None:
    print("Starting remote training monitor loop (Press Ctrl+C to exit)...")
    time.sleep(1.0)
    try:
        while True:
            # Clear terminal screen for a clean, updating view
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()

            res = subprocess.run(
                [
                    "ssh",
                    "ma012@100.117.159.47",
                    "bash -lc 'cd ~/projects/microgrid-simulator && uv run scripts/monitor_queue.py --json f3_jobs_e1_hard.json'",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            sys.stdout.write(res.stdout)
            if res.stderr.strip():
                sys.stderr.write(res.stderr)
            sys.stdout.flush()
            time.sleep(10)
    except KeyboardInterrupt:
        sys.stdout.write("\nMonitor loop exited.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
