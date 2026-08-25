#!/usr/bin/env python3
import json
from pathlib import Path

inc_path = Path("artifacts/agent-loop/e5/incumbent.json")
if inc_path.is_file():
    inc = json.loads(inc_path.read_text())
    print(f"=== Incumbent (Frozen E5 raw-F3) ===")
    print(f"  Mean unserved:  {inc['mean_unserved_kwh']:.2f} kWh/ep")
    print(f"  Worst episode:  {inc['worst_episode_unserved_kwh']:.2f} kWh")
    print(f"  Mean reward:    {inc['mean_reward']:.1f}")
    print()

print(f"{'It':2s} | {'Type':9s} | {'Parameter & Value':32s} | {'Status':8s} | {'Time':7s} | {'Avoidable':9s} | {'Gate Violations'}")
print("-" * 120)

for i in range(1, 6):
    p_file = Path(f"artifacts/agent-loop/e5/iteration-00{i}/proposal.json")
    o_file = Path(f"reports/agent-loop/e5/iteration-00{i}/outcome.json")
    if o_file.is_file():
        o = json.loads(o_file.read_text())
        status = o.get("status", "?")
        elapsed = o.get("elapsed_min", 0)
        reasons = o.get("gate_reasons", [])
        detail = o.get("detail", "")
        
        avoidable = "—"
        for r in reasons:
            if "avoidable_unserved=" in r:
                avoidable = r.split("=")[1] + " kWh"

        if p_file.is_file():
            p = json.loads(p_file.read_text())
            param = p["change"]["parameter"]
            val = p["change"]["new_value"]
            exp = p["experiment_type"]
            pv_str = f"{param}={val}"
        else:
            exp = "—"
            pv_str = detail

        print(f"{i:02d} | {exp:9s} | {pv_str:32s} | {status:8s} | {elapsed:5.1f}m | {avoidable:9s} | {', '.join(reasons) if reasons else detail}")
