# Held-out pandapower scheduling-state check

- Full AC solves: **288**; all converged: **true**
- Maximum scheduling-field delta from the simple backend: **2.13e-13**
- Voltage range: **0.983312–1.005066 pu**
- Maximum line loading: **25.987%**
- Voltage / line-loading violation intervals: **0 / 0**

## Selected states

| Labels | Timestamp | Load kW | PV kW | Diesel kW | Battery kW | SOC % | Unserved kW | V min | V max | Line % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| representative_load | 2026-03-08 07:15:00 | 73.600 | 10.818 | 74.968 | 0.000 | 50.000 | 0.000 | 0.992995 | 1.000000 | 10.910 |
| peak_pv | 2026-03-08 10:30:00 | 74.800 | 131.181 | 0.000 | 10.427 | 50.501 | 0.000 | 0.992880 | 1.004769 | 12.243 |
| peak_load, peak_unserved | 2026-03-09 07:30:00 | 204.400 | 3.707 | 76.229 | 0.000 | 52.356 | 124.464 | 0.992387 | 1.000208 | 11.857 |
| worst_voltage, peak_line_loading | 2026-03-09 07:45:00 | 173.600 | 8.376 | 144.016 | -21.209 | 51.251 | 0.000 | 0.983312 | 1.001059 | 25.987 |
| minimum_soc | 2026-03-10 19:30:00 | 153.600 | 0.000 | 150.000 | -3.600 | 49.210 | 0.000 | 0.985264 | 1.000180 | 22.947 |

## Claim boundary

The configured 0.4 kV buses, balanced lines, load split, and reactive power are
assumed. The islanded slack is a numerical grid-forming reference. These results
test implementation consistency and sensitivity only; they do not validate the
physical campus feeder, losses, voltage, phase behavior, or equipment ratings.
