"""Shared data types spoken across every layer.

The RL environment, controllers, digital twin, API, and all backends exchange
only these types — no layer above ``backends/`` ever touches pandapower, PyPSA
or OpenDSS objects directly.

Plain dataclasses (not pydantic) on the hot path: a ``GridState`` is produced
every simulator tick, so construction cost matters for RL training throughput.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ControlAction:
    """One EMS decision for one tick, in physical units (MW).

    Sign convention: ``battery_p_mw > 0`` charges the battery, ``< 0``
    discharges it (matches :class:`~microgrid_simulator.components.battery.BatteryModel`).
    """

    battery_p_mw: float = 0.0
    ev_p_mw: list[float] = field(default_factory=list)  # >= 0, one per EV charger
    pv_curtail: float = 0.0  # fraction of available PV to spill, in [0, 1]
    diesel_on: bool = False
    diesel_setpoint_mw: float = 0.0

    def copy(self) -> ControlAction:
        return ControlAction(
            battery_p_mw=self.battery_p_mw,
            ev_p_mw=list(self.ev_p_mw),
            pv_curtail=self.pv_curtail,
            diesel_on=self.diesel_on,
            diesel_setpoint_mw=self.diesel_setpoint_mw,
        )


@dataclass
class ConstraintViolation:
    """One violated operational/electrical limit in a solved tick."""

    kind: str  # e.g. "voltage", "line_loading", "import_limit", "soc_band", "unserved"
    device: str = ""
    magnitude: float = 0.0  # how far past the limit, in the limit's own unit
    limit: float = 0.0
    message: str = ""


@dataclass
class DeviceStatus:
    """Uniform per-device snapshot for dashboards and logs."""

    name: str
    kind: str  # battery | diesel | pv | load | ev | grid
    p_mw: float = 0.0
    on: bool | None = None
    soc: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class GridState:
    """Full physical snapshot after one simulated tick.

    Every backend returns exactly this schema; fields a backend cannot compute
    stay at their defaults (e.g. the simple backend reports flat 1.0 pu
    voltages and no line loading).
    """

    v_bus: list[float] = field(default_factory=list)
    p_load: list[float] = field(default_factory=list)
    p_gen: list[float] = field(default_factory=list)
    soc: list[float] = field(default_factory=list)
    soh: list[float] = field(default_factory=list)
    ev_soc: list[float] = field(default_factory=list)
    line_loading: list[float] = field(default_factory=list)
    grid_import_mw: float = 0.0
    pv_available_mw: float = 0.0
    pv_used_mw: float = 0.0
    load_demand_mw: float = 0.0
    load_served_mw: float = 0.0
    unserved_mw: float = 0.0
    islanded: bool = False
    battery_p_mw: float = 0.0
    diesel_p_mw: float = 0.0
    diesel_on: bool = False
    demand_is_real: bool = False
    delta_soh: float = 0.0
    solver_ok: bool = True
    timestamp: float = 0.0
    violations: list[ConstraintViolation] = field(default_factory=list)

    @property
    def blackout(self) -> bool:
        """True when the island cannot serve all demand this tick."""
        return self.unserved_mw > 1e-6


@dataclass
class BackendResult:
    """A solved tick plus what the backend actually did with the request.

    ``applied_action`` may differ from the request after feasibility clamping
    (SoC headroom, ramp limits, min up/down lockouts, intertie limits) — the
    digital-twin replay and validation layers compare the two.
    """

    state: GridState
    requested_action: ControlAction
    applied_action: ControlAction
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    """Per-term reward decomposition (all values are the raw penalty magnitudes)."""

    carbon: float = 0.0
    autonomy: float = 0.0
    health: float = 0.0
    waste: float = 0.0
    unserved: float = 0.0
    constraint: float = 0.0
    total: float = 0.0

    def as_info(self) -> dict[str, float]:
        return {f"reward/{k}": v for k, v in asdict(self).items()}
