"""PyPSAOperationalBackend — trusted operational scheduling via PyPSA.

Per-step physics are inherited from :class:`SimpleBackend` (identical numbers,
so an RL policy can be swapped between the two without re-tuning). What PyPSA
adds is :func:`optimize_dispatch` — a linopy unit-commitment/dispatch
optimization over a forecast horizon enforcing:

* diesel max/min output, ramp up/down, minimum stable generation,
* diesel minimum up time, minimum down time, startup/shutdown cost,
* battery charge/discharge power limits, energy capacity, efficiencies,
* grid import limit, PV availability and curtailment,
* load shedding only as a last (heavily priced) resort.

This is deliberately *not* run inside every RL step (a MILP per tick is far too
slow for training). Use it as the MPC baseline (:class:`PyPSAMPCController`),
an expert/teacher policy, an operational feasibility checker, or an offline
dispatch benchmark.

Objective costs mirror the reward function's fuel, carbon, and autonomy terms:
``grid = w_autonomy*import_price + w_carbon*grid_carbon*1000`` per MWh and
``diesel = fuel_cost + w_carbon*diesel_carbon*1000`` per MWh, so the MPC
optimum is a meaningful lower bound for the RL policy's operating cost.

Requires PyPSA, which ships with the default install (``pip install -e .``).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from microgrid_simulator.backends.simple_backend import SimpleBackend
from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import ControlAction

LOGGER = logging.getLogger(__name__)

def _require_pypsa() -> Any:
    try:
        import pypsa
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyPSA is not installed. Reinstall project dependencies with "
            "`pip install -e .` to use the PyPSA backend/MPC baseline."
        ) from exc
    return pypsa


def build_operational_network(
    settings: Settings,
    demand_mw: np.ndarray,
    pv_available_mw: np.ndarray,
    soc_init: float,
    diesel_on_init: bool = False,
) -> Any:
    """Build a single-bus PyPSA network for one dispatch horizon.

    The microgrid is treated as a copper plate (operational constraints, not
    electrical ones — validate the resulting dispatch with pandapower/OpenDSS).
    """
    pypsa = _require_pypsa()
    n_steps = len(demand_mw)
    dt = float(settings.topology.timestep_hours)
    reward = settings.reward

    net = pypsa.Network()
    net.set_snapshots(pd.RangeIndex(n_steps, name="step"))
    net.snapshot_weightings.loc[:, :] = dt

    net.add("Bus", "microgrid")
    net.add("Load", "demand", bus="microgrid", p_set=pd.Series(demand_mw, index=net.snapshots))

    # Penalizing curtailed PV is objective-equivalent (up to a constant) to
    # crediting every MWh of used PV by the configured per-kWh waste weight.
    pv_nom = max(sum(pv.p_mw for pv in settings.pv_arrays[: settings.topology.n_pv]), 1e-9)
    net.add(
        "Generator",
        "pv",
        bus="microgrid",
        p_nom=pv_nom,
        p_max_pu=pd.Series(np.clip(pv_available_mw / pv_nom, 0.0, 1.0), index=net.snapshots),
        marginal_cost=-reward.w_waste * 1000.0,
    )

    # Grid intertie (only when the topology has a utility bus).
    grid_connected = any(b.role.lower() in {"grid", "slack", "utility"} for b in settings.buses)
    if grid_connected:
        import_limit = settings.intertie.max_import_mw
        p_nom_grid = (
            import_limit if import_limit is not None else max(float(np.max(demand_mw)) * 2.0, 1.0)
        )
        grid_cost = (
            reward.w_autonomy * reward.import_price
            + reward.w_carbon * reward.grid_carbon_kg_per_kwh * 1000.0
        )
        net.add("Generator", "grid", bus="microgrid", p_nom=p_nom_grid, marginal_cost=grid_cost)

    # Diesel genset with unit-commitment constraints.
    d = settings.diesel
    if d.enabled:
        p_nom = d.max_kw / 1000.0
        ramp_mw_per_h = math.inf if d.ramp_kw_per_min <= 0 else d.ramp_kw_per_min * 60.0 / 1000.0
        ramp_pu = min(1.0, ramp_mw_per_h * dt / p_nom) if math.isfinite(ramp_mw_per_h) else 1.0
        committable = bool(settings.backend.unit_commitment)
        net.add(
            "Generator",
            "diesel",
            bus="microgrid",
            p_nom=p_nom,
            committable=committable,
            p_min_pu=min(d.min_kw, d.max_kw) / max(d.max_kw, 1e-9) if committable else 0.0,
            ramp_limit_up=ramp_pu,
            ramp_limit_down=ramp_pu,
            min_up_time=(
                max(1, round(d.min_up_time_min / 60.0 / dt))
                if committable and d.min_up_time_min > 0.0
                else 0
            ),
            min_down_time=(
                max(1, round(d.min_down_time_min / 60.0 / dt))
                if committable and d.min_down_time_min > 0.0
                else 0
            ),
            start_up_cost=reward.diesel_start_cost,
            shut_down_cost=d.shut_down_cost,
            up_time_before=1 if diesel_on_init else 0,
            marginal_cost=(
                reward.diesel_fuel_cost_per_kwh * 1000.0
                + reward.w_carbon * d.carbon_kg_per_kwh * 1000.0
            ),
        )

    # Battery: usable energy window [soc_min, soc_max] mapped to [0, usable].
    b = settings.battery
    p_nom_batt = max(b.max_charge_mw, b.max_discharge_mw, 1e-9)
    usable_mwh = max(b.capacity_mwh * (b.soc_max - b.soc_min), 1e-9)
    soc0_mwh = float(np.clip((soc_init - b.soc_min) * b.capacity_mwh, 0.0, usable_mwh))
    net.add(
        "StorageUnit",
        "battery",
        bus="microgrid",
        p_nom=p_nom_batt,
        p_max_pu=b.max_discharge_mw / p_nom_batt,
        p_min_pu=-b.max_charge_mw / p_nom_batt,
        max_hours=usable_mwh / p_nom_batt,
        efficiency_store=b.charge_eff,
        efficiency_dispatch=b.discharge_eff,
        state_of_charge_initial=soc0_mwh,
        cyclic_state_of_charge=False,
    )

    # Load shedding keeps islanded / import-limited horizons feasible. The
    # simulator penalty is w_unserved per unserved kWh.
    net.add(
        "Generator",
        "shed",
        bus="microgrid",
        p_nom=max(float(np.max(demand_mw)) * 2.0, 1.0),
        marginal_cost=reward.w_unserved * 1000.0,
    )
    return net


def _add_common_reward_objective(net: Any, snapshots: Any) -> None:
    """Add battery wear and physically exclusive charge/discharge operation.

    The simulator's baseline degradation is ``2e-4 * MWh / capacity_mwh`` and
    reward scales delta-SOH by 1e4, yielding ``2 / capacity_mwh`` per MWh
    before ``w_health``. Its nonlinear SOC-edge stress multiplier is not
    represented in this linear MPC approximation.
    """

    settings: Settings = net.meta["microgrid_settings"]
    p_dispatch = net.model.variables["StorageUnit-p_dispatch"].loc[:, "battery"]
    p_store = net.model.variables["StorageUnit-p_store"].loc[:, "battery"]

    # StorageUnit otherwise permits simultaneous charge/discharge. A binary
    # mode matches the controller's one signed battery action per tick and
    # prevents artificial loss-cycling to collect the PV-use credit.
    mode = net.model.add_variables(
        binary=True,
        coords=[snapshots],
        name="battery-discharge-mode",
    )
    net.model.add_constraints(
        p_dispatch <= settings.battery.max_discharge_mw * mode,
        name="battery-dispatch-mode-upper",
    )
    net.model.add_constraints(
        p_store <= settings.battery.max_charge_mw * (1.0 - mode),
        name="battery-store-mode-upper",
    )

    wear_per_mwh = settings.reward.w_health * 2.0 / max(
        settings.battery.capacity_mwh, 1e-9
    )
    if wear_per_mwh <= 0.0:
        return
    dt = float(settings.topology.timestep_hours)
    net.model.objective += wear_per_mwh * dt * (p_dispatch + p_store).sum()


def optimize_dispatch(
    settings: Settings,
    demand_mw: np.ndarray,
    pv_available_mw: np.ndarray,
    soc_init: float,
    diesel_on_init: bool = False,
    solver_name: str | None = None,
) -> pd.DataFrame:
    """Solve one horizon and return the optimal dispatch per step.

    Columns: ``demand_mw, pv_used_mw, diesel_p_mw, diesel_on, battery_p_mw``
    (positive = charging, matching :class:`ControlAction`), ``grid_import_mw``,
    ``shed_mw``, ``soc`` (fraction of nameplate capacity, within the SoC band).
    """
    demand_mw = np.asarray(demand_mw, dtype=float)
    pv_available_mw = np.asarray(pv_available_mw, dtype=float)
    if len(demand_mw) != len(pv_available_mw):
        raise ValueError("demand and PV forecasts must have the same length")

    net = build_operational_network(settings, demand_mw, pv_available_mw, soc_init, diesel_on_init)
    net.meta["microgrid_settings"] = settings
    status, condition = net.optimize(
        solver_name=solver_name or settings.backend.solver,
        log_to_console=False,
        extra_functionality=_add_common_reward_objective,
    )
    if status != "ok":  # pragma: no cover - solver dependent
        raise RuntimeError(f"PyPSA dispatch optimization failed: {status} / {condition}")

    b = settings.battery
    gen = net.generators_t.p
    zeros = pd.Series(0.0, index=net.snapshots)
    diesel_p = gen.get("diesel", zeros)
    if "diesel" in net.generators_t.status:
        diesel_on = net.generators_t.status["diesel"].astype(bool)
    else:
        diesel_on = diesel_p > 1e-6
    soc_mwh = net.storage_units_t.state_of_charge["battery"]
    return pd.DataFrame(
        {
            "demand_mw": demand_mw,
            "pv_used_mw": gen["pv"].to_numpy(),
            "diesel_p_mw": diesel_p.to_numpy(),
            "diesel_on": diesel_on.to_numpy(),
            # PyPSA StorageUnit p > 0 means discharging into the bus.
            "battery_p_mw": -net.storage_units_t.p["battery"].to_numpy(),
            "grid_import_mw": gen.get("grid", zeros).to_numpy(),
            "shed_mw": gen.get("shed", zeros).to_numpy(),
            "soc": (soc_mwh / max(b.capacity_mwh, 1e-9) + b.soc_min).to_numpy(),
        }
    )


def plan_to_actions(plan: pd.DataFrame) -> list[ControlAction]:
    """Convert an optimized dispatch into per-step control actions."""
    actions: list[ControlAction] = []
    for _, row in plan.iterrows():
        actions.append(
            ControlAction(
                battery_p_mw=float(row["battery_p_mw"]),
                pv_curtail=0.0,  # backends spill surplus PV implicitly
                diesel_on=bool(row["diesel_on"]),
                diesel_setpoint_mw=float(row["diesel_p_mw"]),
            )
        )
    return actions


class PyPSAOperationalBackend(SimpleBackend):
    """Backend whose stepping physics match ``SimpleBackend`` and which exposes
    PyPSA rolling-horizon optimization for MPC baselines and feasibility checks."""

    def __init__(self, settings: Settings) -> None:
        _require_pypsa()  # fail fast with the install hint
        super().__init__(settings)

    def horizon_steps(self) -> int:
        return max(1, round(self.settings.backend.horizon_hours / self.dt))

    def rolling_steps(self) -> int:
        return max(1, round(self.settings.backend.rolling_horizon_hours / self.dt))

    def advance_window(self) -> None:
        """Advance the demand/PV window position by one control tick.

        Called by an MPC controller each time it consumes one plan step so the
        next re-plan reads the telemetry window from the correct offset.
        """
        self.demand.advance()
        self.pv.advance()

    def forecast(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Perfect-foresight forecast from the backend's own demand/PV models."""
        demand = np.empty(n_steps)
        pv = np.empty(n_steps)
        pos0 = self.demand.window_pos
        pv_pos0 = self.pv.window_pos
        for k in range(n_steps):
            t = self.timestamp + (k + 1) * self.dt
            self.demand.window_pos = pos0 + k + 1
            self.pv.window_pos = pv_pos0 + k + 1
            demand[k] = self.demand.total_demand_mw(t)
            pv[k] = self.pv.available_mw(t)
        self.demand.window_pos = pos0
        self.pv.window_pos = pv_pos0
        return demand, pv

    def optimize_horizon(
        self,
        n_steps: int | None = None,
        soc_init: float | None = None,
        diesel_on_init: bool | None = None,
        demand_mw: np.ndarray | None = None,
        pv_mw: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Optimize dispatch for the next ``n_steps`` from the current state.

        ``soc_init`` / ``diesel_on_init`` default to this backend's own (un-stepped)
        values; pass the live environment state when the MPC plays against a
        separately-stepped env so each re-plan starts from the true battery/diesel
        condition rather than a stale reset value.

        ``demand_mw`` / ``pv_mw`` override the perfect-foresight forecast with an
        explicit horizon (e.g. a leakage-free cached forecast). When omitted, the
        backend's own perfect-foresight telemetry forecast is used.
        """
        n = n_steps or self.horizon_steps()
        if demand_mw is None or pv_mw is None:
            demand_mw, pv_mw = self.forecast(n)
        return optimize_dispatch(
            self.settings,
            demand_mw,
            pv_mw,
            soc_init=(
                self.battery.soc
                if soc_init is None
                else float(
                    np.clip(
                        soc_init,
                        self.settings.battery.soc_min,
                        self.settings.battery.soc_max,
                    )
                )
            ),
            diesel_on_init=(
                self.diesel.is_on if diesel_on_init is None else bool(diesel_on_init)
            ),
        )
