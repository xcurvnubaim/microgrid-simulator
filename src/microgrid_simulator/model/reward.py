"""The five-term reward from the architecture diagram's Reward / Punish box.

    reward = -(
        w_carbon   * carbon_emission
      + w_autonomy * grid_import_penalty
      + w_health   * battery_degradation_penalty   # from PINN
      + w_waste    * curtailed_solar_penalty
      + w_excess   * dumped_overgeneration_penalty
      + w_unserved * unserved_load_penalty
      + constraint_penalty                          # hard SoC / voltage / solver
    )

Reward is always <= 0; a controller that minimises import, avoids peaks, wastes
no solar, and keeps the battery healthy earns near-zero.
"""

from __future__ import annotations

from microgrid_simulator.components.battery import BatteryLike
from microgrid_simulator.config import RewardCfg
from microgrid_simulator.core.types import GridState, RewardBreakdown

__all__ = ["RewardBreakdown", "compute_reward"]


def compute_reward(
    state: GridState,
    battery: BatteryLike,
    cfg: RewardCfg,
    dt_hours: float,
) -> tuple[float, RewardBreakdown]:
    """Return ``(reward, breakdown)`` for one tick.

    ``state`` is the post-solve physical snapshot; ``battery`` gives access to
    SoC-band violations for the hard constraint term.
    """
    if not state.solver_ok:
        # Infeasible grid: strong fixed penalty so the agent avoids it.
        b = RewardBreakdown(constraint=1000.0, total=-1000.0)
        return b.total, b

    if cfg.mode == "pymgrid":
        return _compute_pymgrid_reward(state, battery, cfg, dt_hours)

    import_mw = max(0.0, state.grid_import_mw)
    import_kwh = import_mw * dt_hours * 1000.0

    # 1. Carbon emission — grid carbon intensity x kWh imported, plus the
    #    diesel term from the plan: diesel_kw * carbon_per_kwh_diesel * dt.
    diesel_kwh = max(0.0, state.diesel_p_mw) * dt_hours * 1000.0
    carbon = cfg.grid_carbon_kg_per_kwh * import_kwh
    carbon += cfg.diesel_carbon_kg_per_kwh * diesel_kwh

    # 2. External-energy / autonomy penalty — economic cost of import + peak.
    autonomy = import_mw * dt_hours * cfg.import_price
    autonomy += max(0.0, import_mw - cfg.peak_threshold_mw) * cfg.peak_penalty

    # 3. Battery health (PINN placeholder) — degradation this tick, scaled up
    #    so SoH loss (~1e-4) is commensurate with the other terms.
    health = state.delta_soh * 1.0e4

    # 4. Energy waste — curtailed / spilled solar that had nowhere to go.
    wasted_mw = max(0.0, state.pv_available_mw - state.pv_used_mw)
    waste = wasted_mw * dt_hours * 1000.0  # kWh of spilled solar

    # 5. Non-exportable overgeneration routed to the dump load (kWh).
    excess = max(0.0, state.excess_generation_mw) * dt_hours * 1000.0

    # 6. Energy insufficient — unserved load (kWh).
    unserved_mw = max(0.0, state.load_demand_mw - state.load_served_mw)
    unserved = unserved_mw * dt_hours * 1000.0

    # Hard constraints: voltage band + SoC band violations.
    voltage_excursion = sum(max(0.0, 0.95 - v) + max(0.0, v - 1.05) for v in state.v_bus)
    constraint = voltage_excursion * cfg.voltage_penalty
    constraint += battery.soc_violation() * cfg.soc_violation_penalty

    breakdown = RewardBreakdown(
        carbon=cfg.w_carbon * carbon,
        autonomy=cfg.w_autonomy * autonomy,
        health=cfg.w_health * health,
        waste=cfg.w_waste * waste,
        excess=cfg.w_excess * excess,
        unserved=cfg.w_unserved * unserved,
        constraint=constraint,
    )
    breakdown.total = -(
        breakdown.carbon
        + breakdown.autonomy
        + breakdown.health
        + breakdown.waste
        + breakdown.excess
        + breakdown.unserved
        + breakdown.constraint
    )
    return breakdown.total, breakdown


def _compute_pymgrid_reward(
    state: GridState,
    battery: BatteryLike,
    cfg: RewardCfg,
    dt_hours: float,
) -> tuple[float, RewardBreakdown]:
    """Reproduce pymgrid's additive native module costs in canonical units."""

    battery_terminal_mwh = abs(state.battery_p_mw) * dt_hours
    if state.battery_p_mw >= 0.0:
        battery_internal_kwh = battery_terminal_mwh * battery.cfg.charge_eff * 1000.0
    else:
        battery_internal_kwh = battery_terminal_mwh / battery.cfg.discharge_eff * 1000.0

    diesel_kwh = max(0.0, state.diesel_p_mw) * dt_hours * 1000.0
    diesel_marginal_cost = cfg.pymgrid_genset_cost + (
        cfg.pymgrid_co2_per_unit * cfg.pymgrid_cost_per_unit_co2
    )
    unserved_kwh = max(0.0, state.unserved_mw) * dt_hours * 1000.0

    supply_mw = (
        state.pv_used_mw
        + state.diesel_p_mw
        + max(0.0, -state.battery_p_mw)
        + max(0.0, state.grid_import_mw)
        + state.unserved_mw
    )
    sinks_mw = state.load_demand_mw + max(0.0, state.battery_p_mw) + max(0.0, -state.grid_import_mw)
    excess_kwh = max(0.0, supply_mw - sinks_mw) * dt_hours * 1000.0

    battery_cost = battery_internal_kwh * cfg.pymgrid_battery_cost_cycle
    diesel_cost = diesel_kwh * diesel_marginal_cost
    unserved_cost = unserved_kwh * cfg.pymgrid_loss_load_cost
    excess_cost = excess_kwh * cfg.pymgrid_overgeneration_cost
    breakdown = RewardBreakdown(
        carbon=diesel_cost,
        autonomy=0.0,
        health=battery_cost,
        waste=0.0,
        excess=excess_cost,
        unserved=unserved_cost,
        constraint=0.0,
    )
    breakdown.total = -(battery_cost + diesel_cost + unserved_cost + excess_cost)
    return breakdown.total, breakdown
