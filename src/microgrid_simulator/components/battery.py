"""Battery state-of-charge (SoC) physics — the empirical baseline.

Implements the transition equation from the build guide:

    soc_next = soc_now
             + charge_eff    * charge_kw   * dt / capacity_kwh
             - discharge_kw  * dt / (discharge_eff * capacity_kwh)

Sign convention (matches the action schema): ``p_mw > 0`` charges the battery,
``p_mw < 0`` discharges it.

A ``degradation`` hook returns an empirical per-step State-of-Health loss that
stands in for the ``dSoH = physics + NN_theta`` PINN residual described in the
research brainstorm. Swap ``degradation_step`` for a learned model later without
touching the rest of the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.config import BatteryCfg


@dataclass
class BatteryModel:
    """Single battery unit with efficiency-aware SoC integration."""

    cfg: BatteryCfg
    soc: float = 0.5
    soh: float = 1.0  # state of health in [0, 1]; 1.0 = fresh
    throughput_mwh: float = 0.0  # cumulative energy cycled (for degradation)

    @classmethod
    def from_cfg(cls, cfg: BatteryCfg) -> BatteryModel:
        return cls(cfg=cfg, soc=cfg.soc_init, soh=1.0, throughput_mwh=0.0)

    def reset(self) -> None:
        self.soc = self.cfg.soc_init
        self.soh = 1.0
        self.throughput_mwh = 0.0

    def clamp_power(self, p_mw: float) -> float:
        """Clamp requested power to charge/discharge limits."""
        return max(-self.cfg.max_discharge_mw, min(self.cfg.max_charge_mw, p_mw))

    def usable_capacity_mwh(self) -> float:
        """Capacity faded by state of health."""
        return self.cfg.capacity_mwh * self.soh

    def apply(self, p_mw: float, dt_hours: float) -> tuple[float, float]:
        """Advance SoC by one tick.

        Returns ``(applied_p_mw, delta_soh)`` where ``applied_p_mw`` is the power
        actually realised after clamping to power *and* SoC headroom limits.
        """
        p_mw = self.clamp_power(p_mw)
        capacity = max(self.usable_capacity_mwh(), 1e-9)

        if p_mw >= 0.0:  # charging
            delta_soc = self.cfg.charge_eff * p_mw * dt_hours / capacity
        else:  # discharging (p_mw < 0)
            delta_soc = p_mw * dt_hours / (self.cfg.discharge_eff * capacity)

        new_soc = self.soc + delta_soc
        # Clamp to operating band; back out the power actually delivered.
        clamped = min(self.cfg.soc_max, max(self.cfg.soc_min, new_soc))
        realised_delta_soc = clamped - self.soc
        self.soc = clamped

        # Recover the true realised power from the clamped SoC change.
        if realised_delta_soc >= 0.0:
            applied_p_mw = realised_delta_soc * capacity / (self.cfg.charge_eff * dt_hours)
        else:
            applied_p_mw = realised_delta_soc * capacity * self.cfg.discharge_eff / dt_hours

        delta_soh = self.degradation_step(applied_p_mw, dt_hours)
        self.soh = max(0.0, self.soh - delta_soh)
        self.throughput_mwh += abs(applied_p_mw) * dt_hours
        return applied_p_mw, delta_soh

    def degradation_step(self, p_mw: float, dt_hours: float) -> float:
        """Empirical State-of-Health loss for this tick.

        Baseline: degradation scales with energy throughput and is amplified at
        SoC extremes (deep discharge / high charge stress). This is the
        ``physics`` term; add the learned ``NN_theta`` residual on top later.
        """
        energy_mwh = abs(p_mw) * dt_hours
        # cycle-throughput wear: ~0.02% SoH per full-capacity MWh cycled
        cycle_wear = 2.0e-4 * energy_mwh / max(self.cfg.capacity_mwh, 1e-9)
        # stress multiplier: worse near the SoC band edges
        stress = 1.0 + 2.0 * max(
            0.0,
            abs(self.soc - 0.5) - 0.3,
        )
        return cycle_wear * stress

    def soc_violation(self) -> float:
        """Magnitude of SoC excursion outside the operating band (0 if inside)."""
        return max(0.0, self.cfg.soc_min - self.soc) + max(0.0, self.soc - self.cfg.soc_max)
