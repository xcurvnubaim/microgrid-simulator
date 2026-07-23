"""Battery transition that follows native pymgrid internal-energy semantics."""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.config import BatteryCfg


@dataclass
class PymgridBatteryModel:
    """Stored-energy model with native limits, efficiency, and fixed SOH."""

    cfg: BatteryCfg
    current_charge_mwh: float
    soh: float = 1.0
    throughput_mwh: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: BatteryCfg) -> PymgridBatteryModel:
        if cfg.limit_basis != "internal_energy_per_step":
            raise ValueError("pymgrid battery requires internal_energy_per_step limits")
        if (
            cfg.max_charge_internal_mwh_per_step is None
            or cfg.max_discharge_internal_mwh_per_step is None
        ):
            raise ValueError("pymgrid battery requires both internal-energy limits")
        return cls(cfg=cfg, current_charge_mwh=cfg.soc_init * cfg.capacity_mwh)

    @property
    def soc(self) -> float:
        return self.current_charge_mwh / max(self.cfg.capacity_mwh, 1e-12)

    def reset(self) -> None:
        self.current_charge_mwh = self.cfg.soc_init * self.cfg.capacity_mwh
        self.soh = 1.0
        self.throughput_mwh = 0.0

    def usable_capacity_mwh(self) -> float:
        return self.cfg.capacity_mwh

    def clamp_power(self, p_mw: float, dt_hours: float = 1.0) -> float:
        dt = max(dt_hours, 1e-12)
        max_charge_internal = self.cfg.max_charge_internal_mwh_per_step or 0.0
        max_discharge_internal = self.cfg.max_discharge_internal_mwh_per_step or 0.0
        max_charge_terminal_mw = max_charge_internal / self.cfg.charge_eff / dt
        max_discharge_terminal_mw = max_discharge_internal * self.cfg.discharge_eff / dt
        return max(-max_discharge_terminal_mw, min(max_charge_terminal_mw, p_mw))

    def apply(self, p_mw: float, dt_hours: float) -> tuple[float, float]:
        """Apply project-sign terminal power through pymgrid's internal transition."""

        if dt_hours <= 0.0:
            raise ValueError("dt_hours must be positive")
        p_mw = self.clamp_power(p_mw, dt_hours)
        min_charge_mwh = self.cfg.soc_min * self.cfg.capacity_mwh
        max_charge_mwh = self.cfg.soc_max * self.cfg.capacity_mwh

        if p_mw >= 0.0:
            requested_internal_mwh = p_mw * dt_hours * self.cfg.charge_eff
            internal_change_mwh = min(
                requested_internal_mwh,
                self.cfg.max_charge_internal_mwh_per_step or 0.0,
                max(0.0, max_charge_mwh - self.current_charge_mwh),
            )
            applied_p_mw = internal_change_mwh / self.cfg.charge_eff / dt_hours
        else:
            requested_internal_mwh = -p_mw * dt_hours / self.cfg.discharge_eff
            internal_discharge_mwh = min(
                requested_internal_mwh,
                self.cfg.max_discharge_internal_mwh_per_step or 0.0,
                max(0.0, self.current_charge_mwh - min_charge_mwh),
            )
            internal_change_mwh = -internal_discharge_mwh
            applied_p_mw = -internal_discharge_mwh * self.cfg.discharge_eff / dt_hours

        self.current_charge_mwh += internal_change_mwh
        self.throughput_mwh += abs(applied_p_mw) * dt_hours
        return applied_p_mw, 0.0

    def degradation_step(self, p_mw: float, dt_hours: float) -> float:
        del p_mw, dt_hours
        return 0.0

    def soc_violation(self) -> float:
        return max(0.0, self.cfg.soc_min - self.soc) + max(0.0, self.soc - self.cfg.soc_max)
