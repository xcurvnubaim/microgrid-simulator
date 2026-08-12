"""Opt-in, deterministic uncertainty injection for robust (robust-SAC) fine-tuning.

Two distinct planes are modelled so that *plant truth* and *what the policy
observes* never collapse into one noisy vector:

* **Plant mismatch** — true battery/diesel parameters differ from the nominal
  values the policy was trained on. Sampled once per episode at ``reset()``,
  applied to a throwaway :class:`Settings` so the shared nominal object is never
  mutated.
* **Forecast-service uncertainty** — residuals on the perfect Chronos cached
  forecast plus stale/missing updates caused by service dropout.

All RNG is independent, per-environment, seed-derived and reproducible. A fully
disabled :class:`UncertaintyCfg` is an exact no-op so the nominal protocol is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from microgrid_simulator.config import Settings


@dataclass
class SampledUncertainty:
    """A per-episode draw of plant-mismatch factors and forecast profile."""

    # multiplicative factors (>0); 1.0 means unperturbed
    battery_capacity: float = 1.0
    battery_charge_eff: float = 1.0
    battery_discharge_eff: float = 1.0
    battery_max_charge: float = 1.0
    battery_max_discharge: float = 1.0
    battery_soc_init: float = 1.0
    diesel_max_kw: float = 1.0
    diesel_min_kw: float = 1.0
    diesel_ramp: float = 1.0
    forecast_pv_residual_std: float = 0.0  # already scaled by abs horizon
    forecast_demand_residual_std: float = 0.0
    forecast_dropout_until_step: int = 0  # suppressed refreshes until this step
    persisted_stale: bool = False  # previous snapshot intentionally kept
    extra: dict[str, Any] = field(default_factory=dict)

    def as_info(self) -> dict[str, Any]:
        return {
            "unc_battery_capacity": self.battery_capacity,
            "unc_battery_charge_eff": self.battery_charge_eff,
            "unc_battery_discharge_eff": self.battery_discharge_eff,
            "unc_battery_max_charge": self.battery_max_charge,
            "unc_battery_max_discharge": self.battery_max_discharge,
            "unc_battery_soc_init": self.battery_soc_init,
            "unc_diesel_max_kw": self.diesel_max_kw,
            "unc_diesel_min_kw": self.diesel_min_kw,
            "unc_diesel_ramp": self.diesel_ramp,
            "unc_forecast_pv_residual_std": self.forecast_pv_residual_std,
            "unc_forecast_demand_residual_std": self.forecast_demand_residual_std,
            "unc_forecast_dropout_until_step": self.forecast_dropout_until_step,
            **self.extra,
        }


@dataclass
class UncertaintySampler:
    """Samples per-episode plant-mismatch factors from a config-block profile.

    Each field's ``std`` is the scale of a zero-mean Gaussian on the log-factor,
    so factors stay strictly positive and are symmetric on a relative scale.
    ``std == 0`` leaves that factor at 1.0 exactly.
    """

    settings: Settings
    rng: np.random.Generator

    @classmethod
    def from_settings(cls, settings: Settings, seed: int) -> UncertaintySampler:
        return cls(settings, np.random.default_rng(seed))

    @property
    def cfg(self):
        return self.settings.uncertainty

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @staticmethod
    def _factor(std: float, rng: np.random.Generator) -> float:
        if std <= 0.0:
            return 1.0
        return float(np.exp(rng.normal(0.0, std)))

    def sample(self, episode_index: int = 0) -> SampledUncertainty:
        if not self.enabled:
            return SampledUncertainty()
        b = self.cfg.battery
        d = self.cfg.diesel
        f = self.cfg.forecast
        rng = self.rng
        return SampledUncertainty(
            battery_capacity=self._factor(b.capacity_std, rng),
            battery_charge_eff=self._factor(b.charge_eff_std, rng),
            battery_discharge_eff=self._factor(b.discharge_eff_std, rng),
            battery_max_charge=self._factor(b.max_charge_std, rng),
            battery_max_discharge=self._factor(b.max_discharge_std, rng),
            battery_soc_init=self._factor(b.soc_init_std, rng),
            diesel_max_kw=self._factor(d.max_kw_std, rng),
            diesel_min_kw=self._factor(d.min_kw_std, rng),
            diesel_ramp=self._factor(d.ramp_std, rng),
            forecast_pv_residual_std=f.residual_pv_std,
            forecast_demand_residual_std=f.residual_demand_std,
            forecast_dropout_until_step=0,
            persisted_stale=False,
        )


def apply_plant_mismatch(settings: Settings, sample: SampledUncertainty) -> Settings:
    """Return a *thumbnail* Settings copy with perturbed plant parameters applied.

    Only the battery/diesel fields the environment reads from ``settings`` at
    build time are modified. The nominal ``settings`` object is never mutated;
    callers must construct the backend/battery/diesel from the returned copy.
    """
    cfg = settings.model_copy(deep=True)
    b = cfg.battery
    d = cfg.diesel
    b.capacity_mwh *= sample.battery_capacity
    b.charge_eff *= sample.battery_charge_eff
    b.discharge_eff *= sample.battery_discharge_eff
    b.max_charge_mw *= sample.battery_max_charge
    b.max_discharge_mw *= sample.battery_max_discharge
    b.soc_init *= sample.battery_soc_init
    b.soc_init = float(np.clip(b.soc_init, b.soc_min, b.soc_max))
    if d.enabled:
        d.max_kw *= sample.diesel_max_kw
        d.min_kw *= sample.diesel_min_kw
        d.min_kw = min(d.min_kw, d.max_kw)
        d.ramp_kw_per_min *= sample.diesel_ramp
    return cfg
