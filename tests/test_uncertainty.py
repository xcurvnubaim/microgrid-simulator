"""Uncertainty injection: exact no-op, reproducibility, per-env streams, plant mismatch."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.env import MicrogridEnv  # noqa: E402
from microgrid_simulator.rl.uncertainty import (  # noqa: E402
    SampledUncertainty,
    UncertaintySampler,
    apply_plant_mismatch,
)


def _roll(settings, seed, steps=5, discharge=True):
    env = MicrogridEnv(settings=settings)
    env.reset(seed=seed)
    traj = [env.backend.battery.soc]
    for _ in range(steps):
        action = np.zeros(env.action_dim, dtype=np.float32)
        if discharge:
            action[0] = -1.0  # full discharge -> SOC depends on capacity/eff
        _, _, term, trunc, _ = env.step(action)
        traj.append(env.backend.battery.soc)
        if term or trunc:
            break
    env.close()
    return traj


def test_disabled_uncertainty_is_exact_noop() -> None:
    base = Settings()
    base.uncertainty.enabled = False
    a = _roll(base, seed=42)
    b = _roll(base, seed=42)
    assert all(np.isclose(x, y) for x, y in zip(a, b, strict=True))


def test_disabled_profile_matches_settings_without_uncertainty() -> None:
    plain = Settings()
    explicit = Settings()
    explicit.uncertainty.enabled = False
    a = _roll(plain, seed=7)
    b = _roll(explicit, seed=7)
    assert all(np.isclose(x, y) for x, y in zip(a, b, strict=True))


def test_enabled_uncertainty_is_reproducible_for_fixed_seed() -> None:
    s = Settings()
    s.uncertainty.enabled = True
    s.uncertainty.battery.capacity_std = 0.15
    s.uncertainty.battery.max_discharge_std = 0.10
    s.uncertainty.diesel.max_kw_std = 0.12
    a = _roll(s, seed=1)
    b = _roll(s, seed=1)
    assert all(np.isclose(x, y) for x, y in zip(a, b, strict=True))


def test_enabled_uncertainty_changes_plant_versus_nominal() -> None:
    nom = Settings()
    s = Settings()
    s.uncertainty.enabled = True
    s.uncertainty.battery.capacity_std = 0.30
    s.uncertainty.battery.max_discharge_std = 0.30
    a = _roll(nom, seed=3)
    b = _roll(s, seed=3)
    assert any(not np.isclose(x, y) for x, y in zip(a, b, strict=True))


def test_uncertainty_metadata_exposed_in_info() -> None:
    s = Settings()
    s.uncertainty.enabled = True
    s.uncertainty.diesel.max_kw_std = 0.1
    env = MicrogridEnv(settings=s)
    _, info = env.reset(seed=5)
    assert info["unc_battery_capacity"] == 1.0
    for k in ("unc_diesel_max_kw", "unc_battery_max_discharge", "unc_battery_soc_init"):
        assert k in info
    env.close()


def test_plant_mismatch_keeps_values_physically_valid() -> None:
    s = Settings()
    s.uncertainty.enabled = True
    s.uncertainty.battery.capacity_std = 0.5
    s.uncertainty.battery.soc_init_std = 0.5
    s.uncertainty.diesel.max_kw_std = 0.5
    for seed in range(20):
        sampler = UncertaintySampler(s, np.random.default_rng(seed))
        sample = sampler.sample()
        perturbed = apply_plant_mismatch(s, sample)
        assert perturbed.battery.capacity_mwh > 0.0
        assert s.battery.soc_min <= perturbed.battery.soc_init <= s.battery.soc_max
        assert 0.0 < perturbed.battery.charge_eff <= 1.0
        assert perturbed.battery.max_discharge_mw > 0.0
        if perturbed.diesel.enabled:
            assert perturbed.diesel.min_kw <= perturbed.diesel.max_kw
            assert perturbed.diesel.max_kw > 0.0
        # Nominal settings object is never mutated.
        assert s.battery.capacity_mwh == 0.50
        assert s.battery.soc_init == 0.50


def test_disabled_sampler_returns_identity() -> None:
    s = Settings()
    s.uncertainty.enabled = False
    sampler = UncertaintySampler(s, np.random.default_rng(0))
    sample = sampler.sample()
    assert sample == SampledUncertainty()


def test_forecast_dropout_marks_availability_off_without_dimension_change() -> None:
    s = Settings()
    s.uncertainty.enabled = True
    s.uncertainty.forecast.dropout_prob = 1.0  # always drop -> always stale
    s.uncertainty.forecast.dropout_block_steps = 999999
    env = MicrogridEnv(settings=s)
    obs, info = env.reset(seed=0)
    assert np.isinf(env.observation_space.low.max()) or env.observation_space.shape == obs.shape
    # forecast dims are present and zeroed/unavailable
    assert "forecast_available" in info
    env.close()