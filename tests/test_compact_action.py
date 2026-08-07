"""RL Plan Week 1 Days 1–2: compact full-EMS action mapping and terminal-SOC.

Covers the RL Plan §Policy and environment contract decisions:

* compact action ``[battery_command, diesel_command, pv_curtail]`` — a single
  continuous diesel command whose positive part commits the genset and maps
  linearly onto its [min_kw, max_kw] setpoint band while non-positive commands
  request off;
* terminal-SOC return-to-start handling — an episode ending outside the
  configured tolerance terminates with a one-shot deviation penalty.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from microgrid_simulator.config import Settings  # noqa: E402
from microgrid_simulator.core.types import ControlAction  # noqa: E402
from microgrid_simulator.rl.env import MicrogridEnv, decode_action, encode_action  # noqa: E402


def _settings() -> Settings:
    s = Settings()
    s.diesel.enabled = True
    s.diesel.max_kw = 150.0
    s.diesel.min_kw = 45.0
    s.topology.n_ev = 0
    return s


# -- compact action decoding -------------------------------------------------


def test_action_space_is_compact() -> None:
    env = MicrogridEnv(settings=_settings())
    # [battery_command, diesel_command, pv_curtail] — no EVs on campus topology
    assert env.action_dim == 3
    assert env.action_space.shape == (3,)
    env.close()


def test_decode_positive_diesel_command_maps_onto_setpoint_band() -> None:
    settings = _settings()
    # a = 1 -> nameplate max
    action = decode_action(np.array([0.0, 1.0, 0.0]), settings, n_ev=0, diesel_enabled=True)
    assert action.diesel_on
    assert action.diesel_setpoint_mw == pytest.approx(0.150)
    # a -> 0+ -> minimum stable load
    action = decode_action(np.array([0.0, 1e-6, 0.0]), settings, n_ev=0, diesel_enabled=True)
    assert action.diesel_on
    assert action.diesel_setpoint_mw == pytest.approx(0.045, abs=1e-5)
    # midpoint -> band midpoint
    action = decode_action(np.array([0.0, 0.5, 0.0]), settings, n_ev=0, diesel_enabled=True)
    assert action.diesel_on
    assert action.diesel_setpoint_mw == pytest.approx((0.045 + 0.150) / 2.0)


def test_decode_nonpositive_diesel_command_requests_off() -> None:
    settings = _settings()
    for raw in (0.0, -0.5, -1.0):
        action = decode_action(
            np.array([0.0, raw, 0.0]), settings, n_ev=0, diesel_enabled=True
        )
        assert not action.diesel_on
        assert action.diesel_setpoint_mw == 0.0


def test_decode_battery_command_symmetric_map() -> None:
    settings = _settings()
    charge = decode_action(np.array([1.0, 0.0, 0.0]), settings, n_ev=0, diesel_enabled=True)
    discharge = decode_action(np.array([-1.0, 0.0, 0.0]), settings, n_ev=0, diesel_enabled=True)
    assert charge.battery_p_mw == pytest.approx(settings.battery.max_charge_mw)
    assert discharge.battery_p_mw == pytest.approx(-settings.battery.max_discharge_mw)


def test_encode_is_inverse_of_decode_for_diesel_and_battery() -> None:
    settings = _settings()
    for physical in (
        ControlAction(battery_p_mw=0.1, diesel_on=True, diesel_setpoint_mw=0.150),
        ControlAction(battery_p_mw=-0.1, diesel_on=True, diesel_setpoint_mw=0.045),
        ControlAction(battery_p_mw=0.0, diesel_on=False, diesel_setpoint_mw=0.0),
    ):
        decoded = decode_action(
            encode_action(physical, settings, n_ev=0, diesel_enabled=True),
            settings,
            n_ev=0,
            diesel_enabled=True,
        )
        assert decoded.battery_p_mw == pytest.approx(physical.battery_p_mw)
        assert decoded.diesel_on == physical.diesel_on
        # float32 rounding through the normalised space: tolerance ~1e-6 of band
        assert decoded.diesel_setpoint_mw == pytest.approx(
            physical.diesel_setpoint_mw, abs=1e-5
        )


def test_encode_off_diesel_command_is_strictly_negative() -> None:
    settings = _settings()
    a = encode_action(
        ControlAction(diesel_on=False, diesel_setpoint_mw=0.0), settings, 0, True
    )
    assert a[1] < 0.0  # zero would decode as "off" only by the > 0 edge; keep it unambiguous


# -- terminal-SOC return-to-start --------------------------------------------


def test_terminal_soc_within_tolerance_truncates_without_penalty() -> None:
    env = MicrogridEnv(settings=_settings())
    obs, info = env.reset(seed=0)
    assert info["initial_soc"] == pytest.approx(env.backend.battery.soc)
    idle = np.zeros(env.action_dim, dtype=np.float32)
    idle[1] = -1.0  # diesel off
    terminated = truncated = False
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(idle)
    assert truncated  # horizon reached without a SOC violation
    assert not terminated
    assert info["terminal_soc_met"]
    assert info["terminal_soc_deviation"] <= info["terminal_soc_tolerance"]
    assert info["terminal_soc_penalty"] == 0.0
    env.close()


def test_terminal_soc_violation_terminates_with_penalty() -> None:
    env = MicrogridEnv(settings=_settings())
    env.reset(seed=0)
    # Discharge the battery as hard as possible for the whole episode so the
    # final SOC lands far below the starting SOC.
    drain = np.zeros(env.action_dim, dtype=np.float32)
    drain[0] = -1.0
    drain[1] = -1.0  # diesel off
    reward_total = 0.0
    terminated = truncated = False
    for _ in range(env.max_steps + 5):
        _, reward, terminated, truncated, info = env.step(drain)
        reward_total += reward
        if terminated or truncated:
            break
    assert terminated, "a terminal-SOC violation must flip the episode to terminated"
    assert not truncated
    assert not info["terminal_soc_met"]
    assert info["terminal_soc_deviation"] > info["terminal_soc_tolerance"]
    expected = info["terminal_soc_deviation"] * env.settings.episode.terminal_soc_penalty
    assert info["terminal_soc_penalty"] == pytest.approx(expected)
    env.close()


def test_terminal_soc_penalty_scales_with_config() -> None:
    settings = _settings()
    settings.episode.terminal_soc_penalty = 7.5
    env = MicrogridEnv(settings=settings)
    env.reset(seed=0)
    drain = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(env.max_steps + 5):
        _, _, terminated, truncated, info = env.step(drain)
        if terminated or truncated:
            break
    assert terminated
    assert info["terminal_soc_penalty"] == pytest.approx(
        info["terminal_soc_deviation"] * 7.5
    )
    env.close()


def test_reset_clears_terminal_soc_tracking() -> None:
    env = MicrogridEnv(settings=_settings())
    env.reset(seed=0)
    drain = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(env.max_steps + 5):
        _, _, terminated, truncated, _ = env.step(drain)
        if terminated or truncated:
            break
    # A fresh episode starts clean: no carried-over deviation or penalty.
    _, info = env.reset(seed=1)
    assert info["initial_soc"] == pytest.approx(env.settings.battery.soc_init)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    assert info["terminal_soc_deviation"] == pytest.approx(
        abs(info["soc"] - info["initial_soc"])
    )
    assert info["terminal_soc_penalty"] == 0.0
    env.close()
