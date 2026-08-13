"""Dashboard configuration and rollout contract for trained RL policies."""

from __future__ import annotations

import numpy as np

from microgrid_simulator.config import Settings
from microgrid_simulator.rl.env import MicrogridEnv
from microgrid_simulator.ui.server import _defaults_payload, _merge_settings, _playback_settings


def test_dashboard_defaults_expose_rl_scenario_and_verified_artifact() -> None:
    payload = _defaults_payload()

    rl_settings = payload["policy_settings"]["rl"]
    assert rl_settings["scenario"]["name"] == "islanded_72h_hardunserved_2026-01-15"
    assert rl_settings["rl"]["hard_unserved"] is True
    assert rl_settings["digital_twin"]["measurements"]["load"]["file"].endswith(
        "demand_15min_weekly_seasonal_reconstruction_candidate.csv"
    )
    assert payload["rl_presets"][0] == {
        "label": "v3 seed-0 (60k steps)",
        "artifact": "artifacts/sac/hardunserved-v3-60k/seed-0/sac_microgrid.zip",
        "algo": "sac",
    }


def test_rl_dashboard_honors_episode_and_topology_settings() -> None:
    settings = _merge_settings(
        {
            "scenario": {"name": "dashboard-experiment"},
            "episode": {
                "horizon_hours": 48.0,
                "telemetry_start": "2026-01-20 00:00:00",
            },
            "topology": {"timestep_hours": 0.5},
        },
        policy="rl",
    )

    assert settings.scenario.name == "dashboard-experiment"
    assert settings.episode.horizon_hours == 48.0
    assert settings.episode.telemetry_start == "2026-01-20 00:00:00"
    assert settings.topology.timestep_hours == 0.5
    assert settings.rl.hard_unserved is True


def test_rl_dashboard_uses_the_policy_specific_frontend_settings() -> None:
    """The UI now switches to the RL scenario before submitting the complete
    settings object, so applying the browser payload preserves that scenario
    without silently overriding later dashboard edits."""
    from microgrid_simulator.ui.server import _base_settings

    dashboard_settings = _base_settings("rl").model_dump()
    dashboard_settings["battery"]["capacity_mwh"] = 0.8
    dashboard_settings["battery"]["max_discharge_mw"] = 0.3
    dashboard_settings["battery"]["soc_init"] = 0.7
    dashboard_settings["reward"]["w_unserved"] = 35.0
    dashboard_settings["rl"]["hard_unserved"] = False

    settings = _merge_settings(dashboard_settings, policy="rl")

    assert settings.scenario.name == "islanded_72h_hardunserved_2026-01-15"
    assert settings.digital_twin.measurements["load"].file.endswith(
        "demand_15min_weekly_seasonal_reconstruction_candidate.csv"
    )
    assert settings.digital_twin.measurements["pv"].file.endswith(
        "pv_15min_chronos_reconstruction_candidate.csv"
    )
    assert settings.rl.hard_unserved is False
    assert settings.rl.hard_unserved_penalty == 50000.0
    assert settings.rl.forecast_mode == "cached"
    assert settings.reward.w_carbon == 4.0
    assert settings.reward.w_unserved == 35.0
    assert settings.battery.capacity_mwh == 0.8
    assert settings.battery.max_discharge_mw == 0.3
    assert settings.battery.soc_init == 0.7


def test_rl_dashboard_honors_complete_battery_settings() -> None:
    settings = _merge_settings(
        {
            "battery": {
                "capacity_mwh": 66.116,
                "max_charge_mw": 18.36,
                "soc_init": 0.7,
            }
        },
        policy="rl",
    )

    assert settings.battery.soc_init == 0.7
    assert settings.battery.capacity_mwh == 66.116
    assert settings.battery.max_charge_mw == 18.36
    assert settings.battery.max_discharge_mw == 0.25


def test_rl_dashboard_playback_disables_hard_termination_only() -> None:
    raw = _defaults_payload()["policy_settings"]["rl"]

    configured = _merge_settings(raw, policy="rl")
    playback = _playback_settings(raw, policy="rl")

    assert configured.rl.hard_unserved is True
    assert playback.rl.hard_unserved is False
    assert playback.rl.hard_unserved_tol_mw == configured.rl.hard_unserved_tol_mw
    assert playback.rl.hard_unserved_penalty == configured.rl.hard_unserved_penalty
    assert playback.episode.horizon_hours == configured.episode.horizon_hours


def test_islanded_72h_episode_feasible_with_diesel() -> None:
    """The hard-unserved islanded scenario is trivially feasible: keeping the
    genset committed at min stable output serves the whole 72-hour window.

    Guards against the dashboard/scheduler ever reporting 'the physics cannot
    support the episode' when an RL policy instead fails to commit diesel —
    those are different failure modes and must stay distinguishable.
    """
    settings = Settings.from_yaml("configs/islanded-baseline-72h-hardunserved.yaml")
    settings.rl.forecast_mode = "none"
    env = MicrogridEnv(settings=settings)
    env.reset(seed=0)

    for step in range(env.max_steps):
        action = np.array([-1.0, 0.3, -1.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if info.get("hard_unserved_triggered"):
            assert False, f"hard unserved termination at step {step}: unserved={info['unserved_mw']:.4f} MW"
        assert info["unserved_mw"] <= 0.002, (
            f"unserved {info['unserved_mw']:.4f} MW exceeds 2 kW tolerance at step {step}"
        )
        if truncated:
            break
    env.close()
