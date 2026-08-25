"""Dashboard configuration and rollout contract for trained RL policies."""

from __future__ import annotations

import numpy as np

from microgrid_simulator.config import Settings
from microgrid_simulator.rl.env import MicrogridEnv
from microgrid_simulator.ui.server import (
    _defaults_payload,
    _experiment_catalog,
    _merge_settings,
    _playback_settings,
)


def test_dashboard_defaults_expose_active_rl_scenario_and_artifact(monkeypatch) -> None:
    monkeypatch.delenv("MGS_CONFIG", raising=False)
    payload = _defaults_payload()

    rl_settings = payload["policy_settings"]["rl"]
    assert rl_settings["scenario"]["name"] == "islanded_72h_2026-01-15"
    assert rl_settings["rl"]["hard_unserved"] is False
    assert rl_settings["digital_twin"]["measurements"]["load"]["file"].endswith(
        "demand_15min_weekly_seasonal_reconstruction_candidate.csv"
    )
    assert payload["rl_presets"][0] == {
        "label": "No-forecast SAC seed 1 (1M steps)",
        "artifact": "artifacts/sac/noforecast-1m/seed-1/sac_microgrid.zip",
        "algo": "sac",
    }


def test_dashboard_exposes_march_experiment_and_documented_policy_extensions() -> None:
    catalog = _experiment_catalog()

    assert catalog["id"] == "continuous_march_policy_catalog"
    assert [scenario["id"] for scenario in catalog["scenarios"]] == [
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
    ]
    policies = {policy["id"]: policy for policy in catalog["policies"]}
    assert list(policies) == [
        "rule_f3",
        "schedule",
        "pypsa_rh_f3",
        "sac_f3",
        "sac_none_f3",
        "sac_f3_summary",
        "sac_f3_finetuned",
    ]
    assert policies["pypsa_rh_f3"]["label"] == "PyPSA-RH-F3"
    assert policies["sac_f3"]["runtime_policy"] == "rl"
    assert policies["sac_f3"]["seeds"] == [0, 1, 2]
    assert policies["sac_f3"]["mask_episode_progress"] is True
    assert policies["sac_f3"]["artifacts"]["E5"]["2"] == (
        "artifacts/sac/f3-15min/E5/f3/seed-2/sac_microgrid.zip"
    )
    assert policies["sac_none_f3"]["settings_overrides"]["rl"]["forecast_mode"] == "none"
    assert policies["sac_f3_summary"]["settings_overrides"]["rl"] == {
        "forecast_mode": "cached",
        "forecast_representation": "summary",
    }
    assert policies["sac_f3_summary"]["artifacts"]["E3"]["1"] == (
        "artifacts/sac/f3-summary/E3/seed-1/sac_microgrid.zip"
    )
    assert policies["sac_f3_finetuned"]["promotion_status"] == "not promoted"
    assert policies["sac_f3_finetuned"]["training_change"] == "fine_tune + gamma=0.995"
    assert policies["sac_f3_finetuned"]["artifacts"]["E0"]["2"] == (
        "artifacts/agent-loop-cross/e0/iteration-001/seed-2/sac_microgrid.zip"
    )
    assert policies["sac_f3_finetuned"]["artifacts"]["E5"]["2"] == (
        "artifacts/agent-loop/e5/iteration-002/seed-2/sac_microgrid.zip"
    )


def test_dashboard_experiment_rl_masks_normalized_episode_progress() -> None:
    from types import SimpleNamespace

    from microgrid_simulator.ui.rollout import policy_action

    class RecordingModel:
        def __init__(self) -> None:
            self.observation = None

        def predict(self, observation, deterministic=True):  # noqa: ANN001, ARG002
            self.observation = observation
            return np.zeros(3, dtype=np.float32), None

    env = SimpleNamespace(_last_observation=np.asarray([1.0, 2.0, 3.0, 4.0]))
    norm = SimpleNamespace(mean=np.zeros(4), var=np.ones(4))
    model = RecordingModel()

    policy_action(
        env,
        "rl",
        rl_model=model,
        rl_norm=norm,
        rl_mask_progress_index=2,
    )

    assert model.observation.tolist() == [1.0, 2.0, 0.0, 4.0]


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
    assert settings.rl.hard_unserved is False


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

    assert settings.scenario.name == "islanded_72h_2026-01-15"
    assert settings.digital_twin.measurements["load"].file.endswith(
        "demand_15min_weekly_seasonal_reconstruction_candidate.csv"
    )
    assert settings.digital_twin.measurements["pv"].file.endswith(
        "pv_15min_chronos_reconstruction_candidate.csv"
    )
    assert settings.rl.hard_unserved is False
    assert settings.rl.hard_unserved_penalty == 1000.0
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


def test_rl_dashboard_playback_preserves_nominal_scenario(monkeypatch) -> None:
    monkeypatch.delenv("MGS_CONFIG", raising=False)
    raw = _defaults_payload()["policy_settings"]["rl"]

    configured = _merge_settings(raw, policy="rl")
    playback = _playback_settings(raw, policy="rl")

    assert configured.rl.hard_unserved is False
    assert playback.rl.hard_unserved is False
    assert playback.rl.hard_unserved_tol_mw == configured.rl.hard_unserved_tol_mw
    assert playback.rl.hard_unserved_penalty == configured.rl.hard_unserved_penalty
    assert playback.episode.horizon_hours == configured.episode.horizon_hours


def test_islanded_72h_episode_feasible_with_diesel() -> None:
    """The nominal islanded scenario is feasible: keeping the
    genset committed at min stable output serves the whole 72-hour window.

    Guards against the dashboard/scheduler ever reporting 'the physics cannot
    support the episode' when an RL policy instead fails to commit diesel —
    those are different failure modes and must stay distinguishable.
    """
    settings = Settings.from_yaml("configs/islanded-baseline-72h.yaml")
    settings.rl.forecast_mode = "none"
    env = MicrogridEnv(settings=settings)
    env.reset(seed=0)

    for step in range(env.max_steps):
        action = np.array([-1.0, 0.3, -1.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if info.get("hard_unserved_triggered"):
            raise AssertionError(
                f"hard unserved termination at step {step}: "
                f"unserved={info['unserved_mw']:.4f} MW"
            )
        assert info["unserved_mw"] <= 0.002, (
            f"unserved {info['unserved_mw']:.4f} MW exceeds 2 kW tolerance at step {step}"
        )
        if truncated:
            break
    env.close()
