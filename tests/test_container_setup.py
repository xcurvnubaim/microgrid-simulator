from __future__ import annotations

from pathlib import Path

import yaml

from microgrid_simulator.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_compose_defines_normal_and_ems_runtime_profiles() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())

    assert compose["services"]["nats"]["image"].startswith("nats:")
    assert "-js" in compose["services"]["nats"]["command"]
    assert compose["services"]["nats"]["volumes"] == ["nats-data:/data"]
    assert compose["services"]["dashboard-normal"]["profiles"] == ["normal"]
    assert compose["services"]["dashboard-normal"]["build"]["target"] == "cpu-rl"
    assert compose["services"]["dashboard-normal"]["image"] == (
        "microgrid-ems-cpu-rl:local"
    )
    normal_mounts = {
        mount["target"]: mount
        for mount in compose["services"]["dashboard-normal"]["volumes"]
        if isinstance(mount, dict)
    }
    assert normal_mounts["/app/artifacts"]["read_only"] is True
    assert compose["services"]["dashboard-ems"]["profiles"] == ["ems"]
    assert compose["services"]["dashboard-ems"]["ports"] == [
        "${EMS_DASHBOARD_PORT:-8501}:8501"
    ]
    assert compose["services"]["dashboard-ems"]["environment"]["MGS_NATS_URL"] == (
        "nats://nats:4222"
    )
    assert "telemetry-serve" in compose["services"]["telemetry"]["command"]
    assert "ems-control-serve" in compose["services"]["ems"]["command"]
    assert "plant-serve" in compose["services"]["plant"]["command"]
    assert "simulator" not in compose["services"]
    assert "nats://nats:4222" in compose["services"]["telemetry"]["command"]
    assert "nats://nats:4222" in compose["services"]["ems"]["command"]
    assert "nats://nats:4222" in compose["services"]["plant"]["command"]
    assert compose["services"]["ems"]["build"]["target"] == "cpu-rl"
    assert compose["services"]["ems"]["environment"]["MGS_EMS_POLICY"].endswith(
        ":-rule}"
    )
    assert "MGS_EMS_ARTIFACT" in compose["services"]["ems"]["environment"]
    assert "docker-islanded-72h.yaml" in compose["x-runtime"]["environment"]["MGS_CONFIG"]
    telemetry_mounts = {
        mount["target"]: mount
        for mount in compose["services"]["telemetry"]["volumes"]
        if isinstance(mount, dict)
    }
    assert telemetry_mounts["/data"]["read_only"] is True
    ems_mounts = {mount["target"]: mount for mount in compose["services"]["ems"]["volumes"]}
    assert ems_mounts["/app/configs"]["read_only"] is True
    assert ems_mounts["/app/artifacts"]["read_only"] is True
    assert compose["services"]["dashboard-ems"]["environment"]["MGS_MODULAR_MODE"] == "true"
    assert compose["services"]["dashboard-normal"]["environment"]["MGS_MODULAR_MODE"] == "false"
    assert "nats-data" in compose["volumes"]


def test_docker_runtime_includes_websocket_server_support() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert '"uvicorn[standard]>=0.29"' in dockerfile
    assert "FROM runtime AS cpu-rl" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile


def test_container_scenario_uses_mounted_telemetry_and_baked_forecasts() -> None:
    settings = Settings.from_yaml(ROOT / "configs" / "docker-islanded-72h.yaml")

    assert settings.digital_twin.measurements["load"].file.startswith("/data/")
    assert settings.digital_twin.measurements["pv"].file.startswith("/data/")
    assert settings.forecast.cache_path == "data/forecasts.jsonl"
    assert settings.forecast.manifest_path == "data/forecast_manifest.json"
