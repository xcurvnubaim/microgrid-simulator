from __future__ import annotations

from microgrid_simulator.ui.server import _runtime_payload


def test_runtime_exposes_external_ems_contract(monkeypatch) -> None:
    monkeypatch.setenv("MGS_RUNTIME_MODE", "ems")
    monkeypatch.setenv("MGS_EMS_POLICY", "sac")
    monkeypatch.setenv("MGS_CONFIG", "/app/configs/docker-islanded-72h.yaml")
    monkeypatch.setenv("MGS_MODULAR_MODE", "true")

    payload = _runtime_payload()

    assert payload == {
        "runtime_mode": "ems",
        "ems_policy": "sac",
        "scenario_config": "/app/configs/docker-islanded-72h.yaml",
        "modular_mode": True,
    }
