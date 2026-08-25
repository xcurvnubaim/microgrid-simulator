"""FastAPI dashboard server — replaces the old Streamlit UI.

Serves a JSON API for the React frontend bundled in ``ui/web/dist``:

    GET  /api/defaults        resolved default settings + demand-trace status
    POST /api/simulate        {settings, policy, seed} -> per-timestep rollout
    POST /api/simulate/stream same, but NDJSON events streamed tick by tick
    POST /api/demand/upload   multipart xlsx/csv -> stored trace + stats
    GET  /api/demand/template example csv in the exact shape the upload expects

Run with ``microgrid-sim dashboard`` (uvicorn under the hood).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from math import pi, sin
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from microgrid_simulator.config import Settings, load_settings
from microgrid_simulator.grid.demand_trace import DemandTrace
from microgrid_simulator.ui.rollout import POLICIES, run_rollout, stream_rollout

LOGGER = logging.getLogger(__name__)

WEB_DIST = Path(__file__).parent / "web" / "dist"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "microgrid-sim-demand"
RL_PRESETS = (
    {
        "label": "No-forecast SAC seed 1 (1M steps)",
        "artifact": "artifacts/sac/noforecast-1m/seed-1/sac_microgrid.zip",
        "algo": "sac",
    },
    {
        "label": "F0 forecast-aware SAC seed 0 (1M steps)",
        "artifact": "artifacts/sac/cached-1m-v3/seed-0/sac_microgrid.zip",
        "algo": "sac",
    },
)

EXPERIMENT_POLICY_LABELS = {
    "rule_f3": "Rule-F3",
    "schedule": "Schedule",
    "pypsa_rh_f3": "PyPSA-RH-F3",
    "sac_f3": "SAC-F3",
    "sac_none_f3": "SAC-none-F3",
    "sac_f3_summary": "SAC-F3-summary",
    # "sac_f3_finetuned": "FT SAC",
}

SERVICE_DEFINITIONS = (
    {
        "id": "control-room",
        "label": "Control room",
        "kind": "browser",
        "description": "Human-facing React dashboard and live event view.",
    },
    {
        "id": "microgrid-plant",
        "label": "Headless plant",
        "kind": "service",
        "description": "Owns authoritative physics, feasibility, and state transitions.",
    },
    {
        "id": "microgrid-telemetry",
        "label": "Telemetry",
        "kind": "service",
        "description": "Serves immutable historical telemetry windows.",
    },
    {
        "id": "microgrid-ems",
        "label": "EMS",
        "kind": "orchestrator",
        "description": "Owns scenario time, observations, dispatch, and episode lifecycle.",
    },
    {
        "id": "nats",
        "label": "NATS broker",
        "kind": "broker",
        "description": "Routes Core NATS requests and stores dashboard events in JetStream.",
    },
)

SERVICE_COMMUNICATION_PATHS = (
    {
        "id": "ems-telemetry",
        "source": "microgrid-ems",
        "target": "microgrid-telemetry",
        "label": "Historical telemetry window",
        "subject": "mgs.v1.telemetry.window",
        "transport": "Core NATS request / reply",
        "purpose": "The EMS asks for the exact replay window for the current episode.",
    },
    {
        "id": "ems-plant-step",
        "source": "microgrid-ems",
        "target": "microgrid-plant",
        "label": "Idempotent plant transition",
        "subject": "mgs.v1.plant.step",
        "transport": "Core NATS request / reply",
        "purpose": "The EMS sends one command; the plant returns authoritative realized state.",
    },
    {
        "id": "ems-plant-session",
        "source": "microgrid-ems",
        "target": "microgrid-plant",
        "label": "Plant session lifecycle",
        "subject": "mgs.v1.plant.start",
        "transport": "Core NATS request / reply",
        "purpose": "The EMS initializes each window with optional carried physical state.",
    },
    {
        "id": "ems-dashboard-events",
        "source": "microgrid-ems",
        "target": "nats",
        "label": "Durable dashboard trace",
        "subject": "mgs.v1.dashboard.<session>.*",
        "transport": "JetStream",
        "purpose": "Ordered event history is retained for replay and acknowledgment inspection.",
    },
    {
        "id": "ems-control-room",
        "source": "microgrid-ems",
        "target": "control-room",
        "label": "Live simulation events",
        "subject": "/api/ems/events/{run_id}",
        "transport": "WebSocket",
        "purpose": (
            "The passive browser replays EMS events through its same-origin dashboard server."
        ),
    },
)

MESSAGE_BUFFER_LIMIT = 100


def _message_route(subject: str, reply: str | None = None) -> dict[str, str]:
    if subject == "mgs.v1.telemetry.window":
        return {
            "kind": "request",
            "source": "EMS",
            "target": "Telemetry",
        }
    if subject in {"mgs.v1.plant.start", "mgs.v1.plant.step", "mgs.v1.plant.close"}:
        return {
            "kind": "request",
            "source": "EMS",
            "target": "Headless plant",
        }
    if subject.startswith("mgs.v1.dashboard."):
        return {
            "kind": "event",
            "source": "EMS",
            "target": "NATS / dashboard history",
        }
    if subject.startswith("_INBOX."):
        return {
            "kind": "reply",
            "source": "Service",
            "target": "Simulator",
        }
    return {
        "kind": "message",
        "source": "Application",
        "target": "NATS",
    }


class NatsMessageObserver:
    """Read-only application-message tap used by the human service map."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._nc: Any = None
        self._sequence = 0
        self._messages: deque[dict[str, Any]] = deque(maxlen=MESSAGE_BUFFER_LIMIT)
        self._reply_routes: dict[str, str] = {}
        self.error: str | None = None

    async def start(self) -> None:
        try:
            from nats.aio.client import Client as NATS

            self._nc = NATS()
            await self._nc.connect(
                servers=[self.url],
                name="microgrid-control-room-observer",
                connect_timeout=1,
                reconnect_time_wait=1,
                max_reconnect_attempts=0,
            )
            await self._nc.subscribe("mgs.v1.>", cb=self._record)
            await self._nc.subscribe("_INBOX.>", cb=self._record)
            await self._nc.flush()
        except Exception as exc:  # noqa: BLE001 - dashboard must degrade gracefully
            self.error = str(exc)
            self._nc = None

    async def _record(self, message: Any) -> None:
        subject = str(message.subject)
        route = _message_route(subject, message.reply)
        if route["kind"] == "request" and message.reply:
            self._reply_routes[str(message.reply)] = route["target"]
        elif route["kind"] == "reply":
            matched_prefix = next(
                (prefix for prefix in self._reply_routes if subject.startswith(prefix)),
                None,
            )
            if matched_prefix:
                route["source"] = self._reply_routes.pop(matched_prefix)

        raw_text = message.data.decode("utf-8", errors="replace")
        try:
            payload_text = json.dumps(json.loads(raw_text), indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            payload_text = raw_text
        truncated = len(payload_text) > 8000
        if truncated:
            payload_text = payload_text[:8000] + "\n… payload truncated"

        self._sequence += 1
        headers = {}
        if message.headers:
            for key in ("Nats-Msg-Id", "X-Trace-Id"):
                if key in message.headers:
                    headers[key] = message.headers[key]
        self._messages.appendleft(
            {
                "id": self._sequence,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "subject": subject,
                "reply": message.reply or None,
                "kind": route["kind"],
                "source": route["source"],
                "target": route["target"],
                "bytes": len(message.data),
                "payload": payload_text,
                "truncated": truncated,
                "headers": headers,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        if self._nc is None:
            status = "unavailable"
        elif self._nc.is_connected:
            status = "connected"
        else:
            status = "reconnecting"
        return {
            "status": status,
            "error": self.error,
            "captured": self._sequence,
            "limit": MESSAGE_BUFFER_LIMIT,
            "messages": list(self._messages),
        }

    async def stop(self) -> None:
        if self._nc is not None:
            await self._nc.drain()


class SimulateRequest(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    policy: str = "rule"
    seed: int = 0
    rl_artifact: str | None = None
    rl_algo: str | None = None
    rl_mask_episode_progress: bool = False


class StreamRequest(SimulateRequest):
    # Minimum wall time per tick so the live charts animate instead of the
    # whole episode arriving in one burst. 0 streams at solver speed.
    pace_ms: int = Field(default=25, ge=0, le=500)


def _base_settings(policy: str | None = None) -> Settings:
    """Use the shared default scenario, including explicit MGS_CONFIG overrides.

    The ``rl`` policy plays a policy trained on the campus controller-study
    scenario (forecast-enabled, islanded pandapower); the plain default
    scenario has forecasting disabled and produces a different observation
    shape, so it is swapped in for RL playback.

    The RL base honors ``$MGS_CONFIG`` first, then falls back to the active
    nominal islanded campus scenario. Archived policy variants must be selected
    explicitly rather than becoming implicit dashboard defaults.
    """
    if policy == "rl":
        from microgrid_simulator.config import Settings as S

        env_cfg = os.environ.get("MGS_CONFIG")
        if env_cfg and Path(env_cfg).is_file():
            return S.from_yaml(Path(env_cfg))
        campus = Path(__file__).resolve().parents[3] / "configs" / "islanded-baseline-72h.yaml"
        if campus.is_file():
            return S.from_yaml(campus)
    return load_settings()


def _merge_settings(overrides: dict[str, Any], policy: str | None = None) -> Settings:
    raw = _base_settings(policy).model_dump()
    # The artifact supplies the controller, not the plant. Dashboard settings
    # therefore remain authoritative for RL just as they are for rule and PyPSA-RH
    # playback. The policy-specific base only gives the UI a compatible initial
    # scenario; every explicit dashboard edit is applied below.
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return Settings(**raw)


def _playback_settings(overrides: dict[str, Any], policy: str) -> Settings:
    settings = _merge_settings(overrides, policy)
    if policy == "rl" and settings.rl.hard_unserved:
        # Hard unserved-load termination is a training constraint. Dashboard
        # inference should finish the requested horizon so the user can inspect
        # the complete blackout and unserved-energy trajectory.
        settings = settings.model_copy(
            update={"rl": settings.rl.model_copy(update={"hard_unserved": False})}
        )
    return settings


def _demand_template_csv(settings: Settings) -> str:
    """Example demand export in the exact shape the upload parser expects:
    ``header_row`` note rows first, then the header, then one row per tick."""
    cfg = settings.demand
    step_hours = settings.topology.timestep_hours
    unit = cfg.unit.lower()

    note = (
        f"# Demand trace template — keep this note row: the header must stay on "
        f"row {cfg.header_row + 1}. Replace the sample rows with your own "
        f"timestamps and total demand in {unit.upper()}."
    )
    lines = [f"{note},"] * cfg.header_row
    lines.append(f"{cfg.timestamp_column},{cfg.value_column}")

    start = datetime(2025, 1, 1)
    n_steps = int(round(48.0 / step_hours))  # two example days
    for i in range(n_steps):
        t = start + timedelta(hours=i * step_hours)
        hour = (i * step_hours) % 24.0
        # Plausible campus shape: ~180 kW base with a daytime bump and a bit of ripple.
        kw = 180.0 + 160.0 * max(0.0, sin(pi * (hour - 7.0) / 13.0)) + 8.0 * sin(hour * 2.1)
        value = kw if unit == "kw" else kw / 1000.0
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},{value:.1f}")
    return "\n".join(lines) + "\n"


def _demand_status(settings: Settings) -> dict[str, Any]:
    trace = DemandTrace.from_file(settings.demand, settings.topology.timestep_hours)
    if trace is None:
        return {"available": False, "stats": None}
    return {"available": True, "stats": trace.stats()}


def _defaults_payload() -> dict[str, Any]:
    settings = _base_settings()
    return {
        "settings": settings.model_dump(),
        "policy_settings": {"rl": _base_settings("rl").model_dump()},
        "policies": list(POLICIES),
        "rl_presets": list(RL_PRESETS),
        "experiment_catalog": _experiment_catalog(),
        "demand": _demand_status(settings),
    }


def _experiment_catalog() -> dict[str, Any]:
    """Expose the March matrix and its documented SAC extensions.

    The monthly evaluator remains the source of truth for scenario configs,
    policy/runtime mappings, seeds, and checkpoint paths. The summary arm is
    the completed 18-job March extension. FT SAC† is a February diagnostic arm,
    not a promoted policy or a result from the continuous-March comparison.
    """
    from microgrid_simulator.experiments.monthly_evaluation import (
        DETERMINISTIC_POLICIES,
        SAC_POLICIES,
        SAC_SEEDS,
        SAC_SUMMARY_POLICIES,
        SCENARIOS,
    )

    configured = Path(os.environ.get("MGS_CONFIG", ""))
    root_candidates = [
        configured.parent.parent if configured.is_absolute() else Path.cwd(),
        Path.cwd(),
        Path(__file__).resolve().parents[3],
    ]
    repo_root = next(
        (root for root in root_candidates if (root / "configs" / "f3-e0-monthly.yaml").is_file()),
        root_candidates[0],
    )
    scenario_paths = {
        scenario_id: repo_root / "configs" / f"f3-{scenario_id.lower()}-monthly.yaml"
        for scenario_id in SCENARIOS
    }
    scenarios = []
    for scenario_id, config_path in scenario_paths.items():
        scenario_settings = Settings.from_yaml(config_path)
        scenarios.append(
            {
                "id": scenario_id,
                "label": f"{scenario_id} · {scenario_settings.scenario.description}",
                "config": str(config_path.relative_to(repo_root)),
                "settings": scenario_settings.model_dump(),
            }
        )

    policies = []
    for policy_id, runtime_policy in DETERMINISTIC_POLICIES.items():
        policies.append(
            {
                "id": policy_id,
                "label": EXPERIMENT_POLICY_LABELS[policy_id],
                "runtime_policy": runtime_policy,
                "seeds": [0],
                "artifacts": {},
                "settings_overrides": {},
            }
        )
    for policy_id, forecast_mode in SAC_POLICIES.items():
        artifacts = {
            scenario_id: {
                str(seed): (
                    f"artifacts/sac/f3-15min/{scenario_id}/{forecast_mode}/seed-{seed}/"
                    "sac_microgrid.zip"
                )
                for seed in SAC_SEEDS
            }
            for scenario_id in SCENARIOS
        }
        policies.append(
            {
                "id": policy_id,
                "label": EXPERIMENT_POLICY_LABELS[policy_id],
                "runtime_policy": "rl",
                "seeds": list(SAC_SEEDS),
                "artifacts": artifacts,
                "settings_overrides": {
                    "rl": {
                        "forecast_mode": "none" if forecast_mode == "none" else "cached",
                        "forecast_representation": "raw",
                    }
                },
                "mask_episode_progress": True,
            }
        )

    for policy_id, forecast_representation in SAC_SUMMARY_POLICIES.items():
        artifacts = {
            scenario_id: {
                str(seed): (
                    f"artifacts/sac/f3-summary/{scenario_id}/seed-{seed}/"
                    "sac_microgrid.zip"
                )
                for seed in SAC_SEEDS
            }
            for scenario_id in SCENARIOS
        }
        policies.append(
            {
                "id": policy_id,
                "label": EXPERIMENT_POLICY_LABELS[policy_id],
                "runtime_policy": "rl",
                "seeds": list(SAC_SEEDS),
                "artifacts": artifacts,
                "settings_overrides": {
                    "rl": {
                        "forecast_mode": "cached",
                        "forecast_representation": forecast_representation,
                    }
                },
                "mask_episode_progress": True,
                "evidence_scope": "continuous March 2026 extension (18 jobs)",
                "promotion_status": "frozen comparison arm",
            }
        )

    fine_tuned_artifacts = {
        scenario_id: {
            str(seed): (
                f"artifacts/agent-loop/e5/iteration-002/seed-{seed}/sac_microgrid.zip"
                if scenario_id == "E5"
                else (
                    f"artifacts/agent-loop-cross/{scenario_id.lower()}/iteration-001/"
                    f"seed-{seed}/sac_microgrid.zip"
                )
            )
            for seed in SAC_SEEDS
        }
        for scenario_id in SCENARIOS
    }
    policies.append(
        {
            "id": "sac_f3_finetuned",
            "label": EXPERIMENT_POLICY_LABELS["sac_f3_finetuned"],
            "runtime_policy": "rl",
            "seeds": list(SAC_SEEDS),
            "artifacts": fine_tuned_artifacts,
            "settings_overrides": {
                "rl": {
                    "forecast_mode": "cached",
                    "forecast_representation": "raw",
                }
            },
            "mask_episode_progress": True,
            "evidence_scope": "72-hour February validation diagnostic",
            "promotion_status": "not promoted",
            "training_change": "fine_tune + gamma=0.995",
        }
    )

    return {
        "id": "continuous_march_policy_catalog",
        "label": "Continuous March policies + documented SAC extensions",
        "default_scenario": "E0",
        "default_policy": "rule_f3",
        "scenarios": scenarios,
        "policies": policies,
    }


def _runtime_payload() -> dict[str, Any]:
    modular_mode = os.environ.get("MGS_MODULAR_MODE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "runtime_mode": os.environ.get("MGS_RUNTIME_MODE", "normal"),
        "ems_policy": os.environ.get("MGS_EMS_POLICY"),
        "scenario_config": os.environ.get("MGS_CONFIG"),
        "modular_mode": modular_mode,
    }


def _fetch_nats_monitor_json(base_url: str, endpoint: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    with url_request.urlopen(url_request.Request(url, method="GET"), timeout=2.0) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"NATS monitor returned a non-object from /{endpoint}")
    return payload


def _connection_row(connection: dict[str, Any]) -> dict[str, Any]:
    subscriptions = []
    for subscription in connection.get("subscriptions_list_detail") or []:
        if not isinstance(subscription, dict):
            continue
        subscriptions.append(
            {
                "subject": str(subscription.get("subject", "")),
                "queue": subscription.get("qgroup") or None,
                "messages": int(subscription.get("msgs", 0) or 0),
                "pending": int(subscription.get("pending", 0) or 0),
            }
        )
    return {
        "name": str(connection.get("name") or f"client-{connection.get('cid', '?')}"),
        "uptime": connection.get("uptime"),
        "in_msgs": int(connection.get("in_msgs", 0) or 0),
        "out_msgs": int(connection.get("out_msgs", 0) or 0),
        "in_bytes": int(connection.get("in_bytes", 0) or 0),
        "out_bytes": int(connection.get("out_bytes", 0) or 0),
        "subscriptions": subscriptions,
    }


def _service_communications_payload(
    varz: dict[str, Any],
    connz: dict[str, Any],
    *,
    monitor_error: str | None = None,
) -> dict[str, Any]:
    connections_by_name: dict[str, list[dict[str, Any]]] = {}
    for raw_connection in connz.get("connections") or []:
        if not isinstance(raw_connection, dict):
            continue
        row = _connection_row(raw_connection)
        connections_by_name.setdefault(row["name"], []).append(row)

    nats_available = monitor_error is None
    service_rows = []
    service_status: dict[str, str] = {}
    for definition in SERVICE_DEFINITIONS:
        service_id = definition["id"]
        instances = connections_by_name.get(service_id, [])
        if service_id == "control-room":
            status = "connected"
        elif service_id == "nats":
            status = "connected" if nats_available else "unavailable"
        elif instances:
            status = "connected"
        else:
            status = "not connected"
        service_status[service_id] = status
        displayed_instances = len(instances)
        if service_id in {"control-room", "nats"} and status == "connected":
            displayed_instances = 1
        service_rows.append(
            {
                **definition,
                "status": status,
                "instances": displayed_instances,
                "in_msgs": sum(item["in_msgs"] for item in instances),
                "out_msgs": sum(item["out_msgs"] for item in instances),
                "in_bytes": sum(item["in_bytes"] for item in instances),
                "out_bytes": sum(item["out_bytes"] for item in instances),
                "subscriptions": [
                    subscription for item in instances for subscription in item["subscriptions"]
                ],
            }
        )

    def subject_stats(target: str, subject: str) -> tuple[int, int]:
        delivered = 0
        pending = 0
        for connection in connections_by_name.get(target, []):
            for subscription in connection["subscriptions"]:
                if subscription["subject"] == subject:
                    delivered += subscription["messages"]
                    pending += subscription["pending"]
        return delivered, pending

    paths = []
    for definition in SERVICE_COMMUNICATION_PATHS:
        source_status = service_status.get(definition["source"], "unknown")
        target_status = service_status.get(definition["target"], "unknown")
        subject = definition["subject"]
        delivered = None
        pending = None

        if definition["transport"] == "WebSocket":
            status = "available"
            active = True
        elif definition["target"] == "nats":
            status = "active" if nats_available and source_status == "connected" else "unavailable"
            active = status == "active"
        elif monitor_error:
            status = "unknown"
            active = None
        else:
            delivered, pending = subject_stats(definition["target"], subject)
            active = target_status == "connected" and delivered is not None
            status = "active" if active else target_status

        paths.append(
            {
                **definition,
                "status": status,
                "active": active,
                "delivered_messages": delivered,
                "pending_messages": pending,
            }
        )

    return {
        "status": "degraded" if monitor_error else "ok",
        "error": monitor_error,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "server": {
            "version": varz.get("version"),
            "uptime": varz.get("uptime"),
            "connections": int(varz.get("connections", len(connz.get("connections") or [])) or 0),
            "subscriptions": int(varz.get("subscriptions", 0) or 0),
            "in_msgs": int(varz.get("in_msgs", 0) or 0),
            "out_msgs": int(varz.get("out_msgs", 0) or 0),
            "in_msgs_per_sec": float(varz.get("in_msgs_per_sec", 0) or 0),
            "out_msgs_per_sec": float(varz.get("out_msgs_per_sec", 0) or 0),
            "jetstream": bool(varz.get("jetstream")),
        },
        "services": service_rows,
        "paths": paths,
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.message_observer = None
        app.state.ems_nats = None
        app.state.ems_jetstream = None
        if os.environ.get("MGS_RUNTIME_MODE") == "ems":
            from microgrid_simulator.messaging.nats import connect_nats, ensure_dashboard_stream

            app.state.ems_nats = await connect_nats(
                os.environ.get("MGS_NATS_URL", "nats://127.0.0.1:4222"),
                name="microgrid-dashboard",
            )
            app.state.ems_jetstream = await ensure_dashboard_stream(app.state.ems_nats)
        yield
        observer = app.state.message_observer
        if observer is not None:
            await observer.stop()
        if app.state.ems_nats is not None:
            await app.state.ems_nats.drain()

    app = FastAPI(
        title="Microgrid Simulator",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    @app.get("/api/defaults")
    def defaults() -> dict[str, Any]:
        return _defaults_payload()

    @app.get("/api/runtime")
    def runtime() -> dict[str, Any]:
        """Expose the external controller contract used by the Compose UI."""
        return _runtime_payload()

    def ems_json(method: str, path: str) -> dict[str, Any]:
        base = os.environ.get("MGS_EMS_INTERNAL_URL")
        if not base:
            raise HTTPException(404, "EMS control plane is not configured")
        request = url_request.Request(f"{base.rstrip('/')}{path}", method=method)
        try:
            with url_request.urlopen(request, timeout=5.0) as response:
                payload = json.load(response)
        except (OSError, TimeoutError, url_error.URLError) as exc:
            raise HTTPException(503, f"EMS control plane unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(502, "EMS control plane returned a non-object")
        return payload

    @app.get("/api/ems/latest")
    async def ems_latest() -> dict[str, Any]:
        """Same-origin discovery for the passive dashboard."""
        return await asyncio.to_thread(ems_json, "GET", "/runs/latest")

    @app.get("/api/ems/runs/{run_id}")
    async def ems_status(run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(ems_json, "GET", f"/runs/{run_id}")

    @app.websocket("/api/ems/events/{run_id}")
    async def ems_events(websocket: WebSocket, run_id: str) -> None:
        """Replay and follow the EMS JetStream trace without browser broker access."""
        from nats.errors import TimeoutError as NatsTimeoutError

        from microgrid_simulator.messaging.nats import DASHBOARD_STREAM, decode_json, trace_subject

        if app.state.ems_jetstream is None:
            await websocket.close(code=4404, reason="EMS event store is not configured")
            return
        await websocket.accept()
        subscription = await app.state.ems_jetstream.pull_subscribe(
            trace_subject(run_id), stream=DASHBOARD_STREAM
        )
        try:
            while True:
                try:
                    messages = await subscription.fetch(1, timeout=15)
                except NatsTimeoutError:
                    status_payload = await asyncio.to_thread(ems_json, "GET", f"/runs/{run_id}")
                    if status_payload.get("state") in {"completed", "cancelled", "failed"}:
                        return
                    continue
                message = messages[0]
                event = decode_json(message.data)
                await websocket.send_json(event)
                await message.ack()
                if event.get("type") == "ems_end":
                    return
        except WebSocketDisconnect:
            return
        finally:
            await subscription.unsubscribe()

    async def ensure_message_observer() -> NatsMessageObserver | None:
        if os.environ.get("MGS_RUNTIME_MODE") != "ems" and not os.environ.get("MGS_NATS_URL"):
            return None
        monitor_url = os.environ.get("MGS_NATS_URL", "nats://127.0.0.1:4222")
        observer = app.state.message_observer
        if observer is None or observer.url != monitor_url:
            observer = NatsMessageObserver(monitor_url)
            await observer.start()
            app.state.message_observer = observer
        return observer

    @app.get("/api/service-communications")
    async def service_communications() -> dict[str, Any]:
        """Return a human-readable service map backed by NATS monitoring data."""
        monitor_url = os.environ.get("MGS_NATS_MONITOR_URL", "http://127.0.0.1:8222")
        observer = await ensure_message_observer()
        try:
            varz, connz = await asyncio.to_thread(
                lambda: (
                    _fetch_nats_monitor_json(monitor_url, "/varz"),
                    _fetch_nats_monitor_json(monitor_url, "/connz?subs=detail"),
                )
            )
        except (OSError, TimeoutError, ValueError, url_error.URLError) as exc:
            LOGGER.warning("NATS service map unavailable: %s", exc)
            payload = _service_communications_payload({}, {}, monitor_error=str(exc))
        else:
            payload = _service_communications_payload(varz, connz)
        payload["message_tap"] = (
            observer.snapshot()
            if observer
            else {
                "status": "unavailable",
                "error": "NATS disabled in standalone mode",
                "captured": 0,
                "limit": MESSAGE_BUFFER_LIMIT,
                "messages": [],
            }
        )
        return payload

    @app.post("/api/simulate")
    def simulate(req: SimulateRequest) -> dict[str, Any]:
        if req.policy not in POLICIES:
            raise HTTPException(422, f"policy must be one of {POLICIES}")
        if req.policy == "rl" and not req.rl_artifact:
            raise HTTPException(422, "policy 'rl' requires rl_artifact")
        try:
            settings = _playback_settings(req.settings, req.policy)
        except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
            raise HTTPException(422, f"invalid settings: {exc}") from exc

        try:
            result = run_rollout(
                settings,
                policy=req.policy,
                seed=req.seed,
                rl_artifact=req.rl_artifact,
                rl_algo=req.rl_algo,
                rl_mask_episode_progress=req.rl_mask_episode_progress,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result["demand"] = _demand_status(settings)
        return result

    @app.post("/api/simulate/stream")
    def simulate_stream(req: StreamRequest) -> StreamingResponse:
        if req.policy not in POLICIES:
            raise HTTPException(422, f"policy must be one of {POLICIES}")
        if req.policy == "rl" and not req.rl_artifact:
            raise HTTPException(422, "policy 'rl' requires rl_artifact")
        try:
            settings = _playback_settings(req.settings, req.policy)
        except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
            raise HTTPException(422, f"invalid settings: {exc}") from exc

        pace = req.pace_ms / 1000.0

        def ndjson() -> Iterator[str]:
            last = time.monotonic()
            for event in stream_rollout(
                settings,
                policy=req.policy,
                seed=req.seed,
                rl_artifact=req.rl_artifact,
                rl_algo=req.rl_algo,
                rl_mask_episode_progress=req.rl_mask_episode_progress,
            ):
                if pace and event["type"] == "row":
                    now = time.monotonic()
                    wait = pace - (now - last)
                    if wait > 0:
                        time.sleep(wait)
                    last = time.monotonic()
                yield json.dumps(event) + "\n"

        return StreamingResponse(
            ndjson(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/demand/template")
    def demand_template() -> Response:
        return Response(
            content=_demand_template_csv(_base_settings()),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="demand_template.csv"'},
        )

    @app.post("/api/demand/upload")
    async def upload_demand(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        suffix = Path(file.filename or "demand.xlsx").suffix.lower()
        if suffix not in {".xlsx", ".xlsm", ".xls", ".csv"}:
            raise HTTPException(422, "expected an .xlsx or .csv demand export")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / f"demand{suffix}"
        dest.write_bytes(await file.read())

        settings = _base_settings()
        cfg = settings.demand.model_copy(update={"file": str(dest), "enabled": True})
        trace = DemandTrace.from_file(cfg, settings.topology.timestep_hours)
        if trace is None:
            raise HTTPException(
                422,
                "could not parse the file — expected columns "
                f"'{cfg.timestamp_column}' and '{cfg.value_column}' with the "
                f"header on row {cfg.header_row + 1}",
            )
        return {"file": str(dest), "stats": trace.stats()}

    # --- static frontend (built React app) ---------------------------------
    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:  # pragma: no cover - dev convenience

        @app.get("/", include_in_schema=False)
        def missing() -> dict[str, str]:
            return {
                "detail": "frontend not built — run `npm install && npm run build` "
                "in src/microgrid_simulator/ui/web, or use the /api endpoints"
            }

    return app


app = create_app()
