"""Standalone EMS inference service operating only on broker messages."""

from __future__ import annotations

import asyncio
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np

from microgrid_simulator.backends.pypsa_backend import PyPSAOperationalBackend
from microgrid_simulator.config import Settings
from microgrid_simulator.controllers import RuleBasedController
from microgrid_simulator.controllers.pypsa_rolling_horizon import (
    PyPSARollingHorizonController,
)
from microgrid_simulator.ems.broker import EventBroker
from microgrid_simulator.ems.shield import SafetyShield
from microgrid_simulator.ems.types import DispatchCommand, TelemetryFrame, utc_now
from microgrid_simulator.rl.env import decode_action, encode_action


class EMSService:
    """Consume telemetry and emit one deadline-bound, shielded command per frame."""

    def __init__(
        self,
        settings: Settings,
        broker: EventBroker | None = None,
        *,
        policy: Literal["rule", "sac", "ppo", "pypsa_rh", "pypsa_mpc"] = "rule",
        artifact: str | Path | None = None,
        command_ttl_ms: int = 1000,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.policy = "pypsa_rh" if policy == "pypsa_mpc" else policy
        self.command_ttl_ms = command_ttl_ms
        self.shield = SafetyShield(settings)
        self.rule = RuleBasedController(settings)
        self.model: Any = None
        self.norm: Any = None
        self.policy_version = "rule-v1"
        if self.policy == "pypsa_rh":
            backend = PyPSAOperationalBackend(self.settings)
            self.controller = PyPSARollingHorizonController(backend=backend)
            self.policy_version = "pypsa-rh-v1"
        elif self.policy != "rule":
            if artifact is None:
                raise ValueError(f"{self.policy} policy requires an artifact")
            from microgrid_simulator.rl.train import ALGOS

            self.model = ALGOS[self.policy].load(str(artifact))
            stats = Path(str(artifact)).with_suffix("").as_posix() + "_vecnormalize.pkl"
            if Path(stats).exists():
                with open(stats, "rb") as file:
                    self.norm = pickle.load(file).obs_rms
            self.policy_version = Path(artifact).name

    def _rule_action(self, frame: TelemetryFrame) -> np.ndarray:
        control = self.rule.act(
            frame.to_grid_state(),
            pv_forecast_mw=(
                frame.forecast_pv_kw / 1000.0 if frame.forecast_pv_kw is not None else None
            ),
            demand_forecast_mw=(
                frame.forecast_demand_kw / 1000.0 if frame.forecast_demand_kw is not None else None
            ),
        )
        return encode_action(
            control,
            self.settings,
            self.settings.topology.n_ev,
            self.settings.diesel.enabled,
        )

    def _pypsa_rh_action(self, frame: TelemetryFrame) -> np.ndarray:
        grid_state = frame.to_grid_state()
        control = self.controller.act(grid_state)
        return encode_action(
            control,
            self.settings,
            self.settings.topology.n_ev,
            self.settings.diesel.enabled,
        )

    def _policy_action(self, frame: TelemetryFrame) -> np.ndarray:
        if self.policy == "rule":
            return self._rule_action(frame)
        if self.policy == "pypsa_rh":
            return self._pypsa_rh_action(frame)
        obs = np.asarray(frame.observation, dtype=np.float32)
        expected = tuple(self.model.observation_space.shape)
        if obs.shape != expected:
            raise ValueError(f"policy observation shape {expected} != telemetry {obs.shape}")
        if self.norm is not None:
            obs = ((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8)).clip(-10.0, 10.0)
            obs = obs.astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (frame.action_shape,) or not np.all(np.isfinite(action)):
            raise ValueError("policy returned an invalid action")
        return np.clip(action, -1.0, 1.0)

    def build_command(self, frame: TelemetryFrame) -> DispatchCommand:
        started = time.perf_counter()
        fallback = frame.data_quality != "good"
        try:
            normalized = self._rule_action(frame) if fallback else self._policy_action(frame)
        except Exception:
            normalized = self._rule_action(frame)
            fallback = True
        requested = decode_action(
            normalized,
            self.settings,
            self.settings.topology.n_ev,
            self.settings.diesel.enabled,
        )
        shielded = self.shield.apply(requested, frame)
        normalized = encode_action(
            shielded.action,
            self.settings,
            self.settings.topology.n_ev,
            self.settings.diesel.enabled,
        )
        issued = utc_now()
        return DispatchCommand(
            session_id=frame.session_id,
            telemetry_sequence_id=frame.sequence_id,
            controller=self.policy,
            policy_version=self.policy_version,
            issued_at=issued,
            expires_at=issued + timedelta(milliseconds=self.command_ttl_ms),
            normalized_action=normalized.tolist(),
            requested_battery_power_kw=shielded.action.battery_p_mw * 1000.0,
            requested_diesel_on=shielded.action.diesel_on,
            requested_diesel_power_kw=shielded.action.diesel_setpoint_mw * 1000.0,
            requested_pv_curtailment=shielded.action.pv_curtail,
            safety_shield_active=bool(shielded.reasons),
            safety_reasons=list(shielded.reasons),
            fallback_active=fallback,
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def validate_contract(self, observation_shape: int, action_shape: int) -> None:
        """Fail before a run when an artifact cannot consume the plant contract."""
        expected_action = 2 + self.settings.topology.n_ev + int(self.settings.diesel.enabled)
        if action_shape != expected_action:
            raise ValueError(
                f"plant action shape {action_shape} != configured EMS action shape "
                f"{expected_action}"
            )
        if self.model is None:
            return
        model_observation = tuple(self.model.observation_space.shape)
        model_action = tuple(self.model.action_space.shape)
        if model_observation != (observation_shape,):
            raise ValueError(
                f"policy observation shape {model_observation} != plant {(observation_shape,)}"
            )
        if model_action != (action_shape,):
            raise ValueError(f"policy action shape {model_action} != plant {(action_shape,)}")

    async def run_once(self) -> DispatchCommand:
        if self.broker is None:
            raise RuntimeError("broker transport is not configured")
        frame = await self.broker.receive_telemetry()
        command = await asyncio.to_thread(self.build_command, frame)
        await self.broker.publish_command(command)
        return command

    async def run(self) -> None:
        if self.broker is None:
            raise RuntimeError("broker transport is not configured")
        while True:
            await self.run_once()
