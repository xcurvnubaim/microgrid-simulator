"""Versioned messages exchanged across the EMS event boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from microgrid_simulator.core.types import ControlAction, GridState

SCHEMA_VERSION: Literal["1.0"] = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TelemetryFrame(WireModel):
    """One solved plant state and the canonical controller observation."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    sequence_id: int = Field(ge=0)
    source_id: str
    observed_at: datetime = Field(default_factory=utc_now)
    simulation_time_hours: float = Field(ge=0.0)
    timestep_hours: float = Field(gt=0.0)
    observation: list[float]
    observation_shape: int = Field(gt=0)
    action_shape: int = Field(gt=0)
    pv_available_kw: float = Field(ge=0.0)
    pv_used_kw: float = Field(ge=0.0)
    load_demand_kw: float = Field(ge=0.0)
    load_served_kw: float = Field(ge=0.0)
    battery_soc: float = Field(ge=0.0, le=1.0)
    battery_soh: float = Field(ge=0.0, le=1.0)
    battery_power_kw: float
    diesel_on: bool
    diesel_power_kw: float = Field(ge=0.0)
    grid_connected: bool
    grid_import_kw: float
    forecast_available: bool = False
    forecast_pv_kw: float | None = Field(default=None, ge=0.0)
    forecast_demand_kw: float | None = Field(default=None, ge=0.0)
    data_quality: Literal["good", "degraded", "invalid"] = "good"

    @field_validator("observed_at")
    @classmethod
    def observed_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value

    @field_validator("observation")
    @classmethod
    def observation_is_finite(cls, value: list[float]) -> list[float]:
        import math

        if not value or not all(math.isfinite(item) for item in value):
            raise ValueError("observation must contain finite values")
        return value

    def to_grid_state(self) -> GridState:
        """Return the physical subset needed by baseline controllers."""
        return GridState(
            soc=[self.battery_soc],
            soh=[self.battery_soh],
            grid_import_mw=self.grid_import_kw / 1000.0,
            pv_available_mw=self.pv_available_kw / 1000.0,
            pv_used_mw=self.pv_used_kw / 1000.0,
            load_demand_mw=self.load_demand_kw / 1000.0,
            load_served_mw=self.load_served_kw / 1000.0,
            islanded=not self.grid_connected,
            battery_p_mw=self.battery_power_kw / 1000.0,
            diesel_p_mw=self.diesel_power_kw / 1000.0,
            diesel_on=self.diesel_on,
            timestamp=self.simulation_time_hours,
        )


class DispatchCommand(WireModel):
    """A controller decision tied to exactly one telemetry frame."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    command_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    telemetry_sequence_id: int = Field(ge=0)
    controller: Literal["rule", "sac", "ppo", "pypsa_mpc"]
    policy_version: str
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    normalized_action: list[float]
    requested_battery_power_kw: float
    requested_diesel_on: bool
    requested_diesel_power_kw: float = Field(ge=0.0)
    requested_pv_curtailment: float = Field(ge=0.0, le=1.0)
    safety_shield_active: bool = False
    safety_reasons: list[str] = Field(default_factory=list)
    fallback_active: bool = False
    inference_latency_ms: float = Field(ge=0.0)

    @field_validator("issued_at", "expires_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("command timestamps must include a timezone")
        return value

    def to_control_action(self, *, diesel_index: int = 1) -> ControlAction:
        """Expose physical setpoints without treating the wire action as plant truth."""
        del diesel_index
        return ControlAction(
            battery_p_mw=self.requested_battery_power_kw / 1000.0,
            pv_curtail=self.requested_pv_curtailment,
            diesel_on=self.requested_diesel_on,
            diesel_setpoint_mw=self.requested_diesel_power_kw / 1000.0,
        )


class DispatchResult(WireModel):
    """Acknowledgment recording the request and the plant's realized response."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    telemetry_sequence_id: int = Field(ge=0)
    command_id: str | None
    status: Literal["applied", "fallback", "rejected"]
    status_message: str
    applied_at: datetime = Field(default_factory=utc_now)
    realized_battery_power_kw: float
    realized_diesel_on: bool
    realized_diesel_power_kw: float = Field(ge=0.0)
    realized_pv_used_kw: float = Field(ge=0.0)
    unserved_load_kw: float = Field(ge=0.0)
    terminated: bool = False
    truncated: bool = False
