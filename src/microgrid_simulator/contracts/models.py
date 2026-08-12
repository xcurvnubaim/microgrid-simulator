"""Messages shared by telemetry, simulator, dashboard, forecaster, and EMS."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from microgrid_simulator.ems.types import DispatchCommand, DispatchResult, TelemetryFrame

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TelemetrySessionRequest(ContractModel):
    """Request one deterministic historical replay window."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    start_at: str | None = None
    n_steps: int | None = Field(default=None, ge=1)


class TelemetrySample(ContractModel):
    """Measured exogenous input; simulated outputs deliberately do not belong here."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    source_id: str
    session_id: str
    sequence_id: int = Field(ge=0)
    observed_at: str
    pv_available_kw: float = Field(ge=0.0)
    load_demand_kw: float = Field(ge=0.0)
    battery_soc: float | None = Field(default=None, ge=0.0, le=1.0)
    weather: dict[str, float] = Field(default_factory=dict)
    data_quality: Literal["good", "degraded", "invalid"] = "good"

    @field_validator("pv_available_kw", "load_demand_kw")
    @classmethod
    def power_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("telemetry power values must be finite")
        return value

    @field_validator("weather")
    @classmethod
    def weather_is_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(item) for item in value.values()):
            raise ValueError("weather values must be finite")
        return value


class TelemetrySession(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    n_steps: int = Field(ge=1)
    sample_count: int = Field(ge=2)
    context_time: str
    first_evaluated_time: str
    last_evaluated_time: str
    timestamps_are_observed: bool

    @model_validator(mode="after")
    def context_plus_steps(self) -> TelemetrySession:
        if self.sample_count != self.n_steps + 1:
            raise ValueError("telemetry sessions require one context sample plus n_steps")
        return self


class TelemetryWindowPayload(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session: TelemetrySession
    samples: list[TelemetrySample]
    source_files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def samples_match_session(self) -> TelemetryWindowPayload:
        if len(self.samples) != self.session.sample_count:
            raise ValueError("sample count does not match telemetry session")
        if any(sample.session_id != self.session.session_id for sample in self.samples):
            raise ValueError("all samples must belong to the telemetry session")
        if [sample.sequence_id for sample in self.samples] != list(range(len(self.samples))):
            raise ValueError("telemetry sample sequence must be contiguous from zero")
        return self


class PlantObservation(ContractModel):
    """Simulator-owned state and canonical controller observation sent to the EMS."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    session_id: str
    sequence_id: int = Field(ge=0)
    source_id: str
    observed_at: datetime
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
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def dimensions_and_values_are_valid(self) -> PlantObservation:
        if len(self.observation) != self.observation_shape:
            raise ValueError("observation length does not match observation_shape")
        if not all(math.isfinite(item) for item in self.observation):
            raise ValueError("observation must contain finite values")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return self

    @classmethod
    def from_telemetry_frame(cls, frame: TelemetryFrame) -> PlantObservation:
        return cls(**frame.model_dump())

    def to_telemetry_frame(self) -> TelemetryFrame:
        return TelemetryFrame(**self.model_dump(exclude={"constraints"}))


class PlantPhysicalState(ContractModel):
    """State that must survive between chronological EMS episode windows."""

    battery_soc: float = Field(ge=0.0, le=1.0)
    battery_soh: float = Field(ge=0.0, le=1.0)
    battery_throughput_mwh: float = Field(ge=0.0)
    diesel_on: bool
    diesel_power_mw: float = Field(ge=0.0)
    diesel_runtime_hours: float = Field(ge=0.0)
    diesel_starts: int = Field(ge=0)
    diesel_hours_in_state: float = Field(ge=0.0)


class PlantFrame(ContractModel):
    """Raw solved plant frame. EMS turns this into the controller observation contract."""

    sequence_id: int = Field(ge=0)
    source_id: str
    observed_at: datetime
    simulation_time_hours: float = Field(ge=0.0)
    observation: list[float]
    action_shape: int = Field(gt=0)
    state: dict[str, Any]
    info: dict[str, Any] = Field(default_factory=dict)
    physical_state: PlantPhysicalState


class PlantSessionStartRequest(ContractModel):
    session_id: str
    episode_index: int = Field(ge=0)
    seed: int
    config_fingerprint: str
    telemetry: TelemetryWindowPayload
    carried_state: PlantPhysicalState | None = None


class PlantSessionStartResponse(ContractModel):
    session_id: str
    episode_index: int = Field(ge=0)
    frame: PlantFrame
    meta: dict[str, Any] = Field(default_factory=dict)


class PlantStepRequest(ContractModel):
    session_id: str
    episode_index: int = Field(ge=0)
    sequence_id: int = Field(ge=0)
    command: DispatchCommand


class PlantStepResponse(ContractModel):
    session_id: str
    episode_index: int = Field(ge=0)
    sequence_id: int = Field(ge=0)
    frame: PlantFrame
    result: DispatchResult
    reward: float
    row: dict[str, Any]
    terminated: bool = False
    truncated: bool = False


class PlantSessionCloseRequest(ContractModel):
    session_id: str


class PlantSessionCloseResponse(ContractModel):
    session_id: str
    closed: bool
