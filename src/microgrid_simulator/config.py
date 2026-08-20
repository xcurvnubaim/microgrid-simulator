"""Typed configuration for the microgrid simulator.

``configs/islanded-baseline-72h.yaml`` is the default runtime scenario. Every
entry point (CLI, dashboard server, RL env) must obtain settings through
:func:`load_settings`, which resolves that file (override with ``$MGS_CONFIG``
or an explicit config path). The field defaults on the models below exist only
so tests can build small in-memory fixtures; they are not a runtime scenario.

Env overrides use prefix ``MGS_`` and ``__`` as the nested delimiter, e.g.
``MGS_REWARD__W_CARBON=2.0``.

Layering rule: this is the lowest layer — nothing here imports from
``grid/``, ``model/`` or ``env``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RUNTIME_PHYSICS_ENGINE = "pandapower"


def _yaml_default(section: str, key: str) -> Any:
    """Read a schema default from the canonical scenario YAML.

    Scenario values must live in YAML. This helper is only for legacy direct
    model construction; runtime code still uses ``load_settings``.
    """
    config_path = Path(__file__).resolve().parents[2] / "configs" / "islanded-baseline-72h.yaml"
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text()) or {}
        value = raw.get(section, {}).get(key)
        if value is not None:
            return value
    raise ValueError(f"missing {section}.{key} in canonical scenario YAML")


class TopologyCfg(BaseModel):
    timestep_hours: float = 0.25
    # "ac" runs a Newton-Raphson power flow per tick (real voltages / line
    # loading); "balance" skips the solver and closes the energy balance
    # algebraically — ~100x cheaper, voltages reported as a flat 1.0 pu.
    solver: Literal["ac", "balance"] = "ac"
    n_pv: int = 2
    n_storage: int = 1
    n_ev: int = 2
    n_load: int = 3


class BusCfg(BaseModel):
    """Editable bus/node definition for the dashboard and pandapower backend.

    ``id`` is the Packet-Tracer-like stable bus number shown in the UI. The
    backend maps it to pandapower's internal bus index. ``x``/``y`` are optional
    diagram coordinates, not electrical parameters.
    """

    id: int
    name: str
    vn_kv: float = 0.4
    role: str = "bus"
    x: float | None = None
    y: float | None = None


class LineCfg(BaseModel):
    """Editable connection between two bus ids.

    Use ``kind='transformer'`` for different-voltage connections. ``kind='auto'``
    automatically creates a transformer when the two buses have different
    voltages and a line otherwise.
    """

    name: str
    from_bus: int
    to_bus: int
    length_km: float = 0.20
    max_i_ka: float = 1.0
    kind: str = "auto"


class PvArrayCfg(BaseModel):
    name: str
    bus: int = 2
    p_mw: float


class LoadCfg(BaseModel):
    name: str
    bus: int
    p_mw: float
    q_mvar: float = 0.0


class BatteryCfg(BaseModel):
    model: Literal["project", "pymgrid"] = "project"
    capacity_mwh: float = 0.50
    charge_eff: float = 0.96
    discharge_eff: float = 0.96
    max_charge_mw: float = 0.25
    max_discharge_mw: float = 0.25
    limit_basis: Literal["terminal_power", "internal_energy_per_step"] = "terminal_power"
    max_charge_internal_mwh_per_step: float | None = None
    max_discharge_internal_mwh_per_step: float | None = None
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50
    bus: int = 5  # independent storage bus; it can charge from PV, grid, or diesel
    degradation_enabled: bool = True


class EvCfg(BaseModel):
    capacity_mwh: float = 0.06
    max_charge_mw: float = 0.04
    soc_init: float = 0.35
    bus: int = 3


class DieselCfg(BaseModel):
    """Dispatchable diesel genset (plan decision 1).

    ``max_kw`` is a re-anchored placeholder (above the 352.8 kW observed peak)
    until a real nameplate spec is available. Defaults define a normal
    15-minute scheduling scenario: an illustrative 30% committed minimum, a
    nominally nonbinding 40%-of-rating-per-minute ramp, simple anti-cycling
    lockouts, and a short startup delay. They are not validated campus
    equipment parameters.
    """

    enabled: bool = Field(default_factory=lambda: _yaml_default("diesel", "enabled"))
    max_kw: float = Field(default_factory=lambda: _yaml_default("diesel", "max_kw"))
    min_kw: float = Field(default_factory=lambda: _yaml_default("diesel", "min_kw"))
    ramp_kw_per_min: float = Field(default_factory=lambda: _yaml_default("diesel", "ramp_kw_per_min"))
    start_delay_min: float = Field(default_factory=lambda: _yaml_default("diesel", "start_delay_min"))
    min_up_time_min: float = Field(default_factory=lambda: _yaml_default("diesel", "min_up_time_min"))
    min_down_time_min: float = Field(default_factory=lambda: _yaml_default("diesel", "min_down_time_min"))
    bus: int = Field(default_factory=lambda: _yaml_default("diesel", "bus"))
    carbon_kg_per_kwh: float = Field(default_factory=lambda: _yaml_default("diesel", "carbon_kg_per_kwh"))
    # Unit-commitment shutdown cost. Fuel and startup costs are shared with the
    # per-step reward through RewardCfg so MPC and RL use one objective definition.
    shut_down_cost: float = Field(default_factory=lambda: _yaml_default("diesel", "shut_down_cost"))
    initial_on: bool = Field(default_factory=lambda: _yaml_default("diesel", "initial_on"))


class GridIntertieCfg(BaseModel):
    """Utility interconnect limits. ``None`` means unconstrained (legacy behaviour)."""

    max_import_mw: float | None = None
    max_export_mw: float | None = None


class ScheduleSegmentCfg(BaseModel):
    """One clock-driven dispatch block; ``start_hour > end_hour`` wraps past
    midnight (e.g. 22 → 6)."""

    start_hour: float = Field(ge=0.0, le=24.0)
    end_hour: float = Field(ge=0.0, le=24.0)

    def contains(self, hour: float) -> bool:
        hour = hour % 24.0
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour  # overnight wrap


class DieselScheduleSegmentCfg(ScheduleSegmentCfg):
    """One clock-driven diesel dispatch block for the manual-schedule controller.

    ``level`` is either a named level — ``"max"`` (nameplate), ``"min"``
    (minimum stable load), ``"off"`` — or an explicit setpoint in kW (clamped
    to the genset's [min_kw, max_kw] band when the unit is on).
    """

    level: Literal["max", "min", "off"] | float = "max"


class DieselScheduleCfg(BaseModel):
    """Time-of-day diesel timetable consumed by ``ManualScheduleController``.

    Hours not covered by any segment mean diesel off; the first matching
    segment wins. The default reproduces the timetable hand-derived from the
    2026-01-15 window's hourly load/PV profile (full nameplate outside
    09:00-14:00, minimum stable load inside it) and is a DESIGNED scenario
    assumption, not a site fact — scenario YAMLs should override it.
    """

    segments: list[DieselScheduleSegmentCfg] = Field(
        default_factory=lambda: [
            DieselScheduleSegmentCfg(start_hour=0.0, end_hour=9.0, level="max"),
            DieselScheduleSegmentCfg(start_hour=9.0, end_hour=14.0, level="min"),
            DieselScheduleSegmentCfg(start_hour=14.0, end_hour=24.0, level="max"),
        ]
    )

    def level_at(self, hour: float) -> Literal["max", "min", "off"] | float:
        for segment in self.segments:
            if segment.contains(hour):
                return segment.level
        return "off"


class RuleCfg(BaseModel):
    """Tunable parameters for the interpretable rule controller."""

    forecast_headroom_fraction: float = Field(default=0.12, ge=0.0)
    night_discharge_start_hour: float = Field(default=18.0, ge=0.0, le=24.0)
    night_discharge_end_hour: float = Field(default=8.0, ge=0.0, le=24.0)
    night_reserve_soc_floor: float = Field(default=0.30, ge=0.0, le=1.0)


class BatteryScheduleSegmentCfg(ScheduleSegmentCfg):
    """One clock-driven battery dispatch block for the manual-schedule controller.

    ``mode`` mirrors how operators actually schedule storage:

    * ``"charge"`` — ``level`` is ``"max"`` or an explicit kW value, clamped
      to ``max_charge_mw``; the backend clips by SOC headroom and resolves the
      charging-source split (PV → diesel → grid merit order).
    * ``"discharge"`` — ``level`` is ``"max"`` or an explicit kW value,
      clamped to ``max_discharge_mw``.
    * ``"reactive"`` — charge from measured PV surplus or discharge only for
      load remaining after PV and scheduled diesel.
    * ``"idle"`` — hold (same as an uncovered hour).
    """

    mode: Literal["charge", "discharge", "reactive", "idle"] = "idle"
    level: Literal["max"] | float = "max"  # explicit levels are kW


class BatteryScheduleCfg(BaseModel):
    """Time-of-day battery timetable consumed by ``ManualScheduleController``.

    Empty segments (the default) keep the controller's reactive battery
    behaviour (charge on PV surplus, discharge to backstop the scheduled
    diesel) so existing baselines are unchanged. With segments the battery
    follows the clock instead: uncovered hours are idle; the first matching
    segment wins.
    """

    segments: list[BatteryScheduleSegmentCfg] = Field(default_factory=list)

    def segment_at(self, hour: float) -> BatteryScheduleSegmentCfg | None:
        for segment in self.segments:
            if segment.contains(hour):
                return segment
        return None


class BackendCfg(BaseModel):
    """Runtime backend settings.

    ``name`` is temporarily normalized to :data:`RUNTIME_PHYSICS_ENGINE`.
    Other backend-specific fields remain so research utilities and old scenario
    files still deserialize while the user-facing runtime is locked.
    """

    name: str = RUNTIME_PHYSICS_ENGINE
    validation_backend: str | None = None  # e.g. "pandapower"; used by eval tooling
    timestep_hours: float | None = None  # overrides topology.timestep_hours when set
    # --- PyPSA operational options ---
    horizon_hours: float = 24.0
    rolling_horizon_hours: float = 6.0
    solver: str = "highs"
    unit_commitment: bool = True


class MeasurementCfg(BaseModel):
    """One measured time series (grid meter, PV, SOC, load) for the digital twin."""

    file: str
    timestamp_column: str | None = "statstime"
    value_column: str = "value"
    unit: str = "kw"  # kw | mw | w | kwh_per_step | mwh_per_step | pct | fraction
    header_row: int = 0
    sheet: str | int | None = None  # Excel sheet, None -> first
    value_multiplier: float = 1.0  # e.g. -1 converts pymgrid load demand to positive
    synthetic_start: str | None = None  # used when the source has a positional index
    synthetic_step_hours: float | None = None
    prepend_first_as_context: bool = False


class DigitalTwinCfg(BaseModel):
    measurements: dict[str, MeasurementCfg] = Field(default_factory=dict)
    # Alignment policy: how missing samples are treated after resampling.
    fill_strategy: Literal["interpolate", "ffill", "error"] = "interpolate"
    max_gap_steps: int = 8  # refuse to silently bridge gaps longer than this
    max_missing_fraction: float = 0.05  # hard error above this share of missing data


class RLCfg(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    algo: str = "sac"  # sac (preferred for continuous battery control) | ppo
    total_timesteps: int = 50_000
    seed: int = 0
    n_envs: int | None = None  # None -> 4 for ppo, 1 for sac
    normalize: bool = True  # VecNormalize on observations/rewards
    eval_freq: int = 5_000
    eval_episodes: int = 3
    checkpoint_freq: int = 10_000
    log_dir: str = "runs"
    artifact_dir: str = "artifacts"
    train_split: Literal["train"] = "train"
    eval_split: Literal["val"] = "val"
    forecast_mode: Literal["cached", "none"] = "cached"
    device: str = "auto"  # auto | cpu | cuda / cuda:0
    # Hard unserved-load constraint (islanded outage scenario). When True, any
    # tick with unserved load above the solver-noise tolerance terminates the
    # episode immediately and applies a one-shot infeasibility penalty, rather
    # than only charging ``w_unserved`` per kWh. This is the RL-side enforcement
    # of the scenario's "load service is non-negotiable" contract; the reward
    # term remains for reporting and as a residual signal.
    hard_unserved: bool = False
    hard_unserved_penalty: float = 1000.0  # one-shot penalty applied on a blackout tick
    # Brownout threshold for the hard constraint (MW). Shortfalls below this are
    # tolerated (e.g. sub-kW command-precision residuals around the SOC floor):
    # continuous battery actions cannot hit demand to 1e-6 MW precision, and
    # terminating on every microscopic shortfall kills every episode within the
    # first ~15 steps, so training never observes a full-length episode.
    hard_unserved_tol_mw: float = 0.002  # 2 kW


class APICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501


class BatteryPerturbationCfg(BaseModel):
    """Relative (multiplicative) plant mismatch on battery parameters.

    Each field is the std of a Gaussian drawn once per episode: ``true =
    nominal * (1 + u)`` with ``u ~ N(0, std)``, clipped to ``[lo, hi]`` when
    set. ``std == 0`` disables that parameter's perturbation. These are SYNTHETIC
    engineering assumptions, not campus-calibrated values, until nameplate and
    health evidence exists.
    """

    enabled: bool = False
    capacity_std: float = 0.0
    charge_eff_std: float = 0.0
    discharge_eff_std: float = 0.0
    max_charge_std: float = 0.0
    max_discharge_std: float = 0.0
    soc_init_std: float = 0.0
    lo: float | None = None  # clip factor, e.g. 0.85
    hi: float | None = None  # clip factor, e.g. 1.15


class DieselPerturbationCfg(BaseModel):
    """Relative plant mismatch on the genset (rate/rating) parameters."""

    enabled: bool = False
    max_kw_std: float = 0.0
    min_kw_std: float = 0.0
    ramp_std: float = 0.0
    lo: float | None = None
    hi: float | None = None


class ForecastUncertaintyCfg(BaseModel):
    """Forecast-service failure and residual-noise injection.

    Residuals are additive Gaussian noise that grows with horizon, matched to
    the reduced-order forecast service (hourly values). ``dropout_prob`` is the
    per-refresh probability that a forecast update is suppressed, so the
    retained horizon goes stale and eventually turns availability off.
    """

    enabled: bool = False
    residual_pv_std: float = 0.0  # relative std of PV error at horizon 0
    residual_demand_std: float = 0.0  # relative std of demand error at horizon 0
    horizon_slope: float = 0.0  # extra relative-std growth per forecast hour
    dropout_prob: float = 0.0  # [0,1) probability an update is suppressed
    dropout_block_steps: int = Field(default=0, ge=0)  # steps a drop stays suppressed


class UncertaintyCfg(BaseModel):
    """Opt-in, dimension-preserving environment perturbation for robustness.

    ``enabled`` must be False for the nominal protocol (exact no-op). Only
    plant mismatch (battery/diesel) and forecast-service uncertainty are
    injected; action, observation, and forecast-horizon dimensions never change.
    """

    enabled: bool = False
    seed: int | None = None  # independent stream base; per-env derived if None
    battery: BatteryPerturbationCfg = Field(default_factory=BatteryPerturbationCfg)
    diesel: DieselPerturbationCfg = Field(default_factory=DieselPerturbationCfg)
    forecast: ForecastUncertaintyCfg = Field(default_factory=ForecastUncertaintyCfg)


class ScenarioCfg(BaseModel):
    name: str = "default"
    description: str = ""


class ExternalReferenceCfg(BaseModel):
    """Provenance and native parameters for an external comparison scenario."""

    implementation: Literal["pymgrid"]
    benchmark: str
    scenario_number: int = Field(ge=0)
    source_yaml: str
    source_commit: str
    reference_priority: bool = True
    native_parameters: dict[str, Any] = Field(default_factory=dict)


class DemandCfg(BaseModel):
    """Real historical demand trace (plan: demand-driver swap only)."""

    enabled: bool = True
    file: str | None = None  # xlsx or csv; None -> synthetic sinusoid fallback
    header_row: int = 1  # header on row 2 of the export -> header=1
    timestamp_column: str = "statstime"
    value_column: str = "demand"
    unit: str = "kw"
    random_window: bool = True  # plan decision 2: random window per reset()


class EpisodeCfg(BaseModel):
    horizon_hours: float = 24.0
    start_hour: float = 0.0
    # Wall-clock timestamp of the first telemetry sample evaluated by a fixed
    # replay. The environment also loads one preceding sample as controller
    # context, so the first action does not peek at the first target interval.
    telemetry_start: str | None = None
    # Terminal-SOC return-to-start contract (RL Plan §Acceptance criteria): an
    # episode whose final SOC is farther than this tolerance (fraction, e.g.
    # 0.01 = one percentage point) from the recorded starting SOC terminates
    # with a one-shot penalty of deviation * ``terminal_soc_penalty``.
    terminal_soc_tolerance: float = 0.01
    terminal_soc_penalty: float = 100.0


class ForecastCfg(BaseModel):
    """Chronos service connection and observation horizon.

    The service forecasts PV and demand in kW at ``target_frequency_hours``
    spacing over ``horizon_hours``. The simulator converts values to MW before
    appending them to the Gym observation; neither forecast substitutes for the
    plant's measured or synthetic trajectories.

    ``forecast_steps`` is *derived* from ``horizon_hours / target_frequency_hours``
    (e.g. 24 h at 0.25 h = 96 quarter-hour points). It is never configured
    directly, matching the 15-minute forecast resolution migration contract.
    """

    enabled: bool = False
    service_url: str = "http://127.0.0.1:8000"
    horizon_hours: int = Field(default=24, ge=1, le=168)
    # Spacing between predicted PV/load points.
    target_frequency_hours: float = Field(default=1.0, gt=0.0)
    # How often a new forecast snapshot is issued (aligned one-to-one with
    # controller decisions for the 15-minute migration).
    issue_frequency_hours: float = Field(default=1.0, gt=0.0)
    target: str = "pv_avg"  # PV target; retained for config compatibility
    demand_target: str = "demand"
    source_unit: Literal["kw", "mw"] = "kw"
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    refresh_each_step: bool = True
    cache_path: str | None = None
    manifest_path: str | None = None
    # Split-specific cache overrides (used for the validation/eval environment so
    # the eval env samples from its own leakage-safe cache rather than the train
    # cache). When unset, ``cache_path``/``manifest_path`` serve every split.
    val_cache_path: str | None = None
    val_manifest_path: str | None = None
    source_id: str | None = None
    strict_cache: bool = False

    @model_validator(mode="after")
    def _derive_forecast_steps(self) -> ForecastCfg:
        """Validate the horizon/frequency timing contract.

        ``forecast_steps`` must be a positive integer and the target/issue
        frequencies must not exceed the horizon. Failing here fails closed
        before any cache, observation, or scenario is built.
        """
        steps = self.horizon_hours / self.target_frequency_hours
        if not math.isclose(steps, round(steps)) or steps < 1:
            raise ValueError(
                "forecast.horizon_hours / target_frequency_hours must be a positive integer; "
                f"got {self.horizon_hours} / {self.target_frequency_hours} = {steps}"
            )
        if self.issue_frequency_hours > self.horizon_hours:
            raise ValueError(
                "forecast.issue_frequency_hours must not exceed forecast.horizon_hours"
            )
        return self

    @property
    def forecast_steps(self) -> int:
        """Number of forecast points per snapshot: ``horizon / target_frequency``."""
        return int(round(self.horizon_hours / self.target_frequency_hours))


class RewardCfg(BaseModel):
    mode: Literal["project", "pymgrid"] = "project"
    w_carbon: float = 1.0
    w_autonomy: float = 1.0
    w_health: float = 1.0
    w_waste: float = 1.0
    w_excess: float = 1.0
    w_unserved: float = 1.0
    grid_carbon_kg_per_kwh: float = 0.45
    diesel_carbon_kg_per_kwh: float = 0.70  # plan: diesel carbon reward term
    diesel_fuel_cost_per_kwh: float = 0.0  # $/kWh fuel, added to the carbon term
    diesel_start_cost: float = 0.0  # $ per engine start, penalized discretely
    import_price: float = 180.0
    peak_threshold_mw: float = 0.35
    peak_penalty: float = 200.0
    voltage_penalty: float = 100.0
    soc_violation_penalty: float = 50.0
    # Native pymgrid marginal costs, used only when mode == "pymgrid".
    pymgrid_battery_cost_cycle: float = 0.0
    pymgrid_genset_cost: float = 0.0
    pymgrid_co2_per_unit: float = 0.0
    pymgrid_cost_per_unit_co2: float = 0.0
    pymgrid_loss_load_cost: float = 0.0
    pymgrid_overgeneration_cost: float = 0.0


def _default_buses() -> list[BusCfg]:
    return [
        BusCfg(id=0, name="Grid interconnect", vn_kv=20.0, role="grid", x=70, y=150),
        BusCfg(id=1, name="Main distribution", vn_kv=0.4, role="main", x=290, y=150),
        BusCfg(id=2, name="PV yard", vn_kv=0.4, role="pv", x=520, y=60),
        BusCfg(id=3, name="EV charging hub", vn_kv=0.4, role="ev", x=520, y=240),
        BusCfg(id=4, name="Academic loads", vn_kv=0.4, role="load", x=760, y=150),
        BusCfg(id=5, name="Battery storage", vn_kv=0.4, role="battery", x=760, y=240),
    ]


def _default_lines() -> list[LineCfg]:
    return [
        LineCfg(name="20/0.4 kV service transformer", from_bus=0, to_bus=1, kind="transformer"),
        LineCfg(name="PV feeder", from_bus=1, to_bus=2, length_km=0.18),
        LineCfg(name="EV feeder", from_bus=1, to_bus=3, length_km=0.22),
        LineCfg(name="Academic feeder", from_bus=1, to_bus=4, length_km=0.28),
        LineCfg(name="Battery feeder", from_bus=1, to_bus=5, length_km=0.16),
        LineCfg(name="Campus tie", from_bus=2, to_bus=3, length_km=0.16),
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MGS_", env_nested_delimiter="__", extra="ignore")

    scenario: ScenarioCfg = Field(default_factory=ScenarioCfg)
    external_reference: ExternalReferenceCfg | None = None
    topology: TopologyCfg = Field(default_factory=TopologyCfg)
    backend: BackendCfg = Field(default_factory=BackendCfg)
    intertie: GridIntertieCfg = Field(default_factory=GridIntertieCfg)
    digital_twin: DigitalTwinCfg = Field(default_factory=DigitalTwinCfg)
    rl: RLCfg = Field(default_factory=RLCfg)
    api: APICfg = Field(default_factory=APICfg)
    battery: BatteryCfg = Field(default_factory=BatteryCfg)
    ev: EvCfg = Field(default_factory=EvCfg)
    diesel: DieselCfg = Field(default_factory=DieselCfg)
    rule: RuleCfg = Field(default_factory=RuleCfg)
    diesel_schedule: DieselScheduleCfg = Field(default_factory=DieselScheduleCfg)
    battery_schedule: BatteryScheduleCfg = Field(default_factory=BatteryScheduleCfg)
    demand: DemandCfg = Field(default_factory=DemandCfg)
    episode: EpisodeCfg = Field(default_factory=EpisodeCfg)
    forecast: ForecastCfg = Field(default_factory=ForecastCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    uncertainty: UncertaintyCfg = Field(default_factory=UncertaintyCfg)
    buses: list[BusCfg] = Field(default_factory=_default_buses)
    lines: list[LineCfg] = Field(default_factory=_default_lines)
    pv_arrays: list[PvArrayCfg] = Field(
        default_factory=lambda: [
            PvArrayCfg(name="PV array 0", bus=2, p_mw=0.12),
            PvArrayCfg(name="PV array 1", bus=2, p_mw=0.08),
        ]
    )
    loads: list[LoadCfg] = Field(
        default_factory=lambda: [
            LoadCfg(name="Lab load", bus=4, p_mw=0.16, q_mvar=0.040),
            LoadCfg(name="Library load", bus=4, p_mw=0.10, q_mvar=0.025),
            LoadCfg(name="Admin load", bus=1, p_mw=0.07, q_mvar=0.018),
        ]
    )

    @model_validator(mode="before")
    @classmethod
    def _unpack_devices_section(cls, data: Any) -> Any:
        """Allow the new-style ``devices:`` grouping alongside the flat legacy keys.

        ``devices: {battery: ..., ev: ..., diesel: ..., pv_arrays: [...], loads: [...]}``
        maps onto the top-level fields; explicit top-level keys win.
        """
        if isinstance(data, dict) and isinstance(data.get("devices"), dict):
            data = dict(data)
            devices = data.pop("devices")
            for key in ("battery", "ev", "diesel", "pv_arrays", "loads", "intertie"):
                if key in devices and key not in data:
                    data[key] = devices[key]
        return data

    @model_validator(mode="after")
    def _apply_backend_timestep(self) -> Settings:
        """``backend.timestep_hours`` (new style) overrides ``topology.timestep_hours``."""
        if self.backend.timestep_hours is not None:
            self.topology.timestep_hours = float(self.backend.timestep_hours)
        return self

    @model_validator(mode="after")
    def _lock_runtime_physics_engine(self) -> Settings:
        """Temporarily force every configured environment onto pandapower AC."""
        self.backend.name = RUNTIME_PHYSICS_ENGINE
        self.topology.solver = "ac"
        return self

    @model_validator(mode="after")
    def _derive_terminal_battery_limits(self) -> Settings:
        """Expose terminal action bounds while preserving native internal limits."""

        battery = self.battery
        if battery.model == "pymgrid" and battery.limit_basis != "internal_energy_per_step":
            raise ValueError("pymgrid battery model requires internal_energy_per_step limits")
        if battery.model == "project" and battery.limit_basis != "terminal_power":
            raise ValueError("project battery model requires terminal_power limits")
        if battery.limit_basis != "internal_energy_per_step":
            return self
        if (
            battery.max_charge_internal_mwh_per_step is None
            or battery.max_discharge_internal_mwh_per_step is None
        ):
            raise ValueError(
                "internal_energy_per_step battery limits require both internal MWh fields"
            )
        dt = float(self.topology.timestep_hours)
        if dt <= 0 or battery.charge_eff <= 0 or battery.discharge_eff <= 0:
            raise ValueError("battery timestep and efficiencies must be positive")
        battery.max_charge_mw = battery.max_charge_internal_mwh_per_step / battery.charge_eff / dt
        battery.max_discharge_mw = (
            battery.max_discharge_internal_mwh_per_step * battery.discharge_eff / dt
        )
        return self

    @model_validator(mode="after")
    def _check_bus_references(self) -> Settings:
        """Every line/device must reference a declared bus id.

        Backends used to silently remap a missing bus by role, which let the
        device sections drift out of sync with the ``buses`` list. Fail at
        load time instead, naming each stale reference.
        """
        ids = {b.id for b in self.buses}
        if not ids:
            raise ValueError("config must declare at least one bus")
        if len(ids) != len(self.buses):
            raise ValueError("bus ids must be unique")
        problems: list[str] = []
        for line in self.lines:
            for end in (line.from_bus, line.to_bus):
                if end not in ids:
                    problems.append(f"line {line.name!r} references bus {end}")
        for pv in self.pv_arrays:
            if pv.bus not in ids:
                problems.append(f"pv array {pv.name!r} references bus {pv.bus}")
        for load in self.loads:
            if load.bus not in ids:
                problems.append(f"load {load.name!r} references bus {load.bus}")
        if self.topology.n_storage > 0 and self.battery.bus not in ids:
            problems.append(f"battery references bus {self.battery.bus}")
        if self.topology.n_ev > 0 and self.ev.bus not in ids:
            problems.append(f"ev references bus {self.ev.bus}")
        if self.diesel.enabled and self.diesel.bus not in ids:
            problems.append(f"diesel references bus {self.diesel.bus}")
        if problems:
            raise ValueError(
                f"config does not match its own bus list (declared ids {sorted(ids)}): "
                + "; ".join(problems)
            )
        return self

    @classmethod
    def _load_raw_yaml(cls, path: Path) -> dict[str, Any]:
        raw = yaml.safe_load(path.read_text()) or {}
        parent = raw.pop("extends", None)
        if parent is not None:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            base = cls._load_raw_yaml(parent_path)

            def merge(target: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
                for key, value in overrides.items():
                    if isinstance(value, dict) and isinstance(target.get(key), dict):
                        target[key] = merge(target[key], value)
                    else:
                        target[key] = value
                return target

            raw = merge(base, raw)
        return raw

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        path = Path(path)
        raw = cls._load_raw_yaml(path)
        return cls(**raw)


DEFAULT_CONFIG_ENV = "MGS_CONFIG"
_CONFIG_RELPATH = Path("configs") / "islanded-baseline-72h.yaml"


def find_config_path() -> Path | None:
    """Locate the active nominal islanded campus scenario.

    Order: ``$MGS_CONFIG`` if set, then a search upward from the current
    directory, then upward from this package (covers editable installs).
    """
    env = os.environ.get(DEFAULT_CONFIG_ENV)
    if env:
        # Strip potential docker compose default artifact '-/' or leading '-'
        cleaned = env.lstrip("-") if env.startswith("-/") else env
        return Path(cleaned)
    for root in (Path.cwd(), Path(__file__).resolve()):
        for parent in (root, *root.parents):
            candidate = parent / _CONFIG_RELPATH
            if candidate.is_file():
                return candidate
    return None


def load_settings(path: str | Path | None = None) -> Settings:
    """Single runtime entry point for configuration.

    All processes (CLI, dashboard server, RL env) must come through here so
    they read the same default scenario — there is no silent fallback to the
    in-code field defaults.
    """
    resolved = Path(path) if path is not None else find_config_path()
    if resolved is None:
        raise FileNotFoundError(
            f"{_CONFIG_RELPATH} not found (searched upward from the current "
            f"directory and from the package); set ${DEFAULT_CONFIG_ENV} to point at it"
        )
    return Settings.from_yaml(resolved)
