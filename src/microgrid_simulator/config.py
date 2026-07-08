"""Typed configuration for the microgrid simulator.

``configs/simulator.yaml`` is the single source of truth at runtime. Every
entry point (CLI, dashboard server, RL env) must obtain settings through
:func:`load_settings`, which resolves that file (override with ``$MGS_CONFIG``).
The field defaults on the models below exist only so tests can build a
``Settings()`` in memory — they are not a second configuration source.

Env overrides use prefix ``MGS_`` and ``__`` as the nested delimiter, e.g.
``MGS_REWARD__W_CARBON=2.0``.

Layering rule: this is the lowest layer — nothing here imports from
``grid/``, ``model/`` or ``env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    capacity_mwh: float = 0.50
    charge_eff: float = 0.96
    discharge_eff: float = 0.96
    max_charge_mw: float = 0.25
    max_discharge_mw: float = 0.25
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50
    bus: int = 5  # independent storage bus; it can charge from PV, grid, or diesel


class EvCfg(BaseModel):
    capacity_mwh: float = 0.06
    max_charge_mw: float = 0.04
    soc_init: float = 0.35
    bus: int = 3


class DieselCfg(BaseModel):
    """Dispatchable diesel genset (plan decision 1).

    ``max_kw`` is a placeholder (~half the 352.8 kW observed peak) until a real
    nameplate spec is available; the operating constraints below default to
    typical prime-rated genset figures scaled to it.
    """

    enabled: bool = True
    max_kw: float = 150.0
    min_kw: float = 45.0  # minimum stable load (~30% of nameplate; wet stacking below)
    ramp_kw_per_min: float = 30.0  # ~20% of nameplate per minute; 0 = unlimited
    min_up_time_min: float = 30.0  # must run this long once started (anti-cycling)
    min_down_time_min: float = 15.0  # cool-down before a restart is allowed
    bus: int = 1
    carbon_kg_per_kwh: float = 0.70  # diesel genset emission intensity
    # Unit-commitment costs (used by the PyPSA operational backend/MPC baseline;
    # the per-step simulators do not price starts).
    start_up_cost: float = 0.0
    shut_down_cost: float = 0.0
    fuel_cost_per_kwh: float = 0.0  # 0 = only carbon-weighted cost in the MPC objective


class GridIntertieCfg(BaseModel):
    """Utility interconnect limits. ``None`` means unconstrained (legacy behaviour)."""

    max_import_mw: float | None = None
    max_export_mw: float | None = None


class BackendCfg(BaseModel):
    """Which simulation backend drives the environment.

    ``name``:
      * ``auto``       -> pandapower when ``topology.solver == "ac"`` and pandapower
                          is importable, otherwise the pure-python simple backend.
      * ``simple``     -> fast algebraic balance (default for RL training).
      * ``pandapower`` -> AC power flow (electrical validation).
      * ``pypsa``      -> operational backend; step physics match ``simple`` and a
                          rolling-horizon unit-commitment optimizer is exposed for
                          MPC/baseline dispatch (requires the ``ops`` extra).
      * ``opendss``    -> OpenDSS validation solve per tick (requires the ``dss`` extra).
    """

    name: str = "auto"
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
    timestamp_column: str = "statstime"
    value_column: str = "value"
    unit: str = "kw"  # kw | mw | w | kwh_per_step | pct | fraction
    header_row: int = 0
    sheet: str | int | None = None  # Excel sheet, None -> first


class DigitalTwinCfg(BaseModel):
    measurements: dict[str, MeasurementCfg] = Field(default_factory=dict)
    # Alignment policy: how missing samples are treated after resampling.
    fill_strategy: Literal["interpolate", "ffill", "error"] = "interpolate"
    max_gap_steps: int = 8  # refuse to silently bridge gaps longer than this
    max_missing_fraction: float = 0.05  # hard error above this share of missing data


class RLCfg(BaseModel):
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


class APICfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8501


class ScenarioCfg(BaseModel):
    name: str = "default"
    description: str = ""


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


class RewardCfg(BaseModel):
    w_carbon: float = 1.0
    w_autonomy: float = 1.0
    w_health: float = 1.0
    w_waste: float = 1.0
    w_unserved: float = 1.0
    grid_carbon_kg_per_kwh: float = 0.45
    diesel_carbon_kg_per_kwh: float = 0.70  # plan: diesel carbon reward term
    import_price: float = 180.0
    peak_threshold_mw: float = 0.35
    peak_penalty: float = 200.0
    voltage_penalty: float = 100.0
    soc_violation_penalty: float = 50.0


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
    model_config = SettingsConfigDict(
        env_prefix="MGS_", env_nested_delimiter="__", extra="ignore"
    )

    scenario: ScenarioCfg = Field(default_factory=ScenarioCfg)
    topology: TopologyCfg = Field(default_factory=TopologyCfg)
    backend: BackendCfg = Field(default_factory=BackendCfg)
    intertie: GridIntertieCfg = Field(default_factory=GridIntertieCfg)
    digital_twin: DigitalTwinCfg = Field(default_factory=DigitalTwinCfg)
    rl: RLCfg = Field(default_factory=RLCfg)
    api: APICfg = Field(default_factory=APICfg)
    battery: BatteryCfg = Field(default_factory=BatteryCfg)
    ev: EvCfg = Field(default_factory=EvCfg)
    diesel: DieselCfg = Field(default_factory=DieselCfg)
    demand: DemandCfg = Field(default_factory=DemandCfg)
    episode: EpisodeCfg = Field(default_factory=EpisodeCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
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
    def from_yaml(cls, path: str | Path) -> Settings:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)


DEFAULT_CONFIG_ENV = "MGS_CONFIG"
_CONFIG_RELPATH = Path("configs") / "simulator.yaml"


def find_config_path() -> Path | None:
    """Locate the canonical ``configs/simulator.yaml``.

    Order: ``$MGS_CONFIG`` if set, then a search upward from the current
    directory, then upward from this package (covers editable installs).
    """
    env = os.environ.get(DEFAULT_CONFIG_ENV)
    if env:
        return Path(env)
    for root in (Path.cwd(), Path(__file__).resolve()):
        for parent in (root, *root.parents):
            candidate = parent / _CONFIG_RELPATH
            if candidate.is_file():
                return candidate
    return None


def load_settings(path: str | Path | None = None) -> Settings:
    """Single runtime entry point for configuration.

    All processes (CLI, dashboard server, RL env) must come through here so
    they read the same ``configs/simulator.yaml`` — there is no silent
    fallback to the in-code field defaults.
    """
    resolved = Path(path) if path is not None else find_config_path()
    if resolved is None:
        raise FileNotFoundError(
            "configs/simulator.yaml not found (searched upward from the current "
            f"directory and from the package); set ${DEFAULT_CONFIG_ENV} to point at it"
        )
    return Settings.from_yaml(resolved)
