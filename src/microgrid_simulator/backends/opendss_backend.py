"""OpenDSSBackend — minimal skeleton for future distribution-grid validation.

Exports the scenario topology (buses, lines, transformers, loads, PV, diesel,
battery) into OpenDSS elements via OpenDSSDirect.py and runs one snapshot solve
per tick to fill bus voltages. Dispatch physics (SoC, ramps, lockouts, energy
balance) are inherited unchanged from :class:`SimpleBackend` — OpenDSS is used
purely as an electrical validation solve.

This is intentionally minimal: unbalanced/multi-phase modelling, real line
codes, and inverter control modes are out of scope for the first version.

Requires OpenDSSDirect.py, which ships with the default install
(``pip install -e .``).
"""

from __future__ import annotations

import logging
from typing import Any

from microgrid_simulator.backends.simple_backend import SimpleBackend
from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import GridState

LOGGER = logging.getLogger(__name__)


def _require_opendss() -> Any:
    try:
        import opendssdirect as dss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "OpenDSSDirect.py is not installed. Reinstall project dependencies "
            "with `pip install -e .` to use the OpenDSS backend."
        ) from exc
    return dss


class OpenDSSBackend(SimpleBackend):
    """SimpleBackend physics + an OpenDSS snapshot solve for voltages."""

    def __init__(self, settings: Settings) -> None:
        self._dss = _require_opendss()
        self._circuit_ready = False
        super().__init__(settings)

    def _configure(self, settings: Settings) -> None:
        # SimpleBackend._configure takes an initial snapshot, but the circuit
        # export needs the topology it sets up — so skip the electrical solve
        # until the circuit exists, then backfill voltages on that snapshot.
        self._circuit_ready = False
        super()._configure(settings)
        self._build_circuit()
        self._circuit_ready = True
        self._electrical_snapshot(self._last_state)

    # -- circuit export ------------------------------------------------------
    def _cmd(self, command: str) -> None:
        result = self._dss.Text.Command(command)
        if result:  # pragma: no cover - OpenDSS reports errors as text
            LOGGER.warning("OpenDSS: %s -> %s", command, result)

    def _build_circuit(self) -> None:
        s = self.settings
        buses = s.buses
        self._bus_order = [int(b.id) for b in buses]
        by_id = {int(b.id): b for b in buses}

        ref = next(
            (b for b in buses if b.role.lower() in {"grid", "slack", "utility"}), buses[0]
        )
        self._cmd("clear")
        self._cmd(
            f"new circuit.microgrid bus1=bus{ref.id} basekv={ref.vn_kv} pu=1.0 phases=3 mvasc3=200"
        )

        for link in s.lines:
            a, b = int(link.from_bus), int(link.to_bus)
            if a not in by_id or b not in by_id:
                continue
            cfg_a, cfg_b = by_id[a], by_id[b]
            name = link.name.replace(" ", "_").replace(".", "_")
            if link.kind == "transformer" or (
                link.kind == "auto" and abs(cfg_a.vn_kv - cfg_b.vn_kv) > 1e-9
            ):
                hv, lv = (cfg_a, cfg_b) if cfg_a.vn_kv >= cfg_b.vn_kv else (cfg_b, cfg_a)
                self._cmd(
                    f"new transformer.{name} phases=3 windings=2 "
                    f"buses=(bus{hv.id}, bus{lv.id}) kvs=({hv.vn_kv}, {lv.vn_kv}) "
                    f"kvas=(1600, 1600) xhl=6"
                )
            else:
                self._cmd(
                    f"new line.{name} bus1=bus{a} bus2=bus{b} phases=3 "
                    f"length={link.length_km} units=km r1=0.05 x1=0.02 c1=0"
                )

        for i, load in enumerate(s.loads[: self.topo.n_load]):
            kv = by_id.get(int(load.bus), ref).vn_kv
            self._cmd(
                f"new load.static_{i} bus1=bus{load.bus} phases=3 kv={kv} "
                f"kw={load.p_mw * 1000.0} kvar={load.q_mvar * 1000.0} model=1"
            )
        for i, pv in enumerate(s.pv_arrays[: self.topo.n_pv]):
            kv = by_id.get(int(pv.bus), ref).vn_kv
            self._cmd(
                f"new generator.pv_{i} bus1=bus{pv.bus} phases=3 kv={kv} kw=0 pf=1 model=1"
            )
        if s.diesel.enabled:
            kv = by_id.get(int(s.diesel.bus), ref).vn_kv
            self._cmd(
                f"new generator.diesel bus1=bus{s.diesel.bus} phases=3 kv={kv} kw=0 pf=1 model=1"
            )
        # Battery as a load: kw > 0 charging, kw < 0 discharging (matches our sign).
        kv = by_id.get(int(s.battery.bus), ref).vn_kv
        self._cmd(f"new load.battery bus1=bus{s.battery.bus} phases=3 kv={kv} kw=0 kvar=0 model=1")
        # EV chargers aggregate into one load at the EV bus.
        kv = by_id.get(int(s.ev.bus), ref).vn_kv
        self._cmd(f"new load.ev bus1=bus{s.ev.bus} phases=3 kv={kv} kw=0 kvar=0 model=1")
        self._cmd(
            f"new load.dump bus1=bus{ref.id} phases=3 kv={ref.vn_kv} kw=0 kvar=0 model=1"
        )

        kvs = sorted({b.vn_kv for b in buses})
        self._cmd(f"set voltagebases={kvs}")
        self._cmd("calcvoltagebases")
        self._cmd("set mode=snapshot")

    # -- per-tick validation solve --------------------------------------------
    def _electrical_snapshot(self, state: GridState) -> None:
        if not self._circuit_ready:
            return
        dss = self._dss
        served_fraction = (
            state.load_served_mw / state.load_demand_mw
            if state.load_demand_mw > 1e-12
            else 1.0
        )
        served_fraction = max(0.0, min(1.0, served_fraction))
        for i, p_mw in enumerate(state.p_load):
            q_base = (
                self.settings.loads[i].q_mvar
                if i < len(self.settings.loads)
                else 0.0
            )
            p_base = (
                self.settings.loads[i].p_mw
                if i < len(self.settings.loads)
                else 0.0
            )
            q_mvar = q_base * (p_mw / p_base) if p_base > 1e-12 else 0.0
            self._cmd(
                f"edit load.static_{i} kw={p_mw * served_fraction * 1000.0} "
                f"kvar={q_mvar * served_fraction * 1000.0}"
            )
        for i, p_mw in enumerate(state.p_gen):
            self._cmd(f"edit generator.pv_{i} kw={p_mw * 1000.0}")
        if self.settings.diesel.enabled:
            self._cmd(f"edit generator.diesel kw={state.diesel_p_mw * 1000.0}")
        self._cmd(f"edit load.battery kw={state.battery_p_mw * 1000.0}")
        ev_kw = max(0.0, state.load_demand_mw - sum(state.p_load))
        ev_kw *= served_fraction * 1000.0
        self._cmd(f"edit load.ev kw={ev_kw}")
        self._cmd(f"edit load.dump kw={state.dump_load_mw * 1000.0}")

        dss.Solution.Solve()
        if not dss.Solution.Converged():  # pragma: no cover - solver dependent
            state.solver_ok = False
            return

        state.network_loss_mw = max(0.0, float(dss.Circuit.Losses()[0]) / 1.0e6)
        if state.islanded:
            # The OpenDSS circuit source is the numerical voltage reference.
            # With the explicit dump load it should supply active network loss,
            # not absorb unreported diesel overgeneration.
            state.reference_balance_mw = state.network_loss_mw

        # Mean per-bus voltage magnitude (pu), reported in topology bus order.
        volts: dict[str, float] = {}
        for name in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(name)
            mags = dss.Bus.puVmagAngle()[0::2]
            if mags:
                volts[name.lower()] = float(sum(mags) / len(mags))
        state.v_bus = [volts.get(f"bus{bid}", 1.0) for bid in self._bus_order]

    def close(self) -> None:
        self._cmd("clear")
