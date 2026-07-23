import React from "react";
import { COLORS } from "../api.js";
import { Num, Check } from "./fields.jsx";
import ScheduleGraph from "./ScheduleGraph.jsx";
import BatteryScheduleGraph, { DEFAULT_BATTERY_SEGMENTS } from "./BatteryScheduleGraph.jsx";

function Section({ title, accent, children }) {
  return (
    <div className="asset-section" style={{ "--accent": accent }}>
      <div className="asset-section-head">{title}</div>
      {children}
    </div>
  );
}

/* Editors for every device attached to the selected bus. Singleton devices
   (battery, diesel, EV) follow their bus assignment; PV arrays and loads are
   the list entries whose `bus` matches. */
export function BusAssets({ bus, settings, onPatch }) {
  const id = Number(bus.id);
  const patchSection = (section, part) =>
    onPatch({ [section]: { ...settings[section], ...part } });

  const pvHere = settings.pv_arrays
    .map((pv, index) => ({ pv, index }))
    .filter(({ pv }) => Number(pv.bus) === id);
  const loadsHere = settings.loads
    .map((load, index) => ({ load, index }))
    .filter(({ load }) => Number(load.bus) === id);
  const batteryHere = Number(settings.battery.bus) === id;
  const dieselHere = Number(settings.diesel.bus) === id;
  const evHere = Number(settings.ev.bus) === id;

  const setPv = (index, part) =>
    onPatch({
      pv_arrays: settings.pv_arrays.map((pv, i) => (i === index ? { ...pv, ...part } : pv)),
    });
  const setLoad = (index, part) =>
    onPatch({
      loads: settings.loads.map((load, i) => (i === index ? { ...load, ...part } : load)),
    });

  const addPv = () =>
    onPatch({
      pv_arrays: [
        ...settings.pv_arrays,
        { name: `PV array ${settings.pv_arrays.length}`, bus: id, p_mw: 0.05 },
      ],
    });
  const addLoad = () =>
    onPatch({
      loads: [
        ...settings.loads,
        { name: `Load ${settings.loads.length}`, bus: id, p_mw: 0.05, q_mvar: 0.01 },
      ],
    });

  const empty = !pvHere.length && !loadsHere.length && !batteryHere && !dieselHere && !evHere;

  return (
    <div className="bus-assets">
      <div className="mini-head"><span>Devices on this bus</span></div>

      {batteryHere && (
        <Section title="Battery" accent={COLORS.battery}>
          <div className="field-row">
            <Num label="Capacity (MWh)" value={settings.battery.capacity_mwh} step={0.05} min={0.05} onChange={(v) => patchSection("battery", { capacity_mwh: v })} />
            <Num label="Initial SoC" value={settings.battery.soc_init} min={0} max={1} onChange={(v) => patchSection("battery", { soc_init: v })} />
          </div>
          <div className="field-row">
            <Num label="Max charge (MW)" value={settings.battery.max_charge_mw} min={0} onChange={(v) => patchSection("battery", { max_charge_mw: v })} />
            <Num label="Max discharge (MW)" value={settings.battery.max_discharge_mw} min={0} onChange={(v) => patchSection("battery", { max_discharge_mw: v })} />
          </div>
          <div className="field-row">
            <Num label="SoC min" value={settings.battery.soc_min} min={0} max={1} onChange={(v) => patchSection("battery", { soc_min: v })} />
            <Num label="SoC max" value={settings.battery.soc_max} min={0} max={1} onChange={(v) => patchSection("battery", { soc_max: v })} />
          </div>
          <Check
            label="Manual timetable (schedule policy)"
            checked={(settings.battery_schedule?.segments ?? []).length > 0}
            onChange={(v) =>
              onPatch({ battery_schedule: { segments: v ? DEFAULT_BATTERY_SEGMENTS : [] } })
            }
          />
          {(settings.battery_schedule?.segments ?? []).length > 0 && (
            <BatteryScheduleGraph settings={settings} onPatch={onPatch} />
          )}
        </Section>
      )}

      {dieselHere && (
        <Section title="Diesel genset" accent={COLORS.diesel}>
          <Check label="Enabled" checked={settings.diesel.enabled} onChange={(v) => patchSection("diesel", { enabled: v })} />
          <div className="field-row">
            <Num label="Max output (kW)" value={settings.diesel.max_kw} step={10} min={0} onChange={(v) => patchSection("diesel", { max_kw: v })} />
            <Num label="Min stable (kW)" value={settings.diesel.min_kw} step={5} min={0} onChange={(v) => patchSection("diesel", { min_kw: v })} />
          </div>
          <Num label="Carbon (kgCO₂/kWh)" value={settings.reward.diesel_carbon_kg_per_kwh} step={0.05} min={0} onChange={(v) => patchSection("reward", { diesel_carbon_kg_per_kwh: v })} />
          {settings.diesel.enabled && (
            <>
              <div className="asset-section-head" style={{ marginTop: 4 }}>Manual timetable</div>
              <ScheduleGraph settings={settings} onPatch={onPatch} />
            </>
          )}
        </Section>
      )}

      {evHere && (
        <Section title="EV chargers" accent={COLORS.volt}>
          <Num label="Chargers" value={settings.topology.n_ev} step={1} min={0} max={8} onChange={(v) => patchSection("topology", { n_ev: Math.round(v) })} />
          <div className="field-row">
            <Num label="Each max (MW)" value={settings.ev.max_charge_mw} min={0} onChange={(v) => patchSection("ev", { max_charge_mw: v })} />
            <Num label="Each capacity (MWh)" value={settings.ev.capacity_mwh} min={0.01} onChange={(v) => patchSection("ev", { capacity_mwh: v })} />
          </div>
        </Section>
      )}

      {pvHere.map(({ pv, index }) => (
        <Section key={`pv-${index}`} title={pv.name || `PV array ${index}`} accent={COLORS.solar}>
          <label className="field">
            <span>Name</span>
            <input type="text" value={pv.name} onChange={(e) => setPv(index, { name: e.target.value })} />
          </label>
          <div className="asset-row-foot">
            <Num label="Peak (MW)" value={pv.p_mw} min={0} onChange={(v) => setPv(index, { p_mw: v })} />
            <button
              type="button"
              className="mini-btn danger"
              onClick={() => onPatch({ pv_arrays: settings.pv_arrays.filter((_, i) => i !== index) })}
            >
              Remove
            </button>
          </div>
        </Section>
      ))}

      {loadsHere.map(({ load, index }) => (
        <Section key={`load-${index}`} title={load.name || `Load ${index}`} accent={COLORS.load}>
          <label className="field">
            <span>Name</span>
            <input type="text" value={load.name} onChange={(e) => setLoad(index, { name: e.target.value })} />
          </label>
          <div className="field-row">
            <Num label="Base (MW)" value={load.p_mw} min={0} onChange={(v) => setLoad(index, { p_mw: v })} />
            <Num label="Q (MVAr)" value={load.q_mvar ?? 0} onChange={(v) => setLoad(index, { q_mvar: v })} />
          </div>
          <div className="asset-row-foot">
            <span className="asset-hint">With a real trace, base MW sets this load's share of the total.</span>
            <button
              type="button"
              className="mini-btn danger"
              onClick={() => onPatch({ loads: settings.loads.filter((_, i) => i !== index) })}
            >
              Remove
            </button>
          </div>
        </Section>
      ))}

      {empty && (
        <div className="asset-hint">
          No devices here. Change the bus role to battery / diesel / EV to move that
          device onto it, or attach one below.
        </div>
      )}

      <div className="asset-attach">
        <button type="button" className="mini-btn" onClick={addPv}>+ PV array</button>
        <button type="button" className="mini-btn" onClick={addLoad}>+ Load</button>
      </div>
    </div>
  );
}

/* Shown when nothing is selected: every device in the scenario and the bus it
   lives on. Clicking an entry selects that bus on the canvas. */
export function AssetDirectory({ settings, onSelectBus }) {
  const busName = (id) => {
    const bus = (settings.buses ?? []).find((b) => Number(b.id) === Number(id));
    return bus ? `bus ${bus.id} · ${bus.name}` : `bus ${id}`;
  };
  const entries = [
    { label: settings.diesel.enabled ? "Diesel genset" : "Diesel (off)", bus: settings.diesel.bus, color: COLORS.diesel },
    { label: "Battery", bus: settings.battery.bus, color: COLORS.battery },
    { label: `EV chargers ×${settings.topology.n_ev}`, bus: settings.ev.bus, color: COLORS.volt },
    ...settings.pv_arrays.map((pv) => ({ label: pv.name, bus: pv.bus, color: COLORS.solar })),
    ...settings.loads.map((load) => ({ label: load.name, bus: load.bus, color: COLORS.load })),
  ];

  return (
    <div className="asset-directory">
      <div className="mini-head"><span>Devices</span></div>
      <div className="asset-hint">
        Click a node on the canvas — or a device below — to edit it here.
      </div>
      {entries.map((entry, i) => (
        <button
          key={i}
          type="button"
          className="asset-chip"
          style={{ "--accent": entry.color }}
          onClick={() => onSelectBus(entry.bus)}
        >
          <span className="asset-chip-name">{entry.label}</span>
          <span className="asset-chip-bus">{busName(entry.bus)}</span>
        </button>
      ))}
      <div className="asset-hint dim">
        Move: drag nodes · Link: click two buses · Delete / Backspace removes the selection.
      </div>
    </div>
  );
}
