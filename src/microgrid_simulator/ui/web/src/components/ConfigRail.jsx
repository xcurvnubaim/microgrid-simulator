import React, { useState } from "react";
import { uploadDemand, fmt, COLORS } from "../api.js";
import { Num, Check } from "./fields.jsx";

function Group({ title, accent, children, open = false }) {
  return (
    <details className="group" open={open} style={{ "--accent": accent }}>
      <summary>{title}</summary>
      <div className="group-body">{children}</div>
    </details>
  );
}

export default function ConfigRail({ settings, setSettings, demand, setDemand, open, onClose }) {
  const [uploadMsg, setUploadMsg] = useState(null);
  if (!settings) return <aside className={`rail ${open ? "open" : ""}`} />;

  const patch = (section, part) => setSettings({ ...settings, [section]: { ...settings[section], ...part } });

  const onFile = async (file) => {
    if (!file) return;
    setUploadMsg("Parsing…");
    try {
      const res = await uploadDemand(file);
      patch("demand", { file: res.file, enabled: true });
      setDemand({ available: true, stats: res.stats });
      setUploadMsg(null);
    } catch (err) {
      setUploadMsg(err.message);
    }
  };

  const stats = demand?.stats;

  return (
    <aside className={`rail ${open ? "open" : ""}`}>
      <div className="rail-title">
        Scenario
        <button className="rail-close" onClick={onClose} aria-label="Close scenario panel">
          ✕
        </button>
      </div>

      <Group title="Episode" accent={COLORS.accent} open>
        <div className="field-row">
          <Num label="Horizon (h)" value={settings.episode.horizon_hours} step={1} min={1} max={168} onChange={(v) => patch("episode", { horizon_hours: v })} />
          <label className="field">
            <span>Timestep (h)</span>
            <select value={settings.topology.timestep_hours} onChange={(e) => patch("topology", { timestep_hours: Number(e.target.value) })}>
              <option value={0.25}>0.25 — 15 min</option>
              <option value={0.5}>0.5 — 30 min</option>
              <option value={1}>1 — hourly</option>
            </select>
          </label>
        </div>
        <Num label="Start hour (synthetic mode)" value={settings.episode.start_hour} step={0.25} min={0} max={23.75} onChange={(v) => patch("episode", { start_hour: v })} />
        <label className="field">
          <span>Physics engine</span>
          <select value="pandapower" disabled aria-label="Physics engine locked to pandapower">
            <option value="pandapower">pandapower — AC power flow (locked)</option>
          </select>
        </label>
        <div style={{ fontSize: 11, color: "var(--faint)" }}>
          Temporarily hardcoded: every tick runs a pandapower Newton-Raphson AC solve.
        </div>
      </Group>

      <Group title="Demand source" accent={COLORS.load} open>
        <label className="upload">
          <input type="file" accept=".xlsx,.xlsm,.xls,.csv" onChange={(e) => onFile(e.target.files?.[0])} />
          {stats ? "Replace demand trace" : "Load real demand trace"}
          <span className="hint">xlsx / csv — header on row 2, columns statstime + demand (kW)</span>
        </label>
        <a className="mini-btn dl-template" href="/api/demand/template" download="demand_template.csv">
          ⭳ Download CSV template
        </a>
        {uploadMsg && <div style={{ fontSize: 12, color: "var(--grid)" }}>{uploadMsg}</div>}
        {stats ? (
          <div className="demand-stats">
            <span><b>{fmt(stats.rows, 0)}</b> ticks · {stats.start?.slice(0, 10)} → {stats.end?.slice(0, 10)}</span>
            <span>min <b>{fmt(stats.min_kw)}</b> · mean <b>{fmt(stats.mean_kw)}</b> · peak <b>{fmt(stats.max_kw)} kW</b></span>
          </div>
        ) : (
          <div className="demand-stats"><span>No trace loaded — synthetic sinusoid drives the loads.</span></div>
        )}
        <Check label="Random 24h window per episode" checked={settings.demand.random_window} onChange={(v) => patch("demand", { random_window: v })} />
      </Group>

      <Group title="Reward weights" accent={COLORS.grid}>
        {["w_carbon", "w_autonomy", "w_health", "w_waste", "w_unserved"].map((k) => (
          <Num key={k} label={k.replace("w_", "weight · ")} value={settings.reward[k]} step={0.1} min={0} onChange={(v) => patch("reward", { [k]: v })} />
        ))}
      </Group>

      <div className="rail-note">
        Devices — PV, battery, diesel, EV chargers and loads — are edited in the
        topology designer: click a node on the canvas to open its settings.
      </div>
    </aside>
  );
}
