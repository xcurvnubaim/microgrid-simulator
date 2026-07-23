import React from "react";
import { COLORS, fmt } from "../api.js";

export function KpiStrip({ totals, meta }) {
  if (!totals) return null;
  const items = [
    ["Episode reward", fmt(totals.total_reward, 0), "", COLORS.accent],
    ["Grid import", fmt(totals.grid_import_kwh, 0), "kWh", COLORS.grid],
    ["Diesel energy", fmt(totals.diesel_kwh, 0), "kWh", COLORS.diesel],
    [
      "Diesel overgeneration",
      fmt(totals.diesel_overgeneration_kwh, 1),
      "kWh",
      COLORS.diesel,
    ],
    [
      "Peak diesel excess",
      fmt(totals.peak_diesel_overgeneration_kw, 1),
      "kW",
      COLORS.diesel,
    ],
    ["Diesel useful output", fmt(totals.diesel_useful_pct, 1), "%", COLORS.diesel],
    ["PV used", fmt(totals.pv_used_kwh, 0), "kWh", COLORS.solar],
    ["PV spilled", fmt(totals.pv_wasted_kwh, 0), "kWh", COLORS.solar],
    ["Excess generation", fmt(totals.excess_generation_kwh, 0), "kWh", COLORS.diesel],
    ["Dump-load energy", fmt(totals.dump_load_kwh, 0), "kWh", COLORS.diesel],
    ["Peak import", fmt(totals.peak_import_kw, 0), "kW", COLORS.grid],
    ["Carbon", fmt(totals.carbon_kg, 0), "kg", COLORS.diesel],
    ["Final SoC", fmt(totals.final_soc_pct, 0), "%", COLORS.battery],
  ];
  if (meta?.diesel_enabled) {
    items.push(["Diesel starts", fmt(meta.diesel_starts, 0), "", COLORS.diesel]);
  }
  if (meta?.islanded || (totals.unserved_kwh ?? 0) > 0.05) {
    items.push(["Unserved energy", fmt(totals.unserved_kwh, 0), "kWh", COLORS.unserved]);
    items.push(["Blackout", fmt(totals.blackout_hours, 2), "h", COLORS.unserved]);
  }
  if ((totals.network_loss_kwh ?? 0) > 0.001) {
    items.push(["Network losses", fmt(totals.network_loss_kwh, 2), "kWh", COLORS.load]);
  }
  if ((totals.reference_balance_absorption_kwh ?? 0) > 0.001) {
    items.push([
      "Unresolved slack sink",
      fmt(totals.reference_balance_absorption_kwh, 3),
      "kWh",
      COLORS.unserved,
    ]);
  }
  return (
    <div className="kpis">
      {items.map(([k, v, u, c]) => (
        <div className="kpi" key={k} style={{ "--kc": c }}>
          <div className="k">{k}</div>
          <div className="v">
            {v}
            {u && <span className="u">{u}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

const TABLE_COLS = [
  ["step", "step", 0],
  ["hour", "hour", 2],
  ["dispatch_rule", "dispatched rule", null],
  ["requested_battery_kw", "batt request kW", 1],
  ["requested_diesel_kw", "diesel request kW", 1],
  ["load_kw", "load kW", 1],
  ["pv_used_kw", "pv kW", 1],
  ["battery_kw", "batt kW", 1],
  ["diesel_kw", "diesel kW", 1],
  ["diesel_load_serving_kw", "diesel→load kW", 1],
  ["diesel_overgeneration_kw", "diesel excess kW", 1],
  ["excess_generation_kw", "excess kW", 1],
  ["dump_load_kw", "dump kW", 1],
  ["network_loss_kw", "loss kW", 2],
  ["reference_balance_kw", "ref balance kW", 2],
  ["grid_import_kw", "grid kW", 1],
  ["unserved_kw", "unserved kW", 1],
  ["soc_pct", "soc %", 1],
  ["min_voltage_pu", "min V pu", 3],
  ["reward", "reward", 2],
];

// Render only the newest slice of a long run — thousands of live-updating DOM
// rows get expensive. The CSV download still contains every tick.
const TABLE_ROW_LIMIT = 200;

export function StepTable({ rows }) {
  if (!rows?.length) return null;
  // Newest first, so the freshest ticks are visible without scrolling.
  const visible = (rows.length > TABLE_ROW_LIMIT ? rows.slice(-TABLE_ROW_LIMIT) : rows)
    .slice()
    .reverse();

  const downloadCsv = () => {
    const csvCell = (value) => {
      const text = value == null ? "" : String(value);
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const keys = Object.keys(rows[0]).filter(
      (k) => rows[0][k] === null || typeof rows[0][k] !== "object"
    );
    const lines = [keys.map(csvCell).join(",")];
    for (const r of rows) lines.push(keys.map((k) => csvCell(r[k])).join(","));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "microgrid_timestep_data.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Per-timestep data</span>
        <span className="note">
          {visible.length < rows.length
            ? `newest ${visible.length} of ${rows.length} ticks, latest first — CSV has all`
            : `${rows.length} ticks · latest first`}
        </span>
        <button className="dl-btn" onClick={downloadCsv} style={{ marginLeft: 10 }}>
          ↓ CSV
        </button>
      </div>
      <div className="step-table-wrap">
        <table className="step-table">
          <thead>
            <tr>
              {TABLE_COLS.map(([k, label]) => (
                <th key={k}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={`${r.episode ?? 0}:${r.step}`}>
                {TABLE_COLS.map(([k, , d]) => (
                  <td key={k}>{d == null ? r[k] : fmt(r[k], d)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
