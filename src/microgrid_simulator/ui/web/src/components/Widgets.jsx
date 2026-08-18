import React, { useEffect, useRef, useState } from "react";
import { COLORS, fmt, hourLabel } from "../api.js";

export function ForecastDiagnostics({ meta, rows }) {
  if (!meta?.forecast_enabled) return null;
  const current = rows?.[rows.length - 1] ?? meta;
  const source = current.forecast_source ?? meta.forecast_source ?? "not returned";
  const requested = current.forecast_requested_source ?? meta.forecast_requested_source;
  const stale = current.forecast_stale ?? meta.forecast_stale;
  const coldStart = current.forecast_cold_start ?? meta.forecast_cold_start;
  const error = current.forecast_error ?? meta.forecast_error;
  const contextTime = current.forecast_context_time ?? meta.forecast_context_time;
  const issuedAt = current.forecast_issued_at ?? meta.forecast_issued_at;
  const contextSteps = current.forecast_context_steps ?? meta.forecast_context_steps;
  const age = current.forecast_age_steps ?? meta.forecast_age_steps;
  const covariateMode = current.forecast_covariate_mode ?? meta.forecast_covariate_mode;

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Forecast diagnostics</span>
        <span className="note">controller context only — never plant input</span>
      </div>
      <div className="demand-stats">
        <span>source <b>{source}</b>{requested && ` · expected ${requested}`}</span>
        <span>issue {issuedAt ?? "not returned"} · context {contextTime ?? "unanchored"}</span>
        <span>{stale ? "stale — advancing retained horizon" : "fresh response"} · age {age ?? 0} step(s)</span>
        <span>{coldStart ? "cold-start context" : "warm context"} · {contextSteps ?? 0} sample(s) · covariates {covariateMode ?? "unknown"}</span>
        {error && <span style={{ color: "var(--grid)" }}>forecast error: <b>{error}</b></span>}
      </div>
    </div>
  );
}

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
  ["demand_forecast_kw", "load forecast kW", 1],
  ["pv_used_kw", "pv kW", 1],
  ["pv_forecast_kw", "forecast kW", 1],
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

const TABLE_PAGE_SIZE = 50;

export function StepTable({ rows, activeHour, selectedHour, onHover, onSelect }) {
  const [page, setPage] = useState(0);
  const selectedRowRef = useRef(null);
  const pageCount = Math.ceil((rows?.length ?? 0) / TABLE_PAGE_SIZE);

  useEffect(() => {
    setPage((current) => Math.min(current, Math.max(0, pageCount - 1)));
  }, [pageCount]);

  useEffect(() => {
    if (selectedHour == null) return;
    const selectedIndex = rows?.findIndex((row) => row.hour === selectedHour) ?? -1;
    if (selectedIndex < 0) return;
    // Rows are displayed newest first, so calculate the page in that order.
    setPage(Math.floor((rows.length - 1 - selectedIndex) / TABLE_PAGE_SIZE));
  }, [rows, selectedHour]);

  useEffect(() => {
    if (selectedHour == null || !selectedRowRef.current) return;
    selectedRowRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [page, selectedHour]);

  if (!rows?.length) return null;

  const visible = rows
    .slice()
    .reverse()
    .slice(page * TABLE_PAGE_SIZE, (page + 1) * TABLE_PAGE_SIZE);

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
             {rows.length} ticks · page {page + 1} of {pageCount} · latest first
        </span>
        {(selectedHour ?? activeHour) != null && (
         <span className="step-selection">t = {hourLabel(selectedHour ?? activeHour)}</span>
        )}
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
              <tr
                key={`${r.episode ?? 0}:${r.step}`}
                className={activeHour === r.hour ? "is-hovered" : ""}
                data-selected={selectedHour === r.hour ? "true" : undefined}
                ref={selectedHour === r.hour ? selectedRowRef : null}
                onMouseEnter={() => onHover?.(r.hour)}
                onMouseLeave={() => onHover?.(null)}
                onClick={() => onSelect?.(r.hour, true)}
              >
                {TABLE_COLS.map(([k, , d]) => (
                  <td key={k}>{k === "hour" ? hourLabel(r[k]) : d == null ? r[k] : fmt(r[k], d)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pageCount > 1 && (
        <div className="step-table-pagination" aria-label="Per-timestep data pages">
          <button className="mini-btn" onClick={() => setPage(0)} disabled={page === 0}>
            First
          </button>
          <button className="mini-btn" onClick={() => setPage((current) => current - 1)} disabled={page === 0}>
            Previous
          </button>
          <span>page {page + 1} / {pageCount}</span>
          <button className="mini-btn" onClick={() => setPage((current) => current + 1)} disabled={page === pageCount - 1}>
            Next
          </button>
          <button className="mini-btn" onClick={() => setPage(pageCount - 1)} disabled={page === pageCount - 1}>
            Last
          </button>
        </div>
      )}
    </div>
  );
}
