import React, { useEffect, useMemo, useRef, useState } from "react";
import { getDefaults, simulateStream } from "./api.js";
import ConfigRail from "./components/ConfigRail.jsx";
import SLD from "./components/SLD.jsx";
import TopologyDesigner from "./components/TopologyDesigner.jsx";
import { DispatchChart, BatteryChart, PvSocChart, GridHealthChart, RewardChart, BusCharts, OutageChart, GenerationChart, OvergenerationChart } from "./components/Charts.jsx";
import { ForecastDiagnostics, KpiStrip, StepTable } from "./components/Widgets.jsx";
import { applyTopologyChange } from "./topologySettings.js";

// Keep the charts responsive on long runs: totals keep accumulating, but the
// charted window is capped (≈6 weeks of 15-min ticks) by dropping oldest rows.
const MAX_ROWS = 4032;

const TOTAL_SUM_KEYS = [
  "total_reward", "grid_import_kwh", "grid_export_kwh", "diesel_kwh", "pv_used_kwh", "pv_wasted_kwh",
  "load_kwh", "served_kwh", "unserved_kwh", "blackout_steps", "blackout_hours", "carbon_kg",
  "excess_generation_kwh", "dump_load_kwh", "network_loss_kwh",
  "diesel_load_serving_kwh", "diesel_overgeneration_kwh",
  "reference_balance_import_kwh", "reference_balance_absorption_kwh",
];
const TOTAL_MAX_KEYS = [
  "peak_unserved_kw", "peak_import_kw", "peak_load_kw", "peak_excess_generation_kw",
  "peak_diesel_overgeneration_kw",
];

function mergeTotals(prev, t) {
  if (!prev) return t;
  if (!t) return prev;
  const out = { ...prev };
  for (const k of TOTAL_SUM_KEYS) out[k] = (prev[k] ?? 0) + (t[k] ?? 0);
  for (const k of TOTAL_MAX_KEYS) out[k] = Math.max(prev[k] ?? 0, t[k] ?? 0);
  out.diesel_useful_pct =
    (out.diesel_kwh ?? 0) > 0
      ? (100 * ((out.diesel_kwh ?? 0) - (out.diesel_overgeneration_kwh ?? 0))) /
        out.diesel_kwh
      : 100;
  out.final_soc_pct = t.final_soc_pct ?? prev.final_soc_pct;
  return out;
}

// Client-side mirror of the backend's episode totals, so the KPI strip ticks
// up live while an episode is still streaming.
function liveTotals(rows, dt) {
  if (!rows.length) return null;
  const sum = (key) => rows.reduce((acc, r) => acc + (r[key] ?? 0), 0);
  const blackoutSteps = rows.reduce((acc, r) => acc + (r.blackout ? 1 : 0), 0);
  return {
    total_reward: sum("reward"),
    grid_import_kwh: sum("grid_import_positive_kw") * dt,
    grid_export_kwh: sum("grid_export_kw") * dt,
    diesel_kwh: sum("diesel_kw") * dt,
    diesel_load_serving_kwh: sum("diesel_load_serving_kw") * dt,
    diesel_overgeneration_kwh: sum("diesel_overgeneration_kw") * dt,
    diesel_useful_pct:
      sum("diesel_kw") > 0
        ? (100 * (sum("diesel_kw") - sum("diesel_overgeneration_kw"))) / sum("diesel_kw")
        : 100,
    pv_used_kwh: sum("pv_used_kw") * dt,
    pv_wasted_kwh: sum("pv_wasted_kw") * dt,
    excess_generation_kwh: sum("excess_generation_kw") * dt,
    dump_load_kwh: sum("dump_load_kw") * dt,
    network_loss_kwh: sum("network_loss_kw") * dt,
    reference_balance_import_kwh:
      rows.reduce((acc, r) => acc + Math.max(0, r.reference_balance_kw ?? 0), 0) * dt,
    reference_balance_absorption_kwh:
      rows.reduce((acc, r) => acc + Math.max(0, -(r.reference_balance_kw ?? 0)), 0) * dt,
    load_kwh: sum("load_kw") * dt,
    served_kwh: sum("served_kw") * dt,
    unserved_kwh: sum("unserved_kw") * dt,
    blackout_steps: blackoutSteps,
    blackout_hours: blackoutSteps * dt,
    peak_unserved_kw: Math.max(...rows.map((r) => r.unserved_kw ?? 0)),
    peak_import_kw: Math.max(...rows.map((r) => r.grid_import_positive_kw ?? Math.max(0, r.grid_import_kw ?? 0))),
    peak_load_kw: Math.max(...rows.map((r) => r.load_kw ?? 0)),
    peak_excess_generation_kw: Math.max(...rows.map((r) => r.excess_generation_kw ?? 0)),
    peak_diesel_overgeneration_kw: Math.max(
      ...rows.map((r) => r.diesel_overgeneration_kw ?? 0)
    ),
    final_soc_pct: rows[rows.length - 1].soc_pct ?? 0,
    carbon_kg: sum("carbon_kg"),
  };
}

export default function App() {
  const [settings, setSettings] = useState(null);
  const [policies, setPolicies] = useState(["rule", "idle", "random", "deterministic"]);
  const [demand, setDemand] = useState(null);
  const [policy, setPolicy] = useState("rule");
  const [seed, setSeed] = useState(0);
  const [error, setError] = useState(null);
  const [railOpen, setRailOpen] = useState(false);

  const [rows, setRows] = useState([]);
  const [totals, setTotals] = useState(null);
  const [meta, setMeta] = useState(null);
  const [running, setRunning] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [autoContinue, setAutoContinue] = useState(false);

  const settingsRef = useRef(null);
  const abortRef = useRef(null);
  const episodeRef = useRef(0);
  const hourOffsetRef = useRef(0);
  const lastHourRef = useRef(0);
  const lastSocRef = useRef(null);
  const autoContinueRef = useRef(false);
  const pendingRef = useRef([]);
  const flushTimerRef = useRef(null);

  useEffect(() => { settingsRef.current = settings; }, [settings]);
  useEffect(() => { autoContinueRef.current = autoContinue; }, [autoContinue]);
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    getDefaults()
      .then((d) => {
        setSettings(d.settings);
        setPolicies(d.policies);
        setDemand(d.demand);
      })
      .catch((e) => setError(e.message));
  }, []);

  const flushRows = () => {
    if (flushTimerRef.current) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    if (!pendingRef.current.length) return;
    const batch = pendingRef.current;
    pendingRef.current = [];
    setRows((prev) => {
      const next = [...prev, ...batch];
      return next.length > MAX_ROWS ? next.slice(next.length - MAX_ROWS) : next;
    });
  };

  const pushRow = (row) => {
    // Each episode restarts its clock at 0, so shift onto a global hour axis.
    const globalRow = {
      ...row,
      hour: Number((hourOffsetRef.current + row.hour).toFixed(4)),
      episode: episodeRef.current,
    };
    lastHourRef.current = globalRow.hour;
    if (row.soc_pct != null) lastSocRef.current = row.soc_pct;
    pendingRef.current.push(globalRow);
    if (!flushTimerRef.current) flushTimerRef.current = setTimeout(flushRows, 250);
  };

  const runEpisode = async (fresh) => {
    const current = settingsRef.current;
    if (!current || abortRef.current) return;
    if (fresh) {
      setRows([]);
      setTotals(null);
      setMeta(null);
      episodeRef.current = 0;
      hourOffsetRef.current = 0;
      lastHourRef.current = 0;
      lastSocRef.current = null;
      pendingRef.current = [];
    }
    setError(null);
    setRunning(true);
    setRailOpen(false);
    episodeRef.current += 1;
    setEpisode(episodeRef.current);
    hourOffsetRef.current = fresh ? 0 : lastHourRef.current;

    // Battery SoC carries over between chained episodes.
    const soc = lastSocRef.current;
    const runSettings =
      !fresh && soc != null
        ? {
            ...current,
            battery: {
              ...current.battery,
              soc_init: Math.min(
                current.battery.soc_max ?? 1,
                Math.max(current.battery.soc_min ?? 0, soc / 100),
              ),
            },
          }
        : current;

    const controller = new AbortController();
    abortRef.current = controller;
    let failed = false;
    try {
      await simulateStream(runSettings, policy, seed + episodeRef.current - 1, {
        signal: controller.signal,
        onEvent: (ev) => {
          if (ev.type === "meta") {
            setMeta((prev) => ({
              ...ev.meta,
              steps: prev?.steps ?? 0,
              diesel_starts: prev?.diesel_starts ?? 0,
              diesel_runtime_hours: prev?.diesel_runtime_hours ?? 0,
            }));
          } else if (ev.type === "row") {
            pushRow(ev.row);
          } else if (ev.type === "end") {
            setTotals((t) => mergeTotals(t, ev.totals));
            setMeta((m) => ({
              ...m,
              ...ev.meta,
              steps: (m?.steps ?? 0) + (ev.meta.steps ?? 0),
              diesel_starts: (m?.diesel_starts ?? 0) + (ev.meta.diesel_starts ?? 0),
              diesel_runtime_hours:
                (m?.diesel_runtime_hours ?? 0) + (ev.meta.diesel_runtime_hours ?? 0),
            }));
          }
        },
      });
    } catch (e) {
      failed = e.name !== "AbortError";
      if (failed) setError(e.message);
    } finally {
      flushRows();
      abortRef.current = null;
    }

    if (!failed && !controller.signal.aborted && autoContinueRef.current) {
      runEpisode(false);
      return;
    }
    setRunning(false);
  };

  const stop = () => abortRef.current?.abort();
  const reset = () => {
    setRows([]);
    setTotals(null);
    setMeta(null);
    setEpisode(0);
    episodeRef.current = 0;
    hourOffsetRef.current = 0;
    lastHourRef.current = 0;
    lastSocRef.current = null;
  };

  const hasRun = rows.length > 0 || running;
  // While an episode streams, blend the completed-episode totals with a live
  // recount of the in-flight episode's rows.
  const displayTotals = useMemo(() => {
    if (!running) return totals;
    const dt = meta?.timestep_hours ?? settings?.topology?.timestep_hours ?? 0.25;
    const current = rows.filter((r) => r.episode === episode);
    return mergeTotals(totals, liveTotals(current, dt));
  }, [running, totals, rows, episode, meta, settings]);
  // Charts get a stride-sampled view past ~700 points; the full rows still
  // feed the table, CSV, SLD scrubber, and outage/energy stats.
  const chartRows = useMemo(() => {
    const MAX_POINTS = 700;
    if (rows.length <= MAX_POINTS) return rows;
    const stride = Math.ceil(rows.length / MAX_POINTS);
    const sampled = rows.filter((_, i) => i % stride === 0);
    if (sampled[sampled.length - 1] !== rows[rows.length - 1]) {
      sampled.push(rows[rows.length - 1]);
    }
    return sampled;
  }, [rows]);
  const demandReal = demand?.available && settings?.demand?.file;
  const islanded =
    settings != null &&
    !(settings.buses ?? []).some((b) => String(b.role ?? "").toLowerCase() === "grid");
  const blackoutHours = displayTotals?.blackout_hours ?? 0;
  const updateTopology = (topology) => {
    if (!settings) return;
    setSettings(applyTopologyChange(settings, topology));
  };
  const patchSettings = (part) => {
    setSettings((current) => {
      if (!current) return current;
      const next = { ...current, ...part };
      if (part.pv_arrays || part.loads) {
        next.topology = {
          ...next.topology,
          n_pv: next.pv_arrays.length,
          n_load: next.loads.length,
        };
      }
      return next;
    });
  };

  return (
    <div className="shell">
      <header className="topbar">
        <button className="rail-toggle" onClick={() => setRailOpen((o) => !o)}>
          ☰ Scenario
        </button>
        <div className="brand">
          <h1>Microgrid Control Room</h1>
          <span className="sub">editable topology · pandapower AC physics · Region 4</span>
        </div>

        <span className={`badge ${demandReal ? "real" : "synthetic"}`}>
          {demandReal ? "● demand: historical trace" : "○ demand: synthetic sinusoid"}
        </span>
        {meta?.backend && (
          <span className="badge" title="Physics engine that solved the last run. Runtime selection is temporarily locked.">
            ⚙ engine: {meta.backend}
          </span>
        )}
        {meta?.forecast_enabled && (
          <span
            className={`badge ${meta.forecast_available ? "real" : "blackout"}`}
            title={
              meta.forecast_available
                ? `${meta.forecast_source ?? "unknown source"} · ${meta.forecast_stale ? "stale" : "fresh"} · ${meta.forecast_model_version} · ${meta.forecast_horizon_hours} h hourly PV + demand forecast`
                : meta.forecast_error || "Forecast service did not return a usable curve."
            }
          >
            {meta.forecast_available
              ? `◌ forecast: ${meta.forecast_stale ? "stale" : "fresh"}`
              : "⚠ forecast unavailable"}
          </span>
        )}
        <span className={`badge ${islanded ? "islanded" : "real"}`} title={
          islanded
            ? "No utility intertie in the topology — demand is served only by PV, diesel and battery. Add a Grid node to reconnect."
            : "Utility intertie present — the grid backfills any shortfall."
        }>
          {islanded ? "⚡ islanded" : "● grid-connected"}
        </span>
        {blackoutHours > 0 && (
          <span className="badge blackout" title="Ticks where local supply could not meet demand.">
            ⚠ blackout {blackoutHours.toFixed(2)} h
          </span>
        )}
        {hasRun && episode > 0 && (
          <span className={`badge ${running ? "live" : ""}`} title="Episodes streamed this run — battery state carries over between them.">
            {running ? "◉ live · " : ""}ep {episode} · {rows.length} ticks
          </span>
        )}

        <div className="controls">
          <label htmlFor="policy">policy</label>
          <select id="policy" value={policy} onChange={(e) => setPolicy(e.target.value)} disabled={running}>
            {policies.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <label htmlFor="seed">seed</label>
          <input
            id="seed"
            type="number"
            value={seed}
            step={1}
            disabled={running}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            style={{ width: 64 }}
          />
          <label
            className="check auto-check"
            title="Chain episodes without stopping — the battery state carries over. Uncheck (or hit Stop) to pause at the next episode boundary."
          >
            <input
              type="checkbox"
              checked={autoContinue}
              onChange={(e) => setAutoContinue(e.target.checked)}
            />
            auto-continue
          </label>
          {running ? (
            <button className="run-btn stop" onClick={stop}>
              ■ Stop
            </button>
          ) : rows.length ? (
            <>
              <button className="run-btn" onClick={() => runEpisode(false)}>
                ▶ Continue · ep {episode + 1}
              </button>
              <button className="ghost-btn" onClick={reset} title="Clear all results and start from scratch">
                ↺ Reset
              </button>
            </>
          ) : (
            <button className="run-btn" onClick={() => runEpisode(true)} disabled={!settings}>
              ▶ Run episode
            </button>
          )}
        </div>
      </header>

      <div className="main">
        {railOpen && <div className="rail-backdrop" onClick={() => setRailOpen(false)} />}
        <ConfigRail
          settings={settings}
          setSettings={setSettings}
          demand={demand}
          setDemand={setDemand}
          open={railOpen}
          onClose={() => setRailOpen(false)}
        />

        <main className="content">
          {error && (
            <div className="error-bar">
              <span>{error}</span>
              <button className="error-dismiss" onClick={() => setError(null)} aria-label="Dismiss error">
                ✕
              </button>
            </div>
          )}

          {settings && (
            <div className="panel topology-workspace">
              <div className="panel-head">
                <span className="eyebrow">Topology designer</span>
                <span className="note">
                  click a node to edit its devices · {settings.buses?.length ?? 0} buses · {settings.lines?.length ?? 0} links
                </span>
              </div>
              <TopologyDesigner
                buses={settings.buses ?? []}
                lines={settings.lines ?? []}
                onChange={updateTopology}
                settings={settings}
                onSettingsPatch={patchSettings}
                className="wide"
              />
            </div>
          )}

          {!hasRun ? (
            <div className="empty">
              <div className="big">No episode yet</div>
              Set up the topology and scenario, then run an episode — every tick
              streams into the charts live as the physics engine works through it.
              <div>
                <button className="run-btn" onClick={() => runEpisode(true)} disabled={!settings}>
                  ▶ Run episode
                </button>
              </div>
            </div>
          ) : (
            <>
              {!running && rows.length > 0 && (
                <div className="episode-banner">
                  <span>
                    Paused after episode {episode} — battery at{" "}
                    {lastSocRef.current != null ? `${lastSocRef.current.toFixed(0)}% SoC` : "—"}.
                    Continue with the next 24h window (battery state carries over) or reset.
                  </span>
                  <button className="mini-btn" onClick={() => runEpisode(false)}>
                    ▶ Continue
                  </button>
                  <button className="mini-btn" onClick={reset}>
                    ↺ Reset
                  </button>
                </div>
              )}
              <KpiStrip totals={displayTotals ?? {}} meta={meta} />
              <ForecastDiagnostics meta={meta} rows={rows} />
              <SLD rows={rows} meta={meta} />
              <PvSocChart rows={chartRows} />
              <DispatchChart rows={chartRows} peakKw={(settings?.reward?.peak_threshold_mw ?? 0) * 1000} />
              <GenerationChart rows={chartRows} meta={meta} fullRows={rows} />
              <OvergenerationChart rows={chartRows} meta={meta} fullRows={rows} />
              <OutageChart rows={chartRows} meta={meta} fullRows={rows} />
              <BusCharts rows={chartRows} meta={meta} />
              <div className="chart-grid">
                <BatteryChart rows={chartRows} />
                <GridHealthChart rows={chartRows} />
              </div>
              <RewardChart rows={chartRows} />
              <StepTable rows={rows} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
