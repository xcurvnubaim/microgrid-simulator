import React from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import { COLORS, THEME, fmt, hourLabel } from "../api.js";

const axis = { stroke: THEME.faint, fontSize: 10.5, fontFamily: "IBM Plex Mono" };
// Shared cursor: every chart carries the same syncId so hovering one shows the
// tooltip + vertical cursor at the same time step on all of them. Match by the
// `hour` value (not array index) so downsampled and full-resolution panels stay
// aligned on the real time step.
const SYNC = { syncId: "microgrid", syncMethod: "value" };
const tooltipStyle = {
  contentStyle: {
    background: THEME.panel,
    border: `1px solid ${THEME.line}`,
    color: THEME.text,
    borderRadius: 8,
    fontFamily: "IBM Plex Mono",
    fontSize: 11.5,
  },
  labelFormatter: (h) => `t = ${hourLabel(h)}`,
  formatter: (v, name) => [fmt(v, 1), name],
  cursor: { stroke: THEME.muted, strokeWidth: 1, strokeDasharray: "3 3" },
};

function Panel({ title, note, height = 260, children }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">{title}</span>
        {note && <span className="note">{note}</span>}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}

export function DispatchChart({ rows, peakKw }) {
  return (
    <Panel
      title="Dispatch — who serves the load"
      note="stacked supply vs demand · battery discharge counts as supply"
      height={300}
    >
      <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis tick={axis} unit=" kW" width={64} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <Area isAnimationActive={false}
          stackId="supply"
          dataKey="pv_used_kw"
          name="PV"
          fill={COLORS.solar}
          stroke={COLORS.solar}
          fillOpacity={0.55}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="supply"
          dataKey={(r) => Math.max(0, -r.battery_kw)}
          name="Battery discharge"
          fill={COLORS.battery}
          stroke={COLORS.battery}
          fillOpacity={0.5}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="supply"
          dataKey="diesel_kw"
          name="Diesel"
          fill={COLORS.diesel}
          stroke={COLORS.diesel}
          fillOpacity={0.55}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="supply"
          dataKey={(r) => Math.max(0, r.grid_import_kw)}
          name="Grid import"
          fill={COLORS.grid}
          stroke={COLORS.grid}
          fillOpacity={0.42}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="supply"
          dataKey={(r) => Math.max(0, r.unserved_kw ?? 0)}
          name="Unserved (blackout)"
          fill={COLORS.unserved}
          stroke={COLORS.unserved}
          fillOpacity={0.55}
          type="monotone"
        />
        <Line isAnimationActive={false}
          dataKey="load_kw"
          name="Campus demand"
          stroke={THEME.text}
          strokeWidth={1.8}
          dot={false}
          type="monotone"
        />
        {rows.some((row) => row.demand_forecast_kw != null) && (
          <Line
            isAnimationActive={false}
            dataKey="demand_forecast_kw"
            name="Chronos demand forecast"
            stroke={COLORS.accent}
            strokeWidth={1.6}
            strokeDasharray="6 4"
            dot={false}
            connectNulls={false}
            type="stepAfter"
          />
        )}
        {peakKw > 0 && (
          <ReferenceLine
            y={peakKw}
            stroke={COLORS.grid}
            strokeDasharray="5 4"
            label={{ value: "peak cap", fill: COLORS.grid, fontSize: 10, position: "insideTopRight" }}
          />
        )}
      </ComposedChart>
    </Panel>
  );
}

export function BatteryChart({ rows }) {
  return (
    <Panel title="Battery" note="SoC band vs power command">
      <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis yAxisId="soc" tick={axis} unit=" %" domain={[0, 100]} width={52} />
        <YAxis yAxisId="kw" orientation="right" tick={axis} unit=" kW" width={58} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <Area isAnimationActive={false}
          yAxisId="kw"
          dataKey="battery_kw"
          name="Battery power (＋charge)"
          fill={COLORS.battery}
          stroke={COLORS.battery}
          fillOpacity={0.25}
          type="stepAfter"
        />
        <Line isAnimationActive={false}
          yAxisId="soc"
          dataKey="soc_pct"
          name="SoC"
          stroke={THEME.text}
          strokeWidth={1.8}
          dot={false}
          type="monotone"
        />
      </ComposedChart>
    </Panel>
  );
}


export function PvSocChart({ rows }) {
  const hasForecast = rows.some((row) => row.pv_forecast_kw != null);
  return (
    <Panel
      title="PV and battery SoC"
      note={
        hasForecast
          ? "Chronos forecast vs PV availability/used and storage level"
          : "PV availability/used vs storage level"
      }
      height={280}
    >
      <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis yAxisId="kw" tick={axis} unit=" kW" width={64} />
        <YAxis yAxisId="soc" orientation="right" tick={axis} unit=" %" domain={[0, 100]} width={52} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <Area isAnimationActive={false}
          yAxisId="kw"
          dataKey="pv_available_kw"
          name="PV available"
          fill={COLORS.solar}
          stroke={COLORS.solar}
          fillOpacity={0.18}
          type="monotone"
        />
        <Line isAnimationActive={false}
          yAxisId="kw"
          dataKey="pv_used_kw"
          name="PV used"
          stroke={COLORS.solar}
          strokeWidth={1.8}
          dot={false}
          type="monotone"
        />
        {hasForecast && (
          <Line
            isAnimationActive={false}
            yAxisId="kw"
            dataKey="pv_forecast_kw"
            name="Chronos PV forecast"
            stroke={COLORS.accent}
            strokeWidth={1.6}
            strokeDasharray="6 4"
            dot={false}
            connectNulls={false}
            type="stepAfter"
          />
        )}
        <Line isAnimationActive={false}
          yAxisId="soc"
          dataKey="soc_pct"
          name="Battery SoC"
          stroke={COLORS.battery}
          strokeWidth={1.9}
          dot={false}
          type="monotone"
        />
      </ComposedChart>
    </Panel>
  );
}

export function GridHealthChart({ rows }) {
  return (
    <Panel title="Grid health" note="voltage band 0.95–1.05 pu">
      <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis yAxisId="v" tick={axis} domain={[0.9, 1.1]} width={52} />
        <YAxis yAxisId="pct" orientation="right" tick={axis} unit=" %" width={52} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <ReferenceLine yAxisId="v" y={0.95} stroke={COLORS.grid} strokeDasharray="4 4" />
        <ReferenceLine yAxisId="v" y={1.05} stroke={COLORS.grid} strokeDasharray="4 4" />
        <Line isAnimationActive={false}
          yAxisId="v"
          dataKey="min_voltage_pu"
          name="Min bus voltage (pu)"
          stroke={COLORS.volt}
          strokeWidth={1.6}
          dot={false}
          type="monotone"
        />
        <Line isAnimationActive={false}
          yAxisId="pct"
          dataKey="max_line_loading_pct"
          name="Max line loading"
          stroke={COLORS.load}
          strokeWidth={1.4}
          dot={false}
          type="monotone"
        />
      </ComposedChart>
    </Panel>
  );
}

// ── demand vs power serving load ──────────────────────────────────────────
export function GenerationChart({ rows, meta, fullRows }) {
  if (!rows.length) return null;
  const dt = meta?.timestep_hours ?? 0.25;
  const grossGenerationKw = (r) =>
    r.gross_generation_kw ??
    (r.pv_used_kw ?? 0) +
      (r.diesel_kw ?? 0) +
      Math.max(0, -(r.battery_kw ?? 0)) +
      Math.max(0, r.grid_import_kw ?? 0);
  const deficitKw = (r) => {
    if (r.unserved_kw != null) return Math.max(0, r.unserved_kw);
    return Math.max(0, r.load_kw - grossGenerationKw(r));
  };
  const loadServingSupplyKw = (r) =>
    r.load_serving_supply_kw ?? r.served_kw ?? Math.max(0, (r.load_kw ?? 0) - deficitKw(r));
  const data = rows.map((r) => {
    const deficit = deficitKw(r);
    const serving = loadServingSupplyKw(r);
    const gross = grossGenerationKw(r);
    return {
      ...r,
      load_serving_supply_kw: serving,
      gross_generation_kw: gross,
      deficit_kw: deficit,
      served_base_kw: serving,
    };
  });
  // Energy totals from the full-resolution rows; the chart itself may be
  // downsampled for performance.
  const statRows = fullRows ?? rows;
  const servedKwh = statRows.reduce((a, r) => a + loadServingSupplyKw(r), 0) * dt;
  const grossKwh = statRows.reduce((a, r) => a + grossGenerationKw(r), 0) * dt;
  const loadKwh = statRows.reduce((a, r) => a + r.load_kw, 0) * dt;
  const deficitKwh = statRows.reduce((a, r) => a + deficitKw(r), 0) * dt;
  const hasDeficit = data.some((d) => d.deficit_kw > 1e-6);
  const hasGrossGap = data.some((d) => Math.abs(d.gross_generation_kw - d.load_serving_supply_kw) > 1e-6);
  const note =
    `served ${fmt(servedKwh, 0)} kWh vs demand ${fmt(loadKwh, 0)} kWh` +
    (hasGrossGap ? ` · gross supply ${fmt(grossKwh, 0)} kWh` : "") +
    (hasDeficit ? ` · ${fmt(deficitKwh, 0)} kWh unserved (tinted)` : "");

  return (
    <Panel title="Demand vs power serving load" note={note} height={260}>
      <ComposedChart data={data} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis tick={axis} unit=" kW" width={62} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <Area isAnimationActive={false}
          dataKey="load_serving_supply_kw"
          name="Power serving load"
          fill={COLORS.ok}
          stroke={COLORS.ok}
          fillOpacity={0.28}
          strokeWidth={1.8}
          type="monotone"
        />
        {hasGrossGap && (
          <Line isAnimationActive={false}
            dataKey="gross_generation_kw"
            name="Gross supply"
            stroke={COLORS.ok}
            strokeWidth={1.3}
            strokeDasharray="5 3"
            dot={false}
            type="monotone"
          />
        )}
        {hasDeficit && (
          // Invisible base lifts the deficit band so it spans served → demand.
          <Area isAnimationActive={false}
            stackId="deficit"
            dataKey="served_base_kw"
            legendType="none"
            tooltipType="none"
            fill="none"
            stroke="none"
            activeDot={false}
            type="monotone"
          />
        )}
        {hasDeficit && (
          <Area isAnimationActive={false}
            stackId="deficit"
            dataKey="deficit_kw"
            name="Deficit (unmet demand)"
            fill={COLORS.unserved}
            stroke={COLORS.unserved}
            fillOpacity={0.45}
            type="monotone"
          />
        )}
        <Line isAnimationActive={false}
          dataKey="load_kw"
          name="Demand"
          stroke={THEME.text}
          strokeWidth={1.8}
          dot={false}
          type="monotone"
        />
      </ComposedChart>
    </Panel>
  );
}

// ── source-specific overgeneration accounting ──────────────────────────────
export function OvergenerationChart({ rows, meta, fullRows }) {
  if (!rows.length) return null;
  const dt = meta?.timestep_hours ?? 0.25;
  const statRows = fullRows ?? rows;
  const energy = (key) =>
    statRows.reduce((total, row) => total + Math.max(0, row[key] ?? 0), 0) * dt;
  const dieselKwh = energy("diesel_kw");
  const dieselExcessKwh = energy("diesel_overgeneration_kw");
  const dumpKwh = energy("dump_load_kw");
  const lossKwh = energy("network_loss_kw");
  const dieselUsefulPct =
    dieselKwh > 1e-9 ? (100 * (dieselKwh - dieselExcessKwh)) / dieselKwh : 100;
  const dieselExcessPct =
    dieselKwh > 1e-9 ? (100 * dieselExcessKwh) / dieselKwh : 0;
  const peakDieselExcessKw = Math.max(
    ...statRows.map((row) => row.diesel_overgeneration_kw ?? 0)
  );
  const note =
    `total ${fmt(dieselKwh, 1)} kWh` +
    ` · useful ${fmt(dieselUsefulPct, 1)}%` +
    ` · excess ${fmt(dieselExcessPct, 1)}% (${fmt(dieselExcessKwh, 1)} kWh)` +
    ` · peak ${fmt(peakDieselExcessKw, 1)} kW` +
    ` · excess → dump ${fmt(dumpKwh, 1)} kWh` +
    (lossKwh > 0.001 ? ` · AC loss ${fmt(lossKwh, 2)} kWh` : "");

  return (
    <Panel title="Diesel output allocation" note={note} height={280}>
      <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis tick={axis} unit=" kW" width={62} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        <ReferenceLine y={0} stroke={THEME.faint} />
        <Area isAnimationActive={false}
          stackId="diesel-allocation"
          dataKey="diesel_load_serving_kw"
          name="Diesel → load"
          fill={COLORS.ok}
          stroke={COLORS.ok}
          fillOpacity={0.42}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="diesel-allocation"
          dataKey="battery_charge_from_diesel_kw"
          name="Diesel → battery"
          fill={COLORS.battery}
          stroke={COLORS.battery}
          fillOpacity={0.42}
          type="monotone"
        />
        <Area isAnimationActive={false}
          stackId="diesel-allocation"
          dataKey="diesel_overgeneration_kw"
          name="Diesel excess"
          fill={COLORS.unserved}
          stroke={COLORS.unserved}
          fillOpacity={0.38}
          type="monotone"
        />
        <Line isAnimationActive={false}
          dataKey="diesel_kw"
          name="Total diesel output"
          stroke={COLORS.diesel}
          strokeWidth={2}
          dot={false}
          type="monotone"
        />
        <Line isAnimationActive={false}
          dataKey="dump_load_kw"
          name="Dump sink (= excess here)"
          stroke={COLORS.volt}
          strokeWidth={1.5}
          strokeDasharray="5 3"
          dot={false}
          type="monotone"
        />
        <Line isAnimationActive={false}
          dataKey="network_loss_kw"
          name="AC network loss"
          stroke={COLORS.load}
          strokeWidth={1.3}
          strokeDasharray="4 3"
          dot={false}
          type="monotone"
        />
      </ComposedChart>
    </Panel>
  );
}

// ── power outage timeline ─────────────────────────────────────────────────
function outageEvents(rows, dt) {
  const events = [];
  let cur = null;
  rows.forEach((r) => {
    if (r.blackout) {
      if (!cur) cur = { start: r.hour, end: r.hour, peak_kw: 0, kwh: 0 };
      cur.end = r.hour;
      cur.peak_kw = Math.max(cur.peak_kw, r.unserved_kw ?? 0);
      cur.kwh += (r.unserved_kw ?? 0) * dt;
    } else if (cur) {
      events.push(cur);
      cur = null;
    }
  });
  if (cur) events.push(cur);
  return events;
}

export function OutageChart({ rows, meta, fullRows }) {
  if (!rows.length || rows[0].unserved_kw == null) return null;
  const dt = meta?.timestep_hours ?? 0.25;
  // Detect outages on the full-resolution rows so short blackouts are not
  // lost to chart downsampling.
  const events = outageEvents(fullRows ?? rows, dt);
  if (!events.length && !meta?.islanded) return null;

  const note = events.length
    ? `${events.length} outage${events.length > 1 ? "s" : ""} · first begins at ${hourLabel(events[0].start)} · shaded = lights out`
    : "islanded · all demand served — no outages this episode";

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Power outage timeline</span>
        <span className="note">{note}</span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
          <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
          <YAxis tick={axis} unit=" kW" width={62} />
          <Tooltip {...tooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
          {events.map((ev, i) => (
            <ReferenceArea
              key={i}
              x1={ev.start}
              x2={ev.end}
              fill={COLORS.unserved}
              fillOpacity={0.12}
              strokeOpacity={0}
            />
          ))}
          <Area isAnimationActive={false}
            dataKey="unserved_kw"
            name="Unserved (blackout)"
            fill={COLORS.unserved}
            stroke={COLORS.unserved}
            fillOpacity={0.55}
            type="stepAfter"
          />
          <Line isAnimationActive={false}
            dataKey="served_kw"
            name="Served"
            stroke={COLORS.ok}
            strokeWidth={1.6}
            dot={false}
            type="monotone"
          />
          <Line isAnimationActive={false}
            dataKey="load_kw"
            name="Demand"
            stroke={THEME.text}
            strokeWidth={1.8}
            dot={false}
            type="monotone"
          />
        </ComposedChart>
      </ResponsiveContainer>
      {events.length > 0 && (
        <div className="outage-events">
          {events.map((ev, i) => (
            <span key={i} className="outage-chip">
              <b>#{i + 1}</b> begins {hourLabel(ev.start)} · ends {hourLabel(ev.end + dt)} ·{" "}
              {fmt(ev.end - ev.start + dt, 2)} h · peak {fmt(ev.peak_kw, 0)} kW ·{" "}
              {fmt(ev.kwh, 0)} kWh lost
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── per-bus state explorer ────────────────────────────────────────────────
const nonzero = (data, key) => data.some((d) => Math.abs(d[key] ?? 0) > 1e-6);

function dieselOnSegments(data) {
  const segments = [];
  let start = null;
  data.forEach((d, i) => {
    if (d.diesel_on && start == null) start = d.hour;
    if ((!d.diesel_on || i === data.length - 1) && start != null) {
      segments.push({ x1: start, x2: d.diesel_on ? d.hour : data[Math.max(0, i - 1)].hour });
      start = null;
    }
  });
  return segments;
}

function BusChart({ bus, rows }) {
  const data = rows.map((r) => {
    const b = r.per_bus?.[bus.id] ?? r.per_bus?.[String(bus.id)] ?? {};
    return {
      hour: r.hour,
      demand_kw: b.demand_kw ?? 0,
      pv_kw: b.pv_kw ?? 0,
      diesel_kw: b.diesel_kw ?? 0,
      diesel_excess_kw: b.diesel_excess_kw ?? 0,
      diesel_on: Boolean(b.diesel_on),
      grid_kw: Math.max(0, b.grid_kw ?? 0),
      battery_discharge_kw: Math.max(0, -(b.battery_kw ?? 0)),
      battery_charge_kw: Math.max(0, b.battery_kw ?? 0),
      unserved_kw: Math.max(0, b.unserved_kw ?? 0),
      v_pu: b.v_pu,
    };
  });

  const hasDiesel = data.some((d) => d.diesel_on) || nonzero(data, "diesel_kw");
  const series = {
    pv: nonzero(data, "pv_kw"),
    grid: nonzero(data, "grid_kw"),
    dischg: nonzero(data, "battery_discharge_kw"),
    chg: nonzero(data, "battery_charge_kw"),
    demand: nonzero(data, "demand_kw"),
    diesel: hasDiesel,
    dieselExcess: nonzero(data, "diesel_excess_kw"),
    unserved: nonzero(data, "unserved_kw"),
  };
  const anyPower = Object.values(series).some(Boolean);
  const role = String(bus.role ?? "bus").toUpperCase();
  const title = `Bus ${bus.id} — ${bus.name}`;

  if (!anyPower) {
    return (
      <Panel title={title} note={`${role} · no assets — bus voltage`} height={220}>
        <ComposedChart data={data} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
          <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
          <YAxis tick={axis} unit=" pu" domain={[0.9, 1.1]} width={58} />
          <Tooltip {...tooltipStyle} formatter={(v, name) => [fmt(v, 4), name]} />
          <Line isAnimationActive={false} dataKey="v_pu" name="Voltage" stroke={COLORS.volt} strokeWidth={1.6} dot={false} type="monotone" />
        </ComposedChart>
      </Panel>
    );
  }

  return (
    <Panel
      title={title}
      note={`${role} · sources vs demand${series.diesel ? " · shaded = diesel ON" : ""}${series.dieselExcess ? " · right axis = diesel excess" : ""}`}
      height={220}
    >
      <ComposedChart data={data} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis tick={axis} unit=" kW" width={62} />
        {series.dieselExcess && (
          <YAxis
            yAxisId="dieselExcess"
            orientation="right"
            tick={axis}
            unit=" kW"
            width={62}
          />
        )}
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        {series.diesel &&
          dieselOnSegments(data).map((seg, i) => (
            <ReferenceArea
              key={i}
              x1={seg.x1}
              x2={seg.x2}
              fill={COLORS.diesel}
              fillOpacity={0.1}
              strokeOpacity={0}
            />
          ))}
        {series.pv && (
          <Area isAnimationActive={false} stackId="src" dataKey="pv_kw" name="PV" fill={COLORS.solar} stroke={COLORS.solar} fillOpacity={0.5} type="monotone" />
        )}
        {series.dischg && (
          <Area isAnimationActive={false} stackId="src" dataKey="battery_discharge_kw" name="Battery discharge" fill={COLORS.battery} stroke={COLORS.battery} fillOpacity={0.5} type="monotone" />
        )}
        {series.diesel && (
          <Area isAnimationActive={false} stackId="src" dataKey="diesel_kw" name="Diesel" fill={COLORS.diesel} stroke={COLORS.diesel} fillOpacity={0.5} type="monotone" />
        )}
        {series.dieselExcess && (
          <Line isAnimationActive={false}
            yAxisId="dieselExcess"
            dataKey="diesel_excess_kw"
            name="Diesel excess (kW)"
            stroke={COLORS.unserved}
            strokeWidth={2.2}
            strokeDasharray="6 3"
            dot={false}
            type="monotone"
          />
        )}
        {series.grid && (
          <Area isAnimationActive={false} stackId="src" dataKey="grid_kw" name="Grid import" fill={COLORS.grid} stroke={COLORS.grid} fillOpacity={0.4} type="monotone" />
        )}
        {series.unserved && (
          <Area isAnimationActive={false} stackId="src" dataKey="unserved_kw" name="Unserved (blackout)" fill={COLORS.unserved} stroke={COLORS.unserved} fillOpacity={0.55} type="monotone" />
        )}
        {series.chg && (
          <Line isAnimationActive={false} dataKey="battery_charge_kw" name="Battery charge" stroke={COLORS.battery} strokeWidth={1.5} strokeDasharray="5 3" dot={false} type="stepAfter" />
        )}
        {series.demand && (
          <Line isAnimationActive={false} dataKey="demand_kw" name="Demand" stroke={THEME.text} strokeWidth={1.8} dot={false} type="monotone" />
        )}
      </ComposedChart>
    </Panel>
  );
}

export function BusCharts({ rows, meta }) {
  const buses = meta?.topology?.buses ?? [];
  if (!buses.length || !rows.length || !rows[0].per_bus) return null;
  return (
    <section className="bus-charts">
      <div className="panel-head">
        <span className="eyebrow">Per-bus state</span>
        <span className="note">sources, demand, and diesel on/off at every bus over the episode</span>
      </div>
      <div className="chart-grid">
        {buses.map((bus) => (
          <BusChart key={bus.id} bus={bus} rows={rows} />
        ))}
      </div>
    </section>
  );
}

const PENALTIES = [
  ["penalty_carbon", "carbon", COLORS.diesel],
  ["penalty_autonomy", "autonomy", COLORS.grid],
  ["penalty_health", "battery health", COLORS.battery],
  ["penalty_waste", "solar waste", COLORS.solar],
  ["penalty_excess", "dumped excess", COLORS.diesel],
  ["penalty_unserved", "unserved", COLORS.unserved],
  ["penalty_constraint", "constraints", COLORS.volt],
];

export function RewardChart({ rows }) {
  return (
    <Panel title="Reward decomposition" note="stacked penalty magnitudes per tick (lower is better)">
      <AreaChart data={rows} {...SYNC} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={THEME.lineSoft} vertical={false} />
        <XAxis dataKey="hour" tick={axis} tickFormatter={hourLabel} minTickGap={40} />
        <YAxis tick={axis} width={64} />
        <Tooltip {...tooltipStyle} />
        <Legend wrapperStyle={{ fontSize: 11.5, fontFamily: "IBM Plex Mono" }} />
        {PENALTIES.map(([key, name, color]) => (
          <Area isAnimationActive={false}
            key={key}
            stackId="p"
            dataKey={key}
            name={name}
            fill={color}
            stroke={color}
            fillOpacity={0.5}
            type="monotone"
          />
        ))}
      </AreaChart>
    </Panel>
  );
}
