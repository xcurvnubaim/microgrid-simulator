import React, { useEffect, useMemo, useRef, useState } from "react";
import { COLORS, fmt, hourLabel } from "../api.js";

const FALLBACK_TOPOLOGY = {
  buses: [
    { id: 0, name: "Grid interconnect", vn_kv: 20.0, role: "grid", x: 70, y: 150 },
    { id: 1, name: "Main distribution", vn_kv: 0.4, role: "main", x: 290, y: 150 },
    { id: 2, name: "PV yard", vn_kv: 0.4, role: "pv", x: 520, y: 60 },
    { id: 3, name: "EV charging hub", vn_kv: 0.4, role: "ev", x: 520, y: 240 },
    { id: 4, name: "Academic loads", vn_kv: 0.4, role: "load", x: 760, y: 150 },
    { id: 5, name: "Battery storage", vn_kv: 0.4, role: "battery", x: 760, y: 240 },
  ],
  lines: [
    { name: "20/0.4 kV service transformer", from_bus: 0, to_bus: 1, kind: "transformer" },
    { name: "PV feeder", from_bus: 1, to_bus: 2, kind: "line" },
    { name: "EV feeder", from_bus: 1, to_bus: 3, kind: "line" },
    { name: "Academic feeder", from_bus: 1, to_bus: 4, kind: "line" },
    { name: "Battery feeder", from_bus: 1, to_bus: 5, kind: "line" },
  ],
  battery_bus: 5,
  ev_bus: 3,
  diesel_bus: 1,
  pv_buses: [2],
  load_buses: [4],
};

function withLayout(buses) {
  const missing = buses.some((b) => b.x == null || b.y == null);
  if (!missing) return buses;
  const cx = 480;
  const cy = 160;
  const rx = 350;
  const ry = 110;
  return buses.map((b, i) => {
    if (b.x != null && b.y != null) return b;
    const a = (2 * Math.PI * i) / Math.max(buses.length, 1) - Math.PI;
    return { ...b, x: Math.round(cx + rx * Math.cos(a)), y: Math.round(cy + ry * Math.sin(a)) };
  });
}

function FlowDots({ x1, y1, x2, y2, active, reverse, color, title }) {
  if (!active) return null;
  const [ax, ay, bx, by] = reverse ? [x2, y2, x1, y1] : [x1, y1, x2, y2];
  return (
    <g>
      {[0, 1, 2].map((i) => (
        <circle key={i} r="3" fill={color} opacity="0.9">
          <animateMotion dur="2.4s" begin={`${i * 0.8}s`} repeatCount="indefinite" path={`M ${ax} ${ay} L ${bx} ${by}`} />
        </circle>
      ))}
      <title>{title}</title>
    </g>
  );
}

function Chip({ x, y, color, title, value, unit = "kW", on = true }) {
  return (
    <g opacity={on ? 1 : 0.38}>
      <rect x={x} y={y} width="128" height="34" rx="7" fill="var(--inset)" stroke={color} strokeOpacity="0.55" />
      <circle cx={x + 13} cy={y + 17} r="4" fill={color} />
      <text x={x + 24} y={y + 14} fontSize="9" letterSpacing="0.08em" fill="var(--faint)">
        {title.toUpperCase()}
      </text>
      <text x={x + 24} y={y + 27} fontSize="12" fill="var(--text)" fontWeight="500">
        {value} <tspan fontSize="9" fill="var(--muted)">{unit}</tspan>
      </text>
    </g>
  );
}

export default function SLD({ rows, meta }) {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);
  const n = rows?.length ?? 0;

  useEffect(() => setIdx(n ? n - 1 : 0), [rows]);
  useEffect(() => {
    if (!playing || !n) return undefined;
    timer.current = setInterval(() => setIdx((i) => (i + 1 >= n ? 0 : i + 1)), 140);
    return () => clearInterval(timer.current);
  }, [playing, n]);

  const r = rows?.[Math.min(idx, n - 1)] ?? null;
  const topology = meta?.topology ?? FALLBACK_TOPOLOGY;
  const buses = useMemo(() => withLayout(topology.buses ?? FALLBACK_TOPOLOGY.buses), [topology]);
  const busById = useMemo(() => Object.fromEntries(buses.map((b) => [Number(b.id), b])), [buses]);
  const busIndex = useMemo(() => Object.fromEntries(buses.map((b, i) => [Number(b.id), i])), [buses]);
  const lines = topology.lines ?? FALLBACK_TOPOLOGY.lines;

  const flows = useMemo(() => buildFlows(lines, busById, topology, r), [lines, busById, topology, r]);

  return (
      <div className="panel">
        <div className="panel-head">
          <span className="eyebrow">Dynamic topology diagram</span>
        {r && <span className="note">topology comes from the interactive designer — drag to scrub, ▶ to animate</span>}
      </div>

      <div className="sld-wrap sld">
        <svg viewBox="0 0 960 340" role="img" aria-label="Editable campus microgrid topology with live power flows">
          {lines.map((line, i) => {
            const a = busById[Number(line.from_bus)];
            const b = busById[Number(line.to_bus)];
            if (!a || !b) return null;
            return (
              <g key={`${line.name}-${i}`}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--line)" strokeWidth={line.kind === "transformer" ? 3 : 2} strokeDasharray={line.kind === "transformer" ? "0" : undefined} />
                <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 7} textAnchor="middle" fontSize="8.5" fill="var(--faint)">
                  {line.kind === "transformer" ? "XFMR" : line.name}
                </text>
                <title>{line.name}</title>
              </g>
            );
          })}

          {flows.map((f, i) => <FlowDots key={`${f.title}-${i}`} {...f} />)}

          {buses.map((b) => (
            <g key={b.id}>
              <circle cx={b.x} cy={b.y} r="13" fill="var(--panel-2)" stroke="var(--line)" strokeWidth="2" />
              <circle cx={b.x} cy={b.y} r="5" fill={busColor(b, r, busIndex)} />
              <text x={b.x} y={b.y - 24} textAnchor="middle" fontSize="10" fill="var(--text)" fontFamily="var(--mono)">
                BUS {b.id} · {b.vn_kv} kV
              </text>
              <text x={b.x} y={b.y + 28} textAnchor="middle" fontSize="9" fill="var(--faint)">
                {b.name}
              </text>
              <text x={b.x} y={b.y + 41} textAnchor="middle" fontSize="8.5" fill={roleColor(b.role)}>
                {String(b.role ?? "bus").toUpperCase()}
              </text>
            </g>
          ))}

          {r && (
            <>
              <Chip x={8} y={12} color={COLORS.grid} title="Grid import" value={fmt(r.grid_import_kw)} />
              <Chip x={152} y={12} color={COLORS.solar} title="PV used" value={fmt(r.pv_used_kw)} />
              <Chip x={296} y={12} color={COLORS.solar} title="PV available" value={fmt(r.pv_available_kw)} />
              <Chip x={440} y={12} color={COLORS.battery} title={r.battery_kw >= 0 ? "Battery charge" : "Battery discharge"} value={fmt(Math.abs(r.battery_kw))} />
              <Chip x={584} y={12} color={COLORS.battery} title="Battery SoC" value={fmt(r.soc_pct)} unit="%" />
              <Chip x={728} y={12} color={COLORS.load} title="Campus load" value={fmt(r.load_kw)} />
              <Chip x={8} y={292} color={COLORS.diesel} title={r.diesel_on ? "Diesel · ON" : "Diesel · off"} value={fmt(r.diesel_kw)} on={r.diesel_on} />
              <Chip x={152} y={292} color={COLORS.volt} title="EV SoC" value={(r.ev_soc_pct ?? []).map((v) => fmt(v, 0)).join(" · ") || "—"} unit="%" />
              <Chip x={296} y={292} color={r.min_voltage_pu < 0.95 ? COLORS.grid : COLORS.ok} title="Min bus V" value={fmt(r.min_voltage_pu, 3)} unit="pu" />
            </>
          )}
        </svg>

        {n > 0 && (
          <div className="scrubber">
            <button className="play-btn" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause replay" : "Play replay"}>
              {playing ? "❚❚" : "▶"}
            </button>
            <input
              type="range"
              min="0"
              max={n - 1}
              value={Math.min(idx, n - 1)}
              onChange={(e) => {
                setPlaying(false);
                setIdx(Number(e.target.value));
              }}
              aria-label="Scrub through timesteps"
            />
            <span className="scrub-time">{r ? `t = ${hourLabel(r.hour)} · step ${r.step}/${n}` : ""}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function buildFlows(lines, busById, topology, r) {
  if (!r) return [];
  const specs = [];
  const batteryBus = Number(topology.battery_bus);
  const evBus = Number(topology.ev_bus);
  const dieselBus = Number(topology.diesel_bus);
  const pvBuses = new Set((topology.pv_buses ?? []).map(Number));
  const loadBuses = new Set((topology.load_buses ?? []).map(Number));

  for (const line of lines) {
    const a = busById[Number(line.from_bus)];
    const b = busById[Number(line.to_bus)];
    if (!a || !b) continue;
    const base = { x1: a.x, y1: a.y, x2: b.x, y2: b.y };
    const addFromTo = (fromId, toId, active, color, title) => {
      if (!active) return;
      const forward = Number(line.from_bus) === Number(fromId) && Number(line.to_bus) === Number(toId);
      const reverse = Number(line.from_bus) === Number(toId) && Number(line.to_bus) === Number(fromId);
      if (forward || reverse) specs.push({ ...base, active, reverse, color, title });
    };
    const otherOf = (id) => Number(line.from_bus) === Number(id) ? Number(line.to_bus) : Number(line.from_bus);
    const touches = (id) => Number(line.from_bus) === Number(id) || Number(line.to_bus) === Number(id);

    const gridEnd = [a, b].find((x) => String(x.role).toLowerCase() === "grid");
    if (gridEnd) {
      const other = otherOf(gridEnd.id);
      addFromTo(gridEnd.id, other, r.grid_import_kw > 1, COLORS.grid, "Grid import");
      addFromTo(other, gridEnd.id, r.grid_import_kw < -1, COLORS.ok, "Grid export");
    }

    for (const pvBus of pvBuses) {
      if (touches(pvBus)) addFromTo(pvBus, otherOf(pvBus), r.pv_used_kw > 1, COLORS.solar, "PV export");
    }
    if (touches(batteryBus)) {
      const other = otherOf(batteryBus);
      addFromTo(other, batteryBus, r.battery_kw > 1, COLORS.battery, "Battery charging");
      addFromTo(batteryBus, other, r.battery_kw < -1, COLORS.battery, "Battery discharging");
    }
    if (touches(dieselBus)) addFromTo(dieselBus, otherOf(dieselBus), r.diesel_kw > 1, COLORS.diesel, "Diesel generation");
    for (const loadBus of loadBuses) {
      if (touches(loadBus)) addFromTo(otherOf(loadBus), loadBus, r.load_kw > 1, COLORS.load, "Load supply");
    }
    if (touches(evBus)) addFromTo(otherOf(evBus), evBus, (r.ev_soc_pct ?? []).length > 0, COLORS.volt, "EV charging");
  }
  return specs.slice(0, 18);
}

function roleColor(role) {
  const key = String(role ?? "").toLowerCase();
  if (key === "grid") return COLORS.grid;
  if (key === "pv") return COLORS.solar;
  if (key === "battery") return COLORS.battery;
  if (key === "ev") return COLORS.volt;
  if (key === "load") return COLORS.load;
  if (key === "main") return COLORS.ok;
  return "var(--muted)";
}

function busColor(bus, r, busIndex) {
  if (!r || !r.v_bus) return roleColor(bus.role);
  const v = r.v_bus[busIndex[Number(bus.id)]];
  if (v == null) return roleColor(bus.role);
  if (v < 0.95 || v > 1.05) return COLORS.grid;
  return roleColor(bus.role);
}
