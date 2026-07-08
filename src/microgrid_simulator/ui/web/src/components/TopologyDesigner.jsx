import React, { useEffect, useMemo, useRef, useState } from "react";
import { COLORS } from "../api.js";
import { BusAssets, AssetDirectory } from "./AssetInspector.jsx";

const CANVAS_W = 960;
const CANVAS_H = 340;
const MIN_X = 34;
const MAX_X = CANVAS_W - 34;
const MIN_Y = 42;
const MAX_Y = CANVAS_H - 42;
const ZOOM_MIN = 1;
const ZOOM_MAX = 3;
const ZOOM_STEP = 1.25;

const ROLES = [
  { value: "grid", label: "Ext. grid", color: COLORS.grid, kv: 20.0, source: true },
  { value: "main", label: "Main", color: COLORS.ok, kv: 0.4 },
  { value: "pv", label: "PV", color: COLORS.solar, kv: 0.4, source: true },
  { value: "diesel", label: "Diesel", color: COLORS.diesel, kv: 0.4, source: true },
  { value: "battery", label: "Battery", color: COLORS.battery, kv: 0.4 },
  { value: "ev", label: "EV", color: COLORS.volt, kv: 0.4 },
  { value: "load", label: "Load", color: COLORS.load, kv: 0.4 },
  { value: "bus", label: "Bus", color: "var(--muted)", kv: 0.4 },
];

const LINE_KINDS = [
  { value: "auto", label: "auto" },
  { value: "line", label: "line" },
  { value: "transformer", label: "xfmr" },
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function roleInfo(role) {
  return ROLES.find((r) => r.value === String(role ?? "").toLowerCase()) ?? ROLES[ROLES.length - 1];
}

function nextBusId(buses) {
  if (!buses.length) return 0;
  return Math.max(...buses.map((b) => Number(b.id))) + 1;
}

function makeBus(role, buses, point) {
  const info = roleInfo(role);
  const id = nextBusId(buses);
  return {
    id,
    name: `${info.label} bus ${id}`,
    vn_kv: info.kv,
    role: info.value,
    x: Math.round(point.x),
    y: Math.round(point.y),
  };
}

function makeLine(fromId, toId, busById) {
  const a = busById.get(Number(fromId));
  const b = busById.get(Number(toId));
  return {
    name: `${a?.name ?? `Bus ${fromId}`} to ${b?.name ?? `Bus ${toId}`}`,
    from_bus: Number(fromId),
    to_bus: Number(toId),
    kind: "auto",
    length_km: 0.2,
    max_i_ka: 1.0,
  };
}

function shortName(value, max = 18) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function isInputTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName);
}

function normalizedViewport(zoom, cx = CANVAS_W / 2, cy = CANVAS_H / 2) {
  const z = clamp(zoom, ZOOM_MIN, ZOOM_MAX);
  const halfW = CANVAS_W / (2 * z);
  const halfH = CANVAS_H / (2 * z);
  return {
    zoom: z,
    cx: clamp(cx, halfW, CANVAS_W - halfW),
    cy: clamp(cy, halfH, CANVAS_H - halfH),
  };
}

function toViewBox(viewport) {
  const width = CANVAS_W / viewport.zoom;
  const height = CANVAS_H / viewport.zoom;
  return {
    x: viewport.cx - width / 2,
    y: viewport.cy - height / 2,
    width,
    height,
  };
}

export default function TopologyDesigner({
  buses = [],
  lines = [],
  onChange,
  settings = null,
  onSettingsPatch = null,
  className = "",
}) {
  const svgRef = useRef(null);
  const [tool, setTool] = useState("select");
  const [selected, setSelected] = useState(null);
  const [linkStart, setLinkStart] = useState(null);
  const [dragId, setDragId] = useState(null);
  const [viewport, setViewport] = useState(() => normalizedViewport(1));
  const [panning, setPanning] = useState(false);
  const panRef = useRef(null);

  const busById = useMemo(
    () => new Map(buses.map((bus) => [Number(bus.id), bus])),
    [buses]
  );
  const selectedBus =
    selected?.type === "bus" ? busById.get(Number(selected.id)) ?? null : null;
  const selectedLine = selected?.type === "line" ? lines[selected.index] ?? null : null;
  const placingRole = tool.startsWith("place:") ? tool.replace("place:", "") : null;
  const viewBox = useMemo(() => toViewBox(viewport), [viewport]);

  useEffect(() => {
    if (selected?.type === "bus" && !busById.has(Number(selected.id))) {
      setSelected(null);
    }
    if (selected?.type === "line" && !lines[selected.index]) {
      setSelected(null);
    }
    if (linkStart != null && !busById.has(Number(linkStart))) {
      setLinkStart(null);
    }
  }, [busById, lines, linkStart, selected]);

  const commit = (nextBuses = buses, nextLines = lines) => {
    onChange({ buses: nextBuses, lines: nextLines });
  };

  const svgPointFromEvent = (event) => {
    const svg = svgRef.current;
    if (!svg) return { x: CANVAS_W / 2, y: CANVAS_H / 2 };
    const pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: CANVAS_W / 2, y: CANVAS_H / 2 };
    return pt.matrixTransform(ctm.inverse());
  };

  const pointFromEvent = (event) => {
    const loc = svgPointFromEvent(event);
    return {
      x: clamp(loc.x, MIN_X, MAX_X),
      y: clamp(loc.y, MIN_Y, MAX_Y),
    };
  };

  const zoomBy = (factor, anchor = null) => {
    setViewport((current) => {
      const currentBox = toViewBox(current);
      const targetZoom = clamp(current.zoom * factor, ZOOM_MIN, ZOOM_MAX);
      const targetWidth = CANVAS_W / targetZoom;
      const targetHeight = CANVAS_H / targetZoom;
      const focus = anchor ?? { x: current.cx, y: current.cy };
      const ratioX = (focus.x - currentBox.x) / currentBox.width;
      const ratioY = (focus.y - currentBox.y) / currentBox.height;
      const nextCx = focus.x + (0.5 - ratioX) * targetWidth;
      const nextCy = focus.y + (0.5 - ratioY) * targetHeight;
      return normalizedViewport(targetZoom, nextCx, nextCy);
    });
  };

  const fitViewport = () => {
    if (!buses.length) {
      setViewport(normalizedViewport(1));
      return;
    }

    const xs = buses.map((bus) => Number(bus.x ?? CANVAS_W / 2));
    const ys = buses.map((bus) => Number(bus.y ?? CANVAS_H / 2));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = 90;
    const boundsW = Math.max(maxX - minX + padding * 2, 1);
    const boundsH = Math.max(maxY - minY + padding * 2, 1);
    const targetZoom = Math.max(
      ZOOM_MIN,
      Math.min(ZOOM_MAX, CANVAS_W / boundsW, CANVAS_H / boundsH)
    );
    setViewport(normalizedViewport(targetZoom, (minX + maxX) / 2, (minY + maxY) / 2));
  };

  const handleWheel = (event) => {
    // Only zoom on ctrl/cmd + wheel (trackpad pinch sends ctrlKey too) so a
    // plain scroll over the big canvas keeps scrolling the page.
    if (!event.ctrlKey && !event.metaKey) return;
    event.preventDefault();
    const point = svgPointFromEvent(event);
    const direction = event.deltaY > 0 ? 1 / ZOOM_STEP : ZOOM_STEP;
    zoomBy(direction, {
      x: clamp(point.x, 0, CANVAS_W),
      y: clamp(point.y, 0, CANVAS_H),
    });
  };

  const setBusField = (id, key, value) => {
    const next = buses.map((bus) =>
      Number(bus.id) === Number(id) ? { ...bus, [key]: value } : bus
    );
    commit(next, lines);
  };

  const setLineField = (index, key, value) => {
    const next = lines.map((line, i) => (i === index ? { ...line, [key]: value } : line));
    commit(buses, next);
  };

  const deleteSelected = () => {
    if (selected?.type === "bus") {
      const id = Number(selected.id);
      if (buses.length <= 1) return;
      commit(
        buses.filter((bus) => Number(bus.id) !== id),
        lines.filter((line) => Number(line.from_bus) !== id && Number(line.to_bus) !== id)
      );
      setSelected(null);
      setLinkStart((current) => (Number(current) === id ? null : current));
      return;
    }

    if (selected?.type === "line") {
      commit(
        buses,
        lines.filter((_, index) => index !== selected.index)
      );
      setSelected(null);
    }
  };

  const handleCanvasPointerDown = (event) => {
    const point = pointFromEvent(event);
    if (placingRole) {
      const bus = makeBus(placingRole, buses, point);
      commit([...buses, bus], lines);
      setSelected({ type: "bus", id: bus.id });
      setTool("select");
      return;
    }
    if (tool === "link") {
      setLinkStart(null);
      setSelected(null);
      return;
    }

    if (tool === "select") {
      const svg = svgRef.current;
      const rect = svg?.getBoundingClientRect();
      const box = toViewBox(viewport);
      panRef.current = {
        clientX: event.clientX,
        clientY: event.clientY,
        moved: false,
        viewport,
        unitsPerPxX: rect ? box.width / rect.width : 1,
        unitsPerPxY: rect ? box.height / rect.height : 1,
      };
      setPanning(true);
      event.currentTarget.setPointerCapture?.(event.pointerId);
    }
  };

  const handleBusPointerDown = (event, bus) => {
    event.stopPropagation();
    const busId = Number(bus.id);

    if (tool === "link") {
      if (linkStart == null || Number(linkStart) === busId) {
        setLinkStart(busId);
        setSelected({ type: "bus", id: busId });
        return;
      }

      const from = Number(linkStart);
      const duplicate = lines.findIndex(
        (line) =>
          (Number(line.from_bus) === from && Number(line.to_bus) === busId) ||
          (Number(line.from_bus) === busId && Number(line.to_bus) === from)
      );

      if (duplicate >= 0) {
        setSelected({ type: "line", index: duplicate });
      } else {
        commit(buses, [...lines, makeLine(from, busId, busById)]);
        setSelected({ type: "line", index: lines.length });
      }
      setLinkStart(null);
      return;
    }

    if (placingRole) return;

    setSelected({ type: "bus", id: busId });
    setDragId(busId);
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (panRef.current && dragId == null && tool === "select") {
      const pan = panRef.current;
      const dx = (event.clientX - pan.clientX) * pan.unitsPerPxX;
      const dy = (event.clientY - pan.clientY) * pan.unitsPerPxY;
      if (Math.abs(event.clientX - pan.clientX) > 2 || Math.abs(event.clientY - pan.clientY) > 2) {
        pan.moved = true;
      }
      setViewport(normalizedViewport(pan.viewport.zoom, pan.viewport.cx - dx, pan.viewport.cy - dy));
      return;
    }

    if (dragId == null || tool !== "select") return;
    const point = pointFromEvent(event);
    commit(
      buses.map((bus) =>
        Number(bus.id) === Number(dragId)
          ? { ...bus, x: Math.round(point.x), y: Math.round(point.y) }
          : bus
      ),
      lines
    );
  };

  const handlePointerUp = () => {
    if (panRef.current) {
      if (!panRef.current.moved) setSelected(null);
      panRef.current = null;
      setPanning(false);
    }
    setDragId(null);
  };

  const handleLinePointerDown = (event, index) => {
    event.stopPropagation();
    setSelected({ type: "line", index });
    setLinkStart(null);
  };

  const handleKeyDown = (event) => {
    if (isInputTarget(event.target)) return;
    if ((event.key === "Delete" || event.key === "Backspace") && selected) {
      event.preventDefault();
      deleteSelected();
    }
    if (event.key === "Escape") {
      setTool("select");
      setLinkStart(null);
      setDragId(null);
    }
  };

  const toolLabel =
    linkStart != null
      ? `link from BUS ${linkStart}`
      : placingRole
        ? `place ${roleInfo(placingRole).label}`
        : tool;
  const zoomPct = Math.round(viewport.zoom * 100);
  const hasGrid = buses.some((bus) => String(bus.role ?? "").toLowerCase() === "grid");

  return (
    <div className={`topology-designer ${className}`} tabIndex={0} onKeyDown={handleKeyDown}>
      <div className="topology-tools">
        <div className="tool-row">
          <button
            type="button"
            className={`tool-btn ${tool === "select" ? "active" : ""}`}
            onClick={() => {
              setTool("select");
              setLinkStart(null);
            }}
            title="Select and move devices"
          >
            Move
          </button>
          <button
            type="button"
            className={`tool-btn ${tool === "link" ? "active" : ""}`}
            onClick={() => {
              setTool("link");
              setLinkStart(null);
            }}
            title="Connect two buses"
          >
            Link
          </button>
          <button
            type="button"
            className="tool-btn danger"
            onClick={deleteSelected}
            disabled={!selected || (selected?.type === "bus" && buses.length <= 1)}
            title="Delete selected bus or link"
          >
            Delete
          </button>
          <div className="zoom-controls" aria-label="Canvas zoom controls">
            <button
              type="button"
              className="tool-btn zoom-btn"
              onClick={() => zoomBy(1 / ZOOM_STEP)}
              disabled={viewport.zoom <= ZOOM_MIN}
              title="Zoom out"
            >
              -
            </button>
            <button
              type="button"
              className="tool-btn zoom-level"
              onClick={fitViewport}
              title="Fit all nodes — Ctrl+scroll (pinch) on the canvas also zooms"
            >
              Fit {zoomPct}%
            </button>
            <button
              type="button"
              className="tool-btn zoom-btn"
              onClick={() => zoomBy(ZOOM_STEP)}
              disabled={viewport.zoom >= ZOOM_MAX}
              title="Zoom in"
            >
              +
            </button>
          </div>
        </div>

        <div className="device-palette">
          {ROLES.map((role) => (
            <button
              key={role.value}
              type="button"
              className={`device-btn ${placingRole === role.value ? "active" : ""}`}
              style={{ "--role-color": role.color }}
              onClick={() => {
                setTool(`place:${role.value}`);
                setLinkStart(null);
              }}
              title={
                role.source
                  ? `Place ${role.label} power-source bus (auto-wires the ${role.label} asset)`
                  : `Place ${role.label} bus`
              }
            >
              <span className="device-swatch" />
              {role.label}
            </button>
          ))}
        </div>
      </div>

      <svg
        ref={svgRef}
        className={`topology-canvas ${tool === "link" ? "is-linking" : ""} ${
          placingRole ? "is-placing" : ""
        } ${panning ? "is-panning" : ""} ${tool === "select" && viewport.zoom > 1 ? "can-pan" : ""}`}
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        role="img"
        aria-label="Interactive microgrid topology editor"
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onWheel={handleWheel}
      >
        <defs>
          <pattern id="topology-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" className="grid-path" />
          </pattern>
        </defs>
        <rect className="canvas-bg" x="0" y="0" width={CANVAS_W} height={CANVAS_H} />
        <rect className="canvas-grid" x="0" y="0" width={CANVAS_W} height={CANVAS_H} />

        {lines.map((line, index) => {
          const a = busById.get(Number(line.from_bus));
          const b = busById.get(Number(line.to_bus));
          if (!a || !b) return null;
          const selectedLineIndex = selected?.type === "line" ? selected.index : null;
          const isSelected = selectedLineIndex === index;
          const isTransformer = String(line.kind ?? "").toLowerCase() === "transformer";
          return (
            <g
              key={`${line.name}-${line.from_bus}-${line.to_bus}-${index}`}
              className={`topology-link ${isSelected ? "selected" : ""} ${
                isTransformer ? "transformer" : ""
              }`}
              onPointerDown={(event) => handleLinePointerDown(event, index)}
            >
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
              <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 8} textAnchor="middle">
                {isTransformer ? "XFMR" : shortName(line.name)}
              </text>
            </g>
          );
        })}

        {buses.map((bus) => {
          const info = roleInfo(bus.role);
          const isSelected =
            selected?.type === "bus" && Number(selected.id) === Number(bus.id);
          const isLinkStart = Number(linkStart) === Number(bus.id);
          return (
            <g
              key={bus.id}
              className={`topology-node ${isSelected ? "selected" : ""} ${
                isLinkStart ? "link-start" : ""
              }`}
              style={{ "--node-color": info.color }}
              onPointerDown={(event) => handleBusPointerDown(event, bus)}
            >
              <circle className="node-target" cx={bus.x} cy={bus.y} r="24" />
              <circle className="node-shell" cx={bus.x} cy={bus.y} r="15" />
              <circle className="node-core" cx={bus.x} cy={bus.y} r="6" />
              <text className="node-id" x={bus.x} y={bus.y - 23} textAnchor="middle">
                BUS {bus.id}
              </text>
              <text className="node-name" x={bus.x} y={bus.y + 31} textAnchor="middle">
                {shortName(bus.name, 16)}
              </text>
              <text className="node-role" x={bus.x} y={bus.y + 44} textAnchor="middle">
                {String(bus.role ?? "bus").toUpperCase()}
              </text>
              <title>
                {bus.name} - Bus {bus.id}, {bus.vn_kv} kV
              </title>
            </g>
          );
        })}
      </svg>

      <div className="topology-status">
        <span>{buses.length} buses</span>
        <span>{lines.length} links</span>
        <span>{toolLabel}</span>
        <span
          className={hasGrid ? "mode-grid" : "mode-island"}
          title={
            hasGrid
              ? "A Grid node ties the microgrid to the utility, which backfills any shortfall."
              : "No Grid node — the microgrid is islanded. PV, diesel and battery must meet demand or the shortfall becomes a blackout."
          }
        >
          {hasGrid ? "grid-connected" : "⚡ islanded — blackouts possible"}
        </span>
      </div>

      <div className="topology-inspector">
        {selectedBus ? (
          <>
            <div className="mini-head">
              <span>Selected bus {selectedBus.id}</span>
              <button
                type="button"
                className="icon-btn"
                onClick={deleteSelected}
                disabled={buses.length <= 1}
                aria-label="Delete bus"
                title="Delete bus"
              >
                ×
              </button>
            </div>
            <label className="field">
              <span>Name</span>
              <input
                type="text"
                value={selectedBus.name}
                onChange={(event) => setBusField(selectedBus.id, "name", event.target.value)}
              />
            </label>
            <div className="field-row">
              <label className="field">
                <span>Role</span>
                <select
                  value={selectedBus.role ?? "bus"}
                  onChange={(event) => setBusField(selectedBus.id, "role", event.target.value)}
                >
                  {ROLES.map((role) => (
                    <option key={role.value} value={role.value}>
                      {role.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>kV</span>
                <input
                  type="number"
                  min={0.1}
                  step={0.1}
                  value={selectedBus.vn_kv}
                  onChange={(event) =>
                    setBusField(selectedBus.id, "vn_kv", Number(event.target.value) || 0.1)
                  }
                />
              </label>
            </div>
            {settings && onSettingsPatch && (
              <BusAssets bus={selectedBus} settings={settings} onPatch={onSettingsPatch} />
            )}
          </>
        ) : selectedLine ? (
          <>
            <div className="mini-head">
              <span>Selected link</span>
              <button
                type="button"
                className="icon-btn"
                onClick={deleteSelected}
                aria-label="Delete link"
                title="Delete link"
              >
                ×
              </button>
            </div>
            <label className="field">
              <span>Name</span>
              <input
                type="text"
                value={selectedLine.name}
                onChange={(event) => setLineField(selected.index, "name", event.target.value)}
              />
            </label>
            <div className="topology-endpoints">
              <span>BUS {selectedLine.from_bus}</span>
              <span>BUS {selectedLine.to_bus}</span>
            </div>
            <div className="field-row">
              <label className="field">
                <span>Kind</span>
                <select
                  value={selectedLine.kind ?? "auto"}
                  onChange={(event) => setLineField(selected.index, "kind", event.target.value)}
                >
                  {LINE_KINDS.map((kind) => (
                    <option key={kind.value} value={kind.value}>
                      {kind.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>km</span>
                <input
                  type="number"
                  min={0.01}
                  step={0.01}
                  value={selectedLine.length_km ?? 0.2}
                  onChange={(event) =>
                    setLineField(selected.index, "length_km", Number(event.target.value) || 0.01)
                  }
                />
              </label>
            </div>
          </>
        ) : settings ? (
          <AssetDirectory
            settings={settings}
            onSelectBus={(id) => setSelected({ type: "bus", id: Number(id) })}
          />
        ) : (
          <div className="topology-empty-selection">No selection</div>
        )}
      </div>
    </div>
  );
}
