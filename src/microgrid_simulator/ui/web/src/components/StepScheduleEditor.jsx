import React, {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

/* Generic editor for a 24-hour step schedule. The graph, exact-value table,
   history, and expanded dialog are shared by the diesel and battery wrappers;
   the wrappers only supply asset-specific value labels and controls. */

const SNAP_HOURS = 0.25;
const MAX_HISTORY = 24;

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

const cloneMeta = (meta) =>
  meta && typeof meta === "object" ? { ...meta } : meta;

const cloneSteps = (steps) =>
  steps.map((step) => ({ ...step, meta: cloneMeta(step.meta) }));

const sameSteps = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const snapHour = (hour) => Math.round(Number(hour) / SNAP_HOURS) * SNAP_HOURS;

export const fmtTime = (hour) => {
  const snapped = snapHour(clamp(Number(hour) || 0, 0, 24));
  if (snapped >= 24) return "24:00";
  const totalMinutes = Math.round(snapped * 60);
  const hh = Math.floor(totalMinutes / 60);
  const mm = totalMinutes % 60;
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
};

const parseTime = (text) => {
  const match = /^(\d{1,2}):(\d{2})$/.exec(String(text ?? "").trim());
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (hours < 0 || hours > 24 || minutes < 0 || minutes > 59) return null;
  if (hours === 24 && minutes !== 0) return null;
  return snapHour(hours + minutes / 60);
};

/* segments -> steps. `segValue(segment)` maps a config segment to a number;
   hours no segment covers are 0 (off / idle). Authored boundaries between
   equal-value blocks are deliberately retained. Optional metadata lets an
   untouched named level (for example diesel "min") survive time-only edits. */
export function segmentsToSteps(segments, segValue, segMeta = null) {
  const segmentAt = (hour) => {
    for (const segment of segments) {
      const inSegment =
        segment.start_hour <= segment.end_hour
          ? hour >= segment.start_hour && hour < segment.end_hour
          : hour >= segment.start_hour || hour < segment.end_hour;
      if (inSegment) return segment;
    }
    return null;
  };

  const bounds = new Set([0]);
  for (const segment of segments) {
    for (const boundary of [segment.start_hour, segment.end_hour]) {
      const hour = boundary % 24;
      if (hour > 0 && hour < 24) bounds.add(hour);
    }
  }

  return [...bounds]
    .sort((a, b) => a - b)
    .map((start) => {
      const segment = segmentAt(start);
      const step = { start, value: segment ? segValue(segment) : 0 };
      if (segment && segMeta) step.meta = segMeta(segment);
      return step;
    });
}

/* steps -> segments. `segFromValue(value, step)` maps the numeric graph value
   and optional preserved metadata back to the config segment fields. */
export function stepsToSegments(steps, segFromValue) {
  return steps.map((step, index) => ({
    start_hour: step.start,
    end_hour: index + 1 < steps.length ? steps[index + 1].start : 24,
    ...segFromValue(step.value, step),
  }));
}

function useElementSize(ref, expanded) {
  const [size, setSize] = useState(() => ({
    width: expanded ? 760 : 340,
    height: expanded ? 440 : 210,
  }));

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return undefined;

    const measure = () => {
      const rect = element.getBoundingClientRect();
      const next = {
        width: Math.max(220, Math.round(rect.width)),
        height: Math.max(170, Math.round(rect.height)),
      };
      setSize((current) =>
        current.width === next.width && current.height === next.height ? current : next,
      );
    };

    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }

    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [expanded, ref]);

  return size;
}

function ScheduleCanvas({
  steps,
  commit,
  splitBlock,
  mergeBoundary,
  updateBoundary,
  deleteBlock,
  selected,
  onSelect,
  yMin,
  yMax,
  snap,
  nudgeValue,
  label,
  guides,
  color,
  title,
  expanded,
  helpId,
}) {
  const wrapRef = useRef(null);
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const { width: W, height: H } = useElementSize(wrapRef, expanded);
  const [hover, setHover] = useState(null);
  const [boundaryHover, setBoundaryHover] = useState(null);
  const [readout, setReadout] = useState(null);
  const [introVisible, setIntroVisible] = useState(false);
  const [introSeen, setIntroSeen] = useState(false);

  const PAD = expanded
    ? { l: 58, r: 18, t: 24, b: 34 }
    : { l: 43, r: 12, t: 18, b: 27 };
  const plotWidth = Math.max(1, W - PAD.l - PAD.r);
  const plotHeight = Math.max(1, H - PAD.t - PAD.b);
  const span = Math.max(Number.EPSILON, yMax - yMin);
  const x = (hour) => PAD.l + (hour / 24) * plotWidth;
  const y = (value) => {
    const bounded = clamp(value, yMin, yMax);
    return PAD.t + (1 - (bounded - yMin) / span) * plotHeight;
  };

  const pointerHourValue = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * W;
    const py = ((event.clientY - rect.top) / rect.height) * H;
    return {
      hour: clamp(((px - PAD.l) / plotWidth) * 24, 0, 24),
      value: clamp(yMin + (1 - (py - PAD.t) / plotHeight) * span, yMin, yMax),
    };
  };

  const startDrag = (event, mode, index) => {
    event.preventDefault();
    setIntroVisible(false);
    setIntroSeen(true);
    onSelect(mode === "time" ? Math.max(0, index - 1) : index);
    svgRef.current?.focus({ preventScroll: true });
    dragRef.current = {
      mode,
      index,
      before: cloneSteps(steps),
      latest: cloneSteps(steps),
      recorded: false,
    };
    svgRef.current?.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event) => {
    const drag = dragRef.current;
    if (!drag) return;

    const { hour, value } = pointerHourValue(event);
    const next = cloneSteps(drag.latest);
    if (drag.mode === "level") {
      const snappedValue = snap(value);
      next[drag.index] = { start: next[drag.index].start, value: snappedValue };
      const end = drag.index + 1 < next.length ? next[drag.index + 1].start : 24;
      setReadout(
        `${fmtTime(next[drag.index].start)}–${fmtTime(end)} · ${label(snappedValue)}`,
      );
    } else {
      const low = next[drag.index - 1].start + SNAP_HOURS;
      const high =
        (drag.index + 1 < next.length ? next[drag.index + 1].start : 24) - SNAP_HOURS;
      next[drag.index].start = clamp(snapHour(hour), low, high);
      setReadout(`switch at ${fmtTime(next[drag.index].start)}`);
    }

    if (sameSteps(next, drag.latest)) return;
    commit(next, {
      record: !drag.recorded,
      before: drag.before,
    });
    drag.recorded = true;
    drag.latest = cloneSteps(next);
  };

  const endDrag = () => {
    dragRef.current = null;
    setReadout(null);
  };

  const onKeyDown = (event) => {
    if (!steps[selected]) return;

    if (event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      const boundaryIndex = event.key === "ArrowLeft" ? selected : selected + 1;
      if (boundaryIndex > 0 && boundaryIndex < steps.length) {
        event.preventDefault();
        const delta = event.key === "ArrowLeft" ? -SNAP_HOURS : SNAP_HOURS;
        updateBoundary(boundaryIndex, steps[boundaryIndex].start + delta);
      }
      return;
    }

    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const direction = event.key === "ArrowUp" ? 1 : -1;
      const value = nudgeValue(steps[selected].value, direction);
      if (value === steps[selected].value) return;
      const next = cloneSteps(steps);
      next[selected] = { start: next[selected].start, value };
      commit(next);
      return;
    }

    if (!event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      onSelect(
        clamp(selected + (event.key === "ArrowRight" ? 1 : -1), 0, steps.length - 1),
      );
      return;
    }

    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteBlock(selected);
    }
  };

  const outline = steps
    .map((step, index) => {
      const end = index + 1 < steps.length ? steps[index + 1].start : 24;
      return `${index === 0 ? "M" : "L"} ${x(step.start)} ${y(step.value)} L ${x(end)} ${y(step.value)}`;
    })
    .join(" ");
  const baseline = y(clamp(0, yMin, yMax));
  const area = `${outline} L ${x(24)} ${baseline} L ${x(0)} ${baseline} Z`;
  const boundaryTooltip = boundaryHover == null ? null : steps[boundaryHover];
  const tooltipWidth = expanded ? 144 : 122;
  const tooltipX = boundaryTooltip
    ? clamp(x(boundaryTooltip.start) - tooltipWidth / 2, PAD.l, W - PAD.r - tooltipWidth)
    : 0;

  return (
    <div
      ref={wrapRef}
      className={`schedule-canvas-wrap ${expanded ? "is-expanded" : ""}`}
      onPointerEnter={() => {
        if (!introSeen) setIntroVisible(true);
      }}
      onPointerLeave={() => {
        if (!dragRef.current) {
          setHover(null);
          setBoundaryHover(null);
          setIntroVisible(false);
          setIntroSeen(true);
        }
      }}
    >
      <svg
        ref={svgRef}
        className="schedule-canvas"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        tabIndex="0"
        role="img"
        aria-label={`${title}. Interactive 24-hour step graph.`}
        aria-describedby={helpId}
        onFocus={() => {
          if (!introSeen) setIntroVisible(true);
        }}
        onBlur={() => setIntroVisible(false)}
        onKeyDown={onKeyDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onLostPointerCapture={endDrag}
      >
        <rect
          x={PAD.l}
          y={PAD.t}
          width={plotWidth}
          height={plotHeight}
          fill="var(--inset)"
        />

        {guides.map(({ value, text }, index) => (
          <g key={`${text}-${value}-${index}`}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={y(value)}
              y2={y(value)}
              stroke="var(--line)"
              strokeDasharray="3 3"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={PAD.l - 5}
              y={y(value) + 3}
              textAnchor="end"
              fontSize={expanded ? 10 : 9}
              fill="var(--faint)"
              fontFamily="IBM Plex Mono"
            >
              {text}
            </text>
          </g>
        ))}

        {yMin < 0 && (
          <line
            x1={PAD.l}
            x2={W - PAD.r}
            y1={y(0)}
            y2={y(0)}
            stroke="var(--muted)"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {[0, 6, 12, 18, 24].map((hour) => (
          <g key={hour}>
            <line
              x1={x(hour)}
              x2={x(hour)}
              y1={PAD.t}
              y2={H - PAD.b}
              stroke="var(--line-soft)"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={x(hour)}
              y={H - 8}
              textAnchor="middle"
              fontSize={expanded ? 10 : 9}
              fill="var(--faint)"
              fontFamily="IBM Plex Mono"
            >
              {hour}h
            </text>
          </g>
        ))}

        <path d={area} fill={color} opacity="0.16" />
        <path
          d={outline}
          fill="none"
          stroke={color}
          strokeWidth={expanded ? 2.2 : 1.8}
          vectorEffect="non-scaling-stroke"
        />

        {steps.map((step, index) => {
          const end = index + 1 < steps.length ? steps[index + 1].start : 24;
          const middle = (step.start + end) / 2;
          const isSelected = selected === index;
          return (
            <g key={`plateau-${index}`}>
              {isSelected && (
                <rect
                  x={x(step.start) + 1}
                  y={PAD.t + 1}
                  width={Math.max(0, x(end) - x(step.start) - 2)}
                  height={Math.max(0, plotHeight - 2)}
                  fill={color}
                  fillOpacity="0.07"
                  stroke={color}
                  strokeOpacity="0.72"
                  strokeDasharray="4 3"
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              )}
              {hover === index && (
                <line
                  x1={x(step.start)}
                  x2={x(end)}
                  y1={y(step.value)}
                  y2={y(step.value)}
                  stroke={color}
                  strokeWidth="4"
                  opacity="0.55"
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              )}
              <rect
                x={x(step.start)}
                y={PAD.t}
                width={Math.max(1, x(end) - x(step.start))}
                height={plotHeight}
                fill="transparent"
                style={{ cursor: "ns-resize" }}
                onPointerDown={(event) => startDrag(event, "level", index)}
                onDoubleClick={(event) => {
                  event.preventDefault();
                  splitBlock(index, pointerHourValue(event).hour);
                }}
                onPointerEnter={() => setHover(index)}
              >
                <title>
                  {fmtTime(step.start)}–{fmtTime(end)} · {label(step.value)}. Drag vertically;
                  double-click to split.
                </title>
              </rect>
              {(hover === index || isSelected || x(end) - x(step.start) > 58) && (
                <text
                  x={x(middle)}
                  y={
                    y(step.value) +
                    (y(step.value) - PAD.t < plotHeight * 0.18 ? 13 : -6)
                  }
                  textAnchor="middle"
                  fontSize={expanded ? 10 : 9}
                  fill="var(--text)"
                  fontFamily="IBM Plex Mono"
                  pointerEvents="none"
                >
                  {label(step.value)}
                </text>
              )}
            </g>
          );
        })}

        {steps.map((step, index) =>
          index === 0 ? null : (
            <g
              key={`boundary-${index}`}
              style={{ cursor: "ew-resize" }}
              onPointerEnter={() => {
                setBoundaryHover(index);
                setIntroVisible(false);
              }}
              onPointerLeave={() => setBoundaryHover(null)}
              onPointerDown={(event) => startDrag(event, "time", index)}
              onDoubleClick={(event) => {
                event.preventDefault();
                mergeBoundary(index);
              }}
            >
              <line
                x1={x(step.start)}
                x2={x(step.start)}
                y1={PAD.t}
                y2={H - PAD.b}
                stroke="transparent"
                strokeWidth="18"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={x(step.start)}
                x2={x(step.start)}
                y1={PAD.t}
                y2={H - PAD.b}
                stroke={color}
                opacity="0.6"
                vectorEffect="non-scaling-stroke"
                pointerEvents="none"
              />
              <circle
                cx={x(step.start)}
                cy={(y(step.value) + y(steps[index - 1].value)) / 2}
                r={expanded ? 7 : 6}
                fill="var(--panel-2)"
                stroke={color}
                strokeWidth="1.8"
                vectorEffect="non-scaling-stroke"
                pointerEvents="none"
              />
              <title>Switch at {fmtTime(step.start)}. Drag sideways; double-click to merge.</title>
            </g>
          ),
        )}

        {boundaryTooltip && (
          <g
            transform={`translate(${tooltipX} ${PAD.t + 7})`}
            pointerEvents="none"
          >
            <rect
              width={tooltipWidth}
              height={expanded ? 26 : 23}
              rx="5"
              fill="var(--panel)"
              stroke={color}
              strokeOpacity="0.75"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={tooltipWidth / 2}
              y={expanded ? 17 : 15}
              textAnchor="middle"
              fontSize={expanded ? 10 : 9}
              fill="var(--text)"
              fontFamily="IBM Plex Mono"
            >
              switch · {fmtTime(boundaryTooltip.start)}
            </text>
          </g>
        )}

        {readout && (
          <text
            x={W - PAD.r}
            y={H - PAD.b - 7}
            textAnchor="end"
            fontSize={expanded ? 10 : 9}
            fill="var(--text)"
            fontFamily="IBM Plex Mono"
            pointerEvents="none"
          >
            {readout}
          </text>
        )}
      </svg>

      {introVisible && (
        <div className="schedule-first-hover-tip">
          Drag blocks ↑↓ · drag switch points ←→ · double-click to split / merge
        </div>
      )}
    </div>
  );
}

function BoundaryTimeInput({ value, disabled, min, max, onChange, ariaLabel, field }) {
  if (disabled) {
    return (
      <input
        type="text"
        className="schedule-time-input"
        value={fmtTime(value)}
        disabled
        aria-label={ariaLabel}
        data-field={field}
      />
    );
  }

  return (
    <input
      type="time"
      className="schedule-time-input"
      step="900"
      min={fmtTime(min)}
      max={fmtTime(max)}
      value={fmtTime(value)}
      aria-label={ariaLabel}
      data-field={field}
      onChange={(event) => {
        const hour = parseTime(event.target.value);
        if (hour != null) onChange(hour);
      }}
    />
  );
}

function SegmentTable({
  steps,
  selected,
  onSelect,
  updateBoundary,
  updateValue,
  deleteBlock,
  renderValueEditor,
  valueHeader,
}) {
  return (
    <div className="schedule-table-panel">
      <div className="mini-head schedule-table-title">
        <span>Segment table</span>
        <span>{steps.length} block{steps.length === 1 ? "" : "s"} · 15 min snap</span>
      </div>
      <div className="schedule-segment-table" role="table" aria-label="Schedule blocks">
        <div className="schedule-table-row schedule-table-header" role="row">
          <span role="columnheader">Start</span>
          <span role="columnheader">End</span>
          <span role="columnheader">{valueHeader}</span>
          <span role="columnheader" aria-label="Actions" />
        </div>

        {steps.map((step, index) => {
          const end = index + 1 < steps.length ? steps[index + 1].start : 24;
          const startMin = index > 0 ? steps[index - 1].start + SNAP_HOURS : 0;
          const startMax = end - SNAP_HOURS;
          const endMin = step.start + SNAP_HOURS;
          const endMax =
            (index + 2 < steps.length ? steps[index + 2].start : 24) - SNAP_HOURS;
          return (
            <div
              key={`row-${index}`}
              className={`schedule-table-row ${selected === index ? "selected" : ""}`}
              role="row"
              aria-selected={selected === index}
              data-block-index={index}
              onClick={() => onSelect(index)}
            >
              <div className="schedule-table-cell" role="cell" data-label="Start">
                <BoundaryTimeInput
                  value={step.start}
                  disabled={index === 0}
                  min={startMin}
                  max={startMax}
                  ariaLabel={`Block ${index + 1} start time`}
                  field="start"
                  onChange={(hour) => updateBoundary(index, hour)}
                />
              </div>
              <div className="schedule-table-cell" role="cell" data-label="End">
                <BoundaryTimeInput
                  value={end}
                  disabled={index === steps.length - 1}
                  min={endMin}
                  max={endMax}
                  ariaLabel={`Block ${index + 1} end time`}
                  field="end"
                  onChange={(hour) => updateBoundary(index + 1, hour)}
                />
              </div>
              <div
                className="schedule-table-cell schedule-value-cell"
                role="cell"
                data-label={valueHeader}
              >
                {renderValueEditor({
                  step,
                  index,
                  value: step.value,
                  onChange: (value) => updateValue(index, value),
                })}
              </div>
              <div className="schedule-table-cell schedule-delete-cell" role="cell">
                <button
                  type="button"
                  className="icon-btn schedule-delete-btn"
                  aria-label={`Delete block ${index + 1}`}
                  title={
                    steps.length <= 1
                      ? "Keep one block. Use the timetable toggle to return the battery to reactive control."
                      : `Delete ${fmtTime(step.start)}–${fmtTime(end)} block`
                  }
                  data-action="delete-block"
                  disabled={steps.length <= 1}
                  onClick={(event) => {
                    event.stopPropagation();
                    deleteBlock(index);
                  }}
                >
                  ×
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function EditorWorkspace({
  expanded,
  steps,
  selected,
  onSelect,
  commit,
  splitBlock,
  mergeBoundary,
  updateBoundary,
  updateValue,
  deleteBlock,
  yMin,
  yMax,
  snap,
  nudgeValue,
  label,
  guides,
  color,
  title,
  hint,
  renderValueEditor,
  valueHeader,
  helpId,
}) {
  const workspaceHelpId = `${helpId}-${expanded ? "expanded" : "inline"}`;
  return (
    <div className={`schedule-workspace ${expanded ? "is-expanded" : ""}`}>
      <div className="schedule-graph-panel">
        <ScheduleCanvas
          steps={steps}
          commit={commit}
          splitBlock={splitBlock}
          mergeBoundary={mergeBoundary}
          updateBoundary={updateBoundary}
          deleteBlock={deleteBlock}
          selected={selected}
          onSelect={onSelect}
          yMin={yMin}
          yMax={yMax}
          snap={snap}
          nudgeValue={nudgeValue}
          label={label}
          guides={guides}
          color={color}
          title={title}
          expanded={expanded}
          helpId={workspaceHelpId}
        />
        <div id={workspaceHelpId} className="schedule-legend asset-hint">
          <span><b>Mouse</b> drag block ↑↓ · drag switch point ←→ · double-click split / merge</span>
          <span><b>Keyboard</b> ↑↓ level · ←→ select · Shift+←/→ boundary · Delete block</span>
        </div>
        {hint && <div className="asset-hint schedule-domain-hint">{hint}</div>}
      </div>

      <SegmentTable
        steps={steps}
        selected={selected}
        onSelect={onSelect}
        updateBoundary={updateBoundary}
        updateValue={updateValue}
        deleteBlock={deleteBlock}
        renderValueEditor={renderValueEditor}
        valueHeader={valueHeader}
      />
    </div>
  );
}

export default function StepScheduleEditor({
  steps,
  onChange,
  yMin = 0,
  yMax,
  snap,
  nudgeValue,
  label,
  guides,
  color,
  hint,
  title = "24-hour schedule",
  kind = "schedule",
  valueHeader = "Level",
  renderValueEditor,
}) {
  const [history, setHistory] = useState([]);
  const [selected, setSelected] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const expandButtonRef = useRef(null);
  const closeButtonRef = useRef(null);
  const dialogTitleId = useId();
  const helpId = useId();

  useEffect(() => {
    if (selected >= steps.length) setSelected(Math.max(0, steps.length - 1));
  }, [selected, steps.length]);

  useEffect(() => {
    if (!expanded) return undefined;
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onDocumentKeyDown = (event) => {
      if (event.key === "Escape") setExpanded(false);
    };
    document.addEventListener("keydown", onDocumentKeyDown);
    requestAnimationFrame(() => closeButtonRef.current?.focus());
    return () => {
      document.body.style.overflow = oldOverflow;
      document.removeEventListener("keydown", onDocumentKeyDown);
      requestAnimationFrame(() => expandButtonRef.current?.focus());
    };
  }, [expanded]);

  const commit = (next, { record = true, before = steps } = {}) => {
    if (sameSteps(next, steps) && sameSteps(next, before)) return false;
    if (record && !sameSteps(before, next)) {
      const snapshot = cloneSteps(before);
      setHistory((current) => [...current.slice(-(MAX_HISTORY - 1)), snapshot]);
    }
    onChange(cloneSteps(next));
    return true;
  };

  const splitBlock = (index, rawHour) => {
    const end = index + 1 < steps.length ? steps[index + 1].start : 24;
    const hour = clamp(
      snapHour(rawHour),
      steps[index].start + SNAP_HOURS,
      end - SNAP_HOURS,
    );
    if (end - steps[index].start < SNAP_HOURS * 2) return;
    const next = cloneSteps(steps);
    next.splice(index + 1, 0, { ...next[index], start: hour, meta: cloneMeta(next[index].meta) });
    if (commit(next)) setSelected(index + 1);
  };

  const addBlock = () => {
    let largestIndex = -1;
    let largestDuration = 0;
    steps.forEach((step, index) => {
      const end = index + 1 < steps.length ? steps[index + 1].start : 24;
      const duration = end - step.start;
      if (duration >= SNAP_HOURS * 2 && duration > largestDuration) {
        largestIndex = index;
        largestDuration = duration;
      }
    });
    if (largestIndex < 0) return;
    const start = steps[largestIndex].start;
    const end = largestIndex + 1 < steps.length ? steps[largestIndex + 1].start : 24;
    splitBlock(largestIndex, (start + end) / 2);
  };

  const mergeBoundary = (index) => {
    if (index <= 0 || index >= steps.length) return;
    const next = steps.filter((_, stepIndex) => stepIndex !== index);
    if (commit(next)) setSelected(Math.max(0, index - 1));
  };

  const deleteBlock = (index) => {
    if (steps.length <= 1 || !steps[index]) return;
    const next = cloneSteps(steps);
    if (index === 0) {
      next[0] = { ...next[1], start: 0, meta: cloneMeta(next[1].meta) };
      next.splice(1, 1);
    } else {
      next.splice(index, 1);
    }
    if (commit(next)) setSelected(clamp(index - 1, 0, next.length - 1));
  };

  const updateBoundary = (index, rawHour) => {
    if (index <= 0 || index >= steps.length) return;
    const low = steps[index - 1].start + SNAP_HOURS;
    const high = (index + 1 < steps.length ? steps[index + 1].start : 24) - SNAP_HOURS;
    const hour = clamp(snapHour(rawHour), low, high);
    if (hour === steps[index].start) return;
    const next = cloneSteps(steps);
    next[index].start = hour;
    commit(next);
  };

  const updateValue = (index, rawValue) => {
    const authored = rawValue && typeof rawValue === "object" ? rawValue : null;
    const value = Number(authored ? authored.value : rawValue);
    if (!Number.isFinite(value) || !steps[index]) return;
    const next = cloneSteps(steps);
    // Numeric edits deliberately replace source metadata; preset/mode controls
    // may supply new metadata even when the effective numeric value is equal.
    next[index] = {
      start: next[index].start,
      value,
      ...(authored?.meta ? { meta: cloneMeta(authored.meta) } : {}),
    };
    if (sameSteps(next, steps)) return;
    commit(next);
  };

  const undo = () => {
    if (!history.length) return;
    const target = cloneSteps(history[history.length - 1]);
    setHistory(history.slice(0, -1));
    setSelected(clamp(selected, 0, target.length - 1));
    onChange(target);
  };

  const canAdd = steps.some((step, index) => {
    const end = index + 1 < steps.length ? steps[index + 1].start : 24;
    return end - step.start >= SNAP_HOURS * 2;
  });

  const workspaceProps = {
    steps,
    selected,
    onSelect: setSelected,
    commit,
    splitBlock,
    mergeBoundary,
    updateBoundary,
    updateValue,
    deleteBlock,
    yMin,
    yMax,
    snap,
    nudgeValue,
    label,
    guides,
    color,
    title,
    hint,
    renderValueEditor,
    valueHeader,
    helpId,
  };

  const toolbarActions = (
    <>
      <button
        type="button"
        className="mini-btn"
        data-action="add-block"
        onClick={addBlock}
        disabled={!canAdd}
        title={canAdd ? "Split the longest block at its midpoint" : "All blocks are already 15 minutes"}
      >
        + Add block
      </button>
      <button
        type="button"
        className="mini-btn"
        data-action="undo"
        onClick={undo}
        disabled={!history.length}
        title={history.length ? `Undo last change (${history.length} saved)` : "Nothing to undo"}
      >
        ↶ Undo
      </button>
    </>
  );

  return (
    <div
      className="schedule-editor"
      style={{ "--schedule-accent": color }}
      data-schedule-kind={kind}
    >
      <div className="schedule-toolbar">
        <div className="schedule-toolbar-title">
          <span>{title}</span>
          <small>{steps.length} block{steps.length === 1 ? "" : "s"}</small>
        </div>
        <div className="schedule-actions">
          {toolbarActions}
          <button
            ref={expandButtonRef}
            type="button"
            className="mini-btn schedule-expand-btn"
            data-action="expand"
            aria-haspopup="dialog"
            onClick={() => setExpanded(true)}
          >
            Expand ↗
          </button>
        </div>
      </div>

      <EditorWorkspace expanded={false} {...workspaceProps} />

      {expanded &&
        createPortal(
          <div
            className="schedule-modal-backdrop"
            onPointerDown={(event) => {
              if (event.target === event.currentTarget) setExpanded(false);
            }}
          >
            <div
              className="schedule-modal"
              style={{ "--schedule-accent": color }}
              role="dialog"
              aria-modal="true"
              aria-labelledby={dialogTitleId}
              data-schedule-modal={kind}
            >
              <div className="schedule-modal-head">
                <div>
                  <div className="eyebrow">Expanded schedule editor</div>
                  <h2 id={dialogTitleId}>{title}</h2>
                </div>
                <div className="schedule-modal-actions">
                  {toolbarActions}
                  <button
                    ref={closeButtonRef}
                    type="button"
                    className="mini-btn"
                    data-action="close-modal"
                    aria-label="Close expanded schedule editor"
                    onClick={() => setExpanded(false)}
                  >
                    Close ×
                  </button>
                </div>
              </div>
              <div className="schedule-modal-body">
                <EditorWorkspace expanded {...workspaceProps} />
              </div>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
