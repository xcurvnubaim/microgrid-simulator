import React, { useMemo } from "react";
import { COLORS } from "../api.js";
import StepScheduleEditor, { segmentsToSteps, stepsToSegments } from "./StepScheduleEditor.jsx";

/* Drag-editable 24h diesel timetable (settings.diesel_schedule) — the
   `schedule` policy plays it back by clock time. Named levels (max/min/off)
   survive the segments↔steps round trip, other values become explicit kW.
   Drawing and drag interactions live in StepScheduleEditor. */

const levelToKw = (level, minKw, maxKw) =>
  level === "max" ? maxKw : level === "min" ? minKw : level === "off" ? 0 : Number(level);

const kwToLevel = (kw, minKw, maxKw) =>
  kw <= 0 ? "off" : kw === maxKw ? "max" : kw === minKw ? "min" : kw;

export default function ScheduleGraph({ settings, onPatch }) {
  const minKw = settings.diesel?.min_kw ?? 45;
  const maxKw = Math.max(settings.diesel?.max_kw ?? 150, minKw);
  const chartMaxKw = Math.max(1, maxKw);
  const segments = settings.diesel_schedule?.segments ?? [];
  const steps = useMemo(
    () =>
      segmentsToSteps(
        segments,
        (seg) => levelToKw(seg.level, minKw, maxKw),
        (seg) => ({ level: seg.level }),
      ),
    [segments, minKw, maxKw],
  );

  const snap = (kw) => {
    const snapBand = maxKw * 0.07;
    for (const target of [0, minKw, maxKw]) {
      if (Math.abs(kw - target) < snapBand) return target;
    }
    if (kw < minKw) return kw < minKw / 2 ? 0 : minKw; // no wet-stacking band
    return Math.round(kw / 5) * 5;
  };

  const label = (kw) =>
    kw <= 0 ? "off" : kw === maxKw ? `max ${maxKw} kW` : kw === minKw ? `min ${minKw} kW` : `${kw} kW`;

  const nudgeValue = (kw, direction) => {
    if (direction > 0) {
      if (kw <= 0) return Math.min(maxKw, Math.max(minKw, Math.min(5, maxKw)));
      return Math.min(maxKw, Math.max(minKw, Math.round((kw + 5) / 5) * 5));
    }
    if (kw <= minKw) return 0;
    return Math.max(minKw, Math.round((kw - 5) / 5) * 5);
  };

  const renderValueEditor = ({ step, index, value, onChange }) => {
    const sourceLevel = step.meta?.level;
    const preset =
      typeof sourceLevel === "string"
        ? sourceLevel
        : value <= 0
          ? "off"
          : value === maxKw
            ? "max"
            : value === minKw
              ? "min"
              : "custom";
    return (
      <div className="schedule-value-fields">
        <select
          value={preset}
          aria-label={`Block ${index + 1} diesel level preset`}
          data-field="level"
          onChange={(event) => {
            const next = event.target.value;
            if (next === "off") onChange({ value: 0, meta: { level: "off" } });
            else if (next === "min") onChange({ value: minKw, meta: { level: "min" } });
            else if (next === "max") onChange({ value: maxKw, meta: { level: "max" } });
            else onChange(value);
          }}
        >
          <option value="off">Off</option>
          <option value="min">Min · {minKw} kW</option>
          <option value="max">Max · {maxKw} kW</option>
          <option value="custom">Custom kW</option>
        </select>
        <label className="schedule-kw-field">
          <input
            type="number"
            min="0"
            max={maxKw}
            step="1"
            value={Math.max(0, value)}
            aria-label={`Block ${index + 1} diesel kilowatts`}
            data-field="kw"
            onChange={(event) => {
              const raw = Number(event.target.value);
              if (!Number.isFinite(raw)) return;
              const bounded = Math.min(maxKw, Math.max(0, raw));
              onChange(bounded > 0 && bounded < minKw ? minKw : bounded);
            }}
          />
          <span>kW</span>
        </label>
      </div>
    );
  };

  return (
    <StepScheduleEditor
      steps={steps}
      onChange={(next) =>
        onPatch({
          diesel_schedule: {
            segments: stepsToSegments(next, (kw, step) => ({
              level:
                step.meta && Object.prototype.hasOwnProperty.call(step.meta, "level")
                  ? step.meta.level
                  : kwToLevel(kw, minKw, maxKw),
            })),
          },
        })
      }
      yMax={chartMaxKw}
      snap={snap}
      nudgeValue={nudgeValue}
      label={label}
      guides={[
        { value: 0, text: "off" },
        { value: minKw, text: `${minKw}` },
        { value: maxKw, text: `${maxKw}` },
      ]}
      color={COLORS.diesel}
      kind="diesel"
      title="Diesel 24-hour timetable"
      valueHeader="Level / kW"
      renderValueEditor={renderValueEditor}
      hint={
        <>
          <b>schedule</b> policy only — output is off, the named minimum/maximum, or an exact
          stable-band kW setpoint.
        </>
      }
    />
  );
}
