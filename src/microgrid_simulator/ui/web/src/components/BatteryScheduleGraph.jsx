import React, { useMemo } from "react";
import { COLORS } from "../api.js";
import StepScheduleEditor, { segmentsToSteps, stepsToSegments } from "./StepScheduleEditor.jsx";

/* Drag-editable 24h battery timetable (settings.battery_schedule) — the
   `schedule` policy plays it back by clock time on a signed axis:

     above zero  -> discharge, continuous kW (snaps to 0 / max, else 5 kW grid)
     below zero  -> charge, binary on/off (any drag past halfway = full rate;
                    the backend clips by SOC and picks the PV→diesel→grid split)
     at zero     -> idle

   An empty timetable (the enable toggle off) keeps the schedule policy's
   reactive default: charge on PV surplus, discharge to backstop diesel. */

export const DEFAULT_BATTERY_SEGMENTS = [
  { start_hour: 10.0, end_hour: 15.0, mode: "charge" },
  { start_hour: 18.0, end_hour: 22.0, mode: "discharge", level: "max" },
];

export default function BatteryScheduleGraph({ settings, onPatch }) {
  const chargeKw = Math.max(1, (settings.battery?.max_charge_mw ?? 0.25) * 1000);
  const dischargeKw = Math.max(1, (settings.battery?.max_discharge_mw ?? 0.25) * 1000);
  const segments = settings.battery_schedule?.segments ?? [];

  const segValue = (seg) =>
    seg.mode === "charge"
      ? -chargeKw
      : seg.mode === "discharge"
        ? seg.level === "max"
          ? dischargeKw
          : Number(seg.level)
        : 0;
  const steps = useMemo(
    () =>
      segmentsToSteps(segments, segValue, (seg) => ({
        mode: seg.mode,
        ...(seg.level !== undefined ? { level: seg.level } : {}),
      })),
    [segments, chargeKw, dischargeKw],
  );

  const segFromValue = (v) =>
    v < 0
      ? { mode: "charge" }
      : v > 0
        ? { mode: "discharge", level: v === dischargeKw ? "max" : v }
        : { mode: "idle" };

  const snap = (v) => {
    if (v < 0) return v < -chargeKw / 2 ? -chargeKw : 0; // charge is on/off
    const band = dischargeKw * 0.07;
    if (v < band) return 0;
    if (Math.abs(v - dischargeKw) < band) return dischargeKw;
    return Math.round(v / 5) * 5;
  };

  const label = (v) =>
    v < 0 ? "charge (full rate)" : v === 0 ? "idle" : v === dischargeKw ? `max ${dischargeKw} kW` : `${v} kW`;

  const nudgeValue = (value, direction) => {
    if (direction > 0) {
      if (value < 0) return 0;
      if (value === 0) return Math.min(5, dischargeKw);
      return Math.min(dischargeKw, Math.round((value + 5) / 5) * 5);
    }
    if (value > 0) return value <= 5 ? 0 : Math.max(5, Math.round((value - 5) / 5) * 5);
    if (value === 0) return -chargeKw;
    return -chargeKw;
  };

  const renderValueEditor = ({ step, index, value, onChange }) => {
    const mode = step.meta?.mode ?? (value < 0 ? "charge" : value > 0 ? "discharge" : "idle");
    return (
      <div className="schedule-value-fields">
        <select
          value={mode}
          aria-label={`Block ${index + 1} battery mode`}
          data-field="mode"
          onChange={(event) => {
            const nextMode = event.target.value;
            if (nextMode === "charge") {
              onChange({ value: -chargeKw, meta: { mode: "charge" } });
            } else if (nextMode === "idle") {
              onChange({ value: 0, meta: { mode: "idle" } });
            } else {
              const kw = value > 0 ? Math.min(value, dischargeKw) : dischargeKw;
              onChange({
                value: kw,
                meta: { mode: "discharge", level: kw === dischargeKw ? "max" : kw },
              });
            }
          }}
        >
          <option value="charge">Charge</option>
          <option value="discharge">Discharge</option>
          <option value="idle">Idle</option>
        </select>
        <label className="schedule-kw-field">
          <input
            type="number"
            min="0"
            max={dischargeKw}
            step="1"
            value={mode === "discharge" ? Math.max(0, value) : ""}
            placeholder={mode === "charge" ? `full ${chargeKw}` : "—"}
            disabled={mode !== "discharge"}
            aria-label={`Block ${index + 1} battery discharge kilowatts`}
            data-field="kw"
            onChange={(event) => {
              const raw = Number(event.target.value);
              if (!Number.isFinite(raw)) return;
              onChange(Math.min(dischargeKw, Math.max(0, raw)));
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
          battery_schedule: {
            segments: stepsToSegments(next, (value, step) =>
              step.meta?.mode ? { ...step.meta } : segFromValue(value),
            ),
          },
        })
      }
      yMin={-chargeKw}
      yMax={dischargeKw}
      snap={snap}
      nudgeValue={nudgeValue}
      label={label}
      guides={[
        { value: -chargeKw, text: "chg" },
        { value: 0, text: "idle" },
        { value: dischargeKw, text: `${dischargeKw}` },
      ]}
      color={COLORS.battery}
      kind="battery"
      title="Battery 24-hour timetable"
      valueHeader="Mode / kW"
      renderValueEditor={renderValueEditor}
      hint={
        <>
          <b>schedule</b> policy only — charge runs at full configured rate, discharge accepts an
          exact kW setpoint, and idle commands zero. Disable the timetable for reactive fallback.
        </>
      }
    />
  );
}
