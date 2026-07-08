import React from "react";

export function Num({ label, value, onChange, step = 0.01, min, max }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value ?? 0}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))}
      />
    </label>
  );
}

export function Check({ label, checked, onChange }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}
