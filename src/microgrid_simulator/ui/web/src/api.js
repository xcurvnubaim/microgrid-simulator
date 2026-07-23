export async function getDefaults() {
  const res = await fetch("/api/defaults");
  if (!res.ok) throw new Error(`defaults failed: ${res.status}`);
  return res.json();
}

export async function simulate(settings, policy, seed) {
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, policy, seed }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `simulate failed: ${res.status}`);
  }
  return res.json();
}

/* Stream one episode tick-by-tick. Calls onEvent for every NDJSON event
   ({type:"meta"|"row"|"end", ...}); resolves when the episode finishes and
   rejects on network/server errors. Abort via an AbortController signal. */
export async function simulateStream(settings, policy, seed, { signal, onEvent } = {}) {
  const res = await fetch("/api/simulate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, policy, seed }),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `simulate failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.trim()) onEvent?.(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent?.(JSON.parse(buffer));
}

export async function uploadDemand(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/demand/upload", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `upload failed: ${res.status}`);
  }
  return res.json();
}

export const COLORS = {
  solar: "var(--solar)",
  grid: "var(--grid)",
  battery: "var(--battery)",
  diesel: "var(--diesel)",
  load: "var(--load)",
  volt: "var(--volt)",
  ok: "var(--ok)",
  unserved: "var(--unserved)",
  accent: "var(--accent)",
};

export const THEME = {
  panel: "var(--panel)",
  panelAlt: "var(--panel-2)",
  inset: "var(--inset)",
  line: "var(--line)",
  lineSoft: "var(--line-soft)",
  text: "var(--text)",
  muted: "var(--muted)",
  faint: "var(--faint)",
};

export const fmt = (v, d = 1) =>
  v == null || Number.isNaN(v)
    ? "—"
    : Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });

export const hourLabel = (h) => {
  const hh = Math.floor(h % 24);
  const mm = Math.round((h % 1) * 60);
  const day = Math.floor(h / 24);
  const base = `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  return day > 0 ? `d${day} ${base}` : base;
};
