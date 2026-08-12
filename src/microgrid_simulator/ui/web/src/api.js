export async function getDefaults() {
  const res = await fetch("/api/defaults");
  if (!res.ok) throw new Error(`defaults failed: ${res.status}`);
  return res.json();
}

export async function getRuntime() {
  const res = await fetch("/api/runtime", { cache: "no-store" });
  if (!res.ok) throw new Error(`runtime configuration failed: ${res.status}`);
  return res.json();
}

export async function getServiceCommunications() {
  const res = await fetch("/api/service-communications", { cache: "no-store" });
  if (!res.ok) throw new Error(`service communications failed: ${res.status}`);
  return res.json();
}

export async function simulate(settings, policy, seed, rl = {}) {
  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, policy, seed, ...rl }),
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
export async function simulateStream(settings, policy, seed, { signal, onEvent, rl = {} } = {}) {
  const res = await fetch("/api/simulate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, policy, seed, ...rl }),
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

/* Simulation-only EMS stream. The WebSocket carries versioned telemetry,
   requested commands, and realized acknowledgments; it never controls hardware. */
export function emsStream(_settings, _policy, _seed, { signal, onEvent } = {}) {
  return new Promise((resolve, reject) => {
    let socket;
    let finished = false;
    const close = () => socket?.readyState < WebSocket.CLOSING && socket.close();
    signal?.addEventListener("abort", close, { once: true });
    const connect = (url, initialMessage) => {
      socket = new WebSocket(url);
      socket.onopen = () => initialMessage && socket.send(JSON.stringify(initialMessage));
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data);
        onEvent?.(event);
        if (event.type === "error") {
          finished = true;
          close();
          reject(new Error(event.message));
        } else if (event.type === "ems_end") {
          finished = true;
          close();
          resolve();
        }
      };
      socket.onerror = () => {
        if (!finished) reject(new Error("EMS WebSocket connection failed"));
      };
      socket.onclose = () => {
        signal?.removeEventListener("abort", close);
        if (!finished) {
          if (signal?.aborted) resolve();
          else reject(new Error("EMS WebSocket closed before the episode ended"));
        }
      };
    };

    const discover = async () => {
      while (!signal?.aborted) {
        const response = await fetch("/api/ems/latest", { cache: "no-store", signal });
        if (!response.ok) throw new Error(`EMS discovery failed: ${response.status}`);
        const { run } = await response.json();
        if (run) return run;
        await new Promise((done) => setTimeout(done, 750));
      }
      return null;
    };

    discover()
      .then((run) => {
        if (!run) return resolve();
        const scheme = window.location.protocol === "https:" ? "wss" : "ws";
        connect(`${scheme}://${window.location.host}/api/ems/events/${run.run_id}`);
      })
      .catch(reject);
  });
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
