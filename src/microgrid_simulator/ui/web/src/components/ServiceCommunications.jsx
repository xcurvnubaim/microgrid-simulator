import React, { useCallback, useEffect, useMemo, useState } from "react";
import { fmt, getServiceCommunications } from "../api.js";

const statusLabel = (status) => {
  if (status === "connected" || status === "active" || status === "available") return "live";
  if (status === "unknown") return "checking";
  if (status === "degraded" || status === "unavailable") return "degraded";
  return "offline";
};

const prettyStatus = (status) => {
  if (status === "connected") return "Connected";
  if (status === "active") return "Active";
  if (status === "available") return "Available";
  if (status === "degraded") return "Degraded";
  if (status === "not connected") return "Not connected";
  if (status === "unavailable") return "Unavailable";
  if (status === "unknown") return "Unknown";
  return status || "Unknown";
};

function StatusPill({ status }) {
  return (
    <span className={`service-status ${statusLabel(status)}`}>
      <span className="service-status-dot" />
      {prettyStatus(status)}
    </span>
  );
}

function ServiceCard({ service }) {
  return (
    <article className={`service-card ${service.kind}`}>
      <div className="service-card-head">
        <div>
          <span className="service-kind">{service.kind}</span>
          <h3>{service.label}</h3>
        </div>
        <StatusPill status={service.status} />
      </div>
      <p>{service.description}</p>
      <div className="service-card-stats">
        <span><b>{service.instances || 0}</b> instance{service.instances === 1 ? "" : "s"}</span>
        <span><b>{fmt(service.in_msgs, 0)}</b> in</span>
        <span><b>{fmt(service.out_msgs, 0)}</b> out</span>
      </div>
      {service.subscriptions?.length > 0 && (
        <div className="service-subscriptions">
          {service.subscriptions.map((subscription) => (
            <div className="service-subscription" key={`${service.id}-${subscription.subject}`}>
              <code>{subscription.subject}</code>
              {subscription.queue && <span>queue: {subscription.queue}</span>}
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function CommunicationPath({ path, labels }) {
  const delivered = path.delivered_messages == null ? null : fmt(path.delivered_messages, 0);
  return (
    <article className={`communication-path ${statusLabel(path.status)}`}>
      <div className="communication-path-head">
        <div>
          <span className="service-kind">{path.transport}</span>
          <h3>{path.label}</h3>
        </div>
        <StatusPill status={path.status} />
      </div>
      <div className="communication-route" aria-label={`${labels[path.source]} to ${labels[path.target]}`}>
        <span>{labels[path.source] || path.source}</span>
        <span className="communication-arrow" aria-hidden="true">→</span>
        <span>{labels[path.target] || path.target}</span>
      </div>
      <code className="communication-subject">{path.subject}</code>
      <p>{path.purpose}</p>
      <div className="communication-path-meta">
        <span>{delivered == null ? "event stream" : `${delivered} delivered`}</span>
        {path.pending_messages != null && <span>{fmt(path.pending_messages, 0)} pending</span>}
      </div>
    </article>
  );
}

export default function ServiceCommunications() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [messageFilter, setMessageFilter] = useState("all");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const payload = await getServiceCommunications();
      setData(payload);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const labels = useMemo(
    () => Object.fromEntries((data?.services ?? []).map((service) => [service.id, service.label])),
    [data],
  );
  const connectedServices = (data?.services ?? []).filter(
    (service) => service.id !== "control-room" && service.status === "connected",
  ).length;
  const activePaths = (data?.paths ?? []).filter((path) => path.active === true).length;
  const messages = data?.message_tap?.messages ?? [];
  const visibleMessages = useMemo(
    () => messages.filter((message) => messageFilter === "all" || message.kind === messageFilter),
    [messages, messageFilter],
  );

  return (
    <div className="communications-page">
      <section className="communications-hero">
        <div>
          <span className="eyebrow">Service communications</span>
          <h2>How the control room talks to its services</h2>
          <p>
            A readable view of the simulator, telemetry, EMS, NATS, and browser event paths.
            The browser receives events through the simulator and never needs NATS credentials.
          </p>
        </div>
        <div className="communications-actions">
          <StatusPill status={data?.status === "ok" ? "connected" : data?.status || "unknown"} />
          <button className="ghost-btn" onClick={refresh} disabled={loading}>
            {loading ? "Refreshing…" : "↻ Refresh"}
          </button>
        </div>
      </section>

      {error && <div className="error-bar"><span>{error}</span></div>}
      {data?.error && (
        <div className="communications-warning">
          NATS monitoring is unavailable. The map is still shown, but live connection and
          subscription status cannot be confirmed: {data.error}
        </div>
      )}

      {!data ? (
        <div className="empty communications-loading">Loading service communications…</div>
      ) : (
        <>
          <div className="communications-kpis">
            <div className="communication-kpi"><span>Services connected</span><b>{connectedServices} / {data.services.length - 1}</b><small>excluding the browser</small></div>
            <div className="communication-kpi"><span>Active paths</span><b>{activePaths} / {data.paths.length}</b><small>logical service links</small></div>
            <div className="communication-kpi"><span>NATS traffic</span><b>{fmt(data.server.in_msgs_per_sec, 1)} in/s</b><small>{fmt(data.server.out_msgs_per_sec, 1)} out/s</small></div>
            <div className="communication-kpi"><span>Broker</span><b>{data.server.version || "NATS"}</b><small>{data.server.connections} client connections</small></div>
          </div>

          <section className="panel service-map-panel">
            <div className="panel-head">
              <span className="eyebrow">Live service map</span>
              <span className="note">refreshes every 5 seconds · arrows show logical message direction</span>
            </div>
            <div className="service-map">
              <div className="service-map-column">
                <div className="service-map-node browser-node"><span>UI</span><b>Control room</b><small>human view</small></div>
                <div className="service-map-node simulator-node"><span>CORE</span><b>Simulator</b><small>orchestrator</small></div>
              </div>
              <div className="service-map-bus">
                <div className="service-map-node broker-node"><span>BUS</span><b>NATS</b><small>Core + JetStream</small></div>
                <div className="service-map-bus-note">request / reply<br />durable event history</div>
              </div>
              <div className="service-map-column">
                <div className="service-map-node telemetry-node"><span>DATA</span><b>Telemetry</b><small>replay windows</small></div>
                <div className="service-map-node ems-node"><span>EMS</span><b>EMS</b><small>dispatch + audit</small></div>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <span className="eyebrow">Communication paths</span>
              <span className="note">what each link carries</span>
            </div>
            <div className="communication-path-grid">
              {data.paths.map((path) => <CommunicationPath key={path.id} path={path} labels={labels} />)}
            </div>
          </section>

          <section className="panel message-inspector-panel">
            <div className="panel-head">
              <span className="eyebrow">Message inspector</span>
              <span className="note">newest {messages.length} of {data.message_tap?.limit ?? 100} messages</span>
            </div>
            <div className="message-toolbar">
              <StatusPill status={data.message_tap?.status || "unknown"} />
              <label htmlFor="message-filter">show</label>
              <select id="message-filter" value={messageFilter} onChange={(event) => setMessageFilter(event.target.value)}>
                <option value="all">all messages</option>
                <option value="request">requests</option>
                <option value="reply">replies</option>
                <option value="event">dashboard events</option>
              </select>
              <span className="message-toolbar-note">Read-only tap · payloads are formatted when they contain JSON</span>
            </div>
            {visibleMessages.length > 0 ? (
              <div className="message-feed">
                {visibleMessages.map((message, index) => (
                  <details className={`message-entry ${message.kind}`} key={message.id} open={index === 0}>
                    <summary>
                      <span className="message-kind">{message.kind}</span>
                      <code>{message.subject}</code>
                      <span className="message-route">{message.source} → {message.target}</span>
                      <span className="message-time">
                        {message.received_at ? new Date(message.received_at).toLocaleTimeString() : "—"}
                      </span>
                    </summary>
                    <div className="message-body">
                      <div className="message-meta">
                        <span>{message.bytes} bytes</span>
                        {message.reply && <span>reply: {message.reply}</span>}
                        {Object.entries(message.headers ?? {}).map(([key, value]) => (
                          <span key={key}>{key}: {value}</span>
                        ))}
                      </div>
                      <pre>{message.payload || "(empty payload)"}</pre>
                      {message.truncated && <small>Payload display capped at 8,000 characters.</small>}
                    </div>
                  </details>
                ))}
              </div>
            ) : (
              <div className="message-empty">
                {data.message_tap?.status === "connected"
                  ? "No application messages captured yet — start a simulation or EMS event run."
                  : "The message observer is not connected; the service map remains available without payloads."}
              </div>
            )}
          </section>

          <section className="panel">
            <div className="panel-head">
              <span className="eyebrow">Connected services</span>
              <span className="note">readable summary of NATS clients and subscriptions</span>
            </div>
            <div className="service-card-grid">
              {data.services.map((service) => <ServiceCard key={service.id} service={service} />)}
            </div>
            <div className="communications-footnote">
              Last updated {data.refreshed_at ? new Date(data.refreshed_at).toLocaleTimeString() : "—"}. Message counts are cumulative for each current NATS client connection.
            </div>
          </section>
        </>
      )}
    </div>
  );
}
