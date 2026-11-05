"use strict";

const STATUS_INTERVAL_MS = 500;
const ALERTS_INTERVAL_MS = 2000;

const HEADLINES = {
  normal: "Driver attentive",
  warning: "Attention needed",
  critical: "Driver at risk",
  unknown: "No data",
};

const REASON_LABELS = {
  prolonged_eye_closure: "Eyes closed too long",
  high_perclos: "Eyes often closed",
  frequent_yawning: "Frequent yawning",
  head_nodding: "Head dropping",
  head_tilted: "Head tilted",
  looking_away: "Looking away",
  driver_not_visible: "Driver not visible",
  model_drowsy: "Classifier: drowsy",
};

const byId = (id) => document.getElementById(id);
const fixed = (value, digits) => (value === null || value === undefined ? "—" : value.toFixed(digits));
const signed = (value) => (value >= 0 ? "+" : "") + value.toFixed(0);

function setText(id, text) {
  const element = byId(id);
  if (element.textContent !== text) {
    element.textContent = text;
  }
}

function renderReasons(reasons) {
  const list = byId("state-reasons");
  const items = reasons.map((reason) => {
    const item = document.createElement("li");
    item.textContent = REASON_LABELS[reason] || reason;
    return item;
  });
  list.replaceChildren(...items);
}

function renderMetrics(status) {
  const metrics = status.metrics || {};
  const fatigue = status.fatigue || {};
  const pose = metrics.head_pose;

  setText("metric-perclos", fatigue.perclos == null ? "—" : `${(fatigue.perclos * 100).toFixed(0)}%`);
  setText("metric-closure", fatigue.eye_closure_seconds == null ? "—" : `${fatigue.eye_closure_seconds.toFixed(1)} s`);
  setText("metric-yawns", fatigue.yawns_in_window == null ? "—" : String(fatigue.yawns_in_window));
  setText("metric-ear", metrics.ear_left == null ? "—" : `${fixed(metrics.ear_left, 2)} / ${fixed(metrics.ear_right, 2)}`);
  setText("metric-mar", fixed(metrics.mar, 2));
  setText("metric-pose", pose ? `${signed(pose.pitch)} / ${signed(pose.yaw)} / ${signed(pose.roll)}` : "—");
}

// A dropped MJPEG <img> keeps showing its last frame and never reconnects by itself, which on a
// monitoring console would look like live video of a driver who may have moved on.
let streamInterrupted = false;

function reconnectStream() {
  const stream = byId("stream");
  if (stream && streamInterrupted) {
    const url = new URL(stream.src);
    url.searchParams.set("reconnect", Date.now());
    stream.src = url.toString();
  }
  streamInterrupted = false;
}

function renderStatus(status) {
  reconnectStream();
  document.body.dataset.level = status.level;
  setText("state-heading", HEADLINES[status.level] || status.level);
  renderReasons(status.reasons || []);
  renderMetrics(status);

  setText("service-camera", status.service.camera_connected ? "Connected" : "Disconnected");
  setText("service-fps", `${status.service.fps.toFixed(1)} fps`);
  if (status.calibrated !== undefined) {
    setText("service-calibration", status.calibrated ? "Complete" : "In progress");
  }
}

function renderOffline() {
  streamInterrupted = true;
  document.body.dataset.level = "unknown";
  setText("state-heading", "Service unreachable");
  setText("service-camera", "—");
  setText("service-fps", "—");
  renderReasons([]);
}

function renderAlerts(alerts) {
  byId("log-empty").hidden = alerts.length > 0;
  const rows = alerts.map((alert) => {
    const row = document.createElement("li");

    const time = document.createElement("time");
    time.dateTime = alert.time;
    time.textContent = new Date(alert.time).toLocaleTimeString();

    const level = document.createElement("span");
    level.className = `level level-${alert.level}`;
    level.textContent = alert.level;

    row.append(time, level);
    if (alert.reasons.length > 0) {
      const detail = document.createElement("span");
      detail.className = "detail";
      detail.textContent = alert.reasons.map((reason) => REASON_LABELS[reason] || reason).join(", ");
      row.append(detail);
    }
    return row;
  });
  byId("log-list").replaceChildren(...rows);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }
  return response.json();
}

function poll(url, intervalMs, onData, onError) {
  const tick = async () => {
    try {
      onData(await fetchJson(url));
    } catch (error) {
      onError(error);
    }
    window.setTimeout(tick, intervalMs);
  };
  tick();
}

function connectTabs() {
  const stream = byId("stream");
  if (!stream) {
    return;
  }
  const tabs = document.querySelectorAll(".tabs button");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((other) => other.setAttribute("aria-selected", String(other === tab)));
      stream.src = `/api/v1/streams/${tab.dataset.view}`;
    });
  });
}

connectTabs();
poll("/api/v1/status", STATUS_INTERVAL_MS, renderStatus, renderOffline);
poll("/api/v1/alerts?limit=50", ALERTS_INTERVAL_MS, (body) => renderAlerts(body.alerts), () => {});
