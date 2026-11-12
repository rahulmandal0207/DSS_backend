// The demo page: camera or video file in, MediaPipe face landmarks, then the DSS rules in
// fatigue.js. Nothing leaves the browser.

import {
  EYE_OUTER_CORNERS,
  FreezeDetector,
  INNER_LIPS,
  LEFT_EYE,
  MOUTH_CORNERS,
  RIGHT_EYE,
  TemporalAnalyzer,
  measure,
} from "./fatigue.js";

const MEDIAPIPE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Sampled, not averaged, so the camera's own noise shows a live picture. Matches dss/vision/freeze.py.
const FREEZE_STRIDE = 8;
const HISTORY_LIMIT = 50;
const FPS_WINDOW = 30;
const TICK_MS = 1000 / 30;
// Without new frames for this long, the page says so rather than showing a stale verdict.
const STALL_MS = 3000;

const HEADLINES = {
  normal: "Driver attentive",
  warning: "Attention needed",
  critical: "Driver at risk",
  idle: "Starting",
};

const REASON_LABELS = {
  prolonged_eye_closure: "Eyes closed too long",
  high_perclos: "Eyes often closed",
  frequent_yawning: "Frequent yawning",
  head_nodding: "Head dropping",
  head_tilted: "Head tilted",
  looking_away: "Looking away",
  driver_not_visible: "Driver not visible",
  camera_frozen: "Camera image frozen",
};

const COLOURS = { normal: "#3fb950", warning: "#e3a008", critical: "#f0524b" };

// Frequency in Hz, then beep length and the gap between beeps in milliseconds.
const TONES = { warning: [660, 250, 2000], critical: [880, 400, 600] };

const byId = (id) => document.getElementById(id);
const video = byId("video");
const overlay = byId("overlay");
const stage = byId("stage");
const sampler = document.createElement("canvas");
const samplerContext = sampler.getContext("2d", { willReadFrequently: true });

const state = {
  landmarker: null,
  loading: null,
  stream: null,
  objectUrl: null,
  analyzer: null,
  freeze: null,
  running: false,
  timer: null,
  lastAnalysedAt: 0,
  stalled: false,
  lastFrameCount: 0,
  lastVideoTime: -1,
  lastTimestamp: 0,
  frameTimes: [],
  lastAlertKey: "",
  history: [],
  audio: null,
  alarmLevel: "normal",
  alarmTimer: null,
};

// ---- Loading ----------------------------------------------------------------------------

function showMessage(text, info = false) {
  const element = byId("message");
  element.textContent = text;
  element.classList.toggle("info", info);
}

async function createLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import(`${MEDIAPIPE_URL}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_URL}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  try {
    return await FaceLandmarker.createFromOptions(files, options("GPU"));
  } catch {
    // Some devices have no usable WebGL. The CPU is slower but works everywhere.
    return FaceLandmarker.createFromOptions(files, options("CPU"));
  }
}

function loadLandmarker() {
  if (!state.loading) {
    state.loading = createLandmarker().then(
      (landmarker) => (state.landmarker = landmarker),
      (error) => {
        state.loading = null;
        throw error;
      },
    );
  }
  return state.loading;
}

function setBusy(busy) {
  byId("start-camera").disabled = busy;
  byId("video-file").disabled = busy;
}

// ---- Sources ----------------------------------------------------------------------------

function cameraError(error) {
  switch (error?.name) {
    case "NotAllowedError":
      return "Camera permission was refused. Allow it in your browser's site settings, or use a video file.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera was found. You can use a video file instead.";
    case "NotReadableError":
      return "The camera is in use by another application. Close it and try again.";
    default:
      return `Could not start: ${error?.message ?? error}`;
  }
}

async function startCamera(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    showMessage("This browser cannot open a camera here. Try a current browser over HTTPS, or use a video file.");
    return;
  }
  stop();
  setBusy(true);
  showMessage("Loading the face model. The first visit takes a few seconds.", true);
  try {
    const constraints = deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } };
    const [stream] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: constraints, audio: false }),
      loadLandmarker(),
    ]);
    state.stream = stream;
    await play({ srcObject: stream });
    await listCameras(stream.getVideoTracks()[0]?.getSettings().deviceId);
    begin({ mirrored: true });
  } catch (error) {
    stop();
    showMessage(error?.name ? cameraError(error) : `Could not load the face model: ${error?.message ?? error}`);
  } finally {
    setBusy(false);
  }
}

async function startFile(file) {
  stop();
  setBusy(true);
  showMessage("Loading the face model. The first visit takes a few seconds.", true);
  try {
    await loadLandmarker();
    state.objectUrl = URL.createObjectURL(file);
    await play({ src: state.objectUrl, loop: true });
    byId("camera-field").hidden = true;
    begin({ mirrored: false });
  } catch (error) {
    stop();
    showMessage(`Could not play that file: ${error?.message ?? error}`);
  } finally {
    setBusy(false);
  }
}

async function play({ srcObject = null, src = "", loop = false }) {
  video.srcObject = srcObject;
  if (src) {
    video.src = src;
  }
  video.loop = loop;
  video.muted = true;
  await video.play();
}

async function listCameras(activeId) {
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const select = byId("camera-select");
  select.replaceChildren(
    ...cameras.map((camera, index) => {
      const option = new Option(camera.label || `Camera ${index + 1}`, camera.deviceId);
      option.selected = camera.deviceId === activeId;
      return option;
    }),
  );
  // Virtual cameras are common, and some show only a still image, so let people choose.
  byId("camera-field").hidden = cameras.length < 2;
}

function stop() {
  state.running = false;
  clearTimeout(state.timer);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  setAlarm("normal");
}

// ---- Analysis loop ----------------------------------------------------------------------

function begin({ mirrored }) {
  state.analyzer = new TemporalAnalyzer();
  state.freeze = new FreezeDetector();
  state.lastAnalysedAt = 0;
  state.stalled = false;
  state.lastFrameCount = 0;
  state.lastVideoTime = -1;
  state.frameTimes = [];
  state.lastAlertKey = "";
  state.history = [];
  renderHistory();
  stage.classList.toggle("mirrored", mirrored);
  showMessage("");
  byId("intro").hidden = true;
  byId("monitor").hidden = false;
  if (byId("sound").checked) {
    enableAudio();
  }
  state.running = true;
  scheduleNext();
}

// A timer drives the loop rather than requestAnimationFrame or requestVideoFrameCallback. Those
// only fire while the video is being painted, so scrolling it out of view on a phone would quietly
// stop the monitoring. A timer keeps running for as long as the tab is open.
function scheduleNext(delay = 0) {
  if (state.running) {
    state.timer = setTimeout(onTick, delay);
  }
}

/** True once per decoded frame, so a repeated frame is never mistaken for a frozen camera. */
function hasNewFrame() {
  const decoded = video.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
  if (decoded > 0) {
    const fresh = decoded !== state.lastFrameCount;
    state.lastFrameCount = decoded;
    return fresh;
  }
  // Browsers without a frame counter: fall back to the playback clock.
  const fresh = video.currentTime !== state.lastVideoTime;
  state.lastVideoTime = video.currentTime;
  return fresh;
}

function onTick() {
  const started = performance.now();
  if (video.readyState >= 2 && video.videoWidth && hasNewFrame()) {
    if (state.stalled) {
      // Holds and windows would otherwise span the gap as if the driver had been watched.
      restartRules();
      state.stalled = false;
    }
    try {
      analyse();
    } catch (error) {
      console.error("Skipped a frame that failed analysis", error);
    }
    state.lastAnalysedAt = performance.now();
  } else if (state.lastAnalysedAt && started - state.lastAnalysedAt > STALL_MS && !state.stalled) {
    showStalled();
  }
  scheduleNext(Math.max(0, TICK_MS - (performance.now() - started)));
}

/** Start the time-based rules afresh, keeping a finished head-position calibration. */
function restartRules() {
  const calibrator = state.analyzer.calibrator;
  state.analyzer = new TemporalAnalyzer();
  if (calibrator.calibrated) {
    state.analyzer.calibrator = calibrator;
  }
  state.freeze = new FreezeDetector();
  state.frameTimes = [];
}

/** Like the service's "unknown" level: no frames means no verdict, not "attentive". */
function showStalled() {
  state.stalled = true;
  document.body.dataset.level = "idle";
  setText("state-heading", "No video");
  const item = document.createElement("li");
  item.textContent = "The camera stopped sending pictures";
  const list = byId("state-reasons");
  list.replaceChildren(item);
  list.dataset.key = "stalled";
  setText("stats", "0 fps");
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  setAlarm("normal");
}

function sampleFrame(width, height) {
  sampler.width = Math.ceil(width / FREEZE_STRIDE);
  sampler.height = Math.ceil(height / FREEZE_STRIDE);
  samplerContext.imageSmoothingEnabled = false;
  samplerContext.drawImage(video, 0, 0, sampler.width, sampler.height);
  const pixels = samplerContext.getImageData(0, 0, sampler.width, sampler.height).data;
  const green = new Uint8Array(pixels.length / 4);
  for (let index = 0; index < green.length; index += 1) {
    green[index] = pixels[index * 4 + 1];
  }
  return green;
}

function analyse() {
  // MediaPipe needs strictly increasing timestamps.
  const now = Math.max(performance.now(), state.lastTimestamp + 1);
  state.lastTimestamp = now;
  const width = video.videoWidth;
  const height = video.videoHeight;

  const face = state.landmarker.detectForVideo(video, now).faceLandmarks?.[0];
  // MediaPipe scales z by the image width, so all three axes end up in pixels.
  const points = face ? face.map((p) => ({ x: p.x * width, y: p.y * height, z: p.z * width })) : null;
  const metrics = measure(points, now / 1000, state.freeze.isUnchanged(sampleFrame(width, height)));
  const assessment = state.analyzer.update(metrics);

  draw(points, assessment.level, width, height);
  render(metrics, assessment);
  record(assessment);
  setAlarm(assessment.level);
  trackFps(now);
}

// ---- Drawing ----------------------------------------------------------------------------

function draw(points, level, width, height) {
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
    // The stage takes the video's exact shape, so the overlay lines up with no letterboxing.
    stage.style.setProperty("--aspect", String(width / height));
  }
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (!points) {
    return;
  }
  const colour = COLOURS[level];
  const radius = Math.max(2, width / 260);
  context.fillStyle = colour;
  for (const index of [...RIGHT_EYE, ...LEFT_EYE, ...MOUTH_CORNERS, ...INNER_LIPS]) {
    context.beginPath();
    context.arc(points[index].x, points[index].y, radius, 0, Math.PI * 2);
    context.fill();
  }
  const [left, right] = EYE_OUTER_CORNERS.map((index) => points[index]);
  context.strokeStyle = colour;
  context.lineWidth = radius;
  context.beginPath();
  context.moveTo(left.x, left.y);
  context.lineTo(right.x, right.y);
  context.stroke();
}

// ---- Panels -----------------------------------------------------------------------------

function setText(id, text) {
  const element = byId(id);
  if (element.textContent !== text) {
    element.textContent = text;
  }
}

const fixed = (value, digits) => (value === undefined || value === null ? "—" : value.toFixed(digits));
const signed = (value) => (value >= 0 ? "+" : "") + value.toFixed(0);

function render(metrics, assessment) {
  const frozen = assessment.reasons.includes("camera_frozen");
  document.body.dataset.level = assessment.level;
  setText("state-heading", HEADLINES[assessment.level]);
  byId("calibration").hidden = assessment.calibrated || !metrics.faceDetected;

  const chips = assessment.reasons.map((reason) => {
    const item = document.createElement("li");
    item.textContent = REASON_LABELS[reason] ?? reason;
    return item;
  });
  const reasonsKey = assessment.reasons.join();
  const list = byId("state-reasons");
  if (list.dataset.key !== reasonsKey) {
    list.replaceChildren(...chips);
    list.dataset.key = reasonsKey;
  }

  setText("metric-perclos", assessment.perclos === null ? "—" : `${(assessment.perclos * 100).toFixed(0)}%`);
  setText("metric-closure", `${assessment.eyeClosureS.toFixed(1)} s`);
  setText("metric-yawns", String(assessment.yawnsInWindow));
  setText("metric-ear", frozen || !metrics.faceDetected ? "—" : `${fixed(metrics.earLeft, 2)} / ${fixed(metrics.earRight, 2)}`);
  setText("metric-mar", frozen ? "—" : fixed(metrics.mar, 2));
  const pose = frozen ? null : metrics.pose;
  setText("metric-pose", pose ? `${signed(pose.pitch)} / ${signed(pose.yaw)} / ${signed(pose.roll)}` : "—");
}

function record(assessment) {
  const key = `${assessment.level}|${[...assessment.reasons].sort().join()}`;
  if (key === state.lastAlertKey) {
    return;
  }
  const first = state.lastAlertKey === "";
  state.lastAlertKey = key;
  if (first && assessment.level === "normal") {
    return;
  }
  state.history.unshift({ time: new Date(), level: assessment.level, reasons: assessment.reasons });
  state.history.length = Math.min(state.history.length, HISTORY_LIMIT);
  renderHistory();
}

function renderHistory() {
  byId("log-empty").hidden = state.history.length > 0;
  const rows = state.history.map((entry) => {
    const row = document.createElement("li");
    const time = document.createElement("time");
    time.dateTime = entry.time.toISOString();
    time.textContent = entry.time.toLocaleTimeString();
    const level = document.createElement("span");
    level.className = `level level-${entry.level}`;
    level.textContent = entry.level;
    row.append(time, level);
    if (entry.reasons.length) {
      const detail = document.createElement("span");
      detail.className = "detail";
      detail.textContent = entry.reasons.map((reason) => REASON_LABELS[reason] ?? reason).join(", ");
      row.append(detail);
    }
    return row;
  });
  byId("log-list").replaceChildren(...rows);
}

function trackFps(now) {
  state.frameTimes.push(now);
  if (state.frameTimes.length > FPS_WINDOW) {
    state.frameTimes.shift();
  }
  const span = state.frameTimes.at(-1) - state.frameTimes[0];
  const fps = span > 0 ? ((state.frameTimes.length - 1) * 1000) / span : 0;
  setText("stats", `${fps.toFixed(0)} fps`);
}

// ---- Alarm ------------------------------------------------------------------------------

function enableAudio() {
  state.audio ??= new AudioContext();
  state.audio.resume();
}

function beep(frequency, milliseconds) {
  if (!state.audio || !byId("sound").checked) {
    return;
  }
  const oscillator = state.audio.createOscillator();
  const gain = state.audio.createGain();
  oscillator.frequency.value = frequency;
  gain.gain.value = 0.15;
  oscillator.connect(gain).connect(state.audio.destination);
  oscillator.start();
  oscillator.stop(state.audio.currentTime + milliseconds / 1000);
}

function setAlarm(level) {
  if (level === state.alarmLevel) {
    return;
  }
  state.alarmLevel = level;
  clearInterval(state.alarmTimer);
  state.alarmTimer = null;
  const tone = TONES[level];
  if (tone) {
    const [frequency, length, gap] = tone;
    beep(frequency, length);
    state.alarmTimer = setInterval(() => beep(frequency, length), gap);
  }
}

// ---- Wiring -----------------------------------------------------------------------------

byId("start-camera").addEventListener("click", () => startCamera());
byId("video-file").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    startFile(file);
  }
  event.target.value = "";
});
byId("camera-select").addEventListener("change", (event) => startCamera(event.target.value));
byId("sound").addEventListener("change", (event) => {
  if (event.target.checked) {
    enableAudio();
  }
});
byId("stop").addEventListener("click", () => {
  stop();
  document.body.dataset.level = "idle";
  byId("monitor").hidden = true;
  byId("intro").hidden = false;
});

// Time spent in a background tab is not time watching the driver, so start the rules afresh.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.running) {
    restartRules();
  }
});
