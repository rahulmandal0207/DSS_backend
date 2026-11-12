// Detection logic for the browser demo, ported from the Python package in src/dss.
//
// Keep it in step with dss/vision/geometry.py, dss/vision/head_pose.py, dss/vision/freeze.py,
// dss/detection/temporal.py and the defaults in dss/config.py. tests/web/fatigue.test.mjs replays
// a scenario exported from the Python code and fails on any difference.
//
// Landmarks are objects with x, y and z in pixels, indexed like MediaPipe's 478-point face mesh.

export const THRESHOLDS = Object.freeze({
  earClosed: 0.2,
  marYawn: 0.6,
  eyeClosureCriticalS: 1.5,
  perclosWindowS: 60,
  perclosWarning: 0.15,
  perclosMinCoverage: 0.5,
  yawnMinDurationS: 1.5,
  yawnWindowS: 300,
  yawnWarningCount: 3,
  poseCalibrationS: 5,
  headPitchLimitDeg: 20,
  headYawLimitDeg: 30,
  headRollLimitDeg: 25,
  headPoseHoldS: 2,
  faceLostHoldS: 2,
  cameraFrozenHoldS: 3,
  cameraMinChange: 0.05,
});

export const LEVELS = ["normal", "warning", "critical"];

export const REASON_LEVELS = Object.freeze({
  prolonged_eye_closure: "critical",
  head_nodding: "critical",
  high_perclos: "warning",
  frequent_yawning: "warning",
  head_tilted: "warning",
  looking_away: "warning",
  driver_not_visible: "warning",
  camera_frozen: "warning",
});

// ---- Geometry -------------------------------------------------------------------------------

export const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
export const LEFT_EYE = [362, 385, 387, 263, 373, 380];
export const MOUTH_CORNERS = [61, 291];
export const INNER_LIPS = [13, 14];
export const FOREHEAD = 10;
export const CHIN = 152;
export const EYE_OUTER_CORNERS = [33, 263];

const EPSILON = 1e-6;

function distance(points, a, b) {
  const dx = points[a].x - points[b].x;
  const dy = points[a].y - points[b].y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Eye height over width. Roughly 0.3 when open, falling towards 0 as the eye closes. */
export function eyeAspectRatio(points, eye) {
  const [p1, p2, p3, p4, p5, p6] = eye;
  const height = distance(points, p2, p6) + distance(points, p3, p5);
  return height / Math.max(2 * distance(points, p1, p4), EPSILON);
}

/** Inner-lip gap over mouth width. Near 0 when closed, and rises as the mouth opens. */
export function mouthAspectRatio(points) {
  return distance(points, ...INNER_LIPS) / Math.max(distance(points, ...MOUTH_CORNERS), EPSILON);
}

// ---- Head pose ------------------------------------------------------------------------------

function unit(vector) {
  const length = Math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2);
  return length < EPSILON ? null : vector.map((component) => component / length);
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

const degrees = (radians) => (radians * 180) / Math.PI;

/**
 * Head orientation in degrees, read from the face's own axes. Positive pitch is chin-down.
 * Returns null if the landmarks are degenerate.
 */
export function estimateHeadPose(points) {
  const left = points[EYE_OUTER_CORNERS[0]];
  const right = points[EYE_OUTER_CORNERS[1]];
  const top = points[FOREHEAD];
  const bottom = points[CHIN];
  const across = unit([right.x - left.x, right.y - left.y, right.z - left.z]);
  const down = unit([bottom.x - top.x, bottom.y - top.y, bottom.z - top.z]);
  if (!across || !down) {
    return null;
  }
  const forward = unit(cross(across, down));
  if (!forward) {
    return null;
  }
  // Columns of the rotation matrix are across, the rebuilt down axis, and forward.
  const rebuiltDown = cross(forward, across);
  return {
    pitch: degrees(Math.atan2(rebuiltDown[2], forward[2])),
    yaw: degrees(Math.asin(Math.min(1, Math.max(-1, -across[2])))),
    roll: degrees(Math.atan2(across[1], across[0])),
  };
}

/** Measure one frame on its own. Pass null landmarks when no face was found. */
export function measure(points, timestamp, frameUnchanged = false) {
  if (!points) {
    return { timestamp, faceDetected: false, frameUnchanged };
  }
  return {
    timestamp,
    faceDetected: true,
    earLeft: eyeAspectRatio(points, LEFT_EYE),
    earRight: eyeAspectRatio(points, RIGHT_EYE),
    mar: mouthAspectRatio(points),
    pose: estimateHeadPose(points),
    frameUnchanged,
  };
}

// ---- Frozen picture -------------------------------------------------------------------------

/**
 * Reports whether a sample of pixels shows the same picture as before. Pass pixels sampled
 * sparsely, not averaged, because averaging would smooth away the sensor noise this relies on.
 */
export class FreezeDetector {
  constructor(minChange = THRESHOLDS.cameraMinChange) {
    this.minChange = minChange;
    this.reference = null;
  }

  isUnchanged(sample) {
    const reference = this.reference;
    if (!reference || reference.length !== sample.length) {
      this.reference = Array.from(sample);
      return false;
    }
    let total = 0;
    for (let index = 0; index < sample.length; index += 1) {
      total += Math.abs(sample[index] - reference[index]);
    }
    if (total / sample.length < this.minChange) {
      return true;
    }
    this.reference = Array.from(sample);
    return false;
  }
}

// ---- Rules over time ------------------------------------------------------------------------

export class Hold {
  constructor() {
    this.since = null;
  }

  /** Record the condition and return how many seconds it has been continuously true. */
  update(active, now) {
    if (!active) {
      this.since = null;
      return 0;
    }
    if (this.since === null) {
      this.since = now;
    }
    return now - this.since;
  }
}

export class Perclos {
  constructor(windowS, minCoverage) {
    this.windowS = windowS;
    this.minCoverage = minCoverage;
    this.samples = [];
    this.last = null;
  }

  /** Record whether the eyes are closed, or null when none are visible. */
  update(closed, now) {
    if (this.last !== null) {
      this.samples.push({ started: this.last.now, duration: now - this.last.now, closed: this.last.closed });
    }
    this.last = closed === null ? null : { now, closed };

    const cutoff = now - this.windowS;
    while (this.samples.length && this.samples[0].started < cutoff) {
      this.samples.shift();
    }

    let observed = 0;
    for (const sample of this.samples) {
      observed += sample.duration;
    }
    if (observed < this.windowS * this.minCoverage) {
      return null;
    }
    let shut = 0;
    for (const sample of this.samples) {
      if (sample.closed) {
        shut += sample.duration;
      }
    }
    return shut / observed;
  }
}

export class EventCounter {
  constructor(windowS) {
    this.windowS = windowS;
    this.events = [];
  }

  add(now) {
    this.events.push(now);
  }

  count(now) {
    while (this.events.length && this.events[0] < now - this.windowS) {
      this.events.shift();
    }
    return this.events.length;
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export class PoseCalibrator {
  constructor(durationS) {
    this.durationS = durationS;
    this.started = null;
    this.samples = [];
    this.neutral = durationS > 0 ? null : { pitch: 0, yaw: 0, roll: 0 };
  }

  get calibrated() {
    return this.neutral !== null;
  }

  /** The pose relative to this driver's neutral position, or null while still calibrating. */
  adjust(pose, now) {
    if (this.neutral === null) {
      this.started = this.started === null ? now : this.started;
      this.samples.push(pose);
      if (now - this.started < this.durationS) {
        return null;
      }
      this.neutral = {
        pitch: median(this.samples.map((sample) => sample.pitch)),
        yaw: median(this.samples.map((sample) => sample.yaw)),
        roll: median(this.samples.map((sample) => sample.roll)),
      };
      this.samples = [];
    }
    return {
      pitch: pose.pitch - this.neutral.pitch,
      yaw: pose.yaw - this.neutral.yaw,
      roll: pose.roll - this.neutral.roll,
    };
  }
}

export function highestLevel(reasons) {
  return reasons.reduce(
    (level, reason) => (LEVELS.indexOf(REASON_LEVELS[reason]) > LEVELS.indexOf(level) ? REASON_LEVELS[reason] : level),
    "normal",
  );
}

/** Turns per-frame measurements into an alert level. Use one per monitored driver. */
export class TemporalAnalyzer {
  constructor(thresholds = THRESHOLDS) {
    this.t = thresholds;
    this.perclos = new Perclos(thresholds.perclosWindowS, thresholds.perclosMinCoverage);
    this.yawns = new EventCounter(thresholds.yawnWindowS);
    this.calibrator = new PoseCalibrator(thresholds.poseCalibrationS);
    this.holds = {
      eyesClosed: new Hold(),
      mouthOpen: new Hold(),
      headDown: new Hold(),
      headTilted: new Hold(),
      lookingAway: new Hold(),
      faceLost: new Hold(),
      cameraFrozen: new Hold(),
    };
    this.yawnCounted = false;
  }

  update(metrics) {
    const now = metrics.timestamp;
    const frozen = this.holds.cameraFrozen.update(Boolean(metrics.frameUnchanged), now) >= this.t.cameraFrozenHoldS;
    if (frozen) {
      // A stale picture says nothing about the driver, so no other rule may count it.
      metrics = { timestamp: now, faceDetected: false };
    }

    const reasons = [];
    const ear =
      metrics.earLeft === undefined || metrics.earRight === undefined
        ? null
        : (metrics.earLeft + metrics.earRight) / 2;
    const mar = metrics.mar === undefined ? null : metrics.mar;

    // Averaging both eyes means a wink, or one eye hidden by a turned head, is not a closure.
    const closed = ear === null ? null : ear < this.t.earClosed;
    const closureS = this.holds.eyesClosed.update(Boolean(closed), now);
    const perclos = this.perclos.update(closed, now);
    if (closureS >= this.t.eyeClosureCriticalS) {
      reasons.push("prolonged_eye_closure");
    }
    if (perclos !== null && perclos >= this.t.perclosWarning) {
      reasons.push("high_perclos");
    }

    this.updateYawns(mar, now);
    const yawns = this.yawns.count(now);
    if (yawns >= this.t.yawnWarningCount) {
      reasons.push("frequent_yawning");
    }

    reasons.push(...this.assessHead(metrics.pose ?? null, now));

    if (this.holds.faceLost.update(!metrics.faceDetected && !frozen, now) >= this.t.faceLostHoldS) {
      reasons.push("driver_not_visible");
    }
    if (frozen) {
      reasons.push("camera_frozen");
    }

    return {
      level: highestLevel(reasons),
      reasons,
      perclos,
      eyeClosureS: closureS,
      yawnsInWindow: yawns,
      calibrated: this.calibrator.calibrated,
    };
  }

  updateYawns(mar, now) {
    const wideOpen = mar !== null && mar >= this.t.marYawn;
    const openS = this.holds.mouthOpen.update(wideOpen, now);
    if (!wideOpen) {
      this.yawnCounted = false;
    } else if (openS >= this.t.yawnMinDurationS && !this.yawnCounted) {
      this.yawnCounted = true;
      this.yawns.add(now);
    }
  }

  assessHead(pose, now) {
    const relative = pose === null ? null : this.calibrator.adjust(pose, now);
    const pitch = relative ? relative.pitch : 0;
    const yaw = relative ? relative.yaw : 0;
    const roll = relative ? relative.roll : 0;
    const rules = [
      [this.holds.headDown, pitch >= this.t.headPitchLimitDeg, "head_nodding"],
      [this.holds.headTilted, Math.abs(roll) >= this.t.headRollLimitDeg, "head_tilted"],
      [this.holds.lookingAway, Math.abs(yaw) >= this.t.headYawLimitDeg, "looking_away"],
    ];
    return rules
      .filter(([hold, active]) => hold.update(active, now) >= this.t.headPoseHoldS)
      .map(([, , reason]) => reason);
  }
}
