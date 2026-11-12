// The browser demo must make the same decisions as the Python service. The fixture is produced
// from the Python code by tests/test_web_parity.py, which also checks it is still current.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  FreezeDetector,
  LEFT_EYE,
  RIGHT_EYE,
  THRESHOLDS,
  TemporalAnalyzer,
  estimateHeadPose,
  eyeAspectRatio,
  measure,
  mouthAspectRatio,
} from "../../web/js/fatigue.js";

const fixture = JSON.parse(readFileSync(new URL("./parity_fixture.json", import.meta.url), "utf8"));
const TOLERANCE = 1e-8;

const toPoints = (raw) =>
  Object.fromEntries(Object.entries(raw).map(([index, [x, y, z]]) => [index, { x, y, z }]));

const close = (actual, expected, label) =>
  assert.ok(Math.abs(actual - expected) <= TOLERANCE, `${label}: ${actual} != ${expected}`);

test("thresholds match the Python defaults", () => {
  const camel = (name) => name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
  const expected = Object.fromEntries(Object.entries(fixture.thresholds).map(([k, v]) => [camel(k), v]));
  assert.deepEqual({ ...THRESHOLDS }, expected);
});

test("geometry matches the Python implementation", () => {
  fixture.geometry.forEach((item, index) => {
    const points = toPoints(item.points);
    close(eyeAspectRatio(points, LEFT_EYE), item.ear_left, `case ${index} left eye`);
    close(eyeAspectRatio(points, RIGHT_EYE), item.ear_right, `case ${index} right eye`);
    close(mouthAspectRatio(points), item.mar, `case ${index} mouth`);
    const pose = estimateHeadPose(points);
    if (item.pose === null) {
      assert.equal(pose, null, `case ${index} pose`);
    } else {
      close(pose.pitch, item.pose[0], `case ${index} pitch`);
      close(pose.yaw, item.pose[1], `case ${index} yaw`);
      close(pose.roll, item.pose[2], `case ${index} roll`);
    }
  });
});

test("every frame of the scenario gets the same decision as Python", () => {
  const analyzer = new TemporalAnalyzer();
  const { inputs, expected } = fixture.scenario;
  inputs.forEach(([timestamp, face, earLeft, earRight, mar, pose, unchanged], index) => {
    const metrics = face
      ? {
          timestamp,
          faceDetected: true,
          earLeft,
          earRight,
          mar,
          pose: pose && { pitch: pose[0], yaw: pose[1], roll: pose[2] },
          frameUnchanged: Boolean(unchanged),
        }
      : { timestamp, faceDetected: false, frameUnchanged: Boolean(unchanged) };
    const result = analyzer.update(metrics);
    const [level, reasons, perclos, closure, yawns, calibrated] = expected[index];
    const at = `frame ${index} (t=${timestamp}s)`;

    assert.equal(result.level, level, `${at} level`);
    assert.deepEqual(result.reasons, reasons, `${at} reasons`);
    if (perclos === null) {
      assert.equal(result.perclos, null, `${at} perclos`);
    } else {
      close(result.perclos, perclos, `${at} perclos`);
    }
    close(result.eyeClosureS, closure, `${at} eye closure`);
    assert.equal(result.yawnsInWindow, yawns, `${at} yawns`);
    assert.equal(result.calibrated, Boolean(calibrated), `${at} calibrated`);
  });
});

test("no face gives a complete result, not a crash", () => {
  assert.deepEqual(measure(null, 1.5), { timestamp: 1.5, faceDetected: false, frameUnchanged: false });
});

test("a repeated picture is frozen, and sensor noise is not", () => {
  const detector = new FreezeDetector();
  const still = Array.from({ length: 4800 }, (_, i) => (i * 37) % 256);
  assert.equal(detector.isUnchanged(still), false); // nothing to compare with yet
  assert.equal(detector.isUnchanged([...still]), true);

  const noisy = still.map((value, i) => value + (i % 3) - 1);
  assert.equal(detector.isUnchanged(noisy), false);
});

test("a slow drift is not mistaken for a freeze", () => {
  const detector = new FreezeDetector(0.5);
  const frame = new Array(1000).fill(100);
  detector.isUnchanged(frame);
  const step = (amount) => frame.map((value, i) => (i < 300 ? value + amount : value));
  assert.equal(detector.isUnchanged(step(1)), true);
  assert.equal(detector.isUnchanged(step(2)), false);
});
