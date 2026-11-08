# API reference

Base URL: `http://127.0.0.1:8000`. Errors are JSON: `{"error": "...", "message": "..."}`.

## Authentication

Authentication is off unless `DSS_API_KEY` is set. When it is, every endpoint except `/api/v1/health`
and `/api/v1/ready` needs the key, in any one of these forms:

| Form | Example |
|---|---|
| Header | `X-API-Key: <key>` |
| Bearer token | `Authorization: Bearer <key>` |
| HTTP Basic | Any username, with the key as the password |

Basic is what a browser uses for the dashboard. It prompts once, then sends the credentials with
every request, including the video streams.

## `GET /api/v1/status`

The driver's current state.

```json
{
  "level": "warning",
  "reasons": ["high_perclos"],
  "face_detected": true,
  "calibrated": true,
  "metrics": {
    "ear_left": 0.31,
    "ear_right": 0.32,
    "mar": 0.08,
    "head_pose": { "pitch": 4.2, "yaw": -1.5, "roll": 2.8 }
  },
  "fatigue": {
    "perclos": 0.18,
    "eye_closure_seconds": 0.0,
    "yawns_in_window": 1,
    "model_probability": null
  },
  "service": { "camera_connected": true, "camera_live": true, "healthy": true, "fps": 29.7 }
}
```

`level` is `normal`, `warning` or `critical`. If frames have stopped arriving, the body is only
`{"level": "unknown", "service": {...}}`: treat `unknown` as "do not trust this", not as "safe".

`perclos` is `null` until half of the 60 second window has been observed. `head_pose` values are
raw angles, before the driver's baseline is removed. `metrics` fields are `null` when no face is
visible.

| Reason | Meaning |
|---|---|
| `prolonged_eye_closure` | Eyes closed longer than the limit |
| `head_nodding` | Chin dropped and held |
| `high_perclos` | Eyes closed too much of the recent window |
| `frequent_yawning` | Too many yawns in the window |
| `looking_away` | Head turned and held |
| `head_tilted` | Head rolled and held |
| `driver_not_visible` | No face in view |
| `model_drowsy` | The optional classifier's smoothed score is over its limit |
| `camera_frozen` | Frames are arriving, but the picture has stopped changing |

While `camera_frozen` is active no other rule runs, because a stale picture says nothing about
the driver. Treat it as "the driver is not being monitored".

## `GET /api/v1/alerts?limit=50`

Alert changes, newest first. A long alert is one entry, and recovery is an entry with level
`normal`. `limit` is clamped to 1–200.

```json
{
  "alerts": [
    { "time": "2026-11-05T16:48:31+00:00", "level": "normal", "reasons": [] },
    { "time": "2026-11-05T16:48:28+00:00", "level": "warning", "reasons": ["driver_not_visible"] }
  ]
}
```

History is held in memory and is lost on restart.

## `GET /api/v1/streams/<view>`

Live `multipart/x-mixed-replace` MJPEG, usable directly as an `<img src>`.

| View | Shows |
|---|---|
| `original` | The camera image, unmodified |
| `annotated` | Eye, mouth and head markers with a status panel |
| `eye_closure` | Eye markers, EAR and PERCLOS |
| `yawn` | Mouth markers, MAR and yawn count |
| `head_tilt` | Eye line and head angles |

Returns `404` for an unknown view or when `DSS_STREAM_ENABLED=false`, and `503` with `Retry-After`
when all stream slots are taken.

The original `/video_feed/<view>` paths are aliases for these.

## `GET /api/v1/health`

Liveness. Always `200` while the process can answer. Use it to decide whether to restart the
service.

```json
{ "status": "ok", "version": "1.0.1" }
```

## `GET /api/v1/ready`

Readiness. `200` while a live picture is being analysed, otherwise `503`, with the failing check
named. Use it to decide whether the results can be trusted.

```json
{
  "status": "not_ready",
  "checks": { "monitor_running": true, "camera_connected": true, "camera_live": false }
}
```

## `GET /metrics`

Prometheus text format.

| Metric | Type | Meaning |
|---|---|---|
| `dss_frames_processed_total` | Counter | Frames analysed |
| `dss_frame_errors_total` | Counter | Frames dropped after an analysis error |
| `dss_frame_processing_seconds` | Histogram | Analysis time per frame |
| `dss_alerts_total{level}` | Counter | Changes into or between alert states |
| `dss_camera_connected` | Gauge | 1 while the source is connected |
| `dss_camera_frozen` | Gauge | 1 while the picture has stopped changing |
| `dss_alert_rank` | Gauge | 0 normal, 1 warning, 2 critical |
