# Operations guide

## Configuration

Settings come from `DSS_*` environment variables, or a `.env` file in the working directory.
Environment variables win when both set the same name. Names are not case-sensitive.

DSS checks the whole configuration before starting, and lists every invalid value at once. A
`DSS_*` name that matches no setting is logged as a warning, so a typo does not silently fall
back to the default.

### Service

| Variable | Default | Meaning |
|---|---|---|
| `DSS_VIDEO_SOURCE` | `0` | Camera index, file path or stream URL |
| `DSS_VIDEO_LOOP` | `false` | Restart a video file when it ends |
| `DSS_CAMERA_BACKEND` | `auto` | `auto`, `dshow`, `msmf` or `v4l2`. See [Choosing a camera](#choosing-a-camera) |
| `DSS_CAMERA_RETRY_S` | `2.0` | Wait between reconnection attempts |
| `DSS_HOST` | `127.0.0.1` | Bind address |
| `DSS_PORT` | `8000` | Port |
| `DSS_SERVER_THREADS` | `8` | HTTP worker threads; two are kept back from video |
| `DSS_API_KEY` | unset | Required to bind beyond localhost |
| `DSS_ALLOW_INSECURE_BIND` | `false` | Override that requirement, for a trusted network only |
| `DSS_CORS_ORIGINS` | empty | Comma-separated allowed browser origins |
| `DSS_STREAM_ENABLED` | `true` | Set `false` to serve measurements with no video |
| `DSS_JPEG_QUALITY` | `80` | Stream quality, 10–100 |
| `DSS_SOUND_ENABLED` | `false` | Beep on the local machine during alerts |
| `DSS_ALERT_HISTORY_SIZE` | `200` | Alert changes kept in memory |
| `DSS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `DSS_LOG_FORMAT` | `text` | `json` for log shippers |

### Classifier

| Variable | Default | Meaning |
|---|---|---|
| `DSS_MODEL_PATH` | unset | Model from `dss train`. Only load models you trained |
| `DSS_MODEL_DROWSY_PROBABILITY` | `0.7` | Smoothed score that raises a warning |
| `DSS_MODEL_SMOOTHING_S` | `3.0` | Averaging window |

### Detection thresholds

| Variable | Default | Meaning |
|---|---|---|
| `DSS_EAR_CLOSED` | `0.20` | Mean eye ratio below this counts as closed |
| `DSS_EYE_CLOSURE_CRITICAL_S` | `1.5` | Continuous closure that raises a critical alert |
| `DSS_PERCLOS_WINDOW_S` | `60` | Rolling PERCLOS window |
| `DSS_PERCLOS_WARNING` | `0.15` | Closed fraction that raises a warning |
| `DSS_PERCLOS_MIN_COVERAGE` | `0.5` | Share of the window needed before PERCLOS is reported |
| `DSS_MAR_YAWN` | `0.60` | Mouth ratio at or above this counts as wide open |
| `DSS_YAWN_MIN_DURATION_S` | `1.5` | Time open before it counts as a yawn, so talking does not |
| `DSS_YAWN_WINDOW_S` | `300` | Rolling yawn window |
| `DSS_YAWN_WARNING_COUNT` | `3` | Yawns in the window that raise a warning |
| `DSS_POSE_CALIBRATION_S` | `5` | Time spent learning the neutral head position |
| `DSS_HEAD_PITCH_LIMIT_DEG` | `20` | Chin-down angle from neutral |
| `DSS_HEAD_YAW_LIMIT_DEG` | `30` | Turn from neutral |
| `DSS_HEAD_ROLL_LIMIT_DEG` | `25` | Tilt from neutral |
| `DSS_HEAD_POSE_HOLD_S` | `2` | How long a pose must be held |
| `DSS_FACE_LOST_HOLD_S` | `2` | Time without a face before alerting |
| `DSS_CAMERA_FROZEN_HOLD_S` | `3` | Time with an unchanged picture before alerting |
| `DSS_CAMERA_MIN_CHANGE` | `0.05` | Mean pixel change, out of 255, below which frames count as identical |

### Choosing a camera

Virtual cameras from phone-link or conferencing software count as devices, and on Windows the
same camera can have a different index under each backend. If the dashboard shows a still picture
or finds no face, try the DirectShow backend, then the next index:

```text
DSS_CAMERA_BACKEND=dshow
DSS_VIDEO_SOURCE=1
```

### Tuning

Run for a while with the dashboard open and watch the raw EAR and MAR tiles for the people and
camera you actually use. EAR varies between people: if an alert driver hovers near `0.20`, lower
`DSS_EAR_CLOSED` rather than the time limits.

## Deployment

### Exposing it on a network

The video is a live image of a person, and the API key travels with every request. Anything beyond
localhost must therefore sit behind a TLS-terminating reverse proxy, with `DSS_API_KEY` set to a
long random value. `DSS_ALLOW_INSECURE_BIND` exists for isolated networks and turns the guard off,
so avoid it elsewhere.

### Docker

```bash
DSS_API_KEY=change-me docker compose up --build
```

The image runs as a non-root user. Compose publishes the port on localhost, drops all Linux
capabilities and blocks privilege escalation. Its health check uses `/api/v1/health`, which stays
up during a camera outage, so a lost camera is reported rather than treated as a crash.

Webcam passthrough needs a Linux host with `/dev/video0`. On Docker Desktop for Windows or macOS,
remove `devices` and `group_add` from `docker-compose.yml` and set `DSS_VIDEO_SOURCE` to a video
URL, or run DSS directly on the host.

### Monitoring

Scrape `/metrics`. Useful alerts:

- `dss_camera_connected == 0` for more than a minute.
- `dss_camera_frozen == 1`, which means the driver is not being monitored.
- `rate(dss_frame_errors_total[5m]) > 0`.
- A falling `rate(dss_frames_processed_total[1m])`, which means detection is not keeping up.

Probe `/api/v1/health` for liveness and `/api/v1/ready` for readiness. Restarting on a failed
readiness probe would not bring a camera back.

## Privacy

DSS processes images of identifiable people, which is regulated in many places, so check what
applies before using it on anyone but yourself. By design:

- Frames are held in memory only, and never written to disk or logged.
- Alert history holds levels and reasons, with no images, and is lost on restart.
- `DSS_STREAM_ENABLED=false` removes the video from the HTTP surface entirely.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Refusing to serve a driver video feed` at startup | Binding beyond localhost with no `DSS_API_KEY` |
| `pip` cannot find `mediapipe==0.10.21` | Python is newer than 3.12 |
| `An Application Control policy has blocked this file` | Windows Smart App Control is refusing an unsigned compiled library. DSS avoids the ones known to be blocked; report any other |
| `/ready` returns 503 | The camera is missing, or held by another application |
| `Camera image frozen` alert | The source is a virtual camera or a stalled backend. See [Choosing a camera](#choosing-a-camera) |
| `Camera image frozen` on a video file | The clip contains a still image, which is treated the same way |
| Level stays `unknown` | No frames are arriving: check `DSS_VIDEO_SOURCE` and the logs |
| False head alerts just after start | The driver moved during calibration: restart, or lengthen `DSS_POSE_CALIBRATION_S` |
| No sound | `DSS_SOUND_ENABLED` is off, or the logs show no usable audio device |
| Frame rate is low | Reduce the camera resolution; rendering and encoding video also costs CPU |
