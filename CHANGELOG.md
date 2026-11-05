# Changelog

## 1.0.0

A rebuild of the original college project into an installable, tested service.

### Added

- Time-based fatigue rules: prolonged closure, PERCLOS, yawn rate, and sustained head pose.
- Pitch and yaw head pose alongside roll, with per-driver calibration.
- A driver-not-visible alert. Losing the face used to freeze the video with no warning.
- Operator dashboard, and a versioned REST API for status and alert history.
- Health and readiness probes, Prometheus metrics, and optional JSON logs.
- API key authentication, a safe default bind, and a guard against unauthenticated network exposure.
- Non-blocking audio alarm, and automatic camera reconnection.
- `dss` command with `serve`, `monitor`, `extract` and `train`.
- Tests, type checking, linting, CI, Docker image and documentation.

### Fixed

- The API could not serve video: detectors returned a tuple that was passed to the JPEG encoder.
- `main.py` crashed on the first frame, unpacking five values from a function returning three.
- A frame with no face crashed the fatigue loop, returning two values where three were unpacked.
- The classifier crashed on any alert driver, because EAR and MAR were `None` below the threshold.
- `requirements.txt` could not be installed: `mediapipe` and `protobuf` pins conflicted, and
  `scikit-learn` and `playsound` were missing.
- EAR and MAR scaled with the camera's aspect ratio. They are now computed in pixels.
- A single blink, or a wink, counted as eye closure.
- Two video viewers competed for frames on a shared queue.
- The API's front page linked only to endpoints that did not exist.
- CORS was open to every origin, and the debug server was used to run the API.

### Changed

- **Breaking:** the package moved to `src/dss`. Use `dss serve` and `dss monitor` in place of
  `src/api/api.py` and `src/main.py`.
- **Breaking:** MAR is now inner-lip gap over mouth width, so it rises with a yawn. It was previously
  width over outer-lip height, which fell. Thresholds and old datasets, including `top_20.csv`, do
  not carry over.
- **Breaking:** the default port is 8000, and CORS is closed by default. A frontend on another origin
  must be listed in `DSS_CORS_ORIGINS`.
- The classifier now uses head pose as well as EAR and MAR, is saved rather than retrained at every
  start, and holds out whole people for testing when subject folders are available.
- `/video_feed/yawn` and `/video_feed/eye_closure` still work. `head_tilt`, `annotated` and
  `original` are new views.

### Removed

- The `dlib` experiment, and the unused TensorFlow imports. TensorFlow, Keras and `dlib` are no
  longer installed at all. `jax` remains, as a dependency of MediaPipe.
