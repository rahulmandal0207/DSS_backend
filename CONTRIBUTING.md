# Contributing

## Setup

Use Python 3.10, 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` elsewhere.

```bash
pip install -r requirements-dev.txt
pip install --no-deps -e .
pre-commit install
```

## Before opening a pull request

```bash
ruff format src tests
ruff check src tests
mypy
pytest --cov=dss
```

CI runs the same checks on Linux and Windows across every supported Python, and fails below 85%
coverage. Use `pytest -m "not mediapipe"` for a faster loop that skips the real model.

## Workflow

Branch from `main`, keep each pull request to one change, and use short imperative commit messages.

## Testing

- Regression tests come first: reproduce a bug in a failing test, then fix it.
- Fatigue logic is about time, so test it as a played-out scenario rather than one frame. See
  `tests/test_temporal.py`.
- Use `ScriptedDetector` and `synthetic_face` from `tests/conftest.py` rather than a webcam.
- Anything depending on what MediaPipe really outputs belongs in
  `tests/test_mediapipe_integration.py`.

## Changing dependencies

Edit `pyproject.toml`, then regenerate the lockfiles with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile pyproject.toml --universal --python-version 3.10 --no-emit-index-url -o requirements.txt
```

```bash
uv pip compile pyproject.toml --extra dev --universal --python-version 3.10 --no-emit-index-url -o requirements-dev.txt
```

## What must stay true

- A detector returns a complete result for every frame, including when no face is present.
- All measurements stay in pixels, and all durations in seconds.
- Nothing writes frames to disk.
- A new alert reason needs a level in `REASON_LEVELS`, and a label in `dashboard.js`.
