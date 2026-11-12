# Browser demo

The demo at <https://rahulmandal0207.github.io/DSS_backend/> lets anyone try DSS with their own
webcam, or with a video file, without installing anything.

## How it works

A server cannot see a visitor's webcam, and a server that could would receive video of their face.
The demo therefore runs everything in the browser:

1. The camera or video file plays in the page.
2. MediaPipe's [Face Landmarker for the web](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
   finds the same 478 face landmarks the Python service uses.
3. `web/js/fatigue.js` applies the DSS rules: eye and mouth ratios in pixels, head pose from the
   face's own axes, a frozen-picture check, and the time-based alerts.

The page is static files only, so it is hosted free on GitHub Pages and stays up whether or not
any other machine is running. The first visit downloads about 17 MB: MediaPipe's WebAssembly
runtime from jsDelivr, pinned to version 1.1.1, and the face model from Google.

## Keeping it identical to the service

The rules exist twice, in Python and in JavaScript, so a shared fixture keeps them in step.

- `tests/test_web_parity.py` builds `tests/web/parity_fixture.json` from the Python code: 80
  seconds of scripted driver behaviour that triggers every alert, plus geometry cases. It fails if
  the stored fixture no longer matches the Python rules.
- `tests/web/fatigue.test.mjs` replays the same fixture through `fatigue.js` and fails if any frame
  gets a different decision.

Both run in CI. After changing a rule on purpose, change it in both languages, then regenerate the
fixture:

```bash
PYTHONPATH=src:tests python tests/test_web_parity.py --write
```

On Windows, use `;` instead of `:` to separate the two paths.

## Differences from the service

- No REST API, metrics or video streams: the page itself is the dashboard.
- Alert history lasts only as long as the page is open.
- The optional classifier is not included.
- Returning from another tab, or from a camera that stopped sending pictures, restarts the
  time-based rules, because that gap was not time spent watching the driver. A finished
  head-position calibration is kept.
- If no new frame arrives for 3 seconds, the page shows "No video" instead of its last verdict,
  as the service reports `unknown`.

## Running it locally

Browsers only allow camera access from HTTPS pages or `localhost`, so serve the folder rather than
opening the file directly:

```bash
python -m http.server 8080 --directory web
```

Then open <http://localhost:8080>. Run the JavaScript tests with
`node --test tests/web/fatigue.test.mjs`.

## Publishing

`.github/workflows/pages.yml` publishes `web/` whenever it changes on `main`, after checking the
demo still matches the Python rules. Before the first run, open the repository's
**Settings > Pages** and set **Source** to **GitHub Actions**.
