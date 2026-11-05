# Security policy

## Reporting a vulnerability

Please report security problems privately, using
[GitHub's private vulnerability reporting](https://github.com/rahulmandal0207/DSS_backend/security/advisories/new),
rather than in a public issue. Include the steps to reproduce it and the version affected.

## Supported versions

Only the latest release receives fixes.

## Security model

DSS handles live video of an identifiable person, so its defaults assume that video is sensitive.

- It listens on localhost unless configured otherwise, and refuses to start on a network address
  without an API key.
- API keys are compared in constant time.
- CORS is closed unless origins are listed. Responses set `Content-Security-Policy`, `nosniff`,
  frame-denial and `no-store` headers.
- Frames are never written to disk.
- The container runs as a non-root user, with all Linux capabilities dropped under Compose.

## What is left to the operator

- **TLS.** DSS speaks plain HTTP. Anything beyond localhost needs a TLS-terminating proxy, or the
  API key and video travel in the clear.
- **Key management.** There is one shared key, with no users, roles or rotation.
- **Model files.** `DSS_MODEL_PATH` loads a pickle, which can run code. Only load models you trained.
- **Rate limiting.** There is none, apart from the cap on concurrent video streams.
