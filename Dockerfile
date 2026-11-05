# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps .


FROM python:3.11-slim-bookworm

# OpenCV and MediaPipe link against these at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 dss

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    DSS_HOST=0.0.0.0

USER dss
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
    CMD ["python", "-c", "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"]

ENTRYPOINT ["dss"]
CMD ["serve"]
