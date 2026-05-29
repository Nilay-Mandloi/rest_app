ARG PYTHON_VERSION=3.11.9

# ─── Build stage: produce a wheel of rest_app ────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .


# ─── Runtime stage ───────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim

# OCI image labels — supersede the deprecated `LABEL maintainer=`.
LABEL org.opencontainers.image.title="rest_app" \
      org.opencontainers.image.description="Lean FastAPI inference service. One model baked at build time via MODEL_URL/MANIFEST_URL." \
      org.opencontainers.image.source="https://github.com/Nilay-Mandloi/rest_app" \
      org.opencontainers.image.licenses="proprietary"

# Native libs:
#   libgomp1     — OpenMP runtime required by LightGBM's native shared library.
#   libglib2.0-0 — required by XGBoost on slim images.
#   curl         — downloads model artifacts via presigned URL at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

ARG WITH_MODEL=true
ARG MODEL_URL=""
ARG MANIFEST_URL=""
ARG MODEL_SHA256=""
RUN set -eu; \
    if [ "${WITH_MODEL}" = "true" ]; then \
        if [ -z "${MODEL_URL}" ] || [ -z "${MANIFEST_URL}" ]; then \
            echo "ERROR: MODEL_URL and MANIFEST_URL are required when WITH_MODEL=true." >&2; \
            echo "       pass --build-arg MODEL_URL=... --build-arg MANIFEST_URL=... or set WITH_MODEL=false" >&2; \
            exit 1; \
        fi; \
        curl -fsSL "${MODEL_URL}"    -o /app/model.pkl; \
        curl -fsSL "${MANIFEST_URL}" -o /app/manifest.json; \
        if [ -n "${MODEL_SHA256}" ]; then \
            echo "${MODEL_SHA256}  /app/model.pkl" | sha256sum -c - || \
                { echo "ERROR: model.pkl sha256 mismatch — refusing build." >&2; exit 1; }; \
        fi; \
    else \
        echo "WITH_MODEL=false: building an empty image. /predict will return 503 until a model is mounted in."; \
    fi

RUN chown -R appuser:appuser /app

ENV MODEL_PKL_PATH=/app/model.pkl \
    MANIFEST_PATH=/app/manifest.json

USER appuser
EXPOSE 8000
STOPSIGNAL SIGTERM

# In-image health gate. The deploy workflow ALSO polls /ready externally, but
# this lets `docker ps` / orchestrators report the container's real state.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "rest_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
