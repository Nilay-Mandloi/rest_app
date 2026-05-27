FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim
# libgomp1: OpenMP runtime required by LightGBM's native shared library.
# libglib2.0-0: required by XGBoost on slim images.
# curl: used to download model artifacts via presigned URL at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# ARGs declared AFTER heavy layers so pip install stays cached across deploys.
# CI generates 1-hour presigned URLs — no AWS credentials are stored in the image.
# To swap to GCS: change the CI presign step; this RUN is identical.
ARG MODEL_URL
ARG MANIFEST_URL
RUN if [ -n "${MODEL_URL}" ] && [ -n "${MANIFEST_URL}" ]; then \
      curl -fsSL "${MODEL_URL}" -o /app/model.pkl && \
      curl -fsSL "${MANIFEST_URL}" -o /app/manifest.json; \
    fi

RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "rest_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
