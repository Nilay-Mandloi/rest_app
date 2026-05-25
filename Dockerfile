FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim
# libgomp1: OpenMP runtime required by LightGBM's native shared library.
# libglib2.0-0: required by XGBoost on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
# JSON Schemas + static admin UI ship inside the rest_app wheel via the
# package-data declaration in pyproject.toml (rest_app/schemas/*.json,
# rest_app/static/*). No separate COPY required.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
CMD ["uvicorn", "rest_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
