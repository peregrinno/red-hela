FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV RED_HELA_ROOT=/app
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY resources/vectors_q8.npy ./resources/vectors_q8.npy
COPY resources/vectors_f16.npy ./resources/vectors_f16.npy
COPY resources/labels.npy ./resources/labels.npy
COPY resources/centroids.npy ./resources/centroids.npy
COPY resources/cluster_indices.npy ./resources/cluster_indices.npy
COPY resources/cluster_offsets.npy ./resources/cluster_offsets.npy
COPY resources/tree.npz ./resources/tree.npz

EXPOSE 8000

ENTRYPOINT ["uvicorn", "red_hela.adapters.http.app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--workers", "1", "--log-level", "error", "--timeout-keep-alive", "30", "--limit-concurrency", "1000", "--no-access-log"]
