# ==============================================================================
# Dockerfile — Iris Prediction API / Kafka Worker image
# ==============================================================================
# This single image is used by BOTH the `api` service and the `kafka-worker`
# service in docker-compose.yml (the worker overrides CMD). Building one
# image for both keeps dependency versions in sync and simplifies CI.
#
# Base image: python:3.9-slim
#   - "slim" variants strip out build tools/docs to minimize image size
#     and reduce the attack surface for production deployments, while
#     still providing a glibc-based Debian environment compatible with
#     scikit-learn's compiled wheels.
# ==============================================================================
FROM python:3.9-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr,
# which keeps container logs flushed immediately (important for
# `docker-compose logs -f` during the Action Phase demo).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# `curl` is required by the HEALTHCHECK below to probe /health.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------
# Dependency layer: copied and installed BEFORE application code so that
# Docker's build cache reuses this (slow) layer whenever only application
# code changes — a core container-build-optimization best practice.
# ------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------
# Application code
# ------------------------------------------------------------------------
COPY train_model.py api_server.py kafka_client.py ./

# ------------------------------------------------------------------------
# Train the model AT BUILD TIME so the resulting image is self-contained
# and reproducible: `docker build` always produces an image that already
# contains a freshly-trained model.pkl, with no separate "training step"
# required before the container can serve predictions.
# ------------------------------------------------------------------------
RUN python train_model.py

# Port exposed by the FastAPI/uvicorn server (api service only — the
# kafka-worker service does not bind a port, but EXPOSE is harmless for it).
EXPOSE 8000

# Container-level health check, consumed by docker-compose's
# `condition: service_healthy` dependency ordering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the FastAPI service. The kafka-worker service in
# docker-compose.yml overrides this with:
#   command: ["python", "kafka_client.py", "worker"]
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
