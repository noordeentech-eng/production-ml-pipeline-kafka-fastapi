"""
api_server.py
================================================================================
Technical Phase — FastAPI Inference Service (Step 2 of the pipeline)

Loads `model.pkl` (produced by `train_model.py`) ONCE at application startup
via a FastAPI `lifespan` handler, and exposes:

    GET  /          -> service / model metadata
    GET  /health     -> health check (used by the Docker Compose healthcheck
                         and by container orchestrators such as Kubernetes)
    POST /predict    -> run inference on a single Iris sample

--------------------------------------------------------------------------------
Kafka audit-trail integration (optional, off by default)
--------------------------------------------------------------------------------
If the environment variable `KAFKA_ENABLED=true` is set, every `/predict`
response is ALSO published to the `ml-predictions` Kafka topic (via
`kafka_client.publish_prediction`). This gives the synchronous REST path the
same audit trail / real-time-analytics benefits as the asynchronous Kafka
path (see README.md "Why Kafka for ML Pipelines?").

The API remains fully functional if Kafka is unreachable or disabled —
Kafka publishing is fire-and-forget and never blocks or fails a prediction
response. This "degrade gracefully" pattern is a production deployment
best practice: a logging/analytics side-channel should never be allowed to
take down the primary serving path.
--------------------------------------------------------------------------------
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api_server")

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"

# Shared application state populated at startup (avoids re-loading the
# model on every request, and avoids global mutable module state).
ml_state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------------------------------------------
    # Startup
    # ---------------------------------------------------------------
    logger.info("Loading model artifact from '%s'", MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    ml_state["model"] = artifact["model"]
    ml_state["feature_names"] = artifact["feature_names"]
    ml_state["target_names"] = artifact["target_names"]
    ml_state["test_accuracy"] = artifact.get("test_accuracy")

    ml_state["producer"] = None
    if KAFKA_ENABLED:
        try:
            from kafka_client import get_producer
            ml_state["producer"] = get_producer()
            logger.info("Kafka producer initialized (KAFKA_ENABLED=true)")
        except Exception as exc:  # pragma: no cover - defensive startup guard
            logger.warning("Kafka enabled but producer init failed (%s). "
                            "Continuing without Kafka audit logging.", exc)
    else:
        logger.info("Kafka audit logging disabled (KAFKA_ENABLED=false)")

    yield

    # ---------------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------------
    producer = ml_state.get("producer")
    if producer is not None:
        producer.flush()
        producer.close()
        logger.info("Kafka producer closed")


app = FastAPI(
    title="Iris Prediction Service",
    description=(
        "Production inference API for a scikit-learn LogisticRegression "
        "Iris classifier, with optional Kafka audit-trail publishing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Request / Response schemas
# --------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """A single Iris sample: [sepal_length, sepal_width, petal_length, petal_width] in cm."""

    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[sepal_length, sepal_width, petal_length, petal_width] in cm",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    prediction_index: int
    probabilities: Dict[str, float]
    latency_ms: float


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    """Basic service/model metadata — useful for smoke-testing deployments."""
    return {
        "service": "iris-prediction-api",
        "status": "running",
        "model_features": ml_state["feature_names"],
        "model_classes": ml_state["target_names"],
        "model_test_accuracy": ml_state["test_accuracy"],
        "kafka_enabled": KAFKA_ENABLED,
    }


@app.get("/health")
def health():
    """
    Liveness/readiness probe.

    Docker Compose (and Kubernetes) use this endpoint to decide whether the
    container is ready to receive traffic and whether it needs restarting.
    """
    return {"status": "ok", "model_loaded": "model" in ml_state}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Run inference on a single Iris sample.

    Request body:
        {"features": [5.1, 3.5, 1.4, 0.2]}

    Response:
        {
          "request_id": "...",
          "prediction": "setosa",
          "prediction_index": 0,
          "probabilities": {"setosa": 0.97, "versicolor": 0.02, "virginica": 0.01},
          "latency_ms": 0.42
        }
    """
    start = time.perf_counter()

    model = ml_state["model"]
    target_names = ml_state["target_names"]

    X = np.array(payload.features, dtype=float).reshape(1, -1)

    try:
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000
    request_id = str(uuid.uuid4())

    response = {
        "request_id": request_id,
        "prediction": target_names[pred_idx],
        "prediction_index": pred_idx,
        "probabilities": {
            target_names[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "latency_ms": round(latency_ms, 3),
    }

    # Fire-and-forget audit publish — never blocks or fails the response.
    producer = ml_state.get("producer")
    if producer is not None:
        try:
            from kafka_client import publish_prediction
            publish_prediction(producer, request_id, payload.features, response)
        except Exception as exc:  # pragma: no cover - never fail the request
            logger.warning("Failed to publish prediction to Kafka: %s", exc)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
