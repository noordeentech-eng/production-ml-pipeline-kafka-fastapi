"""
kafka_client.py
================================================================================
Kafka Producer / Consumer Logic for Asynchronous Prediction Streaming
(Step 3 of the pipeline)

Topics
--------------------------------------------------------------------------------
  ml-requests     Raw feature payloads awaiting inference. Produced by the
                  `producer` mode below (simulating an upstream system —
                  e.g. an IoT sensor, batch job, or another microservice —
                  emitting prediction requests).

  ml-predictions  Inference results (prediction + probabilities + latency).
                  Written by:
                    (a) the `worker` mode below, which consumes ml-requests,
                        runs the model, and republishes the result; and
                    (b) api_server.py, when KAFKA_ENABLED=true, as an audit
                        trail for every synchronous /predict call.

Modes (run via `python kafka_client.py <mode>`)
--------------------------------------------------------------------------------
  producer  python kafka_client.py producer [N]
            Publish N (default 10) sample Iris feature vectors to
            'ml-requests'. This is the "Action Phase" producer used to send
            10 sample requests for the demo.

  worker    python kafka_client.py worker
            Long-running consumer that subscribes to 'ml-requests', loads
            model.pkl, runs inference for each message, and publishes the
            result to 'ml-predictions'. This is the asynchronous inference
            service that decouples request ingestion from model execution.

  consumer  python kafka_client.py consumer
            Long-running consumer that subscribes to 'ml-predictions' and
            prints each result to the console in real time — simulating a
            downstream analytics/monitoring/audit consumer.

Environment variables
--------------------------------------------------------------------------------
  KAFKA_BROKER_URL   Kafka bootstrap server (default: "localhost:9092").
                     Inside Docker Compose this is set to "kafka:29092".
  MODEL_PATH         Path to the joblib model artifact (default: "model.pkl").
"""

import os
import sys
import json
import time
import uuid
import random
import logging

import joblib
import numpy as np
from kafka import KafkaProducer, KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kafka_client")

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
REQUEST_TOPIC = "ml-requests"
PREDICTION_TOPIC = "ml-predictions"
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

# A handful of well-known Iris samples (with their true labels) used to
# generate realistic demo traffic and to sanity-check predictions against
# known ground truth.
SAMPLE_IRIS_RECORDS = [
    {"features": [5.1, 3.5, 1.4, 0.2], "true_label": "setosa"},
    {"features": [4.9, 3.0, 1.4, 0.2], "true_label": "setosa"},
    {"features": [5.0, 3.4, 1.5, 0.2], "true_label": "setosa"},
    {"features": [4.6, 3.1, 1.5, 0.2], "true_label": "setosa"},
    {"features": [6.7, 3.1, 4.4, 1.4], "true_label": "versicolor"},
    {"features": [5.9, 3.0, 4.2, 1.5], "true_label": "versicolor"},
    {"features": [6.0, 2.9, 4.5, 1.5], "true_label": "versicolor"},
    {"features": [6.3, 3.3, 6.0, 2.5], "true_label": "virginica"},
    {"features": [6.5, 3.0, 5.2, 2.0], "true_label": "virginica"},
    {"features": [6.9, 3.1, 5.4, 2.1], "true_label": "virginica"},
]


# --------------------------------------------------------------------------
# Shared producer/consumer factories
# --------------------------------------------------------------------------
def get_producer() -> KafkaProducer:
    """Create a JSON-serializing Kafka producer with retry on transient errors."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
        linger_ms=5,
    )


def get_consumer(topic: str, group_id: str) -> KafkaConsumer:
    """
    Create a JSON-deserializing Kafka consumer.

    `group_id` matters for scalability: multiple worker replicas sharing the
    same group_id will have partitions of `topic` load-balanced across them
    automatically by Kafka, enabling horizontal scaling of the inference
    workers without any code changes.
    """
    return KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=group_id,
    )


def publish_prediction(producer: KafkaProducer, request_id: str, features, response: dict):
    """
    Publish a completed prediction to 'ml-predictions'.

    Called from api_server.py (source="api_server") so that synchronous
    REST predictions are captured in the same audit-trail topic as
    asynchronous worker predictions (source="kafka_worker").
    """
    message = {
        "request_id": request_id,
        "features": features,
        "prediction": response["prediction"],
        "probabilities": response["probabilities"],
        "latency_ms": response["latency_ms"],
        "source": "api_server",
        "processed_at": time.time(),
    }
    producer.send(PREDICTION_TOPIC, value=message)
    producer.flush()


# --------------------------------------------------------------------------
# MODE: producer — publish sample requests to 'ml-requests'
# --------------------------------------------------------------------------
def run_producer(n: int = 10):
    producer = get_producer()
    logger.info(
        "Producer connected to '%s'. Sending %d sample request(s) to topic '%s'...",
        KAFKA_BROKER, n, REQUEST_TOPIC,
    )

    for i in range(n):
        record = random.choice(SAMPLE_IRIS_RECORDS)
        message = {
            "request_id": str(uuid.uuid4()),
            "features": record["features"],
            "true_label": record["true_label"],
            "sent_at": time.time(),
        }
        producer.send(REQUEST_TOPIC, value=message)
        producer.flush()
        logger.info(
            "[%d/%d] -> ml-requests | request_id=%s | features=%s | true_label=%s",
            i + 1, n, message["request_id"], message["features"], message["true_label"],
        )
        time.sleep(0.5)  # small delay so output is easy to follow live

    producer.close()
    logger.info("Done. Sent %d request(s) to '%s'.", n, REQUEST_TOPIC)


# --------------------------------------------------------------------------
# MODE: worker — consume 'ml-requests', run inference, produce 'ml-predictions'
# --------------------------------------------------------------------------
def run_worker():
    logger.info("Loading model artifact from '%s'", MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    target_names = artifact["target_names"]

    consumer = get_consumer(REQUEST_TOPIC, group_id="ml-prediction-workers")
    producer = get_producer()

    logger.info(
        "Worker ready. Consuming '%s' -> running inference -> producing '%s'...",
        REQUEST_TOPIC, PREDICTION_TOPIC,
    )

    for msg in consumer:
        request = msg.value
        start = time.perf_counter()

        X = np.array(request["features"], dtype=float).reshape(1, -1)
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "request_id": request["request_id"],
            "features": request["features"],
            "true_label": request.get("true_label"),
            "prediction": target_names[pred_idx],
            "probabilities": {
                target_names[i]: round(float(p), 4) for i, p in enumerate(proba)
            },
            "latency_ms": round(latency_ms, 3),
            "source": "kafka_worker",
            "sent_at": request.get("sent_at"),
            "processed_at": time.time(),
        }

        producer.send(PREDICTION_TOPIC, value=result)
        producer.flush()

        logger.info(
            "%s -> %s | prediction=%s | latency=%.3fms",
            REQUEST_TOPIC, PREDICTION_TOPIC, result["prediction"], result["latency_ms"],
        )


# --------------------------------------------------------------------------
# MODE: consumer — real-time monitoring of 'ml-predictions'
# --------------------------------------------------------------------------
def run_consumer():
    consumer = get_consumer(PREDICTION_TOPIC, group_id="ml-prediction-monitors")
    logger.info("Consumer ready. Listening on '%s' for real-time prediction output...",
                 PREDICTION_TOPIC)
    logger.info("(Run 'python kafka_client.py producer' in another terminal to generate traffic)")

    for msg in consumer:
        result = msg.value

        match = ""
        if result.get("true_label"):
            ok = "MATCH" if result["true_label"] == result["prediction"] else "MISMATCH"
            match = f" | true_label={result['true_label']} ({ok})"

        # end-to-end latency: time from request creation to consumption here
        e2e_ms = ""
        if result.get("sent_at"):
            e2e_ms = f" | end_to_end={ (time.time() - result['sent_at']) * 1000:.1f}ms"

        logger.info(
            "request_id=%s | prediction=%s | inference_latency=%.3fms | source=%s%s%s",
            result["request_id"], result["prediction"], result["latency_ms"],
            result.get("source", "unknown"), match, e2e_ms,
        )


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    valid_modes = ("producer", "worker", "consumer")
    if len(sys.argv) < 2 or sys.argv[1] not in valid_modes:
        print(f"Usage: python kafka_client.py [{'|'.join(valid_modes)}] [N]")
        print("  producer [N]  - send N sample requests to 'ml-requests' (default N=10)")
        print("  worker        - consume 'ml-requests', predict, produce 'ml-predictions'")
        print("  consumer      - print real-time results from 'ml-predictions'")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "producer":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        run_producer(n)
    elif mode == "worker":
        run_worker()
    elif mode == "consumer":
        run_consumer()
