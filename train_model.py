"""
train_model.py
================================================================================
Discovery Phase — Modular Setup (Step 1 of the pipeline)

Trains a scikit-learn `LogisticRegression` classifier on the classic Iris
dataset and serializes the fitted model — along with the metadata needed
for inference (feature names, target class names, test accuracy) — to
`model.pkl` using `joblib`.

This single artifact is the source of truth consumed by:
  - `api_server.py`   (synchronous REST inference via FastAPI)
  - `kafka_client.py` (asynchronous inference inside the Kafka worker)

Both consumers load the exact same `model.pkl`, guaranteeing prediction
parity between the REST and streaming code paths.

Usage:
    python train_model.py
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "model.pkl"
RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # 1. Load the Iris dataset (built into scikit-learn, no download
    #    required — important for offline/CI/Docker build environments).
    # ------------------------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Train a LogisticRegression model.
    #    max_iter=200 ensures convergence on the (small, well-separated)
    #    Iris feature space.
    # ------------------------------------------------------------------
    model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Evaluate on the held-out test split.
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ------------------------------------------------------------------
    # 4. Serialize the model + inference metadata as a single artifact.
    #    Bundling feature/target names with the model avoids any
    #    train/serve skew about label ordering or feature ordering.
    # ------------------------------------------------------------------
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
        "test_accuracy": accuracy,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model artifact saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
