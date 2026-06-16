"""
test_api.py
================================================================================
Action Phase — API Endpoint Test Script

Verifies that the /predict endpoint is functioning correctly by sending
requests with known test images and checking the response schema.

Usage:
    python test_api.py --url http://localhost:8000 --image_dir dataset/new_images
"""

import os
import sys
import glob
import argparse
import requests

BASE_URL = "http://localhost:8000"
VALID_ACTIONS = {"REJECT", "FLAG_FOR_REVIEW", "PASS"}
VALID_CLASSES = {"good", "defect"}


def test_health(base_url):
    print("── Health check ──────────────────────────────────")
    resp = requests.get(f"{base_url}/health", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok", f"Expected status=ok, got {data}"
    assert data["model_loaded"] is True, "Model is not loaded!"
    print(f"  PASS  /health -> {data}")


def test_root(base_url):
    print("── Root endpoint ─────────────────────────────────")
    resp = requests.get(f"{base_url}/", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    print(f"  PASS  / -> {data}")


def test_predict(base_url, image_dir):
    print("── /predict endpoint ─────────────────────────────")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        print(f"  SKIP  No images found in '{image_dir}'")
        return

    passed = 0
    failed = 0
    for path in image_paths:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{base_url}/predict",
                files={"file": (os.path.basename(path), f, "image/jpeg")},
                timeout=10,
            )

        if resp.status_code != 200:
            print(f"  FAIL  {os.path.basename(path)} -> HTTP {resp.status_code}: {resp.text}")
            failed += 1
            continue

        data = resp.json()

        # Schema checks
        errors = []
        if "defect_probability" not in data:
            errors.append("missing 'defect_probability'")
        elif not (0.0 <= data["defect_probability"] <= 1.0):
            errors.append(f"defect_probability out of range: {data['defect_probability']}")
        if data.get("predicted_class") not in VALID_CLASSES:
            errors.append(f"invalid predicted_class: {data.get('predicted_class')}")
        if data.get("factory_action") not in VALID_ACTIONS:
            errors.append(f"invalid factory_action: {data.get('factory_action')}")

        if errors:
            print(f"  FAIL  {os.path.basename(path)} -> {errors}")
            failed += 1
        else:
            print(
                f"  PASS  {os.path.basename(path):30s} "
                f"class={data['predicted_class']:6s} "
                f"P(defect)={data['defect_probability']:.3f} "
                f"action={data['factory_action']:15s} "
                f"latency={data['latency_ms']:.1f}ms"
            )
            passed += 1

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(image_paths)} images")
    if failed > 0:
        sys.exit(1)


def test_invalid_input(base_url):
    print("── Invalid input handling ────────────────────────")
    # Send a text file instead of an image
    resp = requests.post(
        f"{base_url}/predict",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
        timeout=5,
    )
    assert resp.status_code == 400, f"Expected 400 for bad input, got {resp.status_code}"
    print(f"  PASS  Bad input correctly rejected with HTTP 400")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the defect detection API")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--image_dir", default=os.path.join("dataset", "new_images"))
    args = parser.parse_args()

    print(f"\nTesting API at {args.url}\n")
    try:
        test_health(args.url)
        test_root(args.url)
        test_predict(args.url, args.image_dir)
        test_invalid_input(args.url)
        print("\nAll tests passed.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to {args.url}. Is the API running?")
        print("Run: docker-compose up --build")
        sys.exit(1)
