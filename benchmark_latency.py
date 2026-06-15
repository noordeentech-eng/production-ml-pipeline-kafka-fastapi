"""
benchmark_latency.py
================================================================================
Action Phase — Latency Benchmarking

Sends a configurable number of requests to the running `/predict` endpoint
and reports request-to-prediction latency statistics: mean, median (p50),
p95, p99, min, and max — both as measured by the client (round-trip,
including the Docker network hop) and as reported by the server
(`latency_ms` in the response body, i.e. pure model-inference time).

Usage:
    python benchmark_latency.py --url http://localhost:8000/predict --n 100
"""

import argparse
import statistics
import time

import requests

SAMPLE_FEATURES = [
    [5.1, 3.5, 1.4, 0.2],   # setosa
    [6.7, 3.1, 4.4, 1.4],   # versicolor
    [6.3, 3.3, 6.0, 2.5],   # virginica
]


def percentile(data, p):
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark /predict latency")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n", type=int, default=100, help="number of requests")
    args = parser.parse_args()

    client_latencies_ms = []
    server_latencies_ms = []
    errors = 0

    print(f"Sending {args.n} requests to {args.url} ...")
    for i in range(args.n):
        payload = {"features": SAMPLE_FEATURES[i % len(SAMPLE_FEATURES)]}

        start = time.perf_counter()
        try:
            resp = requests.post(args.url, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            errors += 1
            print(f"  request {i + 1} failed: {exc}")
            continue
        end = time.perf_counter()

        client_latencies_ms.append((end - start) * 1000)
        server_latencies_ms.append(data["latency_ms"])

    if not client_latencies_ms:
        print("All requests failed — is the API running?")
        return

    def summarize(label, values):
        print(f"\n{label} (ms), n={len(values)}")
        print(f"  mean   : {statistics.mean(values):.3f}")
        print(f"  median : {statistics.median(values):.3f}")
        print(f"  p95    : {percentile(values, 95):.3f}")
        print(f"  p99    : {percentile(values, 99):.3f}")
        print(f"  min    : {min(values):.3f}")
        print(f"  max    : {max(values):.3f}")

    summarize("Client-observed round-trip latency", client_latencies_ms)
    summarize("Server-reported inference latency", server_latencies_ms)
    print(f"\nErrors: {errors}/{args.n}")


if __name__ == "__main__":
    main()
