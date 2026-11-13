"""Measure detector latency on every available OpenVINO device — evidence for the NPU/GPU/CPU split."""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import openvino as ov
from anomalib.deploy import OpenVINOInferencer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/anomaly/weights/openvino/model.xml")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    img = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    print(f"{'device':<8}{'compile_s':>10}{'p50_ms':>9}{'p95_ms':>9}{'fps':>7}")
    for dev in ov.Core().available_devices:
        t0 = time.perf_counter()
        inf = OpenVINOInferencer(path=args.model, device=dev, config={"PERFORMANCE_HINT": "LATENCY"})
        compile_s = time.perf_counter() - t0
        for _ in range(5):
            inf.predict(img)
        ts = []
        for _ in range(args.iters):
            t = time.perf_counter()
            inf.predict(img)
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        p50, p95 = statistics.median(ts), ts[int(0.95 * len(ts)) - 1]
        print(f"{dev:<8}{compile_s:>10.2f}{p50:>9.1f}{p95:>9.1f}{1000 / p50:>7.1f}")


if __name__ == "__main__":
    main()
