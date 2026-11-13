"""Stage 1 — Anomalib model compiled with OpenVINO (target: NPU)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from anomalib.deploy import OpenVINOInferencer

from .devices import resolve_device


@dataclass
class Detection:
    score: float
    is_anomalous: bool
    borderline: bool
    anomaly_map: np.ndarray | None
    latency_ms: float
    device: str


class AnomalyDetector:
    def __init__(self, model_xml: str, device: str = "NPU", threshold: float | None = None, borderline_margin: float = 0.1):
        self.device = resolve_device(device)
        self.inferencer = OpenVINOInferencer(path=model_xml, device=self.device, config={"PERFORMANCE_HINT": "LATENCY"})
        self.threshold = threshold
        self.margin = borderline_margin

    def __call__(self, bgr: np.ndarray) -> Detection:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        pred = self.inferencer.predict(rgb)
        dt = (time.perf_counter() - t0) * 1000
        # Anomalib exports normalise scores so 0.5 is the learned threshold.
        score = float(np.asarray(pred.pred_score).reshape(-1)[0])
        thr = 0.5 if self.threshold is None else self.threshold
        amap = None if pred.anomaly_map is None else np.asarray(pred.anomaly_map).squeeze()
        return Detection(
            score=score,
            is_anomalous=score >= thr,
            borderline=abs(score - thr) <= self.margin,
            anomaly_map=amap,
            latency_ms=dt,
            device=self.device,
        )
