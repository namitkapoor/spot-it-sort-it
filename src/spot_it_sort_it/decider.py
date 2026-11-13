"""Stage 2 — decide what to do about an item with a VLM via OpenVINO GenAI (target: iGPU).

The VLM only runs on items the detector flags (or scores near the threshold), so clean items never
pay its latency. If the VLM isn't available, a rule-based fallback keeps the line moving.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection
from .devices import resolve_device

PROMPT = """You are a quality-control inspector on a production line.
Scenario: {scenario}
An anomaly detector scored this item {score:.2f} (0 = normal, 1 = very anomalous; threshold 0.5).
The second image is a heatmap of where the detector saw the anomaly.
Choose exactly one action from {actions}.
Reply with JSON only: {{"action": "...", "defect": "<short defect name or none>", "reason": "<one sentence>"}}"""


@dataclass
class Decision:
    action: str
    defect: str
    reason: str
    latency_ms: float = 0.0
    source: str = "rules"
    raw: str = field(default="", repr=False)

    @property
    def is_good(self) -> bool:
        return self.action == "accept"


def heatmap_overlay(bgr: np.ndarray, amap: np.ndarray | None) -> np.ndarray:
    if amap is None:
        return bgr
    m = cv2.resize(amap.astype(np.float32), (bgr.shape[1], bgr.shape[0]))
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.addWeighted(bgr, 0.6, cv2.applyColorMap(m, cv2.COLORMAP_JET), 0.4, 0)


class Decider:
    def __init__(self, model_dir: str, device: str = "GPU", scenario: str = "", actions: list[str] | None = None,
                 max_new_tokens: int = 96, enabled: bool = True):
        self.scenario = scenario.strip()
        self.actions = actions or ["accept", "reject"]
        self.max_new_tokens = max_new_tokens
        self.pipe = None
        self.device = None
        if enabled and Path(model_dir).exists():
            import openvino_genai as ov_genai

            self.device = resolve_device(device)
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device)
        elif enabled:
            print(f"[decider] VLM not found at {model_dir} — using rule-based fallback. Run scripts/export_vlm.sh.")

    def rules(self, det: Detection) -> Decision:
        if det.is_anomalous:
            return Decision("reject", "anomaly", f"score {det.score:.2f} above threshold")
        return Decision("accept", "none", f"score {det.score:.2f} below threshold")

    def __call__(self, bgr: np.ndarray, det: Detection) -> Decision:
        needs_vlm = det.is_anomalous or det.borderline
        if self.pipe is None or not needs_vlm:
            return self.rules(det)

        import openvino as ov
        import openvino_genai as ov_genai

        def to_tensor(img: np.ndarray) -> ov.Tensor:
            rgb = cv2.cvtColor(cv2.resize(img, (448, 448)), cv2.COLOR_BGR2RGB)
            return ov.Tensor(rgb[None])

        images = [to_tensor(bgr), to_tensor(heatmap_overlay(bgr, det.anomaly_map))]
        prompt = PROMPT.format(scenario=self.scenario, score=det.score, actions=self.actions)
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = self.max_new_tokens
        t0 = time.perf_counter()
        out = str(self.pipe.generate(prompt, images=images, generation_config=cfg))
        dt = (time.perf_counter() - t0) * 1000
        return self._parse(out, det, dt)

    def _parse(self, text: str, det: Detection, dt: float) -> Decision:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
            action = d.get("action", "")
            if action not in self.actions:
                raise ValueError(action)
            return Decision(action, d.get("defect", "unknown"), d.get("reason", ""), dt, "vlm", text)
        except (ValueError, json.JSONDecodeError):
            fb = self.rules(det)
            fb.latency_ms, fb.source, fb.raw = dt, "vlm-unparsed->rules", text
            return fb
