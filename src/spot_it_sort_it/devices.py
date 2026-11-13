"""Pick the best available OpenVINO device, falling back gracefully (e.g. on a dev Mac with CPU only)."""
from __future__ import annotations

import openvino as ov

FALLBACK = {"NPU": ["NPU", "GPU", "CPU"], "GPU": ["GPU", "CPU"], "CPU": ["CPU"]}


def resolve_device(preferred: str) -> str:
    available = ov.Core().available_devices
    for dev in FALLBACK.get(preferred.upper(), [preferred]):
        if any(a.startswith(dev) for a in available):
            return dev
    return "CPU"
