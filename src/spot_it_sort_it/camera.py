from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def frames(source: int | str, width: int = 1280, height: int = 720) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (frame_id, BGR image). Source can be a camera index, stream URL, or image folder."""
    if isinstance(source, str) and Path(source).is_dir():
        for p in sorted(Path(source).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                yield str(p), cv2.imread(str(p))
        return
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"cam-{i:06d}", frame
            i += 1
    finally:
        cap.release()
