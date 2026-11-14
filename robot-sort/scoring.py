"""Per-part normal / hold / anomalous from YOLO part boxes and ONE PatchCore anomaly map.

Boxes come from our one-class YOLO detector (vision-6). Each part's score is how much hotter PatchCore's anomaly map
is on the part's own pixels (90th percentile) than on the empty surface of the same frame (median). Subtracting the
frame's background cancels global shifts (cardboard moved, lighting), and using part pixels ignores creases in the box.
The scoring is copied from the team's sortit:vision:objects module (objects.py); only the box source changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from viam.proto.service.vision import Detection


@dataclass
class ObjectBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float  # part heat minus background heat (0 = looks like the surface; ~0.3+ = clearly anomalous)
    label: str    # normal | hold | anomalous

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


def ignore_mask(h: int, w: int) -> np.ndarray:
    """Regions that are never parts: gripper fingers (bottom) and the table strips at the frame edges."""
    m = np.zeros((h, w), np.uint8)
    m[int(h * 0.88):, :] = 255          # gripper fingers along the bottom
    m[:, : int(w * 0.09)] = 255         # left table strip / cardboard edge
    m[:, int(w * 0.91):] = 255          # right edge band
    return m


def local_background(bgr: np.ndarray) -> np.ndarray:
    """Estimate the empty surface from the frame itself (robust to the cardboard shifting or lighting drift)."""
    small = cv2.resize(bgr, (bgr.shape[1] // 8, bgr.shape[0] // 8), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.medianBlur(small, 15), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def foreground_mask(bgr: np.ndarray, diff_thresh: float = 22.0) -> np.ndarray:
    """Pixels that differ from the surface. Used only to pick a part's own pixels inside its YOLO box."""
    ref = local_background(bgr)
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = cv2.cvtColor(cv2.GaussianBlur(ref, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = (np.linalg.norm(lab - rl, axis=2) > diff_thresh).astype(np.uint8)
    fg[ignore_mask(*fg.shape) > 0] = 0
    return fg


def label_objects(bgr: np.ndarray, boxes, anomaly_map: np.ndarray, box_lo: float, reject_hi: float) -> list[ObjectBox]:
    h, w = bgr.shape[:2]
    amap = cv2.resize(anomaly_map.astype(np.float32), (w, h))
    fg = foreground_mask(bgr)
    in_box = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes:
        in_box[y1:y2, x1:x2] = True
    surface = (ignore_mask(h, w) == 0) & ~in_box
    bg = float(np.median(amap[surface])) if surface.any() else float(np.median(amap))
    out = []
    for x1, y1, x2, y2 in boxes:
        mk = cv2.dilate(fg[y1:y2, x1:x2], np.ones((7, 7), np.uint8)) > 0
        vals = amap[y1:y2, x1:x2][mk] if mk.any() else amap[y1:y2, x1:x2].ravel()
        s = float(np.percentile(vals, 90)) - bg
        label = "normal" if s < box_lo else "anomalous" if s > reject_hi else "hold"
        out.append(ObjectBox(x1, y1, x2, y2, s, label))
    return out


def score_parts(bgr: np.ndarray, detections: Sequence[Detection], anomaly_map: np.ndarray,
                min_confidence: float, box_lo: float, reject_hi: float) -> list[ObjectBox]:
    """Keep YOLO boxes with confidence >= min_confidence, then label each from the anomaly map."""
    h, w = bgr.shape[:2]
    boxes = [(max(0, int(d.x_min)), max(0, int(d.y_min)), min(w, int(d.x_max)), min(h, int(d.y_max)))
             for d in detections if d.confidence >= min_confidence]
    return label_objects(bgr, boxes, anomaly_map, box_lo, reject_hi)
