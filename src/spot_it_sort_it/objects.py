"""Per-object boxes + per-object normal/anomalous from ONE PatchCore inference.

The inspection surface is plain cardboard, so objects are found by colour difference from an empty-surface
reference (no detector training needed). Each object's score is how much hotter PatchCore's anomaly map is on the
object's own pixels (90th percentile) than on the empty surface of the same frame (median). Subtracting the frame's
background cancels global shifts (cardboard moved, lighting), and using object pixels ignores creases in the box.
A frame with a good bolt and a bad screw yields one 'normal' box and one 'anomalous' box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float  # object heat minus background heat (0 = looks like the surface; ~0.3+ = clearly anomalous)
    label: str    # normal | hold | anomalous

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


def ignore_mask(h: int, w: int) -> np.ndarray:
    """Regions that are never objects: gripper fingers (bottom) and the table strips at the frame edges."""
    m = np.zeros((h, w), np.uint8)
    m[int(h * 0.88):, :] = 255          # gripper fingers along the bottom
    m[:, : int(w * 0.09)] = 255         # left table strip / cardboard edge
    m[:, int(w * 0.91):] = 255          # right edge band
    return m


def local_background(bgr: np.ndarray) -> np.ndarray:
    """Estimate the empty surface from the frame itself (robust to the cardboard shifting or lighting drift)."""
    small = cv2.resize(bgr, (bgr.shape[1] // 8, bgr.shape[0] // 8), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.medianBlur(small, 15), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def find_objects(bgr: np.ndarray, background_bgr: np.ndarray | None = None, min_area: int = 120, diff_thresh: float = 22.0):
    """Boxes of compact things that stand out from the surface (colour difference in Lab)."""
    ref_bgr = local_background(bgr) if background_bgr is None else background_bgr
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(cv2.GaussianBlur(ref_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.linalg.norm(lab - ref, axis=2)
    mask = (diff > diff_thresh).astype(np.uint8) * 255
    mask[ignore_mask(*mask.shape) > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        fill = area / float(w * h)
        H, W = bgr.shape[:2]
        # partly out of view (frame border or the masked edge bands): can't judge or pick it reliably
        touches_edge = y <= 6 or x <= int(W * 0.09) + 3 or x + w >= int(W * 0.91) - 3 or y + h >= int(H * 0.88) - 3
        if area >= min_area and fill > 0.12 and w * h < 0.25 * H * W and not touches_edge:
            pad = 6
            boxes.append((max(0, x - pad), max(0, y - pad), min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)))
    return boxes


def foreground_mask(bgr: np.ndarray, diff_thresh: float = 22.0) -> np.ndarray:
    ref = local_background(bgr)
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = cv2.cvtColor(cv2.GaussianBlur(ref, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = (np.linalg.norm(lab - rl, axis=2) > diff_thresh).astype(np.uint8)
    fg[ignore_mask(*fg.shape) > 0] = 0
    return fg


def label_objects(bgr: np.ndarray, boxes, anomaly_map: np.ndarray, box_lo: float = 0.15, reject_hi: float = 0.28):
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


COLORS = {"normal": (60, 170, 60), "hold": (0, 170, 230), "anomalous": (40, 40, 220)}


def draw(bgr: np.ndarray, objs: list[ObjectBox]) -> np.ndarray:
    img = bgr.copy()
    for o in objs:
        c = COLORS[o.label]
        cv2.rectangle(img, (o.x1, o.y1), (o.x2, o.y2), c, 2)
        cv2.putText(img, f"{o.label} {o.score:.2f}", (o.x1, max(14, o.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img
