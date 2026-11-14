"""Unit test for robot-sort/scoring.py: YOLO boxes + anomaly map -> per-part labels."""
import os
import sys

import numpy as np
from viam.proto.service.vision import Detection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoring import score_parts  # noqa: E402


def det(x1, y1, x2, y2, conf):
    return Detection(x_min=x1, y_min=y1, x_max=x2, y_max=y2, confidence=conf, class_name="part")


def scene():
    bgr = np.full((480, 640, 3), (120, 150, 180), np.uint8)  # plain cardboard
    bgr[150:190, 200:240] = 40   # part A (dark square)
    bgr[150:190, 400:440] = 40   # part B
    amap = np.full((256, 256), 0.1, np.float32)
    amap[80:102, 160:176] = 0.9  # PatchCore heat over part B only (256-px map, 640x480 frame)
    return bgr, amap


def test_labels_each_box_from_its_own_heat_and_drops_low_confidence():
    bgr, amap = scene()
    dets = [det(195, 145, 245, 195, 0.9), det(395, 145, 445, 195, 0.85), det(100, 300, 140, 340, 0.3)]
    parts = score_parts(bgr, dets, amap, min_confidence=0.5, box_lo=0.3, reject_hi=0.5)
    assert [p.label for p in parts] == ["normal", "anomalous"]
    assert parts[0].center == (220, 170)
    assert parts[1].score > 0.5


def test_box_outside_frame_is_clipped():
    bgr, amap = scene()
    parts = score_parts(bgr, [det(-10, 150, 30, 190, 0.9)], amap, min_confidence=0.5, box_lo=0.3, reject_hi=0.5)
    assert (parts[0].x1, parts[0].y1, parts[0].x2, parts[0].y2) == (0, 150, 30, 190)
