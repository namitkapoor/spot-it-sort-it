"""Wiring test for robot-sort/main.py with fake camera, YOLO vision service, and PatchCore mlmodel."""
import asyncio
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from viam.media.video import ViamImage
from viam.proto.service.vision import Detection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeCamera:
    def __init__(self, jpeg):
        self.jpeg = jpeg

    async def get_images(self):
        depth = ViamImage(b"\x00" * 10, "image/vnd.viam.dep")
        return [depth, ViamImage(self.jpeg, "image/jpeg")], None


class FakeDetector:
    async def get_detections(self, image, **kw):
        return [Detection(x_min=195, y_min=145, x_max=245, y_max=195, confidence=0.9, class_name="part"),
                Detection(x_min=395, y_min=145, x_max=445, y_max=195, confidence=0.85, class_name="part")]


class FakePatchCore:
    async def metadata(self):
        return SimpleNamespace(input_info=[SimpleNamespace(name="input", shape=[1, 3, 256, 256])])

    async def infer(self, inputs):
        assert inputs["input"].shape == (1, 3, 256, 256)
        amap = np.full((1, 1, 256, 256), 0.1, np.float32)
        amap[..., 80:102, 160:176] = 0.9
        return {"anomaly_map": amap, "pred_score": np.array([0.8], np.float32)}


def make(tmp_path):
    os.environ["VIAM_MODULE_DATA"] = str(tmp_path)
    from main import RobotSortParts
    from item_log import ItemLog
    assert not RobotSortParts.__abstractmethods__
    bgr = np.full((480, 640, 3), (120, 150, 180), np.uint8)
    bgr[150:190, 200:240] = 40
    bgr[150:190, 400:440] = 40
    jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
    s = RobotSortParts("parts")
    s.detector, s.mlmodel, s.camera, s.camera_name = FakeDetector(), FakePatchCore(), FakeCamera(jpeg), "cam"
    s.min_confidence, s.box_lo, s.reject_hi, s._input, s.log = 0.5, 0.3, 0.5, None, ItemLog(root=tmp_path)
    return s


def test_objects_decide_and_arm_start(tmp_path):
    s = make(tmp_path)
    objs = asyncio.run(s.do_command({"command": "objects"}))
    assert [o["label"] for o in objs["objects"]] == ["normal", "anomalous"]
    d = asyncio.run(s.do_command({"command": "decide", "frames": 2}))
    assert [(i["item"], i["destination"]) for i in d["new"]] == [(1, "box"), (2, "reject")]
    again = asyncio.run(s.do_command({"command": "decide", "frames": 1}))
    assert again["new"] == []  # same parts still pending -> not logged twice
    lat = asyncio.run(s.do_command({"command": "arm_start", "item": 1}))
    assert lat["item"] == 1 and lat["latency_ms"] >= 0
    lines = open(d["log_path"]).read().strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("item,decision,destination,confidence")
