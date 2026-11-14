"""Tests for picker.py: target choice, and the order of calls in one cycle (with fake resources)."""
import asyncio
import os
import sys

import cv2
import numpy as np
import pytest
from viam.media.video import ViamImage
from viam.proto.component.camera import GetPropertiesResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from picker import RobotSortSorter, pick_target  # noqa: E402
from test_geometry import CAM, INTR  # noqa: E402


def test_pick_target_oldest_still_in_view():
    pending = [{"item": 1, "center": [100, 100], "box": [90, 90, 110, 110], "destination": "box"},
               {"item": 2, "center": [300, 200], "box": [290, 190, 310, 210], "destination": "reject"}]
    current = [{"center": [305, 210], "box": [295, 200, 315, 220], "label": "anomalous"}]  # item 1 was moved away
    t = pick_target(pending, current)
    assert t["item"] == 2 and t["destination"] == "reject"
    assert list(t["box"]) == [295, 200, 315, 220]  # where the part is now
    assert pick_target(pending, []) is None


class Rec:
    """Records every call as (resource, method, arg)."""
    calls: list

    def __init__(self, calls, name):
        self.calls, self.name = calls, name


class FakeSwitch(Rec):
    async def set_position(self, p):
        self.calls.append((self.name, "set_position", p))

    async def get_position(self):
        return 0


class FakeArm(Rec):
    async def is_moving(self):
        return False

    async def stop(self):
        self.calls.append(("arm", "stop", None))


class FakeGripper(Rec):
    async def open(self):
        self.calls.append(("gripper", "open", None))

    async def grab(self):
        self.calls.append(("gripper", "grab", None))
        return True


class FakeParts(Rec):
    async def do_command(self, cmd):
        self.calls.append(("parts", cmd["command"], cmd.get("item")))
        # DoCommand results travel as protobuf Struct, so every number arrives as a float
        part = {"item": 1.0, "center": [320.0, 240.0], "box": [275.0, 225.0, 365.0, 255.0], "destination": "reject"}
        if cmd["command"] == "decide":
            return {"pending": [part], "new": [part]}
        if cmd["command"] == "objects":
            return {"objects": [{"center": [320.0, 240.0], "box": [275.0, 225.0, 365.0, 255.0], "label": "anomalous"}]}
        return {}


class FakeCamera(Rec):
    async def get_images(self):
        bgr = np.full((480, 640, 3), (120, 150, 180), np.uint8)
        bgr[230:250, 280:360] = 40
        depth = np.full((480, 640), 322, np.uint16)
        depth[230:250, 280:360] = 312
        dep = b"DEPTHMAP" + (640).to_bytes(8, "big") + (480).to_bytes(8, "big") + depth.astype(">u2").tobytes()
        return [ViamImage(cv2.imencode(".jpg", bgr)[1].tobytes(), "image/jpeg"), ViamImage(dep, "image/vnd.viam.dep")], None

    async def get_properties(self):
        return GetPropertiesResponse(intrinsic_parameters=INTR)


class FakeMotion(Rec):
    async def get_pose(self, component_name, destination_frame):
        return type("P", (), {"pose": CAM})()

    async def move(self, component_name, destination, constraints=None):
        self.calls.append(("motion", "move", (component_name, round(destination.pose.z, 1), constraints is not None)))
        return True


def sorter(dry_run):
    calls = []
    s = RobotSortSorter("sorter")
    s.parts, s.camera, s.arm = FakeParts(calls, "parts"), FakeCamera(calls, "cam"), FakeArm(calls, "arm")
    s.gripper, s.motion = FakeGripper(calls, "gripper"), FakeMotion(calls, "motion")
    s.poses = {k: FakeSwitch(calls, k) for k in ("inspect", "box", "reject", "hold")}
    s.camera_name, s.gripper_name = "cam", "gripper"
    s.approach_mm, s.grasp_depth_mm, s.floor_clearance_mm, s.yaw_offset_deg = 80, 5, 3, 0
    s.dry_run, s._task, s._results, s._error = dry_run, None, [], ""
    return s, calls


def test_plan_does_not_move_after_inspect():
    s, calls = sorter(dry_run=True)
    r = asyncio.run(s.do_command({"command": "plan"}))
    assert r["status"] == "planned" and r["destination"] == "reject"
    assert [c for c in calls if c[0] in ("motion", "gripper")] == []
    with pytest.raises(ValueError, match="dry_run"):
        asyncio.run(s.do_command({"command": "sort_one"}))


def test_sort_one_order():
    s, calls = sorter(dry_run=False)
    r = asyncio.run(s.do_command({"command": "sort_one"}))
    assert r["status"] == "sorted" and r["held"] is True
    z_top, z_grasp = r["grasp"]["z_top"], r["grasp"]["z_grasp"]
    assert calls == [
        ("gripper", "open", None),
        ("inspect", "set_position", 2),
        ("parts", "decide", None),
        ("parts", "objects", None),
        ("parts", "arm_start", 1),                                   # logged before the first arm motion
        ("motion", "move", ("gripper", round(z_top + 80, 1), False)),  # approach
        ("motion", "move", ("gripper", round(z_grasp, 1), True)),      # straight down
        ("gripper", "grab", None),
        ("motion", "move", ("gripper", round(z_top + 80, 1), True)),   # straight up
        ("reject", "set_position", 2),
        ("gripper", "open", None),
    ]
