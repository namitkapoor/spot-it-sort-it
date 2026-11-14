"""robotsort:generic:sorter — the pick-and-place loop around robotsort:vision:parts.

One cycle: arm to inspect-pose -> parts.decide (logs new items) -> choose a logged item that is still in view ->
depth + frame system -> top-down grasp -> log first arm motion -> approach, descend, grab, lift -> destination
pose -> open. Every cycle takes a fresh photo, so a part a judge moved is simply seen again (recovery).

Config attributes (defaults in brackets):
  parts_name (required)  robotsort:vision:parts service
  camera_name [cam], arm_name [arm], gripper_name [gripper], motion_name [builtin]
  inspect_pose [inspect-pose], box_pose [good-pose], reject_pose [bad-pose], hold_pose [unsure-pose]
                         erh:vmodutils:arm-position-saver switches; SetPosition(2) moves the arm there
  approach_mm [80]       height above the part top for the approach and lift
  grasp_depth_mm [5]     how far below the part top the gripper frame goes
  floor_clearance_mm [3] the gripper frame never goes closer than this to the cardboard
  yaw_offset_deg [0]     finger direction correction (calibrate at the station)
  dry_run [true]         when true, only "plan" works; nothing moves
DoCommand:
  {"command": "plan"}                  one cycle without arm motion after inspect-pose: target + grasp (logs the item)
  {"command": "sort_one"}              one full cycle
  {"command": "start", "max_items": 6} run cycles in the background until no parts are left, max_items, or stop
  {"command": "stop"}                  stop the background run and the arm
  {"command": "status"}                background run state and results
"""
import asyncio
from dataclasses import asdict
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.services.vision import Vision
from viam.utils import ValueTypes, struct_to_dict

from geometry import decode_depth, plan_grasp

LOGGER = getLogger(__name__)
MATCH_PX = 45  # same as item_log.ItemLog.match_px


def pick_target(pending: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Oldest logged-but-not-picked item that is still in view now (a current part within MATCH_PX of its centre).
    Returns the item with its box/centre updated to where the part is now."""
    for item in pending:
        for part in current:
            if all(abs(part["center"][i] - item["center"][i]) <= MATCH_PX for i in (0, 1)):
                return {**item, "box": part["box"], "center": part["center"]}
    return None


class RobotSortSorter(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("robotsort", "generic"), "sorter")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @staticmethod
    def _names(a: Mapping[str, Any]) -> dict:
        return {
            "parts": a["parts_name"], "camera": a.get("camera_name", "cam"), "arm": a.get("arm_name", "arm"),
            "gripper": a.get("gripper_name", "gripper"), "motion": a.get("motion_name", "builtin"),
            "inspect": a.get("inspect_pose", "inspect-pose"), "box": a.get("box_pose", "good-pose"),
            "reject": a.get("reject_pose", "bad-pose"), "hold": a.get("hold_pose", "unsure-pose"),
        }

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        a = struct_to_dict(config.attributes)
        if not a.get("parts_name"):
            raise ValueError("parts_name is required")
        n = cls._names(a)
        deps = [n[k] for k in ("parts", "camera", "arm", "gripper", "inspect", "box", "reject", "hold")]
        return deps + [f"rdk:service:motion/{n['motion']}"], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        a = struct_to_dict(config.attributes)
        n = self._names(a)
        self.parts: Vision = dependencies[Vision.get_resource_name(n["parts"])]  # type: ignore[assignment]
        self.camera: Camera = dependencies[Camera.get_resource_name(n["camera"])]  # type: ignore[assignment]
        self.arm: Arm = dependencies[Arm.get_resource_name(n["arm"])]  # type: ignore[assignment]
        self.gripper: Gripper = dependencies[Gripper.get_resource_name(n["gripper"])]  # type: ignore[assignment]
        self.motion: Motion = dependencies[Motion.get_resource_name(n["motion"])]  # type: ignore[assignment]
        self.camera_name, self.gripper_name = n["camera"], n["gripper"]
        self.poses = {k: dependencies[Switch.get_resource_name(n[k])] for k in ("inspect", "box", "reject", "hold")}
        self.approach_mm = float(a.get("approach_mm", 80))
        self.grasp_depth_mm = float(a.get("grasp_depth_mm", 5))
        self.floor_clearance_mm = float(a.get("floor_clearance_mm", 3))
        self.yaw_offset_deg = float(a.get("yaw_offset_deg", 0))
        self.dry_run = bool(a.get("dry_run", True))
        if not hasattr(self, "_task"):
            self._task: Optional[asyncio.Task] = None
            self._results: list = []
            self._error = ""

    async def _goto(self, name: str, timeout_s: float = 30) -> None:
        """Move to a saved pose and wait until the switch is back to idle and the arm has stopped."""
        sw: Switch = self.poses[name]  # type: ignore[assignment]
        await sw.set_position(2)
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout_s
        while await sw.get_position() != 0 or await self.arm.is_moving():
            if loop.time() > end:
                raise TimeoutError(f"arm did not reach {name} in {timeout_s} s")
            await asyncio.sleep(0.1)

    async def _move(self, x: float, y: float, z: float, theta: float, straight: bool) -> None:
        """Move the gripper frame to (x, y, z) in world, pointing straight down, rotated theta about world z."""
        dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta))
        constraints = Constraints(linear_constraint=[LinearConstraint()]) if straight else None
        await self.motion.move(component_name=self.gripper_name, destination=dest, constraints=constraints)

    async def _cycle(self, move: bool) -> dict:
        await self._goto("inspect")
        decided = await self.parts.do_command({"command": "decide", "frames": 1})
        current = (await self.parts.do_command({"command": "objects"}))["objects"]
        target = pick_target(decided["pending"], current)  # type: ignore[arg-type]
        if target is None:
            return {"status": "empty", "in_view": len(current)}  # type: ignore[arg-type]
        images, _ = await self.camera.get_images()
        color = next(i for i in images if i.mime_type in ("image/jpeg", "image/png"))
        depth = next(i for i in images if i.mime_type == "image/vnd.viam.dep")
        bgr = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode camera image")
        intr = (await self.camera.get_properties()).intrinsic_parameters
        cam_pose = (await self.motion.get_pose(component_name=self.camera_name, destination_frame="world")).pose
        g = plan_grasp([int(v) for v in target["box"]], bgr, decode_depth(depth.data), intr, cam_pose,
                       self.grasp_depth_mm, self.floor_clearance_mm, self.yaw_offset_deg)
        out = {"item": target["item"], "destination": target["destination"], "grasp": asdict(g)}
        if not move:
            return {"status": "planned", **out}
        await self.parts.do_command({"command": "arm_start", "item": target["item"]})
        above = g.z_top + self.approach_mm
        await self._move(g.x, g.y, above, g.theta, straight=False)
        await self._move(g.x, g.y, g.z_grasp, g.theta, straight=True)
        held = await self.gripper.grab()
        await self._move(g.x, g.y, above, g.theta, straight=True)
        await self._goto(str(target["destination"]))
        await self.gripper.open()
        return {"status": "sorted", "held": held, **out}

    async def _run(self, max_items: int) -> None:
        try:
            await self.gripper.open()
            while len([r for r in self._results if r["status"] == "sorted"]) < max_items:
                r = await self._cycle(move=True)
                self._results.append(r)
                LOGGER.info("cycle: %s", r)
                if r["status"] == "empty":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # report through "status"; the run stops
            self._error = repr(e)
            LOGGER.exception("sorter run failed")

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        running = self._task is not None and not self._task.done()
        if cmd == "status":
            return {"running": running, "dry_run": self.dry_run, "error": self._error, "results": self._results}
        if cmd == "stop":
            if running:
                self._task.cancel()  # type: ignore[union-attr]
            await self.arm.stop()
            return {"stopped": True}
        if cmd == "plan":
            return await self._cycle(move=False)
        if cmd not in ("sort_one", "start"):
            raise ValueError("command must be one of: plan, sort_one, start, stop, status")
        if self.dry_run:
            raise ValueError("dry_run is true: set dry_run false in the config to allow arm motion")
        if running:
            raise ValueError("a run is already in progress; send stop first")
        if cmd == "sort_one":
            await self.gripper.open()
            return await self._cycle(move=True)
        self._results, self._error = [], ""
        self._task = asyncio.create_task(self._run(int(command.get("max_items", 6))))  # type: ignore[arg-type]
        return {"started": True}
