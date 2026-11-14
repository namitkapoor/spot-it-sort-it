"""robotsort:vision:parts — one box per part (from our YOLO detector), each labelled normal / hold / anomalous.

Takes ONE camera frame, sends it at the same time to the YOLO vision service (NPU) and to the PatchCore mlmodel
(GPU f32), then scores each YOLO box from PatchCore's anomaly map (scoring.py). Based on the team's
sortit:vision:objects module; the only change is that boxes come from YOLO instead of colour difference.

Config attributes:
  detector_name   (required)  vision service with the one-class 'part' YOLO model (vision-6)
  mlmodel_name    (required)  mlmodel serving the Anomalib export (outputs pred_score, anomaly_map)
  camera_name     (required)  camera that both models look through
  min_confidence  (0.5)       YOLO boxes below this are ignored
  box_lo          (0.3)       part score (heat above background) below this -> normal (box)
  reject_hi       (0.5)       part score above this -> anomalous (reject); in between -> hold
DoCommand:
  {"command": "objects"}                    every part now: box, centre, score, label (no logging)
  {"command": "decide", "frames": 3}        average N frames, LOG each new item (number, decision, confidence,
                                            timestamp) to disk, return new + pending items. Call before moving the arm.
  {"command": "next"}                       oldest logged item not yet picked: where it is and which bin
  {"command": "arm_start", "item": N}       record first arm motion for item N (gives latency in the log)
  {"command": "log"}                        the whole run's log (also on disk: $VIAM_MODULE_DATA/items-<run>.csv)
  {"command": "new_run"}                    start a fresh log (do this right before the judged run)
"""
import asyncio
import time
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.module.module import Module
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, Detection3D, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.mlmodel import MLModel
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

import picker  # noqa: F401  (defining RobotSortSorter registers robotsort:generic:sorter)
from item_log import ItemLog
from scoring import ObjectBox, score_parts

LOGGER = getLogger(__name__)


class RobotSortParts(Vision, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("robotsort", "vision"), "parts")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        a = struct_to_dict(config.attributes)
        for key in ("detector_name", "mlmodel_name", "camera_name"):
            if not a.get(key):
                raise ValueError(f"{key} is required")
        return [a["detector_name"], a["mlmodel_name"], a["camera_name"]], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        a = struct_to_dict(config.attributes)
        self.detector: Vision = dependencies[Vision.get_resource_name(a["detector_name"])]  # type: ignore[assignment]
        self.mlmodel: MLModel = dependencies[MLModel.get_resource_name(a["mlmodel_name"])]  # type: ignore[assignment]
        self.camera: Camera = dependencies[Camera.get_resource_name(a["camera_name"])]  # type: ignore[assignment]
        self.camera_name = a["camera_name"]
        self.min_confidence = float(a.get("min_confidence", 0.5))
        self.box_lo = float(a.get("box_lo", 0.3))
        self.reject_hi = float(a.get("reject_hi", 0.5))
        self._input: Optional[Tuple[str, int, int]] = None
        if not hasattr(self, "log"):  # keep the run's log across reconfigures
            self.log = ItemLog()

    async def _input_info(self) -> Tuple[str, int, int]:
        if self._input is None:
            md = await self.mlmodel.metadata()
            t = md.input_info[0]
            h, w = int(t.shape[-2]), int(t.shape[-1])
            self._input = (t.name, h, w)
        return self._input

    async def _camera_image(self) -> ViamImage:
        images, _ = await self.camera.get_images()
        for im in images:  # RealSense also returns depth; use the colour frame
            if "jpeg" in im.mime_type or "png" in im.mime_type:
                return ViamImage(im.data, im.mime_type)
        return ViamImage(images[0].data, images[0].mime_type)

    async def _analyze(self, image: ViamImage) -> Tuple[List[ObjectBox], float, Tuple[int, int]]:
        bgr = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode camera image")
        name, h, w = await self._input_info()
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None])
        dets, out = await asyncio.gather(self.detector.get_detections(image), self.mlmodel.infer({name: tensor}))
        amap = np.asarray(out["anomaly_map"], np.float32).squeeze()
        frame_score = float(np.asarray(out["pred_score"]).reshape(-1)[0])
        objs = score_parts(bgr, dets, amap, self.min_confidence, self.box_lo, self.reject_hi)
        return objs, frame_score, bgr.shape[:2]

    @staticmethod
    def _detections(objs: List[ObjectBox], hw: Tuple[int, int]) -> List[Detection]:
        h, w = hw
        return [Detection(x_min=o.x1, y_min=o.y1, x_max=o.x2, y_max=o.y2, confidence=o.score, class_name=o.label,
                          x_min_normalized=o.x1 / w, y_min_normalized=o.y1 / h, x_max_normalized=o.x2 / w,
                          y_max_normalized=o.y2 / h) for o in objs]

    @staticmethod
    def _classifications(objs: List[ObjectBox]) -> List[Classification]:
        """Frame summary: label of the highest-scoring part (empty -> 'empty')."""
        if not objs:
            return [Classification(class_name="empty", confidence=1.0)]
        worst = max(objs, key=lambda o: o.score)
        return [Classification(class_name=worst.label, confidence=worst.score)]

    async def get_detections(self, image: ViamImage, *, extra=None, timeout=None) -> List[Detection]:
        objs, _, hw = await self._analyze(image)
        return self._detections(objs, hw)

    async def get_detections_from_camera(self, camera_name: str, *, extra=None, timeout=None) -> List[Detection]:
        return await self.get_detections(await self._camera_image())

    async def get_classifications(self, image: ViamImage, count: int, *, extra=None, timeout=None) -> List[Classification]:
        objs, _, _ = await self._analyze(image)
        return self._classifications(objs)

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra=None, timeout=None):
        return await self.get_classifications(await self._camera_image(), count)

    async def capture_all_from_camera(self, camera_name: str, return_image: bool = False, return_classifications: bool = False,
                                      return_detections: bool = False, return_object_point_clouds: bool = False,
                                      return_detections_3d: bool = False, *, extra=None, timeout=None) -> CaptureAllResult:
        result = CaptureAllResult()
        image = await self._camera_image()
        if return_image:
            result.image = image
        if return_classifications or return_detections:
            objs, _, hw = await self._analyze(image)
            if return_detections:
                result.detections = self._detections(objs, hw)
            if return_classifications:
                result.classifications = self._classifications(objs)
        return result

    async def get_object_point_clouds(self, camera_name: str, *, extra=None, timeout=None) -> List[PointCloudObject]:
        raise NotSupportedError("robotsort:vision:parts does not support point clouds")

    async def get_detections_3d(self, camera_name: str, *, extra=None, timeout=None) -> List[Detection3D]:
        raise NotSupportedError("robotsort:vision:parts does not support 3D detections")

    async def get_properties(self, *, extra=None, timeout=None) -> GetPropertiesResponse:
        return GetPropertiesResponse(classifications_supported=True, detections_supported=True,
                                     object_point_clouds_supported=False, detections_3d_supported=False,
                                     default_camera=self.camera_name)

    async def _decide(self, frames: int) -> dict:
        seen = time.time()
        runs = []
        for _ in range(max(1, frames)):
            objs, _, _ = await self._analyze(await self._camera_image())
            runs.append(objs)
        base = runs[0]
        for o in base:  # average each part's score over the frames it was matched in
            scores = [o.score]
            for other in runs[1:]:
                m = min(other, key=lambda b: abs(b.center[0] - o.center[0]) + abs(b.center[1] - o.center[1]), default=None)
                if m and abs(m.center[0] - o.center[0]) + abs(m.center[1] - o.center[1]) < 60:
                    scores.append(m.score)
            o.score = float(np.mean(scores))
            o.label = "normal" if o.score < self.box_lo else "anomalous" if o.score > self.reject_hi else "hold"
        new = self.log.record(base, seen, len(runs), self.box_lo, self.reject_hi)
        pend = {i.item for i in self.log.pending()}
        return {"run": self.log.run, "log_path": str(self.log.path),
                "decided_ms": round((time.time() - seen) * 1000, 1),
                "new": [d for d in self.log.as_dicts() if d["item"] in {i.item for i in new}],
                "pending": [d for d in self.log.as_dicts() if d["item"] in pend]}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "decide":
            return await self._decide(int(command.get("frames", 3)))
        if cmd == "next":
            pend = self.log.pending()
            return {"item": next((d for d in self.log.as_dicts() if pend and d["item"] == pend[0].item), None)}
        if cmd == "arm_start":
            it = self.log.arm_start(int(command["item"]))
            return {"item": it.item, "latency_ms": round((it.arm_start - it.seen) * 1000)}
        if cmd == "log":
            return {"run": self.log.run, "log_path": str(self.log.path), "items": self.log.as_dicts()}
        if cmd == "new_run":
            return {"run": self.log.new_run(), "log_path": str(self.log.path)}
        if cmd != "objects":
            raise ValueError("command must be one of: objects, decide, next, arm_start, log, new_run")
        t0 = time.perf_counter()
        objs, frame_score, (h, w) = await self._analyze(await self._camera_image())
        return {
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "frame_score": frame_score,
            "image_size": [w, h],
            "objects": [{"label": o.label, "score": round(o.score, 4), "box": [o.x1, o.y1, o.x2, o.y2],
                         "center": list(o.center)} for o in objs],
        }


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
