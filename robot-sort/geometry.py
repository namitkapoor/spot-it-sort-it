"""Camera pixel + depth -> point in the world frame, and the grasp pose for one part.

Frames (verified on armfarm01 with RobotService.TransformPose): the camera's +z axis points out of the lens (down at
inspect-pose), +x is image right, +y is image down. Depth photos are 'image/vnd.viam.dep': 'DEPTHMAP', width, height
(big-endian int64), then big-endian uint16 millimetres; 0 = no reading. Colour and depth share pixels
(align_color_depth is on in the fragment).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from viam.proto.common import Pose
from viam.proto.component.camera import IntrinsicParameters

from scoring import foreground_mask

_POLE_RADIUS = 0.0001  # rdk spatialmath orientationVectorPoleRadius


def ov_to_matrix(ox: float, oy: float, oz: float, theta_deg: float) -> np.ndarray:
    """Rotation matrix of a Viam orientation vector (degrees), same as rdk spatialmath OrientationVector.Quaternion."""
    v = np.array([ox, oy, oz], float)
    v /= np.linalg.norm(v)
    lat, lon, th = np.arccos(np.clip(v[2], -1, 1)), np.arctan2(v[1], v[0]), np.radians(theta_deg)
    if 1 - abs(v[2]) <= _POLE_RADIUS:  # near a pole rdk measures theta as if lon were 0
        th = th - lon if v[2] > 0 else th + lon

    def rz(a: float) -> np.ndarray:
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    ry = np.array([[np.cos(lat), 0, np.sin(lat)], [0, 1, 0], [-np.sin(lat), 0, np.cos(lat)]])
    return rz(lon) @ ry @ rz(th)


def to_world(frame_pose: Pose, p: np.ndarray) -> np.ndarray:
    """Point p (mm) given in a frame whose world pose is frame_pose -> the same point in world (mm)."""
    R = ov_to_matrix(frame_pose.o_x, frame_pose.o_y, frame_pose.o_z, frame_pose.theta)
    return np.array([frame_pose.x, frame_pose.y, frame_pose.z]) + R @ np.asarray(p, float)


def decode_depth(data: bytes) -> np.ndarray:
    """Viam raw depth bytes -> (height, width) uint16 array in millimetres."""
    w, h = int.from_bytes(data[8:16], "big"), int.from_bytes(data[16:24], "big")
    return np.frombuffer(data[24:24 + 2 * w * h], ">u2").reshape(h, w).astype(np.uint16)


def deproject(u: float, v: float, depth_mm: float, intr: IntrinsicParameters) -> np.ndarray:
    """Pixel (u, v) at depth_mm -> point in the camera frame (mm), pinhole model."""
    return np.array([(u - intr.center_x_px) * depth_mm / intr.focal_x_px,
                     (v - intr.center_y_px) * depth_mm / intr.focal_y_px, depth_mm], float)


def part_axis_deg(mask: np.ndarray) -> float | None:
    """Long-axis angle (degrees, image x right / y down) of the pixels in mask, or None if the blob is round."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    vals, vecs = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(float)))
    if vals[1] < 2.25 * max(vals[0], 1e-9):  # long side < 1.5x short side: nuts, washers -> any angle works
        return None
    ang = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    return (ang + 90) % 180 - 90


@dataclass
class Grasp:
    x: float          # world mm, part centre
    y: float
    z_top: float      # world z of the part's top surface
    z_grasp: float    # world z the gripper frame goes down to
    surface_z: float  # world z of the cardboard around the part
    theta: float      # gripper rotation about world z (degrees), gripper pointing straight down


def plan_grasp(box: list[int], bgr: np.ndarray, depth: np.ndarray, intr: IntrinsicParameters, cam_pose: Pose,
               grasp_depth_mm: float, floor_clearance_mm: float, yaw_offset_deg: float) -> Grasp:
    """Top-down grasp for the part in box [x1, y1, x2, y2] (pixels).

    Part top = 10th percentile of depth inside the box (closest valid pixels); surface = median depth in a 20 px ring
    around the box. The gripper goes grasp_depth_mm below the top, but never closer than floor_clearance_mm to the
    surface. Fingers close across the part's long axis; yaw_offset_deg corrects for how the fingers sit on the
    gripper frame (calibrate at the station).
    """
    x1, y1, x2, y2 = box
    h, w = depth.shape
    d = depth.astype(np.float32)
    inside = d[y1:y2, x1:x2]
    top_d = float(np.percentile(inside[inside > 0], 10))
    ring = np.zeros((h, w), bool)
    ring[max(0, y1 - 20):min(h, y2 + 20), max(0, x1 - 20):min(w, x2 + 20)] = True
    ring[y1:y2, x1:x2] = False
    surface_d = float(np.median(d[ring & (d > 0)]))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    top = to_world(cam_pose, deproject(cx, cy, top_d, intr))
    surface_z = float(to_world(cam_pose, deproject(cx, cy, surface_d, intr))[2])
    z_grasp = max(float(top[2]) - grasp_depth_mm, surface_z + floor_clearance_mm)
    axis = part_axis_deg(foreground_mask(bgr)[y1:y2, x1:x2])
    theta = yaw_offset_deg
    if axis is not None:
        a = np.radians(axis)
        p2 = to_world(cam_pose, deproject(cx + 20 * np.cos(a), cy + 20 * np.sin(a), top_d, intr))
        theta += float(np.degrees(np.arctan2(p2[1] - top[1], p2[0] - top[0]))) + 90
    theta = (theta + 90) % 180 - 90  # two-finger gripper: 180 degrees apart is the same grasp
    return Grasp(float(top[0]), float(top[1]), float(top[2]), z_grasp, surface_z, theta)
