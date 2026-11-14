"""Unit tests for geometry.py. Reference points were read from armfarm01 with RobotService.TransformPose."""
import os
import sys

import numpy as np
from viam.proto.common import Pose
from viam.proto.component.camera import IntrinsicParameters

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geometry import decode_depth, deproject, ov_to_matrix, part_axis_deg, plan_grasp, to_world  # noqa: E402

# cam pose in world at the time of the readings (MotionService.GetPose cam -> world); almost straight down,
# so this also covers rdk's near-pole rule.
CAM = Pose(x=407.8980501245819, y=17.481432701222815, z=332.68473145443056,
           o_x=-0.009348377958051471, o_y=0.0070423787652807174, o_z=-0.9999315040195905, theta=-91.27215821781293)
LIVE = [((0, 0, 300), (405.09353673716646, 19.594146330807035, 32.70528024855339)),
        ((100, 0, 300), (407.3103012014594, -80.3786529325736, 31.980461130663006)),
        ((37, -52, 280), (458.08565625732336, -16.38047494783993, 51.95786170194105))]
INTR = IntrinsicParameters(width_px=640, height_px=480, focal_x_px=603.7205810546875, focal_y_px=603.6961669921875,
                           center_x_px=320.1231384277344, center_y_px=239.8363800048828)


def test_to_world_matches_station_transform_pose():
    for cam_pt, world_pt in LIVE:
        assert np.allclose(to_world(CAM, np.array(cam_pt, float)), world_pt, atol=1e-6)


def test_ov_away_from_pole():
    # OV pointing along +x with theta 0: frame z -> world +x
    R = ov_to_matrix(1, 0, 0, 0)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9)


def test_decode_depth():
    arr = np.array([[300, 0], [301, 65535]], ">u2")
    data = b"DEPTHMAP" + (2).to_bytes(8, "big") + (2).to_bytes(8, "big") + arr.tobytes()
    assert decode_depth(data).tolist() == [[300, 0], [301, 65535]]


def test_deproject_centre_and_offset():
    assert np.allclose(deproject(INTR.center_x_px, INTR.center_y_px, 300, INTR), [0, 0, 300])
    assert np.allclose(deproject(INTR.center_x_px + INTR.focal_x_px, INTR.center_y_px, 300, INTR), [300, 0, 300])


def test_part_axis():
    m = np.zeros((60, 60), np.uint8)
    m[28:32, 5:55] = 1                 # horizontal bar
    assert abs(part_axis_deg(m)) < 2 or abs(abs(part_axis_deg(m)) - 180) < 2
    m = np.zeros((60, 60), np.uint8)
    np.fill_diagonal(m, 1)
    m = np.maximum(m, np.roll(m, 1, axis=1))  # bar going down-right in image coordinates
    assert abs(part_axis_deg(m) - 45) < 3
    disc = np.zeros((60, 60), np.uint8)
    yy, xx = np.mgrid[:60, :60]
    disc[(yy - 30) ** 2 + (xx - 30) ** 2 < 15 ** 2] = 1
    assert part_axis_deg(disc) is None


def test_plan_grasp_on_flat_scene():
    # Cardboard 322 mm below the camera; a 10 mm tall horizontal bar at the image centre.
    depth = np.full((480, 640), 322, np.uint16)
    depth[230:250, 280:360] = 312
    bgr = np.full((480, 640, 3), (120, 150, 180), np.uint8)
    bgr[230:250, 280:360] = 40
    g = plan_grasp([275, 225, 365, 255], bgr, depth, INTR, CAM, grasp_depth_mm=5, floor_clearance_mm=3, yaw_offset_deg=0)
    top = to_world(CAM, deproject(320, 240, 312, INTR))
    surface = to_world(CAM, deproject(320, 240, 322, INTR))
    assert abs(g.x - top[0]) < 1.5 and abs(g.y - top[1]) < 1.5
    assert abs(g.z_top - top[2]) < 0.5 and abs(g.surface_z - surface[2]) < 0.5
    assert abs(g.z_grasp - (g.z_top - 5)) < 1e-6
    # bar runs along image x = world -y here (see LIVE), so fingers close across it: gripper theta = bar yaw + 90
    bar_yaw = np.degrees(np.arctan2(*(to_world(CAM, [100, 0, 300]) - to_world(CAM, [0, 0, 300]))[1::-1]))
    want = (bar_yaw + 90 + 90) % 180 - 90
    assert abs(g.theta - want) < 3


def test_plan_grasp_never_below_surface_clearance():
    depth = np.full((480, 640), 322, np.uint16)
    depth[230:250, 280:360] = 320   # 2 mm thin washer
    bgr = np.full((480, 640, 3), (120, 150, 180), np.uint8)
    bgr[230:250, 280:360] = 40
    g = plan_grasp([275, 225, 365, 255], bgr, depth, INTR, CAM, grasp_depth_mm=5, floor_clearance_mm=3, yaw_offset_deg=0)
    assert abs(g.z_grasp - (g.surface_z + 3)) < 1e-6
