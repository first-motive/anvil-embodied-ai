"""Behaviour tests for turning chest-camera detections into world poses."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from classical_control.camera_model import FisheyeCamera, pose_to_matrix, project_points
from classical_control.localise import (
    ChestCameraConfig,
    locate_can,
    locate_paper,
    paper_yaw,
    yaw_to_quaternion,
)
from scipy.spatial.transform import Rotation

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

CAMERA = FisheyeCamera(
    width=960,
    height=540,
    fx=425.0,
    fy=425.0,
    cx=480.0,
    cy=270.0,
    distortion=(0.03, -0.01, 0.002, -0.0005),
)
TABLE_Z = 0.22
CAN_HEIGHT = 0.12
MM = 1e-3


def _camera_above_table(pitch_deg: float = 50.0) -> np.ndarray:
    """Optical frame 0.4 m above the table at the origin, facing +x, pitched down."""
    pitch = np.radians(pitch_deg)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    quat = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return pose_to_matrix([0.0, 0.0, TABLE_Z + 0.4], quat)


T_WORLD_OPTICAL = _camera_above_table()


def _world_to_pixels(points_world: np.ndarray) -> np.ndarray:
    """Project world points through the synthetic camera, as a detector would see them."""
    T_optical_world = np.linalg.inv(T_WORLD_OPTICAL)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    optical = (T_optical_world[:3, :3] @ points.T).T + T_optical_world[:3, 3]
    return project_points(CAMERA, optical)


def _paper_corners_world(center_xy: tuple[float, float], yaw: float) -> np.ndarray:
    """Corners of an A4 sheet on the table, in cyclic order."""
    half_long, half_short = 0.297 / 2, 0.210 / 2
    local = np.array(
        [[-half_long, -half_short], [half_long, -half_short], [half_long, half_short],
         [-half_long, half_short]]
    )  # fmt: skip
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    xy = local @ rotation.T + np.asarray(center_xy)
    return np.hstack([xy, np.full((4, 1), TABLE_Z)])


def test_can_pixel_maps_back_to_its_world_centre():
    can_center = np.array([0.45, -0.08, TABLE_Z + CAN_HEIGHT / 2])
    pixel = _world_to_pixels(can_center)[0]

    located = locate_can(CAMERA, T_WORLD_OPTICAL, tuple(pixel), TABLE_Z, CAN_HEIGHT)

    assert located is not None
    np.testing.assert_allclose(located, can_center, atol=1 * MM)


def test_paper_corners_give_its_centre_and_heading():
    center_xy, yaw = (0.40, 0.06), 0.3
    corners_pixels = _world_to_pixels(_paper_corners_world(center_xy, yaw))

    located = locate_paper(CAMERA, T_WORLD_OPTICAL, corners_pixels, TABLE_Z)

    assert located is not None
    np.testing.assert_allclose(located.position, [*center_xy, TABLE_Z], atol=1 * MM)
    assert located.yaw == pytest.approx(yaw, abs=1e-3)


@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_paper_yaw_ignores_corner_order(start, reverse):
    corners = _paper_corners_world((0.0, 0.0), -0.7)[:, :2]
    if reverse:
        corners = corners[::-1]
    corners = np.roll(corners, start, axis=0)

    assert paper_yaw(corners) == pytest.approx(-0.7, abs=1e-9)


def test_paper_yaw_folds_half_turns_into_one_range():
    corners = _paper_corners_world((0.0, 0.0), 0.2 + math.pi)[:, :2]

    assert paper_yaw(corners) == pytest.approx(0.2, abs=1e-9)


def test_missing_detections_give_none():
    assert locate_can(CAMERA, T_WORLD_OPTICAL, None, TABLE_Z, CAN_HEIGHT) is None
    assert locate_paper(CAMERA, T_WORLD_OPTICAL, None, TABLE_Z) is None


def test_pixel_above_the_horizon_gives_none():
    # A level camera's top-centre pixel looks upward, so its ray never meets the table.
    level = _camera_above_table(pitch_deg=0.0)

    assert locate_can(CAMERA, level, (480.0, 0.0), TABLE_Z, CAN_HEIGHT) is None


def test_yaw_to_quaternion_matches_scipy():
    expected = Rotation.from_euler("z", 0.9).as_quat()

    np.testing.assert_allclose(yaw_to_quaternion(0.9), expected, atol=1e-12)


def test_shipped_chest_config_loads():
    config = ChestCameraConfig.from_yaml(CONFIG_DIR / "camera_chest.yaml")

    assert (config.camera.width, config.camera.height) == (1920, 1080)
    assert config.parent_frame == "follower_body_link0"
    assert np.linalg.norm(config.rotation_xyzw) == pytest.approx(1.0, abs=1e-6)
