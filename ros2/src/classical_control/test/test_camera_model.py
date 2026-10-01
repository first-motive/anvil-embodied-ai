"""Behaviour tests for the fisheye camera model and horizontal-plane geometry."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.camera_model import (
    FisheyeCamera,
    intersect_plane,
    pixel_to_ray,
    pixel_to_world_on_plane,
    pose_to_matrix,
    project_points,
)
from scipy.spatial.transform import Rotation

CAMERA = FisheyeCamera(
    width=1920,
    height=1080,
    fx=900.0,
    fy=898.0,
    cx=965.0,
    cy=535.0,
    distortion=(0.05, -0.01, 0.002, -0.0005),
)


def _camera_looking_down(height: float, pitch_deg: float) -> np.ndarray:
    """Pose of an optical frame at (0, 0, height) facing world +x, pitched down."""
    pitch = np.radians(pitch_deg)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    quat = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return pose_to_matrix([0.0, 0.0, height], quat)


def test_from_dict_reads_yaml_style_mapping():
    camera = FisheyeCamera.from_dict(
        {
            "width": 1920,
            "height": 1080,
            "fx": 900,
            "fy": 898,
            "cx": 965,
            "cy": 535,
            "distortion": [0.05, -0.01, 0.002, -0.0005],
        }
    )

    assert camera == CAMERA


def test_from_dict_rejects_wrong_distortion_length():
    with pytest.raises(ValueError):
        FisheyeCamera.from_dict(
            {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0, "distortion": [0.1]}
        )


def test_scaled_halves_intrinsics():
    half = CAMERA.scaled(960, 540)

    assert (half.width, half.height) == (960, 540)
    assert (half.fx, half.fy, half.cx, half.cy) == pytest.approx((450.0, 449.0, 482.5, 267.5))
    assert half.distortion == CAMERA.distortion


def test_scaled_camera_sees_same_ray_at_scaled_pixel():
    half = CAMERA.scaled(960, 540)

    np.testing.assert_allclose(
        pixel_to_ray(half, [100.0, 400.0]), pixel_to_ray(CAMERA, [200.0, 800.0]), atol=1e-9
    )


def test_pixel_ray_pixel_round_trip_across_image():
    us, vs = np.meshgrid(np.linspace(0, 1919, 9), np.linspace(0, 1079, 7))
    pixels = np.column_stack([us.ravel(), vs.ravel()])

    rays = pixel_to_ray(CAMERA, pixels)

    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
    assert np.all(rays[:, 2] > 0)
    np.testing.assert_allclose(project_points(CAMERA, rays), pixels, atol=1e-3)


def test_principal_point_unprojects_to_optical_axis():
    np.testing.assert_allclose(pixel_to_ray(CAMERA, [965.0, 535.0]), [[0.0, 0.0, 1.0]], atol=1e-12)


def test_ray_pixel_ray_round_trip_at_wide_angles():
    angles = np.radians([5.0, 30.0, 60.0, 75.0])
    rays = np.column_stack([np.sin(angles) * 0.6, np.sin(angles) * -0.8, np.cos(angles)])

    np.testing.assert_allclose(pixel_to_ray(CAMERA, project_points(CAMERA, rays)), rays, atol=1e-6)


def test_intersect_plane_hits_expected_point():
    hit = intersect_plane(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, -1.0]), 0.0)

    np.testing.assert_allclose(hit, [1.0, 0.0, 0.0])


def test_intersect_plane_parallel_ray_returns_none():
    assert intersect_plane(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.0) is None


def test_intersect_plane_behind_origin_returns_none():
    assert intersect_plane(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]), 0.0) is None


def test_principal_point_hits_table_along_camera_axis():
    T_world_optical = _camera_looking_down(height=1.0, pitch_deg=45.0)

    hit = pixel_to_world_on_plane(CAMERA, T_world_optical, [965.0, 535.0], plane_z=0.0)

    np.testing.assert_allclose(hit, [1.0, 0.0, 0.0], atol=1e-9)


def test_off_centre_pixel_recovers_world_point_on_raised_plane():
    T_world_optical = _camera_looking_down(height=1.2, pitch_deg=50.0)
    world_point = np.array([0.7, 0.35, 0.1])
    point_optical = (np.linalg.inv(T_world_optical) @ np.append(world_point, 1.0))[:3]
    pixel = project_points(CAMERA, point_optical)[0]

    hit = pixel_to_world_on_plane(CAMERA, T_world_optical, pixel, plane_z=0.1)

    np.testing.assert_allclose(hit, world_point, atol=1e-6)


def test_pixel_above_horizon_never_reaches_table():
    T_world_optical = _camera_looking_down(height=1.0, pitch_deg=10.0)

    assert pixel_to_world_on_plane(CAMERA, T_world_optical, [965.0, 0.0], plane_z=0.0) is None
