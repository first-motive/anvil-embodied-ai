"""Fisheye camera model and horizontal-plane geometry for look-then-move localisation.

The chest camera is fixed, so an object resting on a known horizontal surface can be
localised from a single pixel: unproject the pixel to a ray in the camera optical frame,
carry the ray into the robot ``world`` frame (z up), and intersect it with the plane
``z = plane_z``. This module holds only that geometry so detectors and nodes can share it
without pulling in ROS.

Frames follow the ROS optical convention: x right, y down, z forward. Distortion uses the
OpenCV fisheye (equidistant) model with coefficients k1..k4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_PARALLEL_EPSILON = 1e-9


@dataclass(frozen=True)
class FisheyeCamera:
    """Intrinsics of a fisheye camera at a given image resolution.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        distortion: Fisheye coefficients (k1, k2, k3, k4) for ``cv2.fisheye``.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FisheyeCamera:
        """Build a camera from a mapping such as a parsed yaml block.

        Args:
            d: Mapping with keys width, height, fx, fy, cx, cy, distortion.

        Returns:
            The camera described by ``d``.
        """
        distortion = tuple(float(k) for k in d["distortion"])
        if len(distortion) != 4:
            raise ValueError(f"fisheye distortion needs 4 coefficients, got {len(distortion)}")
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            distortion=distortion,  # type: ignore[arg-type]
        )

    def scaled(self, width: int, height: int) -> FisheyeCamera:
        """Return the intrinsics for the same lens on a resized image.

        Fisheye distortion acts on normalised angles, so only the pixel-unit terms scale.

        Args:
            width: Resized image width in pixels.
            height: Resized image height in pixels.

        Returns:
            A camera whose intrinsics match the resized image.
        """
        sx = width / self.width
        sy = height / self.height
        # tradeoff: scales the principal point directly, ignoring the half-pixel centre
        # shift ((c + 0.5) * s - 0.5); the error is under one pixel at any downscale.
        return FisheyeCamera(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            distortion=self.distortion,
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion_array(self) -> np.ndarray:
        """Distortion coefficients as the (4,) float64 array ``cv2.fisheye`` expects."""
        return np.asarray(self.distortion, dtype=np.float64)


def pixel_to_ray(camera: FisheyeCamera, pixels: Sequence[float] | np.ndarray) -> np.ndarray:
    """Unproject pixels to unit rays in the camera optical frame.

    Args:
        camera: Camera intrinsics matching the image the pixels came from.
        pixels: A single (u, v) pixel or an (N, 2) array of pixels.

    Returns:
        (N, 3) unit direction vectors (x right, y down, z forward).

    Note:
        ``cv2.fisheye.undistortPoints`` returns normalised image coordinates, which only
        exist for rays less than 90 degrees off the optical axis; pixels beyond that
        unproject to meaningless rays. A table-facing camera never needs them.
    """
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    normalised = cv2.fisheye.undistortPoints(
        points, camera.camera_matrix, camera.distortion_array
    ).reshape(-1, 2)
    rays = np.hstack([normalised, np.ones((len(normalised), 1))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def project_points(camera: FisheyeCamera, points_optical: np.ndarray) -> np.ndarray:
    """Project 3D points in the camera optical frame to pixels.

    Args:
        camera: Camera intrinsics.
        points_optical: A single (x, y, z) point or an (N, 3) array, in the optical frame.

    Returns:
        (N, 2) pixel coordinates.
    """
    points = np.asarray(points_optical, dtype=np.float64).reshape(-1, 1, 3)
    pixels, _ = cv2.fisheye.projectPoints(
        points, np.zeros(3), np.zeros(3), camera.camera_matrix, camera.distortion_array
    )
    return pixels.reshape(-1, 2)


def pose_to_matrix(position_xyz: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a position and quaternion.

    Args:
        position_xyz: Translation (x, y, z).
        quat_xyzw: Rotation as a quaternion in (x, y, z, w) order, the ROS convention.

    Returns:
        4x4 transform mapping child-frame points into the parent frame.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    matrix[:3, 3] = position_xyz
    return matrix


def intersect_plane(origin: np.ndarray, direction: np.ndarray, plane_z: float) -> np.ndarray | None:
    """Intersect a ray with the horizontal plane ``z = plane_z``.

    Args:
        origin: Ray origin (x, y, z) in the plane's frame.
        direction: Ray direction (x, y, z); need not be unit length.
        plane_z: Height of the plane.

    Returns:
        The (3,) hit point, or None when the ray is parallel to the plane or the plane
        lies behind the origin.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if abs(direction[2]) < _PARALLEL_EPSILON:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0.0:
        return None
    return origin + t * direction


def pixel_to_world_on_plane(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    pixel: Sequence[float] | np.ndarray,
    plane_z: float,
) -> np.ndarray | None:
    """Locate the world point on ``z = plane_z`` that a single pixel sees.

    Args:
        camera: Camera intrinsics matching the image the pixel came from.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        pixel: One (u, v) pixel.
        plane_z: Height of the plane in ``world``.

    Returns:
        The (3,) world point, or None when the pixel's ray never reaches the plane.
    """
    ray_optical = pixel_to_ray(camera, pixel)[0]
    direction_world = T_world_optical[:3, :3] @ ray_optical
    return intersect_plane(T_world_optical[:3, 3], direction_world, plane_z)
