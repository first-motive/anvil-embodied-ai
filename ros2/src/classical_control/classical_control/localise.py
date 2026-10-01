"""Turn chest-camera pixel detections into world poses for the can and the paper.

`detector.detect` reports where the can and paper are in the image; the task node needs
where they are on the table. Both rest on the table, whose height in ``world`` is known,
so each pixel is carried onto a horizontal plane with `camera_model`: the paper lies on
the table top, and the can's blob centroid is taken to sit at half the can's height.

The paper's heading is measured from its corners after they are mapped onto the table,
not in the image, because the fisheye lens bends straight edges and foreshortens depth.

Pure library code: numpy, scipy, and PyYAML only, no ROS imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from classical_control.camera_model import FisheyeCamera, pixel_to_world_on_plane


@dataclass(frozen=True)
class ChestCameraConfig:
    """Chest camera intrinsics plus its fixed mounting on the robot body.

    Attributes:
        camera: Intrinsics at the camera's native resolution.
        parent_frame: Robot frame the camera is bolted to.
        translation_xyz: Optical-frame origin in ``parent_frame``, in metres.
        rotation_xyzw: Optical-frame orientation in ``parent_frame``, as a quaternion.
    """

    camera: FisheyeCamera
    parent_frame: str
    translation_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChestCameraConfig:
        """Load the config from a yaml file with ``intrinsics`` and ``extrinsic`` blocks.

        Args:
            path: Path to a file shaped like ``config/camera_chest.yaml``.

        Returns:
            The parsed config.
        """
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        extrinsic = data["extrinsic"]
        translation = tuple(float(v) for v in extrinsic["translation_xyz"])
        rotation = tuple(float(v) for v in extrinsic["rotation_xyzw"])
        if len(translation) != 3 or len(rotation) != 4:
            raise ValueError("extrinsic needs translation_xyz[3] and rotation_xyzw[4]")
        return cls(
            camera=FisheyeCamera.from_dict(data["intrinsics"]),
            parent_frame=str(extrinsic["parent_frame"]),
            translation_xyz=translation,  # type: ignore[arg-type]
            rotation_xyzw=rotation,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PaperPose:
    """Where the paper lies on the table.

    Attributes:
        position: (3,) centre of the paper in ``world``, in metres.
        yaw: Heading of the paper's long edge about world z, in radians, in [-pi/2, pi/2).
    """

    position: np.ndarray
    yaw: float


def locate_can(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    can_pixel: Sequence[float] | None,
    table_z: float,
    can_height: float,
) -> np.ndarray | None:
    """Place the can's blob centroid on the plane halfway up the can.

    Args:
        camera: Intrinsics matching the image the pixel came from.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        can_pixel: Can centroid as (u, v) pixels, or None when no can was seen.
        table_z: Height of the table top in ``world``, in metres.
        can_height: Height of an upright can, in metres.

    Returns:
        The (3,) can centre in ``world``, or None when there is no can or its ray misses.
    """
    if can_pixel is None:
        return None
    # tradeoff: a can lying on its side has its centre lower than can_height / 2, which
    # shifts the estimate a few millimetres along the viewing ray; the grasp tolerates it.
    return pixel_to_world_on_plane(camera, T_world_optical, can_pixel, table_z + can_height / 2)


def locate_paper(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    paper_corners: np.ndarray | None,
    table_z: float,
) -> PaperPose | None:
    """Map the paper's corners onto the table and derive its centre and heading.

    Args:
        camera: Intrinsics matching the image the corners came from.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        paper_corners: (4, 2) corner pixels in cyclic order, or None when no paper was seen.
        table_z: Height of the table top in ``world``, in metres.

    Returns:
        The paper pose, or None when there is no paper or a corner's ray misses the table.
    """
    if paper_corners is None:
        return None
    corners_world = []
    for pixel in np.asarray(paper_corners, dtype=np.float64).reshape(4, 2):
        point = pixel_to_world_on_plane(camera, T_world_optical, pixel, table_z)
        if point is None:
            return None
        corners_world.append(point)
    corners = np.asarray(corners_world)
    # The mean of the table-plane corners is the true centre; the detector's pixel centre
    # is not, because the fisheye maps equal table distances to unequal pixel distances.
    return PaperPose(position=corners.mean(axis=0), yaw=paper_yaw(corners[:, :2]))


def paper_yaw(corners_xy: np.ndarray) -> float:
    """Return the heading of a rectangle's long edge about z.

    Opposite edges are averaged so a slightly skewed quad still gives one heading. A
    rectangle looks the same turned half a turn, so the result is folded into
    [-pi/2, pi/2).

    Args:
        corners_xy: (4, 2) corners in cyclic order (either direction, any start).

    Returns:
        Yaw in radians.
    """
    c = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    edge_a = ((c[1] - c[0]) + (c[2] - c[3])) / 2
    edge_b = ((c[2] - c[1]) + (c[3] - c[0])) / 2
    long_edge = edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
    yaw = math.atan2(long_edge[1], long_edge[0])
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Return the (x, y, z, w) quaternion for a rotation of ``yaw`` radians about z."""
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
