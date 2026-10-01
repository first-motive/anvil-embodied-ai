"""Calibrate the chest camera from an ArUco marker carried by the right arm.

The arm reports its TCP pose in ``world``, so a marker taped to the hand is a set of 3D
points whose positions are known up to one fixed, unknown offset. Sweeping the hand
through a spread of positions and tilts and recording the marker's corner pixels at each
stop pins down, in one least-squares solve, the fisheye intrinsics, the camera's pose in
``world``, and the marker's offset on the hand. That replaces a tape-measured mount and a
separate checkerboard session.

Frames and conventions:

- ``world``: robot base frame, z up. ``/ee_pose_right`` is in it.
- ``tcp``: the right arm's tool centre point, as ``/ee_pose_right`` reports it.
- ``marker``: ArUco marker frame, origin at the marker centre, z out of the printed face,
  corners at (-s/2, s/2), (s/2, s/2), (s/2, -s/2), (-s/2, -s/2) in ``cv2.aruco`` order
  (top-left, top-right, bottom-right, bottom-left). This is the frame
  ``SOLVEPNP_IPPE_SQUARE`` expects.
- ``optical``: camera optical frame, x right, y down, z forward.
- ``T_a_b`` maps points in frame ``b`` into frame ``a``.

The forward model for corner ``c`` in view ``i`` is::

    pixel = fisheye_project(inv(T_world_optical) @ T_world_tcp_i @ T_tcp_marker @ c)

Pure library code: numpy, OpenCV, scipy only, no ROS imports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from classical_control.camera_model import FisheyeCamera, pose_to_matrix, project_points
from classical_control.eval_offline import matrix_to_pose
from classical_control.safety import SafetyLimiter
from classical_control.trajectory import Pose

#: Fewest views the fit accepts: 20 unknowns, and fewer stops leave the tilts too sparse
#: to separate the marker offset from the camera pose.
MIN_VIEWS = 6

#: Fewest sweep poses worth driving the arm through; below this the fit has no margin for
#: views where the marker is not detected.
MIN_SWEEP_POSES = 8

#: Smallest RMS spread of TCP orientation, about each of two independent axes, the fit
#: accepts. Without rotation about two axes the marker offset and the camera position
#: trade off freely along the missing axes and a near-zero reprojection error means nothing.
MIN_TILT_SPREAD_RAD = 0.05

#: Robust-loss scale in pixels: corners within this of the model count as inliers.
_LOSS_SCALE_PX = 2.0

#: Bound on each fisheye coefficient. Real lenses sit well inside it; the bound only stops
#: the poorly observed high-order terms from running away when views are sparse.
_MAX_ABS_K = 0.5

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@dataclass(frozen=True)
class MarkerSpec:
    """Which ArUco marker is on the hand.

    Attributes:
        dictionary: Name of a ``cv2.aruco`` predefined dictionary, e.g. ``DICT_4X4_50``.
        marker_id: Marker id within the dictionary.
        size_m: Printed side length of the black square, in metres.
    """

    dictionary: str = "DICT_4X4_50"
    marker_id: int = 0
    size_m: float = 0.05

    def __post_init__(self) -> None:
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary!r}")
        if not (math.isfinite(self.size_m) and self.size_m > 0.0):
            raise ValueError("size_m must be positive and finite")

    @property
    def corners_marker(self) -> np.ndarray:
        """(4, 3) corner positions in the marker frame, in ``cv2.aruco`` order."""
        h = self.size_m / 2.0
        return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


@dataclass(frozen=True, eq=False)
class Sample:
    """One stop of the sweep: where the hand was and where the marker corners appeared.

    Attributes:
        tcp: Measured ``/ee_pose_right`` at capture, in ``world``.
        corners: (4, 2) full-resolution corner pixels in ``cv2.aruco`` order.
    """

    tcp: Pose
    corners: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float64).reshape(-1, 2).copy()
        if corners.shape != (4, 2):
            raise ValueError(f"corners must be (4, 2), got {corners.shape}")
        # A NaN corner turns every residual of the fit into NaN; refuse it here.
        if not np.isfinite(corners).all():
            raise ValueError("corners must be finite")
        corners.setflags(write=False)
        object.__setattr__(self, "corners", corners)


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of `fit`.

    Attributes:
        camera: Fitted intrinsics at the samples' resolution.
        T_world_optical: 4x4, optical-frame points into ``world``.
        T_tcp_marker: 4x4, marker-frame points into the TCP frame.
        view_errors_px: (n_views,) mean corner reprojection error per view.
        median_error_px: Median over all corners of the reprojection error.
        success: True when the solve ran and converged to finite values.
        message: Human-readable summary or the reason for failure.
    """

    camera: FisheyeCamera
    T_world_optical: np.ndarray
    T_tcp_marker: np.ndarray
    view_errors_px: np.ndarray
    median_error_px: float
    success: bool
    message: str


def sweep_poses(
    orientation_xyzw: Sequence[float] | np.ndarray,
    limiter: SafetyLimiter,
    *,
    count: int = 15,
    center_xy: Sequence[float] = (0.28, -0.18),
    half_extent_xy: Sequence[float] = (0.10, 0.10),
    heights: Sequence[float] = (0.42, 0.50),
    max_tilt_rad: float = 0.35,
) -> list[Pose]:
    """Plan the TCP poses the arm visits while the camera watches the marker.

    Positions follow a golden-angle spiral over the ellipse ``center_xy +/- half_extent_xy``
    so any prefix of the sweep already covers the region evenly; heights cycle through
    ``heights``. Each orientation is ``orientation_xyzw`` tilted about a horizontal world
    axis whose direction also steps by the golden angle, alternating between the full and
    60% of ``max_tilt_rad``. Tilts about several axes are what make the marker's offset on
    the hand observable separately from the camera pose.

    Args:
        orientation_xyzw: Base TCP orientation, typically the mined grasp orientation.
        limiter: The task's limiter; poses outside its workspace are dropped.
        count: Number of poses to plan before the workspace check.
        center_xy: Centre of the sweep region in ``world``, metres.
        half_extent_xy: Half-widths of the region along x and y, metres.
        heights: TCP heights to cycle through, metres.
        max_tilt_rad: Largest tilt from the base orientation, radians.

    Returns:
        The poses inside the workspace, in visiting order. Deterministic.

    Raises:
        ValueError: If inputs are non-finite or fewer than `MIN_SWEEP_POSES` survive.
    """
    numbers = [*center_xy, *half_extent_xy, *heights, max_tilt_rad]
    if not np.isfinite(numbers).all() or not heights:
        raise ValueError("sweep parameters must be finite and heights non-empty")
    base = Rotation.from_quat(np.asarray(orientation_xyzw, dtype=np.float64))
    poses: list[Pose] = []
    for i in range(count):
        radius = math.sqrt((i + 0.5) / count)
        angle = i * _GOLDEN_ANGLE
        position = (
            center_xy[0] + half_extent_xy[0] * radius * math.cos(angle),
            center_xy[1] + half_extent_xy[1] * radius * math.sin(angle),
            heights[i % len(heights)],
        )
        # Offset the tilt axis from the position angle so tilt does not track position.
        axis_angle = (i + 0.5) * _GOLDEN_ANGLE * 2.0
        axis = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0])
        magnitude = max_tilt_rad if i % 2 == 0 else 0.6 * max_tilt_rad
        tilt = Rotation.from_rotvec(axis * magnitude)
        pose = Pose.from_rotation(position, tilt * base)
        if limiter.in_workspace(pose):
            poses.append(pose)
    if len(poses) < MIN_SWEEP_POSES:
        raise ValueError(
            f"only {len(poses)} of {count} sweep poses lie in the workspace; "
            f"need at least {MIN_SWEEP_POSES}"
        )
    return poses


def detect_marker(image_bgr: np.ndarray, spec: MarkerSpec) -> np.ndarray | None:
    """Find the hand marker's corners in a chest frame.

    Args:
        image_bgr: Full-resolution BGR (or single-channel) image.
        spec: The marker to look for.

    Returns:
        (4, 2) corner pixels in ``cv2.aruco`` order, or None when the marker is absent or
        appears more than once (an ambiguous detection must not enter the fit).
    """
    image = np.asarray(image_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    # OpenCV 4.7 replaced the free function with ArucoDetector; the robot runs 4.6.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None:
        return None
    matches = [c for c, marker_id in zip(corners, ids.ravel()) if marker_id == spec.marker_id]
    if len(matches) != 1:
        return None
    return np.asarray(matches[0], dtype=np.float64).reshape(4, 2)


def _matrix(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def _vector(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]])


def _camera(params: np.ndarray, width: int, height: int) -> FisheyeCamera:
    fx, fy, cx, cy, k1, k2, k3, k4 = (float(v) for v in params[:8])
    return FisheyeCamera(width, height, fx, fy, cx, cy, (k1, k2, k3, k4))


def _unpack(
    params: np.ndarray, width: int, height: int
) -> tuple[FisheyeCamera, np.ndarray, np.ndarray]:
    return (
        _camera(params, width, height),
        _matrix(params[8:11], params[11:14]),
        _matrix(params[14:17], params[17:20]),
    )


def _project(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    T_tcp_marker: np.ndarray,
    T_world_tcp: np.ndarray,
    corners_marker: np.ndarray,
) -> np.ndarray:
    """Project every view's marker corners; returns (n_views, 4, 2) pixels."""
    T_optical_world = np.linalg.inv(T_world_optical)
    corners_h = np.hstack([corners_marker, np.ones((4, 1))])
    # (n, 4, 4) chain applied to the 4 corners at once: optical <- world <- tcp <- marker.
    chain = T_optical_world @ T_world_tcp @ T_tcp_marker
    points = np.einsum("nij,kj->nki", chain, corners_h)[..., :3]
    # tradeoff: cv2.fisheye folds points behind the camera back into the image instead of
    # flagging them. A marker on the hand is always in front of the chest camera, so the
    # fit does not guard against it.
    # cv2 reads a strided view as if it were packed, so hand it a contiguous copy.
    points = np.ascontiguousarray(points.reshape(-1, 3))
    return project_points(camera, points).reshape(-1, 4, 2)


def _pnp_views(
    samples: Sequence[Sample], spec: MarkerSpec, camera: FisheyeCamera
) -> list[np.ndarray | None]:
    """Per-view T_optical_marker from the marker corners under the given intrinsics."""
    poses: list[np.ndarray | None] = []
    for sample in samples:
        # cv2.fisheye.solvePnP is missing from OpenCV 4.6, so solve on undistorted
        # normalised coordinates with an identity camera, as eval_offline does.
        normalised = cv2.fisheye.undistortPoints(
            sample.corners.reshape(-1, 1, 2), camera.camera_matrix, camera.distortion_array
        ).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(
            spec.corners_marker,
            normalised,
            np.eye(3),
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
            poses.append(None)
            continue
        poses.append(_matrix(np.ravel(rvec), np.ravel(tvec)))
    return poses


def _tilt_spread(samples: Sequence[Sample]) -> float:
    """RMS TCP rotation about the second-best-covered axis, relative to the mean orientation."""
    rotations = Rotation.from_quat(np.array([s.tcp.quat_xyzw for s in samples]))
    rotvecs = (rotations * rotations.mean().inv()).as_rotvec()
    singular = np.linalg.svd(rotvecs, compute_uv=False)
    return float(singular[1] / math.sqrt(len(samples)))


def _mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = (
        Rotation.from_matrix(np.array([t[:3, :3] for t in transforms])).mean().as_matrix()
    )
    matrix[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return matrix


def _hand_eye(T_tcp_world: list[np.ndarray], T_optical_marker: list[np.ndarray]) -> np.ndarray:
    """T_world_optical from ``cv2.calibrateHandEye``; raises when it is missing or degenerate.

    ``cv2.calibrateHandEye`` solves the eye-in-hand problem AX = XB for camera-to-gripper.
    The chest camera is eye-to-hand (camera fixed, marker on the hand), which maps onto the
    same solver by passing the *inverse* TCP poses (T_tcp_world) as its "gripper2base" and
    the PnP poses (T_optical_marker) as its "target2cam". Its "cam2gripper" output is then
    T_world_optical.
    """
    # OpenCV 5 moved hand-eye out of the main module; the robot's 4.6 still has it.
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError("cv2.calibrateHandEye unavailable")
    rotation, translation = cv2.calibrateHandEye(
        [t[:3, :3] for t in T_tcp_world],
        [t[:3, 3] for t in T_tcp_world],
        [t[:3, :3] for t in T_optical_marker],
        [t[:3, 3] for t in T_optical_marker],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.ravel(translation)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        raise RuntimeError("cv2.calibrateHandEye returned a degenerate pose")
    return matrix


def _align_points(points_optical: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Rigid T_world_optical best mapping one point set onto the other (Kabsch)."""
    mean_optical = points_optical.mean(axis=0)
    mean_world = points_world.mean(axis=0)
    covariance = (points_world - mean_world).T @ (points_optical - mean_optical)
    u, _, vt = np.linalg.svd(covariance)
    # Flip the weakest axis if needed so the result is a rotation, not a reflection.
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    matrix = np.eye(4)
    matrix[:3, :3] = u @ d @ vt
    matrix[:3, 3] = mean_world - matrix[:3, :3] @ mean_optical
    return matrix


def _initial_transforms(
    T_world_tcp: np.ndarray,
    T_optical_marker: list[np.ndarray | None],
    fallback_T_world_optical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Seed T_world_optical and T_tcp_marker for the refinement.

    Tries hand-eye first. Without it, aligns the PnP marker centres (optical frame) with
    the TCP positions (world): that ignores the marker's few-centimetre offset from the
    TCP, which the refinement then absorbs. With fewer than three PnP views, uses the
    caller's guess.
    """
    usable = [i for i, t in enumerate(T_optical_marker) if t is not None]
    if len(usable) < 3:
        T_world_optical = fallback_T_world_optical
        note = "too few PnP views, used initial extrinsic"
    else:
        observed = [T_optical_marker[i] for i in usable]
        try:
            T_world_optical = _hand_eye([np.linalg.inv(T_world_tcp[i]) for i in usable], observed)
            note = "hand-eye init"
        except (RuntimeError, cv2.error):
            T_world_optical = _align_points(
                np.array([t[:3, 3] for t in observed]), T_world_tcp[usable, :3, 3]
            )
            note = "point-alignment init (hand-eye unavailable)"
    # Each view gives its own marker offset given the camera pose; their mean is the seed.
    offsets = [
        np.linalg.inv(T_world_tcp[i]) @ T_world_optical @ T_optical_marker[i] for i in usable
    ]
    T_tcp_marker = _mean_transform(offsets) if offsets else np.eye(4)
    return T_world_optical, T_tcp_marker, note


def fit(
    samples: Sequence[Sample],
    spec: MarkerSpec,
    initial_camera: FisheyeCamera,
    initial_T_world_optical: np.ndarray,
) -> CalibrationResult:
    """Fit intrinsics, camera pose in ``world`` and marker offset on the hand.

    Seeds the two transforms with per-view PnP plus hand-eye under ``initial_camera``,
    then refines all 20 parameters on corner reprojection with a robust loss. The
    refinement runs twice: transforms only, then everything, because a bad focal-length
    guess biases the PnP depths and freeing the intrinsics against a skewed pose start can
    trade focal length for distance.

    Args:
        samples: Captured views, all at ``initial_camera``'s resolution.
        spec: The marker in the views.
        initial_camera: Starting intrinsics, e.g. the current ``camera_chest.yaml``.
        initial_T_world_optical: Starting camera pose, used only if hand-eye fails.

    Returns:
        The fit; ``success`` is False when there are too few views or the solve fails.
    """
    initial_T_world_optical = np.asarray(initial_T_world_optical, dtype=np.float64)
    if initial_T_world_optical.shape != (4, 4) or not np.isfinite(initial_T_world_optical).all():
        raise ValueError("initial_T_world_optical must be a finite 4x4 matrix")
    width, height = initial_camera.width, initial_camera.height
    n_views = len(samples)

    def failure(message: str) -> CalibrationResult:
        return CalibrationResult(
            camera=initial_camera,
            T_world_optical=initial_T_world_optical,
            T_tcp_marker=np.eye(4),
            view_errors_px=np.full(n_views, np.nan),
            median_error_px=math.nan,
            success=False,
            message=message,
        )

    if n_views < MIN_VIEWS:
        return failure(f"need at least {MIN_VIEWS} views, got {n_views}")
    spread = _tilt_spread(samples)
    if spread < MIN_TILT_SPREAD_RAD:
        return failure(
            f"TCP orientations span under two rotation axes (spread {spread:.3f} rad < "
            f"{MIN_TILT_SPREAD_RAD} rad); the marker offset is unobservable"
        )

    T_world_tcp = np.array([pose_to_matrix(s.tcp.position, s.tcp.quat_xyzw) for s in samples])
    observed = np.array([s.corners for s in samples])
    corners_marker = spec.corners_marker

    T_world_optical, T_tcp_marker, init_note = _initial_transforms(
        T_world_tcp, _pnp_views(samples, spec, initial_camera), initial_T_world_optical
    )
    intrinsics = np.array(
        [
            initial_camera.fx,
            initial_camera.fy,
            initial_camera.cx,
            initial_camera.cy,
            *initial_camera.distortion,
        ]
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        camera, world_optical, tcp_marker = _unpack(params, width, height)
        predicted = _project(camera, world_optical, tcp_marker, T_world_tcp, corners_marker)
        return (predicted - observed).ravel()

    lower = np.full(20, -np.inf)
    upper = np.full(20, np.inf)
    # Focal length within a factor of three of the guess, principal point on the sensor.
    lower[:4] = [intrinsics[0] / 3, intrinsics[1] / 3, 0.0, 0.0]
    upper[:4] = [intrinsics[0] * 3, intrinsics[1] * 3, float(width), float(height)]
    lower[4:8], upper[4:8] = -_MAX_ABS_K, _MAX_ABS_K
    intrinsics[4:8] = np.clip(intrinsics[4:8], -_MAX_ABS_K, _MAX_ABS_K)

    params = np.concatenate([intrinsics, _vector(T_world_optical), _vector(T_tcp_marker)])
    try:
        # Stage 1: settle the transforms under the fixed initial intrinsics.
        poses_only = least_squares(
            lambda pose_params: residuals(np.concatenate([intrinsics, pose_params])),
            params[8:],
            loss="soft_l1",
            f_scale=_LOSS_SCALE_PX,
        )
        params[8:] = poses_only.x
        # Stage 2: free everything.
        solution = least_squares(
            residuals,
            params,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=_LOSS_SCALE_PX,
            x_scale="jac",
        )
    except (ValueError, cv2.error) as error:
        return failure(f"least squares failed: {error}")

    camera, T_world_optical, T_tcp_marker = _unpack(solution.x, width, height)
    errors = np.linalg.norm(residuals(solution.x).reshape(n_views, 4, 2), axis=2)
    view_errors = errors.mean(axis=1)
    median_error = float(np.median(errors))
    finite = np.isfinite(solution.x).all() and math.isfinite(median_error)
    success = bool(solution.success and finite)
    message = (
        f"{init_note}; {n_views} views, median corner error {median_error:.2f} px, "
        f"worst view {float(view_errors.max()):.2f} px; solver: {solution.message}"
    )
    return CalibrationResult(
        camera=camera,
        T_world_optical=T_world_optical,
        T_tcp_marker=T_tcp_marker,
        view_errors_px=view_errors,
        median_error_px=median_error,
        success=success,
        message=message,
    )


def camera_yaml(
    result: CalibrationResult,
    parent_frame: str = "follower_body_link0",
    T_world_parent: np.ndarray | None = None,
) -> dict[str, Any]:
    """Express a fit in the ``config/camera_chest.yaml`` schema.

    Args:
        result: A calibration result.
        parent_frame: Frame the extrinsic is written relative to.
        T_world_parent: Pose of ``parent_frame`` in ``world``; identity by default, since
            ``world`` and ``follower_body_link0`` coincide on this robot.

    Returns:
        ``{"intrinsics": {...}, "extrinsic": {...}}`` with plain Python floats, ready for
        ``yaml.safe_dump``.
    """
    T_world_parent = np.eye(4) if T_world_parent is None else np.asarray(T_world_parent)
    T_parent_optical = np.linalg.inv(T_world_parent) @ result.T_world_optical
    translation, rotation = matrix_to_pose(T_parent_optical)
    camera = result.camera
    return {
        "intrinsics": {
            "width": int(camera.width),
            "height": int(camera.height),
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "distortion": [float(k) for k in camera.distortion],
        },
        "extrinsic": {
            "parent_frame": parent_frame,
            "translation_xyz": translation,
            "rotation_xyzw": rotation,
        },
    }


def save_samples(path: str | Path, samples: Sequence[Sample], spec: MarkerSpec) -> None:
    """Write samples and the marker spec to an ``.npz`` so a fit can be rerun offline.

    Args:
        path: Destination file; numpy appends ``.npz`` if missing.
        samples: Captured views.
        spec: The marker in the views.
    """
    np.savez(
        path,
        tcp_position=np.array([s.tcp.position for s in samples]).reshape(-1, 3),
        tcp_quat_xyzw=np.array([s.tcp.quat_xyzw for s in samples]).reshape(-1, 4),
        corners=np.array([s.corners for s in samples]).reshape(-1, 4, 2),
        dictionary=np.array(spec.dictionary),
        marker_id=np.array(spec.marker_id),
        size_m=np.array(spec.size_m),
    )


def load_samples(path: str | Path) -> tuple[list[Sample], MarkerSpec]:
    """Read samples written by `save_samples`.

    Args:
        path: An ``.npz`` file from `save_samples`.

    Returns:
        The samples and the marker spec.
    """
    with np.load(path, allow_pickle=False) as data:
        spec = MarkerSpec(
            dictionary=str(data["dictionary"]),
            marker_id=int(data["marker_id"]),
            size_m=float(data["size_m"]),
        )
        samples = [
            Sample(tcp=Pose(position, quat), corners=corners)
            for position, quat, corners in zip(
                data["tcp_position"], data["tcp_quat_xyzw"], data["corners"]
            )
        ]
    return samples, spec
