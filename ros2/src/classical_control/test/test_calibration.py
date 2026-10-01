"""Behaviour tests for the arm-driven chest camera calibration solver."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml
from classical_control import calibration
from classical_control.calibration import (
    MarkerSpec,
    Sample,
    camera_yaml,
    detect_marker,
    fit,
    load_samples,
    save_samples,
    sweep_poses,
)
from classical_control.camera_model import FisheyeCamera, pose_to_matrix, project_points
from classical_control.localise import ChestCameraConfig
from classical_control.safety import SafetyLimiter, SafetyLimits
from classical_control.trajectory import Pose
from scipy.spatial.transform import Rotation

#: Grasp orientation measured at home in task.yaml.
GRASP_XYZW = (0.672, 0.064, 0.736, 0.055)

TRUE_CAMERA = FisheyeCamera(
    width=1920,
    height=1080,
    fx=700.0,
    fy=705.0,
    cx=950.0,
    cy=560.0,
    distortion=(0.05, -0.02, 0.01, -0.003),
)
#: The guess camera_chest.yaml ships with: f = 850, no distortion, image centre.
GUESS_CAMERA = FisheyeCamera(1920, 1080, 850.0, 850.0, 960.0, 540.0, (0.0, 0.0, 0.0, 0.0))
#: camera_chest.yaml's placeholder mount, 0.35 m up, pitched 45 degrees.
GUESS_T_WORLD_OPTICAL = pose_to_matrix(
    (0.0, 0.0, 0.35), (-0.65328148, 0.65328148, -0.27059805, 0.27059805)
)
SPEC = MarkerSpec()


def _limiter() -> SafetyLimiter:
    return SafetyLimiter(
        SafetyLimits(
            workspace_min_m=(0.05, -0.55, 0.0),
            workspace_max_m=(0.65, 0.15, 0.70),
            table_z=0.28,
        )
    )


def _looking_forward(position: tuple[float, float, float], pitch_deg: float) -> np.ndarray:
    """Optical frame at `position` facing world +x, pitched down by `pitch_deg`."""
    pitch = np.radians(pitch_deg)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, down, forward])
    matrix[:3, 3] = position
    return matrix


TRUE_T_WORLD_OPTICAL = _looking_forward((0.05, 0.02, 0.62), 44.0)


def _true_tcp_marker() -> np.ndarray:
    """Marker a few cm off the TCP, its face turned toward the camera at the base pose."""
    base = Rotation.from_quat(GRASP_XYZW)
    # Marker z (out of the face) points back along the camera's forward axis.
    facing = TRUE_T_WORLD_OPTICAL[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    rotation = base.inv() * Rotation.from_matrix(facing) * Rotation.from_rotvec([0.1, -0.05, 0.2])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = (0.03, -0.02, 0.04)
    return matrix


TRUE_T_TCP_MARKER = _true_tcp_marker()


def _samples(poses: list[Pose], noise_px: float = 0.3, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    T_optical_world = np.linalg.inv(TRUE_T_WORLD_OPTICAL)
    corners_h = np.hstack([SPEC.corners_marker, np.ones((4, 1))])
    samples = []
    for pose in poses:
        T = T_optical_world @ pose_to_matrix(pose.position, pose.quat_xyzw) @ TRUE_T_TCP_MARKER
        pixels = project_points(TRUE_CAMERA, np.ascontiguousarray((corners_h @ T.T)[:, :3]))
        samples.append(Sample(pose, pixels + rng.normal(0.0, noise_px, pixels.shape)))
    return samples


def _rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


@pytest.fixture(scope="module")
def sweep() -> list[Pose]:
    return sweep_poses(GRASP_XYZW, _limiter())


@pytest.fixture(scope="module")
def recovered(sweep: list[Pose]):
    return fit(_samples(sweep), SPEC, GUESS_CAMERA, GUESS_T_WORLD_OPTICAL)


# --- sweep_poses ---


def test_sweep_stays_in_workspace_with_requested_count(sweep: list[Pose]) -> None:
    limiter = _limiter()
    assert len(sweep) == 15
    assert all(limiter.in_workspace(pose) for pose in sweep)


def test_sweep_tilts_about_at_least_two_axes(sweep: list[Pose]) -> None:
    base = Rotation.from_quat(GRASP_XYZW)
    tilts = [(pose.rotation * base.inv()).as_rotvec() for pose in sweep]
    axes = np.array([t / np.linalg.norm(t) for t in tilts])
    assert np.linalg.matrix_rank(axes, tol=0.2) >= 2
    assert max(np.linalg.norm(t) for t in tilts) <= 0.35 + 1e-9


def test_sweep_is_deterministic(sweep: list[Pose]) -> None:
    again = sweep_poses(GRASP_XYZW, _limiter())
    for a, b in zip(sweep, again):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.quat_xyzw, b.quat_xyzw)


def test_sweep_refuses_a_region_mostly_outside_the_workspace() -> None:
    with pytest.raises(ValueError, match="workspace"):
        sweep_poses(GRASP_XYZW, _limiter(), center_xy=(0.70, -0.18))


# --- fit ---


def test_fit_recovers_extrinsic_from_the_shipped_guess(recovered) -> None:
    assert recovered.success, recovered.message
    translation_error = np.linalg.norm(
        recovered.T_world_optical[:3, 3] - TRUE_T_WORLD_OPTICAL[:3, 3]
    )
    assert translation_error < 0.01
    assert _rotation_error_deg(recovered.T_world_optical, TRUE_T_WORLD_OPTICAL) < 1.0
    assert recovered.median_error_px < 1.0


def test_fit_recovers_marker_offset_and_focal_length(recovered) -> None:
    offset_error = np.linalg.norm(recovered.T_tcp_marker[:3, 3] - TRUE_T_TCP_MARKER[:3, 3])
    assert offset_error < 0.01
    assert _rotation_error_deg(recovered.T_tcp_marker, TRUE_T_TCP_MARKER) < 2.0
    assert recovered.camera.fx == pytest.approx(TRUE_CAMERA.fx, rel=0.05)
    assert recovered.camera.fy == pytest.approx(TRUE_CAMERA.fy, rel=0.05)


def test_fit_reports_one_error_per_view(recovered, sweep: list[Pose]) -> None:
    assert recovered.view_errors_px.shape == (len(sweep),)
    assert np.all(recovered.view_errors_px < 2.0)


def test_fit_with_too_few_views_fails_cleanly(sweep: list[Pose]) -> None:
    result = fit(_samples(sweep[:5]), SPEC, GUESS_CAMERA, GUESS_T_WORLD_OPTICAL)
    assert not result.success
    assert "at least" in result.message


def test_fit_refuses_a_sweep_without_tilt() -> None:
    # Pure translation leaves the marker offset and camera position free to trade off.
    flat = sweep_poses(GRASP_XYZW, _limiter(), max_tilt_rad=0.0)
    result = fit(_samples(flat), SPEC, GUESS_CAMERA, GUESS_T_WORLD_OPTICAL)
    assert not result.success
    assert "unobservable" in result.message


def test_fit_recovers_without_hand_eye(sweep: list[Pose], monkeypatch) -> None:
    # OpenCV 5 dropped cv2.calibrateHandEye; the point-alignment seed must carry the fit.
    def unavailable(*_args):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(calibration, "_hand_eye", unavailable)
    result = fit(_samples(sweep), SPEC, GUESS_CAMERA, GUESS_T_WORLD_OPTICAL)
    assert result.success, result.message
    assert "point-alignment" in result.message
    error = np.linalg.norm(result.T_world_optical[:3, 3] - TRUE_T_WORLD_OPTICAL[:3, 3])
    assert error < 0.01


@pytest.mark.skipif(not hasattr(cv2, "calibrateHandEye"), reason="OpenCV build lacks hand-eye")
def test_hand_eye_seed_uses_the_eye_to_hand_inversion(sweep: list[Pose]) -> None:
    samples = _samples(sweep, noise_px=0.0)
    views = calibration._pnp_views(samples, SPEC, TRUE_CAMERA)
    T_tcp_world = [np.linalg.inv(pose_to_matrix(s.tcp.position, s.tcp.quat_xyzw)) for s in samples]
    seed = calibration._hand_eye(T_tcp_world, views)
    assert np.linalg.norm(seed[:3, 3] - TRUE_T_WORLD_OPTICAL[:3, 3]) < 0.01
    assert _rotation_error_deg(seed, TRUE_T_WORLD_OPTICAL) < 1.0


def test_fit_rejects_a_non_finite_initial_extrinsic(sweep: list[Pose]) -> None:
    with pytest.raises(ValueError, match="finite"):
        fit(_samples(sweep), SPEC, GUESS_CAMERA, np.full((4, 4), np.nan))


def test_sample_rejects_non_finite_corners(sweep: list[Pose]) -> None:
    corners = np.zeros((4, 2))
    corners[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Sample(sweep[0], corners)


# --- detect_marker ---


def _marker_image(spec: MarkerSpec, side_px: int = 200) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, spec.marker_id, side_px)
    else:  # OpenCV 4.6
        marker = cv2.aruco.drawMarker(dictionary, spec.marker_id, side_px)
    image = np.full((720, 1280, 3), 128, dtype=np.uint8)
    # A white quiet zone around the marker, as on a printed sheet.
    image[180:520, 380:720] = 255
    image[250:450, 450:650] = marker[..., None]
    return image


def test_detect_marker_finds_corners_in_aruco_order() -> None:
    corners = detect_marker(_marker_image(SPEC), SPEC)
    assert corners is not None
    expected = np.array([[450, 250], [650, 250], [650, 450], [450, 450]], dtype=float)
    np.testing.assert_allclose(corners, expected, atol=2.0)


def test_detect_marker_returns_none_without_a_marker() -> None:
    assert detect_marker(np.full((720, 1280, 3), 128, dtype=np.uint8), SPEC) is None


def test_detect_marker_ignores_other_ids() -> None:
    assert detect_marker(_marker_image(SPEC), MarkerSpec(marker_id=1)) is None


# --- camera_yaml and persistence ---


def test_camera_yaml_round_trips_through_the_chest_config(recovered, tmp_path) -> None:
    path = tmp_path / "camera_chest.yaml"
    path.write_text(yaml.safe_dump(camera_yaml(recovered)), encoding="utf-8")
    config = ChestCameraConfig.from_yaml(path)
    assert config.parent_frame == "follower_body_link0"
    assert config.camera == recovered.camera
    np.testing.assert_allclose(
        pose_to_matrix(config.translation_xyz, config.rotation_xyzw),
        recovered.T_world_optical,
        atol=1e-9,
    )


def test_camera_yaml_expresses_extrinsic_in_the_parent_frame(recovered) -> None:
    T_world_parent = pose_to_matrix((0.1, -0.2, 0.3), Rotation.from_euler("z", 30, True).as_quat())
    extrinsic = camera_yaml(recovered, "base", T_world_parent)["extrinsic"]
    T_parent_optical = pose_to_matrix(extrinsic["translation_xyz"], extrinsic["rotation_xyzw"])
    np.testing.assert_allclose(
        T_world_parent @ T_parent_optical, recovered.T_world_optical, atol=1e-9
    )


def test_samples_round_trip_through_npz(sweep: list[Pose], tmp_path) -> None:
    samples = _samples(sweep)
    spec = MarkerSpec(marker_id=1, size_m=0.04)
    path = tmp_path / "samples.npz"
    save_samples(path, samples, spec)
    loaded, loaded_spec = load_samples(path)
    assert loaded_spec == spec
    assert len(loaded) == len(samples)
    for a, b in zip(samples, loaded):
        np.testing.assert_array_equal(a.corners, b.corners)
        np.testing.assert_array_equal(a.tcp.position, b.tcp.position)
        np.testing.assert_array_equal(a.tcp.quat_xyzw, b.tcp.quat_xyzw)
