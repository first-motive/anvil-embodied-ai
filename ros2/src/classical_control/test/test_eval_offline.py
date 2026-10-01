"""Tests for the offline perception evaluation: stats, geometry, PnP fit and the CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from classical_control import eval_offline
from classical_control.camera_model import FisheyeCamera, pose_to_matrix
from classical_control.eval_offline import (
    camera_in_world,
    error_stats,
    fit_extrinsic,
    locate_episodes,
    matrix_to_pose,
    mean_bias,
    median_image,
    project_world_points,
    read_ground_truth,
    xy_errors,
)
from classical_control.localise import ChestCameraConfig
from scipy.spatial.transform import Rotation

CAMERA = FisheyeCamera(1920, 1080, 700.0, 700.0, 955.0, 545.0, (0.05, -0.01, 0.002, -0.0005))


def _chest_pose() -> np.ndarray:
    """A chest camera 0.5 m above the table, behind it, pitched down at the workspace."""
    # Optical z forward along world +x, x right along world -y, y down along world -z,
    # then pitched 50 degrees down about the optical x axis.
    level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pitch = Rotation.from_euler("x", -50.0, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = level @ pitch
    T[:3, 3] = (0.05, -0.02, 0.75)
    return T


def _workspace_points(rng: np.random.Generator, count: int) -> np.ndarray:
    """Can-like points at grasp height and paper-like points on the table."""
    xy = np.column_stack([rng.uniform(0.25, 0.65, count), rng.uniform(-0.3, 0.3, count)])
    z = np.where(np.arange(count) % 2 == 0, 0.345, 0.285)
    return np.column_stack([xy, z])


def test_matrix_to_pose_round_trips_pose_to_matrix() -> None:
    quat = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_quat()
    translation, rotation = matrix_to_pose(pose_to_matrix([0.1, -0.2, 0.3], quat))
    assert translation == pytest.approx([0.1, -0.2, 0.3])
    assert rotation[3] >= 0.0
    assert abs(np.dot(rotation, quat)) == pytest.approx(1.0)


def test_project_world_points_marks_points_behind_camera_nan() -> None:
    T = np.eye(4)
    pixels = project_world_points(CAMERA, T, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert pixels[0] == pytest.approx([CAMERA.cx, CAMERA.cy])
    assert np.isnan(pixels[1]).all()


def test_locate_episodes_inverts_projection_and_keeps_missing_rows_nan() -> None:
    T = _chest_pose()
    rng = np.random.default_rng(0)
    can = _workspace_points(rng, 6)
    can[:, 2] = 0.28  # table_z 0.22 + can_height 0.12 / 2
    can_px = project_world_points(CAMERA, T, can)
    can_px[3] = np.nan
    paper_center = np.array([0.45, 0.05, 0.22])
    square = np.array([[-0.1, -0.15, 0.0], [0.1, -0.15, 0.0], [0.1, 0.15, 0.0], [-0.1, 0.15, 0.0]])
    corners_px = project_world_points(CAMERA, T, paper_center + square)
    paper_corners = np.repeat(corners_px[None], 6, axis=0)
    paper_corners[1, 2] = np.nan

    can_world, paper_world, paper_yaw = locate_episodes(
        CAMERA, T, can_px, paper_corners, table_z=0.22, can_height=0.12
    )

    keep = np.arange(6) != 3
    np.testing.assert_allclose(can_world[keep], can[keep], atol=1e-6)
    assert np.isnan(can_world[3]).all()
    np.testing.assert_allclose(paper_world[0], paper_center, atol=1e-6)
    assert np.isnan(paper_world[1]).all() and np.isnan(paper_yaw[1])
    assert abs(paper_yaw[0]) == pytest.approx(np.pi / 2)  # long edge along world y


def _config(parent_frame: str) -> ChestCameraConfig:
    quat = Rotation.from_euler("z", 30, degrees=True).as_quat()
    return ChestCameraConfig(CAMERA, parent_frame, (0.1, 0.0, 0.3), tuple(quat))


@pytest.fixture
def warnings_logged(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record eval_offline warnings directly; ROS test images may stop log propagation."""
    logged: list[str] = []
    monkeypatch.setattr(
        eval_offline.logger, "warning", lambda msg, *args: logged.append(msg % args)
    )
    return logged


def test_camera_in_world_composes_parent_pose(warnings_logged: list[str]) -> None:
    parent_to_world = (0.0, 0.2, 0.5, *Rotation.from_euler("z", 90, degrees=True).as_quat())
    T = camera_in_world(_config("follower_body_link0"), parent_to_world)
    assert not warnings_logged
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.3, 0.8], atol=1e-9)
    yaw = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)[2]
    assert yaw == pytest.approx(120.0)


def test_camera_in_world_warns_when_identity_hides_a_parent_frame(
    warnings_logged: list[str],
) -> None:
    camera_in_world(_config("world"))
    assert not warnings_logged
    T = camera_in_world(_config("follower_body_link0"))
    assert len(warnings_logged) == 1
    assert "follower_body_link0" in warnings_logged[0]
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.0, 0.3])


def test_error_stats_ignores_nan() -> None:
    stats = error_stats(np.array([0.01, 0.02, np.nan, 0.03, 0.04]))
    assert stats["n"] == 4
    assert stats["median"] == pytest.approx(0.025)
    assert stats["p90"] == pytest.approx(0.037)
    assert stats["mean"] == pytest.approx(0.025)
    assert stats["max"] == pytest.approx(0.04)


def test_error_stats_of_nothing_is_empty() -> None:
    assert error_stats(np.array([np.nan])) == {
        "n": 0,
        "median": None,
        "p90": None,
        "mean": None,
        "max": None,
    }


def test_xy_errors_and_bias_recover_a_constant_offset() -> None:
    truth = np.array([[0.4, 0.0, 0.3], [0.5, 0.1, 0.3], [0.6, -0.1, 0.3], [0.5, 0.0, 0.3]])
    estimate = truth[:, :2] - np.array([0.03, -0.04])
    estimate[3] = np.nan
    errors = xy_errors(estimate, truth)
    np.testing.assert_allclose(errors[:3], 0.05)
    assert np.isnan(errors[3])
    assert mean_bias(estimate, truth) == pytest.approx([0.03, -0.04])
    assert mean_bias(np.full((2, 2), np.nan), truth[:2]) is None


def test_fit_extrinsic_recovers_known_pose_despite_outliers() -> None:
    rng = np.random.default_rng(1)
    T_true = _chest_pose()
    world = _workspace_points(rng, 80)
    pixels = project_world_points(CAMERA, T_true, world) + rng.normal(0.0, 0.5, (80, 2))
    outliers = np.arange(0, 80, 10)
    pixels[outliers] += rng.uniform(80.0, 200.0, (len(outliers), 2))

    fit = fit_extrinsic(CAMERA, pixels, world, ransac_px=5.0)

    assert fit is not None
    assert not fit.inliers[outliers].any()
    assert fit.inliers.sum() >= 70
    np.testing.assert_allclose(fit.T_world_optical[:3, 3], T_true[:3, 3], atol=0.005)
    angle = Rotation.from_matrix(fit.T_world_optical[:3, :3].T @ T_true[:3, :3]).magnitude()
    assert np.degrees(angle) < 0.2
    assert np.median(fit.reprojection_px[fit.inliers]) < 1.0


def test_fit_extrinsic_needs_six_points() -> None:
    world = _workspace_points(np.random.default_rng(2), 5)
    assert fit_extrinsic(CAMERA, project_world_points(CAMERA, _chest_pose(), world), world) is None


def test_median_image_drops_a_transient_object() -> None:
    frames = np.full((5, 70, 4, 3), 100, dtype=np.uint8)
    frames[1, 10:20] = 255
    frames[3, 40:60] = 0
    frames[:, 65:] = np.arange(5, dtype=np.uint8)[:, None, None, None] * 10
    median = median_image(frames)
    assert median.dtype == np.uint8
    assert (median[:65] == 100).all()
    assert (median[65:] == 20).all()


def test_read_ground_truth_blanks_become_none(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.csv"
    path.write_text(
        "episode,close_t,open_t,can_x,can_y,can_z,paper_x,paper_y,paper_z\n"
        "0001,1.0,2.0,0.4,0.1,0.345,0.5,-0.1,0.35\n"
        "0002,1.5,,0.4,0.1,0.345,,,\n"
    )
    truth = read_ground_truth(path)
    assert truth["0001"]["paper_y"] == pytest.approx(-0.1)
    assert truth["0002"]["paper_x"] is None


def test_main_writes_report_and_csv_from_fixture_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    fixtures = Path(__file__).parent / "fixtures"
    frames = {name: cv2.imread(str(fixtures / f"chest_{name}.jpg")) for name in ("0001", "0015")}
    recordings = tmp_path / "recordings"
    for name in frames:
        (recordings / name).mkdir(parents=True)
    monkeypatch.setattr(eval_offline, "read_first_frame", lambda d: frames[d.name])
    camera_yaml = tmp_path / "camera.yaml"
    camera_yaml.write_text(
        yaml.safe_dump(
            {
                "intrinsics": {
                    "width": 1920,
                    "height": 1080,
                    "fx": 700.0,
                    "fy": 700.0,
                    "cx": 960.0,
                    "cy": 540.0,
                    "distortion": [0.0, 0.0, 0.0, 0.0],
                },
                "extrinsic": {
                    "parent_frame": "follower_body_link0",
                    "translation_xyz": [0.0, 0.0, 0.5],
                    "rotation_xyzw": [1.0, 0.0, 0.0, 0.0],
                },
            }
        )
    )
    truth = tmp_path / "ground_truth.csv"
    truth.write_text(
        "episode,close_t,open_t,can_x,can_y,can_z,paper_x,paper_y,paper_z\n"
        "0001,1.0,2.0,0.4,0.1,0.345,0.5,-0.1,0.35\n"
    )
    out = tmp_path / "out"

    eval_offline.main(
        [
            f"--recordings={recordings}",
            f"--ground-truth={truth}",
            f"--camera-yaml={camera_yaml}",
            f"--out-dir={out}",
            f"--save-background={out / 'background.jpg'}",
            "--scale=0.25",
            "--parent-to-world",
            *("0", "0", "0.3", "0", "0", "0", "1"),
        ]
    )

    report = yaml.safe_load((out / "eval_report.yaml").read_text())
    assert report["detection"]["episodes"] == 2
    assert report["detection"]["with_ground_truth"] == 1
    assert report["inputs"]["camera_parent_frame"] == "follower_body_link0"
    assert report["inputs"]["parent_to_world"][2] == pytest.approx(0.3)
    assert report["fitted_extrinsic"] is None  # two episodes cannot pin six PnP points
    assert (out / "background.jpg").exists()
    assert len((out / "eval_episodes.csv").read_text().splitlines()) == 3
