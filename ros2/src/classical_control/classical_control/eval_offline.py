"""Score the chest-camera perception against the teleop ground truth, and refit the extrinsic.

The classical baseline localises the can and the paper from one chest frame: detect pixels,
cast their rays through the camera yaml extrinsic, intersect them with planes of known height
(``localise``). Before trusting that on the robot, this tool replays the recorded episodes
through exactly that chain, the perception node's own loader and localisers, and compares every estimate with where the gripper actually closed on the can and
released it over the paper (``ground_truth.csv`` from ``episode_miner``).

Per episode it reads only the first chest frame, which shows the scene before the arm moves.
The per-pixel median of those frames is an empty-table background (the can moves between
episodes, so the median drops it) and can be saved for the perception node. The report gives
detection rates, xy error and its mean bias under the yaml extrinsic, then fits the extrinsic
itself by RANSAC PnP on (detected pixel, ground-truth point) pairs and scores again with it.

The camera yaml's extrinsic is in a robot body frame while the ground truth is in ``world``;
``--parent-to-world`` bridges the two and defaults to identity, with a warning.

The geometry and statistics are pure numpy/OpenCV and unit-tested; only ``read_first_frame``
touches rosbag2, and it imports ROS lazily.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from classical_control.camera_model import FisheyeCamera, pose_to_matrix, project_points
from classical_control.detector import DetectorParams, detect
from classical_control.episode_miner import _bag_uri
from classical_control.localise import ChestCameraConfig, locate_can, locate_paper

CHEST_TOPIC = "/cam_chest/image_raw/compressed"
_MEDIAN_BAND_ROWS = 32
_IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PnPFit:
    """Extrinsic fitted from pixel and world-point correspondences.

    Attributes:
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        inliers: Bool mask over the correspondences that RANSAC kept.
        reprojection_px: Per-correspondence reprojection error in pixels, all points.
    """

    T_world_optical: np.ndarray
    inliers: np.ndarray
    reprojection_px: np.ndarray


def add_parent_to_world_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--parent-to-world``, the pose of the camera yaml's parent frame in ``world``."""
    parser.add_argument(
        "--parent-to-world",
        type=float,
        nargs=7,
        default=list(_IDENTITY_POSE),
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="pose of the camera yaml parent_frame in world (default: identity)",
    )


def camera_in_world(
    config: ChestCameraConfig, parent_to_world: Sequence[float] = _IDENTITY_POSE
) -> np.ndarray:
    """Compose the camera yaml extrinsic with the parent frame's pose in ``world``.

    Args:
        config: Chest camera config; its extrinsic is in ``config.parent_frame``.
        parent_to_world: Pose of ``parent_frame`` in ``world`` as x, y, z, qx, qy, qz, qw.

    Returns:
        4x4 transform mapping optical-frame points into ``world``.
    """
    if config.parent_frame != "world" and tuple(parent_to_world) == _IDENTITY_POSE:
        logger.warning(
            "camera parent_frame is %r and --parent-to-world is identity: treating %r as world",
            config.parent_frame,
            config.parent_frame,
        )
    T_world_parent = pose_to_matrix(parent_to_world[:3], parent_to_world[3:])
    T_parent_optical = pose_to_matrix(config.translation_xyz, config.rotation_xyzw)
    return T_world_parent @ T_parent_optical


def matrix_to_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    """Split a 4x4 transform into the camera yaml's translation and quaternion.

    Args:
        matrix: 4x4 homogeneous transform.

    Returns:
        ``(translation_xyz, rotation_xyzw)`` as plain float lists, ROS quaternion order.
    """
    translation = [float(v) for v in matrix[:3, 3]]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    if quat[3] < 0.0:  # one canonical sign keeps reports diffable
        quat = -quat
    return translation, [float(v) for v in quat]


def project_world_points(
    camera: FisheyeCamera, T_world_optical: np.ndarray, points_world: np.ndarray
) -> np.ndarray:
    """Project world points into the image.

    Args:
        camera: Camera intrinsics.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        points_world: (N, 3) world points.

    Returns:
        (N, 2) pixels; rows are NaN for points at or behind the camera plane, which the
        fisheye model would otherwise fold back into the image.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    T_optical_world = np.linalg.inv(T_world_optical)
    optical = points @ T_optical_world[:3, :3].T + T_optical_world[:3, 3]
    pixels = np.full((len(points), 2), np.nan)
    in_front = optical[:, 2] > 1e-6
    if in_front.any():
        pixels[in_front] = project_points(camera, optical[in_front])
    return pixels


def locate_episodes(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    can_pixels: np.ndarray,
    paper_corners: np.ndarray,
    table_z: float,
    can_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the perception node's localisers over every episode's detections.

    Args:
        camera: Camera intrinsics matching the pixels' image.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        can_pixels: (N, 2) can pixels; NaN rows mark a missing can.
        paper_corners: (N, 4, 2) paper corner pixels; NaN entries mark a missing paper.
        table_z: Table height in ``world``, metres.
        can_height: Can height, metres.

    Returns:
        ``(can_world, paper_world, paper_yaw)``: (N, 3), (N, 3) and (N,) arrays, NaN where
        the object is missing or a ray misses its plane.
    """
    count = len(can_pixels)
    can_world = np.full((count, 3), np.nan)
    paper_world = np.full((count, 3), np.nan)
    paper_yaw = np.full(count, np.nan)
    for i in range(count):
        pixel = can_pixels[i] if np.isfinite(can_pixels[i]).all() else None
        can = locate_can(camera, T_world_optical, pixel, table_z, can_height)
        if can is not None:
            can_world[i] = can
        corners = paper_corners[i] if np.isfinite(paper_corners[i]).all() else None
        paper = locate_paper(camera, T_world_optical, corners, table_z)
        if paper is not None:
            paper_world[i] = paper.position
            paper_yaw[i] = paper.yaw
    return can_world, paper_world, paper_yaw


def xy_errors(estimate_xy: np.ndarray, truth_xy: np.ndarray) -> np.ndarray:
    """Return the per-row planar distance between estimates and truth; NaN where either is."""
    delta = np.asarray(truth_xy, dtype=np.float64)[:, :2] - np.asarray(estimate_xy)[:, :2]
    return np.linalg.norm(delta, axis=1)


def error_stats(errors: np.ndarray) -> dict[str, float | int | None]:
    """Summarise errors, ignoring NaN rows.

    Args:
        errors: (N,) errors in metres.

    Returns:
        ``n``, ``median``, ``p90``, ``mean`` and ``max``; the statistics are None when no
        row is finite.
    """
    finite = np.asarray(errors, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None, "max": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def mean_bias(estimate_xy: np.ndarray, truth_xy: np.ndarray) -> list[float] | None:
    """Return the mean xy offset to add to an estimate to land on the truth.

    This is the correction perception applies as ``can_xy_bias``: corrected = estimate + bias.

    Args:
        estimate_xy: (N, >=2) estimates; NaN rows are skipped.
        truth_xy: (N, >=2) truth, same row order.

    Returns:
        ``[bx, by]`` in metres, or None when no row has both.
    """
    delta = np.asarray(truth_xy, dtype=np.float64)[:, :2] - np.asarray(estimate_xy)[:, :2]
    delta = delta[np.isfinite(delta).all(axis=1)]
    if len(delta) == 0:
        return None
    return [float(v) for v in delta.mean(axis=0)]


def fit_extrinsic(
    camera: FisheyeCamera,
    pixels: np.ndarray,
    points_world: np.ndarray,
    ransac_px: float = 8.0,
) -> PnPFit | None:
    """Fit the camera pose in ``world`` from pixel and world-point pairs with RANSAC PnP.

    ``cv2.fisheye.solvePnP`` is missing from older OpenCV (4.6 on the robot), so the pixels
    are undistorted to normalised coordinates first and solved with an identity camera.

    Args:
        camera: Camera intrinsics matching the pixels' image.
        pixels: (N, 2) pixels, N >= 6, finite.
        points_world: (N, 3) matching world points.
        ransac_px: Inlier threshold in pixels.

    Returns:
        The fit, or None when there are too few pairs or RANSAC finds no pose.
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if len(pixels) < 6:
        return None
    normalised = cv2.fisheye.undistortPoints(
        pixels.reshape(-1, 1, 2), camera.camera_matrix, camera.distortion_array
    ).reshape(-1, 2)
    identity = np.eye(3)
    ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
        points_world,
        normalised,
        identity,
        None,
        iterationsCount=2000,
        reprojectionError=ransac_px / camera.fx,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inlier_idx is None or len(inlier_idx) < 6:
        return None
    inliers = np.zeros(len(pixels), dtype=bool)
    inliers[inlier_idx.ravel()] = True
    # RANSAC's pose comes from its best minimal sample; refining on every inlier settles it.
    _, rvec, tvec = cv2.solvePnP(
        points_world[inliers],
        normalised[inliers],
        identity,
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    T_optical_world = np.eye(4)
    T_optical_world[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_optical_world[:3, 3] = tvec.ravel()
    T_world_optical = np.linalg.inv(T_optical_world)
    reprojected = project_world_points(camera, T_world_optical, points_world)
    return PnPFit(T_world_optical, inliers, np.linalg.norm(reprojected - pixels, axis=1))


def median_image(frames: np.ndarray) -> np.ndarray:
    """Return the per-pixel median of a stack of uint8 images.

    Args:
        frames: (N, H, W, C) uint8 stack.

    Returns:
        (H, W, C) uint8 median image.
    """
    out = np.empty(frames.shape[1:], dtype=np.uint8)
    # np.median promotes to float64; working in row bands keeps that copy small.
    for top in range(0, frames.shape[1], _MEDIAN_BAND_ROWS):
        band = frames[:, top : top + _MEDIAN_BAND_ROWS]
        out[top : top + _MEDIAN_BAND_ROWS] = np.round(np.median(band, axis=0)).astype(np.uint8)
    return out


def read_first_frame(episode_dir: Path) -> np.ndarray | None:
    """Decode the first chest camera frame of one rosbag2 MCAP episode.

    The storage filter skips every other topic and reading stops after one message, so a
    bag costs one JPEG decode however long the episode is.

    Args:
        episode_dir: Episode directory holding the MCAP bag.

    Returns:
        BGR uint8 frame, or None when the bag has no chest frame.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=_bag_uri(episode_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[CHEST_TOPIC]))
    if not reader.has_next():
        return None
    _, data, _ = reader.read_next()
    msg = deserialize_message(data, CompressedImage)
    return cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)


def read_ground_truth(path: Path) -> dict[str, dict[str, float | None]]:
    """Read ``ground_truth.csv`` into ``{episode: {column: value or None}}``."""
    rows: dict[str, dict[str, float | None]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            episode = row.pop("episode")
            rows[episode] = {k: float(v) if v else None for k, v in row.items()}
    return rows


def _xyz(row: Mapping[str, float | None] | None, prefix: str) -> list[float]:
    if row is None:
        return [np.nan] * 3
    return [np.nan if row[f"{prefix}_{a}"] is None else row[f"{prefix}_{a}"] for a in "xyz"]


def _score(
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    can_px: np.ndarray,
    paper_corners: np.ndarray,
    can_gt: np.ndarray,
    paper_gt: np.ndarray,
    table_z: float,
    can_height: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Locate both objects with one extrinsic and score them against the truth."""
    can_est, paper_est, paper_yaw = locate_episodes(
        camera, T_world_optical, can_px, paper_corners, table_z, can_height
    )
    can_err = xy_errors(can_est, can_gt)
    paper_err = xy_errors(paper_est, paper_gt)
    can_bias = mean_bias(can_est, can_gt)
    summary = {
        "can_xy_error_m": error_stats(can_err),
        "paper_xy_error_m": error_stats(paper_err),
        "can_mean_bias_xy_m": can_bias,
        "paper_mean_bias_xy_m": mean_bias(paper_est, paper_gt),
        "suggested_can_xy_bias": can_bias,
    }
    columns = {
        "can_est": can_est,
        "can_err": can_err,
        "paper_est": paper_est,
        "paper_err": paper_err,
        "paper_yaw": paper_yaw,
    }
    return summary, columns


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate perception over every episode and write the report to ``--out-dir``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, required=True, help="episode directories")
    parser.add_argument("--ground-truth", type=Path, required=True, help="ground_truth.csv")
    parser.add_argument("--camera-yaml", type=Path, required=True, help="chest camera yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--table-z", type=float, default=0.22, help="table height in world, m")
    parser.add_argument("--can-height", type=float, default=0.12, help="can height, m")
    parser.add_argument(
        "--scale", type=float, default=0.5, help="detect on frames resized by this factor"
    )
    parser.add_argument("--ransac-px", type=float, default=8.0, help="PnP inlier threshold")
    parser.add_argument(
        "--save-background", type=Path, help="also write the median background jpg here"
    )
    add_parent_to_world_argument(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = ChestCameraConfig.from_yaml(args.camera_yaml)
    T_world_parent = pose_to_matrix(args.parent_to_world[:3], args.parent_to_world[3:])
    T_world_optical = camera_in_world(config, args.parent_to_world)
    truth = read_ground_truth(args.ground_truth)

    episodes: list[str] = []
    # tradeoff: keeps every full-resolution first frame in memory (~6 MB each, ~650 MB for
    # 104 episodes) so the background stays full resolution for the perception node.
    frames: list[np.ndarray] = []
    episode_dirs = sorted(p for p in args.recordings.iterdir() if p.is_dir())
    for i, episode_dir in enumerate(episode_dirs, start=1):
        try:
            frame = read_first_frame(episode_dir)
        except Exception as exc:  # one bad bag must not stop the batch
            logger.warning("[%d/%d] %s: unreadable (%s)", i, len(episode_dirs), episode_dir, exc)
            continue
        if frame is None or (frames and frame.shape != frames[0].shape):
            logger.warning("[%d/%d] %s: no usable chest frame", i, len(episode_dirs), episode_dir)
            continue
        episodes.append(episode_dir.name)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no chest frames under {args.recordings}")
    logger.info("read %d first frames; building the median background", len(frames))

    background = median_image(np.stack(frames))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_background:
        args.save_background.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_background), background)

    height, width = background.shape[:2]
    camera = config.camera.scaled(width, height)
    small_size = (round(width * args.scale), round(height * args.scale))
    small_background = cv2.resize(background, small_size, interpolation=cv2.INTER_AREA)
    to_full = np.array([width / small_size[0], height / small_size[1]])
    params = DetectorParams()

    can_px = np.full((len(frames), 2), np.nan)
    paper_corners = np.full((len(frames), 4, 2), np.nan)
    for i, frame in enumerate(frames):
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
        detection = detect(small, small_background, params)
        # Pixels go back to full resolution so every later step uses one camera model.
        if detection.can_pixel is not None:
            can_px[i] = np.asarray(detection.can_pixel) * to_full
        if detection.paper_corners is not None:
            paper_corners[i] = np.asarray(detection.paper_corners) * to_full
    frames.clear()
    # tradeoff: PnP pairs the paper's pixel centre with its true centre; the fisheye makes
    # these differ by a pixel or two, well inside the RANSAC threshold.
    paper_px = paper_corners.mean(axis=1)

    can_gt = np.array([_xyz(truth.get(e), "can") for e in episodes])
    paper_gt = np.array([_xyz(truth.get(e), "paper") for e in episodes])
    can_plane_z = args.table_z + args.can_height / 2.0
    has_can = np.isfinite(can_px).all(axis=1)
    has_paper = np.isfinite(paper_px).all(axis=1)
    n = len(episodes)
    report: dict[str, Any] = {
        "inputs": {
            "recordings": str(args.recordings),
            "ground_truth": str(args.ground_truth),
            "camera_yaml": str(args.camera_yaml),
            "image_size": [width, height],
            "detect_scale": args.scale,
            "table_z": args.table_z,
            "can_height": args.can_height,
            "can_plane_z": can_plane_z,
            "paper_plane_z": args.table_z,
            "camera_parent_frame": config.parent_frame,
            "parent_to_world": [float(v) for v in args.parent_to_world],
        },
        "detection": {
            "episodes": n,
            "with_ground_truth": int(sum(e in truth for e in episodes)),
            "can": int(has_can.sum()),
            "paper": int(has_paper.sum()),
            "both": int((has_can & has_paper).sum()),
            "can_rate": _rate(int(has_can.sum()), n),
            "paper_rate": _rate(int(has_paper.sum()), n),
            "both_rate": _rate(int((has_can & has_paper).sum()), n),
        },
    }
    score_args = (can_px, paper_corners, can_gt, paper_gt, args.table_z, args.can_height)
    report["yaml_extrinsic"], yaml_cols = _score(camera, T_world_optical, *score_args)

    # The grasp TCP is a 3D point at the can, so its true z is used. The release TCP holds
    # the can over the paper; the paper itself lies a can half-height lower, on the table.
    # tradeoff: assumes the gripper holds the can at mid-height, as perception's can plane does.
    paper_pnp = paper_gt - np.array([0.0, 0.0, args.can_height / 2.0])
    pnp_pixels = np.vstack([can_px, paper_px])
    pnp_world = np.vstack([can_gt, paper_pnp])
    usable = np.isfinite(pnp_pixels).all(axis=1) & np.isfinite(pnp_world).all(axis=1)
    fit = fit_extrinsic(camera, pnp_pixels[usable], pnp_world[usable], args.ransac_px)
    fit_cols: dict[str, np.ndarray] = {}
    if fit is None:
        logger.warning("PnP fit failed on %d correspondences", int(usable.sum()))
        report["fitted_extrinsic"] = None
    else:
        # The yaml stores the pose in its parent frame, so the fit is carried back into it.
        translation, rotation = matrix_to_pose(np.linalg.inv(T_world_parent) @ fit.T_world_optical)
        world_translation, world_rotation = matrix_to_pose(fit.T_world_optical)
        inlier_err = fit.reprojection_px[fit.inliers]
        paper_inliers = fit.inliers[int(usable[:n].sum()) :]
        report["fitted_extrinsic"] = {
            # Paste-ready for the camera yaml extrinsic: pose of the optical frame in
            # parent_frame, derived through --parent-to-world.
            "parent_frame": config.parent_frame,
            "translation_xyz": translation,
            "rotation_xyzw": rotation,
            "in_world": {"translation_xyz": world_translation, "rotation_xyzw": world_rotation},
            "correspondences": int(usable.sum()),
            "inliers": int(fit.inliers.sum()),
            "reprojection_px_inliers": {
                "median": float(np.median(inlier_err)),
                "p90": float(np.percentile(inlier_err, 90)),
                "max": float(np.max(inlier_err)),
            },
            # Implied by the release TCP and the can half-height; compare with --table-z.
            "implied_table_z": float(np.median(paper_pnp[usable[n:], 2][paper_inliers]))
            if paper_inliers.any()
            else None,
        }
        scored, fit_cols = _score(camera, fit.T_world_optical, *score_args)
        report["fitted_extrinsic"].update(scored)

    report_path = args.out_dir / "eval_report.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    _write_episode_csv(
        args.out_dir / "eval_episodes.csv",
        episodes,
        np.hstack([can_px, paper_px, can_gt, paper_gt]),
        yaml_cols,
        fit_cols,
    )
    logger.info("report:\n%s", json.dumps(report, indent=2))
    logger.info("wrote %s", args.out_dir)


def _write_episode_csv(
    path: Path,
    episodes: list[str],
    inputs: np.ndarray,
    yaml_cols: Mapping[str, np.ndarray],
    fit_cols: Mapping[str, np.ndarray],
) -> None:
    """Write one row per episode: pixels, truth, and each extrinsic's estimates and errors.

    Args:
        path: CSV path.
        episodes: Episode names, in row order.
        inputs: (N, 10) can pixel, paper pixel centre, can truth, paper truth.
        yaml_cols: Estimates and errors under the yaml extrinsic.
        fit_cols: Estimates and errors under the fitted extrinsic; empty if the fit failed.
    """
    header = ["episode", "can_u", "can_v", "paper_u", "paper_v"]
    header += ["gt_can_x", "gt_can_y", "gt_can_z", "gt_paper_x", "gt_paper_y", "gt_paper_z"]
    for tag, cols in (("yaml", yaml_cols), ("fit", fit_cols)):
        if cols:
            header += [f"{tag}_can_x", f"{tag}_can_y", f"{tag}_can_err"]
            header += [f"{tag}_paper_x", f"{tag}_paper_y", f"{tag}_paper_err", f"{tag}_paper_yaw"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, episode in enumerate(episodes):
            row: list[Any] = [episode, *inputs[i]]
            for cols in (yaml_cols, fit_cols):
                if cols:
                    row += [*cols["can_est"][i, :2], cols["can_err"][i]]
                    row += [*cols["paper_est"][i, :2], cols["paper_err"][i], cols["paper_yaw"][i]]
            writer.writerow(["" if isinstance(v, float) and np.isnan(v) else v for v in row])


if __name__ == "__main__":
    main()
