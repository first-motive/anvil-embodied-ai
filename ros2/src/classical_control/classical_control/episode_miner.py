"""Mine grasp and place parameters from recorded teleop pick-and-place episodes.

The classical can-on-paper baseline needs a grasp height, a place height and a
tool orientation. Rather than hand-tuning them, this module reads the teleop
demonstrations, finds the moment the gripper closed on the can and the moment it
released it at the paper, and takes the commanded TCP pose at each. Summary
statistics across episodes become ``mined_params.yaml``; the per-episode events
become ``ground_truth.csv`` for checking the perception stack against.

The event detection and statistics are pure numpy/scipy so they are testable off
the robot. Only ``read_episode`` touches rosbag2, and it imports ROS lazily.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

FINGER_JOINT = "follower_r_finger_joint1"
JOINT_STATES_TOPIC = "/joint_states"
EE_POSE_TOPIC = "/ee_pose_right"
CSV_COLUMNS = (
    "episode",
    "close_t",
    "open_t",
    "can_x",
    "can_y",
    "can_z",
    "paper_x",
    "paper_y",
    "paper_z",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GripperEvents:
    """Gripper close and release times found in one finger trace.

    Attributes:
        close_t: Time the finger first fell below the close threshold after being open.
        open_t: Time the finger first rose off its settled closed value, or None if never.
        closed_value: Lowest finger position held between close and release.
    """

    close_t: float
    open_t: float | None
    closed_value: float


@dataclass(frozen=True)
class EpisodeResult:
    """Mined events and TCP poses for one episode.

    Poses are ``[x, y, z, qx, qy, qz, qw]`` in the world frame. Times are seconds
    from the first message in the episode.
    """

    episode: str
    close_t: float
    open_t: float | None
    grasp_pose: np.ndarray
    place_pose: np.ndarray | None


def find_gripper_events(
    times: np.ndarray,
    finger: np.ndarray,
    *,
    open_threshold: float = 0.040,
    close_threshold: float = 0.030,
    release_delta: float = 0.003,
) -> GripperEvents | None:
    """Find when the gripper closed on the object and when it let go.

    A close is the first fall below ``close_threshold`` that follows a sample above
    ``open_threshold``, so a gripper that starts closed is not mistaken for a grasp.
    The can blocks the fingers part-way (~0.0185) while an empty close reaches ~0.0,
    so release is measured relative to the settled closed value rather than a fixed
    level: the first sample more than ``release_delta`` above the lowest value
    seen since the close.

    Args:
        times: Sample times in seconds, ascending.
        finger: Finger joint position per sample, metres.
        open_threshold: Position above which the gripper counts as open.
        close_threshold: Position below which the gripper counts as closed.
        release_delta: Rise above the settled closed value that marks a release.

    Returns:
        The events, or None if the gripper never closed from open.
    """
    times = np.asarray(times, dtype=float)
    finger = np.asarray(finger, dtype=float)
    was_open = np.maximum.accumulate(finger > open_threshold)
    close_candidates = np.flatnonzero(was_open & (finger < close_threshold))
    if close_candidates.size == 0:
        return None
    close_idx = int(close_candidates[0])

    after = finger[close_idx:]
    settled = np.minimum.accumulate(after)
    released = np.flatnonzero(after > settled + release_delta)
    if released.size == 0:
        return GripperEvents(float(times[close_idx]), None, float(settled[-1]))
    release_idx = int(released[0])
    return GripperEvents(
        close_t=float(times[close_idx]),
        open_t=float(times[close_idx + release_idx]),
        closed_value=float(settled[release_idx]),
    )


def pose_at(times: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """Return the pose sample nearest in time to ``t``.

    Args:
        times: Pose sample times in seconds, ascending.
        poses: Array of shape (N, 7), ``[x, y, z, qx, qy, qz, qw]`` per row.
        t: Query time in seconds.

    Returns:
        The nearest pose row, shape (7,).
    """
    # tradeoff: nearest sample, not interpolation; at ~90 Hz the TCP is near-still
    # at grasp and release, so the error is well under a millimetre.
    times = np.asarray(times, dtype=float)
    idx = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    if t - times[idx - 1] <= times[idx] - t:
        idx -= 1
    return np.asarray(poses[idx], dtype=float)


def mine_episode(
    episode: str,
    finger_times: np.ndarray,
    finger: np.ndarray,
    ee_times: np.ndarray,
    ee_poses: np.ndarray,
) -> EpisodeResult | None:
    """Detect gripper events in one episode and look up the TCP pose at each.

    Args:
        episode: Episode name, used as the CSV key.
        finger_times: Finger sample times in seconds.
        finger: Finger joint positions.
        ee_times: TCP pose sample times in seconds.
        ee_poses: TCP poses, shape (N, 7).

    Returns:
        The mined result, or None if there is no data or no close event.
    """
    if len(finger_times) == 0 or len(ee_times) == 0:
        return None
    events = find_gripper_events(finger_times, finger)
    if events is None:
        return None
    t0 = min(finger_times[0], ee_times[0])
    return EpisodeResult(
        episode=episode,
        close_t=events.close_t - t0,
        open_t=None if events.open_t is None else events.open_t - t0,
        grasp_pose=pose_at(ee_times, ee_poses, events.close_t),
        place_pose=None if events.open_t is None else pose_at(ee_times, ee_poses, events.open_t),
    )


def _height_stats(z: np.ndarray) -> dict[str, float | int]:
    return {
        "median": float(np.median(z)),
        "p10": float(np.percentile(z, 10)),
        "p90": float(np.percentile(z, 90)),
        "std": float(np.std(z)),
        "n": int(z.size),
    }


def _orientation_stats(quats_xyzw: np.ndarray) -> tuple[list[float], dict[str, float]]:
    rotations = Rotation.from_quat(quats_xyzw)
    mean = rotations.mean()
    deviation_deg = np.degrees((rotations * mean.inv()).magnitude())
    spread = {
        "median": float(np.median(deviation_deg)),
        "p90": float(np.percentile(deviation_deg, 90)),
        "max": float(np.max(deviation_deg)),
    }
    return [float(v) for v in mean.as_quat()], spread


def summarise(episodes: list[EpisodeResult], total: int | None = None) -> dict:
    """Reduce per-episode results to the parameters the baseline controller uses.

    Args:
        episodes: Results for episodes where a close was found.
        total: Number of episodes attempted, including skipped ones. Defaults to
            ``len(episodes)``.

    Returns:
        A plain dict ready for YAML: episode counts, grasp/place height stats,
        mean orientation quaternions (xyzw) and their angular spread in degrees.

    Raises:
        ValueError: If ``episodes`` is empty.
    """
    if not episodes:
        raise ValueError("no episodes with a gripper close to summarise")
    grasp = np.array([e.grasp_pose for e in episodes])
    placed = [e.place_pose for e in episodes if e.place_pose is not None]
    grasp_quat, grasp_spread = _orientation_stats(grasp[:, 3:])
    params: dict = {
        "episodes": {
            "total": len(episodes) if total is None else total,
            "with_close": len(episodes),
            "with_open": len(placed),
        },
        "grasp_z": _height_stats(grasp[:, 2]),
        "grasp_orientation_xyzw": grasp_quat,
        "grasp_orientation_spread_deg": grasp_spread,
    }
    if placed:
        place = np.array(placed)
        place_quat, place_spread = _orientation_stats(place[:, 3:])
        params["place_z"] = _height_stats(place[:, 2])
        params["place_orientation_xyzw"] = place_quat
        params["place_orientation_spread_deg"] = place_spread
    return params


def write_params_yaml(params: dict, path: Path) -> None:
    """Write mined parameters as YAML."""
    path.write_text(yaml.safe_dump(params, sort_keys=False))


def write_ground_truth_csv(episodes: list[EpisodeResult], path: Path) -> None:
    """Write one row per episode with event times and TCP positions.

    Paper columns and ``open_t`` are blank when the episode has no release.
    """
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for e in episodes:
            paper = ["", "", ""] if e.place_pose is None else list(e.place_pose[:3])
            open_t = "" if e.open_t is None else e.open_t
            writer.writerow([e.episode, e.close_t, open_t, *e.grasp_pose[:3], *paper])


def _bag_uri(episode_dir: Path) -> str:
    # A bag directory without metadata.yaml must be opened by its .mcap file.
    if (episode_dir / "metadata.yaml").exists():
        return str(episode_dir)
    mcaps = sorted(episode_dir.glob("*.mcap"))
    if not mcaps:
        raise FileNotFoundError(f"no .mcap file in {episode_dir}")
    return str(mcaps[0])


def read_episode(
    episode_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the finger joint and TCP pose topics out of one rosbag2 MCAP episode.

    Only the two topics are deserialised; camera topics are filtered out by the
    storage layer so images are never loaded.

    Args:
        episode_dir: Episode directory holding the MCAP bag.

    Returns:
        ``(finger_times, finger, ee_times, ee_poses)`` with times in seconds
        (bag receive time) and ``ee_poses`` of shape (N, 7).
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=_bag_uri(episode_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    # get_message resolves anvil_msgs here, so the proprietary package is only
    # needed where bags are actually read.
    msg_types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
        if t.name in (JOINT_STATES_TOPIC, EE_POSE_TOPIC)
    }
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(msg_types)))

    finger_times: list[float] = []
    finger: list[float] = []
    ee_times: list[float] = []
    ee_poses: list[list[float]] = []
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        msg = deserialize_message(data, msg_types[topic])
        t = stamp_ns * 1e-9
        if topic == JOINT_STATES_TOPIC:
            if FINGER_JOINT in msg.name:
                finger_times.append(t)
                finger.append(msg.position[msg.name.index(FINGER_JOINT)])
        else:
            p, q = msg.pose.position, msg.pose.orientation
            ee_times.append(t)
            ee_poses.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    return (
        np.asarray(finger_times),
        np.asarray(finger),
        np.asarray(ee_times),
        np.asarray(ee_poses).reshape(-1, 7),
    )


def main(argv: list[str] | None = None) -> None:
    """Mine every episode under ``--recordings`` and write the summary files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recordings", type=Path, required=True, help="directory of episode directories"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="where to write the YAML and CSV"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    episode_dirs = sorted(p for p in args.recordings.iterdir() if p.is_dir())
    results: list[EpisodeResult] = []
    for i, episode_dir in enumerate(episode_dirs, start=1):
        name = episode_dir.name
        try:
            result = mine_episode(name, *read_episode(episode_dir))
        except Exception as exc:  # one bad bag must not stop the batch
            logger.warning("[%d/%d] %s: unreadable, skipped (%s)", i, len(episode_dirs), name, exc)
            continue
        if result is None:
            logger.warning("[%d/%d] %s: no gripper close, skipped", i, len(episode_dirs), name)
            continue
        logger.info(
            "[%d/%d] %s: close %.2fs, open %s",
            i,
            len(episode_dirs),
            name,
            result.close_t,
            "none" if result.open_t is None else f"{result.open_t:.2f}s",
        )
        results.append(result)

    if not results:
        raise SystemExit(f"no usable episodes under {args.recordings}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_params_yaml(
        summarise(results, total=len(episode_dirs)), args.out_dir / "mined_params.yaml"
    )
    write_ground_truth_csv(results, args.out_dir / "ground_truth.csv")
    logger.info("wrote %d of %d episodes to %s", len(results), len(episode_dirs), args.out_dir)


if __name__ == "__main__":
    main()
