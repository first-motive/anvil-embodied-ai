"""Skill segments with end-effector features in the handover frame.

A skill segment runs from a SKILL_START event to the matching SKILL_END event
on the fm_skills_msgs/TaskEvent topic. The SKILL_START event carries the
handover frame H: the TCP pose at the moment the skill took over. Every pose
in the segment is expressed relative to H (see frames.py), so the learned
skill sees the same coordinates wherever on the table the handover happened.

Rotations are rotation vectors: relative to H the rotation starts at identity
and normally stays well inside pi, so the rotation vector is continuous over a
segment. Segments that come close to pi are flagged.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..config.schema import DataConfig
from ..utils.image_utils import decode_compressed_image, decode_image, resize_image
from ..utils.logging import get_logger
from .extractor import message_timestamp
from .frames import pose_in_frame, pose_to_vector
from .reader import McapReader

logger = get_logger("core.skill_segments")

# fm_skills_msgs/TaskEvent.kind values
SKILL_START = 2
SKILL_END = 3

# Action pose names match the state's so anvil-trainer's delta transform pairs them by name
POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz"]
STATE_NAMES = POSE_NAMES + ["finger_width_m", "finger_effort"]
ACTION_NAMES = POSE_NAMES + ["gripper"]

# Reads are windowed by log time while alignment uses header stamps; the
# margin covers the transport delay between the two.
_READ_MARGIN_S = 1.0

# Past this angle from H a rotation vector is near its flip at pi.
_ROTATION_WARN_RAD = 2.5

# =============================================================================
# Segments
# =============================================================================


@dataclass(frozen=True)
class SkillSegment:
    """One run of a skill, from its SKILL_START to its SKILL_END event."""

    episode: int
    start: float  # header stamp, seconds
    end: float
    handover: np.ndarray  # H as [x, y, z, qx, qy, qz, qw]
    code: int
    verified: bool

    @property
    def succeeded(self) -> bool:
        return self.code == 0 and self.verified


def find_skill_segments(events: List[Tuple[float, Any]], skill: str) -> List[SkillSegment]:
    """
    Pair each SKILL_START of ``skill`` with the next SKILL_END of the same task episode.

    A start with no matching end (the recording stopped, or another start came
    first) is dropped: it has no outcome to label it with.

    Args:
        events: ``(stamp_seconds, TaskEvent)`` pairs in time order.
        skill: Skill name to cut, e.g. ``"pick"``.
    """
    segments = []
    open_start: Optional[Tuple[float, Any]] = None
    for stamp, event in events:
        if event.skill != skill:
            continue
        if event.kind == SKILL_START:
            open_start = (stamp, event)
        elif event.kind == SKILL_END and open_start is not None:
            start_stamp, start = open_start
            open_start = None
            if start.episode != event.episode:
                continue
            p, q = start.handover_pose.pose.position, start.handover_pose.pose.orientation
            segments.append(
                SkillSegment(
                    episode=start.episode,
                    start=start_stamp,
                    end=stamp,
                    handover=np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w]),
                    code=event.code,
                    verified=event.verified,
                )
            )
    return segments


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index into sorted ``times`` of the sample nearest each target."""
    if len(times) == 1:
        return np.zeros(len(targets), dtype=int)
    right = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left = right - 1
    return np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)


def segment_features(
    handover: np.ndarray,
    frame_times: np.ndarray,
    ee_times: np.ndarray,
    ee_poses: np.ndarray,
    finger_times: np.ndarray,
    finger: np.ndarray,
    finger_effort: np.ndarray,
    command_times: np.ndarray,
    commands: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observation state and action for each frame but the last.

    ``state[t]  = [pose_t in H, finger width_t, finger effort_t]``
    ``action[t] = [pose_{t+1} in H, gripper target_t]``

    The action pose is absolute in H with the state's names, so anvil-trainer's
    ``delta_obs_t`` transform subtracts the observed pose from every chunk row.
    The gripper target is the last finger command published at or before t;
    before the first command (or with none recorded) it falls back to the
    measured width at t+1. A measured target alone never squeezes.

    Returns:
        ``(state, action)`` of shapes ``(N-1, 8)`` and ``(N-1, 7)``.
    """
    ee_idx = nearest_indices(ee_times, frame_times)
    poses = np.array([pose_to_vector(pose_in_frame(ee_poses[i], handover)) for i in ee_idx])
    finger_idx = nearest_indices(finger_times, frame_times)
    width = finger[finger_idx]

    latest = np.searchsorted(command_times, frame_times[:-1], side="right") - 1
    commanded = commands[np.clip(latest, 0, None)] if len(commands) else width[1:]
    gripper_target = np.where(latest >= 0, commanded, width[1:])

    state = np.column_stack([poses, width, finger_effort[finger_idx]])[:-1]
    action = np.column_stack([poses[1:], gripper_target])
    return state.astype(np.float32), action.astype(np.float32)


# =============================================================================
# MCAP extraction
# =============================================================================


def read_skill_segments(mcap_path: str, config: DataConfig) -> List[SkillSegment]:
    """All segments of the configured skill in one MCAP, whatever their outcome."""
    seg_cfg = config.skill_segments
    events = [
        (message_timestamp(m), m.ros_msg)
        for m in McapReader(mcap_path).read_messages(topics=[seg_cfg.event_topic])
    ]
    return find_skill_segments(events, seg_cfg.skill)


def read_gripper_commands(mcap_path: str, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``(times, positions)`` of every control_msgs/GripperCommand in the MCAP, in time order.

    Read over the whole file, not per segment: the finger is commanded on
    change, so the command in force at a segment can predate it by any amount.
    GripperCommand has no header, so times are log times.
    """
    commands = sorted(
        (message_timestamp(m), m.ros_msg.position)
        for m in McapReader(mcap_path).read_messages(topics=[topic])
    )
    if not commands:
        return np.empty(0), np.empty(0)
    times, positions = zip(*commands)
    return np.array(times), np.array(positions)


def sample_indices(times: np.ndarray, fps: int) -> np.ndarray:
    """
    Indices of the samples nearest each point of an ``fps`` grid anchored at ``times[0]``.

    Grid points are computed from their index, not by stepping a running
    timestamp, which accumulates float error and skips frames.
    """
    count = int(np.floor((times[-1] - times[0]) * fps + 0.5)) + 1
    grid = times[0] + np.arange(count) / fps
    return np.unique(nearest_indices(times, grid))


def _decode(msg) -> np.ndarray:
    if hasattr(msg, "format"):
        return decode_compressed_image(msg.data, msg.format)
    return decode_image(msg.data, msg.encoding, msg.height, msg.width)


def extract_segment_frames(
    mcap_path: str,
    segment: SkillSegment,
    config: DataConfig,
    fps: int,
    task: str,
    commands: Tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0)),
) -> Iterator[Dict[str, Any]]:
    """
    LeRobot frames for one segment, one per main-camera grid slot at ``fps``.

    The first camera in ``camera_topic_mapping`` paces the frames; other
    cameras are matched by nearest stamp. ``commands`` are the finger commands
    from :func:`read_gripper_commands`. Images stay as the recorded messages
    until a frame is yielded, so only the kept frames are ever decoded and only
    one is held decoded at a time. Yields nothing when a topic has no data in
    the segment.
    """
    seg_cfg = config.skill_segments
    cameras = list(config.camera_topic_mapping.values())
    images: Dict[str, List[Tuple[float, Any]]] = {cam: [] for cam in cameras}
    ee: List[Tuple[float, np.ndarray]] = []
    fingers: List[Tuple[float, float, float]] = []
    topics = [*config.camera_topic_mapping, seg_cfg.ee_pose_topic, config.robot_state_topic]
    max_skew = 0.0

    reader = McapReader(mcap_path)
    for message in reader.read_messages(
        topics=topics,
        start_time=int((segment.start - _READ_MARGIN_S) * 1e9),
        end_time=int((segment.end + _READ_MARGIN_S) * 1e9),
    ):
        topic, msg, stamp = message.channel.topic, message.ros_msg, message_timestamp(message)
        max_skew = max(max_skew, abs(stamp - message.log_time_ns * 1e-9))
        if topic in config.camera_topic_mapping:
            if segment.start <= stamp <= segment.end:
                images[config.camera_topic_mapping[topic]].append((stamp, msg))
        elif topic == seg_cfg.ee_pose_topic:
            p, q = msg.pose.position, msg.pose.orientation
            ee.append((stamp, np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])))
        elif seg_cfg.gripper_joint in msg.name:
            i = list(msg.name).index(seg_cfg.gripper_joint)
            effort = msg.effort[i] if len(msg.effort) > i else 0.0
            fingers.append((stamp, msg.position[i], effort))

    if max_skew > _READ_MARGIN_S:
        logger.warning(
            f"Episode {segment.episode}: header stamps differ from log times by up to "
            f"{max_skew:.2f} s (> {_READ_MARGIN_S} s read margin); samples may be missing"
        )
    if not ee or not fingers or any(not images[cam] for cam in cameras):
        return

    for frames_of_cam in images.values():
        frames_of_cam.sort(key=lambda item: item[0])
    main = images[cameras[0]]
    main_times = np.array([t for t, _ in main])
    keep = sample_indices(main_times, fps)
    if len(keep) < 2:
        return
    frame_times = main_times[keep]

    ee.sort(key=lambda item: item[0])
    finger_arr = np.array(sorted(fingers))
    state, action = segment_features(
        segment.handover,
        frame_times,
        np.array([t for t, _ in ee]),
        np.array([pose for _, pose in ee]),
        finger_arr[:, 0],
        finger_arr[:, 1],
        finger_arr[:, 2],
        *commands,
    )
    max_angle = float(np.linalg.norm(state[:, 3:6], axis=1).max())
    if max_angle > _ROTATION_WARN_RAD:
        logger.warning(
            f"Episode {segment.episode}: TCP turns {max_angle:.2f} rad from H; the rotation "
            "vector flips near pi"
        )

    size = tuple(config.image_resolution)
    other_idx = {
        cam: nearest_indices(np.array([t for t, _ in images[cam]]), frame_times)
        for cam in cameras[1:]
    }
    for n in range(len(state)):
        frame = {f"observation.images.{cameras[0]}": resize_image(_decode(main[keep[n]][1]), size)}
        for cam, idx in other_idx.items():
            frame[f"observation.images.{cam}"] = resize_image(_decode(images[cam][idx[n]][1]), size)
        frame["observation.state"] = state[n]
        frame["action"] = action[n]
        frame["task"] = task
        yield frame


def iter_skill_episodes(
    mcap_path: str, config: DataConfig, fps: int, task: str
) -> Iterator[Tuple[SkillSegment, Iterator[Dict[str, Any]]]]:
    """
    Yield ``(segment, frames)`` for every segment to keep.

    ``frames`` is a lazy iterator; consume it before advancing to the next
    segment. Successful, verified segments carry ``task``; failures, kept only
    with ``include_failures``, carry ``"<task> [failed]"`` so they can be
    filtered by task index.
    """
    seg_cfg = config.skill_segments
    segments = [
        s for s in read_skill_segments(mcap_path, config) if s.succeeded or seg_cfg.include_failures
    ]
    if not segments:
        return
    commands = (
        read_gripper_commands(mcap_path, seg_cfg.gripper_command_topic)
        if seg_cfg.gripper_command_topic
        else (np.empty(0), np.empty(0))
    )
    for segment in segments:
        label = task if segment.succeeded else f"{task} [failed]"
        yield segment, extract_segment_frames(mcap_path, segment, config, fps, label, commands)
