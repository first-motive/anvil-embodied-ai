"""Tests for skill segments with end-effector features in the handover frame."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from mcap_converter.config.loader import ConfigLoader
from mcap_converter.core.skill_segments import (
    SKILL_END,
    SKILL_START,
    extract_segment_frames,
    find_skill_segments,
    iter_skill_episodes,
    sample_indices,
    segment_features,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def yaw_pose(x, y, z, yaw):
    return np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


# =============================================================================
# Segments
# =============================================================================


def event(kind, skill="pick", episode=1, code=0, verified=True, handover=IDENTITY):
    position = SimpleNamespace(x=handover[0], y=handover[1], z=handover[2])
    orientation = SimpleNamespace(x=handover[3], y=handover[4], z=handover[5], w=handover[6])
    pose = SimpleNamespace(position=position, orientation=orientation)
    return SimpleNamespace(
        kind=kind,
        skill=skill,
        episode=episode,
        code=code,
        verified=verified,
        handover_pose=SimpleNamespace(pose=pose),
    )


class TestFindSkillSegments:
    def test_pairs_start_with_end_and_carries_outcome(self):
        handover = yaw_pose(0.4, 0.0, 0.3, 0.5)
        events = [
            (1.0, event(SKILL_START, handover=handover)),
            (3.0, event(SKILL_END, code=4, verified=False)),
        ]
        [segment] = find_skill_segments(events, "pick")
        assert (segment.start, segment.end, segment.code, segment.verified) == (1.0, 3.0, 4, False)
        np.testing.assert_allclose(segment.handover, handover)
        assert not segment.succeeded

    def test_ignores_other_skills(self):
        events = [
            (1.0, event(SKILL_START, skill="place")),
            (2.0, event(SKILL_END, skill="place")),
            (3.0, event(SKILL_START)),
            (4.0, event(SKILL_END)),
        ]
        [segment] = find_skill_segments(events, "pick")
        assert segment.start == 3.0 and segment.succeeded

    def test_drops_start_without_end(self):
        events = [
            (1.0, event(SKILL_START, episode=1)),
            (2.0, event(SKILL_START, episode=2)),
            (3.0, event(SKILL_END, episode=2)),
            (4.0, event(SKILL_START, episode=3)),
        ]
        [segment] = find_skill_segments(events, "pick")
        assert (segment.episode, segment.start) == (2, 2.0)

    def test_drops_end_from_another_episode(self):
        events = [(1.0, event(SKILL_START, episode=1)), (2.0, event(SKILL_END, episode=2))]
        assert find_skill_segments(events, "pick") == []


class TestSegmentFeatures:
    def setup_method(self):
        self.handover = yaw_pose(0.5, 0.0, 0.3, np.pi / 2)
        self.frame_times = np.arange(5) * 0.1
        # TCP moves along world +y (H's +x) and yaws 1 rad/s
        self.ee_times = np.arange(50) * 0.01
        self.ee_poses = np.array([yaw_pose(0.5, t, 0.3, np.pi / 2 + t) for t in self.ee_times])
        self.finger_times = np.arange(50) * 0.01
        self.finger = 0.05 - self.finger_times * 0.01
        self.effort = np.full(50, 2.0)

    def features(self, command_times=(), commands=()):
        return segment_features(
            self.handover,
            self.frame_times,
            self.ee_times,
            self.ee_poses,
            self.finger_times,
            self.finger,
            self.effort,
            np.array(command_times, dtype=float),
            np.array(commands, dtype=float),
        )

    def test_state_in_h_and_action_is_next_pose_in_h(self):
        state, action = self.features()
        assert state.shape == (4, 8) and action.shape == (4, 7)
        np.testing.assert_allclose(state[:, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(state[:, 1:5], 0.0, atol=1e-6)
        np.testing.assert_allclose(state[:, 5], [0.0, 0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(state[:, 6], 0.05 - self.frame_times[:-1] * 0.01, atol=1e-6)
        np.testing.assert_allclose(state[:, 7], 2.0)
        # Same convention as the state, one frame ahead
        np.testing.assert_allclose(action[:-1, :6], state[1:, :6], atol=1e-6)
        np.testing.assert_allclose(action[-1, [0, 5]], [0.4, 0.4], atol=1e-6)

    def test_gripper_target_falls_back_to_next_measured_width(self):
        state, action = self.features()
        np.testing.assert_allclose(action[:-1, 6], state[1:, 6], atol=1e-6)

    def test_gripper_target_is_last_command_at_or_before_t(self):
        state, action = self.features(command_times=[0.1, 0.25], commands=[0.0, 0.03])
        # Frame 0 precedes the first command, so it falls back to the width at t+1
        np.testing.assert_allclose(action[0, 6], state[1, 6], atol=1e-6)
        np.testing.assert_allclose(action[1:, 6], [0.0, 0.0, 0.03])


# =============================================================================
# End to end, on a synthetic MCAP
# =============================================================================

HEADER = "builtin_interfaces/Time stamp\nstring frame_id"
POSE = "geometry_msgs/Point position\ngeometry_msgs/Quaternion orientation"
DEPS = {
    "std_msgs/Header": HEADER,
    "geometry_msgs/Pose": POSE,
    "geometry_msgs/PoseStamped": "std_msgs/Header header\ngeometry_msgs/Pose pose",
    "geometry_msgs/Point": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/Quaternion": "float64 x\nfloat64 y\nfloat64 z\nfloat64 w",
}
SCHEMAS = {
    "fm_skills_msgs/msg/TaskEvent": (
        "std_msgs/Header header\nuint32 episode\nuint8 TASK_START = 0\nuint8 MODE = 1\n"
        "uint8 SKILL_START = 2\nuint8 SKILL_END = 3\nuint8 TASK_END = 4\nuint8 kind\n"
        "string mode\nstring skill\nstring impl\ngeometry_msgs/PoseStamped handover_pose\n"
        "uint8 code\nbool verified"
    ),
    "anvil_msgs/msg/CommandedEEPose": "std_msgs/Header header\ngeometry_msgs/Pose pose\nfloat64 gripper",
    "sensor_msgs/msg/JointState": (
        "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort"
    ),
    "sensor_msgs/msg/CompressedImage": "std_msgs/Header header\nstring format\nuint8[] data",
    "control_msgs/msg/GripperCommand": "float64 position\nfloat64 max_effort",
}
T0 = 1_700_000_000.0


def msgdef(name):
    parts = [SCHEMAS[name]]
    for dep, text in DEPS.items():
        parts.append("=" * 80 + f"\nMSG: {dep}\n{text}")
    return "\n".join(parts)


def stamp(t):
    sec = int(T0 + t)
    return {"stamp": {"sec": sec, "nanosec": int(round((T0 + t - sec) * 1e9))}, "frame_id": "world"}


def pose_dict(pose):
    return {
        "position": dict(zip("xyz", pose[:3])),
        "orientation": dict(zip("xyzw", pose[3:])),
    }


def write_recording(path, handover):
    """
    5 s recording at 30 fps with a successful pick in [1, 2] s and a failed one in [3, 4] s.

    The finger is commanded open (0.04) long before the pick and closed (0.0) at 1.51 s.
    """
    from mcap_ros2.writer import Writer

    jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 128, dtype=np.uint8))[1].tobytes()
    messages = []  # (t, topic, schema, message)
    for i in range(150):
        t = i / 30
        messages.append(
            (
                t,
                "/cam_wrist_r/image_raw/compressed",
                "sensor_msgs/msg/CompressedImage",
                {"header": stamp(t), "format": "jpeg", "data": jpeg},
            )
        )
    for i in range(500):
        t = i / 100
        pose = handover + np.array([0.0, t - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        messages.append(
            (
                t,
                "/ee_pose_right",
                "anvil_msgs/msg/CommandedEEPose",
                {"header": stamp(t), "pose": pose_dict(pose), "gripper": 0.0},
            )
        )
        messages.append(
            (
                t,
                "/joint_states",
                "sensor_msgs/msg/JointState",
                {
                    "header": stamp(t),
                    "name": ["follower_r_joint1", "follower_r_finger_joint1"],
                    "position": [0.0, 0.03],
                    "velocity": [0.0, 0.0],
                    "effort": [0.0, 1.5],
                },
            )
        )
    for t, position in [(0.2, 0.04), (1.51, 0.0)]:
        messages.append(
            (
                t,
                "/motion/gripper",
                "control_msgs/msg/GripperCommand",
                {"position": position, "max_effort": 10.0},
            )
        )
    for t, kind, code, verified in [
        (1.0, SKILL_START, 0, False),
        (2.0, SKILL_END, 0, True),
        (3.0, SKILL_START, 0, False),
        (4.0, SKILL_END, 5, False),
    ]:
        messages.append(
            (
                t,
                "/task/events",
                "fm_skills_msgs/msg/TaskEvent",
                {
                    "header": stamp(t),
                    "episode": 1,
                    "kind": kind,
                    "mode": "",
                    "skill": "pick",
                    "impl": "classical",
                    "code": code,
                    "verified": verified,
                    "handover_pose": {"header": stamp(t), "pose": pose_dict(handover)},
                },
            )
        )

    messages.sort(key=lambda m: m[0])
    with open(path, "wb") as f:
        writer = Writer(f)
        schemas = {name: writer.register_msgdef(name, msgdef(name)) for name in SCHEMAS}
        for t, topic, schema, message in messages:
            writer.write_message(topic, schemas[schema], message, log_time=int((T0 + t) * 1e9))
        writer.finish()


CONFIG = "configs/mcap_converter/openarm_single_handover_ee.yaml"


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "session" / "0001.mcap"
    path.parent.mkdir()
    handover = yaw_pose(0.4, -0.2, 0.3, np.pi / 2)
    write_recording(path, handover)
    return path


@pytest.fixture
def config(request):
    root = request.config.rootpath
    return ConfigLoader.from_yaml(str(root / CONFIG))


class TestEndToEnd:
    def test_keeps_only_successful_segments_by_default(self, recording, config):
        episodes = list(iter_skill_episodes(str(recording), config, fps=30, task="pick"))
        [(segment, frames)] = episodes
        frames = list(frames)
        assert segment.succeeded
        # 31 camera frames in [1, 2] s, the last has no next pose
        assert len(frames) == 30
        state = np.array([f["observation.state"] for f in frames])
        action = np.array([f["action"] for f in frames])
        # World +y is H's +x; the TCP starts at H and moves 1 m/s
        np.testing.assert_allclose(state[0, :6], 0.0, atol=1e-3)
        np.testing.assert_allclose(state[:, 1:6], 0.0, atol=1e-3)
        # The action pose is the next frame's state pose; /ee_pose_right is at 100 Hz
        np.testing.assert_allclose(action[:-1, :6], state[1:, :6], atol=1e-6)
        np.testing.assert_allclose(action[:, 0], np.arange(1, 31) / 30, atol=0.0051)
        np.testing.assert_allclose(state[:, 6:], [[0.03, 1.5]] * 30, atol=1e-6)
        # Commanded open until 1.51 s, closed after: frames 0-15 open, 16-29 closed
        np.testing.assert_allclose(action[:, 6], [0.04] * 16 + [0.0] * 14, atol=1e-6)
        assert frames[0]["observation.images.wrist_r"].shape == (480, 640, 3)
        assert {f["task"] for f in frames} == {"pick"}

    def test_gripper_target_without_command_topic_is_next_measured_width(self, recording, config):
        config.skill_segments.gripper_command_topic = "/not/recorded"
        [(_, frames)] = iter_skill_episodes(str(recording), config, fps=30, task="pick")
        np.testing.assert_allclose([f["action"][6] for f in frames], 0.03, atol=1e-6)

    @pytest.mark.parametrize("missing", ["ee_pose_topic", "gripper_joint"])
    def test_segment_without_ee_or_finger_data_yields_nothing(self, recording, config, missing):
        [(segment, _)] = iter_skill_episodes(str(recording), config, fps=30, task="pick")
        setattr(config.skill_segments, missing, "/not/recorded")
        assert list(extract_segment_frames(str(recording), segment, config, 30, "pick")) == []

    def test_include_failures_labels_them(self, recording, config):
        config.skill_segments.include_failures = True
        episodes = list(iter_skill_episodes(str(recording), config, fps=30, task="pick"))
        assert [next(frames)["task"] for _, frames in episodes] == ["pick", "pick [failed]"]

    def test_convert_writes_ee_features(self, recording, config, tmp_path):
        from mcap_converter.cli.convert import convert_skill_session

        config.camera_topic_mapping = {"/cam_wrist_r/image_raw/compressed": "wrist_r"}
        config.image_resolution = [64, 48]
        out = tmp_path / "out"
        dataset = convert_skill_session(
            output_dir=str(out),
            repo_id="test/handover",
            mcap_files=[recording],
            config=config,
            fps=30,
            task="pick",
            debug_plot_episodes=0,
        )
        assert dataset.meta.total_episodes == 1
        features = json.loads((out / "meta" / "info.json").read_text())["features"]
        state_names = ["x", "y", "z", "rx", "ry", "rz", "finger_width_m", "finger_effort"]
        action_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
        assert features["observation.state"]["names"] == state_names
        assert features["action"]["names"] == action_names
        assert features["observation.state"]["dtype"] == features["action"]["dtype"] == "float32"
        assert "observation.images.wrist_r" in features


class TestSampleIndices:
    JITTER = np.array([0.0, 1e-4, -1e-4])

    def test_downsamples_to_grid_without_drift(self):
        # 10 s of jittered 60 Hz stamps sampled at 30 Hz: every other frame, none skipped
        times = 1_700_000_000.0 + np.arange(601) / 60 + np.resize(self.JITTER, 601)
        np.testing.assert_array_equal(sample_indices(times, 30), np.arange(0, 601, 2))

    def test_keeps_every_frame_at_source_rate(self):
        times = 1_700_000_000.0 + np.arange(900) / 30 + np.resize(self.JITTER, 900)
        np.testing.assert_array_equal(sample_indices(times, 30), np.arange(900))


def test_resume_is_rejected_with_skill_segments(request, tmp_path):
    from mcap_converter.cli.convert import main

    config = str(request.config.rootpath / CONFIG)
    with pytest.raises(SystemExit) as exit_info:
        main(["-i", str(tmp_path), "--config", config, "--resume", "--hf-user", "test"])
    assert exit_info.value.code == 1
