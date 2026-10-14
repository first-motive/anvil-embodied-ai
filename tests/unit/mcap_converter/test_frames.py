"""Tests for poses expressed relative to another frame."""

import numpy as np
import pytest

from mcap_converter.core.frames import (
    pose_from_frame,
    pose_in_frame,
    pose_to_vector,
    quat_to_rotvec,
    rotvec_to_quat,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def yaw_pose(x, y, z, yaw):
    return np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])


class TestFrameMath:
    def test_identity_frame_leaves_pose_unchanged(self):
        pose = yaw_pose(0.4, -0.2, 0.3, 0.7)
        np.testing.assert_allclose(pose_in_frame(pose, IDENTITY), pose, atol=1e-12)

    def test_pure_translation_subtracts_origin(self):
        frame = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        pose = yaw_pose(0.4, 0.2, 0.5, 0.3)
        local = pose_in_frame(pose, frame)
        np.testing.assert_allclose(local[:3], [0.3, 0.0, 0.2], atol=1e-12)
        np.testing.assert_allclose(local[3:], pose[3:], atol=1e-12)

    def test_rotation_about_z(self):
        # H yawed by +90 degrees: world +y is H's +x, world -x is H's +y.
        frame = yaw_pose(1.0, 0.0, 0.0, np.pi / 2)
        pose = yaw_pose(1.0, 2.0, 0.5, np.pi / 2 + 0.25)
        local = pose_in_frame(pose, frame)
        np.testing.assert_allclose(local[:3], [2.0, 0.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(pose_to_vector(local)[3:], [0.0, 0.0, 0.25], atol=1e-12)

    def test_pose_at_frame_is_zero(self):
        frame = yaw_pose(0.3, -0.1, 0.4, 1.2)
        np.testing.assert_allclose(
            pose_to_vector(pose_in_frame(frame, frame)), np.zeros(6), atol=1e-12
        )

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            frame = np.concatenate([rng.normal(size=3), rotvec_to_quat(rng.normal(size=3))])
            pose = np.concatenate([rng.normal(size=3), rotvec_to_quat(rng.normal(size=3))])
            back = pose_from_frame(pose_in_frame(pose, frame), frame)
            np.testing.assert_allclose(back[:3], pose[:3], atol=1e-12)
            # q and -q are the same rotation
            assert abs(np.dot(back[3:], pose[3:])) == pytest.approx(1.0)

    def test_rotvec_round_trip_and_sign(self):
        r = np.array([0.1, -0.4, 0.8])
        np.testing.assert_allclose(quat_to_rotvec(rotvec_to_quat(r)), r, atol=1e-12)
        np.testing.assert_allclose(quat_to_rotvec(-rotvec_to_quat(r)), r, atol=1e-12)
        np.testing.assert_allclose(quat_to_rotvec(IDENTITY[3:]), np.zeros(3))
