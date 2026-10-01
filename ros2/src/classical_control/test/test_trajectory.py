"""Tests for min-jerk Cartesian trajectories."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.trajectory import (
    Pose,
    min_jerk,
    plan_path,
    plan_segment,
    segment_duration,
)
from scipy.spatial.transform import Rotation

RATE_HZ = 30.0
V_MAX = 0.1
W_MAX = 0.5


def pose_at(x: float, y: float, z: float, rotvec=(0.0, 0.0, 0.0)) -> Pose:
    return Pose.from_rotation((x, y, z), Rotation.from_rotvec(rotvec))


def step_sizes(start: Pose, samples: list[Pose]) -> tuple[np.ndarray, np.ndarray]:
    poses = [start, *samples]
    linear = np.array(
        [np.linalg.norm(b.position - a.position) for a, b in zip(poses[:-1], poses[1:])]
    )
    angular = np.array(
        [(a.rotation.inv() * b.rotation).magnitude() for a, b in zip(poses[:-1], poses[1:])]
    )
    return linear, angular


def test_min_jerk_profile_endpoints_and_peak():
    assert min_jerk(0.0) == pytest.approx(0.0)
    assert min_jerk(1.0) == pytest.approx(1.0)
    assert min_jerk(0.5) == pytest.approx(0.5)
    tau = np.linspace(0.0, 1.0, 100001)
    peak_rate = np.max(np.diff(min_jerk(tau)) / np.diff(tau))
    assert peak_rate == pytest.approx(1.875, rel=1e-4)


def test_segment_ends_exactly_at_the_target():
    start = pose_at(0.3, 0.0, 0.2)
    end = pose_at(0.4, 0.1, 0.15, rotvec=(0.0, 0.0, 0.3))

    samples = plan_segment(start, end, V_MAX, W_MAX, RATE_HZ)

    assert samples[-1] is end
    # The start is already commanded, so the first sample moves away from it.
    assert not np.allclose(samples[0].position, start.position)


def test_sample_count_matches_rate_and_duration():
    start = pose_at(0.3, 0.0, 0.2)
    end = pose_at(0.5, 0.0, 0.2)

    samples = plan_segment(start, end, V_MAX, W_MAX, RATE_HZ)

    duration = 1.875 * 0.2 / V_MAX
    assert len(samples) == int(np.ceil(duration * RATE_HZ))


def test_linear_step_never_exceeds_the_velocity_bound():
    start = pose_at(0.3, -0.1, 0.25)
    end = pose_at(0.45, 0.12, 0.1)

    linear, _ = step_sizes(start, plan_segment(start, end, V_MAX, W_MAX, RATE_HZ))

    assert np.max(linear) <= V_MAX / RATE_HZ + 1e-12
    # Tight, not wildly conservative: the peak step is close to the bound.
    assert np.max(linear) > 0.95 * V_MAX / RATE_HZ


def test_angular_step_never_exceeds_the_velocity_bound():
    start = pose_at(0.3, 0.0, 0.2)
    end = pose_at(0.3, 0.0, 0.2, rotvec=(0.4, -0.9, 1.1))

    _, angular = step_sizes(start, plan_segment(start, end, V_MAX, W_MAX, RATE_HZ))

    assert np.max(angular) <= W_MAX / RATE_HZ + 1e-9
    assert np.max(angular) > 0.95 * W_MAX / RATE_HZ


def test_slerp_takes_the_shortest_arc():
    # Yaw 170° to -170° is 20° through 180°, not 340° back through 0°.
    start = pose_at(0.3, 0.0, 0.2, rotvec=(0.0, 0.0, np.deg2rad(170.0)))
    end = pose_at(0.3, 0.0, 0.2, rotvec=(0.0, 0.0, np.deg2rad(-170.0)))

    samples = plan_segment(start, end, V_MAX, W_MAX, RATE_HZ)
    _, angular = step_sizes(start, samples)

    assert np.sum(angular) == pytest.approx(np.deg2rad(20.0), abs=1e-6)
    yaws = np.abs([s.rotation.as_euler("xyz")[2] for s in samples])
    assert np.all(yaws >= np.deg2rad(170.0) - 1e-6)


def test_negated_quaternion_is_not_a_rotation():
    start = pose_at(0.3, 0.0, 0.2, rotvec=(0.0, 0.0, 0.5))
    end = Pose(start.position, -start.quat_xyzw)

    assert segment_duration(start, end, V_MAX, W_MAX) == pytest.approx(0.0)
    assert plan_segment(start, end, V_MAX, W_MAX, RATE_HZ) == [end]


def test_zero_length_segment_yields_only_the_target():
    start = pose_at(0.3, 0.0, 0.2)
    end = pose_at(0.3, 0.0, 0.2)

    assert plan_segment(start, end, V_MAX, W_MAX, RATE_HZ) == [end]


def test_min_duration_holds_a_short_move():
    start = pose_at(0.3, 0.0, 0.2)
    end = pose_at(0.3, 0.0, 0.2)

    samples = plan_segment(start, end, V_MAX, W_MAX, RATE_HZ, min_duration=1.0)

    assert len(samples) == 30
    for sample in samples:
        np.testing.assert_allclose(sample.position, start.position)


def test_limits_must_be_positive():
    start = pose_at(0.3, 0.0, 0.2)
    with pytest.raises(ValueError):
        plan_segment(start, start, 0.0, W_MAX)
    with pytest.raises(ValueError):
        plan_segment(start, start, V_MAX, W_MAX, rate_hz=0.0)


def test_path_visits_every_waypoint():
    waypoints = [pose_at(0.3, 0.0, 0.2), pose_at(0.35, 0.0, 0.2), pose_at(0.35, 0.05, 0.1)]

    samples = plan_path(waypoints, V_MAX, W_MAX, RATE_HZ)

    assert samples[-1] is waypoints[-1]
    assert any(s is waypoints[1] for s in samples)
    linear, _ = step_sizes(waypoints[0], samples)
    assert np.max(linear) <= V_MAX / RATE_HZ + 1e-12


def test_pose_normalises_and_is_immutable():
    pose = Pose([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 2.0])

    np.testing.assert_allclose(pose.quat_xyzw, [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        pose.position[0] = 1.0
    with pytest.raises(ValueError):
        Pose([0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        Pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
