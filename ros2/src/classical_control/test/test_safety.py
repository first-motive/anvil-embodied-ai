"""Tests for the safety limiter, the last software guard before the hardware."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.safety import (
    FreshnessGate,
    HardwareStateGate,
    SafetyLimiter,
    SafetyLimits,
)
from classical_control.trajectory import Pose
from scipy.spatial.transform import Rotation

TABLE_Z = 0.0
Z_MARGIN = 0.02


@pytest.fixture
def limits() -> SafetyLimits:
    return SafetyLimits(
        workspace_min_m=(-0.5, -0.5, -0.1),
        workspace_max_m=(0.5, 0.5, 0.9),
        table_z=TABLE_Z,
        z_margin_m=Z_MARGIN,
    )


@pytest.fixture
def limiter(limits) -> SafetyLimiter:
    return SafetyLimiter(limits)


def pose_at(x: float, y: float, z: float, rotvec=(0.0, 0.0, 0.0)) -> Pose:
    return Pose.from_rotation((x, y, z), Rotation.from_rotvec(rotvec))


def rotation_between(a: Pose, b: Pose) -> np.ndarray:
    return (a.rotation.inv() * b.rotation).as_rotvec()


def test_defaults_match_the_plan(limits):
    assert limits.max_position_step_m == 0.005
    assert limits.max_rotation_step_rad == 0.05
    assert (limits.gripper_min_m, limits.gripper_max_m) == (0.0, 0.05)


def test_small_step_passes_through(limiter):
    previous = pose_at(0.1, 0.0, 0.3)
    target = pose_at(0.102, 0.0, 0.3, rotvec=(0.0, 0.0, 0.01))

    clamped, report = limiter.clamp(target, previous)

    assert not report.any_clamped
    np.testing.assert_allclose(clamped.position, target.position)
    np.testing.assert_allclose(rotation_between(target, clamped), 0.0, atol=1e-12)


def test_large_position_step_is_clamped_along_its_direction(limiter):
    previous = pose_at(0.1, 0.0, 0.3)
    target = pose_at(0.4, 0.2, 0.4)

    clamped, report = limiter.clamp(target, previous)

    assert report.position_clamped
    step = clamped.position - previous.position
    assert np.linalg.norm(step) == pytest.approx(0.005)
    wanted = target.position - previous.position
    np.testing.assert_allclose(step / np.linalg.norm(step), wanted / np.linalg.norm(wanted))


def test_large_rotation_step_is_clamped_to_the_limit(limiter):
    previous = pose_at(0.1, 0.0, 0.3)
    target = pose_at(0.1, 0.0, 0.3, rotvec=(0.0, 0.0, 0.8))

    clamped, report = limiter.clamp(target, previous)

    assert report.rotation_clamped
    np.testing.assert_allclose(rotation_between(previous, clamped), [0.0, 0.0, 0.05], atol=1e-9)


@pytest.mark.parametrize(
    "axis",
    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.577, 0.577, 0.577)],
)
def test_half_turn_is_shortened_not_discarded(limiter, axis):
    # A half-turn has no axis in the antisymmetric part of its matrix. Reading
    # it as "no rotation" would silently drop the command while still reporting
    # a clamp, so check the clamped step is the limit about the same axis.
    previous = pose_at(0.1, 0.0, 0.3, rotvec=(0.2, -0.1, 0.4))
    unit = np.asarray(axis) / np.linalg.norm(axis)
    target = Pose.from_rotation(
        previous.position, previous.rotation * Rotation.from_rotvec(unit * np.pi)
    )

    clamped, report = limiter.clamp(target, previous)

    assert report.rotation_clamped
    rotvec = rotation_between(previous, clamped)
    assert np.linalg.norm(rotvec) == pytest.approx(0.05, abs=1e-9)
    assert abs(float(np.dot(rotvec / np.linalg.norm(rotvec), unit))) == pytest.approx(1.0)


def test_target_outside_the_box_is_pulled_back(limiter):
    # Sit on the box edge so the step limit does not mask the box limit.
    previous = pose_at(0.5, 0.0, 0.3)
    target = pose_at(0.504, -0.003, 0.3)

    clamped, report = limiter.clamp(target, previous)

    assert report.workspace_clamped
    assert not report.position_clamped
    np.testing.assert_allclose(clamped.position, [0.5, -0.003, 0.3])


def test_z_floor_keeps_the_tcp_off_the_table(limiter):
    previous = pose_at(0.2, 0.0, TABLE_Z + Z_MARGIN + 0.001)
    target = pose_at(0.2, 0.0, TABLE_Z - 0.05)

    clamped, report = limiter.clamp(target, previous)

    assert report.workspace_clamped
    assert clamped.position[2] == pytest.approx(TABLE_Z + Z_MARGIN)


def test_box_floor_wins_when_it_is_above_the_table_floor():
    limits = SafetyLimits(
        workspace_min_m=(-0.5, -0.5, 0.1),
        workspace_max_m=(0.5, 0.5, 0.9),
        table_z=0.0,
        z_margin_m=0.02,
    )

    assert limits.min_z_m == pytest.approx(0.1)


def test_clamped_result_never_leaves_the_box_from_inside(limiter):
    rng = np.random.default_rng(0)
    previous = pose_at(0.49, 0.49, 0.025)
    for _ in range(200):
        target = pose_at(*rng.uniform(-1.0, 1.0, size=3), rotvec=rng.normal(size=3))
        clamped, _ = limiter.clamp(target, previous)
        assert limiter.in_workspace(clamped)
        assert np.linalg.norm(clamped.position - previous.position) <= 0.005 + 1e-12
        assert np.linalg.norm(rotation_between(previous, clamped)) <= 0.05 + 1e-9
        previous = clamped


def test_previous_outside_the_box_walks_back_at_the_step_limit(limiter):
    previous = pose_at(0.2, 0.0, TABLE_Z - 0.03)
    target = pose_at(0.2, 0.0, 0.3)

    clamped, report = limiter.clamp(target, previous)

    assert report.position_clamped
    assert clamped.position[2] == pytest.approx(previous.position[2] + 0.005)


def test_in_workspace(limiter):
    assert limiter.in_workspace(pose_at(0.0, 0.0, 0.3))
    assert limiter.in_workspace(pose_at(0.5, -0.5, TABLE_Z + Z_MARGIN))
    assert not limiter.in_workspace(pose_at(0.6, 0.0, 0.3))
    assert not limiter.in_workspace(pose_at(0.0, 0.0, TABLE_Z + Z_MARGIN - 0.001))
    assert not limiter.in_workspace(pose_at(0.0, 0.0, 0.95))


def test_clamping_does_not_mutate_its_inputs(limiter):
    previous = pose_at(0.1, 0.0, 0.3)
    target = pose_at(0.4, 0.0, 0.3)
    original = target.position.copy()

    limiter.clamp(target, previous)

    np.testing.assert_array_equal(target.position, original)


def test_gripper_is_clamped_into_range(limiter):
    assert limiter.clamp_gripper(0.03) == (0.03, False)
    assert limiter.clamp_gripper(0.08) == (0.05, True)
    assert limiter.clamp_gripper(-0.01) == (0.0, True)


def test_limits_from_dict():
    limits = SafetyLimits.from_dict(
        {
            "workspace_min_m": [-0.5, -0.5, 0.0],
            "workspace_max_m": [0.5, 0.5, 0.9],
            "table_z": 0.01,
            "z_margin_m": 0.02,
            "max_position_step_m": 0.004,
        }
    )

    assert limits.workspace_min_m == (-0.5, -0.5, 0.0)
    assert limits.min_z_m == pytest.approx(0.03)
    assert limits.max_position_step_m == 0.004
    assert limits.max_rotation_step_rad == 0.05


@pytest.mark.parametrize(
    "override",
    [
        {"workspace_min_m": [0.5, 0.0, 0.0]},
        {"workspace_min_m": [0.0, 0.0]},
        {"table_z": 1.0},
        {"max_position_step_m": 0.0},
        {"gripper_min_m": 0.06},
        {"z_margin_m": -0.01},
        {"max_speed": 1.0},
    ],
)
def test_invalid_limits_are_rejected(override):
    data = {
        "workspace_min_m": [-0.5, -0.5, 0.0],
        "workspace_max_m": [0.5, 0.5, 0.9],
        "table_z": 0.0,
    }
    with pytest.raises(ValueError):
        SafetyLimits.from_dict({**data, **override})


def test_gate_is_closed_before_the_probe_answers():
    gate = HardwareStateGate()

    # The controller publishes only on change and boots active, so silence
    # means "not yet confirmed", never "safe to move".
    assert gate.state is None
    assert not gate.allows_commands


def test_gate_opens_when_the_probe_is_accepted():
    gate = HardwareStateGate()

    gate.confirm_active(True)

    assert gate.allows_commands


def test_refused_probe_latches_the_gate_shut():
    gate = HardwareStateGate()

    # The controller refuses only when the hardware is not active, and that
    # needs a restart to clear.
    gate.confirm_active(False)
    gate.confirm_active(True)

    assert not gate.allows_commands


@pytest.mark.parametrize("state", ["pause", "estop", "", "ACTIVE", "unknown"])
def test_a_stop_latches_the_gate_shut(state):
    gate = HardwareStateGate()
    gate.confirm_active(True)
    assert gate.allows_commands

    gate.update(state)

    assert not gate.allows_commands
    assert gate.state == state


def test_active_after_a_stop_does_not_reopen_the_gate():
    gate = HardwareStateGate()
    gate.confirm_active(True)
    gate.update("estop")

    gate.update("active")
    gate.confirm_active(True)

    # Matches the hardware: only a restart clears a stop.
    assert not gate.allows_commands


def test_gate_reports_only_real_changes():
    gate = HardwareStateGate()

    assert gate.update("pause")
    assert not gate.update("pause")


INPUTS = ("can_pose", "paper_pose", "ee_pose_right")


def test_inputs_never_seen_are_stale():
    gate = FreshnessGate(max_age_s=0.2)

    assert gate.stale(INPUTS, now=100.0) == INPUTS


def test_fresh_inputs_are_not_stale():
    gate = FreshnessGate(max_age_s=0.2)
    for key in INPUTS:
        gate.mark(key, now=100.0)

    assert gate.stale(INPUTS, now=100.1) == ()


def test_one_frozen_stream_is_reported_on_its_own():
    gate = FreshnessGate(max_age_s=0.2)
    for key in INPUTS:
        gate.mark(key, now=100.0)
    for key in ("paper_pose", "ee_pose_right"):
        gate.mark(key, now=100.5)

    assert gate.stale(INPUTS, now=100.5) == ("can_pose",)


def test_freshness_recovers_when_the_stream_resumes():
    gate = FreshnessGate(max_age_s=0.2)
    gate.mark("can_pose", now=100.0)
    assert gate.stale(("can_pose",), now=100.5)

    gate.mark("can_pose", now=100.5)

    assert gate.stale(("can_pose",), now=100.5) == ()


def test_age_limit_must_be_positive():
    with pytest.raises(ValueError):
        FreshnessGate(max_age_s=0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_pose_cannot_be_built(bad):
    with pytest.raises(ValueError, match="finite"):
        Pose((bad, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        Pose((0.0, 0.0, 0.3), (bad, 0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    "override",
    [
        {"workspace_min_m": (float("nan"), -0.5, -0.1)},
        {"workspace_max_m": (0.5, float("inf"), 0.9)},
        {"table_z": float("nan")},
        {"z_margin_m": float("nan")},
        {"max_position_step_m": float("nan")},
        {"max_rotation_step_rad": float("inf")},
        {"gripper_max_m": float("nan")},
    ],
)
def test_non_finite_limits_are_rejected(override):
    params = {
        "workspace_min_m": (-0.5, -0.5, -0.1),
        "workspace_max_m": (0.5, 0.5, 0.9),
        "table_z": TABLE_Z,
        **override,
    }
    with pytest.raises(ValueError, match="finite"):
        SafetyLimits(**params)


def test_non_finite_gripper_command_is_rejected(limiter):
    with pytest.raises(ValueError, match="finite"):
        limiter.clamp_gripper(float("nan"))
