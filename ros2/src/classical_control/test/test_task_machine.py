"""Tests for the pick-place phase machine, driven by a fake clock."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.safety import SafetyLimiter, SafetyLimits
from classical_control.task_machine import (
    MIN_SPEED_SCALE,
    ErrorCode,
    Phase,
    PickPlaceTask,
    TaskConfig,
    effective_speed_scale,
    mined_targets,
)
from classical_control.trajectory import Pose

RATE_HZ = 30.0
DT = 1.0 / RATE_HZ
GRASP_QUAT = (0.672, 0.064, 0.736, 0.055)
HOME = Pose([0.396, -0.185, 0.506], GRASP_QUAT)
CAN = (0.40, -0.30, 0.28)
PAPER = (0.30, -0.05, 0.22)
CLOSED_ON_CAN = 0.0185

HAPPY_ORDER = [
    Phase.LOCALISE,
    Phase.PRE_GRASP,
    Phase.DESCEND,
    Phase.CLOSE,
    Phase.LIFT,
    Phase.TRANSIT,
    Phase.LOWER,
    Phase.OPEN,
    Phase.RETREAT,
    Phase.HOME,
]


@pytest.fixture
def limiter() -> SafetyLimiter:
    return SafetyLimiter(
        SafetyLimits(
            workspace_min_m=(0.05, -0.55, 0.0),
            workspace_max_m=(0.65, 0.15, 0.70),
            table_z=0.22,
            z_margin_m=0.01,
        )
    )


@pytest.fixture
def config() -> TaskConfig:
    return TaskConfig(
        grasp_z_m=0.345,
        place_z_m=0.347,
        grasp_orientation_xyzw=GRASP_QUAT,
        home=HOME,
        can_xy_bias_m=(0.01, -0.02),
        rate_hz=RATE_HZ,
    )


def run(task: PickPlaceTask, finger: float = CLOSED_ON_CAN, max_ticks: int = 20_000):
    """Tick until done; return the phases seen per tick and the commands emitted."""
    now = 0.0
    phases, commands = [task.phase], []
    for _ in range(max_ticks):
        if task.done:
            break
        command = task.tick(now, HOME, finger)
        if command is not None:
            commands.append(command)
        phases.append(task.phase)
        now += DT
    return phases, commands


def distinct(phases: list[Phase]) -> list[Phase]:
    return [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]


def step_target(task: PickPlaceTask, phase: Phase) -> Pose:
    return next(step.target for step in task.steps if step.phase is phase)


def test_happy_path_visits_every_phase_in_order(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    phases, commands = run(task)
    assert distinct(phases) == HAPPY_ORDER
    assert task.result is ErrorCode.SUCCESS
    np.testing.assert_allclose(commands[-1].pose.position, HOME.position)
    assert commands[-1].gripper == config.gripper_home_m


def test_first_command_echoes_measured_pose_and_finger(config, limiter):
    measured = Pose([0.39, -0.19, 0.50], (0.6, 0.1, 0.75, 0.05))
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    first = task.tick(0.0, measured, 0.044)
    np.testing.assert_array_equal(first.pose.position, measured.position)
    np.testing.assert_array_equal(first.pose.quat_xyzw, measured.quat_xyzw)
    assert first.gripper == 0.044


def test_streamed_steps_respect_speed_limit(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    _, commands = run(task)
    steps = np.diff([c.pose.position for c in commands], axis=0)
    assert np.linalg.norm(steps, axis=1).max() <= config.v_max_mps / RATE_HZ + 1e-9


def test_out_of_workspace_fails_before_anything_moves(config, limiter):
    task = PickPlaceTask(config, limiter, (0.90, -0.30, 0.28), PAPER)
    assert task.done
    assert task.result is ErrorCode.OUT_OF_WORKSPACE
    assert task.tick(0.0, HOME, 0.045) is None


def test_empty_close_holds_with_grasp_missed(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    phases, commands = run(task, finger=0.0)
    assert task.result is ErrorCode.GRASP_MISSED
    assert task.phase is Phase.HOLD
    assert Phase.LIFT not in phases
    grasp = step_target(task, Phase.DESCEND)
    np.testing.assert_allclose(commands[-1].pose.position, grasp.position)
    assert commands[-1].gripper == config.gripper_closed_m


def test_cancel_mid_transit_holds_last_command(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    now, last, transit_ticks = 0.0, None, 0
    while transit_ticks < 10:
        last = task.tick(now, HOME, CLOSED_ON_CAN)
        transit_ticks += task.phase is Phase.TRANSIT
        now += DT
    task.cancel()
    held = task.tick(now, HOME, CLOSED_ON_CAN)
    assert task.phase is Phase.HOLD
    assert task.result is ErrorCode.CANCELLED
    assert held is last
    assert task.tick(now + DT, HOME, CLOSED_ON_CAN) is last


def test_fault_holds_and_reports_its_code(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    first = task.tick(0.0, HOME, 0.045)
    task.fault(ErrorCode.STALE_POSE)
    task.fault(ErrorCode.HARDWARE_NOT_ACTIVE)
    assert task.tick(DT, None, 0.045) is first
    assert task.result is ErrorCode.STALE_POSE
    assert task.phase is Phase.HOLD


def test_fault_before_first_tick_commands_nothing(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    task.fault(ErrorCode.HARDWARE_NOT_ACTIVE)
    assert task.tick(0.0, HOME, 0.045) is None
    assert task.result is ErrorCode.HARDWARE_NOT_ACTIVE


def test_half_speed_takes_longer(config, limiter):
    full = PickPlaceTask(config, limiter, CAN, PAPER, speed_scale=1.0)
    half = PickPlaceTask(config, limiter, CAN, PAPER, speed_scale=0.5)
    assert half.speed_limits == pytest.approx((config.v_max_mps * 0.5, config.w_max_radps * 0.5))
    dwell_ticks = (config.close_dwell_s + config.open_dwell_s) * RATE_HZ
    full_motion = len(run(full)[1]) - dwell_ticks
    half_motion = len(run(half)[1]) - dwell_ticks
    assert half_motion == pytest.approx(2 * full_motion, rel=0.05)


@pytest.mark.parametrize(
    ("requested", "expected"), [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (0.01, MIN_SPEED_SCALE)]
)
def test_speed_scale_is_clamped(requested, expected):
    assert effective_speed_scale(requested) == expected


def test_dry_run_plans_waypoints_and_commands_nothing(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER, dry_run=True)
    assert task.result is ErrorCode.SUCCESS
    assert [s.phase for s in task.steps] == HAPPY_ORDER[1:]
    assert task.tick(0.0, HOME, 0.045) is None


def test_grasp_at_biased_can_xy_and_place_at_paper_xy(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER)
    grasp = step_target(task, Phase.DESCEND)
    place = step_target(task, Phase.LOWER)
    pre_grasp = step_target(task, Phase.PRE_GRASP)
    np.testing.assert_allclose(grasp.position, [0.41, -0.32, config.grasp_z_m])
    np.testing.assert_allclose(place.position, [PAPER[0], PAPER[1], config.place_z_m])
    np.testing.assert_allclose(pre_grasp.position[2], config.grasp_z_m + config.approach_height_m)
    np.testing.assert_allclose(grasp.quat_xyzw, Pose([0, 0, 0], GRASP_QUAT).quat_xyzw)


def test_mined_targets_reads_medians_and_requires_place():
    mined = {
        "grasp_z": {"median": 0.345},
        "place_z": {"median": 0.347},
        "grasp_orientation_xyzw": list(GRASP_QUAT),
    }
    assert mined_targets(mined) == (0.345, 0.347, GRASP_QUAT)
    del mined["place_z"]
    with pytest.raises(KeyError):
        mined_targets(mined)


def test_non_finite_detection_is_refused_at_construction(config, limiter):
    # The node filters these out as NO_CAN/NO_PAPER; this guards the machine itself.
    with pytest.raises(ValueError, match="finite"):
        PickPlaceTask(config, limiter, (float("nan"), -0.30, 0.28), PAPER)


def test_nan_speed_scale_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        effective_speed_scale(float("nan"))
