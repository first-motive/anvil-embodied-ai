"""Tests for the pick-place phase machine, driven by a fake clock."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.safety import SafetyLimiter, SafetyLimits
from classical_control.task_machine import (
    MIN_SPEED_SCALE,
    ErrorCode,
    GoalOverrides,
    Phase,
    PickPlaceTask,
    TaskConfig,
    effective_speed_scale,
    goal_overrides,
    mined_targets,
    place_target_in_workspace,
    plan_steps,
    required_detections,
)
from classical_control.trajectory import Pose
from scipy.spatial.transform import Rotation

RATE_HZ = 30.0
DT = 1.0 / RATE_HZ
GRASP_QUAT = (0.672, 0.064, 0.736, 0.055)
HOME = Pose([0.396, -0.185, 0.506], GRASP_QUAT)
CAN = (0.40, -0.30, 0.28)
PAPER = (0.30, -0.05, 0.22)
CLOSED_ON_CAN = 0.0185

PLACE_TARGET = (0.20, -0.40, 0.0)
OVERRIDE_LIMITS = {
    "max_grasp_dz_m": 0.02,
    "max_yaw_offset_rad": 0.35,
    "gripper_range_m": (0.0, 0.05),
    "max_closure_m": 0.010,
}

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


def test_height_offset_moves_grasp_and_place_together():
    mined = {
        "grasp_z": {"median": 0.341},
        "place_z": {"median": 0.344},
        "grasp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    grasp_z, place_z, _ = mined_targets(mined, -0.035)
    assert grasp_z == pytest.approx(0.306)
    assert place_z == pytest.approx(0.309)


# ---- goal overrides ----------------------------------------------------------


def test_default_overrides_plan_the_same_steps(config):
    plain = plan_steps(config, CAN, PAPER)
    defaults = plan_steps(config, CAN, PAPER, goal_overrides(0.0, 0.0, 0.0, **OVERRIDE_LIMITS))
    assert [(s.phase, s.gripper, s.dwell_s) for s in plain] == [
        (s.phase, s.gripper, s.dwell_s) for s in defaults
    ]
    for a, b in zip(plain, defaults):
        np.testing.assert_array_equal(a.target.position, b.target.position)
        np.testing.assert_array_equal(a.target.quat_xyzw, b.target.quat_xyzw)


def test_place_target_sets_place_xy(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PLACE_TARGET)
    for phase in (Phase.TRANSIT, Phase.LOWER, Phase.OPEN, Phase.RETREAT):
        np.testing.assert_allclose(step_target(task, phase).position[:2], PLACE_TARGET[:2])
    run(task)
    assert task.result is ErrorCode.SUCCESS


def test_place_target_outside_workspace_is_clamped_into_it(config, limiter):
    target = place_target_in_workspace(limiter.limits, (0.90, -0.80, 0.0))
    assert target == (0.65, -0.55, 0.0)
    task = PickPlaceTask(config, limiter, CAN, target)
    assert task.result is None
    np.testing.assert_allclose(step_target(task, Phase.LOWER).position[:2], (0.65, -0.55))


def test_non_finite_place_target_is_refused(limiter):
    with pytest.raises(ValueError, match="finite"):
        place_target_in_workspace(limiter.limits, (float("nan"), 0.0, 0.0))


def test_grasp_dz_moves_grasp_and_place_heights(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER, overrides=GoalOverrides(grasp_dz_m=-0.01))
    assert step_target(task, Phase.DESCEND).position[2] == pytest.approx(config.grasp_z_m - 0.01)
    assert step_target(task, Phase.LOWER).position[2] == pytest.approx(config.place_z_m - 0.01)
    assert step_target(task, Phase.LIFT).position[2] == pytest.approx(
        config.grasp_z_m - 0.01 + config.approach_height_m
    )


def test_overrides_are_clamped_to_their_limits():
    high = goal_overrides(0.5, 0.2, 2.0, **OVERRIDE_LIMITS)
    low = goal_overrides(-0.5, 0.001, -2.0, **OVERRIDE_LIMITS)
    assert high == GoalOverrides(grasp_dz_m=0.02, closure_m=0.010, yaw_offset_rad=0.35)
    assert low == GoalOverrides(grasp_dz_m=-0.02, closure_m=0.001, yaw_offset_rad=-0.35)
    assert goal_overrides(0.0, -1.0, 0.0, **OVERRIDE_LIMITS).closure_m is None


@pytest.mark.parametrize("field", range(3))
def test_nan_override_is_refused(field):
    values = [0.0, 0.0, 0.0]
    values[field] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        goal_overrides(*values, **OVERRIDE_LIMITS)


def test_closure_is_commanded_at_close(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER, overrides=GoalOverrides(closure_m=0.012))
    close = next(step for step in task.steps if step.phase is Phase.CLOSE)
    assert close.gripper == 0.012
    assert all(
        step.gripper == 0.012 for step in task.steps if step.phase in (Phase.LIFT, Phase.LOWER)
    )


@pytest.mark.parametrize(
    ("finger", "expected"),
    [
        # An empty close stops at the commanded closure, above the plain threshold.
        (0.010, ErrorCode.GRASP_MISSED),
        (0.010 + 0.007, ErrorCode.GRASP_MISSED),
        # At the largest allowed closure a held can still reads as a grasp.
        (CLOSED_ON_CAN, ErrorCode.SUCCESS),
    ],
)
def test_grasp_missed_check_is_measured_from_closure(config, limiter, finger, expected):
    closure = goal_overrides(0.0, 0.05, 0.0, **OVERRIDE_LIMITS).closure_m
    assert closure == 0.010
    task = PickPlaceTask(config, limiter, CAN, PAPER, overrides=GoalOverrides(closure_m=closure))
    run(task, finger=finger)
    assert task.result is expected


def test_yaw_offset_turns_grasp_about_world_z_and_leaves_home(config, limiter):
    task = PickPlaceTask(config, limiter, CAN, PAPER, overrides=GoalOverrides(yaw_offset_rad=0.3))
    base = Rotation.from_quat(GRASP_QUAT)
    for step in task.steps:
        turned = Rotation.from_quat(step.target.quat_xyzw)
        if step.phase is Phase.HOME:
            np.testing.assert_allclose(step.target.quat_xyzw, HOME.quat_xyzw)
            continue
        relative = (turned * base.inv()).as_rotvec()
        np.testing.assert_allclose(relative, [0.0, 0.0, 0.3], atol=1e-9)


def test_home_only_plans_one_home_step_and_succeeds(config, limiter):
    task = PickPlaceTask.home_only(config, limiter)
    assert [(s.phase, s.gripper) for s in task.steps] == [(Phase.HOME, config.gripper_home_m)]
    phases, commands = run(task, finger=0.0)
    assert distinct(phases) == [Phase.LOCALISE, Phase.HOME]
    assert task.result is ErrorCode.SUCCESS
    np.testing.assert_allclose(commands[-1].pose.position, HOME.position)


def test_home_only_dry_run_commands_nothing(config, limiter):
    task = PickPlaceTask.home_only(config, limiter, dry_run=True)
    assert task.result is ErrorCode.SUCCESS
    assert task.tick(0.0, HOME, 0.045) is None


@pytest.mark.parametrize(
    ("home_only", "place_target_set", "needed"),
    [
        (False, False, ("can", "paper")),
        (False, True, ("can",)),
        (True, False, ()),
        (True, True, ()),
    ],
)
def test_required_detections_follow_the_goal(home_only, place_target_set, needed):
    assert required_detections(home_only=home_only, place_target_set=place_target_set) == needed
