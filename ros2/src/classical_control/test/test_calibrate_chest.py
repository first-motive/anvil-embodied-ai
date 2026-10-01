"""Tests for the calibrate_chest sweep sequencer and its pure helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from classical_control.calibrate_chest import (
    Stage,
    SweepSequencer,
    home_from_params,
    limiter_from_params,
    load_task_params,
    outlier_views,
    parse_args,
    workspace_violations,
)
from classical_control.calibration import sweep_poses
from classical_control.trajectory import Pose
from scipy.spatial.transform import Rotation

CONFIG = Path(__file__).resolve().parents[1] / "config"
RATE_HZ = 30.0
V_MAX = 0.05
W_MAX = 0.25
SETTLE_S = 1.0
QUAT = (0.672, 0.064, 0.736, 0.055)


def _pose(x: float, y: float, z: float, quat=QUAT) -> Pose:
    return Pose([x, y, z], quat)


def _sequencer(targets, home):
    return SweepSequencer(
        targets, home, v_max=V_MAX, w_max=W_MAX, rate_hz=RATE_HZ, settle_s=SETTLE_S
    )


def _run(sequencer: SweepSequencer, start: Pose, *, max_ticks: int = 20_000):
    """Tick at RATE_HZ, capturing as soon as allowed; return (commands, stages, captures)."""
    commanded = start
    sequencer.start(start)
    commands, stages, captures = [], [], []
    for tick in range(max_ticks):
        now = tick / RATE_HZ
        if sequencer.stage is Stage.CAPTURE:
            captures.append((now, sequencer.capture_after, commanded))
            sequencer.captured()
        command = sequencer.tick(now, commanded)
        stages.append(sequencer.stage)
        if command is None:
            break
        commands.append(command)
        commanded = command
    return commands, stages, captures


def test_first_command_is_the_measured_pose():
    measured = _pose(0.30, -0.20, 0.45)
    commands, _, _ = _run(_sequencer([_pose(0.32, -0.20, 0.45)], measured), measured)
    assert np.allclose(commands[0].position, measured.position)
    assert np.allclose(commands[0].quat_xyzw, measured.quat_xyzw)


def test_visits_each_pose_settles_captures_then_goes_home():
    home = _pose(0.396, -0.185, 0.506)
    tilted = Rotation.from_rotvec([0.2, 0.0, 0.0]) * Rotation.from_quat(QUAT)
    targets = [_pose(0.30, -0.20, 0.45), Pose.from_rotation([0.25, -0.15, 0.42], tilted)]
    sequencer = _sequencer(targets, home)
    commands, stages, captures = _run(sequencer, home)

    assert sequencer.stage is Stage.DONE
    assert len(captures) == len(targets)
    for (now, after, commanded), target in zip(captures, targets):
        # The arm sits on the target, held there for the settle time before the capture.
        assert np.allclose(commanded.position, target.position)
        assert now >= after
    assert stages.count(Stage.SETTLE) >= (SETTLE_S * RATE_HZ - 1) * len(targets)
    assert np.allclose(commands[-1].position, home.position)
    assert Stage.HOME in stages


def test_steps_stay_within_the_speed_limits():
    home = _pose(0.396, -0.185, 0.506)
    tilted = Rotation.from_rotvec([0.0, 0.35, 0.0]) * Rotation.from_quat(QUAT)
    targets = [Pose.from_rotation([0.22, -0.25, 0.42], tilted), _pose(0.35, -0.10, 0.50)]
    commands, _, _ = _run(_sequencer(targets, home), home)
    for previous, current in zip(commands, commands[1:]):
        assert np.linalg.norm(current.position - previous.position) <= V_MAX / RATE_HZ + 1e-9
        angle = (previous.rotation.inv() * current.rotation).magnitude()
        assert angle <= W_MAX / RATE_HZ + 1e-9


def test_capture_opens_settle_s_after_arrival_and_holds_until_captured():
    start = _pose(0.30, -0.20, 0.45)
    sequencer = _sequencer([start], start)
    sequencer.start(start)
    now, arrived = 0.0, None
    while sequencer.stage is not Stage.CAPTURE:
        assert sequencer.capture_after is None
        sequencer.tick(now, start)
        if arrived is None and sequencer.stage is Stage.SETTLE:
            arrived = now
        now += 1.0 / RATE_HZ
    assert sequencer.capture_after == pytest.approx(arrived + SETTLE_S)
    # Without captured() it keeps holding the pose.
    assert np.allclose(sequencer.tick(now, start).position, start.position)
    assert sequencer.stage is Stage.CAPTURE


def test_abort_stops_commanding_and_keeps_its_reason():
    start = _pose(0.30, -0.20, 0.45)
    sequencer = _sequencer([_pose(0.40, -0.20, 0.45)], start)
    sequencer.start(start)
    sequencer.tick(0.0, start)
    sequencer.abort("stale inputs ee_pose")
    sequencer.abort("interrupted")
    assert sequencer.finished
    assert sequencer.tick(0.1, start) is None
    assert sequencer.abort_reason == "stale inputs ee_pose"


def test_captured_outside_capture_is_an_error():
    start = _pose(0.30, -0.20, 0.45)
    sequencer = _sequencer([start], start)
    with pytest.raises(RuntimeError):
        sequencer.captured()


def test_task_yaml_builds_the_task_limits_and_home():
    params = load_task_params(CONFIG / "task.yaml")
    limiter = limiter_from_params(params)
    home = home_from_params(params)
    assert limiter.limits.table_z == params["table_z"]
    assert np.allclose(home.position, params["home_position_m"])
    assert workspace_violations([home], limiter) == []
    below_table = _pose(0.30, -0.20, params["table_z"])
    assert workspace_violations([home, below_table], limiter) == [1]


def test_outlier_views_uses_three_times_the_median_with_a_floor():
    assert outlier_views([1.0, 1.2, 0.9, 1.1, 9.0]) == [4]
    # Below the 3 px floor nothing is dropped, however tight the rest are.
    assert outlier_views([0.1, 0.1, 0.1, 2.5]) == []
    assert outlier_views([2.0, 2.0, 2.0, 7.0]) == [3]
    assert outlier_views([]) == []


def test_default_sweep_keeps_home_orientation_within_max_tilt():
    # The first live sweep used the demos' grasp orientation and turned the marker away
    # from the chest camera; the defaults now stay near home, where it faces the camera.
    params = load_task_params(CONFIG / "task.yaml")
    home = home_from_params(params)
    args = parse_args(["--camera-yaml", "x", "--task-params", "x"])
    targets = sweep_poses(
        home.quat_xyzw,
        limiter_from_params(params),
        count=args.count,
        center_xy=tuple(args.center),
        half_extent_xy=tuple(args.half_extent),
        heights=tuple(args.heights),
        max_tilt_rad=args.max_tilt,
    )
    assert len(targets) == args.count
    tilts = [(home.rotation.inv() * pose.rotation).magnitude() for pose in targets]
    assert max(tilts) <= args.max_tilt + 1e-9


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-tilt", "0.8"],
        ["--half-extent", "0", "0.05"],
        ["--center", "nan", "-0.15"],
        ["--heights", "0.42", "inf"],
        ["--v-max", "nan"],
    ],
)
def test_sweep_options_out_of_range_are_refused(extra):
    with pytest.raises(SystemExit):
        parse_args(["--camera-yaml", "x", "--task-params", "x", *extra])
