"""Phase machine for one look-then-move can-on-paper pick and place.

The machine turns a localised can and paper into a fixed sequence of TCP
targets and streams them one sample per control tick:

    LOCALISE → PRE_GRASP → DESCEND → CLOSE → LIFT → TRANSIT → LOWER → OPEN
             → RETREAT → HOME

A cancel or a fault in any phase drops into HOLD, which re-commands the last
pose sent and stops there, and the run ends with that error code.

A goal may override the place point, grasp height, finger closure and grasp
yaw (`GoalOverrides`) so an unattended collection loop can vary its episodes,
or ask only to go HOME (`PickPlaceTask.home_only`). Every override at its
default reproduces the plain pick and place.

It is pure (no ROS, no threads, time is passed in) so the whole sequence is
unit-tested off the robot; `task_node` only feeds it measurements and publishes
what it returns. Three robot facts shape it:

- The loader latches the last commanded EE target, which may be stale from an
  earlier session. The first command is therefore the measured current pose,
  never a planned one, so starting never jumps.
- An unreachable pose makes the IK drift silently rather than fail. Every
  waypoint uses the mined grasp orientation (the one the demonstrations proved
  reachable), stood upright by `level_orientation`, and must pass the workspace
  check before anything moves.
- HOME is a commanded move to the stored home pose, not a call to the arms
  resetter, so it is just the last segment of the stream.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .safety import SafetyLimiter, SafetyLimits
from .trajectory import Pose, plan_segment

#: Smallest speed scale honoured, so a typo cannot stall the arm mid-air for minutes.
MIN_SPEED_SCALE = 0.1


class Phase(IntEnum):
    """Task phases; values match the PickPlace feedback constants."""

    LOCALISE = 0
    PRE_GRASP = 1
    DESCEND = 2
    CLOSE = 3
    LIFT = 4
    TRANSIT = 5
    LOWER = 6
    OPEN = 7
    RETREAT = 8
    HOME = 9
    HOLD = 10


class ErrorCode(IntEnum):
    """Run outcomes; values match the PickPlace result constants."""

    SUCCESS = 0
    NO_CAN = 1
    NO_PAPER = 2
    OUT_OF_WORKSPACE = 3
    GRASP_MISSED = 4
    HARDWARE_NOT_ACTIVE = 5
    STALE_POSE = 6
    CANCELLED = 7


@dataclass(frozen=True)
class TaskConfig:
    """Geometry, speed and gripper settings for one run.

    Attributes:
        grasp_z_m: TCP height at which the gripper closes on the can.
        place_z_m: TCP height at which the gripper releases over the paper.
        grasp_orientation_xyzw: TCP orientation held from pre-grasp to retreat.
        home: Pose the run ends at.
        approach_height_m: Height above grasp/place used for pre-grasp, lift,
            transit and retreat.
        can_xy_bias_m: Offset added to the detected can centre to get the TCP xy.
        v_max_mps: Peak linear speed at speed_scale 1.0.
        w_max_radps: Peak angular speed at speed_scale 1.0.
        rate_hz: Control rate the samples are streamed at.
        close_dwell_s: Time the gripper is given to close before the finger check.
        open_dwell_s: Time the gripper is given to release before retreating.
        gripper_open_m: Finger position commanded open.
        gripper_closed_m: Finger position commanded closed.
        gripper_home_m: Finger position commanded on the way home.
        grasp_missed_threshold_m: Finger position below which a close is empty. With a
            goal's closure override it is measured above that closure instead of zero.
    """

    grasp_z_m: float
    place_z_m: float
    grasp_orientation_xyzw: tuple[float, float, float, float]
    home: Pose
    approach_height_m: float = 0.08
    can_xy_bias_m: tuple[float, float] = (0.0, 0.0)
    v_max_mps: float = 0.10
    w_max_radps: float = 0.5
    rate_hz: float = 30.0
    close_dwell_s: float = 1.0
    open_dwell_s: float = 0.8
    gripper_open_m: float = 0.05
    gripper_closed_m: float = 0.0
    gripper_home_m: float = 0.045
    grasp_missed_threshold_m: float = 0.008


@dataclass(frozen=True)
class Command:
    """One tick's output: an absolute TCP target and a finger position in metres."""

    pose: Pose
    gripper: float


@dataclass(frozen=True)
class Step:
    """One phase's goal. A non-zero dwell means hold the pose while the gripper acts."""

    phase: Phase
    target: Pose
    gripper: float
    dwell_s: float = 0.0


@dataclass(frozen=True)
class GoalOverrides:
    """Per-goal changes to the planned grasp, already clamped by `goal_overrides`.

    Attributes:
        grasp_dz_m: Added to both grasp and place height, so the can still hangs
            the same distance below the fingers and lands on the table.
        closure_m: Finger position commanded at CLOSE, or None for the configured
            `gripper_closed_m`.
        yaw_offset_rad: Rotation about world z applied to the grasp orientation
            from pre-grasp to retreat; HOME keeps its own orientation.
    """

    grasp_dz_m: float = 0.0
    closure_m: float | None = None
    yaw_offset_rad: float = 0.0


def goal_overrides(
    grasp_dz: float,
    closure_width: float,
    yaw_offset: float,
    *,
    max_grasp_dz_m: float,
    max_yaw_offset_rad: float,
    gripper_range_m: tuple[float, float],
    max_closure_m: float,
) -> GoalOverrides:
    """Clamp a goal's raw override fields into `GoalOverrides`.

    Args:
        grasp_dz: Requested height change in metres; clamped to ±max_grasp_dz_m.
        closure_width: Requested finger closure in metres. Non-positive is the
            message default and means the configured closure; otherwise clamped
            into `gripper_range_m`, the limiter's (min, max), and capped at
            `max_closure_m`. An empty close reads about the closure, so the missed
            check fires below closure + threshold; the cap keeps that under what a
            held can reads, or every good grasp would count as missed.
        yaw_offset: Requested yaw in radians; clamped to ±max_yaw_offset_rad.

    Raises:
        ValueError: If any field is NaN; np.clip would pass it through.
    """
    if any(math.isnan(v) for v in (grasp_dz, closure_width, yaw_offset)):
        raise ValueError("goal overrides must not be NaN")
    closure = None
    if closure_width > 0.0:
        low, high = gripper_range_m
        closure = float(np.clip(closure_width, low, min(high, max_closure_m)))
    return GoalOverrides(
        grasp_dz_m=float(np.clip(grasp_dz, -max_grasp_dz_m, max_grasp_dz_m)),
        closure_m=closure,
        yaw_offset_rad=float(np.clip(yaw_offset, -max_yaw_offset_rad, max_yaw_offset_rad)),
    )


def place_target_in_workspace(
    limits: SafetyLimits, target: Sequence[float]
) -> tuple[float, float, float]:
    """Clamp a requested place point's xy into the workspace box.

    Only xy is used by the planner (heights come from the demonstrations), so z
    is passed through. The all-waypoints workspace check still runs afterwards.

    Raises:
        ValueError: If the target is not finite.
    """
    x, y, z = (float(v) for v in target)
    if not np.isfinite([x, y, z]).all():
        raise ValueError("place target must be finite")
    lower, upper = limits.workspace_min_m, limits.workspace_max_m
    return float(np.clip(x, lower[0], upper[0])), float(np.clip(y, lower[1], upper[1])), z


def required_detections(*, home_only: bool, place_target_set: bool) -> tuple[str, ...]:
    """The detections a goal needs fresh before it may start: "can" and/or "paper".

    A goal with its own place target needs no paper; a home-only goal needs neither.
    """
    if home_only:
        return ()
    return ("can",) if place_target_set else ("can", "paper")


def level_orientation(quat_xyzw: Sequence[float]) -> tuple[tuple[float, ...], float]:
    """Turn a grasp so its most vertical tool axis is exactly vertical.

    The mined orientation is the median of teleoperated grasps, which leaned about 16
    degrees. In a side grasp the squeezed can lines up with the fingers, so it was carried
    leaning and fell when released. The smallest rotation that stands that axis upright
    leaves the approach heading close to the demonstrated one.

    Returns:
        The levelled quaternion (x, y, z, w) and the tilt removed, in radians.
    """
    rotation = Rotation.from_quat(quat_xyzw)
    axes = rotation.as_matrix()
    column = int(np.argmax(np.abs(axes[2])))
    axis = axes[:, column]
    upright = np.array([0.0, 0.0, math.copysign(1.0, axis[2])])
    cross = np.cross(axis, upright)
    tilt = math.atan2(np.linalg.norm(cross), float(np.dot(axis, upright)))
    if tilt < 1e-9:
        return tuple(float(v) for v in quat_xyzw), 0.0
    correction = Rotation.from_rotvec(cross / np.linalg.norm(cross) * tilt)
    return tuple(float(v) for v in (correction * rotation).as_quat()), tilt


def mined_targets(
    mined: Mapping[str, Any], height_offset_m: float = 0.0
) -> tuple[float, float, tuple[float, ...]]:
    """Pull grasp height, place height and grasp orientation out of `mined_params.yaml`.

    Medians, not means: one sloppy demonstration should not move the grasp.

    Args:
        mined: The parsed file.
        height_offset_m: Added to both heights. Grasp and place move together, so the can
            hangs the same distance below the fingers and still lands on the paper.

    Raises:
        KeyError: If the file has no place statistics (no episode released the can).
    """
    quat = tuple(float(v) for v in mined["grasp_orientation_xyzw"])
    grasp_z = float(mined["grasp_z"]["median"]) + height_offset_m
    place_z = float(mined["place_z"]["median"]) + height_offset_m
    return grasp_z, place_z, quat


def effective_speed_scale(speed_scale: float) -> float:
    """Clamp a goal's speed scale into [MIN_SPEED_SCALE, 1].

    A non-positive value means the field was left at its message default, so it
    runs at the configured speed rather than being refused.

    Raises:
        ValueError: If the scale is NaN; np.clip would pass it through and the
            planner would collapse each segment to a single jump.
    """
    if math.isnan(speed_scale):
        raise ValueError("speed_scale must not be NaN")
    if speed_scale <= 0.0:
        return 1.0
    return float(np.clip(speed_scale, MIN_SPEED_SCALE, 1.0))


def plan_steps(
    config: TaskConfig,
    can_position: Sequence[float],
    place_position: Sequence[float],
    overrides: GoalOverrides = GoalOverrides(),
) -> list[Step]:
    """Build the waypoint sequence from the can centre and the place point.

    Only xy is taken from the positions: heights come from the demonstrations,
    which saw where the TCP actually was when the fingers closed and opened.

    Args:
        config: Run settings.
        can_position: Detected can centre in the world frame.
        place_position: Detected paper centre, or a goal's place target.
        overrides: Per-goal grasp changes; the default changes nothing.
    """
    quat = config.grasp_orientation_xyzw
    if overrides.yaw_offset_rad != 0.0:
        # Left-multiplied, so the turn is about world z, not the tilted tool axis.
        rotation = Rotation.from_euler("z", overrides.yaw_offset_rad) * Rotation.from_quat(quat)
        quat = tuple(rotation.as_quat())
    grasp_xy = np.asarray(can_position, dtype=np.float64)[:2] + np.asarray(config.can_xy_bias_m)
    place_xy = np.asarray(place_position, dtype=np.float64)[:2]
    above = config.approach_height_m
    grasp_z = config.grasp_z_m + overrides.grasp_dz_m
    place_z = config.place_z_m + overrides.grasp_dz_m

    grasp = Pose([*grasp_xy, grasp_z], quat)
    pre_grasp = Pose([*grasp_xy, grasp_z + above], quat)
    place = Pose([*place_xy, place_z], quat)
    above_place = Pose([*place_xy, place_z + above], quat)

    opened = config.gripper_open_m
    closed = config.gripper_closed_m if overrides.closure_m is None else overrides.closure_m
    return [
        Step(Phase.PRE_GRASP, pre_grasp, opened),
        Step(Phase.DESCEND, grasp, opened),
        Step(Phase.CLOSE, grasp, closed, config.close_dwell_s),
        Step(Phase.LIFT, pre_grasp, closed),
        Step(Phase.TRANSIT, above_place, closed),
        Step(Phase.LOWER, place, closed),
        Step(Phase.OPEN, place, opened, config.open_dwell_s),
        Step(Phase.RETREAT, above_place, opened),
        Step(Phase.HOME, config.home, config.gripper_home_m),
    ]


class PickPlaceTask:
    """Streams one pick and place, one command per `tick`.

    Usage: build it once localisation has succeeded, then call `tick` at the
    control rate until `done`. `cancel` and `fault` may be called at any time;
    the next tick returns the hold command.

    Construction runs the pre-flight: if any waypoint is outside the workspace
    the task is born finished with OUT_OF_WORKSPACE and never commands. A dry
    run is born finished with SUCCESS; read `steps` to log what it would do.
    """

    def __init__(
        self,
        config: TaskConfig,
        limiter: SafetyLimiter,
        can_position: Sequence[float],
        place_position: Sequence[float],
        *,
        overrides: GoalOverrides = GoalOverrides(),
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        steps = plan_steps(config, can_position, place_position, overrides)
        # An empty close reads about the commanded closure, not zero, so the
        # missed check is measured from it. None keeps the configured threshold as is.
        missed_below = config.grasp_missed_threshold_m + (overrides.closure_m or 0.0)
        self._setup(config, limiter, steps, missed_below, speed_scale, dry_run)

    @classmethod
    def home_only(
        cls,
        config: TaskConfig,
        limiter: SafetyLimiter,
        *,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> PickPlaceTask:
        """Build a task that only moves HOME with the gripper at `gripper_home_m`.

        Used to recover the arm after a failed or stopped run; it needs no can or
        paper and has no CLOSE, so it never reports GRASP_MISSED.
        """
        task = cls.__new__(cls)
        steps = [Step(Phase.HOME, config.home, config.gripper_home_m)]
        task._setup(config, limiter, steps, config.grasp_missed_threshold_m, speed_scale, dry_run)
        return task

    def _setup(
        self,
        config: TaskConfig,
        limiter: SafetyLimiter,
        steps: list[Step],
        missed_below_m: float,
        speed_scale: float,
        dry_run: bool,
    ) -> None:
        self._config = config
        self._steps = steps
        self._missed_below_m = missed_below_m
        scale = effective_speed_scale(speed_scale)
        self._v_max = config.v_max_mps * scale
        self._w_max = config.w_max_radps * scale

        self._phase = Phase.LOCALISE
        self._result: ErrorCode | None = None
        self._pending_fault: ErrorCode | None = None
        self._last: Command | None = None
        self._index = 0
        self._samples: deque[Pose] = deque()
        self._dwell_started: float | None = None

        if not all(limiter.in_workspace(step.target) for step in self._steps):
            self._finish(ErrorCode.OUT_OF_WORKSPACE, hold=True)
        elif dry_run:
            self._finish(ErrorCode.SUCCESS, hold=False)

    @property
    def phase(self) -> Phase:
        """The phase currently being executed, or HOLD after a cancel or fault."""
        return self._phase

    @property
    def result(self) -> ErrorCode | None:
        """The outcome once finished, else None."""
        return self._result

    @property
    def done(self) -> bool:
        """True once the run has an outcome."""
        return self._result is not None

    @property
    def steps(self) -> list[Step]:
        """The planned waypoints in order, for logging."""
        return list(self._steps)

    @property
    def speed_limits(self) -> tuple[float, float]:
        """The (v_max, w_max) the segments are planned with after scaling."""
        return self._v_max, self._w_max

    def cancel(self) -> None:
        """Stop at the next tick and finish with CANCELLED."""
        self.fault(ErrorCode.CANCELLED)

    def fault(self, code: ErrorCode) -> None:
        """Stop at the next tick and finish with `code`. The first fault wins."""
        if self._result is None and self._pending_fault is None:
            self._pending_fault = code

    def tick(self, now: float, measured_pose: Pose | None, finger: float) -> Command | None:
        """Advance one control tick.

        Args:
            now: Monotonic time in seconds; only dwell phases read it.
            measured_pose: Current TCP pose; required on the first tick only.
            finger: Measured finger position in metres.

        Returns:
            The command to publish, or None when there is nothing to send.
        """
        # tradeoff: one sample per tick rather than indexing by wall time, so a
        # late tick slows the move instead of lengthening a step.
        if self._result is not None:
            return self._last if self._phase is Phase.HOLD else None
        if self._pending_fault is not None:
            return self._finish(self._pending_fault, hold=True)
        if self._last is None:
            return self._start(measured_pose, finger)

        while True:
            step = self._steps[self._index]
            if step.dwell_s <= 0.0:
                if self._samples:
                    self._last = Command(self._samples.popleft(), step.gripper)
                    return self._last
            else:
                if self._dwell_started is None:
                    self._dwell_started = now
                if now - self._dwell_started < step.dwell_s:
                    self._last = Command(step.target, step.gripper)
                    return self._last
                if step.phase is Phase.CLOSE and finger < self._missed_below_m:
                    return self._finish(ErrorCode.GRASP_MISSED, hold=True)
            if not self._advance():
                return self._finish(ErrorCode.SUCCESS, hold=False)

    def _start(self, measured_pose: Pose | None, finger: float) -> Command:
        if measured_pose is None:
            raise ValueError("the first tick needs the measured TCP pose")
        # The latch gotcha: echo where the arm is before asking it to go anywhere.
        self._last = Command(measured_pose, float(finger))
        self._enter(0)
        return self._last

    def _advance(self) -> bool:
        if self._index + 1 >= len(self._steps):
            return False
        self._enter(self._index + 1)
        return True

    def _enter(self, index: int) -> None:
        self._index = index
        step = self._steps[index]
        self._phase = step.phase
        self._dwell_started = None
        self._samples.clear()
        if step.dwell_s <= 0.0:
            # Planned from the last command, not the measured pose, so a
            # tracking lag never shows up as a step in the commanded stream.
            self._samples.extend(
                plan_segment(
                    self._last.pose, step.target, self._v_max, self._w_max, self._config.rate_hz
                )
            )

    def _finish(self, code: ErrorCode, *, hold: bool) -> Command | None:
        self._result = code
        if hold:
            # Holding the last command, not the measured pose: the loader already
            # tracks it, so re-sending it stops the arm without any new step.
            self._phase = Phase.HOLD
            return self._last
        return None
