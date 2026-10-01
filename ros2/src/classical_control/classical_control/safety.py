"""Safety limits for commanded end-effector poses.

The task node publishes absolute TCP poses straight to `/commanded_ee_right`,
and the robot's IK follows them. That path bypasses every upstream limiter, so
apart from the webapp e-stop this module is the only software guard between the
planner and the hardware. It is deliberately dumb: clamp the step, clamp the
box, keep off the table, clamp the gripper, and refuse to command at all unless
the hardware is confirmed active and the inputs are fresh.

Ported from rdt2-bridge's `safety.py`, with scipy's Rotation in place of the
hand-rolled frame maths there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .trajectory import Pose

#: The only hardware state in which the arms accept commands.
ACTIVE_STATE = "active"


class HardwareStateGate:
    """Holds commands unless the workcell has confirmed it is active.

    The loader runs a `hardware_state_controller` inside its 500 Hz control loop
    that holds `active`, `pause` or `estop`. The webapp's Emergency Stop button
    sets that state, so honouring it here means the button stops this node too,
    not just the hardware layer underneath it.

    Two properties of that controller shape this gate:

    - It publishes **only on change**, and it boots active. A node that waited
      for an `active` message would wait forever, so readiness is established by
      asking the `set_state` service for `active` instead. That request is a
      no-op when the hardware already is active, and the controller refuses it
      outright when it is not, so it cannot talk its way past a stop.
    - **A stop latches.** Once paused or stopped, the controller answers
      `Cannot transition to active: restart the system to safely activate the
      hardware`. The gate latches the same way: once it has seen a stop, only a
      restart clears it.
    """

    def __init__(self) -> None:
        self._state: str | None = None
        self._confirmed_active = False
        self._stopped = False

    @property
    def state(self) -> str | None:
        """The last reported state, or None if nothing has been heard yet."""
        return self._state

    @property
    def allows_commands(self) -> bool:
        """True only after an accepted probe and with no stop seen since boot."""
        return self._confirmed_active and not self._stopped

    def confirm_active(self, accepted: bool) -> None:
        """Record the answer to a `set_state active` probe."""
        if accepted:
            self._confirmed_active = True
        else:
            # The controller only refuses when the hardware is not active, and
            # that condition needs a restart to clear.
            self._stopped = True

    def update(self, state: str) -> bool:
        """Record a broadcast state. Returns True if it differs from the last one."""
        changed = state != self._state
        self._state = state
        if state != ACTIVE_STATE:
            self._stopped = True
        return changed


class FreshnessGate:
    """Holds commands when an input has gone stale.

    Observed in rdt2-bridge's first dry run: inputs can stop arriving while the
    control loop keeps running on the last values it cached. Commanding from a
    frozen camera or a frozen pose is exactly the failure this exists to
    prevent, so every input carries a timestamp and anything older than
    `max_age_s` closes the gate.
    """

    def __init__(self, max_age_s: float) -> None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        self._max_age_s = max_age_s
        self._seen: dict[str, float] = {}

    def mark(self, key: str, now: float) -> None:
        """Record that input `key` arrived at monotonic time `now`."""
        self._seen[key] = now

    def stale(self, keys: tuple[str, ...], now: float) -> tuple[str, ...]:
        """Return the inputs that are missing or older than the age limit."""
        return tuple(
            key
            for key in keys
            if key not in self._seen or (now - self._seen[key]) > self._max_age_s
        )


def _vector3(values: Sequence[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    vector = tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))
    if len(vector) != 3:
        raise ValueError(f"{name} must have 3 elements, got {len(vector)}")
    return vector  # type: ignore[return-value]


@dataclass(frozen=True)
class SafetyLimits:
    """Per-tick and workspace limits on commanded poses, in the `world` frame.

    Attributes:
        workspace_min_m: Lower corner of the allowed TCP box (x, y, z).
        workspace_max_m: Upper corner of the allowed TCP box (x, y, z).
        table_z: Height of the table surface in metres.
        z_margin_m: Clearance the TCP keeps above the table.
        max_position_step_m: Largest translation allowed between two ticks.
        max_rotation_step_rad: Largest rotation allowed between two ticks.
        gripper_min_m: Fully closed finger position.
        gripper_max_m: Fully open finger position.
    """

    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    table_z: float
    z_margin_m: float = 0.01
    max_position_step_m: float = 0.005
    max_rotation_step_rad: float = 0.05
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_min_m", _vector3(self.workspace_min_m, "min"))
        object.__setattr__(self, "workspace_max_m", _vector3(self.workspace_max_m, "max"))
        # NaN compares False, so every check below would pass it and clamp() would
        # then hand NaN to the arm. Reject non-finite limits before anything else.
        scalars = (
            self.table_z,
            self.z_margin_m,
            self.max_position_step_m,
            self.max_rotation_step_rad,
            self.gripper_min_m,
            self.gripper_max_m,
        )
        if not np.isfinite([*self.workspace_min_m, *self.workspace_max_m, *scalars]).all():
            raise ValueError("safety limits must be finite")
        if any(lo >= hi for lo, hi in zip(self.workspace_min_m, self.workspace_max_m)):
            raise ValueError("workspace_min_m must be below workspace_max_m on every axis")
        if self.z_margin_m < 0.0:
            raise ValueError("z_margin_m must not be negative")
        if self.max_position_step_m <= 0.0 or self.max_rotation_step_rad <= 0.0:
            raise ValueError("step limits must be positive")
        if self.gripper_min_m >= self.gripper_max_m:
            raise ValueError("gripper_min_m must be below gripper_max_m")
        if self.min_z_m >= self.workspace_max_m[2]:
            raise ValueError("table_z + z_margin_m leaves no room below workspace_max_m")

    @property
    def min_z_m(self) -> float:
        """The effective z floor: the box floor or the table clearance, whichever is higher."""
        return max(self.workspace_min_m[2], self.table_z + self.z_margin_m)

    @property
    def lower_bound_m(self) -> np.ndarray:
        """The box's lower corner with the table floor applied."""
        return np.array([self.workspace_min_m[0], self.workspace_min_m[1], self.min_z_m])

    @property
    def upper_bound_m(self) -> np.ndarray:
        """The box's upper corner."""
        return np.array(self.workspace_max_m)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SafetyLimits:
        """Build limits from a parsed config mapping; unknown keys are an error."""
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown safety keys: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class ClampReport:
    """What the limiter changed, so the node can log it during a dry run."""

    position_clamped: bool
    rotation_clamped: bool
    workspace_clamped: bool

    @property
    def any_clamped(self) -> bool:
        """True if any limit bound."""
        return self.position_clamped or self.rotation_clamped or self.workspace_clamped


class SafetyLimiter:
    """Clamps a target pose against the previous commanded pose and the workspace."""

    def __init__(self, limits: SafetyLimits) -> None:
        self._limits = limits

    @property
    def limits(self) -> SafetyLimits:
        """The limits this limiter enforces."""
        return self._limits

    def in_workspace(self, pose: Pose) -> bool:
        """True if `pose` lies inside the box and above the table floor."""
        position = pose.position
        return bool(
            np.all(position >= self._limits.lower_bound_m)
            and np.all(position <= self._limits.upper_bound_m)
        )

    def clamp(self, target: Pose, previous: Pose) -> tuple[Pose, ClampReport]:
        """Clamp `target` so it is reachable in one tick from `previous` and stays in bounds.

        The box is applied before the step limit, unlike rdt2-bridge. The box is
        convex, so when `previous` is inside it the shortened step stays inside
        too, and when `previous` is outside (a start pose below the floor, say)
        the arm walks back at the step limit instead of jumping.

        Args:
            target: Pose the planner asked for, in the `world` frame.
            previous: Pose that was last commanded, in the `world` frame.

        Returns:
            The clamped pose and a report of which limits bound.
        """
        limits = self._limits

        bounded = np.clip(target.position, limits.lower_bound_m, limits.upper_bound_m)
        workspace_clamped = not np.allclose(bounded, target.position)

        delta_position = bounded - previous.position
        distance = float(np.linalg.norm(delta_position))
        position_clamped = distance > limits.max_position_step_m
        position = bounded
        if position_clamped:
            position = previous.position + delta_position * (limits.max_position_step_m / distance)

        # as_rotvec reads the angle and axis from the quaternion, so a half-turn
        # keeps its axis instead of collapsing to "no rotation" as it does when
        # read off the antisymmetric part of a matrix.
        delta_rotvec = (previous.rotation.inv() * target.rotation).as_rotvec()
        angle = float(np.linalg.norm(delta_rotvec))
        rotation_clamped = angle > limits.max_rotation_step_rad
        rotation = target.rotation
        if rotation_clamped:
            limited = Rotation.from_rotvec(delta_rotvec * (limits.max_rotation_step_rad / angle))
            rotation = previous.rotation * limited

        return Pose.from_rotation(position, rotation), ClampReport(
            position_clamped=position_clamped,
            rotation_clamped=rotation_clamped,
            workspace_clamped=workspace_clamped,
        )

    def clamp_gripper(self, value: float) -> tuple[float, bool]:
        """Clamp a finger position into the gripper range.

        Returns:
            The clamped position in metres and whether the clamp bound.

        Raises:
            ValueError: If the value is not finite; np.clip would pass NaN through.
        """
        if not np.isfinite(value):
            raise ValueError("gripper command must be finite")
        bounded = float(np.clip(value, self._limits.gripper_min_m, self._limits.gripper_max_m))
        return bounded, not np.isclose(bounded, value)
