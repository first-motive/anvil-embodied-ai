"""Min-jerk Cartesian trajectories for the commanded-EE stream.

The task node streams absolute TCP poses at a fixed rate and the robot's own IK
follows them, so every sample must already be a pose the arm can reach smoothly
in one tick. A min-jerk profile starts and ends at rest with zero acceleration,
which keeps the IK from jolting at segment boundaries, and its closed-form peak
speed lets the duration be picked so a speed limit holds by construction instead
of by clipping afterwards.

Orientation follows the same profile through a slerp, so position and rotation
arrive together and the angular speed bound is enforced the same way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

#: Peak of ds/dτ for the min-jerk profile, reached at τ = 0.5.
MIN_JERK_PEAK_RATE = 1.875


@dataclass(frozen=True, eq=False)
class Pose:
    """A TCP pose in the `world` frame.

    Attributes:
        position: Translation in metres, shape (3,).
        quat_xyzw: Unit quaternion in scalar-last order (ROS and scipy), shape (4,).
    """

    position: np.ndarray
    quat_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1).copy()
        quat = np.asarray(self.quat_xyzw, dtype=np.float64).reshape(-1).copy()
        if position.shape != (3,):
            raise ValueError(f"position must have 3 elements, got {position.shape}")
        if quat.shape != (4,):
            raise ValueError(f"quat_xyzw must have 4 elements, got {quat.shape}")
        # NaN compares False against every limit, so a non-finite pose would slip
        # past the safety clamps unnoticed. Refuse it at the one type they all take.
        if not (np.isfinite(position).all() and np.isfinite(quat).all()):
            raise ValueError("pose must be finite")
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            raise ValueError("quat_xyzw must be non-zero")
        quat /= norm
        # Frozen means frozen: callers must not be able to mutate a pose in place.
        position.setflags(write=False)
        quat.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quat_xyzw", quat)

    @property
    def rotation(self) -> Rotation:
        """The orientation as a scipy Rotation."""
        return Rotation.from_quat(self.quat_xyzw)

    @classmethod
    def from_rotation(cls, position: Sequence[float] | np.ndarray, rotation: Rotation) -> Pose:
        """Build a pose from a position and a scipy Rotation."""
        return cls(np.asarray(position, dtype=np.float64), rotation.as_quat())


def min_jerk(tau: np.ndarray | float) -> np.ndarray:
    """Evaluate the min-jerk progress s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ for τ in [0, 1]."""
    tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def segment_duration(
    start: Pose, end: Pose, v_max: float, w_max: float, min_duration: float = 0.0
) -> float:
    """Shortest min-jerk duration that keeps peak speeds within the limits.

    Args:
        start: Pose the segment starts from.
        end: Pose the segment ends at.
        v_max: Peak linear speed limit in m/s.
        w_max: Peak angular speed limit in rad/s.
        min_duration: Floor on the duration in seconds.

    Returns:
        The duration in seconds.
    """
    if v_max <= 0.0 or w_max <= 0.0:
        raise ValueError("v_max and w_max must be positive")
    distance = float(np.linalg.norm(end.position - start.position))
    # magnitude() is in [0, π], i.e. the shortest arc, matching what slerp travels.
    angle = float((start.rotation.inv() * end.rotation).magnitude())
    return max(
        MIN_JERK_PEAK_RATE * distance / v_max,
        MIN_JERK_PEAK_RATE * angle / w_max,
        min_duration,
    )


def plan_segment(
    start: Pose,
    end: Pose,
    v_max: float,
    w_max: float,
    rate_hz: float = 30.0,
    min_duration: float = 0.0,
) -> list[Pose]:
    """Sample a min-jerk move from `start` to `end` at `rate_hz`.

    The sample count is rounded up, so the realised duration is never shorter
    than the one the limits ask for and the per-tick step never exceeds
    `v_max / rate_hz` (or `w_max / rate_hz` in rotation).

    Args:
        start: Current commanded pose; not included in the output.
        end: Target pose; always the last sample, exactly.
        v_max: Peak linear speed limit in m/s.
        w_max: Peak angular speed limit in rad/s.
        rate_hz: Streaming rate in Hz.
        min_duration: Floor on the duration in seconds, for moves that should
            not be rushed even when short.

    Returns:
        The poses to stream, one per tick. A zero-length move with no
        `min_duration` yields just `[end]`.
    """
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive")
    duration = segment_duration(start, end, v_max, w_max, min_duration)
    # The epsilon stops float noise in duration * rate from adding a whole tick.
    count = max(1, math.ceil(duration * rate_hz - 1e-9))

    progress = min_jerk(np.arange(1, count + 1) / count)
    positions = start.position + np.outer(progress, end.position - start.position)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([start.rotation, end.rotation]))
    quats = slerp(progress).as_quat()

    samples = [Pose(p, q) for p, q in zip(positions[:-1], quats[:-1])]
    samples.append(end)
    return samples


def plan_path(
    waypoints: Sequence[Pose],
    v_max: float,
    w_max: float,
    rate_hz: float = 30.0,
    min_duration: float = 0.0,
) -> list[Pose]:
    """Chain `plan_segment` through `waypoints`, stopping at each one.

    Each segment ends at rest, so the arm briefly halts at every waypoint. That
    is what pick-and-place wants at pre-grasp and grasp poses.

    Args:
        waypoints: Poses to visit in order; the first is the current pose and is
            not included in the output.
        v_max: Peak linear speed limit in m/s.
        w_max: Peak angular speed limit in rad/s.
        rate_hz: Streaming rate in Hz.
        min_duration: Floor on each segment's duration in seconds.

    Returns:
        The concatenated poses to stream.
    """
    samples: list[Pose] = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        samples.extend(plan_segment(start, end, v_max, w_max, rate_hz, min_duration))
    return samples
