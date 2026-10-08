"""Decisions for the unattended tactile collection loop.

The loop sends one PickPlace goal per cycle: pick the can, then put it down at a
random free point inside the pick region. Putting it back where it came from
would record the same episode forever; a fresh point resets the scene and varies
it in one move, so the loop needs no operator between cycles. Each attempt is
recorded as one MCAP episode in the loader's layout (`0001/0001_0.mcap` plus a
`metadata.json`), so existing tooling reads the runs unchanged.

This module holds everything the loop decides, with no ROS, no threads and no
clock: knob sampling takes an injected `numpy.random.Generator`, the stop policy
is told the time and free disk space, and the metadata helpers return plain
dicts. A thin node feeds it and does the I/O, so every stop rule is unit-tested
off the robot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from .task_machine import ErrorCode

#: Version of the per-episode `metadata.json`, matching the loader's.
EPISODE_METADATA_VERSION = 1

#: Rejection-sampling budget before a place region is declared unusable.
MAX_PLACE_TRIES = 1000

#: Stop reason recorded when the operator ends the run (SIGINT or a cancelled goal).
OPERATOR_STOP = "operator"


def _check_range(name: str, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    if lo > hi:
        raise ValueError(f"{name} range has lo {lo} > hi {hi}")


@dataclass(frozen=True)
class KnobRanges:
    """Uniform ranges the per-cycle knobs are drawn from.

    Every range defaults to zero width at zero, so a run varies only the place
    point until each knob has been tuned on the robot.

    Attributes:
        grasp_dz_m: (lo, hi) offset added to the grasp height.
        closure_m: (lo, hi) finger position commanded at CLOSE; 0 or below means the
            task node's configured closure.
        yaw_offset_rad: (lo, hi) offset added to the grasp yaw.
    """

    grasp_dz_m: tuple[float, float] = (0.0, 0.0)
    closure_m: tuple[float, float] = (0.0, 0.0)
    yaw_offset_rad: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        _check_range("grasp_dz_m", self.grasp_dz_m)
        _check_range("closure_m", self.closure_m)
        _check_range("yaw_offset_rad", self.yaw_offset_rad)

    def to_dict(self) -> dict[str, list[float]]:
        """The ranges as JSON-ready lists."""
        return {name: [float(v) for v in bounds] for name, bounds in asdict(self).items()}


@dataclass(frozen=True)
class PlaceRegion:
    """Where the can may be put down: the pick region, kept off its edges and the can.

    Attributes:
        min_xy: Lower (x, y) corner of the pick region in the world frame.
        max_xy: Upper (x, y) corner of the pick region.
        edge_margin_m: Distance kept from every edge, so the can lands wholly
            inside the region the detector watches.
        min_place_distance_m: Smallest distance from the can's current centre,
            so consecutive episodes differ.
    """

    min_xy: tuple[float, float]
    max_xy: tuple[float, float]
    edge_margin_m: float = 0.0
    min_place_distance_m: float = 0.0

    def __post_init__(self) -> None:
        _check_range("x", (self.min_xy[0], self.max_xy[0]))
        _check_range("y", (self.min_xy[1], self.max_xy[1]))
        if self.edge_margin_m < 0.0 or self.min_place_distance_m < 0.0:
            raise ValueError("edge_margin_m and min_place_distance_m must be non-negative")

    @property
    def usable_min_xy(self) -> tuple[float, float]:
        """Lower corner after the edge margin is taken off."""
        return (self.min_xy[0] + self.edge_margin_m, self.min_xy[1] + self.edge_margin_m)

    @property
    def usable_max_xy(self) -> tuple[float, float]:
        """Upper corner after the edge margin is taken off."""
        return (self.max_xy[0] - self.edge_margin_m, self.max_xy[1] - self.edge_margin_m)

    def to_dict(self) -> dict[str, Any]:
        """The region as a JSON-ready dict."""
        return {
            "min_xy": [float(v) for v in self.min_xy],
            "max_xy": [float(v) for v in self.max_xy],
            "edge_margin_m": float(self.edge_margin_m),
            "min_place_distance_m": float(self.min_place_distance_m),
        }


@dataclass(frozen=True)
class Knobs:
    """One cycle's sampled settings.

    Attributes:
        place_xy: Where the can is put down.
        grasp_dz_m: Offset added to the grasp height.
        closure_m: Finger position commanded at CLOSE; 0 or below means configured.
        yaw_offset_rad: Offset added to the grasp yaw.
    """

    place_xy: tuple[float, float]
    grasp_dz_m: float
    closure_m: float
    yaw_offset_rad: float

    def to_dict(self) -> dict[str, Any]:
        """The knobs as a JSON-ready dict."""
        return {
            "place_xy": [float(v) for v in self.place_xy],
            "grasp_dz_m": float(self.grasp_dz_m),
            "closure_m": float(self.closure_m),
            "yaw_offset_rad": float(self.yaw_offset_rad),
        }


def sample_place_xy(
    rng: np.random.Generator,
    region: PlaceRegion,
    can_xy: Sequence[float],
    max_tries: int = MAX_PLACE_TRIES,
) -> tuple[float, float]:
    """Draw a uniform place point inside the usable region and clear of the can.

    Args:
        rng: Source of randomness, injected so runs and tests are reproducible.
        region: The pick region with its margins.
        can_xy: The can's current centre.
        max_tries: Draws allowed before giving up.

    Returns:
        The (x, y) place point.

    Raises:
        ValueError: If the margins leave no room, or no draw clears the can.
    """
    lo = np.asarray(region.usable_min_xy, dtype=float)
    hi = np.asarray(region.usable_max_xy, dtype=float)
    if np.any(lo > hi):
        raise ValueError(f"edge margin {region.edge_margin_m} m leaves no room in the pick region")
    can = np.asarray(can_xy, dtype=float)
    # tradeoff: rejection sampling instead of sampling the region minus a disc exactly;
    # the disc is a small part of the region, so a draw almost always lands first time,
    # and the try cap turns a region that cannot fit the disc into an error, not a hang.
    for _ in range(max_tries):
        xy = rng.uniform(lo, hi)
        if np.hypot(*(xy - can)) >= region.min_place_distance_m:
            return (float(xy[0]), float(xy[1]))
    raise ValueError(
        f"no place point at least {region.min_place_distance_m} m from the can "
        f"after {max_tries} tries; the pick region is too small"
    )


def sample_knobs(
    rng: np.random.Generator,
    region: PlaceRegion,
    can_xy: Sequence[float],
    ranges: KnobRanges,
) -> Knobs:
    """Draw one cycle's place point and knob values.

    Args:
        rng: Source of randomness, injected so runs and tests are reproducible.
        region: The pick region with its margins.
        can_xy: The can's current centre.
        ranges: Ranges for the grasp knobs.

    Returns:
        The cycle's knobs. A zero-width range yields exactly its lower bound.

    Raises:
        ValueError: If no place point can be found.
    """
    place_xy = sample_place_xy(rng, region, can_xy)
    return Knobs(
        place_xy=place_xy,
        grasp_dz_m=float(rng.uniform(*ranges.grasp_dz_m)),
        closure_m=float(rng.uniform(*ranges.closure_m)),
        yaw_offset_rad=float(rng.uniform(*ranges.yaw_offset_rad)),
    )


@dataclass(frozen=True)
class StopLimits:
    """When the unattended loop gives up.

    Attributes:
        max_grasp_retries: Extra attempts allowed on one cycle after a missed grasp.
        max_consecutive_failures: Failed attempts in a row that end the run.
        min_free_bytes: Free disk below which no new cycle starts.
        max_cycles: Successful cycles after which the run ends; None for no limit.
        max_duration_s: Run time after which no new cycle starts; None for no limit.
    """

    max_grasp_retries: int = 2
    max_consecutive_failures: int = 3
    min_free_bytes: int = 50_000_000_000
    max_cycles: int | None = None
    max_duration_s: float | None = None

    def __post_init__(self) -> None:
        counts = (self.max_grasp_retries, self.max_consecutive_failures, self.min_free_bytes)
        if min(counts) < 0 or (self.max_cycles is not None and self.max_cycles < 0):
            raise ValueError("stop limits must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """The limits as a JSON-ready dict."""
        return asdict(self)


class Decision(Enum):
    """What the loop does after an attempt."""

    CONTINUE = "continue"  # Start the next cycle with a new can position and knobs.
    RETRY = "retry"  # Run the same cycle again on the same can.
    STOP = "stop"  # End the run; the reason is in `StopPolicy.stop_reason`.


class StopPolicy:
    """Counts attempts and decides whether the run goes on.

    Stop reasons:

    - `"operator"`: `request_stop` was called (SIGINT) or a goal came back CANCELLED.
    - `"error:<CODE>"`: any other failure but GRASP_MISSED, e.g. `"error:NO_CAN"`.
      These leave the scene in a state the loop cannot repair, so they stop at once.
    - `"grasp_retries"`: a grasp missed again after `max_grasp_retries` retries.
    - `"consecutive_failures"`: `max_consecutive_failures` failures in a row.
    - `"disk"`, `"cycles"`, `"time"`: checked before each cycle by `before_cycle`.

    An operator CANCELLED counts as a failed attempt in the summary, since the episode
    it ends is incomplete. When one miss trips both the retry cap and the failure streak, `"grasp_retries"`
    wins because it names the cause; the streak is the backstop for limits set
    tighter than the retry cap. A success resets both counters.
    """

    def __init__(self, limits: StopLimits, started_at: float) -> None:
        """Start the policy.

        Args:
            limits: When to stop.
            started_at: Monotonic time the run began, in seconds.
        """
        self.limits = limits
        self.started_at = started_at
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.consecutive_failures = 0
        self.stop_reason: str | None = None

    def request_stop(self, reason: str = OPERATOR_STOP) -> None:
        """Stop the run from outside; the first reason recorded is kept."""
        if self.stop_reason is None:
            self.stop_reason = reason

    def before_cycle(self, now: float, free_bytes: int) -> str | None:
        """Check whether a new cycle may start.

        Args:
            now: Monotonic time in seconds, on the same clock as `started_at`.
            free_bytes: Free space on the disk the episodes are written to.

        Returns:
            The stop reason, or None to go ahead.
        """
        if self.stop_reason is None:
            limits = self.limits
            if free_bytes < limits.min_free_bytes:
                self.stop_reason = "disk"
            elif limits.max_cycles is not None and self.successes >= limits.max_cycles:
                self.stop_reason = "cycles"
            elif limits.max_duration_s is not None and self.elapsed_s(now) >= limits.max_duration_s:
                self.stop_reason = "time"
        return self.stop_reason

    def after_attempt(self, code: ErrorCode | int) -> Decision:
        """Record one attempt's result and say what happens next.

        Args:
            code: The PickPlace result's error code.

        Returns:
            CONTINUE, RETRY or STOP; on STOP, `stop_reason` says why.
        """
        code = ErrorCode(code)
        self.attempts += 1
        if code == ErrorCode.SUCCESS:
            self.successes += 1
            self.retries = 0
            self.consecutive_failures = 0
            return Decision.STOP if self.stop_reason is not None else Decision.CONTINUE

        self.failures += 1
        self.consecutive_failures += 1
        if code == ErrorCode.CANCELLED:
            self.request_stop(OPERATOR_STOP)
        elif code != ErrorCode.GRASP_MISSED:
            self.request_stop(f"error:{code.name}")
        elif self.retries >= self.limits.max_grasp_retries:
            self.request_stop("grasp_retries")
        elif self.consecutive_failures >= self.limits.max_consecutive_failures:
            self.request_stop("consecutive_failures")
        if self.stop_reason is not None:
            return Decision.STOP
        self.retries += 1
        return Decision.RETRY

    def elapsed_s(self, now: float) -> float:
        """Seconds since the run began."""
        return now - self.started_at


def run_id(now_utc: datetime) -> str:
    """Name a run by its UTC start time, e.g. `20261008T142301Z`.

    Raises:
        ValueError: If `now_utc` is naive, since its zone would be a guess.
    """
    return _as_utc(now_utc).strftime("%Y%m%dT%H%M%SZ")


def iso_utc(when: datetime) -> str:
    """An ISO 8601 UTC timestamp for metadata, e.g. `2026-10-08T14:23:01Z`."""
    return _as_utc(when).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return when.astimezone(UTC)


def episode_name(index: int) -> str:
    """The loader's episode directory name for a 1-based index, e.g. `0001`."""
    if index < 1:
        raise ValueError(f"episode index is 1-based, got {index}")
    return f"{index:04d}"


def episode_metadata(
    *,
    code: ErrorCode | int,
    attempt: int,
    can_xy: Sequence[float],
    knobs: Knobs,
    started_at: str,
    duration_s: float,
    note: str | None = None,
) -> dict[str, Any]:
    """Build one episode's `metadata.json`.

    Keeps the loader's keys (`version`, `status`, `note`, `duration`) so its tools
    read these episodes, and adds what the loop knows about the attempt.

    Args:
        code: The attempt's result code.
        attempt: 0 for a cycle's first attempt, n for its nth retry.
        can_xy: Where the can was detected before the pick.
        knobs: The cycle's sampled knobs.
        started_at: ISO UTC time the attempt began.
        duration_s: Attempt length in seconds.
        note: Free-text note; the loader's default is null.

    Returns:
        A dict ready for `json.dump`.
    """
    code = ErrorCode(code)
    return {
        "version": EPISODE_METADATA_VERSION,
        "status": "success" if code == ErrorCode.SUCCESS else "failure",
        "note": note,
        "duration": round(float(duration_s), 3),
        "error_code": code.name,
        "attempt": int(attempt),
        "can_xy": [float(v) for v in can_xy],
        "place_xy": [float(v) for v in knobs.place_xy],
        "knobs": knobs.to_dict(),
        "started_at": started_at,
    }


def run_metadata(
    *,
    object_name: str,
    object_config: Mapping[str, Any],
    region: PlaceRegion,
    knob_ranges: KnobRanges,
    limits: StopLimits,
    speed_scale: float,
    record: bool,
    git_sha: str | None,
    started_at: str,
) -> dict[str, Any]:
    """Build the run's `metadata.json`, written once before the first cycle.

    Args:
        object_name: Name of the object being collected, e.g. `can`.
        object_config: The object's settings as loaded from config.
        region: Where cans are placed.
        knob_ranges: Ranges the knobs are drawn from.
        limits: When the run stops.
        speed_scale: Speed scale sent with every goal.
        record: Whether episodes are recorded.
        git_sha: Commit the code ran from, or None if unknown.
        started_at: ISO UTC start time.

    Returns:
        A dict ready for `json.dump`.
    """
    return {
        "object": object_name,
        "object_config": dict(object_config),
        "place_region": region.to_dict(),
        "knob_ranges": knob_ranges.to_dict(),
        "stop_limits": limits.to_dict(),
        "speed_scale": float(speed_scale),
        "record": bool(record),
        "git_sha": git_sha,
        "started_at": started_at,
    }


def summary_metadata(
    policy: StopPolicy, *, episodes: int, now: float, ended_at: str
) -> dict[str, Any]:
    """Build the run's `summary.json`, written once when the loop ends.

    Args:
        policy: The run's stop policy, holding its counters and stop reason.
        episodes: Episodes written to disk.
        now: Monotonic time the run ended, on the policy's clock.
        ended_at: ISO UTC end time.

    Returns:
        A dict ready for `json.dump`.
    """
    return {
        "attempts": policy.attempts,
        "episodes": int(episodes),
        "successes": policy.successes,
        "failures": policy.failures,
        "stop_reason": policy.stop_reason,
        "duration_s": round(policy.elapsed_s(now), 3),
        "ended_at": ended_at,
    }
