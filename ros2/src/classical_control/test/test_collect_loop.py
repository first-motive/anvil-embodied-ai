"""Tests for the collection loop's knob sampler, stop policy and metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from classical_control.collect_loop import (
    Decision,
    KnobRanges,
    Knobs,
    PlaceRegion,
    StopLimits,
    StopPolicy,
    episode_metadata,
    episode_name,
    forced_stop_reason,
    iso_utc,
    load_loop_config,
    run_id,
    run_metadata,
    sample_knobs,
    sample_place_xy,
    summary_metadata,
)
from classical_control.task_machine import ErrorCode

PLENTY_OF_DISK = 10**12
REGION = PlaceRegion(
    min_xy=(0.20, -0.15), max_xy=(0.45, 0.15), edge_margin_m=0.03, min_place_distance_m=0.08
)
CAN_XY = (0.30, 0.0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


# --- Knob sampler -------------------------------------------------------------------


def test_place_points_stay_inside_the_margin_and_clear_of_the_can(rng):
    lo, hi = np.array(REGION.usable_min_xy), np.array(REGION.usable_max_xy)
    points = np.array([sample_place_xy(rng, REGION, CAN_XY) for _ in range(2000)])

    assert np.all(points >= lo) and np.all(points <= hi)
    assert np.all(np.hypot(*(points - CAN_XY).T) >= REGION.min_place_distance_m)


def test_place_points_cover_the_usable_region(rng):
    points = np.array([sample_place_xy(rng, REGION, CAN_XY) for _ in range(2000)])

    span = points.max(axis=0) - points.min(axis=0)
    assert np.all(span > 0.9 * (np.array(REGION.usable_max_xy) - REGION.usable_min_xy))


def test_region_too_small_to_clear_the_can_raises(rng):
    tiny = PlaceRegion(min_xy=(0.0, 0.0), max_xy=(0.05, 0.05), min_place_distance_m=0.2)

    with pytest.raises(ValueError, match="too small"):
        sample_place_xy(rng, tiny, (0.025, 0.025))


def test_margin_wider_than_the_region_raises(rng):
    narrow = PlaceRegion(min_xy=(0.0, 0.0), max_xy=(0.05, 0.5), edge_margin_m=0.03)

    with pytest.raises(ValueError, match="no room"):
        sample_place_xy(rng, narrow, (1.0, 1.0))


def test_inverted_ranges_are_rejected():
    with pytest.raises(ValueError):
        KnobRanges(closure_m=(0.01, -0.01))
    with pytest.raises(ValueError):
        PlaceRegion(min_xy=(0.5, 0.0), max_xy=(0.2, 0.1))


def test_knobs_stay_inside_their_ranges(rng):
    ranges = KnobRanges(
        grasp_dz_m=(-0.01, 0.005), closure_m=(-0.002, 0.003), yaw_offset_rad=(-0.3, 0.3)
    )
    draws = [sample_knobs(rng, REGION, CAN_XY, ranges) for _ in range(500)]

    for name in ("grasp_dz_m", "closure_m", "yaw_offset_rad"):
        lo, hi = getattr(ranges, name)
        values = np.array([getattr(k, name) for k in draws])
        assert np.all(values >= lo) and np.all(values <= hi)
        assert values.std() > 0.0


def test_zero_width_ranges_return_exactly_lo(rng):
    ranges = KnobRanges(grasp_dz_m=(-0.004, -0.004), closure_m=(0.002, 0.002))

    knobs = sample_knobs(rng, REGION, CAN_XY, ranges)

    assert knobs.grasp_dz_m == -0.004
    assert knobs.closure_m == 0.002
    assert knobs.yaw_offset_rad == 0.0


def test_default_ranges_are_zero_width():
    assert KnobRanges() == KnobRanges((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def test_same_seed_gives_same_knobs():
    ranges = KnobRanges(yaw_offset_rad=(-0.2, 0.2))
    a = sample_knobs(np.random.default_rng(7), REGION, CAN_XY, ranges)
    b = sample_knobs(np.random.default_rng(7), REGION, CAN_XY, ranges)

    assert a == b


# --- Stop policy --------------------------------------------------------------------


def policy(**limits) -> StopPolicy:
    return StopPolicy(StopLimits(**limits), started_at=100.0)


def test_grasp_miss_retries_up_to_the_cap_then_stops():
    p = policy(max_grasp_retries=2, max_consecutive_failures=10)

    assert p.after_attempt(ErrorCode.GRASP_MISSED) is Decision.RETRY
    assert p.after_attempt(ErrorCode.GRASP_MISSED) is Decision.RETRY
    assert p.after_attempt(ErrorCode.GRASP_MISSED) is Decision.STOP
    assert p.stop_reason == "grasp_retries"


def test_retry_cap_wins_when_it_trips_with_the_failure_streak():
    p = policy()  # defaults: two retries, three consecutive failures

    decisions = [p.after_attempt(ErrorCode.GRASP_MISSED) for _ in range(3)]

    assert decisions == [Decision.RETRY, Decision.RETRY, Decision.STOP]
    assert p.stop_reason == "grasp_retries"


def test_consecutive_failures_stop_when_tighter_than_the_retry_cap():
    p = policy(max_grasp_retries=5, max_consecutive_failures=3)

    decisions = [p.after_attempt(ErrorCode.GRASP_MISSED) for _ in range(3)]

    assert decisions == [Decision.RETRY, Decision.RETRY, Decision.STOP]
    assert p.stop_reason == "consecutive_failures"


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.NO_CAN,
        ErrorCode.NO_PAPER,
        ErrorCode.OUT_OF_WORKSPACE,
        ErrorCode.HARDWARE_NOT_ACTIVE,
        ErrorCode.STALE_POSE,
    ],
)
def test_non_retryable_error_stops_at_once(code):
    p = policy()

    assert p.after_attempt(code) is Decision.STOP
    assert p.stop_reason == f"error:{code.name}"


def test_error_reason_names_the_code():
    p = policy()
    p.after_attempt(int(ErrorCode.NO_CAN))

    assert p.stop_reason == "error:NO_CAN"


def test_cancelled_goal_is_an_operator_stop():
    p = policy()

    assert p.after_attempt(ErrorCode.CANCELLED) is Decision.STOP
    assert p.stop_reason == "operator"


def test_operator_request_stops_before_the_next_cycle():
    p = policy()
    assert p.before_cycle(now=101.0, free_bytes=PLENTY_OF_DISK) is None

    p.request_stop()

    assert p.before_cycle(now=102.0, free_bytes=PLENTY_OF_DISK) == "operator"


def test_operator_request_mid_cycle_stops_after_a_success():
    p = policy()
    p.request_stop("operator")

    assert p.after_attempt(ErrorCode.SUCCESS) is Decision.STOP
    assert p.successes == 1


def test_first_stop_reason_is_kept():
    p = policy()
    p.after_attempt(ErrorCode.NO_PAPER)
    p.request_stop("operator")

    assert p.before_cycle(now=0.0, free_bytes=0) == "error:NO_PAPER"


def test_low_disk_stops_before_a_cycle():
    p = policy(min_free_bytes=50_000_000_000)

    assert p.before_cycle(now=101.0, free_bytes=50_000_000_000) is None
    assert p.before_cycle(now=101.0, free_bytes=49_999_999_999) == "disk"


def test_run_time_limit_stops_before_a_cycle():
    p = policy(max_duration_s=60.0)

    assert p.before_cycle(now=159.9, free_bytes=PLENTY_OF_DISK) is None
    assert p.before_cycle(now=160.0, free_bytes=PLENTY_OF_DISK) == "time"


def test_cycle_limit_counts_successes():
    p = policy(max_cycles=2)
    p.after_attempt(ErrorCode.SUCCESS)
    p.after_attempt(ErrorCode.GRASP_MISSED)
    assert p.before_cycle(now=101.0, free_bytes=PLENTY_OF_DISK) is None

    p.after_attempt(ErrorCode.SUCCESS)

    assert p.before_cycle(now=101.0, free_bytes=PLENTY_OF_DISK) == "cycles"


def test_unlimited_by_default():
    p = policy()
    for _ in range(1000):
        p.after_attempt(ErrorCode.SUCCESS)

    assert p.before_cycle(now=1e9, free_bytes=PLENTY_OF_DISK) is None


def test_success_resets_retries_and_the_failure_streak():
    p = policy()  # two retries, three consecutive failures
    for _ in range(4):
        assert p.after_attempt(ErrorCode.GRASP_MISSED) is Decision.RETRY
        assert p.after_attempt(ErrorCode.GRASP_MISSED) is Decision.RETRY
        assert p.after_attempt(ErrorCode.SUCCESS) is Decision.CONTINUE

    assert p.stop_reason is None
    assert (p.attempts, p.successes, p.failures) == (12, 4, 8)
    assert (p.retries, p.consecutive_failures) == (0, 0)


# --- Metadata -----------------------------------------------------------------------

LOADER_KEYS = {"version", "status", "note", "duration"}
KNOBS = Knobs(place_xy=(0.35, 0.05), grasp_dz_m=-0.002, closure_m=0.001, yaw_offset_rad=0.1)
START = datetime(2026, 10, 8, 14, 23, 1, tzinfo=UTC)


def test_episode_names_follow_the_loader_layout():
    assert episode_name(1) == "0001"
    assert episode_name(42) == "0042"
    with pytest.raises(ValueError):
        episode_name(0)


def test_run_id_and_timestamps_are_utc():
    two_hours_east = timezone(timedelta(hours=2))
    local = START.astimezone(two_hours_east)

    assert run_id(local) == "20261008T142301Z"
    assert iso_utc(local) == "2026-10-08T14:23:01Z"
    with pytest.raises(ValueError):
        run_id(datetime(2026, 10, 8))


@pytest.mark.parametrize(
    ("code", "status"), [(ErrorCode.SUCCESS, "success"), (ErrorCode.GRASP_MISSED, "failure")]
)
def test_episode_metadata_keeps_the_loader_keys(code, status):
    meta = episode_metadata(
        code=code,
        attempt=1,
        can_xy=np.array([0.3, 0.0]),
        knobs=KNOBS,
        started_at=iso_utc(START),
        duration_s=np.float64(12.3456),
    )

    assert meta.keys() >= LOADER_KEYS
    assert meta["version"] == 1 and meta["note"] is None
    assert meta["status"] == status
    assert meta["error_code"] == code.name
    assert meta["duration"] == 12.346
    assert meta["place_xy"] == [0.35, 0.05]
    assert json.loads(json.dumps(meta)) == meta


def test_run_metadata_is_json_ready():
    meta = run_metadata(
        object_name="can",
        object_config={"height_m": 0.12},
        region=REGION,
        knob_ranges=KnobRanges(yaw_offset_rad=(-0.2, 0.2)),
        limits=StopLimits(max_cycles=50),
        speed_scale=0.5,
        record=True,
        git_sha="321b54d",
        started_at=iso_utc(START),
    )

    assert meta["object"] == "can"
    assert meta["stop_limits"]["max_cycles"] == 50
    assert meta["knob_ranges"]["yaw_offset_rad"] == [-0.2, 0.2]
    assert json.loads(json.dumps(meta)) == meta


def test_summary_reports_counters_and_reason():
    p = policy()
    p.after_attempt(ErrorCode.SUCCESS)
    p.after_attempt(ErrorCode.NO_CAN)

    summary = summary_metadata(p, episodes=2, now=130.0, ended_at=iso_utc(START))

    assert summary == {
        "attempts": 2,
        "episodes": 2,
        "successes": 1,
        "failures": 1,
        "stop_reason": "error:NO_CAN",
        "duration_s": 30.0,
        "ended_at": "2026-10-08T14:23:01Z",
    }
    assert json.loads(json.dumps(summary)) == summary


@pytest.mark.parametrize(
    "kwargs", [{"max_grasp_retries": -1}, {"min_free_bytes": -1}, {"max_cycles": -1}]
)
def test_negative_stop_limits_are_refused(kwargs):
    with pytest.raises(ValueError):
        StopLimits(**kwargs)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_shipped_loop_config_uses_perceptions_pick_region():
    loop = load_loop_config(CONFIG_DIR / "collect.yaml", CONFIG_DIR / "perception.yaml")
    assert loop.region.min_xy == (0.10, -0.32)
    assert loop.region.max_xy == (0.45, 0.02)
    assert loop.max_grasp_retries == 2
    # The shipped margins must leave room to place a can away from any can position.
    rng = np.random.default_rng(0)
    for can_xy in [(0.10, -0.32), (0.275, -0.15), (0.45, 0.02)]:
        sample_place_xy(rng, loop.region, can_xy)


def test_unknown_loop_key_is_refused(tmp_path):
    collect = tmp_path / "collect.yaml"
    collect.write_text("loop:\n  edge_margin: 0.05\n")
    with pytest.raises(ValueError, match="unknown loop keys"):
        load_loop_config(collect, CONFIG_DIR / "perception.yaml")


def test_missing_loop_section_is_refused(tmp_path):
    collect = tmp_path / "collect.yaml"
    collect.write_text("recording: {}\n")
    with pytest.raises(ValueError, match="no loop section"):
        load_loop_config(collect, CONFIG_DIR / "perception.yaml")


@pytest.mark.parametrize(
    ("code", "recorder_died", "stop_requested", "reason"),
    [
        (None, False, True, "operator"),
        (None, False, False, "error:NO_RESULT"),
        (ErrorCode.CANCELLED, False, False, "error:GOAL_TIMEOUT"),
        (ErrorCode.CANCELLED, False, True, None),
        (ErrorCode.SUCCESS, True, False, "error:RECORDER"),
        (ErrorCode.GRASP_MISSED, False, False, None),
        (ErrorCode.SUCCESS, False, False, None),
    ],
)
def test_forced_stop_reason(code, recorder_died, stop_requested, reason):
    assert (
        forced_stop_reason(code, recorder_died=recorder_died, stop_requested=stop_requested)
        == reason
    )


def test_timed_out_goal_stops_as_a_timeout_not_the_operator():
    policy = StopPolicy(StopLimits(), started_at=0.0)
    policy.request_stop(
        forced_stop_reason(ErrorCode.CANCELLED, recorder_died=False, stop_requested=False)
    )
    assert policy.after_attempt(ErrorCode.CANCELLED) is Decision.STOP
    assert policy.stop_reason == "error:GOAL_TIMEOUT"


def test_non_positive_loop_timeouts_are_refused(tmp_path):
    collect = tmp_path / "collect.yaml"
    collect.write_text(
        "loop:\n  edge_margin_m: 0.05\n  min_place_distance_m: 0.08\n  max_grasp_retries: 2\n"
        "  max_consecutive_failures: 3\n  can_pose_timeout_s: 0\n  goal_timeout_s: 300\n"
    )
    with pytest.raises(ValueError, match="timeouts"):
        load_loop_config(collect, CONFIG_DIR / "perception.yaml")


def test_perception_file_without_pick_region_is_refused(tmp_path):
    perception = tmp_path / "perception.yaml"
    perception.write_text("/**:\n  ros__parameters: {}\n")
    with pytest.raises(ValueError, match="pick region"):
        load_loop_config(CONFIG_DIR / "collect.yaml", perception)
