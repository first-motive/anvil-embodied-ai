"""Tests for the episode miner's pure core on synthetic gripper and pose series."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml
from classical_control.episode_miner import (
    CSV_COLUMNS,
    EpisodeResult,
    find_gripper_events,
    mine_episode,
    pose_at,
    summarise,
    write_ground_truth_csv,
    write_params_yaml,
)
from scipy.spatial.transform import Rotation

RATE_HZ = 500.0


def _piecewise(knots: list[tuple[float, float]], noise: float = 2e-4, seed: int = 0):
    """Linear finger trace through (time, value) knots, sampled at 500 Hz with noise."""
    t_knots, v_knots = zip(*knots)
    times = np.arange(t_knots[0], t_knots[-1], 1.0 / RATE_HZ)
    finger = np.interp(times, t_knots, v_knots)
    finger += np.random.default_rng(seed).normal(0.0, noise, times.size)
    return times, finger


def _can_trace(closed_on: float = 0.0185):
    # home -> open -> close on can -> hold -> release -> episode ends mid-opening
    return _piecewise(
        [
            (0.0, 0.045),
            (1.0, 0.045),
            (1.5, 0.050),
            (5.0, 0.050),
            (5.5, closed_on),
            (10.0, closed_on),
            (10.5, 0.025),
        ]
    )


def _time_of_value(start: float, end: float, v0: float, v1: float, v: float) -> float:
    return start + (end - start) * (v - v0) / (v1 - v0)


def test_detects_close_and_release_on_noisy_can_trace():
    times, finger = _can_trace()
    events = find_gripper_events(times, finger)

    assert events is not None
    expected_close = _time_of_value(5.0, 5.5, 0.050, 0.0185, 0.030)
    assert events.close_t == pytest.approx(expected_close, abs=0.02)
    expected_open = _time_of_value(10.0, 10.5, 0.0185, 0.025, 0.0185 + 0.003)
    # Noise pulls the running minimum down and spikes the rise up, so release fires early.
    assert events.open_t == pytest.approx(expected_open, abs=0.1)
    assert events.closed_value == pytest.approx(0.0185, abs=1e-3)


def test_empty_close_to_zero_still_detects_close():
    times, finger = _can_trace(closed_on=0.0)
    events = find_gripper_events(times, finger)

    assert events is not None
    expected_close = _time_of_value(5.0, 5.5, 0.050, 0.0, 0.030)
    assert events.close_t == pytest.approx(expected_close, abs=0.02)
    assert events.closed_value == pytest.approx(0.0, abs=1e-3)


def test_no_close_returns_none():
    times, finger = _piecewise([(0.0, 0.045), (2.0, 0.050), (4.0, 0.045)])
    assert find_gripper_events(times, finger) is None


def test_starting_closed_is_not_a_grasp():
    times, finger = _piecewise([(0.0, 0.0185), (3.0, 0.0185)])
    assert find_gripper_events(times, finger) is None


def test_release_absent_gives_open_none():
    times, finger = _piecewise([(0.0, 0.050), (1.0, 0.050), (1.5, 0.0185), (6.0, 0.0185)])
    events = find_gripper_events(times, finger)

    assert events is not None
    assert events.open_t is None


def test_pose_at_picks_nearest_sample():
    times = np.array([0.0, 1.0, 2.0])
    poses = np.arange(21, dtype=float).reshape(3, 7)

    assert pose_at(times, poses, 0.4)[0] == 0.0
    assert pose_at(times, poses, 0.6)[0] == 7.0
    assert pose_at(times, poses, 5.0)[0] == 14.0
    assert pose_at(times, poses, -1.0)[0] == 0.0


def test_mine_episode_reports_times_from_episode_start():
    times, finger = _can_trace()
    times = times + 100.0
    ee_times = np.arange(100.0, 110.5, 1.0 / 90.0)
    ee_poses = np.tile([0.43, -0.06, 0.345, 0.0, 0.0, 0.0, 1.0], (ee_times.size, 1))

    result = mine_episode("0001", times, finger, ee_times, ee_poses)

    assert result is not None
    expected_close = _time_of_value(5.0, 5.5, 0.050, 0.0185, 0.030)
    assert result.close_t == pytest.approx(expected_close, abs=0.02)
    assert result.grasp_pose[2] == pytest.approx(0.345)


def _result(name, grasp_z, place_z, quat=(0.0, 0.0, 0.0, 1.0)):
    grasp = np.array([0.43, -0.06, grasp_z, *quat])
    place = None if place_z is None else np.array([0.35, -0.13, place_z, *quat])
    return EpisodeResult(name, 5.0, None if place_z is None else 10.0, grasp, place)


def test_summarise_medians_and_counts():
    episodes = [
        _result("0001", 0.34, 0.35),
        _result("0002", 0.345, 0.36),
        _result("0003", 0.35, None),
    ]
    params = summarise(episodes, total=4)

    assert params["episodes"] == {"total": 4, "with_close": 3, "with_open": 2}
    assert params["grasp_z"]["median"] == pytest.approx(0.345)
    assert params["grasp_z"]["n"] == 3
    assert params["place_z"]["median"] == pytest.approx(0.355)
    assert params["grasp_z"]["p10"] <= params["grasp_z"]["median"] <= params["grasp_z"]["p90"]


def test_summarise_quaternion_mean_of_symmetric_tilts():
    tilt = 10.0
    quats = [Rotation.from_euler("x", angle, degrees=True).as_quat() for angle in (-tilt, tilt)]
    episodes = [_result(f"{i:04d}", 0.345, 0.35, tuple(q)) for i, q in enumerate(quats)]
    params = summarise(episodes)

    mean = Rotation.from_quat(params["grasp_orientation_xyzw"])
    assert mean.magnitude() == pytest.approx(0.0, abs=1e-9)
    assert params["grasp_orientation_spread_deg"]["max"] == pytest.approx(tilt)


def test_summarise_rejects_empty():
    with pytest.raises(ValueError):
        summarise([])


def test_yaml_round_trip(tmp_path):
    params = summarise([_result("0001", 0.34, 0.35), _result("0002", 0.35, 0.36)])
    path = tmp_path / "mined_params.yaml"

    write_params_yaml(params, path)

    assert yaml.safe_load(path.read_text()) == params


def test_csv_round_trip(tmp_path):
    episodes = [_result("0001", 0.34, 0.35), _result("0002", 0.35, None)]
    path = tmp_path / "ground_truth.csv"

    write_ground_truth_csv(episodes, path)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[0]["episode"] == "0001"
    assert float(rows[0]["can_z"]) == pytest.approx(0.34)
    assert float(rows[0]["paper_z"]) == pytest.approx(0.35)
    assert rows[1]["open_t"] == ""
    assert rows[1]["paper_x"] == ""
