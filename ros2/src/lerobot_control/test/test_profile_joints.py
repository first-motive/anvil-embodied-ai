"""Tests that the inference node and strategy use profile joint names, in policy order,
and never stand in 0.0 for a joint the robot does not publish."""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from anvil_robot_profiles import load_profile
from lerobot_control.inference_node import JOINT_STATE_MAX_AGE_S, LeRobotInferenceNode
from lerobot_control.strategies import multi_process
from lerobot_control.strategies.multi_process import MultiProcessStrategy

PROFILES = ("loader_openarm_v2", "fm_openarm_v2")


def _all_joints(profile) -> dict[str, float]:
    """Every profile joint at a distinct position, listed in an order no policy uses."""
    names = [n for arm in profile.arms.values() for n in arm.command_joints]
    return {name: float(i) for i, name in enumerate(reversed(names))}


def _joint_state(positions: dict[str, float], velocities: bool = True):
    return types.SimpleNamespace(
        name=list(positions),
        position=list(positions.values()),
        velocity=[0.5] * len(positions) if velocities else [],
        effort=[],
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0)),
    )


@pytest.fixture
def clock(monkeypatch):
    """A settable time.monotonic for the strategy's receive stamps."""
    now = types.SimpleNamespace(t=100.0)
    monkeypatch.setattr(multi_process.time, "monotonic", lambda: now.t)
    return now


def _strategy(monkeypatch, profile, joint_names_config) -> MultiProcessStrategy:
    strategy = MultiProcessStrategy()
    for method in ("_setup_shared_memory", "_start_workers", "_setup_joint_subscription"):
        monkeypatch.setattr(strategy, method, lambda *_: None)
    strategy.setup(
        node=MagicMock(),
        config={},
        camera_mapping={},
        joint_names_config=joint_names_config,
        robot_profile=profile,
        joint_state_topic="/joint_states",
        image_shape=(1, 1, 3),
    )
    strategy._image_buffer = MagicMock(read_all_if_ready=lambda: {})
    return strategy


# --- Strategy: observation.state ---


@pytest.mark.parametrize("name", PROFILES)
def test_state_vector_keeps_the_old_layout(monkeypatch, name, joint_names_config) -> None:
    profile = load_profile(name)
    strategy = _strategy(monkeypatch, profile, joint_names_config)
    strategy._joint_callback(_joint_state(_all_joints(profile)))

    state = strategy.get_observation([])["observation.state"].squeeze(0).tolist()

    # The loop the strategy ran before profiles: f"{prefix}_{arm_key}_{joint_id}" over
    # sorted arm_mapping keys (l, r), with the loader prefix "follower".
    order = joint_names_config["model_joint_order"]
    keys = sorted(joint_names_config["arm_mapping"])
    expected = {
        "loader_openarm_v2": [f"follower_{key}_{j}" for key in keys for j in order],
        "fm_openarm_v2": [f"openarm_{side}_{j}" for side in ("left", "right") for j in order],
    }[name]
    assert state == [_all_joints(profile)[n] for n in expected]


def test_missing_state_joint_skips_the_observation(monkeypatch, joint_names_config) -> None:
    profile = load_profile()
    strategy = _strategy(monkeypatch, profile, joint_names_config)
    positions = _all_joints(profile)
    del positions["follower_r_joint4"]
    strategy._joint_callback(_joint_state(positions))

    assert strategy.get_observation([]) is None
    assert "follower_r_joint4" in strategy.get_incomplete_reason()


def test_non_finite_position_counts_as_missing(monkeypatch, joint_names_config) -> None:
    profile = load_profile()
    strategy = _strategy(monkeypatch, profile, joint_names_config)
    strategy._joint_callback(_joint_state({**_all_joints(profile), "follower_l_joint2": math.nan}))

    assert "follower_l_joint2" not in strategy.get_current_joint_positions()
    assert strategy.get_observation([]) is None
    assert "follower_l_joint2" in strategy.get_incomplete_reason()


def test_missing_velocity_skips_a_velocity_policy(monkeypatch, joint_names_config) -> None:
    profile = load_profile()
    config = {**joint_names_config, "state_features": ["position", "velocity"]}
    strategy = _strategy(monkeypatch, profile, config)

    strategy._joint_callback(_joint_state(_all_joints(profile), velocities=False))
    assert strategy.get_observation([]) is None
    assert "velocity" in strategy.get_incomplete_reason()

    strategy._joint_callback(_joint_state(_all_joints(profile)))
    observation = strategy.get_observation([])
    assert observation["observation.velocity"].squeeze(0).tolist() == [0.5] * 16


def test_joint_state_age_counts_from_receipt(monkeypatch, clock, joint_names_config) -> None:
    strategy = _strategy(monkeypatch, load_profile(), joint_names_config)
    assert strategy.joint_state_age() is None

    strategy._joint_callback(_joint_state(_all_joints(load_profile())))
    clock.t += 0.25

    assert strategy.joint_state_age() == pytest.approx(0.25)


# --- Node: commands ---


def _node(profile, joint_names_config, current: dict[str, float], age: float | None = 0.01):
    """A stand-in node carrying only what the publish path reads."""
    arms_config = {
        "left": {"action_start": 0, "action_end": 8},
        "right": {"action_start": 8, "action_end": 16},
    }
    order = joint_names_config["controller_joint_order"]
    logger = MagicMock()
    return types.SimpleNamespace(
        robot_profile=profile,
        joint_names_config=joint_names_config,
        joint_state_topic="/joint_states",
        arms_config=arms_config,
        _command_joint_names={arm: profile.joint_names([arm], order) for arm in arms_config},
        strategy=MagicMock(get_current_joint_positions=lambda: current, joint_state_age=lambda: age),
        action_limiter=MagicMock(process=lambda action, _reference: action),
        arm_publishers={arm: MagicMock() for arm in arms_config},
        _debug=False,
        _monitor_enable=False,
        _smooth_tracker=None,
        metrics=MagicMock(),
        logger=logger,
        get_logger=lambda: logger,
    )


def _published(node) -> set[str]:
    return {arm for arm, pub in node.arm_publishers.items() if pub.publish.called}


@pytest.mark.parametrize("name", PROFILES)
def test_action_is_limited_against_current_positions_in_command_order(
    name, joint_names_config
) -> None:
    profile = load_profile(name)
    current = _all_joints(profile)
    node = _node(profile, joint_names_config, current)
    references = []
    node.action_limiter = MagicMock(
        process=lambda action, reference: references.append(reference) or action
    )

    LeRobotInferenceNode._publish_action(node, np.zeros(16))

    for arm, reference in zip(("left", "right"), references):
        assert reference.tolist() == [current[n] for n in profile.arms[arm].command_joints]


def test_missing_command_joint_skips_only_that_arm(joint_names_config) -> None:
    profile = load_profile()
    current = _all_joints(profile)
    del current["follower_r_finger_joint1"]
    node = _node(profile, joint_names_config, current)

    LeRobotInferenceNode._publish_action(node, np.zeros(16))

    assert _published(node) == {"left"}
    assert "follower_r_finger_joint1" in node.logger.error.call_args.args[0]


def test_no_positions_commands_no_arm(joint_names_config) -> None:
    node = _node(load_profile(), joint_names_config, {})

    LeRobotInferenceNode._publish_action(node, np.zeros(16))

    assert _published(node) == set()


@pytest.mark.parametrize("age", [None, JOINT_STATE_MAX_AGE_S + 0.1])
def test_stale_joint_state_commands_no_arm(joint_names_config, age) -> None:
    profile = load_profile()
    node = _node(profile, joint_names_config, _all_joints(profile), age=age)

    LeRobotInferenceNode._publish_action(node, np.zeros(16))

    assert _published(node) == set()
    node.logger.error.assert_called_once()


def test_non_finite_joint_state_skips_that_arm(monkeypatch, joint_names_config) -> None:
    profile = load_profile()
    strategy = _strategy(monkeypatch, profile, joint_names_config)
    strategy._joint_callback(_joint_state({**_all_joints(profile), "follower_r_joint1": math.inf}))
    node = _node(profile, joint_names_config, {})
    node.strategy = strategy

    LeRobotInferenceNode._publish_action(node, np.zeros(16))

    assert _published(node) == {"left"}


# --- Node: startup ---


def _wait_for(monkeypatch, replies):
    """Stand in for rclpy.wait_for_message, answering each call from `replies`."""
    calls = iter(replies)
    module = types.SimpleNamespace(wait_for_message=lambda *_, **__: next(calls))
    monkeypatch.setitem(sys.modules, "rclpy.wait_for_message", module)


@pytest.mark.parametrize("name", PROFILES)
def test_startup_waits_then_accepts_a_robot_with_every_joint(
    monkeypatch, name, joint_names_config
) -> None:
    profile = load_profile(name)
    message = types.SimpleNamespace(name=list(_all_joints(profile)))
    _wait_for(monkeypatch, [(False, None), (True, message)])
    node = _node(profile, joint_names_config, {})

    LeRobotInferenceNode._require_profile_joints(node)

    node.logger.warning.assert_called_once()


def test_startup_refuses_a_robot_missing_a_profile_joint(monkeypatch, joint_names_config) -> None:
    # The fm profile's names against a robot publishing the loader's names.
    message = types.SimpleNamespace(name=list(_all_joints(load_profile("loader_openarm_v2"))))
    _wait_for(monkeypatch, [(True, message)])
    node = _node(load_profile("fm_openarm_v2"), joint_names_config, {})

    with pytest.raises(RuntimeError, match="openarm_right_joint1"):
        LeRobotInferenceNode._require_profile_joints(node)
