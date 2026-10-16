"""Tests for the hold command the inference node sends each arm on shutdown."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
from anvil_robot_profiles import load_profile
from lerobot_control.inference_node import JOINT_STATE_MAX_AGE_S, LeRobotInferenceNode
from lerobot_control.strategies import multi_process
from lerobot_control.strategies.multi_process import MultiProcessStrategy

ARMS_CONFIG = {
    "left": {"action_start": 0, "action_end": 8},
    "right": {"action_start": 8, "action_end": 16},
}


def _current() -> dict[str, float]:
    profile = load_profile()
    names = [n for arm in profile.arms.values() for n in arm.command_joints]
    return {name: float(i) for i, name in enumerate(names)}


def _node(strategy, joint_names_config: dict):
    profile = load_profile()
    order = joint_names_config["controller_joint_order"]
    logger = MagicMock()
    return types.SimpleNamespace(
        arms_config=ARMS_CONFIG,
        _command_joint_names={arm: profile.joint_names([arm], order) for arm in ARMS_CONFIG},
        strategy=strategy,
        arm_publishers={arm: MagicMock() for arm in ARMS_CONFIG},
        logger=logger,
        get_logger=lambda: logger,
    )


def _fake_strategy(current: dict[str, float], age: float | None):
    return MagicMock(get_current_joint_positions=lambda: current, joint_state_age=lambda: age)


def _sent(node, arm: str) -> list[float] | None:
    publish = node.arm_publishers[arm].publish
    return publish.call_args.args[0].data if publish.called else None


def test_both_arms_hold_their_current_positions_in_command_order(joint_names_config) -> None:
    current = _current()
    node = _node(_fake_strategy(current, 0.01), joint_names_config)

    LeRobotInferenceNode._publish_hold_position(node)

    for arm in ("left", "right"):
        expected = [current[n] for n in load_profile().arms[arm].command_joints]
        assert _sent(node, arm) == expected
    assert "left, right" in node.logger.info.call_args.args[0]


def test_arm_with_a_missing_joint_is_not_held(joint_names_config) -> None:
    current = _current()
    del current["follower_r_joint3"]
    node = _node(_fake_strategy(current, 0.01), joint_names_config)

    LeRobotInferenceNode._publish_hold_position(node)

    assert _sent(node, "right") is None
    assert _sent(node, "left") is not None
    assert "follower_r_joint3" in node.logger.error.call_args.args[0]


def test_no_arm_held_logs_no_success(joint_names_config) -> None:
    current = {k: v for k, v in _current().items() if not k.endswith("joint1")}
    node = _node(_fake_strategy(current, 0.01), joint_names_config)

    LeRobotInferenceNode._publish_hold_position(node)

    assert _sent(node, "left") is None and _sent(node, "right") is None
    node.logger.info.assert_not_called()


@pytest.mark.parametrize(
    ("current", "age", "reason"),
    [
        (_current(), None, "never received"),
        ({}, 0.01, "no finite positions"),
    ],
)
def test_absent_joint_states_send_no_hold(joint_names_config, current, age, reason) -> None:
    node = _node(_fake_strategy(current, age), joint_names_config)

    LeRobotInferenceNode._publish_hold_position(node)

    assert _sent(node, "left") is None and _sent(node, "right") is None
    assert reason in node.logger.error.call_args.args[0]


def test_stale_joint_state_sends_no_hold(monkeypatch, joint_names_config) -> None:
    # A real strategy and its receive stamp: the state arrives, then goes stale.
    now = types.SimpleNamespace(t=100.0)
    monkeypatch.setattr(multi_process.time, "monotonic", lambda: now.t)
    strategy = MultiProcessStrategy()
    current = _current()
    strategy._joint_callback(
        types.SimpleNamespace(
            name=list(current),
            position=list(current.values()),
            velocity=[],
            effort=[],
            header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0)),
        )
    )
    now.t += JOINT_STATE_MAX_AGE_S + 0.1
    node = _node(strategy, joint_names_config)

    LeRobotInferenceNode._publish_hold_position(node)

    assert _sent(node, "left") is None and _sent(node, "right") is None
    assert "s old" in node.logger.error.call_args.args[0]
