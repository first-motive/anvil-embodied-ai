"""Tests for what collect_node loads before a run starts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# collect_node imports ROS and the action interface at module level. Where they are not
# installed (plain pytest), stand-ins let the pure settings loader be tested.
for _module in (
    "rclpy",
    "rclpy.action",
    "rclpy.node",
    "rclpy.qos",
    "rclpy.signals",
    "classical_control_msgs",
    "classical_control_msgs.action",
    "geometry_msgs",
    "geometry_msgs.msg",
    "rcl_interfaces",
    "rcl_interfaces.msg",
    "rcl_interfaces.srv",
):
    try:
        importlib.import_module(_module)
    except ImportError:
        sys.modules[_module] = MagicMock()
        if _module == "rclpy.node":
            sys.modules[_module].Node = type("Node", (), {})

from classical_control.collect_node import RunSettings, parse_args  # noqa: E402


def _args(tmp_path: Path, *extra: str):
    return parse_args(
        [
            "--object=can",
            f"--objects-dir={CONFIG_DIR / 'objects'}",
            f"--collect-config={CONFIG_DIR / 'collect.yaml'}",
            f"--perception-config={CONFIG_DIR / 'perception.yaml'}",
            f"--out-root={tmp_path}",
            *extra,
        ]
    )


@pytest.mark.parametrize(
    ("extra", "topic"),
    [
        ((), "/follower_r_forward_position_controller/commands"),
        (("--robot-profile=fm_openarm_v2",), "/openarm_right_forward_position_controller/commands"),
    ],
)
def test_run_records_the_profile_arm_command_topic(tmp_path, extra, topic) -> None:
    topics = RunSettings.load(_args(tmp_path, *extra)).recording.topics
    assert topics.count(topic) == 1
    assert "/joint_states" in topics


def test_a_command_topic_already_in_the_config_is_not_duplicated(tmp_path) -> None:
    collect = tmp_path / "collect.yaml"
    text = (CONFIG_DIR / "collect.yaml").read_text(encoding="utf-8")
    collect.write_text(
        text.replace(
            "    - /joint_states ",
            "    - /follower_r_forward_position_controller/commands\n    - /joint_states ",
            1,
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, f"--collect-config={collect}")
    topics = RunSettings.load(args).recording.topics
    assert topics.count("/follower_r_forward_position_controller/commands") == 1
