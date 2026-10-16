"""Stand-ins for the ROS modules the nodes import, used only where ROS is not installed.

The node and strategy logic under test is plain Python; mocking rclpy and the message
packages lets it run under plain pytest, as tests/unit/anvil_eval_ros does.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

_ROS_MODULES = (
    "rclpy",
    "rclpy.callback_groups",
    "rclpy.executors",
    "rclpy.node",
    "rclpy.qos",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
)

for _module in _ROS_MODULES:
    try:
        importlib.import_module(_module)
    except ImportError:
        sys.modules[_module] = MagicMock()
        if _module == "rclpy.node":
            sys.modules[_module].Node = type("Node", (), {})
        if _module == "std_msgs.msg":
            sys.modules[_module].Float64MultiArray = type("Float64MultiArray", (), {"data": None})

try:
    import torch  # noqa: F401
except ImportError:
    # These import the node through lerobot_control, which needs torch. Ignored rather
    # than skipped: under ROS's pytest plugins a module-level skip drops the whole run.
    collect_ignore = ["test_profile_joints.py"]

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[4] / "configs" / "lerobot_control" / "inference_default.yaml"
)


@pytest.fixture
def joint_names_config() -> dict:
    """The shipped bimanual config's joint_names section: the orders trained policies use."""
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["joint_names"]
