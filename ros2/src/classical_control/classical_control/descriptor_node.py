"""Publish the controller's control descriptor, latched, on /control/descriptor.

The descriptor (``control/anvil_classical.yaml``, fm-sim's ``control_descriptor``
schema) draws this controller's pipeline. A viewer reads it the same way on the robot
and in sim: as JSON in a ``std_msgs/String``, so it needs no YAML parser. Transient
local durability hands it to a viewer that subscribes after the launch started.
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

DESCRIPTOR_TOPIC = "/control/descriptor"
LATCHED = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)


def checked_control_file(path: str | Path, control_dir: str | Path) -> Path:
    """Resolve ``path`` and refuse anything outside ``control_dir`` or not a yaml file.

    The parameter is settable by anyone on the ROS domain at startup; confining it to the
    package's descriptors keeps the node from publishing an arbitrary file's contents.
    """
    resolved = Path(path).resolve()
    if resolved.suffix != ".yaml" or not resolved.is_relative_to(Path(control_dir).resolve()):
        raise ValueError(f"control_file must be a .yaml file in {control_dir}")
    return resolved


def descriptor_json(path: str | Path) -> str:
    """Read a control yaml and return it as the JSON published on the topic."""
    return json.dumps(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class DescriptorNode(Node):
    """Publishes ``control_file`` once; the latched QoS keeps it for late subscribers."""

    def __init__(self) -> None:
        super().__init__("control_descriptor")
        control_dir = Path(get_package_share_directory("classical_control")) / "control"
        path = checked_control_file(self.declare_parameter("control_file", "").value, control_dir)
        self._publisher = self.create_publisher(String, DESCRIPTOR_TOPIC, LATCHED)
        self._publisher.publish(String(data=descriptor_json(path)))
        self.get_logger().info(f"published {path} on {DESCRIPTOR_TOPIC}")


def main(args: list[str] | None = None) -> None:
    """Publish the descriptor and keep it latched until shutdown."""
    rclpy.init(args=args)
    node = DescriptorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
