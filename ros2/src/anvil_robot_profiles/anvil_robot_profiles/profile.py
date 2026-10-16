"""Load a robot profile: the joint, controller and frame names of one robot build.

The same arm runs under different names (the Anvil loader's ``follower_*``, First
Motive's canonical ``openarm_*``). Nodes read names from a profile instead of
hardcoding them, so switching the joint, gripper and controller names is one
``robot_profile`` parameter. Topics the loader names per side (``/ee_pose_right``) and
camera topics stay fixed.

No ROS imports at module level: offline tools and plain pytest load profiles from the
source tree when the package is not installed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE = "anvil_robot_profiles"
DEFAULT_PROFILE = "loader_openarm_v2"
#: The arm the single-arm nodes (classical_control) drive.
DRIVEN_ARM = "right"
# ROS joint, frame and controller names; also the only shape a bare profile name may take,
# so a name can never step outside the profiles directory.
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SOURCE_PROFILES = Path(__file__).resolve().parent.parent / "profiles"
_ARM_KEYS = {
    "joints",
    "gripper_joint",
    "arm_controller",
    "trajectory_controller",
    "ee_frame",
    "hand_frame",
}


@dataclass(frozen=True)
class ArmProfile:
    """Names for one arm.

    Attributes:
        joints: Arm joints in controller order, gripper excluded.
        gripper_joint: Gripper joint name.
        arm_controller: Forward position controller taking joint commands.
        trajectory_controller: Joint trajectory controller.
        ee_frame: Tool centre point frame.
        hand_frame: Hand (gripper base) frame.
        joint_prefix: Text every joint name of this arm starts with, e.g. ``follower_r_``.
    """

    joints: tuple[str, ...]
    gripper_joint: str
    arm_controller: str
    trajectory_controller: str
    ee_frame: str
    hand_frame: str
    joint_prefix: str

    @property
    def command_joints(self) -> tuple[str, ...]:
        """Joints in the order the arm controller takes commands: arm, then gripper."""
        return (*self.joints, self.gripper_joint)

    @property
    def command_topic(self) -> str:
        """Topic the arm controller reads joint commands from."""
        return f"/{self.arm_controller}/commands"

    def joint_name(self, joint_id: str) -> str:
        """Map a robot-neutral joint id (``joint3``, ``finger_joint1``) to its full name.

        Datasets and policy configs name joints by id, without the arm prefix, so one
        trained policy runs under any profile of the same arm.

        Raises:
            KeyError: If no joint of this arm has that id.
        """
        name = self.joint_prefix + joint_id
        if name not in self.command_joints:
            raise KeyError(f"no joint {joint_id!r} on arm with joints {self.command_joints}")
        return name


@dataclass(frozen=True)
class RobotProfile:
    """Names for one robot build.

    Attributes:
        name: Profile name, the yaml file's stem.
        base_frame: Robot body frame the arms and fixed cameras hang off.
        arms: Arms by side (``right``, ``left``).
        cameras: Optional camera optical frames by camera name.
    """

    name: str
    base_frame: str
    arms: dict[str, ArmProfile]
    cameras: dict[str, str]

    def joint_names(self, arms: Iterable[str], joint_ids: Iterable[str]) -> list[str]:
        """Full joint names arm by arm, each arm's joints in ``joint_ids`` order.

        A policy's state and action vectors are laid out this way, so the order of both
        arguments is the vector order.

        Raises:
            KeyError: On an unknown arm or joint id.
        """
        ids = list(joint_ids)
        return [self.arms[arm].joint_name(joint_id) for arm in arms for joint_id in ids]


def profiles_dir() -> Path:
    """Return the directory of shipped profiles: the installed share, else the source tree."""
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return _SOURCE_PROFILES
    try:
        return Path(get_package_share_directory(PACKAGE)) / "profiles"
    except PackageNotFoundError:
        return _SOURCE_PROFILES


def load_profile(name_or_path: str | Path = DEFAULT_PROFILE) -> RobotProfile:
    """Load a shipped profile by name, or any profile yaml by path.

    Args:
        name_or_path: A shipped profile name such as ``fm_openarm_v2``, or a path to a
            ``.yaml`` file.

    Raises:
        FileNotFoundError: If no such profile exists.
        ValueError: If a bare name is not a plain identifier, or the yaml does not match
            the profile schema.
    """
    text = str(name_or_path)
    if text.endswith((".yaml", ".yml")):
        path = Path(text).expanduser()
    elif _NAME.fullmatch(text):
        path = profiles_dir() / f"{text}.yaml"
    else:
        raise ValueError(f"robot profile {text!r} is neither a profile name nor a .yaml path")
    if not path.is_file():
        shipped = sorted(p.stem for p in profiles_dir().glob("*.yaml"))
        raise FileNotFoundError(f"robot profile {text!r} not found; shipped: {shipped}")
    with open(path, encoding="utf-8") as stream:
        return parse_profile(yaml.safe_load(stream) or {}, source=str(path))


def declare_robot_profile(node: Any) -> RobotProfile:
    """Declare the ``robot_profile`` parameter on a ROS node and load that profile.

    Args:
        node: An ``rclpy.node.Node``.
    """
    from rcl_interfaces.msg import ParameterDescriptor

    descriptor = ParameterDescriptor(
        description=(
            "Robot profile naming joints, controllers and frames: a shipped name "
            f"({', '.join(sorted(p.stem for p in profiles_dir().glob('*.yaml')))}) "
            "or a path to a profile yaml"
        ),
        read_only=True,
    )
    name = node.declare_parameter("robot_profile", DEFAULT_PROFILE, descriptor).value
    profile = load_profile(name)
    node.get_logger().info(f"robot profile: {profile.name}")
    return profile


def parse_profile(data: dict[str, Any], source: str = "<dict>") -> RobotProfile:
    """Validate a profile mapping and build a RobotProfile.

    Raises:
        ValueError: On a missing or unknown key, an arm without joints, or a name that is
            not a valid ROS name.
    """
    missing = {"name", "base_frame", "arms"} - set(data)
    if missing:
        raise ValueError(f"{source}: missing keys {sorted(missing)}")
    arms_data = data["arms"]
    if not isinstance(arms_data, dict) or not arms_data:
        raise ValueError(f"{source}: 'arms' must map each side to its names")
    arms = {}
    for side, arm in arms_data.items():
        keys = set(arm or {})
        if keys != _ARM_KEYS:
            raise ValueError(
                f"{source}: arm {side!r} missing {sorted(_ARM_KEYS - keys)}, "
                f"unknown {sorted(keys - _ARM_KEYS)}"
            )
        if not isinstance(arm["joints"], list) or not arm["joints"]:
            raise ValueError(f"{source}: arm {side!r} has no joints list")
        joints = tuple(_checked(j, source) for j in arm["joints"])
        names = {key: _checked(arm[key], source) for key in _ARM_KEYS - {"joints"}}
        # The prefix is what every joint name shares up to its last "_", e.g. "follower_r_".
        shared = os.path.commonprefix([*joints, names["gripper_joint"]])
        arms[str(side)] = ArmProfile(
            joints=joints, joint_prefix=shared[: shared.rfind("_") + 1], **names
        )
    cameras = data.get("cameras") or {}
    if not isinstance(cameras, dict):
        raise ValueError(f"{source}: 'cameras' must map camera names to frames")
    return RobotProfile(
        name=_checked(data["name"], source),
        base_frame=_checked(data["base_frame"], source),
        arms=arms,
        cameras={str(k): _checked(v, source) for k, v in cameras.items()},
    )


def _checked(name: Any, source: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{source}: {name!r} is not a valid ROS name")
    return name
