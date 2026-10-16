"""Full joint names behind a policy's state and command vectors, resolved from a profile.

The inference config names arms and robot-neutral joint ids; the robot profile turns
them into the names on ``/joint_states``. Resolving once at startup means a joint id the
profile lacks fails there, not inside the control loop.

Kept free of ROS and torch imports so it can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_ARM_MAPPING = {"l": "left", "r": "right"}


def state_joint_names(profile: Any, joint_names_config: Mapping[str, Any]) -> list[str]:
    """Joint names in ``observation.state`` order.

    Arms come in sorted ``arm_mapping`` key order (``l`` before ``r``: left, then right),
    whatever order the mapping lists them in; each arm's joints follow
    ``model_joint_order``. Trained policies depend on this exact layout.

    Args:
        profile: An ``anvil_robot_profiles.RobotProfile``.
        joint_names_config: The inference config's ``joint_names`` section.
    """
    arm_mapping = joint_names_config.get("arm_mapping", DEFAULT_ARM_MAPPING)
    return profile.joint_names(
        [arm_mapping[key] for key in sorted(arm_mapping)],
        joint_names_config.get("model_joint_order", []),
    )


def command_joint_names(
    profile: Any, arm_names: list[str], controller_joint_order: list[str]
) -> dict[str, list[str]]:
    """Each arm's joint names in ``controller_joint_order``, the order commands are sent in.

    Args:
        profile: An ``anvil_robot_profiles.RobotProfile``.
        arm_names: Arms the inference config commands.
        controller_joint_order: Joint ids in the arm controller's order.
    """
    return {arm: profile.joint_names([arm], controller_joint_order) for arm in arm_names}
