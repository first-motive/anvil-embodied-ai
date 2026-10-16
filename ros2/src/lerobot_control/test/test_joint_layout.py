"""Tests for the policy state and command joint layouts resolved from a robot profile."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from anvil_robot_profiles import load_profile

# Loaded by path: importing the lerobot_control package pulls in torch and lerobot.
_spec = importlib.util.spec_from_file_location(
    "joint_layout", Path(__file__).parent.parent / "lerobot_control" / "joint_layout.py"
)
joint_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joint_layout)

PREFIXES = {
    "loader_openarm_v2": {"left": "follower_l_", "right": "follower_r_"},
    "fm_openarm_v2": {"left": "openarm_left_", "right": "openarm_right_"},
}


@pytest.mark.parametrize("profile", sorted(PREFIXES))
@pytest.mark.parametrize(
    "arm_mapping", [{"l": "left", "r": "right"}, {"r": "right", "l": "left"}]
)
def test_state_is_left_then_right_whatever_the_mapping_order(
    profile, arm_mapping, joint_names_config
) -> None:
    order = joint_names_config["model_joint_order"]
    config = {**joint_names_config, "arm_mapping": arm_mapping}
    names = joint_layout.state_joint_names(load_profile(profile), config)
    prefix = PREFIXES[profile]
    assert names == [prefix[arm] + j for arm in ("left", "right") for j in order]


def test_loader_state_matches_the_old_hardcoded_names(joint_names_config) -> None:
    # Before profiles: f"follower_{arm_key}_{joint_id}" over sorted arm_mapping keys.
    order = joint_names_config["model_joint_order"]
    keys = sorted(joint_names_config["arm_mapping"])
    names = joint_layout.state_joint_names(load_profile("loader_openarm_v2"), joint_names_config)
    assert names == [f"follower_{k}_{j}" for k in keys for j in order]


def test_state_defaults_to_both_arms() -> None:
    names = joint_layout.state_joint_names(load_profile(), {"model_joint_order": ["joint1"]})
    assert names == ["follower_l_joint1", "follower_r_joint1"]


def test_single_arm_state(joint_names_config) -> None:
    order = joint_names_config["model_joint_order"]
    config = {**joint_names_config, "arm_mapping": {"r": "right"}}
    names = joint_layout.state_joint_names(load_profile("fm_openarm_v2"), config)
    assert names == [f"openarm_right_{j}" for j in order]


@pytest.mark.parametrize("profile", sorted(PREFIXES))
def test_command_names_follow_the_controller_order(profile, joint_names_config) -> None:
    order = joint_names_config["controller_joint_order"]
    names = joint_layout.command_joint_names(load_profile(profile), ["right", "left"], order)
    prefix = PREFIXES[profile]
    assert names == {arm: [prefix[arm] + j for j in order] for arm in prefix}
    assert names["right"] == list(load_profile(profile).arms["right"].command_joints)


def test_unknown_joint_id_fails_at_resolution() -> None:
    with pytest.raises(KeyError, match="joint9"):
        joint_layout.state_joint_names(load_profile(), {"model_joint_order": ["joint9"]})
