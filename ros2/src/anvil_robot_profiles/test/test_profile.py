"""Tests for loading and validating robot profiles."""

from __future__ import annotations

import sys
import types

import pytest
import yaml
from anvil_robot_profiles import (
    DEFAULT_PROFILE,
    DRIVEN_ARM,
    declare_robot_profile,
    load_profile,
    parse_profile,
    profiles_dir,
)

SHIPPED = ("loader_openarm_v2", "fm_openarm_v2")
JOINT_IDS = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]


def test_every_shipped_profile_is_tested() -> None:
    assert sorted(p.stem for p in profiles_dir().glob("*.yaml")) == sorted(SHIPPED)


def test_default_is_the_loader() -> None:
    assert DEFAULT_PROFILE == "loader_openarm_v2"
    assert load_profile().name == DEFAULT_PROFILE


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_profile_loads_with_both_arms(name: str) -> None:
    profile = load_profile(name)
    assert profile.name == name
    assert set(profile.arms) == {"left", "right"}
    for arm in profile.arms.values():
        assert len(arm.joints) == 7
        assert [arm.joint_name(j) for j in JOINT_IDS] == list(arm.command_joints)


def test_loader_profile_keeps_the_names_the_nodes_hardcoded() -> None:
    # These were constants in classical_control and lerobot_control before profiles.
    profile = load_profile("loader_openarm_v2")
    right = profile.arms["right"]
    assert profile.base_frame == "follower_body_link0"
    assert right.command_joints == (
        *(f"follower_r_joint{i}" for i in range(1, 8)),
        "follower_r_finger_joint1",
    )
    assert right.command_topic == "/follower_r_forward_position_controller/commands"
    assert profile.arms["left"].command_topic == (
        "/follower_l_forward_position_controller/commands"
    )
    assert right.ee_frame == "follower_r_hand_tcp"


def test_fm_profile_uses_openarm_names() -> None:
    right = load_profile("fm_openarm_v2").arms["right"]
    assert right.joint_name("joint1") == "openarm_right_joint1"
    assert right.gripper_joint == "openarm_right_finger_joint1"
    assert right.command_topic == "/openarm_right_forward_position_controller/commands"
    assert right.trajectory_controller == "openarm_right_arm_controller"


@pytest.mark.parametrize(
    ("name", "chest", "wrist_right"),
    [
        ("loader_openarm_v2", "cam_chest_optical", "cam_wrist_r_optical"),
        ("fm_openarm_v2", "openarm_cam_chest_optical", "openarm_right_cam_wrist_optical"),
    ],
)
def test_camera_frames(name: str, chest: str, wrist_right: str) -> None:
    assert load_profile(name).cameras == {"chest": chest, "wrist_right": wrist_right}


def test_joint_name_rejects_an_unknown_id() -> None:
    with pytest.raises(KeyError, match="joint8"):
        load_profile().arms["right"].joint_name("joint8")


def test_load_by_path(tmp_path) -> None:
    data = yaml.safe_load((profiles_dir() / "fm_openarm_v2.yaml").read_text())
    data["name"] = "custom"
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(data))
    assert load_profile(str(path)).name == "custom"


def test_unknown_profile_lists_the_shipped_ones() -> None:
    with pytest.raises(FileNotFoundError, match="fm_openarm_v2"):
        load_profile("no_such_robot")


def _minimal() -> dict:
    arm = {
        "joints": ["a_joint1"],
        "gripper_joint": "a_finger_joint1",
        "arm_controller": "c",
        "trajectory_controller": "t",
        "ee_frame": "e",
        "hand_frame": "h",
    }
    return {"name": "x", "base_frame": "b", "arms": {"right": arm}}


def test_cameras_are_optional() -> None:
    assert parse_profile(_minimal()).cameras == {}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.pop("base_frame"), "base_frame"),
        (lambda d: d.update(arms={}), "arms"),
        (lambda d: d["arms"]["right"].pop("ee_frame"), "ee_frame"),
        (lambda d: d["arms"]["right"].update(extra=1), "extra"),
        (lambda d: d["arms"]["right"].update(joints=[]), "no joints"),
        (lambda d: d["arms"]["right"].update(joints="a_joint1"), "no joints list"),
        (lambda d: d["arms"]["right"].update(arm_controller="ctrl/../x"), "valid ROS name"),
        (lambda d: d["arms"]["right"].update(joints=["a joint1"]), "valid ROS name"),
        (lambda d: d.update(base_frame="1base"), "valid ROS name"),
        (lambda d: d.update(cameras={"chest": "cam-chest"}), "valid ROS name"),
        (lambda d: d.update(cameras=["cam_chest_optical"]), "'cameras' must map"),
    ],
)
def test_parse_rejects_a_bad_schema(mutate, match: str) -> None:
    data = _minimal()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        parse_profile(data)


def test_joint_names_keep_the_lerobot_state_order() -> None:
    # lerobot_control built "follower_{l,r}_{id}" for sorted arm keys l, r: left, then right.
    model_order = ["finger_joint1", *(f"joint{i}" for i in range(1, 8))]
    names = load_profile("loader_openarm_v2").joint_names(["left", "right"], model_order)
    assert names == [f"follower_{k}_{j}" for k in ("l", "r") for j in model_order]


@pytest.mark.parametrize("name", SHIPPED)
def test_joint_names_follow_the_argument_order(name: str) -> None:
    profile = load_profile(name)
    names = profile.joint_names(["right", "left"], ["joint2", "finger_joint1"])
    right, left = profile.arms["right"], profile.arms["left"]
    assert names == [right.joints[1], right.gripper_joint, left.joints[1], left.gripper_joint]


def test_joint_names_reject_an_unknown_arm() -> None:
    with pytest.raises(KeyError):
        load_profile().joint_names(["middle"], ["joint1"])


@pytest.mark.parametrize("name", ["../fm_openarm_v2", "a/b", "a\\b", "..", ""])
def test_bare_name_cannot_leave_the_profiles_dir(name: str) -> None:
    with pytest.raises(ValueError, match="neither a profile name"):
        load_profile(name)


def test_driven_arm_is_in_every_shipped_profile() -> None:
    for name in SHIPPED:
        assert DRIVEN_ARM in load_profile(name).arms


class _FakeNode:
    """Records declare_parameter calls and answers with the declared default."""

    def __init__(self) -> None:
        self.declared = []
        self.logger = types.SimpleNamespace(info=lambda _: None)

    def declare_parameter(self, name, default, descriptor):
        self.declared.append((name, default, descriptor))
        return types.SimpleNamespace(value=default)

    def get_logger(self):
        return self.logger


def test_declare_robot_profile_declares_a_read_only_parameter(monkeypatch) -> None:
    try:
        import rcl_interfaces.msg  # noqa: F401
    except ImportError:
        # Outside ROS: a descriptor stand-in that keeps what it was given.
        msg = types.ModuleType("rcl_interfaces.msg")
        msg.ParameterDescriptor = lambda **fields: types.SimpleNamespace(**fields)
        monkeypatch.setitem(sys.modules, "rcl_interfaces", types.ModuleType("rcl_interfaces"))
        monkeypatch.setitem(sys.modules, "rcl_interfaces.msg", msg)
    node = _FakeNode()

    profile = declare_robot_profile(node)

    [(name, default, descriptor)] = node.declared
    assert (name, default) == ("robot_profile", "loader_openarm_v2")
    assert descriptor.read_only is True
    assert "fm_openarm_v2" in descriptor.description
    assert profile == load_profile("loader_openarm_v2")
