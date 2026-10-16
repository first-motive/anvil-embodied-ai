"""Start one LeRobot policy inside an fm-sim run.

fm-sim's ``run.launch.py`` includes a controller's launch with exactly three arguments,
so each policy has its own thin wrapper (``sim_act.launch.py``, ``sim_smolvla.launch.py``,
``sim_pi0.launch.py``) that includes this file and adds ``policy``:

    policy         act | smolvla | pi0
    use_sim        must be 'true': this launch is sim-only and refuses 'false' (the
                   default), so it can never start inference_node against hardware
    robot_profile  required; the robot entry's profile, e.g. fm_openarm_v2
    scenario       the scenario name; validated, otherwise unused

The run contract carries no checkpoint, so the sim host supplies one. The
``LEROBOT_SIM_CHECKPOINTS`` environment variable names a directory holding one
subdirectory per policy::

    $LEROBOT_SIM_CHECKPOINTS/<policy>/
        inference.yaml          inference config: cameras, joints, arms, task prompt
        config.json ...         the LeRobot checkpoint, here or under pretrained_model/

The inference config sits with the checkpoint because its camera and joint layout
must match what that checkpoint was trained on. The checkpoint's ``config.json``
``type`` must equal ``policy``. Both files must resolve (symlinks followed) inside
``$LEROBOT_SIM_CHECKPOINTS/<policy>``, and ``config.json`` must be at most 1 MiB.
Anything missing or out of bounds fails here, before a node starts.
"""

import json
import os
import re
from pathlib import Path

from anvil_robot_profiles import load_profile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: Catalog policy name -> the LeRobot ``type`` its checkpoint must declare.
POLICIES = {"act": "act", "smolvla": "smolvla", "pi0": "pi0"}
CHECKPOINTS_ENV = "LEROBOT_SIM_CHECKPOINTS"
INFERENCE_CONFIG = "inference.yaml"
# A LeRobot config.json is a few KiB; anything near this is not one.
MAX_MODEL_CONFIG_BYTES = 1024 * 1024
# Same rule fm-sim applies to every name that becomes a path, topic or frame.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
USE_SIM_VALUES = ("true", "false")


def _checked_name(argument, value):
    # The raw value is not echoed: it is caller input and may be anything.
    if not NAME.fullmatch(value):
        raise ValueError(f"{argument} must be a name matching {NAME.pattern}")
    return value


def _contained(checkpoint, path):
    """Return ``path`` resolved, refusing one that leaves the resolved ``checkpoint``."""
    resolved = path.resolve()
    if not resolved.is_relative_to(checkpoint):
        raise PermissionError(f"{path.name} in sim checkpoint {checkpoint} resolves outside it")
    return resolved


def resolve_checkpoint(policy, root):
    """Return ``(model_path, config_file)`` for ``policy`` under checkpoint ``root``.

    Raises:
        ValueError: On an unknown policy, an unset root, or a checkpoint whose
            ``config.json`` declares another policy type.
        FileNotFoundError: When the checkpoint, its ``config.json`` or its
            ``inference.yaml`` is missing.
        PermissionError: When either file resolves outside the checkpoint directory.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of: {', '.join(POLICIES)}")
    if not root:
        raise ValueError(
            f"no sim checkpoint: set {CHECKPOINTS_ENV} to a directory holding {policy}/"
        )
    checkpoint = (Path(root) / policy).resolve()
    config_file = _contained(checkpoint, checkpoint / INFERENCE_CONFIG)
    if not config_file.is_file():
        raise FileNotFoundError(f"no {INFERENCE_CONFIG} in sim checkpoint {checkpoint}")
    candidates = (checkpoint / "config.json", checkpoint / "pretrained_model" / "config.json")
    model_config = next((path for path in candidates if path.is_file()), None)
    if model_config is None:
        raise FileNotFoundError(f"no config.json in sim checkpoint {checkpoint}")
    model_config = _contained(checkpoint, model_config)
    if model_config.stat().st_size > MAX_MODEL_CONFIG_BYTES:
        raise ValueError(f"{model_config} is over {MAX_MODEL_CONFIG_BYTES} bytes")
    try:
        model = json.loads(model_config.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{model_config} is not valid JSON: {error}") from error
    declared = model.get("type") if isinstance(model, dict) else None
    if declared != POLICIES[policy]:
        raise ValueError(
            f"{model_config} declares type {declared!r}, expected {POLICIES[policy]!r}"
        )
    return str(checkpoint), str(config_file)


def _launch_setup(context):
    policy = LaunchConfiguration("policy").perform(context)
    use_sim = LaunchConfiguration("use_sim").perform(context)
    if use_sim not in USE_SIM_VALUES:
        raise ValueError(f"use_sim must be one of: {', '.join(USE_SIM_VALUES)}")
    if use_sim != "true":
        raise ValueError("vla_sim runs only inside fm-sim: use_sim must be 'true'")
    robot_profile = _checked_name(
        "robot_profile", LaunchConfiguration("robot_profile").perform(context)
    )
    _checked_name("scenario", LaunchConfiguration("scenario").perform(context))
    # Fails here, naming the shipped profiles, rather than in the node after launch.
    load_profile(robot_profile)
    model_path, config_file = resolve_checkpoint(policy, os.environ.get(CHECKPOINTS_ENV, ""))
    return [
        Node(
            package="lerobot_control",
            executable="inference_node",
            name="lerobot_inference",
            output="screen",
            parameters=[
                {
                    "model_path": model_path,
                    "config_file": config_file,
                    "robot_profile": robot_profile,
                    "use_sim_time": True,
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("policy", description=f"LeRobot policy: {' | '.join(POLICIES)}."),
            DeclareLaunchArgument(
                "use_sim",
                default_value="false",
                description="Must be 'true'; 'false' is refused (sim-only launch).",
            ),
            DeclareLaunchArgument(
                "robot_profile",
                description="Robot profile naming the joints and controllers.",
            ),
            DeclareLaunchArgument(
                "scenario",
                default_value="empty",
                description="Scenario of the run; validated, otherwise unused.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
