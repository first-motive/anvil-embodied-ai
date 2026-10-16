"""Start the classical baseline inside an fm-sim run, as the ``classical`` controller.

fm-sim's ``run.launch.py`` includes this file for the ``classical`` catalog entry
(``catalog/controllers/classical.yaml``) and passes exactly three arguments:

    use_sim        'true' | 'false' (default 'false'); sets use_sim_time on both nodes
    robot_profile  required; the robot entry's profile, e.g. fm_openarm_v2
    scenario       the scenario name; validated, otherwise unused

Two nodes start:

- ``task_node`` on the ``pose`` interface: it streams ``/commanded_ee_right`` as a
  ``PoseStamped`` in the profile's ``base_frame`` and reads the measured TCP from TF.
  It loads the shipped ``task.yaml`` plus ``config/sim/mined_params.yaml``.
- fm-sim's ``ee_follower`` (package ``fm_sim_ee_follower``), which turns that pose
  into joint commands for the right arm's forward position controller. It ships with
  fm-sim, not here, so it is not a ``package.xml`` dependency; this launch fails
  before starting anything when it is not installed.

No ``perception_node`` starts: its chest-camera calibration and background are the
real cell's. task_node refuses the ``pose`` interface without ``use_sim_time``, so
``use_sim:=false`` makes the run fail at task_node startup.
"""

import re

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from anvil_robot_profiles import load_profile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

FOLLOWER_PACKAGE = "fm_sim_ee_follower"
# Same rule fm-sim applies to every name that becomes a path, topic or frame.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
USE_SIM_VALUES = ("true", "false")


def _checked_name(argument, value):
    # The raw value is not echoed: it is caller input and may be anything.
    if not NAME.fullmatch(value):
        raise ValueError(f"{argument} must be a name matching {NAME.pattern}")
    return value


def _launch_setup(context):
    use_sim = LaunchConfiguration("use_sim").perform(context)
    if use_sim not in USE_SIM_VALUES:
        raise ValueError(f"use_sim must be one of: {', '.join(USE_SIM_VALUES)}")
    robot_profile = _checked_name(
        "robot_profile", LaunchConfiguration("robot_profile").perform(context)
    )
    _checked_name("scenario", LaunchConfiguration("scenario").perform(context))
    # Fails here, naming the shipped profiles, rather than in a node after launch.
    load_profile(robot_profile)
    try:
        get_package_share_directory(FOLLOWER_PACKAGE)
    except PackageNotFoundError as error:
        raise RuntimeError(f"{FOLLOWER_PACKAGE} is not installed; it ships with fm-sim") from error

    in_sim = use_sim == "true"
    config_dir = PathJoinSubstitution([FindPackageShare("classical_control"), "config"])
    return [
        Node(
            package="classical_control",
            executable="task_node",
            output="screen",
            parameters=[
                PathJoinSubstitution([config_dir, "task.yaml"]),
                {
                    "ee_interface": "pose",
                    "mined_params_file": PathJoinSubstitution(
                        [config_dir, "sim", "mined_params.yaml"]
                    ),
                    "robot_profile": robot_profile,
                    "use_sim_time": in_sim,
                },
            ],
        ),
        Node(
            package=FOLLOWER_PACKAGE,
            executable="ee_follower",
            output="screen",
            parameters=[{"robot_profile": robot_profile, "use_sim_time": in_sim}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim",
                default_value="false",
                description="'true' or 'false'; sets use_sim_time on both nodes.",
            ),
            DeclareLaunchArgument(
                "robot_profile",
                description="Robot profile naming the joints, controllers and frames.",
            ),
            DeclareLaunchArgument(
                "scenario",
                default_value="empty",
                description="Scenario of the run; validated, otherwise unused.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
