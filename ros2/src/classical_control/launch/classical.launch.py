"""Starts the classical baseline: perception_node and the pick-place task node.

`config_dir` defaults to the package's installed config. The container points it
at the repo's config directory instead, mounted read-only, so a calibration edit
takes effect on restart without rebuilding the image. `data_dir` holds what the
robot produces at run time: the captured background and the mined params.
`robot_profile` names the joints and controllers (see anvil_robot_profiles).
"""

from anvil_robot_profiles import DEFAULT_PROFILE
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_dir = LaunchConfiguration("config_dir")
    data_dir = LaunchConfiguration("data_dir")
    robot_profile = LaunchConfiguration("robot_profile")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_dir",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("classical_control"), "config"]
                ),
            ),
            DeclareLaunchArgument("data_dir", default_value="/data"),
            DeclareLaunchArgument(
                "robot_profile",
                default_value=DEFAULT_PROFILE,
                description="Robot profile naming joints, controllers and frames: "
                "a name shipped in anvil_robot_profiles or a path to a profile yaml",
            ),
            Node(
                package="classical_control",
                executable="perception_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution([config_dir, "perception.yaml"]),
                    {
                        "camera_config_file": PathJoinSubstitution(
                            [config_dir, "camera_chest.yaml"]
                        ),
                        "background_file": PathJoinSubstitution([data_dir, "chest_background.png"]),
                    },
                ],
            ),
            Node(
                package="classical_control",
                executable="task_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution([config_dir, "task.yaml"]),
                    {
                        "mined_params_file": PathJoinSubstitution([data_dir, "mined_params.yaml"]),
                        "robot_profile": robot_profile,
                    },
                ],
            ),
        ]
    )
