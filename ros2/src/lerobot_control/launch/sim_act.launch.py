"""fm-sim include for the ``act`` catalog entry: vla_sim.launch.py with ``policy:=act``.

fm-sim passes ``use_sim``, ``robot_profile`` and ``scenario``. An include must name
every required argument of the file it includes, so all three are declared here and
forwarded alongside ``policy``.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    vla_sim = get_package_share_directory("lerobot_control") + "/launch/vla_sim.launch.py"
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim", default_value="false"),
            DeclareLaunchArgument("robot_profile"),
            DeclareLaunchArgument("scenario", default_value="empty"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(vla_sim),
                launch_arguments={
                    "policy": "act",
                    "use_sim": LaunchConfiguration("use_sim"),
                    "robot_profile": LaunchConfiguration("robot_profile"),
                    "scenario": LaunchConfiguration("scenario"),
                }.items(),
            ),
        ]
    )
