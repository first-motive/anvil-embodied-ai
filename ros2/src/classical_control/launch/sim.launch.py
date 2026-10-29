"""Start the classical baseline inside an fm-sim run, as the ``anvil_classical`` controller.

fm-sim's ``run.launch.py`` includes this file for the ``anvil_classical`` catalog entry
(``catalog/controllers/anvil_classical.yaml``) and passes three arguments:

    use_sim        'true' | 'false' (default 'false'); sets use_sim_time on every node
    robot_profile  required; the robot entry's profile, e.g. loader_openarm_v2
    scenario       the scenario name; picks the episode goal's place target

``backend`` picks the background. fm-sim does not pass it, but its include does not
scope launch configurations, so the run's own ``backend`` argument reaches this file.
It defaults to ``mujoco``, as run.launch.py does: an included file's argument with no
default must be passed explicitly, which fm-sim's include does not.

What starts:

- ``perception_node`` with ``config/sim/camera_chest.yaml`` (the sim's chest mount) and
  the backend's rendered table_only reference, ``backgrounds/<backend>_openarm_chest.png``
  (``tools/render_background.py``). Isaac has no reference yet, so it is refused.
- ``task_node`` on the ``pose`` interface: it streams ``/commanded_ee_right`` as a
  ``PoseStamped`` in the profile's ``base_frame`` and reads the measured TCP from TF.
  It loads the shipped ``task.yaml`` plus ``config/sim/mined_params.yaml``.
- fm-sim's ``ee_follower`` (package ``fm_sim_ee_follower``), which turns that pose
  into joint commands for the right arm's forward position controller. It ships with
  fm-sim, not here, so it is not a ``package.xml`` dependency; this launch fails
  before starting anything when it is not installed.
- ``control_descriptor``, publishing ``control/anvil_classical.yaml`` latched on
  ``/control/descriptor``.
- For a scenario in ``PLACE_TARGETS``, one ``/classical/pick_place`` goal, sent once the
  first can pose arrives. It places at the scenario's paper: the paper is in the
  rendered reference, so perception never detects it.

task_node refuses the ``pose`` interface without ``use_sim_time``, so ``use_sim:=false``
makes the run fail at task_node startup.
"""

import re

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from anvil_robot_profiles import load_profile
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "classical_control"
FOLLOWER_PACKAGE = "fm_sim_ee_follower"
# Same rule fm-sim applies to every name that becomes a path, topic or frame.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
USE_SIM_VALUES = ("true", "false")
BACKGROUND_BACKENDS = ("mujoco", "gazebo")
# The paper's centre in each scenario that has one (fm-sim catalog/scenarios), world xy.
PLACE_TARGETS = {"can_on_table": (0.40, -0.30)}


def _checked_name(argument, value):
    # The raw value is not echoed: it is caller input and may be anything.
    if not NAME.fullmatch(value):
        raise ValueError(f"{argument} must be a name matching {NAME.pattern}")
    return value


def _episode_goal(place_xy):
    """Wait for the first can pose, then send one pick and place to the paper."""
    wait_for_can = ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "echo",
            "--once",
            "/classical/can_pose",
            "geometry_msgs/msg/PoseStamped",
        ],
        name="wait_for_can",
        output="log",
    )
    x, y = place_xy
    send_goal = ExecuteProcess(
        cmd=[
            "ros2",
            "action",
            "send_goal",
            "/classical/pick_place",
            "classical_control_msgs/action/PickPlace",
            f"{{place_target_set: true, place_target: {{x: {x}, y: {y}, z: 0.0}}}}",
        ],
        name="pick_place_goal",
        output="screen",
    )

    def after_wait(event, _context):
        # A failed wait (shutdown, bad topic) sends nothing rather than a goal that
        # would fail at LOCALISE.
        return [send_goal] if event.returncode == 0 else []

    return [
        wait_for_can,
        RegisterEventHandler(OnProcessExit(target_action=wait_for_can, on_exit=after_wait)),
    ]


def _launch_setup(context):
    use_sim = LaunchConfiguration("use_sim").perform(context)
    if use_sim not in USE_SIM_VALUES:
        raise ValueError(f"use_sim must be one of: {', '.join(USE_SIM_VALUES)}")
    robot_profile = _checked_name(
        "robot_profile", LaunchConfiguration("robot_profile").perform(context)
    )
    scenario = _checked_name("scenario", LaunchConfiguration("scenario").perform(context))
    backend = _checked_name("backend", LaunchConfiguration("backend").perform(context))
    if backend not in BACKGROUND_BACKENDS:
        raise ValueError(
            f"no chest background for backend {backend}; "
            f"rendered for: {', '.join(BACKGROUND_BACKENDS)}"
        )
    # Fails here, naming the shipped profiles, rather than in a node after launch.
    load_profile(robot_profile)
    try:
        get_package_share_directory(FOLLOWER_PACKAGE)
    except PackageNotFoundError as error:
        raise RuntimeError(f"{FOLLOWER_PACKAGE} is not installed; it ships with fm-sim") from error

    in_sim = use_sim == "true"
    share = get_package_share_directory(PACKAGE)
    config_dir = f"{share}/config"
    actions = [
        Node(
            package=PACKAGE,
            executable="perception_node",
            output="screen",
            parameters=[
                f"{config_dir}/perception.yaml",
                {
                    "camera_config_file": f"{config_dir}/sim/camera_chest.yaml",
                    "background_file": f"{share}/backgrounds/{backend}_openarm_chest.png",
                    "use_sim_time": in_sim,
                },
            ],
        ),
        Node(
            package=PACKAGE,
            executable="task_node",
            output="screen",
            parameters=[
                f"{config_dir}/task.yaml",
                {
                    "ee_interface": "pose",
                    "mined_params_file": f"{config_dir}/sim/mined_params.yaml",
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
        Node(
            package=PACKAGE,
            executable="control_descriptor",
            output="screen",
            parameters=[{"control_file": f"{share}/control/anvil_classical.yaml"}],
        ),
    ]
    if scenario in PLACE_TARGETS:
        actions += _episode_goal(PLACE_TARGETS[scenario])
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim",
                default_value="false",
                description="'true' or 'false'; sets use_sim_time on every node.",
            ),
            DeclareLaunchArgument(
                "robot_profile",
                description="Robot profile naming the joints, controllers and frames.",
            ),
            DeclareLaunchArgument(
                "scenario",
                default_value="empty",
                description="Scenario of the run; a scenario with a paper gets one goal.",
            ),
            DeclareLaunchArgument(
                "backend",
                default_value="mujoco",
                description="fm-sim backend, inherited from run.launch.py; picks the background.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
