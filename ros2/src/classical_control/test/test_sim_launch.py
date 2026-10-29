"""The sim include launch, loaded from the installed share and set up as fm-sim would.

fm-sim's run.launch.py includes it with ``use_sim``, ``robot_profile`` and ``scenario``;
``backend`` reaches it from the run's own launch configurations.
"""

import os
from pathlib import Path

import pytest
import yaml

launch = pytest.importorskip("launch")
launch_ros_actions = pytest.importorskip("launch_ros.actions")
packages = pytest.importorskip("ament_index_python.packages")

PROFILE = "fm_openarm_v2"


@pytest.fixture(scope="module")
def sim_launch(launch_tools):
    return launch_tools.load("sim.launch.py")


@pytest.fixture
def follower_installed(sim_launch, monkeypatch):
    # fm_sim_ee_follower ships with fm-sim; stand in for it unless it is installed.
    real = sim_launch.get_package_share_directory

    def lookup(package):
        if package == sim_launch.FOLLOWER_PACKAGE:
            try:
                return real(package)
            except packages.PackageNotFoundError:
                return "/nonexistent/fm_sim_ee_follower"
        return real(package)

    monkeypatch.setattr(sim_launch, "get_package_share_directory", lookup)


def _declared(sim_launch):
    return {
        action.name: action
        for action in sim_launch.generate_launch_description().entities
        if isinstance(action, launch.actions.DeclareLaunchArgument)
    }


def _setup(sim_launch, **arguments):
    """Run the setup function with the launch file's defaults plus ``arguments``."""
    context = launch.LaunchContext()
    for name, declared in _declared(sim_launch).items():
        if declared.default_value is not None:
            context.launch_configurations[name] = declared.default_value[0].text
    context.launch_configurations.update(arguments)
    return context, sim_launch._launch_setup(context)


def _command(process, context):
    return [launch.utilities.perform_substitutions(context, part) for part in process.cmd]


def _processes(actions):
    # launch_ros's Node subclasses ExecuteProcess; only the plain CLI processes count here.
    return [
        action
        for action in actions
        if isinstance(action, launch.actions.ExecuteProcess)
        and not isinstance(action, launch_ros_actions.Node)
    ]


def test_declares_the_run_arguments_and_the_inherited_backend(sim_launch):
    assert set(_declared(sim_launch)) == {"use_sim", "robot_profile", "scenario", "backend"}
    assert _declared(sim_launch)["robot_profile"].default_value is None
    # Declared with a default, or fm-sim's include (which passes no backend) would fail.
    assert _declared(sim_launch)["backend"].default_value[0].text == "mujoco"


@pytest.mark.usefixtures("follower_installed")
def test_sim_run_starts_task_node_on_the_pose_interface_and_the_follower(launch_tools, sim_launch):
    context, actions = _setup(sim_launch, use_sim="true", robot_profile=PROFILE)
    nodes = launch_tools.nodes(actions)
    task, follower = nodes["task_node"], nodes["ee_follower"]
    assert (task.node_package, task.node_executable) == ("classical_control", "task_node")
    parameters, files = launch_tools.parameters(task, context)
    assert os.path.basename(files[0]) == "task.yaml"
    assert parameters["ee_interface"] == "pose"
    assert parameters["robot_profile"] == PROFILE
    assert parameters["use_sim_time"] is True
    mined = parameters["mined_params_file"]
    assert os.path.isfile(mined)
    assert {"grasp_z", "place_z", "grasp_orientation_xyzw"} <= set(
        yaml.safe_load(Path(mined).read_text())
    )

    assert (follower.node_package, follower.node_executable) == (
        "fm_sim_ee_follower",
        "ee_follower",
    )
    assert launch_tools.parameters(follower, context)[0] == {
        "robot_profile": PROFILE,
        "use_sim_time": True,
    }


@pytest.mark.usefixtures("follower_installed")
def test_use_sim_false_leaves_every_clocked_node_on_wall_time(launch_tools, sim_launch):
    context, actions = _setup(sim_launch, robot_profile=PROFILE)
    nodes = launch_tools.nodes(actions)
    del nodes["control_descriptor"]  # publishes once; it has no clock to set
    assert {
        launch_tools.parameters(node, context)[0]["use_sim_time"] for node in nodes.values()
    } == {False}


@pytest.mark.usefixtures("follower_installed")
@pytest.mark.parametrize("value", ["", "True", "yes", "1"])
def test_refuses_a_use_sim_other_than_true_or_false(sim_launch, value):
    with pytest.raises(ValueError, match="use_sim must be one of: true, false$"):
        _setup(sim_launch, use_sim=value, robot_profile=PROFILE)


@pytest.mark.usefixtures("follower_installed")
@pytest.mark.parametrize("argument", ["robot_profile", "scenario", "backend"])
@pytest.mark.parametrize("value", ["", "../fm_openarm_v2", "a/b", "9lives", "x y"])
def test_refuses_a_malformed_name_without_echoing_it(sim_launch, argument, value):
    arguments = {"robot_profile": PROFILE, argument: value}
    with pytest.raises(ValueError, match=f"^{argument} must be a name") as raised:
        _setup(sim_launch, **arguments)
    assert repr(value) not in str(raised.value)


@pytest.mark.usefixtures("follower_installed")
def test_refuses_an_unknown_robot_profile(sim_launch):
    with pytest.raises(FileNotFoundError, match="robot profile 'no_such_robot' not found"):
        _setup(sim_launch, robot_profile="no_such_robot")


def test_fails_before_starting_anything_without_the_follower(sim_launch, monkeypatch):
    def missing(package):
        if package == sim_launch.FOLLOWER_PACKAGE:
            raise packages.PackageNotFoundError(package)
        return packages.get_package_share_directory(package)

    monkeypatch.setattr(sim_launch, "get_package_share_directory", missing)
    with pytest.raises(RuntimeError, match="fm_sim_ee_follower is not installed"):
        _setup(sim_launch, use_sim="true", robot_profile=PROFILE)


@pytest.mark.usefixtures("follower_installed")
@pytest.mark.parametrize("backend", ["mujoco", "gazebo"])
def test_perception_runs_on_the_sim_camera_and_the_backend_background(
    launch_tools, sim_launch, backend
):
    context, actions = _setup(sim_launch, use_sim="true", robot_profile=PROFILE, backend=backend)
    perception = launch_tools.nodes(actions)["perception_node"]
    parameters, files = launch_tools.parameters(perception, context)
    assert os.path.basename(files[0]) == "perception.yaml"
    assert parameters["background_file"].endswith(f"backgrounds/{backend}_openarm_chest.png")
    assert os.path.isfile(parameters["background_file"])
    camera = yaml.safe_load(Path(parameters["camera_config_file"]).read_text())
    # The sim description already mounts cam_chest_optical on chest_cam.
    assert camera["extrinsic"]["parent_frame"] == "chest_cam"
    assert parameters["use_sim_time"] is True


@pytest.mark.usefixtures("follower_installed")
def test_refuses_a_backend_with_no_rendered_background(sim_launch):
    with pytest.raises(ValueError, match="no chest background for backend isaac"):
        _setup(sim_launch, robot_profile=PROFILE, backend="isaac")


@pytest.mark.usefixtures("follower_installed")
def test_publishes_the_installed_descriptor(launch_tools, sim_launch):
    context, actions = _setup(sim_launch, use_sim="true", robot_profile=PROFILE)
    descriptor = launch_tools.nodes(actions)["control_descriptor"]
    control_file = launch_tools.parameters(descriptor, context)[0]["control_file"]
    assert yaml.safe_load(Path(control_file).read_text())["name"] == "anvil_classical"


@pytest.mark.usefixtures("follower_installed")
def test_can_on_table_sends_one_goal_to_the_paper_after_the_first_can_pose(sim_launch):
    context, actions = _setup(
        sim_launch, use_sim="true", robot_profile=PROFILE, scenario="can_on_table"
    )
    (wait,) = _processes(actions)
    wait_cmd = _command(wait, context)
    assert wait_cmd[-3:] == ["--once", "/classical/can_pose", "geometry_msgs/msg/PoseStamped"]
    (handler,) = (a for a in actions if isinstance(a, launch.actions.RegisterEventHandler))
    exited = launch.events.process.ProcessExited(
        action=wait, name="wait_for_can", cmd=wait_cmd, cwd=None, env=None, pid=1, returncode=0
    )
    (goal,) = handler.event_handler.handle(exited, context)
    goal_cmd = _command(goal, context)
    assert goal_cmd[2:5] == [
        "send_goal",
        "/classical/pick_place",
        "classical_control_msgs/action/PickPlace",
    ]
    assert goal_cmd[5] == "{place_target_set: true, place_target: {x: 0.4, y: -0.3, z: 0.0}}"
    failed = launch.events.process.ProcessExited(
        action=wait, name="wait_for_can", cmd=wait_cmd, cwd=None, env=None, pid=1, returncode=1
    )
    assert not handler.event_handler.handle(failed, context)


@pytest.mark.usefixtures("follower_installed")
def test_a_scenario_without_paper_sends_no_goal(sim_launch):
    _, actions = _setup(sim_launch, use_sim="true", robot_profile=PROFILE, scenario="empty")
    assert not _processes(actions)
