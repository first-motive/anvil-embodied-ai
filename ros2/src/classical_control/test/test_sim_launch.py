"""The sim include launch, loaded from the installed share and set up as fm-sim would.

fm-sim's run.launch.py includes it with ``use_sim``, ``robot_profile`` and ``scenario``.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

launch = pytest.importorskip("launch")
launch_ros_actions = pytest.importorskip("launch_ros.actions")
packages = pytest.importorskip("ament_index_python.packages")

PROFILE = "fm_openarm_v2"


@pytest.fixture(scope="module")
def sim_launch():
    try:
        share = packages.get_package_share_directory("classical_control")
    except packages.PackageNotFoundError:
        pytest.skip("classical_control is not installed; run under colcon test")
    path = os.path.join(share, "launch", "sim.launch.py")
    spec = importlib.util.spec_from_file_location("classical_control_sim_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _parameters(node, context):
    # launch_ros has no public getter for a node's parameters: it expands each entry
    # to a yaml file (dicts written out, file entries as given), recorded privately.
    node._perform_substitutions(context)
    expanded = getattr(node, "_Node__expanded_parameter_arguments", None)
    assert expanded is not None, "launch_ros no longer has Node.__expanded_parameter_arguments"
    parameters, files = {}, []
    for path, is_file in expanded:
        assert is_file
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        files.append(path)
        parameters.update(next(iter(loaded.values()))["ros__parameters"])
    return parameters, files


def test_declares_the_three_run_arguments(sim_launch):
    assert set(_declared(sim_launch)) == {"use_sim", "robot_profile", "scenario"}
    assert _declared(sim_launch)["robot_profile"].default_value is None


@pytest.mark.usefixtures("follower_installed")
def test_sim_run_starts_task_node_on_the_pose_interface_and_the_follower(sim_launch):
    context, (task, follower) = _setup(sim_launch, use_sim="true", robot_profile=PROFILE)
    assert (task.node_package, task.node_executable) == ("classical_control", "task_node")
    parameters, files = _parameters(task, context)
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
    assert _parameters(follower, context)[0] == {"robot_profile": PROFILE, "use_sim_time": True}


@pytest.mark.usefixtures("follower_installed")
def test_use_sim_false_leaves_both_nodes_on_wall_time(sim_launch):
    context, nodes = _setup(sim_launch, robot_profile=PROFILE)
    assert [_parameters(node, context)[0]["use_sim_time"] for node in nodes] == [False, False]


@pytest.mark.usefixtures("follower_installed")
@pytest.mark.parametrize("value", ["", "True", "yes", "1"])
def test_refuses_a_use_sim_other_than_true_or_false(sim_launch, value):
    with pytest.raises(ValueError, match="use_sim must be one of: true, false$"):
        _setup(sim_launch, use_sim=value, robot_profile=PROFILE)


@pytest.mark.usefixtures("follower_installed")
@pytest.mark.parametrize("argument", ["robot_profile", "scenario"])
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
