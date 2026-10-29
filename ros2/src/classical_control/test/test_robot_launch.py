"""The robot launch, loaded from the installed share: it publishes the control descriptor."""

import pytest
import yaml

launch = pytest.importorskip("launch")


def test_publishes_the_anvil_classical_descriptor(launch_tools):
    robot_launch = launch_tools.load("classical.launch.py")
    nodes = launch_tools.nodes(robot_launch.generate_launch_description().entities)
    assert set(nodes) == {"perception_node", "task_node", "control_descriptor"}

    parameters, _ = launch_tools.parameters(nodes["control_descriptor"], launch.LaunchContext())
    with open(parameters["control_file"]) as handle:
        assert yaml.safe_load(handle)["name"] == "anvil_classical"
