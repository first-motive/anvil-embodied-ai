"""Shared helpers for the launch-file tests, as the ``launch_tools`` fixture.

The launch files are loaded from the installed share, so these tests skip under plain
pytest and run under colcon test.
"""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import yaml


def _load(file_name):
    """Import an installed classical_control launch file as a module."""
    packages = pytest.importorskip("ament_index_python.packages")
    try:
        share = packages.get_package_share_directory("classical_control")
    except packages.PackageNotFoundError:
        pytest.skip("classical_control is not installed; run under colcon test")
    path = os.path.join(share, "launch", file_name)
    module_name = "classical_control_" + file_name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nodes(actions):
    """The launch_ros Nodes among ``actions``, by executable."""
    from launch_ros.actions import Node

    return {action.node_executable: action for action in actions if isinstance(action, Node)}


def _parameters(node, context):
    """A node's parameters merged in order, and the parameter files they came from."""
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


@pytest.fixture(scope="session")
def launch_tools():
    pytest.importorskip("launch_ros.actions")
    return SimpleNamespace(load=_load, nodes=_nodes, parameters=_parameters)
