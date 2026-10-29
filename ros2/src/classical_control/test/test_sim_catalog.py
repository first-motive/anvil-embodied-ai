"""The fm-sim catalog entry this package installs, read back the way fm-sim reads it.

Runs against an install: the marker and yaml are looked up through the ament index.
Each entry goes through fm-sim's own ``fm_sim_bringup.catalog.load``, which checks the
marker path, the schema and the self-name; with no fm-sim in the workspace those tests
skip, since the schema ships only with fm-sim and is not vendored here.
"""

import os

import pytest

PACKAGE = "classical_control"
RESOURCE_TYPE = "fm_sim.controllers"
# Entry name -> whether run.launch.py may start it.
EXPECTED = {"anvil_classical": False}


@pytest.fixture(scope="module")
def index():
    resources = pytest.importorskip("ament_index_python.resources")
    packages = pytest.importorskip("ament_index_python.packages")
    try:
        packages.get_package_share_directory(PACKAGE)
    except packages.PackageNotFoundError:
        pytest.skip(f"{PACKAGE} is not installed; run under colcon test")
    return resources


@pytest.fixture(scope="module")
def fm_sim_catalog():
    try:
        from fm_sim_bringup import catalog
    except ImportError:
        pytest.skip("fm_sim_bringup is not installed: no catalog schema to validate against")
    return catalog


def _ours(index):
    return sorted(
        name
        for name in index.get_resources(RESOURCE_TYPE)
        if index.get_resource(RESOURCE_TYPE, name)[0].strip().startswith(PACKAGE + "/")
    )


def test_installs_exactly_the_expected_markers(index):
    assert _ours(index) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_marker_points_at_an_installed_entry_yaml(index, name):
    marker, prefix = index.get_resource(RESOURCE_TYPE, name)
    assert marker.strip() == f"{PACKAGE}/catalog/controllers/{name}.yaml"
    assert os.path.isfile(os.path.join(prefix, "share", marker.strip()))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fm_sim_loads_and_validates_the_entry(index, fm_sim_catalog, name):
    entry = fm_sim_catalog.load("controllers", name)
    assert entry["package"] == PACKAGE
    assert entry["available"] is EXPECTED[name]


def test_anvil_classical_launch_is_installed_and_needs_the_forward_position_role(
    index, fm_sim_catalog
):
    entry = fm_sim_catalog.load("controllers", "anvil_classical")
    # The EE follower streams the profile's arm_controller, which the openarm robot
    # entry files under arm_forward_position.
    assert entry["activate"] == ["arm_forward_position"]
    assert os.path.isfile(fm_sim_catalog.share_file(entry["launch"]))


def test_fm_sim_refuses_the_unavailable_entry(index, fm_sim_catalog):
    with pytest.raises(fm_sim_catalog.CatalogError, match="not available"):
        fm_sim_catalog.require_available("controllers", "anvil_classical")


def test_anvil_classical_is_in_the_anvil_family_and_drawn_by_its_descriptor(index, fm_sim_catalog):
    entry = fm_sim_catalog.load("controllers", "anvil_classical")
    assert entry["family"] == "anvil"
    assert entry["control"] == {"package": PACKAGE, "path": "control/anvil_classical.yaml"}
    # The loader validated the file against the schema and attached it.
    assert entry["control_descriptor"]["name"] == "anvil_classical"
