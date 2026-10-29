"""Robot and sim run the same controllers.

Every robot launch publishes a control descriptor, every descriptor is published by a
robot launch and has an fm-sim catalog entry that names it, and every catalog entry has
a descriptor. Reads the source tree only (yaml, and launch files parsed with ``ast``, no
ROS), so plain CI catches a variant added on one side.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PACKAGE = PACKAGE_DIR.name
LAUNCH_DIR = PACKAGE_DIR / "launch"
CONTROL_DIR = PACKAGE_DIR / "control"
CATALOG_DIR = PACKAGE_DIR / "catalog" / "controllers"
MARKER_DIR = PACKAGE_DIR / "resource" / "fm_sim.controllers"
# Launch files that start the controller on the robot. A new launch file must be
# filed here or as SIM_LAUNCH, so a robot variant cannot appear unchecked.
ROBOT_LAUNCHES = ("classical.launch.py",)
SIM_LAUNCH = "sim.launch.py"
VARIANTS = sorted(p.stem for p in CONTROL_DIR.glob("*.yaml"))
ENTRIES = sorted(p.stem for p in CATALOG_DIR.glob("*.yaml"))


def _string_literals(path: Path) -> set[str]:
    """String constants in a launch file's code: comments and docstrings do not count."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def _published(launch_file: str) -> list[str]:
    literals = _string_literals(LAUNCH_DIR / launch_file)
    return [variant for variant in VARIANTS if f"{variant}.yaml" in literals]


def test_every_launch_file_is_filed_as_robot_or_sim():
    launches = sorted(p.name for p in LAUNCH_DIR.glob("*.launch.py"))
    assert launches == sorted((*ROBOT_LAUNCHES, SIM_LAUNCH))


@pytest.mark.parametrize("robot_launch", ROBOT_LAUNCHES)
def test_robot_launch_publishes_a_control_descriptor(robot_launch):
    assert _published(robot_launch), f"{robot_launch} publishes no control/*.yaml descriptor"


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_is_published_by_a_robot_launch(variant):
    launches = [name for name in ROBOT_LAUNCHES if variant in _published(name)]
    assert launches, f"control/{variant}.yaml is published by no robot launch"


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_has_a_sim_entry_drawn_by_its_descriptor(variant):
    descriptor = yaml.safe_load((CONTROL_DIR / f"{variant}.yaml").read_text())
    assert descriptor.get("name") == variant, f"control/{variant}.yaml is not named {variant}"
    entry_path = CATALOG_DIR / f"{variant}.yaml"
    assert entry_path.is_file(), f"{variant} runs on the robot but has no sim catalog entry"
    entry = yaml.safe_load(entry_path.read_text())
    assert entry.get("name") == variant, f"{entry_path.name} is not named {variant}"
    assert entry.get("control") == {
        "package": PACKAGE,
        "path": f"control/{variant}.yaml",
    }, f"{entry_path.name} does not name control/{variant}.yaml as its control"
    marker = MARKER_DIR / variant
    assert marker.is_file(), f"no fm_sim.controllers marker for {variant}"
    assert marker.read_text().strip() == f"{PACKAGE}/catalog/controllers/{variant}.yaml"


def test_every_sim_entry_is_a_robot_variant():
    assert ENTRIES == VARIANTS, "each catalog entry needs a control/<name>.yaml and the reverse"
