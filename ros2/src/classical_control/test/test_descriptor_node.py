"""The shipped control descriptor and the JSON the descriptor node publishes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from classical_control.task_machine import Phase

CONTROL = Path(__file__).resolve().parent.parent / "control" / "anvil_classical.yaml"


@pytest.fixture(scope="module")
def descriptor():
    return yaml.safe_load(CONTROL.read_text())


def test_names_the_sim_entry(descriptor):
    assert descriptor["name"] == "anvil_classical"


def test_state_machine_lists_the_pick_place_feedback_phases_in_order(descriptor):
    (machine,) = (block for block in descriptor["blocks"] if block["kind"] == "state_machine")
    assert machine["states"] == [phase.name for phase in Phase]
    assert machine["source"] == {
        "topic": "/classical/pick_place/_action/feedback",
        "type": "classical_control_msgs/action/PickPlace",
        "field": "feedback.phase",
    }


def test_every_edge_names_a_block(descriptor):
    ids = {block["id"] for block in descriptor["blocks"]}
    assert {end for edge in descriptor["edges"] for end in (edge["from"], edge["to"])} <= ids


def test_published_json_is_the_parsed_yaml(descriptor):
    descriptor_node = pytest.importorskip("classical_control.descriptor_node")
    assert json.loads(descriptor_node.descriptor_json(CONTROL)) == descriptor


def test_publisher_is_latched():
    descriptor_node = pytest.importorskip("classical_control.descriptor_node")
    qos = descriptor_node.LATCHED
    assert (qos.depth, qos.durability.name, qos.reliability.name) == (
        1,
        "TRANSIENT_LOCAL",
        "RELIABLE",
    )


def test_control_file_must_be_a_yaml_inside_the_control_dir(tmp_path):
    descriptor_node = pytest.importorskip("classical_control.descriptor_node")
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / "anvil_classical.yaml").write_text("name: anvil_classical\n")
    (tmp_path / "secret.yaml").write_text("x: 1\n")
    (control_dir / "notes.txt").write_text("")
    inside = control_dir / "anvil_classical.yaml"
    assert descriptor_node.checked_control_file(inside, control_dir) == inside.resolve()
    for outside in (
        control_dir / ".." / "secret.yaml",
        tmp_path / "secret.yaml",
        control_dir / "notes.txt",
        "",
    ):
        with pytest.raises(ValueError, match="control_file must be a .yaml file in"):
            descriptor_node.checked_control_file(outside, control_dir)
