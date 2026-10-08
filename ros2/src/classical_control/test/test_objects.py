"""Tests for the per-object collection configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from classical_control.objects import load_object_config

OBJECTS_DIR = Path(__file__).resolve().parents[1] / "config" / "objects"


def write(tmp_path: Path, name: str, text: str) -> Path:
    (tmp_path / f"{name}.yaml").write_text(text)
    return tmp_path


def test_shipped_can_config_loads_with_zero_width_knobs() -> None:
    can = load_object_config(OBJECTS_DIR, "can")
    assert can.name == "can"
    assert can.height_m == pytest.approx(0.135)
    assert can.grasp_dz_m == can.closure_m == can.yaw_offset_rad == (0.0, 0.0)


def test_missing_knobs_default_to_zero_width(tmp_path: Path) -> None:
    config = load_object_config(write(tmp_path, "box", "height_m: 0.1\n"), "box")
    assert config.closure_m == (0.0, 0.0)


def test_to_dict_is_json_ready(tmp_path: Path) -> None:
    path = write(tmp_path, "box", "height_m: 0.1\nknobs:\n  yaw_offset_rad: [-0.1, 0.2]\n")
    data = json.loads(json.dumps(load_object_config(path, "box").to_dict()))
    assert data["knobs"]["yaw_offset_rad"] == [-0.1, 0.2]


@pytest.mark.parametrize("name", ["../can", "can.yaml", "Can", "", "can\n"])
def test_rejects_names_that_are_not_plain_stems(name: str) -> None:
    with pytest.raises(ValueError, match="object name"):
        load_object_config(OBJECTS_DIR, name)


def test_missing_object_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_object_config(tmp_path, "nothing")


@pytest.mark.parametrize(
    "text",
    [
        "height_m: 0.0\n",
        "height_m: -0.1\n",
        "height_m: 0.1\nknobs:\n  grasp_dz_m: [0.01, -0.01]\n",
        "height_m: 0.1\nknobs:\n  closure_m: [0.0]\n",
        "height_m: 0.1\nknobs:\n  squeeze: [0.0, 0.1]\n",
        "height_m: 0.1\nknobs:\n  yaw_offset_rad: [.nan, 0.1]\n",
        "",
        "knobs: {}\n",
        "height_m: 0.1\nknobs:\n  closure_m: 0.01\n",
        "height_m: 0.1\nknobs: [1, 2]\n",
    ],
)
def test_rejects_malformed_configs(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_object_config(write(tmp_path, "bad", text), "bad")
