"""Per-object settings for the collection loop, read from `config/objects/<name>.yaml`.

A collection run handles one object. Its file gives the object's height, which the
perception node needs to place the object on the table, and the ranges the per-cycle
grasp knobs are drawn from. Keeping these per object means swapping the can for another
object is a new file and a `--object` flag, not an edit to the task or perception config.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Object names are file stems passed on the command line; nothing that could walk paths.
_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")

_KNOBS = ("grasp_dz_m", "closure_m", "yaw_offset_rad")


@dataclass(frozen=True)
class ObjectConfig:
    """One object's collection settings.

    Attributes:
        name: The file stem, e.g. `can`.
        height_m: Height of the upright object, set as the perception node's can_height.
        grasp_dz_m: (low, high) height change added to grasp and place.
        closure_m: (low, high) finger position commanded at CLOSE.
        yaw_offset_rad: (low, high) grasp rotation about world z.
    """

    name: str
    height_m: float
    grasp_dz_m: tuple[float, float]
    closure_m: tuple[float, float]
    yaw_offset_rad: tuple[float, float]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.height_m) and self.height_m > 0.0):
            raise ValueError(f"height_m must be positive, got {self.height_m}")
        for knob in _KNOBS:
            low, high = getattr(self, knob)
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"knobs.{knob} must be finite [low, high], got [{low}, {high}]")

    def to_dict(self) -> dict[str, Any]:
        """The settings as a JSON-ready dict for the run's metadata."""
        return {
            "name": self.name,
            "height_m": self.height_m,
            "knobs": {knob: list(getattr(self, knob)) for knob in _KNOBS},
        }


def load_object_config(objects_dir: str | Path, name: str) -> ObjectConfig:
    """Read `<objects_dir>/<name>.yaml`.

    Raises:
        ValueError: If the name is not a plain file stem, or the file is malformed.
        FileNotFoundError: If there is no config for that object.
    """
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"object name must match {_NAME_PATTERN.pattern}, got {name!r}")
    raw = yaml.safe_load((Path(objects_dir) / f"{name}.yaml").read_text())
    if not isinstance(raw, dict) or "height_m" not in raw:
        raise ValueError(f"{name}.yaml must be a mapping with height_m")
    knobs = raw.get("knobs") or {}
    if not isinstance(knobs, dict):
        raise ValueError(f"knobs in {name}.yaml must be a mapping")
    unknown = set(knobs) - set(_KNOBS)
    if unknown:
        raise ValueError(f"unknown knobs in {name}.yaml: {sorted(unknown)}")
    ranges = {}
    for knob in _KNOBS:
        bounds = knobs.get(knob, [0.0, 0.0])
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"knobs.{knob} must be [low, high], got {bounds}")
        ranges[knob] = (float(bounds[0]), float(bounds[1]))
    return ObjectConfig(name=name, height_m=float(raw["height_m"]), **ranges)
