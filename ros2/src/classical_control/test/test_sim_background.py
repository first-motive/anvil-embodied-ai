"""The baked sim backgrounds localise the can in a rendered can_on_table frame.

Each fixture is the chest frame of fm-sim's can_on_table at seed 0, rendered by
tools/render_background.py with ``--scenario can_on_table``; each background is the
same backend's table_only render. Detection runs as perception_node runs it in sim:
config/sim/camera_chest.yaml, the shipped perception.yaml pick region, and the
camera pose the sim robot description gives cam_chest_optical.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from classical_control.camera_model import pose_to_matrix
from classical_control.detector import DetectorParams, detect
from classical_control.localise import ChestCameraConfig, locate_can, pick_region_mask

PACKAGE = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "sim"
# world <- cam_chest_optical in fm-robot's openarm_loader.sim.urdf.xacro (all fixed joints).
T_WORLD_OPTICAL = pose_to_matrix([0.0888, 0.0, 0.6231], [-0.659701, 0.659704, -0.254545, 0.254544])
# fm_sim_scenarios.scenario.spawn_poses(can_on_table, seed=0)['can'] xy.
CAN_XY_SEED0 = (0.40067, -0.13210)
# The real cell's offline eval puts can xy error at 12-20 mm median; the sim renders
# land in that band.
MAX_CAN_ERROR_M = 0.025


@pytest.fixture(scope="module")
def perception():
    return yaml.safe_load((PACKAGE / "config" / "perception.yaml").read_text())["/**"][
        "ros__parameters"
    ]


@pytest.mark.parametrize("backend", ["mujoco", "gazebo"])
def test_background_finds_the_seed0_can_where_the_scenario_spawned_it(perception, backend):
    size = (perception["process_width"], perception["process_height"])
    background = cv2.imread(str(PACKAGE / "backgrounds" / f"{backend}_openarm_chest.png"))
    frame = cv2.imread(str(FIXTURES / f"{backend}_can_on_table_seed0.jpg"))
    background, frame = (
        cv2.resize(i, size, interpolation=cv2.INTER_AREA) for i in (background, frame)
    )
    camera = ChestCameraConfig.from_yaml(PACKAGE / "config" / "sim" / "camera_chest.yaml").camera
    camera = camera.scaled(*size)
    table_z, can_height = perception["table_z"], perception["can_height"]
    region = pick_region_mask(
        camera,
        T_WORLD_OPTICAL,
        perception["pick_region_min_xy"],
        perception["pick_region_max_xy"],
        heights=(table_z, table_z + can_height / 2.0),
    )

    detection = detect(frame, background, DetectorParams(), region)

    assert detection.can_pixel is not None
    can = locate_can(camera, T_WORLD_OPTICAL, detection.can_pixel, table_z, can_height)
    assert np.hypot(can[0] - CAN_XY_SEED0[0], can[1] - CAN_XY_SEED0[1]) < MAX_CAN_ERROR_M
