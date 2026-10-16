# anvil_robot_profiles

Joint, controller and frame names for each robot build that `classical_control` and
`lerobot_control` drive. The same OpenArm v2 runs under the Anvil loader's `follower_*`
names and under First Motive's canonical `openarm_*` names. Nodes read joint, gripper
and controller names from a profile, so switching them is one `robot_profile` parameter
instead of a code change. Topics the loader names per side (`/ee_pose_right`,
`/commanded_ee_right`) and the camera image topics stay fixed.

## Profiles

| Profile | Names | Source |
|---|---|---|
| `loader_openarm_v2` (default) | `follower_r_joint1`, `follower_r_forward_position_controller`, `follower_body_link0` | The names the nodes hardcoded before profiles existed |
| `fm_openarm_v2` | `openarm_right_joint1`, `openarm_right_forward_position_controller`, `openarm_right_arm_controller` (JTC), `openarm_body_link0` | fm-robot's OpenArm description (openarm_description v2.0, `default_bimanual`); JTC names from fm_bringup's `default_bimanual.controllers.yaml` |

The yaml files live in `profiles/` and install to `share/anvil_robot_profiles/profiles`.
fm-skills uses the same schema:

```yaml
name: loader_openarm_v2
base_frame: follower_body_link0
arms:
  right:
    joints: [follower_r_joint1, ..., follower_r_joint7]   # controller order
    gripper_joint: follower_r_finger_joint1
    arm_controller: follower_r_forward_position_controller
    trajectory_controller: follower_r_jtc
    ee_frame: follower_r_hand_tcp
    hand_frame: follower_r_hand
  left: {...}            # same keys
cameras:                 # optional optical frame names; fm_openarm_v2 uses
  chest: cam_chest_optical          # openarm_cam_chest_optical and
  wrist_right: cam_wrist_r_optical  # openarm_right_cam_wrist_optical
```

Every arm needs all six keys, and every name must be a valid ROS name
(`[A-Za-z][A-Za-z0-9_]*`); `load_profile` rejects anything else. A bare profile name must
be a plain identifier, so it always resolves inside the profiles directory.
`DRIVEN_ARM` (`right`) is the arm the single-arm classical_control nodes drive.

## Usage

```python
from anvil_robot_profiles import declare_robot_profile, load_profile

profile = declare_robot_profile(node)          # in a node: declares `robot_profile`
profile = load_profile("fm_openarm_v2")        # by name, or by path to a .yaml

right = profile.arms["right"]
right.gripper_joint                            # "openarm_right_finger_joint1"
right.command_topic                            # "/openarm_right_forward_position_controller/commands"
right.command_joints                           # joints, then gripper: the controller order
right.joint_name("joint3")                     # "openarm_right_joint3"
profile.joint_names(["left", "right"], ["finger_joint1", "joint1"])
```

Datasets and policy configs name joints by a robot-neutral id (`joint3`,
`finger_joint1`). `joint_name` adds the arm's prefix, which is the text every joint name
of that arm shares up to its last `_`. `joint_names` lays out a policy's state vector arm
by arm in the order given, so a policy sees the same layout under any profile.

Outside a ROS install (plain pytest, offline tools), profiles load from the source tree.
It is a ROS package rather than part of `packages/anvil_shared` because the ROS images
build `ros2/src` with colcon and never install the uv workspace packages.

## Tests

```bash
cd ros2/src/anvil_robot_profiles
PYTHONPATH=$PWD uv run --no-project --with pytest --with pyyaml pytest test
```
