# MCAP Converter

Convert ROS2 MCAP recordings into LeRobot v3.0 datasets.

## Installation

```bash
# From repository root (installs all workspace packages)
uv sync --all-packages
```

## CLI Tools

### mcap-convert

Convert MCAP files to LeRobot dataset format.

```bash
mcap-convert -i /path/to/mcap/dir -o /tmp/dataset --config configs/mcap_converter/openarm_bimanual.yaml
```

Key options:

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input-dir` | Directory containing MCAP files | required |
| `-o, --output-dir` | Output directory | `data/processed/dataset` |
| `--config` | Robot config YAML file | |
| `--fps` | Video framerate | `30` |
| `--vcodec` | Video codec (`h264`, `hevc`, `libsvtav1`) | `h264` |
| `--task` | Task name for the dataset | `manipulation` |
| `--push-to-hub` | Upload to HuggingFace Hub after conversion | |

### mcap-to-video

Extract image topics from MCAP files directly to MP4 videos.

```bash
mcap-to-video -i recording.mcap -o ./videos
mcap-to-video -i recording.mcap --scan-only
```

### dataset-valid

Validate a converted LeRobot dataset by loading and reading frames.

```bash
dataset-valid --root /path/to/dataset
```

### dataset-viz

Browse a converted dataset's episodes, videos, and action curves with lerobot's own Rerun-based viewer.

```bash
dataset-viz /path/to/dataset
```

See [docs/dataset-viz.md](../../docs/dataset-viz.md) for the full flag reference and troubleshooting.

### mcap-upload

Upload a LeRobot dataset to Hugging Face Hub.

```bash
mcap-upload /path/to/dataset --repo-id anvil-robot/my_dataset
```

## Python API

```python
from mcap_converter import McapReader, LeRobotWriter, ConfigLoader

# Load configuration
config = ConfigLoader.from_yaml("configs/mcap_converter/openarm_bimanual.yaml")

# Read MCAP file
reader = McapReader("recording.mcap")

# Write dataset
writer = LeRobotWriter(
    output_dir="output_dataset",
    repo_id="anvil_robot/my_dataset",
    fps=30,
)
dataset = writer.create_dataset(joint_names, camera_names)
writer.add_episode(dataset, frames, episode_index=0)
writer.finalize(dataset)
```

## Configuration

Example for bimanual OpenArm robot (`configs/mcap_converter/openarm_bimanual.yaml`):

```yaml
robot_state_topic: "/joint_states"

joint_names:
  separator: "_"
  source:
    leader: action
    follower: observation
  arms:
    r: right
    l: left

camera_topic_mapping:
  "/usb_cam_waist/image_raw/compressed": "waist"
  "/usb_cam_wrist_r/image_raw/compressed": "wrist_r"

image_resolution: [640, 480]

observation_feature_mapping:
  state: "position"
  others:
    - "velocity"
    - "effort"

action_feature_mapping:
  state: "position"
  others: []
```

## Skill Segments in the Handover Frame

For a learned skill trained on continuous fm-skills recordings, set
`skill_segments` in the config
(`configs/mcap_converter/openarm_single_handover_ee.yaml`). The converter then
cuts one episode per `[SKILL_START, SKILL_END]` pair of the chosen skill on
`/task/events` (`fm_skills_msgs/msg/TaskEvent`), and replaces the joint
features with the TCP pose relative to the handover frame H, the TCP pose the
SKILL_START event carries.

```bash
mcap-convert -i data/raw/my-session -o data/datasets \
    --config configs/mcap_converter/openarm_single_handover_ee.yaml --fps 30
```

| Feature | Names | Meaning |
|---------|-------|---------|
| `observation.state` (8) | `x y z rx ry rz finger_width_m finger_effort` | TCP pose at t in H, finger position and effort at t |
| `action` (7) | `x y z rx ry rz gripper` | TCP pose at t+1 in H; gripper target at t |
| `observation.images.wrist_r` | | wrist camera; the first camera paces the frames |

The frame convention:

- Pose in H: `p = R_H⁻¹ (p_tcp − o_H)` and `r = rotvec(R_H⁻¹ R_tcp)`, with
  o_H, R_H the origin and rotation of H. Relative to H the rotation starts at
  zero and normally stays well inside pi, so the rotation vector is continuous
  over a segment; a segment that turns past 2.5 rad is logged as a warning.
  `pose_in_frame` and its inverse `pose_from_frame` in `core/frames.py`
  implement it.
- The action pose is the next frame's measured TCP in H, absolute, with the
  same names as the state. Train with
  `--action-type=delta_obs_t --delta-exclude-joints=gripper`: anvil-trainer's
  delta transform then subtracts the observed pose from every row of the ACT
  chunk by name, which is how the policy node applies a chunk at inference.
  The gripper target stays absolute.
- The gripper target is the `position` (m) of the last
  `control_msgs/msg/GripperCommand` on `gripper_command_topic`
  (`/motion/gripper`) published at or before t. Before the first command, or
  when the topic is not recorded, it is the measured finger width at t+1; a
  measured target alone never squeezes. GripperCommand has no header, so it
  is aligned on MCAP log time while every other topic is aligned on its
  header stamp.
- The TCP pose comes from `/ee_pose_right` (`anvil_msgs/msg/CommandedEEPose`),
  the finger from `follower_r_finger_joint1` in `/joint_states`.
- Only segments that end with code 0 and `verified` are kept. With
  `include_failures: true` the others are kept too, with task
  `"<task> [failed]"`.
- Frames sit on a `--fps` grid anchored at the segment's first camera frame.
  Each segment is read within a log-time window 1 s wider than its stamps; a
  warning is logged when header stamps and log times differ by more.
- `--task` defaults to the skill name. `--tolerance-s` and `--buffer-seconds`
  are unused in this mode.
- `--resume` is not supported in this mode. `--max-episodes` and
  `--skip-episode-idx` count MCAP files, not segments.

## Module Structure

```
mcap_converter/
├── core/
│   ├── reader.py      # MCAP file reading
│   ├── extractor.py   # Data extraction & streaming
│   ├── aligner.py     # Time synchronization
│   ├── frames.py      # Poses relative to another frame
│   ├── skill_segments.py # Skill segments in the handover frame
│   └── writer.py      # LeRobot dataset writing
├── cli/
│   ├── convert.py     # mcap-convert
│   ├── validate.py    # dataset-valid
│   ├── upload.py      # mcap-upload
│   └── video.py       # mcap-to-video
├── config/
│   ├── schema.py      # Configuration dataclasses
│   ├── loader.py      # YAML config loading
│   └── validators.py  # Config validation
├── utils/
│   ├── image_utils.py # Image processing
│   └── logging.py     # Logging utilities
└── exceptions.py      # Custom exceptions
```
