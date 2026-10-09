# classical_control

A non-learned baseline for the `pick-and-place-can` task on the OpenArm right arm.
It looks once with the fixed chest camera, then moves through a fixed sequence of
Cartesian waypoints. It exists to give ACT and VLA policies a classical number to
beat on the same task, so it deliberately has no retry and no closed-loop servoing.

```
/cam_chest/image_raw/compressed ─▶ perception_node ─▶ /classical/can_pose    (PoseStamped, world)
                                   bg diff, ray ∩ plane ─▶ /classical/paper_pose  (PoseStamped, world)
                                   static TF            ─▶ /classical/debug/compressed
                                   srv /classical/capture_background

/ee_pose_right ───────────────────┐
/joint_states ────────────────────┤
/hardware_state_controller/state ─┼─▶ task_node ─▶ /commanded_ee_right
/classical/{can,paper}_pose ──────┘   action /classical/pick_place
```

The loader's IK follows `/commanded_ee_right`, so there is no IK or MoveIt here.
Grasp and place heights and the gripper orientation come from the 104 teleop demos
(`mine_episodes`), so every target is one a human already reached.

## Task Sequence

```
LOCALISE → PRE_GRASP → DESCEND → CLOSE → (finger check) → LIFT
        → TRANSIT → LOWER → OPEN → RETREAT → HOME

cancel, stale input or hardware stop in any phase → HOLD the last command → error code
```

- The first command repeats the current `/ee_pose_right`. The loader latches the last
  commanded target, so starting anywhere else would jump the arm.
- Every waypoint is checked against the workspace box before anything moves.
- After CLOSE, a finger position below `grasp_missed_threshold_m` means the gripper
  closed on nothing: the task ends with `GRASP_MISSED`. A can holds the fingers near
  0.0185 m; an empty close reads near 0. With a goal `closure_width` an empty close
  reads near that width, so the check is `closure_width + grasp_missed_threshold_m`.
- A goal with `place_target_set` places at its xy instead of on the paper and needs
  no paper detection. A `home_only` goal skips LOCALISE's can and paper checks and
  runs HOME alone with the gripper at `gripper_home_m`.
- HOME is a commanded pose, not `/arms_resetter/reset`, which hands control back to
  the latched target and snaps the arm back.

## Interfaces

### Action `/classical/pick_place` (`classical_control_msgs/action/PickPlace`)

| Field | Type | Meaning |
|---|---|---|
| Goal `dry_run` | bool | Plan and log the waypoints; publish nothing to the arm |
| Goal `speed_scale` | float64 | Scales `v_max_mps` and `w_max_radps`. Clamped to [0.1, `max_speed_scale`]; 0 (the default) runs at 1.0; NaN is rejected |
| Goal `place_target_set`, `place_target` | bool, Point | Place at this `world` xy instead of on the paper; skips the paper checks. xy clamped into the workspace box |
| Goal `grasp_dz` | float64 | Metres added to grasp and place height. Clamped to ±`max_grasp_dz_m` |
| Goal `closure_width` | float64 | Finger position commanded at CLOSE. 0 (the default) uses `gripper_closed_m`; otherwise clamped to [`gripper_min_m`, `max_closure_m`] |
| Goal `yaw_offset` | float64 | Radians about world z added to the grasp orientation, pre-grasp to retreat. Clamped to ±`max_yaw_offset_rad` |
| Goal `home_only` | bool | Move HOME with the gripper at `gripper_home_m` and stop; needs no can or paper |
| Feedback `phase` | uint8 | Current phase, `LOCALISE` (0) to `HOME` (9), `HOLD` (10) |
| Result `error_code` | int8 | `SUCCESS`, `NO_CAN`, `NO_PAPER`, `OUT_OF_WORKSPACE`, `GRASP_MISSED`, `HARDWARE_NOT_ACTIVE`, `STALE_POSE`, `CANCELLED` |
| Result `can_pose`, `paper_pose` | PoseStamped | The detections the task used, in `world` |
| Result `duration_s` | float64 | Wall time from goal to result |

Every new goal field at its default reproduces the plain pick and place. A non-finite
(NaN or inf) float field rejects the goal. Only one goal runs at a time; a second is rejected.

### perception_node

| Topic / service | Type | Direction |
|---|---|---|
| `/cam_chest/image_raw/compressed` | sensor_msgs/CompressedImage | Subscribed, best effort |
| `/classical/can_pose` | geometry_msgs/PoseStamped | Published; can centre at `table_z + can_height/2` |
| `/classical/paper_pose` | geometry_msgs/PoseStamped | Published; paper centre at `table_z`, yaw from its long edge |
| `/classical/debug/compressed` | sensor_msgs/CompressedImage | Published at `debug_rate_hz`; mask, can dot, paper outline |
| `/tf_static` | `follower_body_link0 → cam_chest_optical` | Published from `camera_chest.yaml` |
| `/classical/capture_background` | std_srvs/Trigger | Saves the latest frame as the empty-table reference |

Nothing is published for a missing detection, so the task node sees a stale pose.
Only the pick region is searched: an arm that has moved since the background was
captured differs from it more than a can does, and would otherwise be taken for the can.
Place the can and paper inside the blue outline in `/classical/debug/compressed`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `camera_config_file` | string | `""` | Empty uses the installed `config/camera_chest.yaml` |
| `background_file` | string | `~/.ros/classical_control/chest_background.png` | Loaded at start if present |
| `process_width`, `process_height` | int | 960, 540 | Detection resolution |
| `process_rate_hz` | double | 5.0 | Detection rate; the task needs one fresh pose, not 60 Hz |
| `debug_rate_hz` | double | 1.0 | |
| `world_frame`, `camera_frame` | string | `world`, `cam_chest_optical` | |
| `table_z` | double | 0.207 | Measured; must match `task.yaml` |
| `can_height` | double | 0.135 | Measured (demo can) |
| `pick_region_min_xy`, `pick_region_max_xy` | double[2] | [0.10, −0.32], [0.45, 0.02] | World rectangle searched for the can and paper; outlined blue in the debug image |
| `detector.*` | mixed | see `config/perception.yaml` | Thresholds, as fractions of image size |

### task_node

Subscribes to `/ee_pose_right` and `/hardware_state_controller/state` (reliable),
`/joint_states` and the two perception poses (best effort). Publishes
`/commanded_ee_right` (`anvil_msgs/CommandedEEPose`) at `control_rate_hz` while a goal
runs, and nothing otherwise.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `mined_params_file` | string | `""` | Required; `mined_params.yaml` from `mine_episodes` |
| `frame_id` | string | `world` | |
| `control_rate_hz` | double | 30.0 | |
| `workspace_min_m`, `workspace_max_m` | double[3] | [0.05, -0.55, 0.0], [0.65, 0.15, 0.70] | TCP box; all 104 demo positions fall inside it |
| `table_z`, `z_margin_m` | double | 0.207, 0.01 | TCP floor is `table_z + z_margin_m`; `table_z` is measured |
| `max_position_step_m`, `max_rotation_step_rad` | double | 0.015, 0.075 | Per-tick clamp; 0.45 m/s and 2.25 rad/s at 30 Hz. The node refuses to start if the fastest goal would step past it |
| `gripper_min_m`, `gripper_max_m` | double | 0.0, 0.05 | |
| `v_max_mps`, `w_max_radps` | double | 0.10, 0.5 | Peak segment speeds before `speed_scale` |
| `max_speed_scale` | double | 3.0 | Largest `speed_scale` a goal may ask for |
| `approach_height_m` | double | 0.08 | Clearance for pre-grasp, lift, transit and retreat |
| `grasp_height_offset_m` | double | −0.035 | Added to the demos' grasp and place heights; grips below the can's neck |
| `level_grasp` | bool | true | Stands the mined grasp's most vertical tool axis upright, so the can is carried and released upright |
| `can_xy_bias_m` | double[2] | [0.0, 0.0] | Added to the detected can xy; take it from `eval_offline` |
| `max_grasp_dz_m`, `max_yaw_offset_rad`, `max_closure_m` | double | 0.02, 0.35, 0.010 | Clamp limits for a goal's `grasp_dz`, `yaw_offset` and `closure_width` |
| `close_dwell_s`, `open_dwell_s` | double | 1.0, 0.8 | |
| `gripper_open_m`, `gripper_closed_m`, `gripper_home_m` | double | 0.05, 0.0, 0.045 | |
| `grasp_missed_threshold_m` | double | 0.008 | |
| `home_position_m`, `home_orientation_xyzw` | double[3], double[4] | measured home TCP | |
| `pose_max_age_s` | double | 0.2 | Older arm inputs fault the task with `STALE_POSE` |
| `localise_max_age_s` | double | 1.0 | Older detections at LOCALISE give `STALE_POSE` |

### Tools

| Command | Does |
|---|---|
| `mine_episodes --recordings DIR --out-dir DIR` | Writes `mined_params.yaml` and `ground_truth.csv` (TCP at each gripper close and release) |
| `eval_offline --recordings DIR --ground-truth CSV --camera-yaml YAML --out-dir DIR` | Detection rate and xy error against the demos, plus a PnP-fitted extrinsic |
| `overlay_check --camera-yaml YAML` | Draws the live TCP and a table grid on one chest frame |
| `calibrate_chest --camera-yaml YAML --task-params YAML` | Sweeps the arm with a hand marker and fits the chest camera; `--dry-run`, `--fit-only` |
| `collect_node --object NAME ...` | Unattended pick and random re-place loop, one MCAP episode per cycle; see [Collecting Visuo-Tactile Episodes](#collecting-visuo-tactile-episodes) |

## Running On The Robot

Clone this repo on the robot PC, then use `scripts/run_classical.sh`. Its outputs go
to `data/classical/`, which is gitignored.

```bash
./scripts/run_classical.sh build        # vendors anvil_msgs from the loader image, then builds
./scripts/run_classical.sh mine         # data/classical/mined_params.yaml + ground_truth.csv
./scripts/run_classical.sh calibrate --marker-id 1 --marker-size 0.04   # see below
./scripts/run_classical.sh eval         # data/classical/eval/eval_report.yaml
./scripts/run_classical.sh up           # start both nodes
./scripts/run_classical.sh background   # with the table empty
./scripts/run_classical.sh trial --dry-run
./scripts/run_classical.sh trial --n 20 --speed 0.5
./scripts/run_classical.sh down
```

`trial` prompts before each goal, asks whether the can ended upright on the paper,
and appends one row per goal to `data/classical/trials.csv`. The container reads the
config from the repo on each start, so a calibration edit needs `down` and `up`, not
a rebuild.

`anvil_msgs` is proprietary. `scripts/vendor_anvil_msgs.sh` copies its install tree
from the running loader's image into `vendor/`, which is gitignored.

### Switching The Loader To Commanded EE

`/ee_pose_right` and the `/commanded_ee_right` subscriber exist only in commanded-EE
mode. In `~/anvil-loader/.env.config` set:

```
ENABLE_VR_TELEOP=true
ARMS_CONTROL_CONFIG_FILE=openarm_v2_quest_teleop_commanded_ee.yaml
```

then run `docker compose up -d` in `~/anvil-loader`. Set it back to
`openarm_v2_inference.yaml` for policy runs.

## Collecting Visuo-Tactile Episodes

`collect_node` runs the pick and place unattended to collect paired camera and FlexiTac
data. Each cycle picks the can and places it at a random free point in the pick region,
so the scene resets itself and changes every time. Unlike the baseline it retries a
missed grasp, because a run is only as long as its failures allow.

```
fresh can pose → sample place point + knobs → start bag → PickPlace goal → stop bag
  → episode metadata.json → SUCCESS: next cycle
                            GRASP_MISSED: HOME, same knobs again (at most 2 retries)
                            anything else: stop
stop (any reason, Ctrl-C, crash) → HOME goal → summary.json
```

The goal uses the `place_target`, `grasp_dz`, `closure_width` and `yaw_offset` fields
(see the action table). Knob ranges come from `config/objects/<name>.yaml` and start at
zero width; widen one at a time while attended. A run also sets perception's
`can_height` from that file. `config/collect.yaml` lists the recorded topics (every
loader topic plus `/gripper/tactile/*`), the disk guard and the loop's margins and
timeouts.

### Stop Rules

| Stop reason | When |
|---|---|
| `grasp_retries` | A grasp missed again after 2 retries on the same can |
| `consecutive_failures` | 3 failed attempts in a row; a backstop for limits set tighter than the retry cap |
| `error:<CODE>` | Any other result code, e.g. `error:NO_CAN`; also `GOAL_TIMEOUT`, `NO_RESULT`, `RECORDER`, `HOME` |
| `disk` | Free space below `disk_guard.min_free_gb` (50 GB) before a cycle |
| `cycles`, `time` | `--cycles` successes reached, or `--hours` elapsed, before a cycle |
| `operator` | Ctrl-C, `collect-stop`, or `tactile-collect stop` |

### Output

The layout matches the loader's episodes (`NNNN/NNNN_0.mcap` plus `metadata.json`, as in
`tests/smoke/fixtures/test-session`), which `mcap-valid` and `mcap-convert` read.

```
data/classical/collect/<run id>/
  metadata.json          object, knob ranges, stop limits, speed, git sha
  0001/0001_0.mcap       one episode per attempt, retries included
  0001/metadata.json     status, error_code, attempt, can_xy, place_xy, knobs, duration
  ...
  summary.json           attempts, episodes, successes, failures, stop_reason, home
```

With `--no-record` each episode directory holds only `metadata.json`. Deleting old runs
and uploading them is the team's process, not the loop's.

### Supervision Ramp

A supervised run, with someone in the room to stand a fallen can back up, needs only G1.
G2 and G3 are the ramp to unattended runs, and they are advisory: `start` warns while
the G2 marker is missing but does not refuse, so whoever starts an unattended run owns
that decision. Fix what each gate finds on a branch before moving on.

| Gate | Run | Passes when |
|---|---|---|
| G1 | 20 cycles at the default speed (2.5), attended: first `--no-record`, then recording | At least 18 succeed, and a recorded episode passes `mcap-valid` with the cameras and all four `/gripper/tactile/*` topics |
| G2 | One hour recording, someone nearby but not watching | No unsafe event and the right stop reason logged. Then write the marker: `echo "$(date -u +%F) <run id>" > data/classical/collect/.g2_passed`. Until it exists, `start` warns that someone must watch the run |
| G3 | First unattended run, started from Slack | Ends on its own limit or a clean stop, arm at HOME |

```bash
./scripts/run_classical.sh up                     # perception + task nodes; loader already in commanded EE
./scripts/run_classical.sh collect --object can --cycles 20 --no-record   # --speed defaults to 2.5
./scripts/run_classical.sh collect --object can --hours 1
./scripts/run_classical.sh collect --object can --hours 2 --detach   # background container
./scripts/run_classical.sh collect-stop          # Ctrl-C it: cancel, HOME, summary.json
```

`--dry-run` sends dry-run goals, so the loop, its metadata and its stop rules run with
nothing moving.

### Starting And Stopping From Slack

`fm.json` at the repo root mounts the `fm tactile-collect` verb
(`scripts/run/tactile-collect.sh`). With `--host` it runs on the robot over
`ssh -o BatchMode=yes`, against the checkout at `$ANVIL_EMBODIED_AI_DIR`
(default `~/anvil-embodied-ai`).

```bash
fm tactile-collect start --object can --hours 2 --host fm-rob-01   # prints the run id
fm tactile-collect status --json --host fm-rob-01
fm tactile-collect stop --host fm-rob-01
```

`start` refuses with exit 3 and a fix message unless the classical nodes are up, the
loader is active in commanded-EE mode, the empty-table background exists, free disk is
above the guard, and no loop is running. It starts but warns (`warnings` in the JSON,
stderr in text) when fewer than four `/gripper/tactile/*` topics have a publisher, so
episodes record without tactile, and when the G2 marker is missing, so someone must
watch. Exit codes: 0 done, 1 unhealthy, 2 usage, 3 precondition.

## Calibrating The Chest Camera

`config/camera_chest.yaml` holds the calibration from 2026-10-01; its header lists the
held-out checks it passed. Re-run this procedure if the camera or its mount is touched.
No published pose exists for this camera, so the arm calibrates it. An ArUco marker on the right hand gives known 3D points, and
`calibrate_chest` fits the intrinsics, the camera pose in `world`, and the marker's
offset on the hand in one solve. `world` and `follower_body_link0` are the same frame
on this robot.

```
marker on hand → arm sweeps ~15 poses → chest frame + measured TCP per pose
              → joint fit → data/classical/calibration/camera_chest.yaml (candidate)
```

1. Print a `DICT_4X4_50` marker at 100% scale and measure its black square. Tape it
   flat, with its white border, on the bracket that holds the wrist camera on the right
   gripper: that face points at the chest camera during grasps, and it does not move
   with the fingers. Keep the wrist-camera cable off it. Its exact position does not
   matter; the fit solves for it.
2. Switch the loader to commanded EE (above).
3. `calibrate --dry-run --marker-id 1 --marker-size 0.04` (use the id and size you
   printed). It logs the sweep and checks every pose against the workspace box without
   moving. The sweep keeps the home orientation, where the marker faces the chest
   camera, tilted by at most `--max-tilt` (0.25 rad), with the TCP low and near home
   (`--center 0.35 -0.15`, `--half-extent 0.05 0.05`, `--heights 0.42 0.47`). If views
   miss the marker, look at `views/NN.jpg` and move the sweep with these options.
4. Clear the table around the arm, keep the e-stop in hand, and run the same command
   without `--dry-run`. The arm visits each pose at 0.05 m/s, settles, and captures; it
   holds the gripper as it is and returns home after a complete sweep. An abort (e-stop,
   stale input, Ctrl-C) stops publishing instead, so the arm stays where it was; the
   views captured so far are kept for `--fit-only`. The loader holds that last target,
   so home the arm with `docker compose restart ros2` in `~/anvil-loader`.
5. Read `data/classical/calibration/report.yaml`. Expect the marker in at least 12
   views and a median corner error of 2 px or less. `views/NN.jpg` shows each capture.
   `--fit-only` re-fits the saved views without moving the arm.
6. Review the candidate `data/classical/calibration/camera_chest.yaml` and copy it to
   `config/camera_chest.yaml`, then `down` and `up`.
7. Run `overlay` with the arm at five poses. The projected TCP should land within about
   10 px of the gripper.
8. Measure `table_z` (base plate bottom to table top; 0.207 m on 2026-10-01)
   and set it in both `perception.yaml` and `task.yaml`. Run `eval` and set
   `can_xy_bias_m` from its `suggested_can_xy_bias`.

The fit holds the fisheye terms k3 and k4 at zero: the marker never reaches the image
edges, so they cannot be measured, and left free they distort the rest of the image.

## Safety

Commanded EE bypasses the inference node's `max_position_delta`. The task node's
clamps (5 mm and 0.05 rad per tick, the workspace box, the table floor, finite-value
checks, input freshness, and the hardware-state latch) are the only software guard
besides the webapp e-stop.

- The e-stop (`/hardware_state_controller/set_state estop`) latches until the loader
  restarts. The task node latches with it and aborts every later goal with
  `HARDWARE_NOT_ACTIVE` until it is restarted too.
- The loader does not reject an unreachable pose; the arm drifts toward it. Treat
  unexplained drift as a joint-limit problem and stop.
- Run a dry run first, then the first live trial at `--speed 0.5` with a hand on the
  e-stop. Stand clear of the arm.
- An unattended collection run has no hand on the e-stop. Its protection is the stop
  rules and the ramp above; `tactile-collect start` warns until the G2 marker exists. Every
  sampled place point lies inside the pick region and is clamped into the workspace box
  again by the task node, because the loader does not reject an unreachable pose.
- On a stop rule, Ctrl-C, SIGTERM or an exception, a collection run sends a HOME goal
  that a second Ctrl-C does not cancel. `docker stop` sends SIGINT with a 3-minute grace
  period for the same reason. A SIGKILL or a power loss skips the HOME move.

## Tests

```bash
cd ros2/src/classical_control
PYTHONPATH=$PWD uv run --no-project --with pytest --with numpy \
    --with opencv-python-headless --with scipy --with pyyaml pytest test
```

Or `colcon test --packages-select classical_control` in a Jazzy workspace. The
detector tests use three real chest frames and a median background in `test/fixtures/`.
