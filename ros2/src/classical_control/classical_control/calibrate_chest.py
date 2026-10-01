"""Sweep the right arm through marker poses and fit the chest camera from what it sees.

    hardware_state ─┐
    /ee_pose_right ─┼─▶ calibrate_chest ──▶ /commanded_ee_right
    /joint_states  ─┤        │
    chest jpeg     ─┘        └─▶ out-dir/ samples.npz, views/NN.jpg, report.yaml,
                                          camera_chest.yaml (candidate)

An ArUco marker taped to the right hand gives known 3D points once its offset on the
hand is solved for, so the arm itself is the calibration target. This tool drives the arm
through `calibration.sweep_poses`, pairs each settled chest frame with the *measured*
TCP (the IK may stop short of the command), and hands the pairs to `calibration.fit`,
which solves intrinsics, `T_world_optical` and `T_tcp_marker` together.

Safety: this moves the arm unattended through commanded EE, which bypasses every
upstream limiter. It follows the task node's guards exactly:

- No motion unless the hardware-state probe is accepted and no stop has been seen; the
  ee pose, finger and chest frame must all be arriving before the sweep starts.
- The first command is the measured pose, because the loader latches its last target
  and anything else would jump. Every later command goes through `SafetyLimiter.clamp`
  against the previous one, and every sweep pose and home is checked against the
  workspace box before anything moves.
- The gripper is held at its measured opening, never opened or closed.
- A stale input or a hardware stop at any tick stops publishing, which holds the
  arm where the loader latched it, and ends the sweep; views already captured are kept.
  The arm is not sent home after an abort, since the guard that tripped would make
  that move unsafe too. Ctrl-C does the same.

`--dry-run` logs the sweep and checks it against the workspace without ROS. `--fit-only`
re-fits the samples a previous sweep saved, also without ROS.

Runs on a SingleThreadedExecutor for the reason given in `task_node`'s docstring: a
multi-threaded one starved the 100 Hz pose stream enough to trip the freshness gate.
The sequencing is the pure `SweepSequencer`; the ROS wiring loads only in `main`.
"""

from __future__ import annotations

import argparse
import datetime
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from classical_control.calibration import (
    MIN_VIEWS,
    CalibrationResult,
    MarkerSpec,
    Sample,
    camera_yaml,
    detect_marker,
    fit,
    load_samples,
    save_samples,
    sweep_poses,
)
from classical_control.camera_model import pose_to_matrix
from classical_control.eval_offline import (
    CHEST_TOPIC,
    add_parent_to_world_argument,
    camera_in_world,
    matrix_to_pose,
)
from classical_control.localise import ChestCameraConfig
from classical_control.safety import SafetyLimiter, SafetyLimits
from classical_control.trajectory import Pose, plan_segment, segment_duration

#: Abort reason for Ctrl-C, which skips the fit.
INTERRUPTED = "interrupted"
#: A view whose mean error exceeds max(OUTLIER_FACTOR x median, OUTLIER_FLOOR_PX) is dropped.
OUTLIER_FACTOR = 3.0
OUTLIER_FLOOR_PX = 3.0
#: Longest wait for a chest frame after settling before the camera counts as stale, s.
FRAME_TIMEOUT_S = 3.0
#: Longest wait at startup for the hardware probe and the first of every input, s.
STARTUP_TIMEOUT_S = 15.0
SAMPLES_FILE = "samples.npz"
REPORT_FILE = "report.yaml"
CANDIDATE_FILE = "camera_chest.yaml"
VIEWS_DIR = "views"
MARKER_COLOR = (0, 255, 0)  # BGR green
MISS_COLOR = (0, 0, 255)  # BGR red


# ---- configuration -----------------------------------------------------------


def load_task_params(path: str | Path) -> dict[str, Any]:
    """Read the ``ros__parameters`` block of a task_node parameter file.

    Args:
        path: A file shaped like ``config/task.yaml``.

    Returns:
        The parameter mapping.
    """
    data = yaml.safe_load(Path(path).expanduser().read_text())
    for node_params in data.values():
        if isinstance(node_params, Mapping) and "ros__parameters" in node_params:
            return dict(node_params["ros__parameters"])
    raise ValueError(f"{path} has no ros__parameters block")


def limiter_from_params(params: Mapping[str, Any]) -> SafetyLimiter:
    """Build the task node's limiter from its parameters, so both share one set of limits."""
    keys = SafetyLimits.__dataclass_fields__
    return SafetyLimiter(SafetyLimits.from_dict({k: params[k] for k in keys if k in params}))


def home_from_params(params: Mapping[str, Any]) -> Pose:
    """The pose the task node ends every run at."""
    return Pose(params["home_position_m"], params["home_orientation_xyzw"])


def workspace_violations(poses: Sequence[Pose], limiter: SafetyLimiter) -> list[int]:
    """Indices of poses outside the workspace box or below the table floor."""
    return [i for i, pose in enumerate(poses) if not limiter.in_workspace(pose)]


def describe_sweep(
    targets: Sequence[Pose], home: Pose, v_max: float, w_max: float, settle_s: float
) -> list[str]:
    """One log line per pose plus an estimated total, starting and ending at home."""
    lines = []
    total = 0.0
    previous = home
    for i, pose in enumerate(targets):
        move = segment_duration(previous, pose, v_max, w_max)
        total += move + settle_s
        x, y, z = pose.position
        tilt = math.degrees((home.rotation.inv() * pose.rotation).magnitude())
        lines.append(
            f"  {i:02d} xyz=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"rot from home={tilt:5.1f} deg move={move:.1f}s"
        )
        previous = pose
    total += segment_duration(previous, home, v_max, w_max)
    lines.append(f"  then home; about {total:.0f} s at v_max={v_max} m/s, w_max={w_max} rad/s")
    return lines


# ---- sequencing ----------------------------------------------------------------


class Stage(Enum):
    """Where the sweep is."""

    MOVE = auto()  # streaming the segment to sweep pose `index`
    SETTLE = auto()  # holding the pose while the arm stops shaking
    CAPTURE = auto()  # holding until a frame newer than the settle deadline arrives
    HOME = auto()  # streaming the segment back home
    DONE = auto()
    ABORTED = auto()


class SweepSequencer:
    """Decides what to command each tick: move, settle, capture, repeat, then go home.

    Pure: the caller feeds it the clock and the last commanded pose, clamps whatever it
    returns, and reports back when a capture is done. Each segment is planned from the
    last *commanded* pose when it starts, so a clamp that shortened a step is never
    undone by a jump.
    """

    def __init__(
        self,
        targets: Sequence[Pose],
        home: Pose,
        *,
        v_max: float,
        w_max: float,
        rate_hz: float,
        settle_s: float,
    ) -> None:
        if not targets:
            raise ValueError("the sweep needs at least one pose")
        self._targets = list(targets)
        self._home = home
        self._v_max = v_max
        self._w_max = w_max
        self._rate_hz = rate_hz
        self._settle_s = settle_s
        self._queue: deque[Pose] = deque()
        self._planned = False
        self._hold: Pose | None = None
        self._settle_until = 0.0
        self.stage = Stage.MOVE
        self.index = 0
        self.abort_reason: str | None = None

    @property
    def count(self) -> int:
        """Number of sweep poses."""
        return len(self._targets)

    @property
    def finished(self) -> bool:
        """True once home is reached or the sweep was aborted."""
        return self.stage in (Stage.DONE, Stage.ABORTED)

    @property
    def capture_after(self) -> float | None:
        """While capturing, only frames received after this monotonic time count."""
        return self._settle_until if self.stage is Stage.CAPTURE else None

    def start(self, measured: Pose) -> None:
        """Make the measured pose the first command, so streaming starts without a jump."""
        self._queue = deque([measured])
        self._hold = measured

    def tick(self, now: float, commanded: Pose) -> Pose | None:
        """Return the pose to command this tick (before clamping), or None to stop.

        Args:
            now: Monotonic time in seconds.
            commanded: The pose last published, after the clamp.
        """
        if self.finished:
            return None
        if self.stage in (Stage.MOVE, Stage.HOME):
            if not self._queue and not self._planned:
                goal = self._home if self.stage is Stage.HOME else self._targets[self.index]
                self._queue.extend(
                    plan_segment(commanded, goal, self._v_max, self._w_max, self._rate_hz)
                )
                self._planned = True
            if self._queue:
                self._hold = self._queue.popleft()
                return self._hold
            if self.stage is Stage.HOME:
                self.stage = Stage.DONE
                return None
            self.stage = Stage.SETTLE
            self._settle_until = now + self._settle_s
        if self.stage is Stage.SETTLE and now >= self._settle_until:
            self.stage = Stage.CAPTURE
        # Re-sent while settling so a clamp that lagged the command still converges.
        return self._hold

    def captured(self) -> None:
        """Record that this pose's frame was handled (marker found or not) and move on."""
        if self.stage is not Stage.CAPTURE:
            raise RuntimeError(f"captured() in stage {self.stage.name}")
        self.index += 1
        self._planned = False
        self.stage = Stage.MOVE if self.index < len(self._targets) else Stage.HOME

    def abort(self, reason: str) -> None:
        """Stop for good; the caller stops publishing, which holds the arm."""
        if not self.finished:
            self.stage = Stage.ABORTED
            self.abort_reason = reason


# ---- outputs -------------------------------------------------------------------


def annotate(image: np.ndarray, corners: np.ndarray | None, label: str) -> np.ndarray:
    """Draw the detected marker outline (first corner circled) and a label on a copy."""
    out = image.copy()
    thickness = max(1, round(out.shape[1] / 960))
    color = MISS_COLOR if corners is None else MARKER_COLOR
    if corners is not None:
        points = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [points], True, color, thickness * 2, cv2.LINE_AA)
        first = (int(points[0, 0, 0]), int(points[0, 0, 1]))
        cv2.circle(out, first, 8 * thickness, color, thickness, cv2.LINE_AA)
    cv2.putText(
        out, label, (20, 40 * thickness), cv2.FONT_HERSHEY_SIMPLEX, thickness, color, thickness
    )
    return out


def _pose_entry(matrix: np.ndarray) -> dict[str, list[float]]:
    translation, rotation = matrix_to_pose(matrix)
    return {"translation_xyz": translation, "rotation_xyzw": rotation}


def outlier_views(view_errors_px: Sequence[float]) -> list[int]:
    """Indices of views whose error exceeds max(3 x median, 3 px).

    The robust loss downweights single bad corners but not a whole view shifted
    consistently (a frame paired with a pose from mid-motion, say), so such views are
    dropped by their mean error and the fit is run again without them.
    """
    errors = np.asarray(view_errors_px, dtype=np.float64)
    if errors.size == 0:
        return []
    limit = max(OUTLIER_FACTOR * float(np.median(errors)), OUTLIER_FLOOR_PX)
    return [int(i) for i in np.flatnonzero(errors > limit)]


def build_report(
    result: CalibrationResult,
    spec: MarkerSpec,
    view_names: Sequence[str],
    attempted: int | None,
    dropped: Mapping[str, float],
) -> dict[str, Any]:
    """Collect the fit's errors and parameters into a yaml-ready mapping.

    Args:
        result: The final fit.
        spec: The marker the views show.
        view_names: One name per view in the final fit, in sample order.
        attempted: Views captured in the sweep, or None after `--fit-only`.
        dropped: Outlier views removed before the final fit, with their first-fit error.
    """
    camera = result.camera
    errors = [float(e) for e in result.view_errors_px]
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "views_used": len(errors),
        "views_attempted": attempted,
        "median_error_px": float(result.median_error_px),
        "view_errors_px": {name: round(e, 3) for name, e in zip(view_names, errors)},
        "dropped_views_px": {name: round(float(e), 3) for name, e in dropped.items()},
        "marker": {"dictionary": spec.dictionary, "id": spec.marker_id, "size_m": spec.size_m},
        "intrinsics": {
            "width": camera.width,
            "height": camera.height,
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
            "distortion": [float(k) for k in camera.distortion],
        },
        "T_world_optical": _pose_entry(result.T_world_optical),
        "T_tcp_marker": _pose_entry(result.T_tcp_marker),
    }


def candidate_header(views: int, median_px: float) -> str:
    """The comment block that heads the generated camera yaml."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# GENERATED by calibrate_chest at {stamp} from {views} marker views,\n"
        f"# median corner reprojection {median_px:.2f} px. See report.yaml next to this file.\n"
        "# Review it (overlay check), then copy it to ros2/src/classical_control/config/\n"
        "# camera_chest.yaml. Nothing reads it from here.\n"
    )


def fit_and_write(
    out_dir: Path,
    samples: list[Sample],
    spec: MarkerSpec,
    config: ChestCameraConfig,
    parent_to_world: Sequence[float],
    *,
    view_names: Sequence[str] | None = None,
    attempted: int | None = None,
) -> CalibrationResult | None:
    """Fit the samples, refit without outlier views, and write the report and candidate.

    The candidate camera yaml is written only for a successful fit.

    Returns:
        The final fit, or None when there are too few samples to try.
    """
    if len(samples) < MIN_VIEWS:
        return None
    names = list(view_names) if view_names else [f"sample {i}" for i in range(len(samples))]
    initial = camera_in_world(config, parent_to_world)
    result = fit(samples, spec, config.camera, initial)
    dropped: dict[str, float] = {}
    outliers = outlier_views(result.view_errors_px) if result.success else []
    # tradeoff: one refit, no iteration; a second round of outliers means the data is bad
    # and the report should show it rather than hide it.
    if outliers and len(samples) - len(outliers) >= MIN_VIEWS:
        dropped = {names[i]: result.view_errors_px[i] for i in outliers}
        kept = [i for i in range(len(samples)) if i not in outliers]
        samples, names = [samples[i] for i in kept], [names[i] for i in kept]
        result = fit(samples, spec, config.camera, initial)
    report = build_report(result, spec, names, attempted, dropped)
    (out_dir / REPORT_FILE).write_text(yaml.safe_dump(report, sort_keys=False))
    if not result.success:
        return result
    T_world_parent = pose_to_matrix(parent_to_world[:3], parent_to_world[3:])
    candidate = camera_yaml(result, config.parent_frame, T_world_parent)
    (out_dir / CANDIDATE_FILE).write_text(
        candidate_header(len(samples), float(result.median_error_px))
        + yaml.safe_dump(candidate, sort_keys=False)
    )
    return result


def summary(
    out_dir: Path, detected: int, attempted: int | None, result: CalibrationResult | None
) -> str:
    """The closing log message: counts, error, where files went, what to do next."""
    total = "?" if attempted is None else str(attempted)
    lines = [f"Marker detected in {detected}/{total} views; outputs in {out_dir}"]
    if result is None:
        lines.append(f"Not fitted: need at least {MIN_VIEWS} views with the marker.")
        return "\n".join(lines)
    verdict = "succeeded" if result.success else "FAILED"
    lines.append(
        f"Fit {verdict} on {len(result.view_errors_px)} views: median reprojection "
        f"{result.median_error_px:.2f} px ({result.message})"
    )
    lines.append(f"Report: {out_dir / REPORT_FILE}")
    if not result.success:
        lines.append("No candidate camera yaml written; see the report.")
        return "\n".join(lines)
    lines.append(
        f"Candidate: {out_dir / CANDIDATE_FILE}. Review it, then copy it to "
        "ros2/src/classical_control/config/camera_chest.yaml"
    )
    return "\n".join(lines)


# ---- live sweep ----------------------------------------------------------------


def run_sweep(
    args: argparse.Namespace,
    targets: list[Pose],
    home: Pose,
    limiter: SafetyLimiter,
    params: Mapping[str, Any],
    spec: MarkerSpec,
    expected_size: tuple[int, int],
) -> tuple[list[Sample], list[str], int, str | None]:
    """Drive the sweep on the robot and capture the views.

    Returns:
        ``(samples, view_names, attempted, abort_reason)``: detected samples, the jpg
        each came from, how many views were captured at all, and why the sweep
        stopped early (None when it finished at home).
    """
    import rclpy
    from anvil_msgs.msg import CommandedEEPose, HardwareState
    from anvil_msgs.srv import SetHardwareState
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, JointState

    from classical_control.safety import ACTIVE_STATE, FreshnessGate, HardwareStateGate
    from classical_control.task_node import (
        COMMAND_TOPIC,
        EE_POSE_TOPIC,
        FINGER_JOINT,
        HARDWARE_STATE_SERVICE,
        HARDWARE_STATE_TOPIC,
        JOINT_STATES_TOPIC,
        pose_from_message,
    )

    rate_hz = float(params.get("control_rate_hz", 30.0))
    frame_id = str(params.get("frame_id", "world"))
    inputs = ("ee_pose", "finger")
    views_dir = args.out_dir / VIEWS_DIR
    views_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = rclpy.create_node("calibrate_chest")
    log = node.get_logger()
    gate = HardwareStateGate()
    freshness = FreshnessGate(float(params.get("pose_max_age_s", 0.2)))
    sequencer = SweepSequencer(
        targets, home, v_max=args.v_max, w_max=args.w_max, rate_hz=rate_hz, settle_s=args.settle_s
    )
    state: dict[str, Any] = {
        "measured": None,
        "finger": None,
        "frame": None,  # (receive time, CompressedImage)
        "probe": None,
        "probe_answered": False,
        "commanded": None,
        "gripper": None,
        "started": False,
        "startup_deadline": time.monotonic() + STARTUP_TIMEOUT_S,
    }
    samples: list[Sample] = []
    view_names: list[str] = []
    attempted = 0

    def on_ee_pose(message: CommandedEEPose) -> None:
        try:
            state["measured"] = pose_from_message(message.pose)
        except ValueError:
            # Left unmarked: a corrupt pose goes stale and aborts the sweep.
            state["measured"] = None
            return
        freshness.mark("ee_pose", time.monotonic())

    def on_joint_states(message: JointState) -> None:
        if FINGER_JOINT not in message.name:
            return
        finger = float(message.position[message.name.index(FINGER_JOINT)])
        if math.isfinite(finger):
            state["finger"] = finger
            freshness.mark("finger", time.monotonic())

    def on_frame(message: CompressedImage) -> None:
        state["frame"] = (time.monotonic(), message)

    def on_hardware_state(message: HardwareState) -> None:
        if gate.update(message.state):
            log.warning(f"Hardware state is now {message.state!r}")

    reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    publisher = node.create_publisher(CommandedEEPose, COMMAND_TOPIC, reliable)
    node.create_subscription(CommandedEEPose, EE_POSE_TOPIC, on_ee_pose, reliable)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, on_joint_states, sensor)
    node.create_subscription(CompressedImage, CHEST_TOPIC, on_frame, qos_profile_sensor_data)
    node.create_subscription(HardwareState, HARDWARE_STATE_TOPIC, on_hardware_state, reliable)
    state_client = node.create_client(SetHardwareState, HARDWARE_STATE_SERVICE)

    def probe() -> None:
        """Ask for `active` once; the controller only publishes on change (see safety.py)."""
        if state["probe"] is None:
            if state_client.service_is_ready():
                request = SetHardwareState.Request()
                request.state = ACTIVE_STATE
                state["probe"] = state_client.call_async(request)
            return
        future = state["probe"]
        if state["probe_answered"] or not future.done() or future.result() is None:
            return
        response = future.result()
        gate.confirm_active(response.accepted)
        # A refused probe latches the gate shut, so there is nothing to retry.
        state["probe_answered"] = True
        if not response.accepted:
            sequencer.abort(f"hardware will not activate: {response.message}")

    def publish(pose: Pose) -> None:
        message = CommandedEEPose()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        p, q = message.pose.position, message.pose.orientation
        p.x, p.y, p.z = (float(v) for v in pose.position)
        q.x, q.y, q.z, q.w = (float(v) for v in pose.quat_xyzw)
        message.gripper = state["gripper"]
        publisher.publish(message)

    def capture(now: float) -> None:
        nonlocal attempted
        frame = state["frame"]
        if frame is None or frame[0] <= sequencer.capture_after:
            if now - sequencer.capture_after > FRAME_TIMEOUT_S:
                sequencer.abort(f"no chest frame within {FRAME_TIMEOUT_S:.0f} s of settling")
            return
        # The arm has settled, so the measured pose now is the pose in this frame.
        tcp = state["measured"]
        image = cv2.imdecode(np.frombuffer(bytes(frame[1].data), np.uint8), cv2.IMREAD_COLOR)
        name = f"{VIEWS_DIR}/{sequencer.index:02d}.jpg"
        if image is None:
            log.warning(f"View {sequencer.index:02d}: undecodable frame, skipped")
            sequencer.captured()
            return
        if (image.shape[1], image.shape[0]) != expected_size:
            sequencer.abort(
                f"chest frames are {image.shape[1]}x{image.shape[0]}, the camera yaml is "
                f"{expected_size[0]}x{expected_size[1]}: calibrate at native resolution"
            )
            return
        attempted += 1
        corners = detect_marker(image, spec)
        label = f"{sequencer.index:02d} " + ("marker" if corners is not None else "no marker")
        cv2.imwrite(str(args.out_dir / name), annotate(image, corners, label))
        if corners is None:
            log.warning(f"View {sequencer.index:02d}/{sequencer.count}: no marker, skipped")
        else:
            try:
                samples.append(Sample(tcp=tcp, corners=corners))
            except ValueError as error:
                log.warning(f"View {sequencer.index:02d}: unusable corners ({error}), skipped")
            else:
                view_names.append(name)
                log.info(f"View {sequencer.index:02d}/{sequencer.count}: marker found")
        sequencer.captured()

    def on_tick() -> None:
        now = time.monotonic()
        if not gate.allows_commands:
            if state["started"]:
                sequencer.abort(f"hardware state is {gate.state!r}")
            else:
                probe()
        stale = freshness.stale(inputs, now)
        if state["measured"] is None and "ee_pose" not in stale:
            stale = (*stale, "ee_pose")
        if not state["started"]:
            if sequencer.finished:
                return
            ready = gate.allows_commands and not stale and state["frame"] is not None
            if not ready:
                if now > state["startup_deadline"]:
                    missing = list(stale) + ([] if state["frame"] else ["chest frame"])
                    if not gate.allows_commands:
                        missing.append("hardware not confirmed active")
                    sequencer.abort(f"not ready after {STARTUP_TIMEOUT_S:.0f} s: {missing}")
                return
            # Held, never opened or closed: the marker sits on the hand, not in the fingers.
            state["gripper"], _ = limiter.clamp_gripper(state["finger"])
            state["commanded"] = state["measured"]
            sequencer.start(state["measured"])
            state["started"] = True
            log.info(f"Sweeping {sequencer.count} poses; Ctrl-C or the e-stop to stop")
        if stale:
            sequencer.abort(f"stale inputs {', '.join(stale)}")
        if sequencer.stage is Stage.CAPTURE:
            capture(now)
        target = sequencer.tick(now, state["commanded"])
        if target is None:
            return
        pose, report = limiter.clamp(target, state["commanded"])
        if report.any_clamped:
            log.warning(f"Clamped command: {report}", throttle_duration_sec=1.0)
        state["commanded"] = pose
        publish(pose)

    node.create_timer(1.0 / rate_hz, on_tick)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    log.info("Waiting for the hardware probe, /ee_pose_right, /joint_states and a chest frame")
    try:
        while rclpy.ok() and not sequencer.finished:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl-C lands here either as the exception or as rclpy's handler shutting the
        # context down. Nothing more is published, so the loader holds its last target.
        sequencer.abort(INTERRUPTED)
        if sequencer.stage is Stage.ABORTED:
            log.error(f"Sweep aborted, arm held where it is: {sequencer.abort_reason}")
        else:
            log.info("Sweep complete, arm is home")
        node.destroy_node()
        rclpy.try_shutdown()
    return samples, view_names, attempted, sequencer.abort_reason


# ---- entry point ---------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command line; see the module docstring for the modes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-yaml", type=Path, required=True, help="initial camera yaml")
    parser.add_argument("--task-params", type=Path, help="task.yaml: limits, home, rate")
    parser.add_argument("--out-dir", type=Path, default=Path("/data/calibration"))
    parser.add_argument("--marker-size", type=float, default=0.05, help="marker side, m")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--count", type=int, default=15, help="sweep poses")
    # The first live sweep centred on the demos' grasp orientation and turned the hand
    # bracket edge-on to the chest camera; at home the marker faces it. So the sweep keeps
    # home's orientation and stays low and near home, where the marker sits inside the
    # frame. These defaults came from that run; adjust them on the robot if views miss.
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=(0.35, -0.15),
        metavar=("X", "Y"),
        help="sweep centre in world, m",
    )
    parser.add_argument(
        "--half-extent",
        type=float,
        nargs=2,
        default=(0.05, 0.05),
        metavar=("DX", "DY"),
        help="sweep half-size in x and y, m",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs=2,
        default=(0.42, 0.47),
        metavar=("Z_LOW", "Z_HIGH"),
        help="sweep TCP heights, m",
    )
    parser.add_argument(
        "--max-tilt",
        type=float,
        default=0.25,
        help="largest tilt away from home's orientation, rad",
    )
    parser.add_argument("--v-max", type=float, default=0.05, help="peak linear speed, m/s")
    parser.add_argument("--w-max", type=float, default=0.25, help="peak angular speed, rad/s")
    parser.add_argument("--settle-s", type=float, default=1.0, help="hold before capture, s")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="log the sweep, move nothing")
    mode.add_argument("--fit-only", action="store_true", help=f"re-fit out-dir/{SAMPLES_FILE}")
    add_parent_to_world_argument(parser)
    args = parser.parse_args(argv)
    if not args.fit_only and args.task_params is None:
        parser.error("--task-params is required unless --fit-only")
    if args.v_max <= 0.0 or args.w_max <= 0.0 or args.settle_s < 0.0:
        parser.error("--v-max and --w-max must be positive and --settle-s non-negative")
    numbers = [
        args.marker_size,
        args.v_max,
        args.w_max,
        args.settle_s,
        args.max_tilt,
        *args.center,
        *args.half_extent,
        *args.heights,
    ]
    # argparse takes "nan" and "inf" as floats, and NaN passes every comparison below.
    if not all(math.isfinite(v) for v in numbers):
        parser.error("numeric options must be finite")
    if min(args.half_extent) <= 0.0 or not 0.0 < args.max_tilt <= 0.5:
        parser.error("--half-extent must be positive and --max-tilt in (0, 0.5] rad")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Dry-run, sweep and fit, or re-fit saved samples."""
    args = parse_args(argv)
    config = ChestCameraConfig.from_yaml(args.camera_yaml)

    if args.fit_only:
        path = args.out_dir / SAMPLES_FILE
        if not path.is_file():
            raise SystemExit(f"No {path}: run a sweep first")
        samples, spec = load_samples(path)
        print(f"Loaded {len(samples)} samples, marker {spec}")
        result = fit_and_write(args.out_dir, samples, spec, config, args.parent_to_world)
        print(summary(args.out_dir, len(samples), None, result))
        return

    params = load_task_params(args.task_params)
    limiter = limiter_from_params(params)
    home = home_from_params(params)
    # The sweep must never outrun the task node, whose limits are the ones reviewed for
    # this arm; a typo on the command line should refuse, not move faster.
    if args.v_max > params["v_max_mps"] or args.w_max > params["w_max_radps"]:
        raise SystemExit(
            f"Refusing: --v-max {args.v_max} / --w-max {args.w_max} exceed task.yaml's "
            f"{params['v_max_mps']} m/s / {params['w_max_radps']} rad/s"
        )
    try:
        targets = sweep_poses(
            home.quat_xyzw,
            limiter,
            count=args.count,
            center_xy=tuple(args.center),
            half_extent_xy=tuple(args.half_extent),
            heights=tuple(args.heights),
            max_tilt_rad=args.max_tilt,
        )
    except ValueError as error:
        raise SystemExit(f"Refusing: {error}") from error
    outside = workspace_violations([*targets, home], limiter)
    print(f"Sweep of {len(targets)} poses:")
    print("\n".join(describe_sweep(targets, home, args.v_max, args.w_max, args.settle_s)))
    if outside:
        # Index len(targets) is home.
        raise SystemExit(f"Refusing: poses {outside} lie outside the workspace box")
    print("All poses inside the workspace box and above the table floor")
    if args.dry_run:
        return

    spec = MarkerSpec(dictionary="DICT_4X4_50", marker_id=args.marker_id, size_m=args.marker_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected = (config.camera.width, config.camera.height)
    samples, view_names, attempted, abort_reason = run_sweep(
        args, targets, home, limiter, params, spec, expected
    )
    if samples:
        save_samples(args.out_dir / SAMPLES_FILE, samples, spec)
    if abort_reason == INTERRUPTED:
        print(f"Interrupted after {len(samples)} detected views; re-fit them with --fit-only")
        return
    result = fit_and_write(
        args.out_dir,
        samples,
        spec,
        config,
        args.parent_to_world,
        view_names=view_names,
        attempted=attempted,
    )
    print(summary(args.out_dir, len(samples), attempted, result))


if __name__ == "__main__":
    main()
