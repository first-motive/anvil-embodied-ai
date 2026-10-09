"""ROS2 action server running one classical can-on-paper pick and place per goal.

    /classical/can_pose    ─┐
    /classical/paper_pose  ─┤
    /ee_pose_right         ─┼─▶ task_node ──▶ /commanded_ee_right
    /joint_states          ─┤   (/classical/pick_place action)
    hardware_state         ─┘

The node is wiring only. `task_machine.PickPlaceTask` decides what to command;
this node localises, gates and clamps:

- LOCALISE: the hardware gate must be open, the arm and finger inputs fresh,
  and the can and paper poses newer than `localise_max_age_s`, otherwise the
  goal aborts with HARDWARE_NOT_ACTIVE, STALE_POSE, NO_CAN or NO_PAPER before
  anything moves. A goal with a place target needs no paper; a home-only goal
  needs neither can nor paper.
- Every tick of the 30 Hz timer checks the gate and freshness again (faulting
  the task into HOLD if either fails), then passes the task's command through
  `SafetyLimiter.clamp` and `clamp_gripper`. `/commanded_ee_right` bypasses all
  upstream limiters, so that clamp is the last software guard before the IK.
- The loader latches the last commanded EE target. The limiter's previous pose
  is seeded from the measured `/ee_pose_right` at goal start, and the task's
  first command is that same pose, so a run never starts with a jump.
- HOME is commanded through the same stream; the arms resetter is never called.

Everything runs on a SingleThreadedExecutor. The execute callback is a
coroutine that awaits a future the control timer resolves when the task ends,
so it never blocks the timer or the subscriptions and needs no lock. A
MultiThreadedExecutor was tried first: on Jazzy it left 0.2-1 s gaps in a
100 Hz `/ee_pose_right` stream even when idle, which trips the 0.2 s freshness
gate, while the single-threaded executor held gaps at ~12 ms.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import rclpy
import yaml
from anvil_msgs.msg import CommandedEEPose, HardwareState
from anvil_msgs.srv import SetHardwareState
from classical_control_msgs.action import PickPlace
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import JointState

from .safety import ACTIVE_STATE, FreshnessGate, HardwareStateGate, SafetyLimiter, SafetyLimits
from .task_machine import (
    ErrorCode,
    GoalOverrides,
    Phase,
    PickPlaceTask,
    TaskConfig,
    goal_overrides,
    level_orientation,
    mined_targets,
    place_target_in_workspace,
    required_detections,
    speed_ceiling_error,
)
from .trajectory import Pose

COMMAND_TOPIC = "/commanded_ee_right"
EE_POSE_TOPIC = "/ee_pose_right"
JOINT_STATES_TOPIC = "/joint_states"
CAN_POSE_TOPIC = "/classical/can_pose"
PAPER_POSE_TOPIC = "/classical/paper_pose"
HARDWARE_STATE_TOPIC = "/hardware_state_controller/state"
HARDWARE_STATE_SERVICE = "/hardware_state_controller/set_state"
ACTION_NAME = "/classical/pick_place"
FINGER_JOINT = "follower_r_finger_joint1"

#: Inputs that must stay fresh for the whole run.
ARM_INPUTS = ("ee_pose", "finger")

#: Defaults for every parameter; config/task.yaml documents each one.
PARAMETER_DEFAULTS = {
    "mined_params_file": "",
    "frame_id": "world",
    "control_rate_hz": 30.0,
    "workspace_min_m": [0.05, -0.55, 0.0],
    "workspace_max_m": [0.65, 0.15, 0.70],
    "table_z": 0.207,
    "z_margin_m": 0.01,
    "max_position_step_m": 0.010,
    "max_rotation_step_rad": 0.05,
    "gripper_min_m": 0.0,
    "gripper_max_m": 0.05,
    "v_max_mps": 0.10,
    "w_max_radps": 0.5,
    "approach_height_m": 0.08,
    "grasp_height_offset_m": -0.035,
    "level_grasp": True,
    "can_xy_bias_m": [0.0, 0.0],
    "max_grasp_dz_m": 0.02,
    "max_yaw_offset_rad": 0.35,
    "max_closure_m": 0.010,
    "max_speed_scale": 1.0,
    "close_dwell_s": 1.0,
    "open_dwell_s": 0.8,
    "gripper_open_m": 0.05,
    "gripper_closed_m": 0.0,
    "gripper_home_m": 0.045,
    "grasp_missed_threshold_m": 0.008,
    "home_position_m": [0.396, -0.185, 0.506],
    "home_orientation_xyzw": [0.672, 0.064, 0.736, 0.055],
    "pose_max_age_s": 0.2,
    "localise_max_age_s": 1.0,
}

SAFETY_KEYS = (
    "workspace_min_m",
    "workspace_max_m",
    "table_z",
    "z_margin_m",
    "max_position_step_m",
    "max_rotation_step_rad",
    "gripper_min_m",
    "gripper_max_m",
)


def pose_from_message(pose) -> Pose:
    """Convert a geometry_msgs/Pose into a task Pose."""
    p, q = pose.position, pose.orientation
    return Pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


class PickPlaceNode(Node):
    """Serves `/classical/pick_place` and streams the task to `/commanded_ee_right`."""

    def __init__(self) -> None:
        super().__init__("classical_task")
        self.declare_parameters("", list(PARAMETER_DEFAULTS.items()))
        param = {name: self.get_parameter(name).value for name in PARAMETER_DEFAULTS}

        self._frame_id = param["frame_id"]
        self._rate_hz = float(param["control_rate_hz"])
        self._limiter = SafetyLimiter(SafetyLimits.from_dict({k: param[k] for k in SAFETY_KEYS}))
        self._config = self._load_task_config(param)
        ceiling = speed_ceiling_error(self._config, self._limiter.limits)
        if ceiling is not None:
            raise ValueError(f"Refusing to start: {ceiling}")
        self._max_grasp_dz_m = float(param["max_grasp_dz_m"])
        self._max_yaw_offset_rad = float(param["max_yaw_offset_rad"])
        self._max_closure_m = float(param["max_closure_m"])

        self._gate = HardwareStateGate()
        self._freshness = FreshnessGate(float(param["pose_max_age_s"]))
        self._localisation = FreshnessGate(float(param["localise_max_age_s"]))
        self._measured: Pose | None = None
        self._finger: float | None = None
        self._can: PoseStamped | None = None
        self._paper: PoseStamped | None = None
        self._probe = None
        self._probe_answered = False

        self._busy = False
        self._task: PickPlaceTask | None = None
        self._goal_handle = None
        self._fed_back: Phase | None = None
        self._commanded: Pose | None = None
        self._finished: Future | None = None

        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        # Best effort subscribers match both reliable and best-effort publishers.
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._publisher = self.create_publisher(CommandedEEPose, COMMAND_TOPIC, reliable)
        self.create_subscription(CommandedEEPose, EE_POSE_TOPIC, self._on_ee_pose, reliable)
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_states, sensor)
        self.create_subscription(PoseStamped, CAN_POSE_TOPIC, self._on_can_pose, sensor)
        self.create_subscription(PoseStamped, PAPER_POSE_TOPIC, self._on_paper_pose, sensor)
        self.create_subscription(
            HardwareState,
            HARDWARE_STATE_TOPIC,
            self._on_hardware_state,
            reliable,
        )
        self._state_client = self.create_client(SetHardwareState, HARDWARE_STATE_SERVICE)
        self._action_server = ActionServer(
            self,
            PickPlace,
            ACTION_NAME,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
        )
        self.create_timer(1.0 / self._rate_hz, self._on_tick)
        self.get_logger().info(f"Ready: send a goal to {ACTION_NAME}")

    def _load_task_config(self, param: dict) -> TaskConfig:
        path = param["mined_params_file"]
        if not path:
            raise ValueError("mined_params_file is required (path to mined_params.yaml)")
        grasp_z, place_z, quat = mined_targets(
            yaml.safe_load(Path(path).expanduser().read_text()),
            float(param["grasp_height_offset_m"]),
        )
        if param["level_grasp"]:
            quat, tilt = level_orientation(quat)
            self.get_logger().info(f"Grasp orientation levelled by {math.degrees(tilt):.1f} deg")
        return TaskConfig(
            grasp_z_m=grasp_z,
            place_z_m=place_z,
            grasp_orientation_xyzw=quat,
            home=Pose(param["home_position_m"], param["home_orientation_xyzw"]),
            approach_height_m=float(param["approach_height_m"]),
            can_xy_bias_m=tuple(param["can_xy_bias_m"]),
            v_max_mps=float(param["v_max_mps"]),
            w_max_radps=float(param["w_max_radps"]),
            rate_hz=self._rate_hz,
            close_dwell_s=float(param["close_dwell_s"]),
            open_dwell_s=float(param["open_dwell_s"]),
            gripper_open_m=float(param["gripper_open_m"]),
            gripper_closed_m=float(param["gripper_closed_m"]),
            gripper_home_m=float(param["gripper_home_m"]),
            grasp_missed_threshold_m=float(param["grasp_missed_threshold_m"]),
            max_speed_scale=float(param["max_speed_scale"]),
        )

    # ---- subscriptions -----------------------------------------------------

    def _on_ee_pose(self, message: CommandedEEPose) -> None:
        try:
            pose = pose_from_message(message.pose)
        except ValueError as error:
            # A corrupt measured pose is as unusable as a missing one: keep the
            # last good one out of reach of the clamp and hold the run.
            self._measured = None
            if self._task is not None:
                self._task.fault(ErrorCode.STALE_POSE)
            self.get_logger().error(f"Bad {EE_POSE_TOPIC}: {error}", throttle_duration_sec=1.0)
            return
        self._measured = pose
        self._freshness.mark("ee_pose", time.monotonic())

    def _on_joint_states(self, message: JointState) -> None:
        if FINGER_JOINT not in message.name:
            return
        finger = float(message.position[message.name.index(FINGER_JOINT)])
        if not math.isfinite(finger):
            # Left unmarked, so the freshness gate faults the run with STALE_POSE.
            return
        self._finger = finger
        self._freshness.mark("finger", time.monotonic())

    def _on_can_pose(self, message: PoseStamped) -> None:
        self._can = self._valid_detection(message, CAN_POSE_TOPIC)
        self._localisation.mark("can", time.monotonic())

    def _on_paper_pose(self, message: PoseStamped) -> None:
        self._paper = self._valid_detection(message, PAPER_POSE_TOPIC)
        self._localisation.mark("paper", time.monotonic())

    def _valid_detection(self, message: PoseStamped, topic: str) -> PoseStamped | None:
        """Return the detection, or None (read as NO_CAN / NO_PAPER) if it is not finite."""
        try:
            pose_from_message(message.pose)
        except ValueError as error:
            self.get_logger().error(f"Bad {topic}: {error}", throttle_duration_sec=1.0)
            return None
        return message

    def _on_hardware_state(self, message: HardwareState) -> None:
        changed = self._gate.update(message.state)
        if changed:
            self.get_logger().warning(f"Hardware state is now {message.state!r}")

    def _probe_hardware_state(self) -> None:
        """Ask the controller for `active`; it publishes only on change and boots active."""
        if self._probe is None:
            if self._state_client.service_is_ready():
                request = SetHardwareState.Request()
                request.state = ACTIVE_STATE
                self._probe = self._state_client.call_async(request)
            return
        if not self._probe.done() or self._probe.result() is None:
            return
        response = self._probe.result()
        self._gate.confirm_active(response.accepted)
        if response.accepted:
            self.get_logger().info("Hardware is active: goals enabled")
        else:
            self.get_logger().error(
                f"Hardware will not activate: {response.message}. Restart the loader stack."
            )
        # A refused probe latches the gate shut, so there is nothing to retry.
        self._probe_answered = True

    # ---- action ------------------------------------------------------------

    def _on_goal(self, goal) -> GoalResponse:
        # Any host on the domain can send a goal; refuse one the planner cannot use
        # before it claims the arm.
        if math.isnan(goal.speed_scale):
            self.get_logger().warning("Rejecting goal: speed_scale is NaN")
            return GoalResponse.REJECT
        target = goal.place_target
        overrides = (
            goal.grasp_dz,
            goal.closure_width,
            goal.yaw_offset,
            target.x,
            target.y,
            target.z,
        )
        if not all(math.isfinite(value) for value in overrides):
            self.get_logger().warning("Rejecting goal: an override field is not finite")
            return GoalResponse.REJECT
        busy, self._busy = self._busy, True
        if busy:
            self.get_logger().warning("Rejecting goal: a pick and place is already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    async def _execute(self, goal_handle) -> PickPlace.Result:
        # Release the busy flag however this ends, or one exception would leave the
        # server rejecting every later goal.
        try:
            return await self._execute_goal(goal_handle)
        finally:
            self._busy = False

    async def _execute_goal(self, goal_handle) -> PickPlace.Result:
        started = time.monotonic()
        goal = goal_handle.request
        result = PickPlace.Result()
        self._publish_phase(goal_handle, Phase.LOCALISE)

        needed = required_detections(
            home_only=goal.home_only, place_target_set=goal.place_target_set
        )
        code = self._localise_error(started, dry_run=goal.dry_run, needed=needed)
        can, paper = self._can, self._paper
        if can is not None:
            result.can_pose = can
        if paper is not None:
            result.paper_pose = paper

        task = None
        if code is None:
            overrides = GoalOverrides()
            if goal.home_only:
                task = PickPlaceTask.home_only(
                    self._config,
                    self._limiter,
                    speed_scale=goal.speed_scale,
                    dry_run=goal.dry_run,
                )
            else:
                overrides = goal_overrides(
                    goal.grasp_dz,
                    goal.closure_width,
                    goal.yaw_offset,
                    max_grasp_dz_m=self._max_grasp_dz_m,
                    max_yaw_offset_rad=self._max_yaw_offset_rad,
                    gripper_range_m=(
                        self._limiter.limits.gripper_min_m,
                        self._limiter.limits.gripper_max_m,
                    ),
                    max_closure_m=self._max_closure_m,
                )
                if goal.place_target_set:
                    target = goal.place_target
                    place = place_target_in_workspace(
                        self._limiter.limits, (target.x, target.y, target.z)
                    )
                else:
                    place = _xyz(paper)
                task = PickPlaceTask(
                    self._config,
                    self._limiter,
                    _xyz(can),
                    place,
                    overrides=overrides,
                    speed_scale=goal.speed_scale,
                    dry_run=goal.dry_run,
                )
            if goal.dry_run:
                self._log_plan(task, None if goal.home_only else overrides)
            if not task.done:
                await self._run(goal_handle, task)
            code = task.result

        result.error_code = int(code)
        result.duration_s = time.monotonic() - started
        if code is ErrorCode.SUCCESS:
            goal_handle.succeed()
        elif code is ErrorCode.CANCELLED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        phase = task.phase.name if task is not None else Phase.LOCALISE.name
        self.get_logger().info(f"Pick and place finished: {code.name} in {phase}")
        return result

    def _localise_error(
        self, now: float, *, dry_run: bool, needed: tuple[str, ...]
    ) -> ErrorCode | None:
        """Pre-flight checks that need no plan.

        Args:
            needed: The detections this goal uses, from `required_detections`.
        """
        if not dry_run:
            if not self._gate.allows_commands:
                return ErrorCode.HARDWARE_NOT_ACTIVE
            if self._measured is None or self._freshness.stale(ARM_INPUTS, now):
                return ErrorCode.STALE_POSE
        if "can" in needed and self._can is None:
            return ErrorCode.NO_CAN
        if "paper" in needed and self._paper is None:
            return ErrorCode.NO_PAPER
        if self._localisation.stale(needed, now):
            return ErrorCode.STALE_POSE
        return None

    async def _run(self, goal_handle, task: PickPlaceTask) -> None:
        """Hand the task to the control timer and wait until it finishes."""
        self._finished = Future()
        self._commanded = None
        self._fed_back = Phase.LOCALISE
        self._goal_handle = goal_handle
        self._task = task
        await self._finished
        self._goal_handle = None

    def _publish_phase(self, goal_handle, phase: Phase) -> None:
        feedback = PickPlace.Feedback()
        feedback.phase = int(phase)
        goal_handle.publish_feedback(feedback)

    def _log_plan(self, task: PickPlaceTask, overrides: GoalOverrides | None) -> None:
        """Log the planned waypoints; `overrides` is None for a home-only goal."""
        v_max, w_max = task.speed_limits
        header = f"Dry run plan (v_max={v_max:.3f} m/s, w_max={w_max:.2f} rad/s"
        if overrides is not None:
            closure = "configured" if overrides.closure_m is None else f"{overrides.closure_m:.4f}"
            header += (
                f", grasp_dz={overrides.grasp_dz_m:+.3f} m, closure={closure}, "
                f"yaw_offset={overrides.yaw_offset_rad:+.3f} rad"
            )
        lines = [header + "):"]
        for step in task.steps:
            x, y, z = step.target.position
            qx, qy, qz, qw = step.target.quat_xyzw
            lines.append(
                f"  {step.phase.name:<9} xyz=({x:.3f}, {y:.3f}, {z:.3f}) "
                f"q=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}) "
                f"gripper={step.gripper:.3f} dwell={step.dwell_s:.1f}s"
            )
        self.get_logger().info("\n".join(lines))

    # ---- control loop ------------------------------------------------------

    def _on_tick(self) -> None:
        if not self._gate.allows_commands and not self._probe_answered:
            self._probe_hardware_state()

        now = time.monotonic()
        task, goal_handle = self._task, self._goal_handle
        if task is None:
            return
        if goal_handle is not None and goal_handle.is_cancel_requested:
            task.cancel()
        if not self._gate.allows_commands:
            task.fault(ErrorCode.HARDWARE_NOT_ACTIVE)
        stale = self._freshness.stale(ARM_INPUTS, now)
        if self._measured is None and "ee_pose" not in stale:
            # The last sample was rejected as non-finite; it still counts as fresh.
            stale = (*stale, "ee_pose")
        if stale:
            task.fault(ErrorCode.STALE_POSE)
        if self._commanded is None:
            # Seeded on the tick the task echoes this same pose, so the
            # first clamp is against where the arm really is.
            self._commanded = self._measured
        command = task.tick(now, self._measured, self._finger or 0.0)
        if command is not None:
            target = command.pose
            if task.phase is Phase.HOLD:
                # Hold what was actually published, which the clamp may
                # have shortened, so stopping never moves the arm.
                target = self._commanded
            pose, report = self._limiter.clamp(target, self._commanded)
            gripper, _ = self._limiter.clamp_gripper(command.gripper)
            self._commanded = pose
        if stale:
            self.get_logger().error(f"Holding: stale inputs {', '.join(stale)}")
        if command is not None:
            if report.any_clamped:
                self.get_logger().warning(f"Clamped command: {report}", throttle_duration_sec=1.0)
            self._publisher.publish(self._build_message(pose, gripper))
        if task.phase is not self._fed_back:
            self._fed_back = task.phase
            self._publish_phase(goal_handle, task.phase)
        if task.done:
            self._task = None
            self._finished.set_result(task.result)

    def _build_message(self, pose: Pose, gripper: float) -> CommandedEEPose:
        message = CommandedEEPose()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        x, y, z = (float(v) for v in pose.position)
        qx, qy, qz, qw = (float(v) for v in pose.quat_xyzw)
        message.pose.position.x, message.pose.position.y, message.pose.position.z = x, y, z
        orientation = message.pose.orientation
        orientation.x, orientation.y, orientation.z, orientation.w = qx, qy, qz, qw
        message.gripper = gripper
        return message


def _xyz(message: PoseStamped) -> tuple[float, float, float]:
    p = message.pose.position
    return p.x, p.y, p.z


def main(args=None) -> None:
    """Run the pick-place action server until interrupted."""
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Ctrl-C makes rclpy's signal handler shut the context down first, and
        # shutting it down twice raises.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
