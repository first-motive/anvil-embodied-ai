"""Unattended pick and random re-place loop that records one MCAP episode per cycle.

    /classical/can_pose ─▶ collect_node ──goal──▶ task_node /classical/pick_place
                              │  recorder: ros2 bag record per cycle
                              ▼
    <out-root>/<run>/  metadata.json, 0001/0001_0.mcap + metadata.json, ..., summary.json

The node is wiring only. `collect_loop` samples each cycle's place point and grasp
knobs and decides when to stop; `recorder` runs the bag. Per cycle:

    before_cycle (disk, cycles, time) → fresh can pose → sample knobs
      → start bag → PickPlace goal (place_target + knobs) → stop bag → metadata.json
      → after_attempt: CONTINUE | RETRY (home, same knobs) | STOP

However the run ends (a stop rule, Ctrl-C, an exception), the arm is sent HOME with a
home-only goal and `summary.json` is written. Ctrl-C is handled here rather than by
rclpy, whose default handler would shut the context down before that HOME goal.

Everything runs in the main thread: each wait spins the node until a future completes
or the stop flag is set, so a SIGINT cancels the running goal within one spin period.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from classical_control_msgs.action import PickPlace
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions

from .collect_loop import (
    OPERATOR_STOP,
    Decision,
    KnobRanges,
    Knobs,
    LoopConfig,
    StopLimits,
    StopPolicy,
    episode_metadata,
    episode_name,
    forced_stop_reason,
    iso_utc,
    load_loop_config,
    run_id,
    run_metadata,
    sample_knobs,
    summary_metadata,
)
from .objects import ObjectConfig, load_object_config
from .recorder import (
    EpisodeRecorder,
    RecordingConfig,
    StopResult,
    free_bytes,
    load_recording_config,
)
from .task_machine import ErrorCode

ACTION_NAME = "/classical/pick_place"
CAN_POSE_TOPIC = "/classical/can_pose"
PERCEPTION_PARAMETERS_SERVICE = "/perception_node/set_parameters"
#: How long one spin waits for work, in seconds; bounds the reaction to Ctrl-C.
SPIN_PERIOD_S = 0.1
#: How long to wait for the action server and the perception parameter service.
CONNECT_TIMEOUT_S = 10.0
#: Run directory files the episodes sit beside.
RUN_METADATA_FILE = "metadata.json"
SUMMARY_FILE = "summary.json"
#: Longest the final HOME move may take, in seconds.
HOME_TIMEOUT_S = 120.0
#: Longest a cancelled goal may take to report its end, in seconds.
CANCEL_TIMEOUT_S = 30.0
#: A run id names one directory under --out-root.
RUN_ID_PATTERN = re.compile(r"[0-9A-Za-z_-]+")


class CollectNode(Node):
    """Action client and can-pose listener the loop drives; holds no loop state."""

    def __init__(self) -> None:
        super().__init__("classical_collect")
        self.stop_requested = False
        self._can_xy: tuple[float, float] | None = None
        self._can_received_at = -np.inf
        self._client = ActionClient(self, PickPlace, ACTION_NAME)
        self._parameters = self.create_client(SetParameters, PERCEPTION_PARAMETERS_SERVICE)
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, CAN_POSE_TOPIC, self._on_can_pose, sensor)

    def _on_can_pose(self, message: PoseStamped) -> None:
        position = message.pose.position
        if np.isfinite([position.x, position.y]).all():
            self._can_xy = (position.x, position.y)
            self._can_received_at = time.monotonic()

    def wait_until(self, future, timeout_s: float) -> bool:
        """Spin until `future` is done; False on timeout or a stop request."""
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if self.stop_requested or time.monotonic() > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=SPIN_PERIOD_S)
        return True

    def pause(self, seconds: float) -> None:
        """Keep spinning for `seconds`, so subscriptions stay current."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=SPIN_PERIOD_S)

    def connect(self) -> None:
        """Wait for the task node's action server.

        Raises:
            RuntimeError: If it does not appear in time.
        """
        if not self._client.wait_for_server(timeout_sec=CONNECT_TIMEOUT_S):
            raise RuntimeError(
                f"no action server on {ACTION_NAME}: is `run_classical.sh up` running?"
            )

    def set_can_height(self, height_m: float) -> None:
        """Tell perception the object's height, which its ray-plane intersection uses.

        Raises:
            RuntimeError: If the service is missing or refuses the value.
        """
        if not self._parameters.wait_for_service(timeout_sec=CONNECT_TIMEOUT_S):
            raise RuntimeError(f"no {PERCEPTION_PARAMETERS_SERVICE}: is perception_node up?")
        value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=height_m)
        request = SetParameters.Request(parameters=[Parameter(name="can_height", value=value)])
        future = self._parameters.call_async(request)
        if not self.wait_until(future, CONNECT_TIMEOUT_S) or future.result() is None:
            raise RuntimeError("perception_node did not answer the can_height change")
        result = future.result().results[0]
        if not result.successful:
            raise RuntimeError(f"perception_node refused can_height: {result.reason}")

    def fresh_can_xy(self, since: float, timeout_s: float) -> tuple[float, float] | None:
        """The first can detection received after monotonic time `since`, or None."""
        deadline = time.monotonic() + timeout_s
        while self._can_received_at <= since:
            if self.stop_requested or time.monotonic() > deadline:
                return None
            rclpy.spin_once(self, timeout_sec=SPIN_PERIOD_S)
        return self._can_xy

    def run_goal(self, goal: PickPlace.Goal, timeout_s: float) -> ErrorCode | None:
        """Send a goal and wait until it has ended, so the arm is never left moving.

        A pick-place goal is cancelled on a stop request or after `timeout_s`. A home-only
        goal ignores stop requests: it is what makes the arm safe, so a second Ctrl-C must
        not cancel it.

        Returns:
            The result code, or None if the goal was rejected or its end was never heard.
        """
        interruptible = not goal.home_only
        sent = self._client.send_goal_async(goal)
        # Never abandon an unanswered goal on a stop request: it could be accepted a
        # moment later and run with nobody watching it.
        if not self._spin_until(sent, CONNECT_TIMEOUT_S):
            # Still unanswered: cancel it the moment it is accepted, on a later spin.
            sent.add_done_callback(_cancel_if_accepted)
            self.get_logger().error("goal not answered in time; it is cancelled if accepted")
            return None
        if not sent.result().accepted:
            return None
        handle = sent.result()
        result = handle.get_result_async()
        deadline = time.monotonic() + timeout_s
        while not result.done():
            now = time.monotonic()
            if now > deadline or (interruptible and self.stop_requested):
                break
            rclpy.spin_once(self, timeout_sec=SPIN_PERIOD_S)
        if not result.done():
            # The task drops into HOLD and answers CANCELLED.
            handle.cancel_goal_async()
            if not self._spin_until(result, CANCEL_TIMEOUT_S):
                self.get_logger().error("goal did not end after cancel; check the arm")
                return None
        return ErrorCode(result.result().result.error_code)

    def _spin_until(self, future, timeout_s: float) -> bool:
        """Spin until `future` is done, ignoring stop requests; False on timeout."""
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=SPIN_PERIOD_S)
        return future.done()


def _cancel_if_accepted(sent) -> None:
    handle = sent.result()
    if handle is not None and handle.accepted:
        handle.cancel_goal_async()


def pick_place_goal(knobs: Knobs, speed_scale: float, dry_run: bool) -> PickPlace.Goal:
    """The goal for one attempt: place at the sampled point with the sampled knobs."""
    goal = PickPlace.Goal(dry_run=dry_run, speed_scale=speed_scale)
    goal.place_target_set = True
    goal.place_target.x, goal.place_target.y = knobs.place_xy
    goal.grasp_dz = knobs.grasp_dz_m
    goal.closure_width = knobs.closure_m
    goal.yaw_offset = knobs.yaw_offset_rad
    return goal


def home_goal(speed_scale: float, dry_run: bool) -> PickPlace.Goal:
    """A goal that only drives the arm HOME with the gripper open."""
    return PickPlace.Goal(dry_run=dry_run, speed_scale=speed_scale, home_only=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as indented JSON, via a temporary file so a reader never sees half."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    partial.replace(path)


def run_name(value: str) -> str:
    """argparse type for --run-id: one plain directory name, never a path."""
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"run id must match {RUN_ID_PATTERN.pattern}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line for one collection run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", required=True, help="config/objects/<name>.yaml to collect")
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--collect-config", type=Path, required=True)
    parser.add_argument("--perception-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id", type=run_name, help="run directory; default UTC start time")
    parser.add_argument("--hours", type=float, help="stop starting cycles after this long")
    parser.add_argument("--cycles", type=int, help="stop after this many successful cycles")
    parser.add_argument("--speed", type=float, default=0.5, help="PickPlace speed_scale")
    parser.add_argument("--no-record", action="store_true", help="write metadata, no MCAP")
    parser.add_argument("--dry-run", action="store_true", help="send dry_run goals; nothing moves")
    parser.add_argument("--seed", type=int, help="knob sampler seed, for a repeatable run")
    parser.add_argument("--git-sha", help="commit the run was started from")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class RunSettings:
    """Everything a run reads from config and the command line, loaded before it starts."""

    obj: ObjectConfig
    recording: RecordingConfig
    loop: LoopConfig
    ranges: KnobRanges
    limits: StopLimits

    @classmethod
    def load(cls, args: argparse.Namespace) -> RunSettings:
        obj = load_object_config(args.objects_dir, args.object)
        recording = load_recording_config(args.collect_config)
        loop = load_loop_config(args.collect_config, args.perception_config)
        limits = StopLimits(
            max_grasp_retries=loop.max_grasp_retries,
            max_consecutive_failures=loop.max_consecutive_failures,
            min_free_bytes=recording.min_free_bytes,
            max_cycles=args.cycles,
            max_duration_s=None if args.hours is None else args.hours * 3600.0,
        )
        ranges = KnobRanges(obj.grasp_dz_m, obj.closure_m, obj.yaw_offset_rad)
        return cls(obj, recording, loop, ranges, limits)


def record_attempt(
    node: CollectNode,
    recorder: EpisodeRecorder,
    settings: RunSettings,
    episode_dir: Path,
    goal: PickPlace.Goal,
) -> tuple[ErrorCode | None, StopResult | None, float]:
    """Run one goal inside its own bag; the bag is stopped however the goal ends."""
    started = time.monotonic()
    recorder.start(episode_dir)
    try:
        node.pause(settings.recording.settle_s if recorder.enabled else 0.0)
        code = node.run_goal(goal, settings.loop.goal_timeout_s)
    finally:
        stopped = recorder.stop()
    return code, stopped, time.monotonic() - started


def collect(
    node: CollectNode,
    args: argparse.Namespace,
    settings: RunSettings,
    policy: StopPolicy,
    run_dir: Path,
) -> None:
    """Run cycles until `policy` records a stop reason."""
    loop = settings.loop
    recorder = EpisodeRecorder(settings.recording, enabled=not args.no_record)
    rng = np.random.default_rng(args.seed)

    node.connect()
    node.set_can_height(settings.obj.height_m)
    write_json(
        run_dir / RUN_METADATA_FILE,
        run_metadata(
            object_name=settings.obj.name,
            object_config=settings.obj.to_dict(),
            region=loop.region,
            knob_ranges=settings.ranges,
            limits=settings.limits,
            speed_scale=args.speed,
            record=not args.no_record,
            git_sha=args.git_sha,
            started_at=iso_utc(datetime.now().astimezone()),
        ),
    )

    episode, attempt, knobs, can_xy = 0, 0, None, None
    while policy.before_cycle(time.monotonic(), free_bytes(run_dir)) is None:
        if node.stop_requested:
            policy.request_stop()
            break
        if knobs is None:
            can_xy = node.fresh_can_xy(time.monotonic(), loop.can_pose_timeout_s)
            if can_xy is None:
                policy.request_stop(OPERATOR_STOP if node.stop_requested else "error:NO_CAN")
                break
            knobs = sample_knobs(rng, loop.region, can_xy, settings.ranges)

        episode += 1
        episode_dir = run_dir / episode_name(episode)
        started_at = iso_utc(datetime.now().astimezone())
        goal = pick_place_goal(knobs, args.speed, args.dry_run)
        code, stopped, duration_s = record_attempt(node, recorder, settings, episode_dir, goal)
        note = None if code is not None else "no result from the task node"
        if stopped is not None and not stopped.clean:
            note = f"recorder exit {stopped.returncode}, killed={stopped.killed}"
        write_json(
            episode_dir / "metadata.json",
            episode_metadata(
                code=ErrorCode.CANCELLED if code is None else code,
                attempt=attempt,
                can_xy=can_xy,
                knobs=knobs,
                started_at=started_at,
                duration_s=duration_s,
                note=note,
            ),
        )
        node.get_logger().info(
            f"episode {episode_name(episode)}: {code.name if code is not None else 'no result'}"
        )
        forced = forced_stop_reason(
            code,
            recorder_died=stopped is not None and stopped.died_early,
            stop_requested=node.stop_requested,
        )
        if forced is not None:
            policy.request_stop(forced)
        decision = Decision.STOP if code is None else policy.after_attempt(code)
        if decision is Decision.STOP:
            break
        if decision is Decision.RETRY:
            # The arm is holding over the can; clear the camera's view before trying again.
            home = node.run_goal(home_goal(args.speed, args.dry_run), HOME_TIMEOUT_S)
            if home is not ErrorCode.SUCCESS:
                policy.request_stop("error:HOME")
                break
            attempt += 1
        else:
            attempt, knobs = 0, None


def main(argv: list[str] | None = None) -> None:
    """Run one collection, then drive HOME and write the summary however it ends."""
    args = parse_args(argv)
    settings = RunSettings.load(args)
    run_dir = args.out_root / (args.run_id or run_id(datetime.now().astimezone()))
    run_dir.mkdir(parents=True)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CollectNode()

    def request_stop(_signum, _frame) -> None:
        node.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    policy = StopPolicy(settings.limits, time.monotonic())
    try:
        collect(node, args, settings, policy, run_dir)
    except Exception as error:
        policy.request_stop(f"error:{type(error).__name__}")
        node.get_logger().error(f"collection failed: {error}")
        raise
    finally:
        node.get_logger().info("driving HOME")
        home = node.run_goal(home_goal(args.speed, args.dry_run), HOME_TIMEOUT_S)
        if home is not ErrorCode.SUCCESS:
            node.get_logger().error(f"HOME goal ended {home}; check the arm")
        summary = summary_metadata(
            policy,
            episodes=sum(1 for path in run_dir.iterdir() if path.is_dir()),
            now=time.monotonic(),
            ended_at=iso_utc(datetime.now().astimezone()),
        )
        summary["home"] = home.name if home is not None else None
        write_json(run_dir / SUMMARY_FILE, summary)
        node.get_logger().info(f"run {run_dir.name} stopped: {policy.stop_reason}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
