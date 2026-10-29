"""task_node on the ``pose`` interface, run in-process over real topics.

Drives the shipped node the way an fm-sim run does: ``/clock``, TF for the driven
arm's TCP and ``/joint_states`` for its finger, then one home-only goal. Needs a
sourced Jazzy workspace with classical_control and its action built.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tf2_ros")
pytest.importorskip("launch_ros")
rclpy = pytest.importorskip("rclpy")

from anvil_robot_profiles import DRIVEN_ARM, load_profile  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from classical_control.task_machine import ErrorCode, Phase  # noqa: E402
from classical_control.task_node import COMMAND_TOPIC, PHASE_TOPIC, PickPlaceNode  # noqa: E402
from classical_control_msgs.action import PickPlace  # noqa: E402
from geometry_msgs.msg import PoseStamped, TransformStamped  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

PROFILE = load_profile("fm_openarm_v2")
ARM = PROFILE.arms[DRIVEN_ARM]
MINED = Path(__file__).resolve().parent.parent / "config" / "sim" / "mined_params.yaml"
# Near the configured home TCP, inside the workspace box.
START_XYZ = (0.40, -0.18, 0.50)
START_QUAT = (0.672, 0.064, 0.736, 0.055)


@pytest.fixture
def task_node():
    """Start rclpy with the node's parameters as global ``--ros-args``, then build it."""

    def start(**overrides):
        parameters = {
            "mined_params_file": str(MINED),
            "robot_profile": PROFILE.name,
            "ee_interface": "pose",
            "use_sim_time": "true",
            **overrides,
        }
        args = ["--ros-args"]
        for name, value in parameters.items():
            args += ["-p", f"{name}:={value}"]
        rclpy.init(args=args)
        return PickPlaceNode()

    yield start
    if rclpy.ok():
        rclpy.shutdown()


def test_refuses_an_unknown_ee_interface(task_node):
    with pytest.raises(ValueError, match="ee_interface must be one of loader, pose"):
        task_node(ee_interface="servo")


def test_pose_interface_refuses_wall_time(task_node):
    with pytest.raises(ValueError, match="pose' has no hardware gate"):
        task_node(use_sim_time="false")


class _Sim:
    """Stands in for fm-sim: clock, the arm's TF and joint states, a command listener."""

    def __init__(self, tf_mode="fresh"):
        # "fresh": a new TF stamp each step; "frozen": one stamp repeated; "none": no TF.
        self.tf_mode = tf_mode
        self.joint_states_seen = False
        self.node = rclpy.create_node("fake_sim")
        self.clock = self.node.create_publisher(Clock, "/clock", 10)
        self.tf = self.node.create_publisher(TFMessage, "/tf", 10)
        self.joints = self.node.create_publisher(JointState, "/joint_states", 10)
        self.commands: list[PoseStamped] = []
        self.node.create_subscription(PoseStamped, COMMAND_TOPIC, self.commands.append, 10)
        self.phases: list[str] = []
        # Deeper than the publisher's 1, so every phase of a fast run is kept here.
        latched = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.node.create_subscription(
            String, PHASE_TOPIC, lambda message: self.phases.append(message.data), latched
        )
        self.goals = ActionClient(self.node, PickPlace, "/classical/pick_place")
        self.nanoseconds = 1_000_000_000
        self.first_stamp = Time(sec=1, nanosec=0)

    def step(self):
        self.nanoseconds += 10_000_000
        stamp = Time(sec=self.nanoseconds // 10**9, nanosec=self.nanoseconds % 10**9)
        self.clock.publish(Clock(clock=stamp))
        if self.tf_mode != "none":
            self.tf.publish(self._transform(stamp if self.tf_mode == "fresh" else self.first_stamp))
        joints = JointState(name=[ARM.gripper_joint], position=[0.04])
        joints.header.stamp = stamp
        self.joints.publish(joints)

    def _transform(self, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = PROFILE.base_frame
        transform.child_frame_id = ARM.ee_frame
        t = transform.transform.translation
        t.x, t.y, t.z = START_XYZ
        r = transform.transform.rotation
        r.x, r.y, r.z, r.w = START_QUAT
        return TFMessage(transforms=[transform])


def _run_home_goal(node, sim, until, age_s=0.0):
    """Send one home-only goal once the node has a measured pose (or after 1 s without one).

    ``age_s`` keeps spinning that long after the pose arrives, so a sample that is not
    renewed ages past ``pose_max_age_s`` (0.2 s). Returns the goal's result future; spins
    until ``until(result_future)`` or a timeout.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(sim.node)

    def spin_until(done, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not done() and time.monotonic() < deadline:
            sim.step()
            executor.spin_once(timeout_sec=0.01)
        return done()

    try:
        assert spin_until(sim.goals.server_is_ready, 5.0)
        # Wait for TF to reach the node; the no-TF case waits out the full second.
        spin_until(lambda: node._measured is not None, 1.0)
        spin_until(lambda: False, age_s)
        sent = sim.goals.send_goal_async(PickPlace.Goal(home_only=True))
        assert spin_until(sent.done, 5.0)
        assert sent.result().accepted
        result = sent.result().get_result_async()
        spin_until(lambda: until(result), 10.0)
        return result
    finally:
        executor.shutdown()
        node.destroy_node()
        sim.node.destroy_node()


def test_home_goal_streams_pose_stamped_in_the_profile_base_frame(task_node):
    node = task_node()
    sim = _Sim()
    _run_home_goal(node, sim, lambda _: len(sim.commands) >= 5)

    assert len(sim.commands) >= 5
    first = sim.commands[0]
    assert first.header.frame_id == PROFILE.base_frame
    # The first command repeats the measured TCP read from TF: no jump at goal start.
    p = first.pose.position
    assert (p.x, p.y, p.z) == pytest.approx(START_XYZ)
    # Stamped from the sim clock the fake publishes, which starts at 1 s and runs
    # 10 ms per step, not from wall time.
    stamp_s = first.header.stamp.sec + first.header.stamp.nanosec * 1e-9
    assert 1.0 < stamp_s <= sim.nanoseconds * 1e-9


@pytest.mark.parametrize("tf_mode", ["none", "frozen"])
def test_missing_or_frozen_tf_fails_the_goal_with_stale_pose(task_node, tf_mode):
    # "none": every lookup raises, so no measured pose. "frozen": one stamp repeated
    # counts as a single sample, which ages out like a silent /ee_pose_right.
    node = task_node()
    sim = _Sim(tf_mode=tf_mode)
    result = _run_home_goal(node, sim, lambda future: future.done(), age_s=0.5)

    assert result.done()
    assert result.result().result.error_code == int(ErrorCode.STALE_POSE)


def test_phase_names_are_mirrored_latched_on_the_plain_phase_topic(task_node):
    node = task_node()
    qos = node._phase_publisher.qos_profile
    assert (qos.depth, qos.durability, qos.reliability) == (
        1,
        DurabilityPolicy.TRANSIENT_LOCAL,
        ReliabilityPolicy.RELIABLE,
    )
    sim = _Sim()
    result = _run_home_goal(node, sim, lambda future: future.done())

    assert result.result().result.error_code == int(ErrorCode.SUCCESS)
    # A home-only goal localises, then moves HOME; names match the descriptor's states.
    assert sim.phases == ["LOCALISE", "HOME"]


def test_a_late_subscriber_gets_the_last_phase_including_hold(task_node):
    # The sim runs faster than the wall-clock stale gate, so a live run cannot fault
    # mid-move here; every phase change, HOLD included, goes through _publish_phase.
    node = task_node()
    feedback = []
    node._publish_phase(SimpleNamespace(publish_feedback=feedback.append), Phase.HOLD)
    assert [message.phase for message in feedback] == [int(Phase.HOLD)]

    sim = _Sim()  # subscribes after the publish: only latching delivers it
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(sim.node)
    try:
        deadline = time.monotonic() + 5.0
        while not sim.phases and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown()
        node.destroy_node()
        sim.node.destroy_node()
    assert sim.phases == ["HOLD"]
