"""ROS node that localises the can and the paper on the table from the chest camera.

The classical look-then-move baseline needs the can and paper as poses in ``world``. This
node is the thin ROS shell around the pure modules that do the work:

* publishes the chest camera's fixed mounting as a static transform, since the robot
  description has no frame for it;
* decodes chest JPEG frames at a capped rate and runs `detector.detect` against a stored
  image of the empty table;
* maps the detections onto the table with `localise` and publishes
  ``/classical/can_pose`` and ``/classical/paper_pose`` (PoseStamped in ``world``);
* publishes a throttled debug overlay on ``/classical/debug/compressed``;
* serves ``/classical/capture_background`` (std_srvs/Trigger) to record the empty table.

A pose is only published when its object is seen, so a stale stamp is how downstream
nodes learn an object is missing.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener

from classical_control.camera_model import pose_to_matrix
from classical_control.detector import Detection, DetectorParams, detect
from classical_control.localise import (
    ChestCameraConfig,
    locate_can,
    locate_paper,
    yaw_to_quaternion,
)

_IMAGE_TOPIC = "/cam_chest/image_raw/compressed"
_CAN_POSE_TOPIC = "/classical/can_pose"
_PAPER_POSE_TOPIC = "/classical/paper_pose"
_DEBUG_TOPIC = "/classical/debug/compressed"
_CAPTURE_SERVICE = "/classical/capture_background"
_WARN_PERIOD_S = 5.0
# A background capture refuses frames older than this, so a stalled camera can't save an
# image of a table that has since changed.
_MAX_CAPTURE_AGE_S = 1.0


class PerceptionNode(Node):
    """Chest-camera can and paper localiser. See the module docstring."""

    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("camera_config_file", "")
        self.declare_parameter("background_file", "~/.ros/classical_control/chest_background.png")
        self.declare_parameter("process_width", 960)
        self.declare_parameter("process_height", 540)
        self.declare_parameter("process_rate_hz", 5.0)
        self.declare_parameter("debug_rate_hz", 1.0)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "cam_chest_optical")
        self.declare_parameter("table_z", 0.22)
        self.declare_parameter("can_height", 0.12)
        for field in dataclasses.fields(DetectorParams):
            self.declare_parameter(f"detector.{field.name}", field.default)

        config_path = self._param("camera_config_file") or str(
            Path(get_package_share_directory("classical_control")) / "config" / "camera_chest.yaml"
        )
        config = ChestCameraConfig.from_yaml(config_path)
        self._process_size = (self._param("process_width"), self._param("process_height"))
        self._camera = config.camera.scaled(*self._process_size)
        self._world_frame: str = self._param("world_frame")
        self._camera_frame: str = self._param("camera_frame")
        self._table_z: float = self._param("table_z")
        self._can_height: float = self._param("can_height")
        self._detector_params = DetectorParams.from_dict(
            {
                field.name: self._param(f"detector.{field.name}")
                for field in dataclasses.fields(DetectorParams)
            }
        )
        self._process_period_s = 1.0 / self._param("process_rate_hz")
        self._debug_period_s = 1.0 / self._param("debug_rate_hz")
        self._background_path = Path(self._param("background_file")).expanduser()

        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform(self._mounting_transform(config))
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._background = self._load_background()
        self._latest_msg: CompressedImage | None = None
        self._latest_received_s = 0.0
        self._last_processed_s = -np.inf
        self._last_debug_s = -np.inf

        self._can_pub = self.create_publisher(PoseStamped, _CAN_POSE_TOPIC, 10)
        self._paper_pub = self.create_publisher(PoseStamped, _PAPER_POSE_TOPIC, 10)
        self._debug_pub = self.create_publisher(CompressedImage, _DEBUG_TOPIC, 1)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self.create_subscription(CompressedImage, _IMAGE_TOPIC, self._on_image, camera_qos)
        self.create_service(Trigger, _CAPTURE_SERVICE, self._on_capture_background)

        self.get_logger().info(
            f"camera {config_path} mounted on {config.parent_frame}; processing at "
            f"{self._process_size[0]}x{self._process_size[1]}, "
            f"background {'loaded' if self._background is not None else 'missing'}"
        )

    def _param(self, name: str):
        """Return a declared parameter's value."""
        return self.get_parameter(name).value

    def _mounting_transform(self, config: ChestCameraConfig) -> TransformStamped:
        """Build the static parent -> optical transform from the camera config."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = config.parent_frame
        transform.child_frame_id = self._camera_frame
        t = transform.transform.translation
        t.x, t.y, t.z = config.translation_xyz
        r = transform.transform.rotation
        r.x, r.y, r.z, r.w = config.rotation_xyzw
        return transform

    def _load_background(self) -> np.ndarray | None:
        """Read the saved empty-table image at processing resolution, if there is one."""
        if not self._background_path.is_file():
            return None
        image = cv2.imread(str(self._background_path), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f"cannot read background {self._background_path}")
            return None
        return self._to_process_size(image)

    def _to_process_size(self, image: np.ndarray) -> np.ndarray:
        """Resize a frame to the detection resolution."""
        if (image.shape[1], image.shape[0]) == self._process_size:
            return image
        return cv2.resize(image, self._process_size, interpolation=cv2.INTER_AREA)

    def _on_image(self, msg: CompressedImage) -> None:
        """Keep the latest frame and, at most at process_rate_hz, localise from it."""
        now_s = time.monotonic()
        self._latest_msg = msg
        self._latest_received_s = now_s
        # Throttle before decoding: the JPEG decode is most of the per-frame cost.
        if now_s - self._last_processed_s < self._process_period_s:
            return
        self._last_processed_s = now_s

        logger = self.get_logger()
        if self._background is None:
            logger.warn(
                f"no background yet; call {_CAPTURE_SERVICE} with the table empty",
                throttle_duration_sec=_WARN_PERIOD_S,
            )
            return
        try:
            tf = self._tf_buffer.lookup_transform(self._world_frame, self._camera_frame, Time())
        except TransformException as error:
            logger.warn(
                f"waiting for {self._world_frame} <- {self._camera_frame}: {error}",
                throttle_duration_sec=_WARN_PERIOD_S,
            )
            return
        t, r = tf.transform.translation, tf.transform.rotation
        T_world_optical = pose_to_matrix([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])

        frame = self._decode(msg)
        if frame is None:
            return
        frame = self._to_process_size(frame)
        detection = detect(frame, self._background, self._detector_params)

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        can = locate_can(
            self._camera, T_world_optical, detection.can_pixel, self._table_z, self._can_height
        )
        if can is not None:
            self._can_pub.publish(_pose(stamp, self._world_frame, can, (0.0, 0.0, 0.0, 1.0)))
        paper = locate_paper(self._camera, T_world_optical, detection.paper_corners, self._table_z)
        if paper is not None:
            self._paper_pub.publish(
                _pose(stamp, self._world_frame, paper.position, yaw_to_quaternion(paper.yaw))
            )

        if now_s - self._last_debug_s >= self._debug_period_s:
            self._last_debug_s = now_s
            self._publish_debug(msg, frame, detection)

    def _decode(self, msg: CompressedImage) -> np.ndarray | None:
        """Decode a JPEG frame to BGR, logging and returning None on a corrupt frame."""
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn(
                "dropped an undecodable chest frame", throttle_duration_sec=_WARN_PERIOD_S
            )
        return frame

    def _publish_debug(self, msg: CompressedImage, frame: np.ndarray, detection: Detection) -> None:
        """Publish the frame with the foreground, paper quad, and can centroid drawn on."""
        overlay = frame.copy()
        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        if detection.paper_corners is not None:
            quad = np.round(detection.paper_corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [quad], True, (0, 255, 0), 2)
        if detection.can_pixel is not None:
            center = tuple(int(round(v)) for v in detection.can_pixel)
            cv2.circle(overlay, center, 6, (0, 0, 255), -1)
        ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        debug = CompressedImage()
        debug.header.stamp = msg.header.stamp
        debug.header.frame_id = self._camera_frame
        debug.format = "jpeg"
        debug.data = jpeg.tobytes()
        self._debug_pub.publish(debug)

    def _on_capture_background(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Store the most recent frame as the empty-table background and save it to disk."""
        # tradeoff: uses the latest received frame rather than waiting for the next one; a
        # single-threaded executor can't wait inside a callback, and the age check below
        # rules out a stale image.
        age_s = time.monotonic() - self._latest_received_s
        if self._latest_msg is None or age_s > _MAX_CAPTURE_AGE_S:
            response.success = False
            response.message = f"no chest frame in the last {_MAX_CAPTURE_AGE_S:.0f} s"
            return response
        frame = self._decode(self._latest_msg)
        if frame is None:
            response.success = False
            response.message = "latest chest frame could not be decoded"
            return response

        self._background_path.parent.mkdir(parents=True, exist_ok=True)
        # Saved at native resolution so a later change of process size can reuse it.
        if not cv2.imwrite(str(self._background_path), frame):
            response.success = False
            response.message = f"could not write {self._background_path}"
            return response
        self._background = self._to_process_size(frame)
        response.success = True
        response.message = f"background saved to {self._background_path}"
        self.get_logger().info(response.message)
        return response


def _pose(
    stamp: TimeMsg, frame_id: str, position: np.ndarray, quat_xyzw: tuple[float, ...]
) -> PoseStamped:
    """Build a PoseStamped from a stamp, frame, position, and (x, y, z, w) quaternion."""
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id
    p = pose.pose.position
    p.x, p.y, p.z = (float(v) for v in position)
    q = pose.pose.orientation
    q.x, q.y, q.z, q.w = (float(v) for v in quat_xyzw)
    return pose


def main(args: list[str] | None = None) -> None:
    """Run the perception node until shutdown."""
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
