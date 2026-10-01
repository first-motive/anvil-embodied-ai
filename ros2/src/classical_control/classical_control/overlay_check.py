"""Draw the commanded TCP and a table-plane grid onto a live chest frame to check the extrinsic.

The look-then-move baseline is only as good as the chest camera's extrinsic, and a wrong
extrinsic is invisible until the gripper misses. This one-shot tool grabs one live chest
frame and the current right-arm TCP, projects the TCP and a grid of 5 cm lines on the table
plane through the camera yaml, and writes the annotated image. If the cross sits on the
gripper and the grid lies flat on the table, the extrinsic is right.

The projection and drawing are pure functions; ROS and anvil_msgs load only in ``main``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from classical_control.camera_model import FisheyeCamera
from classical_control.eval_offline import (
    CHEST_TOPIC,
    add_parent_to_world_argument,
    camera_in_world,
    project_world_points,
)
from classical_control.localise import ChestCameraConfig

EE_POSE_TOPIC = "/ee_pose_right"
GRID_COLOR = (0, 255, 255)  # BGR yellow
TCP_COLOR = (255, 0, 255)  # BGR magenta
# The fisheye bends straight world lines, so each line is drawn through this many samples.
_LINE_SAMPLES = 60


def table_grid(
    center_xy: Sequence[float], half_extent: float, spacing: float, table_z: float
) -> list[np.ndarray]:
    """Build a square grid of world-frame lines on the table plane.

    Args:
        center_xy: Grid centre (x, y) in world, metres.
        half_extent: Half the grid's side length, metres.
        spacing: Distance between neighbouring lines, metres.
        table_z: Table height in world, metres.

    Returns:
        Polylines, each an (_LINE_SAMPLES, 3) array of world points: lines of constant x
        first, then lines of constant y.
    """
    cx, cy = center_xy
    count = int(np.floor(half_extent / spacing + 1e-9))
    offsets = np.arange(-count, count + 1) * spacing
    sweep = np.linspace(-half_extent, half_extent, _LINE_SAMPLES)
    z = np.full(_LINE_SAMPLES, table_z)
    lines = [np.column_stack([np.full(_LINE_SAMPLES, cx + o), cy + sweep, z]) for o in offsets]
    lines += [np.column_stack([cx + sweep, np.full(_LINE_SAMPLES, cy + o), z]) for o in offsets]
    return lines


def draw_overlay(
    image: np.ndarray,
    camera: FisheyeCamera,
    T_world_optical: np.ndarray,
    tcp_xyz: Sequence[float],
    grid_lines: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the grid and a TCP marker onto a copy of the image.

    Args:
        image: BGR uint8 frame; ``camera`` must match its resolution.
        camera: Camera intrinsics.
        T_world_optical: 4x4 transform mapping optical-frame points into ``world``.
        tcp_xyz: TCP position in world, metres.
        grid_lines: World-frame polylines from `table_grid`.

    Returns:
        ``(annotated, tcp_pixel)``: the annotated copy, and the TCP's (u, v) pixel (NaN if
        the TCP is behind the camera).
    """
    out = image.copy()
    thickness = max(1, round(out.shape[1] / 960))
    for line in grid_lines:
        pixels = project_world_points(camera, T_world_optical, line)
        # Points behind the camera break a line into separate visible runs.
        visible = np.isfinite(pixels).all(axis=1)
        for run in np.split(pixels, np.flatnonzero(np.diff(visible.astype(int))) + 1):
            if len(run) > 1 and np.isfinite(run).all():
                points = np.round(run).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [points], False, GRID_COLOR, thickness, cv2.LINE_AA)

    tcp_pixel = project_world_points(camera, T_world_optical, np.asarray(tcp_xyz))[0]
    if np.isfinite(tcp_pixel).all():
        center = (int(round(tcp_pixel[0])), int(round(tcp_pixel[1])))
        size = 12 * thickness
        cv2.drawMarker(out, center, TCP_COLOR, cv2.MARKER_CROSS, size * 2, thickness * 2)
        cv2.circle(out, center, size, TCP_COLOR, thickness, cv2.LINE_AA)
    return out, tcp_pixel


def main(argv: Sequence[str] | None = None) -> None:
    """Grab one live frame and TCP pose, draw the overlay, write the jpg."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-yaml", type=Path, required=True, help="chest camera yaml")
    parser.add_argument("--output", type=Path, default=Path("overlay_check.jpg"))
    parser.add_argument("--table-z", type=float, default=0.207, help="table height in world, m")
    parser.add_argument("--spacing", type=float, default=0.05, help="grid spacing, m")
    parser.add_argument(
        "--half-extent", type=float, default=0.3, help="grid half-width around the TCP, m"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for data")
    add_parent_to_world_argument(parser)
    args = parser.parse_args(argv)

    import rclpy
    from anvil_msgs.msg import CommandedEEPose
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    config = ChestCameraConfig.from_yaml(args.camera_yaml)
    T_world_optical = camera_in_world(config, args.parent_to_world)
    received: dict[str, object] = {}
    rclpy.init()
    node = rclpy.create_node("overlay_check")
    try:
        # Best-effort QoS matches any publisher, reliable or not.
        node.create_subscription(
            CompressedImage,
            CHEST_TOPIC,
            lambda msg: received.setdefault("image", msg),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            CommandedEEPose,
            EE_POSE_TOPIC,
            lambda msg: received.setdefault("pose", msg),
            qos_profile_sensor_data,
        )
        deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
        while len(received) < 2 and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(received) < 2:
            missing = {"image": CHEST_TOPIC, "pose": EE_POSE_TOPIC}.keys() - received.keys()
            raise SystemExit(f"timed out waiting for {sorted(missing)}")

        image_msg = received["image"]
        frame = cv2.imdecode(np.frombuffer(bytes(image_msg.data), np.uint8), cv2.IMREAD_COLOR)
        position = received["pose"].pose.position
        tcp = (position.x, position.y, position.z)
        height, width = frame.shape[:2]
        camera = config.camera.scaled(width, height)
        lines = table_grid(tcp[:2], args.half_extent, args.spacing, args.table_z)
        annotated, tcp_pixel = draw_overlay(frame, camera, T_world_optical, tcp, lines)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), annotated)
        node.get_logger().info(
            f"TCP ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}) -> pixel "
            f"({tcp_pixel[0]:.0f}, {tcp_pixel[1]:.0f}); wrote {args.output}"
        )
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
