"""Render the sim chest-camera background reference for one fm-sim backend.

perception_node subtracts an image of the table without the can. On the robot it is
captured live (``/classical/capture_background``); in sim it is rendered once per backend
from fm-sim's ``table_only`` scenario (``can_on_table`` without the can) and baked into
``backgrounds/``, so a sim run needs no capture step.

Runs inside an fm-sim image with fm-sim and this package sourced, from this directory's
package (the default output is ``../backgrounds``):

    python3 tools/render_background.py --backend mujoco

It starts ``run.launch.py`` with the ``idle`` controller, waits for chest frames,
lets the scene settle, writes the latest frame as ``<backend>_<robot>_chest.png`` and
shuts the run down. The frame is a decoded JPEG, as perception_node decodes live frames,
so the reference carries the same compression.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

CHEST_TOPIC = "/cam_chest/image_raw/compressed"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "backgrounds"
# fm-sim's backends. Isaac is accepted so a GPU host can render its reference.
BACKENDS = ("mujoco", "gazebo", "isaac")
# Same rule fm-sim applies to entry names; robot names part of the output file name.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# A 1920x1080 JPEG is well under 1 MiB; anything far larger is not a chest frame and
# is dropped before it reaches the decoder.
MAX_FRAME_BYTES = 16 * 1024 * 1024


def _name(value: str) -> str:
    if not NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {NAME.pattern}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", required=True, choices=BACKENDS, help="fm-sim backend")
    parser.add_argument("--robot", type=_name, default="openarm", help="fm-sim robot entry")
    parser.add_argument(
        "--scenario", type=_name, default="table_only", help="fm-sim scenario entry"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument(
        "--settle-s", type=float, default=20.0, help="wall seconds to wait after the first frame"
    )
    parser.add_argument(
        "--timeout-s", type=float, default=300.0, help="give up without a frame after this long"
    )
    return parser.parse_args(argv)


def latest_frame(settle_s: float, timeout_s: float) -> np.ndarray:
    """Spin until settle_s after the first chest frame and return the latest one, decoded."""
    rclpy.init()
    node = rclpy.create_node("render_background")
    latest: list[bytes] = []

    def keep(message: CompressedImage) -> None:
        if len(message.data) <= MAX_FRAME_BYTES:
            latest[:] = [bytes(message.data)]

    node.create_subscription(CompressedImage, CHEST_TOPIC, keep, qos_profile_sensor_data)
    started = time.monotonic()
    first_s = None
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.5)
            now = time.monotonic()
            if latest and first_s is None:
                first_s = now
                node.get_logger().info(f"first chest frame; settling {settle_s:.0f} s")
            if first_s is not None and now - first_s >= settle_s:
                break
            if first_s is None and now - started > timeout_s:
                raise TimeoutError(f"no frame on {CHEST_TOPIC} in {timeout_s:.0f} s")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    frame = cv2.imdecode(np.frombuffer(latest[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("the latest chest frame did not decode")
    return frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out / f"{args.backend}_{args.robot}_chest.png"
    command = [
        "ros2",
        "launch",
        "fm_sim_bringup",
        "run.launch.py",
        f"backend:={args.backend}",
        f"robot:={args.robot}",
        f"scenario:={args.scenario}",
        "controller:=idle",
    ]
    # Its own process group, so shutdown reaches every node the launch started.
    run = subprocess.Popen(command, start_new_session=True)
    try:
        frame = latest_frame(args.settle_s, args.timeout_s)
    finally:
        os.killpg(run.pid, signal.SIGINT)
        try:
            run.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(run.pid, signal.SIGKILL)
    args.out.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"could not write {out}")
    print(f"wrote {out} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
