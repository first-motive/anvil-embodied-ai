"""Pose math for poses expressed in another frame.

Poses are 7-vectors ``[x, y, z, qx, qy, qz, qw]`` (xyzw unit quaternion).
Feature vectors flatten them to ``[x, y, z, rx, ry, rz]`` with a rotation
vector.
"""

import numpy as np


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit xyzw quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by unit xyzw quaternion ``q``."""
    return quat_multiply(quat_multiply(q, np.array([*v, 0.0])), quat_conjugate(q))[:3]


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle, angle in [0, pi]) of a unit xyzw quaternion."""
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    sin_half = np.linalg.norm(q[:3])
    if sin_half < 1e-12:
        return 2.0 * q[:3]
    return q[:3] / sin_half * 2.0 * np.arctan2(sin_half, q[3])


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit xyzw quaternion of a rotation vector."""
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([*(r / 2.0), 1.0])
    return np.array([*(np.sin(angle / 2.0) * r / angle), np.cos(angle / 2.0)])


def pose_in_frame(pose: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Express a pose given in the parent frame relative to ``frame`` (both in the parent)."""
    q_inv = quat_conjugate(frame[3:])
    position = quat_rotate(q_inv, pose[:3] - frame[:3])
    return np.concatenate([position, quat_multiply(q_inv, pose[3:])])


def pose_from_frame(local: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pose_in_frame`: map a pose relative to ``frame`` back to the parent."""
    position = frame[:3] + quat_rotate(frame[3:], local[:3])
    return np.concatenate([position, quat_multiply(frame[3:], local[3:])])


def pose_to_vector(pose: np.ndarray) -> np.ndarray:
    """``[x, y, z, qx, qy, qz, qw]`` to ``[x, y, z, rx, ry, rz]``."""
    return np.concatenate([pose[:3], quat_to_rotvec(pose[3:])])
