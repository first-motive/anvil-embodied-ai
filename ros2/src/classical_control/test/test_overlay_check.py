"""Tests for the overlay drawing: grid construction and where the TCP marker lands."""

from __future__ import annotations

import numpy as np
import pytest
from classical_control.camera_model import FisheyeCamera
from classical_control.overlay_check import TCP_COLOR, draw_overlay, table_grid

CAMERA = FisheyeCamera(640, 480, 300.0, 300.0, 320.0, 240.0, (0.02, 0.0, 0.0, 0.0))


def _looking_down(height: float) -> np.ndarray:
    """Camera at (0, 0, height) looking straight down at the world origin."""
    T = np.eye(4)
    # Optical x along world x, y along world -y, z along world -z.
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = (0.0, 0.0, height)
    return T


def test_table_grid_spans_the_square_at_table_height() -> None:
    lines = table_grid((0.4, -0.1), half_extent=0.1, spacing=0.05, table_z=0.22)
    assert len(lines) == 10  # five lines each way
    points = np.vstack(lines)
    np.testing.assert_allclose(points[:, 2], 0.22)
    assert points[:, 0].min() == pytest.approx(0.3)
    assert points[:, 0].max() == pytest.approx(0.5)
    assert points[:, 1].min() == pytest.approx(-0.2)
    assert points[:, 1].max() == pytest.approx(0.0)


def test_draw_overlay_puts_tcp_under_the_optical_axis_and_draws_grid() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    lines = table_grid((0.0, 0.0), half_extent=0.2, spacing=0.05, table_z=0.0)

    annotated, tcp_pixel = draw_overlay(image, CAMERA, _looking_down(1.0), (0.0, 0.0, 0.3), lines)

    assert tcp_pixel == pytest.approx([320.0, 240.0])
    assert tuple(annotated[240, 320]) == TCP_COLOR
    # Anti-aliased lines rarely hit the exact colour, so count any drawn pixel; the TCP
    # marker alone covers a few hundred.
    assert annotated.any(axis=2).sum() > 1500
    assert not image.any(), "input image must not be modified"


def test_draw_overlay_skips_points_behind_the_camera() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Table above the camera: every grid point and the TCP are behind it.
    lines = table_grid((0.0, 0.0), half_extent=0.2, spacing=0.05, table_z=2.0)

    annotated, tcp_pixel = draw_overlay(image, CAMERA, _looking_down(1.0), (0.0, 0.0, 1.5), lines)

    assert np.isnan(tcp_pixel).all()
    assert not annotated.any()
