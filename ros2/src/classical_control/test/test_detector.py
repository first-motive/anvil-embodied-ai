"""Behaviour tests for the can and paper detector, on synthetic scenes and real chest frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from classical_control.detector import DetectorParams, detect

FIXTURES = Path(__file__).parent / "fixtures"
HEIGHT, WIDTH = 480, 640
PAPER_CENTER = (420.0, 300.0)
CAN_CENTER = (220, 200)


def _noise(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 4.0, (HEIGHT, WIDTH, 3))


def _table(rng: np.random.Generator) -> np.ndarray:
    """Return a float wood-like table: a light brown with horizontal grain stripes."""
    table = np.empty((HEIGHT, WIDTH, 3))
    table[:] = (150.0, 180.0, 205.0)
    grain = 8.0 * np.sin(np.arange(HEIGHT) / 3.0)[:, None, None]
    return table + grain + _noise(rng)


def _to_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _draw_paper(frame: np.ndarray) -> np.ndarray:
    corners = cv2.boxPoints((PAPER_CENTER, (90.0, 70.0), 20.0))
    cv2.fillConvexPoly(frame, corners.astype(np.int32), (245, 245, 245))
    return corners


def _draw_can(frame: np.ndarray) -> None:
    cv2.ellipse(frame, (CAN_CENTER, (50, 90), 0.0), (40, 60, 200), cv2.FILLED)


@pytest.fixture
def scene() -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Return an empty-table background, a fresh-noise copy to draw on, and the rng."""
    rng = np.random.default_rng(0)
    table = _table(rng)
    background = _to_image(table)
    frame = _to_image(table + _noise(rng))
    return background, frame, rng


def test_finds_and_locates_can_and_paper(scene) -> None:
    background, frame, _ = scene
    corners = _draw_paper(frame)
    _draw_can(frame)

    detection = detect(frame, background)

    assert detection.can_pixel == pytest.approx(CAN_CENTER, abs=3.0)
    assert detection.paper_center == pytest.approx(PAPER_CENTER, abs=3.0)
    assert detection.paper_corners.shape == (4, 2)
    for corner in corners:
        assert np.linalg.norm(detection.paper_corners - corner, axis=1).min() < 4.0


def test_paper_corners_run_clockwise_from_top_left(scene) -> None:
    background, frame, _ = scene
    _draw_paper(frame)

    corners = detect(frame, background).paper_corners

    assert int(np.argmin(corners.sum(axis=1))) == 0
    edge_a, edge_b = corners[1] - corners[0], corners[2] - corners[1]
    # Positive cross product means clockwise on screen, where v grows downward.
    assert edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0] > 0


def test_empty_table_finds_nothing(scene) -> None:
    background, frame, _ = scene

    detection = detect(frame, background)

    assert detection.can_pixel is None
    assert detection.paper_corners is None
    assert detection.paper_center is None
    assert not detection.mask.any()


def test_paper_alone_is_not_a_can(scene) -> None:
    background, frame, _ = scene
    _draw_paper(frame)

    detection = detect(frame, background)

    assert detection.paper_center is not None
    assert detection.can_pixel is None


def test_can_alone_has_no_paper(scene) -> None:
    background, frame, _ = scene
    _draw_can(frame)

    detection = detect(frame, background)

    assert detection.can_pixel == pytest.approx(CAN_CENTER, abs=3.0)
    assert detection.paper_center is None


def test_scales_with_resolution(scene) -> None:
    background, frame, _ = scene
    _draw_paper(frame)
    _draw_can(frame)
    size = (WIDTH * 2, HEIGHT * 2)

    detection = detect(cv2.resize(frame, size), cv2.resize(background, size))

    assert detection.can_pixel == pytest.approx((CAN_CENTER[0] * 2, CAN_CENTER[1] * 2), abs=6.0)
    assert detection.paper_center == pytest.approx(
        (PAPER_CENTER[0] * 2, PAPER_CENTER[1] * 2), abs=6.0
    )


def test_rejects_mismatched_shapes(scene) -> None:
    background, frame, _ = scene

    with pytest.raises(ValueError, match="shape"):
        detect(frame[:-1], background)


def test_params_from_dict_overrides_and_rejects_unknown_keys() -> None:
    params = DetectorParams.from_dict({"diff_threshold": 25})

    assert params.diff_threshold == 25
    assert params.paper_min_value == DetectorParams().paper_min_value
    with pytest.raises(ValueError, match="no_such_param"):
        DetectorParams.from_dict({"no_such_param": 1})


# Hand-measured from the fixture frames: can body centre and paper centre, (u, v) pixels.
REAL_FRAMES = [
    ("chest_0001.jpg", (520, 215), (586, 313)),  # upright can
    ("chest_0015.jpg", (600, 290), (520, 385)),  # upright can, paper nearer the camera
    ("chest_0057.jpg", (725, 410), (524, 306)),  # can lying on its side by the right arm
]


@pytest.mark.parametrize(("name", "can", "paper"), REAL_FRAMES)
def test_real_chest_frames(name: str, can: tuple[int, int], paper: tuple[int, int]) -> None:
    background = cv2.imread(str(FIXTURES / "chest_background.jpg"))
    frame = cv2.imread(str(FIXTURES / name))

    detection = detect(frame, background)

    assert detection.can_pixel is not None
    assert detection.paper_center is not None
    assert np.hypot(*np.subtract(detection.can_pixel, can)) < 30.0
    assert np.hypot(*np.subtract(detection.paper_center, paper)) < 30.0


def test_real_background_against_itself_finds_nothing() -> None:
    background = cv2.imread(str(FIXTURES / "chest_background.jpg"))

    detection = detect(background, background)

    assert detection.can_pixel is None
    assert detection.paper_center is None


def test_region_keeps_a_bigger_blob_outside_it_from_being_the_can(scene) -> None:
    # A moved arm changes more pixels than the can; the pick region must exclude it.
    background, frame, _ = scene
    _draw_can(frame)
    height, width = frame.shape[:2]
    arm = (width - 60, height // 2)
    cv2.circle(frame, arm, 70, (30, 30, 30), cv2.FILLED)
    assert detect(frame, background).can_pixel == pytest.approx(arm, abs=5.0)

    region = np.zeros((height, width), np.uint8)
    region[:, : width - 160] = 255
    assert detect(frame, background, region=region).can_pixel == pytest.approx(CAN_CENTER, abs=3.0)


def test_region_must_match_the_frame(scene) -> None:
    background, frame, _ = scene
    with pytest.raises(ValueError, match="region"):
        detect(frame, background, region=np.zeros((10, 10), np.uint8))
