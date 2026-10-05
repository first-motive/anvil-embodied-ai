"""Can and paper detector for the chest camera, by background subtraction.

The classical look-then-move baseline needs two image points per episode: where the can is
and where the sheet of paper is. Both sit on a table that the chest camera sees from a fixed
pose, so the detector compares each frame against a reference image of the empty table and
classifies what changed. It knows nothing about the can's colour, which keeps it working
for any can the episode uses.

Three facts about the real scene shape the pipeline:

* The robot arms frame the table at the image edges and never sit in exactly the same pose,
  so a plain frame difference lights them up. Changes are kept only where the reference
  image shows table, judged by colour closeness to the table's own centre.
* The reference is a per-pixel median over episodes, and the paper lands in nearly the same
  place every time, so a ghost of it survives in the reference. The paper is therefore
  found by appearance (bright, unsaturated, four-cornered) inside the table, not by
  difference.
* The can is whatever remains: the largest changed blob on the table once the paper is
  removed, upright or lying on its side.

Pure library code: numpy and OpenCV only, no ROS imports.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# The table's reference colour is sampled from this centre window of the background, as
# fractions of image height and width. The arms never reach the middle of the image.
_TABLE_SAMPLE_WINDOW = (0.4, 0.6)


@dataclass(frozen=True)
class DetectorParams:
    """Tuning for `detect`. Sizes are fractions of the image so any resolution works.

    Attributes:
        blur_frac: Gaussian blur kernel size as a fraction of image width.
        diff_threshold: Minimum per-channel absolute difference (0-255) that marks a pixel
            as changed against the background.
        open_frac: Morphological opening kernel size as a fraction of image width.
        close_frac: Morphological closing kernel size as a fraction of image width.
        table_color_distance: Maximum Lab distance from the table's reference colour for a
            background pixel to count as table.
        table_lightness_weight: Weight on the Lab lightness axis in that distance. Below 1
            so the darker, vignetted table edges still count as table.
        paper_max_saturation: Maximum HSV saturation (0-255) of a paper pixel.
        paper_min_value: Minimum HSV value (0-255) of a paper pixel.
        paper_min_area_frac: Smallest paper blob, as a fraction of image area.
        paper_max_area_frac: Largest paper blob, as a fraction of image area.
        paper_approx_epsilon: Polygon approximation tolerance as a fraction of the paper
            contour's perimeter; the approximation must have four vertices.
        paper_min_rectangularity: Minimum ratio of the paper contour's area to its
            minimum-area rectangle's area, which rejects ragged four-cornered blobs. A
            slightly curled sheet on the robot scored 0.84; 0.80 keeps it and finds paper
            in 84 of the 104 demos instead of 73, with no false quads inside the pick region.
        can_min_area_frac: Smallest can blob, as a fraction of image area.
    """

    blur_frac: float = 0.005
    diff_threshold: int = 40
    open_frac: float = 0.005
    close_frac: float = 0.015
    table_color_distance: float = 20.0
    table_lightness_weight: float = 0.4
    paper_max_saturation: int = 20
    paper_min_value: int = 190
    paper_min_area_frac: float = 0.002
    paper_max_area_frac: float = 0.05
    paper_approx_epsilon: float = 0.04
    paper_min_rectangularity: float = 0.80
    can_min_area_frac: float = 0.002

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DetectorParams:
        """Build params from a mapping such as a ROS parameter dict.

        Args:
            values: Field names to values. Missing fields keep their defaults.

        Returns:
            The params.

        Raises:
            ValueError: If a key is not a field of `DetectorParams`.
        """
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown detector params: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class Detection:
    """What `detect` found in one frame.

    Attributes:
        can_pixel: Can blob centroid as (u, v) pixels, or None when no can was found.
        paper_corners: Paper corners as a (4, 2) float array of (u, v) pixels, ordered
            clockwise from the top-left, or None when no paper was found.
        paper_center: Paper centre as (u, v) pixels, or None when no paper was found.
        mask: uint8 foreground mask (255 = changed table pixel) the can was taken from.
    """

    can_pixel: tuple[float, float] | None
    paper_corners: np.ndarray | None
    paper_center: tuple[float, float] | None
    mask: np.ndarray


def detect(
    frame: np.ndarray,
    background: np.ndarray,
    params: DetectorParams | None = None,
    region: np.ndarray | None = None,
) -> Detection:
    """Find the can and the paper in a frame by comparing it with the empty-table background.

    Args:
        frame: BGR uint8 image from the chest camera.
        background: BGR uint8 image of the empty table, same shape as `frame`.
        params: Tuning; defaults to `DetectorParams()`.
        region: Optional uint8 mask, same height and width as `frame`; only pixels where it
            is non-zero can belong to the can or the paper. A moved arm differs from the
            background far more than a can does, so without it the arm can win.

    Returns:
        The detection. Either object may be missing.

    Raises:
        ValueError: If `frame` and `background` differ in shape, or `region` does not
            match them.
    """
    if frame.shape != background.shape:
        raise ValueError(f"frame shape {frame.shape} != background shape {background.shape}")
    if region is not None and region.shape != frame.shape[:2]:
        raise ValueError(f"region shape {region.shape} != frame size {frame.shape[:2]}")
    params = params or DetectorParams()
    height, width = frame.shape[:2]
    image_area = height * width
    blur = _odd_kernel(params.blur_frac, width)
    open_kernel = _ellipse(params.open_frac, width)
    close_kernel = _ellipse(params.close_frac, width)

    # tradeoff: the table mask is rebuilt every call, which is a few milliseconds at 1080p;
    # cache it per background if the detector ever runs at frame rate.
    table = _table_mask(background, params, blur, open_kernel)
    if region is not None:
        table &= region > 0

    frame_blurred = cv2.GaussianBlur(frame, (blur, blur), 0)
    background_blurred = cv2.GaussianBlur(background, (blur, blur), 0)
    changed = cv2.absdiff(frame_blurred, background_blurred).max(axis=2) > params.diff_threshold
    foreground = _clean((changed & table).astype(np.uint8) * 255, open_kernel, close_kernel)

    hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] <= params.paper_max_saturation) & (hsv[..., 2] >= params.paper_min_value)
    paper_mask = _clean((white & table).astype(np.uint8) * 255, open_kernel, close_kernel)
    paper_contour = _find_paper(paper_mask, image_area, params)

    paper_corners = None
    paper_center = None
    if paper_contour is not None:
        rect = cv2.minAreaRect(paper_contour)
        paper_corners = _order_clockwise(cv2.boxPoints(rect))
        paper_center = (float(rect[0][0]), float(rect[0][1]))
        paper_fill = np.zeros_like(foreground)
        cv2.drawContours(paper_fill, [paper_contour], -1, 255, cv2.FILLED)
        foreground[cv2.dilate(paper_fill, close_kernel) > 0] = 0

    can_pixel = _find_can(foreground, image_area, params)
    return Detection(can_pixel, paper_corners, paper_center, foreground)


def _table_mask(
    background: np.ndarray, params: DetectorParams, blur: int, open_kernel: np.ndarray
) -> np.ndarray:
    """Return a bool mask of the background pixels that show table, holes filled."""
    height, width = background.shape[:2]
    blurred = cv2.GaussianBlur(background, (blur, blur), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    low, high = _TABLE_SAMPLE_WINDOW
    window = lab[int(height * low) : int(height * high), int(width * low) : int(width * high)]
    reference = np.median(window.reshape(-1, 3), axis=0)
    delta = lab - reference
    delta[..., 0] *= params.table_lightness_weight
    near = (np.linalg.norm(delta, axis=2) < params.table_color_distance).astype(np.uint8) * 255
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, open_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(near)
    if count < 2:
        return np.zeros((height, width), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours(
        (labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Filling the outline brings back what sat on the table in the reference, such as the
    # paper ghost, so objects there still count as on the table.
    filled = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled > 0


def _find_paper(mask: np.ndarray, image_area: int, params: DetectorParams) -> np.ndarray | None:
    """Return the largest four-cornered blob in the whiteness mask, or None."""
    min_area = params.paper_min_area_frac * image_area
    max_area = params.paper_max_area_frac * image_area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        epsilon = params.paper_approx_epsilon * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        (_, (rect_width, rect_height), _) = cv2.minAreaRect(contour)
        rectangularity = area / max(rect_width * rect_height, 1.0)
        if (
            len(polygon) == 4
            and cv2.isContourConvex(polygon)
            and rectangularity >= params.paper_min_rectangularity
        ):
            candidates.append((area, contour))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def _find_can(
    foreground: np.ndarray, image_area: int, params: DetectorParams
) -> tuple[float, float] | None:
    """Return the centroid of the largest foreground blob above the can size floor."""
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < params.can_min_area_frac * image_area:
        return None
    moments = cv2.moments(largest)
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _clean(mask: np.ndarray, open_kernel: np.ndarray, close_kernel: np.ndarray) -> np.ndarray:
    """Drop speckle with an opening, then knit split blobs back with a closing."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)


def _order_clockwise(corners: np.ndarray) -> np.ndarray:
    """Order four (u, v) corners clockwise on screen, starting from the top-left."""
    corners = np.asarray(corners, dtype=np.float32)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _odd_kernel(fraction: float, width: int) -> int:
    """Return an odd kernel size of at least 3 px for a fraction of image width."""
    size = max(3, int(round(fraction * width)))
    return size if size % 2 else size + 1


def _ellipse(fraction: float, width: int) -> np.ndarray:
    """Return an elliptical structuring element sized as a fraction of image width."""
    size = _odd_kernel(fraction, width)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
