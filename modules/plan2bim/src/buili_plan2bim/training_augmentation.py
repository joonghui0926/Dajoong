"""Exact C4 rotations shared by full-sheet and native-detail training.

The supervision contract includes spatial offsets, width/height, and yaw.  Rotating
only the pixels would create contradictory labels, so every dependent channel is
transformed together here.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


def deterministic_quadrant(key: str | int, *, seed: int) -> int:
    """Return a stable 0/90/180/270-degree training augmentation index."""

    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return digest[0] % 4


def deterministic_detail_crop(
    key: str | int,
    *,
    seed: int,
    size: tuple[int, int],
    minimum_fraction: float = 0.48,
    maximum_fraction: float = 0.78,
) -> tuple[int, int, int, int] | None:
    """Return a stable zoom crop while retaining full sheets in the mixture."""

    if not 0 < minimum_fraction <= maximum_fraction <= 1:
        raise ValueError("detail crop fractions must be in (0, 1]")
    width, height = size
    digest = hashlib.sha256(f"detail:{seed}:{key}".encode()).digest()
    # One third of samples remain complete sheets. The other two thirds teach
    # exactly the partial-window contract used by hierarchical inference.
    if digest[0] % 3 == 0:
        return None
    fraction = minimum_fraction + (maximum_fraction - minimum_fraction) * (
        int.from_bytes(digest[1:3], "little") / 65535.0
    )
    crop_width = max(8, min(width, round(width * fraction)))
    crop_height = max(8, min(height, round(height * fraction)))
    left = round((width - crop_width) * (digest[3] / 255.0))
    top = round((height - crop_height) * (digest[4] / 255.0))
    return left, top, left + crop_width, top + crop_height


def detail_crop_context(
    bbox: tuple[int, int, int, int] | None,
    *,
    size: tuple[int, int],
    frame_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Encode a crop in the actual drawing frame, excluding letterbox padding."""

    width, height = size
    if width < 1 or height < 1:
        raise ValueError("detail context requires a positive sheet size")
    frame_left, frame_top, frame_right, frame_bottom = frame_bbox or (
        0,
        0,
        width,
        height,
    )
    if not (
        0 <= frame_left < frame_right <= width
        and 0 <= frame_top < frame_bottom <= height
    ):
        raise ValueError("detail context frame lies outside the sheet")
    left, top, right, bottom = bbox or (
        frame_left,
        frame_top,
        frame_right,
        frame_bottom,
    )
    if not (
        frame_left <= left < right <= frame_right
        and frame_top <= top < bottom <= frame_bottom
    ):
        raise ValueError("detail context crop lies outside the sheet")
    frame_width = frame_right - frame_left
    frame_height = frame_bottom - frame_top
    return np.asarray(
        (
            (left - frame_left) / frame_width,
            (top - frame_top) / frame_height,
            (right - left) / frame_width,
            (bottom - top) / frame_height,
            float(left == frame_left),
            float(top == frame_top),
            float(right == frame_right),
            float(bottom == frame_bottom),
        ),
        dtype=np.float32,
    )


def rotate_spatial_bbox(
    bbox: tuple[int, int, int, int],
    *,
    size: tuple[int, int],
    quadrants: int,
) -> tuple[int, int, int, int]:
    """Rotate an axis-aligned content frame with the same C4 image transform."""

    width, height = size
    left, top, right, bottom = bbox
    if width != height:
        raise ValueError("C4 bbox rotation currently requires a square frame")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("spatial bbox lies outside its frame")
    quadrants %= 4
    if quadrants == 0:
        return bbox
    if quadrants == 1:
        return top, width - right, bottom, width - left
    if quadrants == 2:
        return width - right, height - bottom, width - left, height - top
    return height - bottom, left, height - top, right


def rotate_normalized_bbox_context(
    context: np.ndarray,
    quadrants: int,
) -> np.ndarray:
    """Rotate normalized `(x, y, width, height)` with a square image."""

    value = np.asarray(context, dtype=np.float32)
    if value.ndim != 1 or value.shape[0] < 4:
        raise ValueError("normalized bbox context must begin with four values")
    left, top, width, height = (float(item) for item in value[:4])
    right = min(1.0, max(left, left + width))
    bottom = min(1.0, max(top, top + height))
    left = min(1.0, max(0.0, left))
    top = min(1.0, max(0.0, top))
    quadrants %= 4
    if quadrants == 0:
        rotated = (left, top, right, bottom)
    elif quadrants == 1:
        rotated = (top, 1.0 - right, bottom, 1.0 - left)
    elif quadrants == 2:
        rotated = (1.0 - right, 1.0 - bottom, 1.0 - left, 1.0 - top)
    else:
        rotated = (1.0 - bottom, left, 1.0 - top, right)
    rotated_bbox = np.asarray(
        (
            rotated[0],
            rotated[1],
            rotated[2] - rotated[0],
            rotated[3] - rotated[1],
        ),
        dtype=np.float32,
    )
    if value.shape[0] == 4:
        return rotated_bbox
    # Room membership, normalized wall distance, and wall contact are invariant
    # under rotating the complete plan with the candidate.
    return np.concatenate((rotated_bbox, value[4:].copy()))


def _resize_channels(
    value: np.ndarray,
    size: tuple[int, int],
    *,
    nearest: bool,
) -> np.ndarray:
    array = np.asarray(value)
    leading = array.shape[:-2]
    flattened = array.reshape((-1, *array.shape[-2:]))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = [
        np.asarray(
            Image.fromarray(channel.astype(np.float32), mode="F").resize(
                size,
                resample=resampling,
            ),
            dtype=np.float32,
        )
        for channel in flattened
    ]
    return np.stack(resized).reshape((*leading, size[1], size[0]))


def crop_dense_training_example(
    *,
    evidence: np.ndarray,
    topology: np.ndarray,
    room_semantics: np.ndarray,
    element_semantics: np.ndarray,
    element_geometry: np.ndarray,
    element_geometry_valid: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Crop and resize every dependent target under one exact transform.

    Element instances cut by a crop boundary are removed from the local target.
    Teaching a clipped chair as a complete chair was one of the ways a detail
    window could learn to promote text, hatches, or furniture fragments. Walls
    and rooms retain their partial supervision because they are spatial fields;
    bounded element instances do not.
    """

    height, width = evidence.shape[-2:]
    left, top, right, bottom = bbox
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("detail crop lies outside the training sample")
    safe_valid = np.asarray(element_geometry_valid, dtype=np.float32).copy()
    safe_elements = np.asarray(element_semantics, dtype=np.int64).copy()
    safe_geometry = np.asarray(element_geometry, dtype=np.float32).copy()
    active = safe_valid > 0.5
    if active.any() and bbox != (0, 0, width, height):
        yy, xx = np.indices((height, width), dtype=np.float32)
        center_x = xx + safe_geometry[0] * width
        center_y = yy + safe_geometry[1] * height
        object_width = np.exp(safe_geometry[2]) * width
        object_height = np.exp(safe_geometry[3]) * height
        # A one-pixel guard prevents interpolation from reintroducing a sliced
        # instance on the resized crop boundary.
        complete = (
            (center_x - object_width / 2 >= left + 1)
            & (center_x + object_width / 2 <= right - 1)
            & (center_y - object_height / 2 >= top + 1)
            & (center_y + object_height / 2 <= bottom - 1)
        )
        safe_valid = np.where(active & complete, safe_valid, 0).astype(np.float32)
        safe_elements = np.where(safe_valid > 0.5, safe_elements, 0).astype(np.int64)
        safe_geometry *= safe_valid[None, ...]
    spatial = (..., slice(top, bottom), slice(left, right))
    target_size = (width, height)
    resized_evidence = _resize_channels(
        evidence[spatial], target_size, nearest=False
    )
    resized_topology = _resize_channels(
        topology[spatial], target_size, nearest=False
    )
    resized_rooms = _resize_channels(
        room_semantics[spatial], target_size, nearest=True
    ).astype(np.int64)
    resized_elements = _resize_channels(
        safe_elements[spatial], target_size, nearest=True
    ).astype(np.int64)
    # Geometry is sparse. Ordinary bilinear resizing blends valid values with
    # zeros outside the instance, corrupting size and offset targets along every
    # boundary. Resize numerator and support separately, then normalize.
    geometry_support = _resize_channels(
        safe_valid[spatial], target_size, nearest=False
    )
    geometry_numerator = _resize_channels(
        safe_geometry[spatial], target_size, nearest=False
    )
    resized_geometry = np.divide(
        geometry_numerator,
        geometry_support[None, ...],
        out=np.zeros_like(geometry_numerator),
        where=geometry_support[None, ...] > 1e-6,
    )
    resized_valid = _resize_channels(
        safe_valid[spatial], target_size, nearest=True
    )
    scale_x = width / (right - left)
    scale_y = height / (bottom - top)
    resized_geometry[0] *= scale_x
    resized_geometry[1] *= scale_y
    resized_geometry[2] += np.log(scale_x)
    resized_geometry[3] += np.log(scale_y)
    resized_geometry *= resized_valid[None, ...]
    return {
        "evidence": np.ascontiguousarray(resized_evidence),
        "topology": np.ascontiguousarray(resized_topology),
        "room_semantics": np.ascontiguousarray(resized_rooms),
        "element_semantics": np.ascontiguousarray(resized_elements),
        "element_geometry": np.ascontiguousarray(resized_geometry),
        "element_geometry_valid": np.ascontiguousarray(resized_valid),
    }


def rotate_spatial(value: np.ndarray, quadrants: int) -> np.ndarray:
    """Rotate the last two axes counter-clockwise and return contiguous storage."""

    quadrants %= 4
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError("spatial rotation requires at least two dimensions")
    return np.ascontiguousarray(np.rot90(array, quadrants, axes=(-2, -1)))


def rotate_oriented_evidence(
    evidence: np.ndarray,
    quadrants: int,
) -> np.ndarray:
    """Rotate one or more four-channel CAD-evidence views coherently.

    Every view is ordered ``[ink, horizontal, vertical, enclosure]``.  The
    horizontal and vertical channel *roles* exchange after an odd quarter turn;
    treating them as anonymous raster bands creates contradictory training data.
    Local element input contains three such views, so this helper deliberately
    supports any positive multiple of four channels.
    """

    value = np.asarray(evidence)
    if value.ndim != 3 or value.shape[0] < 4 or value.shape[0] % 4:
        raise ValueError(
            "oriented evidence must have [4*n, height, width] channels"
        )
    output = rotate_spatial(value, quadrants).copy()
    if quadrants % 2:
        for offset in range(0, output.shape[0], 4):
            horizontal = output[offset + 1].copy()
            output[offset + 1] = output[offset + 2]
            output[offset + 2] = horizontal
    return np.ascontiguousarray(output)


def rotate_element_geometry(
    geometry: np.ndarray,
    quadrants: int,
    *,
    spatial: bool,
) -> np.ndarray:
    """Rotate one six-channel geometry vector or a dense geometry field exactly."""

    quadrants %= 4
    value = np.asarray(geometry, dtype=np.float32)
    if value.shape[0] != 6:
        raise ValueError("element geometry must have six leading channels")
    if spatial:
        if value.ndim != 3:
            raise ValueError("dense geometry must have shape [6, height, width]")
        output = rotate_spatial(value, quadrants).copy()
    else:
        if value.ndim != 1:
            raise ValueError("local geometry must have shape [6]")
        output = value.copy()
    if quadrants == 0:
        return output

    dx = output[0].copy()
    dy = output[1].copy()
    log_width = output[2].copy()
    log_height = output[3].copy()
    sine = output[4].copy()
    cosine = output[5].copy()
    if quadrants == 1:
        output[0], output[1] = dy, -dx
        output[2], output[3] = log_height, log_width
        output[4], output[5] = cosine, -sine
    elif quadrants == 2:
        output[0], output[1] = -dx, -dy
        output[4], output[5] = -sine, -cosine
    else:
        output[0], output[1] = -dy, dx
        output[2], output[3] = log_height, log_width
        output[4], output[5] = -cosine, sine
    return np.ascontiguousarray(output)


def rotate_dense_training_example(
    *,
    evidence: np.ndarray,
    topology: np.ndarray,
    room_semantics: np.ndarray,
    element_semantics: np.ndarray,
    element_geometry: np.ndarray,
    element_geometry_valid: np.ndarray,
    quadrants: int,
) -> dict[str, np.ndarray]:
    """Rotate every full-sheet input and target under one consistent transform."""

    rotated_evidence = rotate_oriented_evidence(evidence, quadrants)
    return {
        "evidence": np.ascontiguousarray(rotated_evidence),
        "topology": rotate_spatial(topology, quadrants),
        "room_semantics": rotate_spatial(room_semantics, quadrants),
        "element_semantics": rotate_spatial(element_semantics, quadrants),
        "element_geometry": rotate_element_geometry(
            element_geometry,
            quadrants,
            spatial=True,
        ),
        "element_geometry_valid": rotate_spatial(
            element_geometry_valid,
            quadrants,
        ),
    }
