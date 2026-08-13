"""Deterministic full-sheet evidence for the tiny AEC residual model.

The learned model should not rediscover cheap CAD facts.  This module computes
long horizontal/vertical support and globally enclosed regions once per sheet,
then supplies aligned crops to the local specialist.  The enclosure channel is
therefore global even when inference itself is tiled.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

LEGACY_EVIDENCE_CONTRACT = "legacy_raster_valid_v1"
GLOBAL_ORIENTED_EVIDENCE_CONTRACT = "cad_global_oriented_v1"
GLOBAL_PROGRAM_INPUT_CONTRACT = "cad_global_oriented_letterbox_v2"
ORIENTED_EVIDENCE_ROTATION_CONTRACT = (
    "c4_spatial_rotate_swap_axis_channels_on_odd_quadrants_v1"
)
EVIDENCE_CONTRACTS = {
    LEGACY_EVIDENCE_CONTRACT,
    GLOBAL_ORIENTED_EVIDENCE_CONTRACT,
}


def _ndimage() -> Any:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover - guarded by train extra
        raise RuntimeError(
            "cad_global_oriented_v1 requires the project train extra (scipy)"
        ) from error
    return ndimage


def raster_ink(image: Image.Image) -> np.ndarray:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return np.clip(1.0 - grayscale, 0.0, 1.0)


def oriented_line_evidence(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return long-line support along the two dominant drafting axes."""

    ndimage = _ndimage()
    horizontal_short = ndimage.uniform_filter1d(ink, size=9, axis=1, mode="nearest")
    horizontal_long = ndimage.uniform_filter1d(ink, size=33, axis=1, mode="nearest")
    vertical_short = ndimage.uniform_filter1d(ink, size=9, axis=0, mode="nearest")
    vertical_long = ndimage.uniform_filter1d(ink, size=33, axis=0, mode="nearest")
    horizontal_support = np.maximum(horizontal_short, horizontal_long)
    vertical_support = np.maximum(vertical_short, vertical_long)
    horizontal = np.sqrt(np.clip(ink * horizontal_support, 0.0, 1.0))
    vertical = np.sqrt(np.clip(ink * vertical_support, 0.0, 1.0))
    return horizontal.astype(np.float32), vertical.astype(np.float32)


def global_enclosure_evidence(
    ink: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    """Compute an aligned global room-closure prior from the complete sheet."""

    ndimage = _ndimage()
    barrier = (horizontal > 0.34) | (vertical > 0.34)
    horizontal_close = ndimage.binary_closing(barrier, structure=np.ones((1, 21), bool))
    vertical_close = ndimage.binary_closing(barrier, structure=np.ones((21, 1), bool))
    barrier = horizontal_close | vertical_close | (ink > 0.92)
    barrier = ndimage.binary_dilation(barrier, iterations=2)
    filled = ndimage.binary_fill_holes(barrier)
    interior = filled & ~barrier
    labels, count = ndimage.label(interior)
    if count:
        support = np.bincount(labels.ravel())
        minimum_area = max(96, round(ink.size * 0.00015))
        keep = support >= minimum_area
        keep[0] = False
        interior = keep[labels]
    distance = ndimage.distance_transform_edt(interior)
    scale = max(8.0, min(32.0, min(ink.shape) / 40.0))
    return np.clip(distance / scale, 0.0, 1.0).astype(np.float32)


def build_cad_evidence(
    image: Image.Image,
    *,
    contract: str = GLOBAL_ORIENTED_EVIDENCE_CONTRACT,
) -> np.ndarray:
    """Build `[4, height, width]` full-sheet evidence for a named contract."""

    if contract not in EVIDENCE_CONTRACTS:
        raise ValueError(f"unsupported evidence contract: {contract}")
    ink = raster_ink(image)
    if contract == LEGACY_EVIDENCE_CONTRACT:
        valid = np.ones_like(ink, dtype=np.float32)
        return np.stack((ink, np.zeros_like(ink), np.zeros_like(ink), valid))
    horizontal, vertical = oriented_line_evidence(ink)
    enclosure = global_enclosure_evidence(ink, horizontal, vertical)
    return np.stack((ink, horizontal, vertical, enclosure)).astype(np.float32)


def letterbox_content_bbox(
    source_size: tuple[int, int],
    target_size: int,
) -> tuple[int, int, int, int]:
    """Return the centered, aspect-preserving content box in a square tensor."""

    source_width, source_height = source_size
    if source_width < 1 or source_height < 1 or target_size < 1:
        raise ValueError("source dimensions and target_size must be positive")
    scale = min(target_size / source_width, target_size / source_height)
    width = max(1, min(target_size, round(source_width * scale)))
    height = max(1, min(target_size, round(source_height * scale)))
    left = (target_size - width) // 2
    top = (target_size - height) // 2
    return left, top, left + width, top + height


def pad_letterbox_content(
    content: np.ndarray,
    target_size: int,
    content_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Place an already resized array into the shared square letterbox frame."""

    value = np.asarray(content)
    if value.ndim < 2:
        raise ValueError("letterbox content requires spatial axes")
    left, top, right, bottom = content_bbox
    if value.shape[-2:] != (bottom - top, right - left):
        raise ValueError("content shape does not match its letterbox bbox")
    if not (0 <= left < right <= target_size and 0 <= top < bottom <= target_size):
        raise ValueError("content bbox falls outside target tensor")
    output = np.zeros((*value.shape[:-2], target_size, target_size), dtype=value.dtype)
    output[..., top:bottom, left:right] = value
    return output


def letterbox_cad_evidence(
    evidence: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize `[channels, height, width]` without changing plan proportions."""

    source = np.asarray(evidence, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("evidence must have shape [channels, height, width]")
    content_bbox = letterbox_content_bbox(
        (int(source.shape[2]), int(source.shape[1])),
        target_size,
    )
    left, top, right, bottom = content_bbox
    channels = []
    for channel in source:
        image = Image.fromarray(channel, mode="F")
        resized = image.resize(
            (right - left, bottom - top),
            resample=Image.Resampling.BILINEAR,
        )
        channels.append(np.asarray(resized, dtype=np.float32))
    content = np.stack(channels)
    return (
        pad_letterbox_content(content, target_size, content_bbox)[None],
        content_bbox,
    )


def augment_drafting_ink(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply label-preserving scan and drafting interference to a local crop."""

    ndimage = _ndimage()
    result = np.asarray(ink, dtype=np.float32).copy()
    contrast = rng.uniform(0.65, 1.4)
    gamma = rng.uniform(0.7, 1.5)
    result = np.clip(result * contrast, 0.0, 1.0) ** gamma
    blur_sigma = rng.uniform(0.0, 0.9)
    if blur_sigma > 0.08:
        result = ndimage.gaussian_filter(result, sigma=blur_sigma, mode="nearest")
    morphology = rng.choice(("none", "none", "dilate", "erode"))
    if morphology == "dilate":
        result = ndimage.maximum_filter(result, size=3, mode="nearest")
    elif morphology == "erode":
        result = ndimage.minimum_filter(result, size=3, mode="nearest")

    canvas = Image.fromarray(np.uint8(np.clip(result, 0.0, 1.0) * 255), mode="L")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for _ in range(rng.randint(0, 7)):
        shade = rng.randint(45, 210)
        line_width = rng.randint(1, 2)
        if rng.random() < 0.65:
            start = rng.randrange(width), rng.randrange(height)
            end = rng.randrange(width), rng.randrange(height)
            draw.line((start, end), fill=shade, width=line_width)
        else:
            left = rng.randrange(width)
            top = rng.randrange(height)
            right = min(width - 1, left + rng.randint(5, max(6, width // 3)))
            bottom = min(height - 1, top + rng.randint(5, max(6, height // 3)))
            draw.rectangle((left, top, right, bottom), outline=shade, width=line_width)
    result = np.asarray(canvas, dtype=np.float32) / 255.0
    generator = np.random.default_rng(rng.getrandbits(64))
    result += generator.normal(0.0, rng.uniform(0.0, 0.035), size=result.shape).astype(np.float32)
    if rng.random() < 0.35:
        dropout = generator.random(result.shape) < rng.uniform(0.0, 0.008)
        result[dropout] *= rng.uniform(0.0, 0.4)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
