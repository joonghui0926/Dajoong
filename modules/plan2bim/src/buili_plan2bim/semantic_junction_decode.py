"""Decode object and opening geometry from dense semantic junction heatmaps.

The network emits global room/icon segmentation together with 21 independent
junction channels.  This module uses the junction channels as geometric evidence
and the semantic channels only to classify the recovered geometry.  The decoder is
dependency-light, deterministic, and linear in the bounded peak candidate count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class JunctionDetection:
    class_index: int
    bbox_px: tuple[int, int, int, int]
    confidence: float
    pixel_area: int
    evidence_mode: str


@dataclass(frozen=True)
class _Peak:
    x: int
    y: int
    score: float
    channel: int


def _peaks(
    heatmap: np.ndarray,
    *,
    channel: int,
    threshold: float,
    radius: int,
    maximum: int,
) -> list[_Peak]:
    values = np.asarray(heatmap, dtype=np.float32)
    local_maximum = ndimage.maximum_filter(
        values,
        size=radius * 2 + 1,
        mode="constant",
        cval=0.0,
    )
    y, x = np.nonzero((values >= threshold) & (values == local_maximum))
    ordered = sorted(
        (
            _Peak(int(px), int(py), float(values[py, px]), channel)
            for py, px in zip(y, x, strict=True)
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    accepted: list[_Peak] = []
    for candidate in ordered:
        if any(
            abs(candidate.x - item.x) <= radius and abs(candidate.y - item.y) <= radius
            for item in accepted
        ):
            continue
        accepted.append(candidate)
        if len(accepted) >= maximum:
            break
    return accepted


def _iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def _source_bbox(
    bbox: tuple[float, float, float, float],
    *,
    model_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    model_width, model_height = model_size
    source_width, source_height = source_size
    x0, y0, x1, y1 = bbox
    return (
        int(np.clip(round(x0 * source_width / model_width), 0, source_width - 1)),
        int(np.clip(round(y0 * source_height / model_height), 0, source_height - 1)),
        int(np.clip(round(x1 * source_width / model_width), 1, source_width)),
        int(np.clip(round(y1 * source_height / model_height), 1, source_height)),
    )


def _non_maximum_suppression(
    detections: list[JunctionDetection],
    *,
    overlap_threshold: float,
) -> list[JunctionDetection]:
    output: list[JunctionDetection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(_iou(candidate.bbox_px, item.bbox_px) >= overlap_threshold for item in output):
            continue
        output.append(candidate)
    return output


def decode_icon_junctions(
    heatmaps: np.ndarray,
    icon_probability: np.ndarray,
    *,
    source_size: tuple[int, int],
    threshold: float = 0.4,
    maximum_peaks: int = 100,
) -> list[JunctionDetection]:
    """Recover rectangular fixed objects from four independently predicted corners."""

    if heatmaps.shape[0] < 21 or icon_probability.shape[0] < 2:
        raise ValueError("semantic output does not contain icon junction channels")
    height, width = heatmaps.shape[1:]
    radius = max(4, min(10, round(max(height, width) * 0.005)))
    tolerance = max(5, round(max(height, width) * 0.006))
    minimum_side = max(4, round(min(height, width) * 0.003))
    maximum_width = width * 0.5
    maximum_height = height * 0.5
    top_left = _peaks(
        heatmaps[17], channel=17, threshold=threshold, radius=radius, maximum=maximum_peaks
    )
    top_right = _peaks(
        heatmaps[18], channel=18, threshold=threshold, radius=radius, maximum=maximum_peaks
    )
    bottom_left = _peaks(
        heatmaps[19], channel=19, threshold=threshold, radius=radius, maximum=maximum_peaks
    )
    bottom_right = _peaks(
        heatmaps[20], channel=20, threshold=threshold, radius=radius, maximum=maximum_peaks
    )
    bottom_right_tree = (
        cKDTree(np.asarray([(item.x, item.y) for item in bottom_right], dtype=np.float64))
        if bottom_right
        else None
    )

    candidates: list[JunctionDetection] = []
    for upper_left in top_left:
        right_options = [
            item
            for item in top_right
            if item.x - upper_left.x >= minimum_side
            and item.x - upper_left.x <= maximum_width
            and abs(item.y - upper_left.y) <= tolerance
        ]
        lower_options = [
            item
            for item in bottom_left
            if item.y - upper_left.y >= minimum_side
            and item.y - upper_left.y <= maximum_height
            and abs(item.x - upper_left.x) <= tolerance
        ]
        for upper_right in sorted(right_options, key=lambda item: item.x)[:12]:
            for lower_left in sorted(lower_options, key=lambda item: item.y)[:12]:
                expected_x = upper_right.x
                expected_y = lower_left.y
                if bottom_right_tree is None:
                    continue
                distance, lower_right_index = bottom_right_tree.query(
                    (expected_x, expected_y),
                    distance_upper_bound=tolerance * math.sqrt(2),
                )
                if not np.isfinite(distance) or lower_right_index >= len(bottom_right):
                    continue
                lower_right = bottom_right[int(lower_right_index)]
                if (
                    abs(lower_right.x - expected_x) > tolerance
                    or abs(lower_right.y - expected_y) > tolerance
                ):
                    continue
                left = (upper_left.x + lower_left.x) / 2
                top = (upper_left.y + upper_right.y) / 2
                right = (upper_right.x + lower_right.x) / 2
                bottom = (lower_left.y + lower_right.y) / 2
                x0 = max(0, int(math.floor(left)))
                y0 = max(0, int(math.floor(top)))
                x1 = min(width, int(math.ceil(right)) + 1)
                y1 = min(height, int(math.ceil(bottom)) + 1)
                if x1 - x0 < minimum_side or y1 - y0 < minimum_side:
                    continue
                class_evidence = icon_probability[:, y0:y1, x0:x1].mean(axis=(1, 2))
                class_index = int(class_evidence.argmax())
                if class_index == 0:
                    continue
                class_confidence = float(class_evidence[class_index])
                corner_confidence = float(
                    np.mean(
                        [
                            upper_left.score,
                            upper_right.score,
                            lower_left.score,
                            lower_right.score,
                        ]
                    )
                )
                confidence = min(1.0, 0.65 * corner_confidence + 0.35 * class_confidence)
                source_bbox = _source_bbox(
                    (left, top, right, bottom),
                    model_size=(width, height),
                    source_size=source_size,
                )
                candidates.append(
                    JunctionDetection(
                        class_index=class_index,
                        bbox_px=source_bbox,
                        confidence=confidence,
                        pixel_area=max(
                            1,
                            (source_bbox[2] - source_bbox[0])
                            * (source_bbox[3] - source_bbox[1]),
                        ),
                        evidence_mode="four_corner_heatmap",
                    )
                )
    return _non_maximum_suppression(candidates, overlap_threshold=0.5)


def _line_class_evidence(
    icon_probability: np.ndarray,
    start: _Peak,
    end: _Peak,
    *,
    horizontal: bool,
    band: int,
) -> tuple[int, float, tuple[int, int, int, int]]:
    height, width = icon_probability.shape[1:]
    if horizontal:
        x0, x1 = sorted((start.x, end.x))
        center = round((start.y + end.y) / 2)
        bbox = (
            max(0, x0),
            max(0, center - band),
            min(width, x1 + 1),
            min(height, center + band + 1),
        )
    else:
        y0, y1 = sorted((start.y, end.y))
        center = round((start.x + end.x) / 2)
        bbox = (
            max(0, center - band),
            max(0, y0),
            min(width, center + band + 1),
            min(height, y1 + 1),
        )
    x0, y0, x1, y1 = bbox
    evidence = icon_probability[1:3, y0:y1, x0:x1].mean(axis=(1, 2))
    class_offset = int(evidence.argmax())
    return class_offset + 1, float(evidence[class_offset]), bbox


def decode_opening_junctions(
    heatmaps: np.ndarray,
    icon_probability: np.ndarray,
    wall_mask: np.ndarray,
    *,
    source_size: tuple[int, int],
    threshold: float = 0.4,
    maximum_peaks: int = 100,
) -> list[JunctionDetection]:
    """Recover door/window spans from reciprocal endpoint heatmaps and wall context."""

    if heatmaps.shape[0] < 17 or icon_probability.shape[0] < 3:
        raise ValueError("semantic output does not contain opening junction channels")
    height, width = heatmaps.shape[1:]
    radius = max(4, min(10, round(max(height, width) * 0.005)))
    tolerance = max(5, round(max(height, width) * 0.006))
    minimum_length = max(5, round(min(height, width) * 0.004))
    maximum_length = max(height, width) * 0.5
    wall_distance = ndimage.distance_transform_edt(~np.asarray(wall_mask, dtype=np.bool_))
    channel_pairs = ((13, 14, True), (15, 16, False))
    candidates: list[JunctionDetection] = []
    for start_channel, end_channel, horizontal in channel_pairs:
        starts = _peaks(
            heatmaps[start_channel],
            channel=start_channel,
            threshold=threshold,
            radius=radius,
            maximum=maximum_peaks,
        )
        ends = _peaks(
            heatmaps[end_channel],
            channel=end_channel,
            threshold=threshold,
            radius=radius,
            maximum=maximum_peaks,
        )
        compatible: dict[_Peak, list[_Peak]] = {}
        for start in starts:
            options = []
            for end in ends:
                along = end.x - start.x if horizontal else end.y - start.y
                across = abs(end.y - start.y) if horizontal else abs(end.x - start.x)
                if minimum_length <= along <= maximum_length and across <= tolerance:
                    options.append(end)
            compatible[start] = sorted(
                options,
                key=lambda item: item.x - start.x if horizontal else item.y - start.y,
            )
        reverse_nearest: dict[_Peak, _Peak] = {}
        for end in ends:
            options = [start for start, values in compatible.items() if end in values]
            if options:
                reverse_nearest[end] = min(
                    options,
                    key=lambda item: end.x - item.x if horizontal else end.y - item.y,
                )
        for start, options in compatible.items():
            if not options:
                continue
            end = options[0]
            if reverse_nearest.get(end) != start:
                continue
            endpoint_distance = (wall_distance[start.y, start.x] + wall_distance[end.y, end.x]) / 2
            if endpoint_distance > tolerance * 2.5:
                continue
            class_index, class_confidence, bbox = _line_class_evidence(
                icon_probability,
                start,
                end,
                horizontal=horizontal,
                band=max(3, tolerance // 2),
            )
            endpoint_confidence = (start.score + end.score) / 2
            confidence = min(1.0, 0.72 * endpoint_confidence + 0.28 * class_confidence)
            source_bbox = _source_bbox(
                bbox,
                model_size=(width, height),
                source_size=source_size,
            )
            candidates.append(
                JunctionDetection(
                    class_index=class_index,
                    bbox_px=source_bbox,
                    confidence=confidence,
                    pixel_area=max(
                        1,
                        (source_bbox[2] - source_bbox[0])
                        * (source_bbox[3] - source_bbox[1]),
                    ),
                    evidence_mode="reciprocal_opening_endpoints",
                )
            )
    return _non_maximum_suppression(candidates, overlap_threshold=0.45)
