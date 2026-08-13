"""Optional full-sheet semantic recognition for raster floor plans.

The recognizer uses a content-addressed ONNX artifact and emits auditable pixel
geometry.  Model licensing and production authorization stay explicit in the
sidecar manifest; a research teacher can never silently become a release model.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .core.hashing import sha256_file, sha256_json
from .core.model.aec_decode import PixelLineProposal, PixelRoomProposal, PixelSymbolProposal
from .semantic_junction_decode import (
    JunctionDetection,
    decode_icon_junctions,
    decode_opening_junctions,
)


class SemanticDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    class_name: str
    symbol_class: str
    bbox_px: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)
    pixel_area: int = Field(ge=1)
    evidence_mode: str = "segmentation_component"
    review_required: bool
    promote_to_bim: bool


class SemanticRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    class_name: str
    polygon_px: list[tuple[float, float]] = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    pixel_area: int = Field(ge=1)
    review_required: bool


class SemanticWallVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = Field(gt=0)


class SemanticRecognitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "buili.semantic-recognition.v1"
    input_path: str
    input_sha256: str
    input_mode: str = "raster_only"
    model_version: str
    model_sha256: str
    decoder_version: str = "dajoong-semantic-junction-decoder-v2"
    decoder_settings: dict[str, float | int] = Field(default_factory=dict)
    license_scope: str
    production_authorized: bool
    source_size: tuple[int, int]
    model_input_size: tuple[int, int]
    wall_pixels: int
    wall_centerlines_px: list[tuple[float, float, float, float]] = Field(default_factory=list)
    wall_vectors_px: list[SemanticWallVector] = Field(default_factory=list)
    detections: list[SemanticDetection]
    rooms: list[SemanticRoom] = Field(default_factory=list)
    counts: dict[str, int]
    evidence_counts: dict[str, int] = Field(default_factory=dict)
    room_counts: dict[str, int] = Field(default_factory=dict)
    inference_ms: float
    total_ms: float
    overlay_path: str = ""
    content_sha256: str = ""

    def finalize(self) -> SemanticRecognitionResult:
        payload = self.model_dump(mode="json", exclude={"content_sha256", "total_ms"})
        self.content_sha256 = sha256_json(payload)
        return self


_SYMBOL_CLASS = {
    "Window": "window",
    "Door": "door",
    "Closet": "closet",
    "Electrical appliance": "electrical_appliance",
    "Toilet": "toilet",
    "Sink": "sink",
    "Sauna bench": "sauna_bench",
    "Fireplace": "fireplace",
    "Bathtub": "bathtub",
    "Chimney": "chimney",
}


def _bbox_iou(
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


def _merge_junction_detections(
    detections: list[SemanticDetection],
    junctions: list[JunctionDetection],
    *,
    icon_classes: tuple[str, ...],
) -> list[SemanticDetection]:
    """Fuse segmentation and geometric evidence without duplicating one object."""

    output = list(detections)
    claimed_existing: set[int] = set()
    for junction in sorted(junctions, key=lambda item: item.confidence, reverse=True):
        if not 0 < junction.class_index < len(icon_classes):
            continue
        class_name = icon_classes[junction.class_index]
        symbol_class = _SYMBOL_CLASS[class_name]
        left, top, right, bottom = junction.bbox_px
        width = right - left
        height = bottom - top
        geometry_complete = not (
            junction.evidence_mode == "four_corner_heatmap" and min(width, height) < 6
        )
        center = ((left + right) / 2, (top + bottom) / 2)
        matching_index = None
        matching_score = 0.0
        for index, item in enumerate(output):
            if index in claimed_existing:
                continue
            if item.class_name != class_name:
                continue
            item_left, item_top, item_right, item_bottom = item.bbox_px
            item_center = (
                (item_left + item_right) / 2,
                (item_top + item_bottom) / 2,
            )
            diagonal = max(
                12.0,
                math.hypot(right - left, bottom - top),
                math.hypot(item_right - item_left, item_bottom - item_top),
            )
            overlap = _bbox_iou(junction.bbox_px, item.bbox_px)
            distance_score = max(0.0, 1.0 - math.dist(center, item_center) / (diagonal * 0.4))
            score = max(overlap, distance_score)
            if score > matching_score and (overlap >= 0.12 or distance_score >= 0.45):
                matching_score = score
                matching_index = index
        if matching_index is None:
            confidence = junction.confidence
            # Four independent corners are only complete geometry when they span
            # both image axes. Near-collinear peaks remain review evidence but
            # cannot become a thin, false BIM object.
            review_required = confidence < 0.68 or not geometry_complete
            output.append(
                SemanticDetection(
                    id=f"semantic:junction:{junction.class_index}:{len(output)}",
                    class_name=class_name,
                    symbol_class=symbol_class,
                    bbox_px=junction.bbox_px,
                    confidence=confidence,
                    pixel_area=junction.pixel_area,
                    evidence_mode=junction.evidence_mode,
                    review_required=review_required,
                    promote_to_bim=not review_required,
                )
            )
            claimed_existing.add(len(output) - 1)
            continue
        existing = output[matching_index]
        confidence = max(existing.confidence, junction.confidence)
        pixel_area = max(existing.pixel_area, junction.pixel_area)
        independently_corroborated = (
            geometry_complete
            and existing.evidence_mode == "segmentation_component"
            and junction.evidence_mode == "four_corner_heatmap"
        )
        review_threshold = 0.64 if independently_corroborated else 0.68
        review_required = confidence < review_threshold or not geometry_complete
        output[matching_index] = existing.model_copy(
            update={
                "bbox_px": junction.bbox_px,
                "confidence": confidence,
                "pixel_area": pixel_area,
                "evidence_mode": f"{existing.evidence_mode}+{junction.evidence_mode}",
                "review_required": review_required,
                "promote_to_bim": not review_required,
            }
        )
        claimed_existing.add(matching_index)
    return output


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _softmax(value: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = value - value.max(axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -30, 30))
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _component_records(
    mask: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_area: int,
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    output: list[dict[str, Any]] = []
    objects = ndimage.find_objects(labels)
    for component, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        local = labels[slices] == component
        pixel_area = int(local.sum())
        if pixel_area < minimum_area:
            continue
        y_slice, x_slice = slices
        values = probability[slices][local]
        output.append(
            {
                "bbox_px": (
                    int(x_slice.start),
                    int(y_slice.start),
                    int(x_slice.stop),
                    int(y_slice.stop),
                ),
                "confidence": float(values.mean()),
                "pixel_area": pixel_area,
            }
        )
    return output


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1], strict=True)
        )
        / 2
    )


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    nearest = start[0] + fraction * dx, start[1] + fraction * dy
    return float(np.hypot(point[0] - nearest[0], point[1] - nearest[1]))


def _rdp(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    distances = [_point_line_distance(point, points[0], points[-1]) for point in points[1:-1]]
    if not distances or max(distances) <= tolerance:
        return [points[0], points[-1]]
    pivot = distances.index(max(distances)) + 1
    return _rdp(points[: pivot + 1], tolerance)[:-1] + _rdp(points[pivot:], tolerance)


def _simplify_loop(
    points: list[tuple[float, float]],
    *,
    tolerance: float,
) -> list[tuple[float, float]]:
    if len(points) <= 4:
        return points
    anchor = max(
        range(1, len(points)),
        key=lambda index: math.dist(points[0], points[index]),
    )
    first = _rdp(points[: anchor + 1], tolerance)
    second = _rdp(points[anchor:] + [points[0]], tolerance)
    simplified = first[:-1] + second[:-1]
    output: list[tuple[float, float]] = []
    for point in simplified:
        if not output or math.dist(output[-1], point) > 1e-6:
            output.append(point)
    return output


def _mask_outer_loop(mask: np.ndarray) -> list[tuple[float, float]]:
    """Trace the largest pixel-boundary loop without an OpenCV dependency."""

    height, width = mask.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    loops: list[list[tuple[float, float]]] = []
    edge_budget = sum(len(value) for value in edges.values()) + 1
    while edges:
        start = next(iter(edges))
        current = start
        loop: list[tuple[float, float]] = []
        for _ in range(edge_budget):
            loop.append((float(current[0]), float(current[1])))
            candidates = edges.get(current)
            if not candidates:
                break
            following = candidates.pop()
            if not candidates:
                del edges[current]
            current = following
            if current == start:
                if len(loop) >= 3:
                    loops.append(loop)
                break
    return max(loops, key=_polygon_area, default=[])


def _room_records(
    class_mask: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_area: int,
    source_size: tuple[int, int],
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(
        class_mask,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8),
    )
    scale_x = source_size[0] / class_mask.shape[1]
    scale_y = source_size[1] / class_mask.shape[0]
    output = []
    for component in range(1, count + 1):
        local = labels == component
        pixel_area = int(local.sum())
        if pixel_area < minimum_area:
            continue
        loop = _mask_outer_loop(local)
        if len(loop) < 3:
            continue
        simplified = _simplify_loop(loop, tolerance=2.0)
        polygon = [(x * scale_x, y * scale_y) for x, y in simplified]
        if len(polygon) < 3 or _polygon_area(polygon) < minimum_area * scale_x * scale_y:
            continue
        output.append(
            {
                "polygon_px": polygon,
                "confidence": float(probability[local].mean()),
                "pixel_area": pixel_area,
            }
        )
    return output


def _merge_axis_segments(
    segments: list[tuple[float, float, float, float]],
    *,
    horizontal: bool,
    coordinate_tolerance: float,
    gap_tolerance: float,
) -> list[tuple[float, float, float, float]]:
    if horizontal:
        normalized = sorted((y0, x0, x1) for x0, y0, x1, _ in segments)
    else:
        normalized = sorted((x0, y0, y1) for x0, y0, _, y1 in segments)
    groups: list[list[tuple[float, float, float]]] = []
    for coordinate, start, end in normalized:
        target = None
        for group in reversed(groups):
            group_coordinate = float(np.median([item[0] for item in group]))
            group_start = min(item[1] for item in group)
            group_end = max(item[2] for item in group)
            if coordinate - group_coordinate > coordinate_tolerance:
                break
            if abs(coordinate - group_coordinate) <= coordinate_tolerance and not (
                end < group_start - gap_tolerance or start > group_end + gap_tolerance
            ):
                target = group
                break
        if target is None:
            groups.append([(coordinate, start, end)])
        else:
            target.append((coordinate, start, end))
    output = []
    for group in groups:
        coordinate = float(np.median([item[0] for item in group]))
        start = float(min(item[1] for item in group))
        end = float(max(item[2] for item in group))
        output.append(
            (start, coordinate, end, coordinate)
            if horizontal
            else (coordinate, start, coordinate, end)
        )
    return output


def _axis_run_segments(
    support: np.ndarray,
    *,
    horizontal: bool,
    minimum_run: int,
) -> list[tuple[float, float, float, float]]:
    """Track long foreground runs across adjacent rows without joining T branches."""

    scan = support if horizontal else support.T
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    for coordinate, row in enumerate(scan):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        runs = [
            (int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
            if end - start >= minimum_run
        ]
        candidates = [item for item in active if item["last"] == coordinate - 1]
        claimed: set[int] = set()
        next_active: list[dict[str, Any]] = []
        for start, end in runs:
            best_index = -1
            best_score = 0.0
            for index, group in enumerate(candidates):
                if index in claimed:
                    continue
                previous_start, previous_end = group["runs"][-1][1:]
                previous_length = previous_end - previous_start
                current_length = end - start
                length_ratio = min(previous_length, current_length) / max(
                    1, previous_length, current_length
                )
                # At a T/X junction an orthogonal wall creates a short run that
                # overlaps the full long-axis run.  Overlap alone would merge both
                # directions into one component and then discard the actual wall
                # as an over-thick blob.  Comparable run length keeps direction
                # identity while still tolerating ordinary endpoint noise.
                if length_ratio < 0.42:
                    continue
                overlap = max(0, min(end, previous_end) - max(start, previous_start))
                score = overlap / max(1, min(end - start, previous_end - previous_start))
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0 and best_score >= 0.55:
                group = candidates[best_index]
                claimed.add(best_index)
                group["runs"].append((coordinate, start, end))
                group["last"] = coordinate
            else:
                group = {"last": coordinate, "runs": [(coordinate, start, end)]}
            next_active.append(group)
        for group in active:
            if group not in next_active:
                finished.append(group)
        active = next_active
    finished.extend(active)

    output: list[tuple[float, float, float, float]] = []
    for group in finished:
        records = group["runs"]
        if len(records) < 2:
            continue
        coordinate = float(np.median([record[0] for record in records]))
        start = float(np.median([record[1] for record in records]))
        end = float(np.median([record[2] for record in records]))
        thickness = records[-1][0] - records[0][0] + 1
        if end - start < max(minimum_run, thickness * 1.5):
            continue
        output.append(
            (start, coordinate, end, coordinate)
            if horizontal
            else (coordinate, start, coordinate, end)
        )
    return output


def _collapse_thick_parallel(
    segments: list[tuple[float, float, float, float]],
    wall_mask: np.ndarray,
    *,
    horizontal: bool,
) -> list[tuple[float, float, float, float]]:
    """Collapse multiple ridges inside one thick wall while preserving nearby walls."""

    pending = list(segments)
    changed = True
    maximum_band = max(24, round(max(wall_mask.shape) * 0.035))
    while changed:
        changed = False
        for left_index, left in enumerate(pending):
            left_coordinate = left[1] if horizontal else left[0]
            left_start, left_end = (left[0], left[2]) if horizontal else (left[1], left[3])
            for right_index in range(left_index + 1, len(pending)):
                right = pending[right_index]
                right_coordinate = right[1] if horizontal else right[0]
                distance = abs(right_coordinate - left_coordinate)
                if distance > maximum_band:
                    continue
                right_start, right_end = (
                    (right[0], right[2]) if horizontal else (right[1], right[3])
                )
                overlap_start = max(left_start, right_start)
                overlap_end = min(left_end, right_end)
                if overlap_end - overlap_start < 20:
                    continue
                c0 = max(0, round(min(left_coordinate, right_coordinate)))
                c1 = min(
                    wall_mask.shape[0 if horizontal else 1],
                    round(max(left_coordinate, right_coordinate)) + 1,
                )
                s0 = max(0, round(overlap_start))
                s1 = min(
                    wall_mask.shape[1 if horizontal else 0],
                    round(overlap_end),
                )
                band = wall_mask[c0:c1, s0:s1] if horizontal else wall_mask[s0:s1, c0:c1]
                if band.size == 0 or float(band.mean()) < 0.55:
                    continue
                coordinate = (left_coordinate + right_coordinate) / 2
                start = min(left_start, right_start)
                end = max(left_end, right_end)
                merged = (
                    (start, coordinate, end, coordinate)
                    if horizontal
                    else (coordinate, start, coordinate, end)
                )
                pending[left_index] = merged
                pending.pop(right_index)
                changed = True
                break
            if changed:
                break
    return pending


def _wall_centerlines(wall_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Vectorize an orthogonal semantic wall mask without consulting a reference graph."""

    mask = np.asarray(wall_mask, dtype=np.bool_)
    minimum_run = max(20, round(max(mask.shape) * 0.01))
    raw: list[tuple[float, float, float, float]] = []
    for horizontal in (True, False):
        # A separable run-support filter is equivalent to the long-axis evidence
        # needed here and avoids the quadratic cost of a large binary footprint.
        opened = (
            ndimage.uniform_filter1d(
                mask.astype(np.float32),
                size=minimum_run,
                axis=1 if horizontal else 0,
                mode="constant",
            )
            >= 0.95
        )
        raw.extend(
            _axis_run_segments(
                opened,
                horizontal=horizontal,
                minimum_run=minimum_run,
            )
        )

    horizontal = [line for line in raw if line[1] == line[3]]
    vertical = [line for line in raw if line[0] == line[2]]
    coordinate_tolerance = max(4.0, minimum_run * 0.45)
    gap_tolerance = max(24.0, max(mask.shape) * 0.075)
    horizontal = _merge_axis_segments(
        horizontal,
        horizontal=True,
        coordinate_tolerance=coordinate_tolerance,
        gap_tolerance=gap_tolerance,
    )
    vertical = _merge_axis_segments(
        vertical,
        horizontal=False,
        coordinate_tolerance=coordinate_tolerance,
        gap_tolerance=gap_tolerance,
    )
    horizontal = _collapse_thick_parallel(
        horizontal,
        mask,
        horizontal=True,
    )
    vertical = _collapse_thick_parallel(
        vertical,
        mask,
        horizontal=False,
    )

    split_horizontal: list[tuple[float, float, float, float]] = []
    split_vertical: list[tuple[float, float, float, float]] = []
    intersection_tolerance = max(8.0, minimum_run * 0.75)
    minimum_segment = max(10.0, minimum_run * 0.5)
    for x0, y, x1, _ in horizontal:
        cuts = [x0, x1]
        for x, y0, _, y1 in vertical:
            if x0 - intersection_tolerance <= x <= x1 + intersection_tolerance and (
                y0 - intersection_tolerance <= y <= y1 + intersection_tolerance
            ):
                cuts.append(float(np.clip(x, x0, x1)))
        cuts = sorted(set(round(value, 3) for value in cuts))
        split_horizontal.extend(
            (left, y, right, y)
            for left, right in zip(cuts, cuts[1:], strict=False)
            if right - left >= minimum_segment
        )
    for x, y0, _, y1 in vertical:
        cuts = [y0, y1]
        for x0, y, x1, _ in horizontal:
            if y0 - intersection_tolerance <= y <= y1 + intersection_tolerance and (
                x0 - intersection_tolerance <= x <= x1 + intersection_tolerance
            ):
                cuts.append(float(np.clip(y, y0, y1)))
        cuts = sorted(set(round(value, 3) for value in cuts))
        split_vertical.extend(
            (x, top, x, bottom)
            for top, bottom in zip(cuts, cuts[1:], strict=False)
            if bottom - top >= minimum_segment
        )
    return [_recenter_wall_segment(line, mask) for line in split_horizontal + split_vertical]


def _foreground_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
            strict=True,
        )
    )


def _wall_cross_sections(
    line: tuple[float, float, float, float],
    wall_mask: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Measure robust perpendicular midpoints and widths along one wall."""

    x0, y0, x1, y1 = line
    horizontal = abs(x1 - x0) >= abs(y1 - y0)
    coordinate = (y0 + y1) / 2 if horizontal else (x0 + x1) / 2
    start, end = (x0, x1) if horizontal else (y0, y1)
    length = abs(end - start)
    sample_count = max(7, min(96, round(length / 8)))
    # Avoid junctions and caps where the perpendicular run is not wall thickness.
    margin = min(length * 0.12, 24.0)
    positions = np.linspace(min(start, end) + margin, max(start, end) - margin, sample_count)
    maximum_offset = max(24.0, max(wall_mask.shape) * 0.04)
    midpoints: list[float] = []
    widths: list[float] = []
    for position in positions:
        sample_index = int(np.clip(round(position), 0, wall_mask.shape[1 if horizontal else 0] - 1))
        values = wall_mask[:, sample_index] if horizontal else wall_mask[sample_index, :]
        runs = _foreground_runs(values)
        if not runs:
            continue
        containing = [run for run in runs if run[0] <= coordinate < run[1]]
        candidates = (
            containing
            or sorted(
                runs,
                key=lambda run: abs(((run[0] + run[1] - 1) / 2) - coordinate),
            )[:1]
        )
        run_start, run_end = candidates[0]
        midpoint = (run_start + run_end - 1) / 2
        if abs(midpoint - coordinate) > maximum_offset:
            continue
        width = float(run_end - run_start)
        if width <= 0:
            continue
        midpoints.append(float(midpoint))
        widths.append(width)
    return midpoints, widths


def _recenter_wall_segment(
    line: tuple[float, float, float, float],
    wall_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    midpoints, widths = _wall_cross_sections(line, wall_mask)
    if len(midpoints) < 3 or len(widths) < 3:
        return line
    center = float(np.median(midpoints))
    x0, y0, x1, y1 = line
    if abs(x1 - x0) >= abs(y1 - y0):
        return x0, center, x1, center
    return center, y0, center, y1


def _wall_vectors(wall_mask: np.ndarray) -> list[SemanticWallVector]:
    mask = np.asarray(wall_mask, dtype=np.bool_)
    output: list[SemanticWallVector] = []
    for line in _wall_centerlines(mask):
        _, widths = _wall_cross_sections(line, mask)
        if not widths:
            continue
        x0, y0, x1, y1 = line
        length = math.dist((x0, y0), (x1, y1))
        thickness = float(np.quantile(widths, 0.35))
        # Very short split segments can lie entirely inside a T/X junction.
        # Re-measure a longer collinear support window before deciding that the
        # junction blob itself is the wall width.
        if thickness > max(8.0, length * 0.6):
            extension = max(32.0, thickness * 2.0)
            if abs(x1 - x0) >= abs(y1 - y0):
                extended = (
                    max(0.0, min(x0, x1) - extension),
                    y0,
                    min(float(mask.shape[1] - 1), max(x0, x1) + extension),
                    y1,
                )
            else:
                extended = (
                    x0,
                    max(0.0, min(y0, y1) - extension),
                    x1,
                    min(float(mask.shape[0] - 1), max(y0, y1) + extension),
                )
            _, extended_widths = _wall_cross_sections(extended, mask)
            if extended_widths:
                thickness = float(np.quantile(extended_widths, 0.25))
        if thickness > max(12.0, length):
            # A segment shorter than its measured cross-section is a junction
            # fragment, not an independently editable wall.
            continue
        output.append(
            SemanticWallVector(
                start_px=(line[0], line[1]),
                end_px=(line[2], line[3]),
                thickness_px=max(1.0, thickness),
            )
        )
    diagonal = _diagonal_wall_vectors(mask, output)
    return _merge_collinear_wall_vectors(output) + diagonal


def _merge_collinear_wall_vectors(
    vectors: list[SemanticWallVector],
) -> list[SemanticWallVector]:
    """Join raster-fragmented runs that describe one continuous BIM wall.

    Doors and windows interrupt the semantic wall mask. Keeping every visible
    run as an independent host wall makes the compiler clamp an opening to the
    end of a short run, moving otherwise accurate source evidence. Only
    axis-aligned runs on the same narrow center band are joined here; nearby
    parallel walls remain separate.
    """

    if len(vectors) < 2:
        return list(vectors)
    parent = list(range(len(vectors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    geometry: list[tuple[str, float, float, float, float]] = []
    for vector in vectors:
        x0, y0 = vector.start_px
        x1, y1 = vector.end_px
        horizontal = abs(x1 - x0) >= abs(y1 - y0)
        if horizontal:
            geometry.append(
                (
                    "horizontal",
                    (y0 + y1) / 2,
                    min(x0, x1),
                    max(x0, x1),
                    vector.thickness_px,
                )
            )
        else:
            geometry.append(
                (
                    "vertical",
                    (x0 + x1) / 2,
                    min(y0, y1),
                    max(y0, y1),
                    vector.thickness_px,
                )
            )

    for left_index, left in enumerate(geometry):
        for right_index in range(left_index + 1, len(geometry)):
            right = geometry[right_index]
            if left[0] != right[0]:
                continue
            center_tolerance = max(3.0, min(left[4], right[4]) * 0.35)
            if abs(left[1] - right[1]) > center_tolerance:
                continue
            gap = max(left[2], right[2]) - min(left[3], right[3])
            gap_tolerance = max(12.0, max(left[4], right[4]) * 1.2)
            if gap <= gap_tolerance:
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        groups.setdefault(find(index), []).append(index)
    merged: list[SemanticWallVector] = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(vectors[indices[0]])
            continue
        members = [geometry[index] for index in indices]
        weights = [max(1.0, member[3] - member[2]) for member in members]
        center = float(np.average([member[1] for member in members], weights=weights))
        start = min(member[2] for member in members)
        end = max(member[3] for member in members)
        thickness = float(
            np.average([member[4] for member in members], weights=weights)
        )
        if members[0][0] == "horizontal":
            start_px = (start, center)
            end_px = (end, center)
        else:
            start_px = (center, start)
            end_px = (center, end)
        merged.append(
            SemanticWallVector(
                start_px=start_px,
                end_px=end_px,
                thickness_px=thickness,
            )
        )
    return merged


def _recover_unclassified_interior_rooms(
    rooms: list[SemanticRoom],
    wall_vectors: list[SemanticWallVector],
    *,
    source_size: tuple[int, int],
) -> list[SemanticRoom]:
    """Preserve large enclosed areas omitted by the semantic room classifier."""

    width, height = source_size
    if not wall_vectors or width < 8 or height < 8:
        return []
    wall_image = Image.new("1", source_size, 0)
    wall_draw = ImageDraw.Draw(wall_image)
    for vector in wall_vectors:
        wall_draw.line(
            (*vector.start_px, *vector.end_px),
            fill=1,
            width=max(3, round(vector.thickness_px)),
        )
    wall_mask = np.asarray(wall_image, dtype=np.bool_)
    free_labels, component_count = ndimage.label(
        ~wall_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    border_labels = set(
        np.unique(
            np.concatenate(
                (
                    free_labels[0],
                    free_labels[-1],
                    free_labels[:, 0],
                    free_labels[:, -1],
                )
            )
        ).tolist()
    )
    enclosed = np.zeros_like(wall_mask)
    for component in range(1, component_count + 1):
        if component not in border_labels:
            enclosed |= free_labels == component
    if not enclosed.any():
        return []

    covered_image = Image.new("1", source_size, 0)
    covered_draw = ImageDraw.Draw(covered_image)
    for room in rooms:
        if len(room.polygon_px) >= 3:
            covered_draw.polygon(room.polygon_px, fill=1)
    residual = enclosed & ~np.asarray(covered_image, dtype=np.bool_)
    residual_labels, residual_count = ndimage.label(
        residual,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    # A high relative floor-area gate prevents wall-edge slivers and door gaps
    # from becoming hundreds of fake rooms. Only a materially omitted enclosed
    # space is preserved as a reviewable, unclassified room.
    minimum_area = max(1024, round(width * height * 0.01))
    output: list[SemanticRoom] = []
    for component in range(1, residual_count + 1):
        local = residual_labels == component
        pixel_area = int(local.sum())
        if pixel_area < minimum_area:
            continue
        loop = _mask_outer_loop(local)
        polygon = _simplify_loop(loop, tolerance=2.0)
        if len(polygon) < 3 or _polygon_area(polygon) < minimum_area:
            continue
        output.append(
            SemanticRoom(
                id=f"semantic:room:unclassified:{len(output)}",
                class_name="Unclassified interior",
                polygon_px=polygon,
                confidence=0.5,
                pixel_area=pixel_area,
                review_required=True,
            )
        )
    return output


def _diagonal_wall_vectors(
    wall_mask: np.ndarray,
    axis_vectors: list[SemanticWallVector],
) -> list[SemanticWallVector]:
    """Recover non-orthogonal wall runs left after axis-vector coverage."""

    height, width = wall_mask.shape
    coverage_image = Image.new("1", (width, height), 0)
    coverage_draw = ImageDraw.Draw(coverage_image)
    for vector in axis_vectors:
        coverage_draw.line(
            (*vector.start_px, *vector.end_px),
            fill=1,
            width=max(3, round(vector.thickness_px + 6)),
        )
    coverage = np.asarray(coverage_image, dtype=np.bool_)
    distance_to_axis = ndimage.distance_transform_edt(~coverage)
    residual = wall_mask & ~coverage
    labels, component_count = ndimage.label(
        residual,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    minimum_length = max(18.0, max(wall_mask.shape) * 0.01)
    output: list[SemanticWallVector] = []
    for component_index in range(1, component_count + 1):
        y, x = np.nonzero(labels == component_index)
        if len(x) < minimum_length * 2:
            continue
        points = np.column_stack((x, y)).astype(np.float64)
        center = points.mean(axis=0)
        covariance = np.cov(points - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[1] <= 1e-9 or eigenvalues[1] / max(eigenvalues[0], 1e-9) < 5.0:
            continue
        direction = eigenvectors[:, 1]
        angle = abs(math.degrees(math.atan2(direction[1], direction[0]))) % 90
        angle_to_axis = min(angle, 90 - angle)
        if angle_to_axis < 10.0:
            continue
        normal = np.asarray((-direction[1], direction[0]))
        along = (points - center) @ direction
        across = (points - center) @ normal
        low, high = np.quantile(along, (0.02, 0.98))
        length = float(high - low)
        if length < minimum_length:
            continue
        thickness = float(np.quantile(across, 0.9) - np.quantile(across, 0.1) + 1)
        if thickness <= 1 or thickness > length * 0.65:
            continue
        start = center + direction * low
        end = center + direction * high
        connection_tolerance = max(12.0, thickness * 1.5)
        start_index = (
            int(np.clip(round(start[1]), 0, height - 1)),
            int(np.clip(round(start[0]), 0, width - 1)),
        )
        end_index = (
            int(np.clip(round(end[1]), 0, height - 1)),
            int(np.clip(round(end[0]), 0, width - 1)),
        )
        if (
            distance_to_axis[start_index] > connection_tolerance
            or distance_to_axis[end_index] > connection_tolerance
        ):
            # Detached title-block graphics and nearby reference plans can also
            # be elongated.  A building wall must join the accepted wall graph
            # at both ends before residual geometry is promoted.
            continue
        output.append(
            SemanticWallVector(
                start_px=(float(start[0]), float(start[1])),
                end_px=(float(end[0]), float(end[1])),
                thickness_px=thickness,
            )
        )
    return output


def _refine_wall_vectors_from_raster(
    source_gray: np.ndarray,
    vectors: list[SemanticWallVector],
) -> list[SemanticWallVector]:
    """Align accepted semantic walls to robust source-raster cross sections."""

    gray = np.asarray(source_gray, dtype=np.uint8)
    height, width = gray.shape
    output: list[SemanticWallVector] = []
    for vector in vectors:
        start = np.asarray(vector.start_px, dtype=np.float64)
        end = np.asarray(vector.end_px, dtype=np.float64)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 8:
            output.append(vector)
            continue
        direction = delta / length
        normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
        sample_count = max(9, min(96, round(length / 10)))
        samples = np.linspace(0.12, 0.88, sample_count)
        search_radius = max(8.0, vector.thickness_px * 1.25)
        offset_count = max(17, round(search_radius * 2) + 1)
        offsets = np.linspace(-search_radius, search_radius, offset_count)
        midpoints: list[float] = []
        spans: list[float] = []
        expected_half_width = max(2.0, vector.thickness_px * 0.75)
        for fraction in samples:
            center = start + delta * fraction
            coordinates = center[None, :] + offsets[:, None] * normal[None, :]
            x = np.clip(np.rint(coordinates[:, 0]).astype(int), 0, width - 1)
            y = np.clip(np.rint(coordinates[:, 1]).astype(int), 0, height - 1)
            darkness = gray[y, x] < 180
            runs = _foreground_runs(darkness)
            relevant = []
            for run_start, run_end in runs:
                run_offsets = offsets[run_start:run_end]
                if not len(run_offsets):
                    continue
                if run_offsets[-1] < -expected_half_width or run_offsets[0] > expected_half_width:
                    continue
                relevant.append((float(run_offsets[0]), float(run_offsets[-1])))
            if not relevant:
                continue
            lower = min(item[0] for item in relevant)
            upper = max(item[1] for item in relevant)
            span = upper - lower + (offsets[1] - offsets[0])
            if span < max(2.0, vector.thickness_px * 0.3):
                continue
            if span > max(12.0, vector.thickness_px * 2.2):
                continue
            midpoints.append((lower + upper) / 2)
            spans.append(float(span))
        if len(midpoints) < max(4, sample_count // 4):
            output.append(vector)
            continue
        normal_shift = float(np.median(midpoints))
        if abs(normal_shift) > max(8.0, vector.thickness_px * 0.7):
            output.append(vector)
            continue
        band_offsets = np.linspace(
            -vector.thickness_px / 2,
            vector.thickness_px / 2,
            max(5, round(vector.thickness_px) + 1),
        )

        def alignment_score(
            shift: float,
            *,
            base_start: np.ndarray = start,
            sample_fractions: np.ndarray = samples,
            base_delta: np.ndarray = delta,
            offsets: np.ndarray = band_offsets,
            base_normal: np.ndarray = normal,
        ) -> float:
            centers = base_start[None, :] + sample_fractions[:, None] * base_delta[None, :]
            coordinates = (
                centers[:, None, :] + (offsets[None, :, None] + shift) * base_normal[None, None, :]
            )
            x = np.clip(np.rint(coordinates[..., 0]).astype(int), 0, width - 1)
            y = np.clip(np.rint(coordinates[..., 1]).astype(int), 0, height - 1)
            return float((gray[y, x] < 180).mean())

        baseline_score = alignment_score(0.0)
        refined_score = alignment_score(normal_shift)
        if refined_score < baseline_score + 0.04:
            output.append(vector)
            continue
        shifted_start = start + normal * normal_shift
        shifted_end = end + normal * normal_shift
        output.append(
            SemanticWallVector(
                start_px=(float(shifted_start[0]), float(shifted_start[1])),
                end_px=(float(shifted_end[0]), float(shifted_end[1])),
                thickness_px=vector.thickness_px,
            )
        )
    return output


class OnnxFloorPlanSemanticRecognizer:
    """Run one dense full-sheet semantic model without SVG or manual coordinates."""

    def __init__(self, model_path: str | Path, *, threads: int = 1) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.manifest_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.model_sha256 = sha256_file(self.model_path)
        if self.manifest.get("onnx_sha256") != self.model_sha256:
            raise ValueError("semantic ONNX artifact does not match its manifest")
        if self.manifest.get("output_contract") != "cubicasa44_dense_logits":
            raise ValueError("unsupported semantic output contract")
        self.model_version = str(self.manifest.get("model_version") or "")
        self.room_classes = tuple(self.manifest.get("room_classes") or ())
        self.icon_classes = tuple(self.manifest.get("icon_classes") or ())
        if len(self.room_classes) != 12 or len(self.icon_classes) != 11:
            raise ValueError("semantic manifest taxonomy is incomplete")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def recognize(
        self,
        image_path: str | Path,
        *,
        max_side: int = 1024,
    ) -> tuple[SemanticRecognitionResult, np.ndarray]:
        started = time.perf_counter()
        source_path = Path(image_path).expanduser().resolve()
        source = Image.open(source_path).convert("RGB")
        scale = min(1.0, max_side / max(source.size))
        width = max(64, round(source.width * scale / 32) * 32)
        height = max(64, round(source.height * scale / 32) * 32)
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor / 127.5 - 1.0)

        inference_started = time.perf_counter()
        prediction = self.session.run(None, {self.input_name: tensor})[0][0]
        inference_ms = (time.perf_counter() - inference_started) * 1000
        room_probability = _softmax(prediction[21:33], axis=0)
        icon_probability = _softmax(prediction[33:44], axis=0)
        room_class = room_probability.argmax(axis=0).astype(np.uint8)
        icon_class = icon_probability.argmax(axis=0).astype(np.uint8)
        room_original = np.asarray(
            Image.fromarray(room_class).resize(source.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        icon_original = np.asarray(
            Image.fromarray(icon_class).resize(source.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        probability_original = np.stack(
            [
                np.asarray(
                    Image.fromarray(channel.astype(np.float32), mode="F").resize(
                        source.size, Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                for channel in icon_probability
            ]
        )

        minimum_area = max(12, round(source.width * source.height * 0.00001))
        detections: list[SemanticDetection] = []
        for class_index, class_name in enumerate(self.icon_classes[1:], start=1):
            records = _component_records(
                icon_original == class_index,
                probability_original[class_index],
                minimum_area=minimum_area,
            )
            for record in records:
                confidence = float(record["confidence"])
                pixel_area = int(record["pixel_area"])
                # Keep low-confidence evidence in the audit result, but flag it for
                # review instead of silently promoting a fragment to BIM geometry.
                review_required = confidence < 0.68 or pixel_area < minimum_area * 8
                detections.append(
                    SemanticDetection(
                        id=f"semantic:{class_index}:{len(detections)}",
                        class_name=class_name,
                        symbol_class=_SYMBOL_CLASS[class_name],
                        bbox_px=record["bbox_px"],
                        confidence=confidence,
                        pixel_area=pixel_area,
                        review_required=review_required,
                        promote_to_bim=not review_required,
                    )
                )
        # Door strokes can be fragmented while the same opening is classified as
        # one strong window region.  Fuse only when at least two independent door
        # components land inside the region; a single weak stroke cannot relabel it.
        consumed: set[str] = set()
        fused: list[SemanticDetection] = []
        for window in [item for item in detections if item.class_name == "Window"]:
            left, top, right, bottom = window.bbox_px
            supports = []
            for door in [item for item in detections if item.class_name == "Door"]:
                door_left, door_top, door_right, door_bottom = door.bbox_px
                center_x = (door_left + door_right) / 2
                center_y = (door_top + door_bottom) / 2
                if left <= center_x <= right and top <= center_y <= bottom:
                    supports.append(door)
            if len(supports) < 2:
                continue
            consumed.add(window.id)
            consumed.update(item.id for item in supports)
            fused.append(
                window.model_copy(
                    update={
                        "id": f"semantic:fused-door:{len(fused)}",
                        "class_name": "Door",
                        "symbol_class": "door",
                        "confidence": min(
                            window.confidence,
                            max(item.confidence for item in supports),
                        ),
                        "review_required": False,
                        "promote_to_bim": True,
                    }
                )
            )
        detections = [item for item in detections if item.id not in consumed] + fused
        icon_junctions = decode_icon_junctions(
            prediction[:21],
            icon_probability,
            source_size=source.size,
        )
        detections = _merge_junction_detections(
            detections,
            icon_junctions,
            icon_classes=self.icon_classes,
        )
        opening_junctions = decode_opening_junctions(
            prediction[:21],
            icon_probability,
            room_class == 2,
            source_size=source.size,
        )
        detections = _merge_junction_detections(
            detections,
            opening_junctions,
            icon_classes=self.icon_classes,
        )
        counts = {
            name: sum(item.class_name == name for item in detections)
            for name in self.icon_classes[1:]
        }
        evidence_counts = {
            mode: sum(item.evidence_mode == mode for item in detections)
            for mode in sorted({item.evidence_mode for item in detections})
        }
        rooms: list[SemanticRoom] = []
        minimum_room_area = max(96, round(width * height * 0.0008))
        # Wall, background, outdoor, and railing pixels are not enclosed BIM spaces.
        room_class_indices = [3, 4, 5, 6, 7, 9, 10, 11]
        for class_index in room_class_indices:
            class_name = self.room_classes[class_index]
            for record in _room_records(
                room_class == class_index,
                room_probability[class_index],
                minimum_area=minimum_room_area,
                source_size=source.size,
            ):
                confidence = float(record["confidence"])
                pixel_area = int(record["pixel_area"])
                rooms.append(
                    SemanticRoom(
                        id=f"semantic:room:{class_index}:{len(rooms)}",
                        class_name=class_name,
                        polygon_px=record["polygon_px"],
                        confidence=confidence,
                        pixel_area=pixel_area,
                        review_required=(confidence < 0.52 or pixel_area < minimum_room_area * 1.5),
                    )
                )
        wall_mask = room_original == 2
        wall_vectors = _refine_wall_vectors_from_raster(
            np.asarray(source.convert("L"), dtype=np.uint8),
            _wall_vectors(wall_mask),
        )
        rooms.extend(
            _recover_unclassified_interior_rooms(
                rooms,
                wall_vectors,
                source_size=source.size,
            )
        )
        room_counts = {
            name: sum(item.class_name == name for item in rooms)
            for name in (*self.room_classes, "Unclassified interior")
            if name not in {"Background", "Outdoor", "Wall", "Railing"}
        }
        result = SemanticRecognitionResult(
            input_path=str(source_path),
            input_sha256=sha256_file(source_path),
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            decoder_settings={"junction_threshold": 0.4, "maximum_peaks_per_channel": 100},
            license_scope=str(self.manifest.get("license_scope") or "unknown"),
            production_authorized=bool(self.manifest.get("production_authorized", False)),
            source_size=source.size,
            model_input_size=(width, height),
            wall_pixels=int(wall_mask.sum()),
            wall_centerlines_px=[(*vector.start_px, *vector.end_px) for vector in wall_vectors],
            wall_vectors_px=wall_vectors,
            detections=detections,
            rooms=rooms,
            counts=counts,
            evidence_counts=evidence_counts,
            room_counts=room_counts,
            inference_ms=round(inference_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
        ).finalize()
        return result, wall_mask

    @staticmethod
    def wall_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelLineProposal]:
        vectors: list[tuple[tuple[float, float, float, float], float | None]] = [
            (
                (*vector.start_px, *vector.end_px),
                vector.thickness_px,
            )
            for vector in result.wall_vectors_px
        ]
        if not vectors:
            vectors = [(line, None) for line in result.wall_centerlines_px]

        return [
            PixelLineProposal(
                id=f"semantic:wall:{index}",
                start_px=(line[0], line[1]),
                end_px=(line[2], line[3]),
                thickness_px=thickness_px,
                confidence=0.82,
                uncertainty=0.18,
                source_ref_ids=source_ref_ids,
                model_version=f"{result.model_version}+{result.decoder_version}",
                review_required=True,
            )
            for index, (line, thickness_px) in enumerate(vectors)
        ]

    @staticmethod
    def symbol_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelSymbolProposal]:
        output = []
        for detection in result.detections:
            if not detection.promote_to_bim:
                continue
            left, top, right, bottom = detection.bbox_px
            output.append(
                PixelSymbolProposal(
                    id=detection.id,
                    symbol_class=detection.symbol_class,
                    center_px=((left + right) / 2, (top + bottom) / 2),
                    bbox_px=(left, top, right, bottom),
                    confidence=detection.confidence,
                    uncertainty=1.0 - detection.confidence,
                    source_ref_ids=source_ref_ids,
                    model_version=f"{result.model_version}+{result.decoder_version}",
                    review_required=detection.review_required,
                )
            )
        return output

    @staticmethod
    def room_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelRoomProposal]:
        return [
            PixelRoomProposal(
                id=room.id,
                name=f"{room.class_name} {index + 1}",
                room_class=room.class_name,
                polygon_px=room.polygon_px,
                confidence=room.confidence,
                uncertainty=1.0 - room.confidence,
                source_ref_ids=source_ref_ids,
                model_version=f"{result.model_version}+{result.decoder_version}",
                review_required=room.review_required,
            )
            for index, room in enumerate(result.rooms)
        ]

    @staticmethod
    def render_overlay(
        image_path: str | Path,
        result: SemanticRecognitionResult,
        wall_mask: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        source = Image.open(image_path).convert("RGBA")
        paint = Image.new("RGBA", source.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(paint)
        wall = Image.fromarray(np.uint8(wall_mask) * 255, mode="L")
        wall_fill = Image.new("RGBA", source.size, (18, 139, 84, 88))
        paint.alpha_composite(Image.composite(wall_fill, Image.new("RGBA", source.size), wall))
        font = _font(max(12, round(source.width / 150)))
        for item in result.detections:
            left, top, right, bottom = item.bbox_px
            color = (
                (222, 122, 20, 255)
                if item.symbol_class in {"door", "window"}
                else (211, 55, 55, 255)
            )
            width = 2 if item.review_required else 4
            draw.rectangle((left, top, right, bottom), outline=color, width=width)
            if item.symbol_class not in {"door", "window"}:
                box = draw.textbbox((0, 0), item.class_name, font=font)
                label_width = box[2] - box[0] + 10
                label_top = max(0, top - (box[3] - box[1] + 7))
                draw.rounded_rectangle(
                    (left, label_top, left + label_width, top),
                    radius=3,
                    fill=(255, 252, 250, 238),
                    outline=color,
                    width=2,
                )
                draw.text(
                    (left + 5, label_top + 2),
                    item.class_name,
                    font=font,
                    fill=(132, 30, 30, 255),
                )
        for x0, y0, x1, y1 in result.wall_centerlines_px:
            draw.line((x0, y0, x1, y1), fill=(17, 92, 62, 210), width=2)
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(source, paint).convert("RGB").save(destination, optimize=True)
        return destination
