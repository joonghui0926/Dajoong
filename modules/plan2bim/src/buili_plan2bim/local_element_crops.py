"""Shared source-coordinate crops for local architectural element inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .core.model.cad_evidence import (
    _ndimage,
    build_cad_evidence,
    letterbox_content_bbox,
)
from .core.model.global_topology_student import ROOM_PROGRAM_CLASSES

LEGACY_LOCAL_ELEMENT_EVIDENCE_CONTRACT = "cad_native_detail_context_pyramid_v2"
WHOLE_SHEET_LOCAL_ELEMENT_EVIDENCE_CONTRACT = (
    "cad_native_detail_context_and_whole_sheet_v3"
)
RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT = (
    "cad_native_detail_context_whole_sheet_and_relations_v4"
)
LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT = (
    "cad_detail_assembly_room_whole_sheet_and_relations_v5"
)
LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT = (
    "cad_focused_detail_assembly_room_whole_sheet_and_relations_v6"
)
LOCAL_ELEMENT_EVIDENCE_CONTRACT = (
    "cad_focused_detail_assembly_room_letterbox_aligned_relations_v7"
)
LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS = 8
LOCAL_ELEMENT_INPUT_CHANNELS = 12
LOCAL_ELEMENT_SEMANTIC_CONTEXT_FEATURES = len(ROOM_PROGRAM_CLASSES) + 3
LEGACY_CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES = 4
CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES = 4
LEGACY_CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT = (
    "nested_proposal_graph_counts_and_extent_ratios_v1"
)
CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT = (
    "nested_proposal_graph_and_aligned_equipment_runs_v2"
)


def _normalized_room_class(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "corridor": "hallway",
        "entry": "hallway",
        "entry-lobby": "hallway",
        "livingroom": "living",
        "living-room": "living",
        "technicalroom": "mechanical",
        "technical-room": "mechanical",
        "laundry": "utility",
        "terrace": "outdoor",
        "balcony": "outdoor",
    }
    value = aliases.get(normalized, normalized)
    return value if value in ROOM_PROGRAM_CLASSES else "other"


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(px - x1, py - y1)
    fraction = min(1.0, max(0.0, ((px - x1) * dx + (py - y1) * dy) / denominator))
    return math.hypot(px - (x1 + fraction * dx), py - (y1 + fraction * dy))


def semantic_element_context(
    proposal_bbox: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
    rooms: list[tuple[str, list[tuple[float, float]]]],
    walls: list[tuple[tuple[float, float], tuple[float, float], float]],
) -> np.ndarray:
    """Encode candidate-to-building relationships, not another pixel thumbnail."""

    left, top, right, bottom = proposal_bbox
    if right <= left or bottom <= top:
        raise ValueError("semantic context requires a positive candidate bbox")
    center = ((left + right) / 2, (top + bottom) / 2)
    room_class = "background"
    inside_known_room = 0.0
    for candidate_class, polygon in rooms:
        if len(polygon) >= 3 and _point_in_polygon(center, polygon):
            room_class = _normalized_room_class(candidate_class)
            inside_known_room = 1.0
            break
    room_one_hot = np.zeros(len(ROOM_PROGRAM_CLASSES), dtype=np.float32)
    room_one_hot[ROOM_PROGRAM_CLASSES.index(room_class)] = 1.0

    diagonal = max(1.0, math.hypot(*image_size))
    nearest_wall_distance = diagonal
    wall_contact = 0.0
    candidate_minor_side = max(1.0, min(right - left, bottom - top))
    for start, end, thickness in walls:
        distance = _point_segment_distance(center, start, end)
        nearest_wall_distance = min(nearest_wall_distance, distance)
        if distance <= max(4.0, thickness * 1.5, candidate_minor_side * 0.65):
            wall_contact = 1.0
    relation = np.asarray(
        (
            inside_known_room,
            min(1.0, nearest_wall_distance / diagonal),
            wall_contact,
        ),
        dtype=np.float32,
    )
    return np.concatenate((room_one_hot, relation))


def normalized_candidate_context(
    proposal_bbox: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
    frame_bbox: tuple[float, float, float, float] | None = None,
    letterbox_size: int | None = None,
) -> np.ndarray:
    """Return candidate origin and extent in the model's complete-plan frame.

    The complete plan is aspect-preserving letterboxed before it reaches the
    model.  Candidate coordinates must undergo the identical transform before
    ``grid_sample`` is used.  The retired contract normalized directly by the
    source width and height, which sampled the wrong location on every
    non-square sheet while appearing correct on square synthetic drawings.
    """

    width, height = image_size
    frame_left, frame_top, frame_right, frame_bottom = frame_bbox or (
        0.0,
        0.0,
        float(width),
        float(height),
    )
    frame_width = frame_right - frame_left
    frame_height = frame_bottom - frame_top
    left, top, right, bottom = proposal_bbox
    if width < 1 or height < 1 or right <= left or bottom <= top:
        raise ValueError("candidate context requires valid image and box dimensions")
    if not (
        0 <= frame_left < frame_right <= width
        and 0 <= frame_top < frame_bottom <= height
        and frame_left <= left < right <= frame_right
        and frame_top <= top < bottom <= frame_bottom
    ):
        raise ValueError("candidate lies outside its whole-plan context frame")
    normalized_left = (left - frame_left) / frame_width
    normalized_top = (top - frame_top) / frame_height
    normalized_width = (right - left) / frame_width
    normalized_height = (bottom - top) / frame_height
    if letterbox_size is not None:
        if letterbox_size < 1:
            raise ValueError("letterbox_size must be positive")
        content_left, content_top, content_right, content_bottom = (
            letterbox_content_bbox(
                (round(frame_width), round(frame_height)),
                letterbox_size,
            )
        )
        content_width = (content_right - content_left) / letterbox_size
        content_height = (content_bottom - content_top) / letterbox_size
        normalized_left = content_left / letterbox_size + normalized_left * content_width
        normalized_top = content_top / letterbox_size + normalized_top * content_height
        normalized_width *= content_width
        normalized_height *= content_height
    return np.asarray(
        (
            normalized_left,
            normalized_top,
            normalized_width,
            normalized_height,
        ),
        dtype=np.float32,
    )


def legacy_candidate_hypothesis_context(
    proposal_bbox: tuple[float, float, float, float],
    candidate_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Encode where one crop sits in the complete proposal hierarchy.

    Native CAD ink deliberately produces tight fragments and larger assembly
    envelopes.  Treating those crops as independent objects loses the very
    distinction the hierarchy was created to preserve.  These bounded features
    describe containing parents and contained children without assigning any
    semantic label; the student can therefore learn when the complete symbol,
    a nested fixture, or a true adjacent object is the correct hypothesis.
    """

    left, top, right, bottom = proposal_bbox
    area = (right - left) * (bottom - top)
    if area <= 0:
        raise ValueError("candidate hypothesis context requires a positive bbox")
    parent_count = 0
    child_count = 0
    largest_parent_scale = 1.0
    largest_child_fraction = 0.0
    for other in candidate_boxes:
        if other == proposal_bbox:
            continue
        other_area = max(0.0, (other[2] - other[0]) * (other[3] - other[1]))
        if other_area <= 0:
            continue
        intersection = max(0.0, min(right, other[2]) - max(left, other[0])) * max(
            0.0, min(bottom, other[3]) - max(top, other[1])
        )
        if intersection <= 0:
            continue
        proposal_coverage = intersection / area
        other_coverage = intersection / other_area
        if other_area >= area * 1.08 and proposal_coverage >= 0.88:
            parent_count += 1
            largest_parent_scale = max(largest_parent_scale, other_area / area)
        elif area >= other_area * 1.08 and other_coverage >= 0.88:
            child_count += 1
            largest_child_fraction = max(largest_child_fraction, other_area / area)
    return np.asarray(
        (
            min(1.0, parent_count / 8.0),
            min(1.0, child_count / 8.0),
            min(1.0, math.log2(largest_parent_scale) / 5.0),
            min(1.0, largest_child_fraction),
        ),
        dtype=np.float32,
    )


def legacy_candidate_hypothesis_contexts(
    candidate_boxes: list[tuple[float, float, float, float]],
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Vectorized proposal-graph context for an entire inference ledger."""

    if block_size < 1:
        raise ValueError("candidate hypothesis block size must be positive")
    if not candidate_boxes:
        return np.zeros((0, CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES), dtype=np.float32)
    boxes = np.asarray(candidate_boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("candidate boxes must have shape [count, 4]")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if np.any(areas <= 0):
        raise ValueError("candidate hypothesis context requires positive boxes")
    output = np.zeros(
        (len(candidate_boxes), CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES),
        dtype=np.float32,
    )
    for start in range(0, len(candidate_boxes), block_size):
        stop = min(len(candidate_boxes), start + block_size)
        current = boxes[start:stop]
        current_area = areas[start:stop, None]
        intersection = np.maximum(
            0.0,
            np.minimum(current[:, None, 2], boxes[None, :, 2])
            - np.maximum(current[:, None, 0], boxes[None, :, 0]),
        ) * np.maximum(
            0.0,
            np.minimum(current[:, None, 3], boxes[None, :, 3])
            - np.maximum(current[:, None, 1], boxes[None, :, 1]),
        )
        proposal_coverage = intersection / current_area
        other_coverage = intersection / areas[None, :]
        distinct = np.ones_like(intersection, dtype=np.bool_)
        row = np.arange(stop - start)
        distinct[row, np.arange(start, stop)] = False
        parent = (
            distinct
            & (areas[None, :] >= current_area * 1.08)
            & (proposal_coverage >= 0.88)
        )
        child = (
            distinct
            & (current_area >= areas[None, :] * 1.08)
            & (other_coverage >= 0.88)
        )
        parent_scale = np.where(parent, areas[None, :] / current_area, 1.0)
        child_fraction = np.where(child, areas[None, :] / current_area, 0.0)
        output[start:stop] = np.stack(
            (
                np.minimum(1.0, parent.sum(axis=1) / 8.0),
                np.minimum(1.0, child.sum(axis=1) / 8.0),
                np.minimum(1.0, np.log2(parent_scale.max(axis=1)) / 5.0),
                np.minimum(1.0, child_fraction.max(axis=1)),
            ),
            axis=1,
        ).astype(np.float32)
    return output


def candidate_hypothesis_contexts(
    candidate_boxes: list[tuple[float, float, float, float]],
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Describe proposal nesting and coherent aligned equipment runs.

    A cabinet, sink, range, and appliance are not independent marks when they
    form one wall-hosted run.  The retired graph encoded only boxes contained
    inside other boxes, so an open-plan kitchen lost its program as soon as the
    room label said ``living``.  The final two features now measure strong
    horizontal and vertical adjacency while retaining parent/child counts.
    """

    if not candidate_boxes:
        return np.zeros((0, CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES), dtype=np.float32)
    boxes = np.asarray(candidate_boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("candidate boxes must have shape [count, 4]")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    if np.any(areas <= 0):
        raise ValueError("candidate hypothesis context requires positive boxes")
    # Index expanded envelopes instead of allocating an N-by-N relation matrix.
    # Expanding each side by 42.5% is an exact spatial superset for the run gate
    # below: gap <= 0.425 * (left_size + right_size).  Boxes that span many grid
    # cells are kept in a small global list, so large parent hypotheses can never
    # disappear from child containment queries.
    extent_width = max(1.0, float(boxes[:, 2].max() - boxes[:, 0].min()))
    extent_height = max(1.0, float(boxes[:, 3].max() - boxes[:, 1].min()))
    cell_size = max(8.0, min(extent_width, extent_height) / 64.0)
    expanded = boxes.copy()
    expanded[:, 0] -= widths * 0.425
    expanded[:, 2] += widths * 0.425
    expanded[:, 1] -= heights * 0.425
    expanded[:, 3] += heights * 0.425
    buckets: dict[tuple[int, int], list[int]] = {}
    global_indices: list[int] = []
    for index, envelope in enumerate(expanded):
        x0 = math.floor(envelope[0] / cell_size)
        y0 = math.floor(envelope[1] / cell_size)
        x1 = math.floor(envelope[2] / cell_size)
        y1 = math.floor(envelope[3] / cell_size)
        cell_count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if cell_count > 256:
            global_indices.append(index)
            continue
        for cell_y in range(y0, y1 + 1):
            for cell_x in range(x0, x1 + 1):
                buckets.setdefault((cell_x, cell_y), []).append(index)

    output = np.zeros(
        (len(candidate_boxes), CANDIDATE_HYPOTHESIS_CONTEXT_FEATURES),
        dtype=np.float32,
    )
    for index, envelope in enumerate(expanded):
        x0 = math.floor(envelope[0] / cell_size)
        y0 = math.floor(envelope[1] / cell_size)
        x1 = math.floor(envelope[2] / cell_size)
        y1 = math.floor(envelope[3] / cell_size)
        neighbors = set(global_indices)
        for cell_y in range(y0, y1 + 1):
            for cell_x in range(x0, x1 + 1):
                neighbors.update(buckets.get((cell_x, cell_y), ()))
        # Exact relation predicates still run after the spatial superset query.
        # This preserves the v2 feature contract while avoiding work between
        # candidates that cannot overlap, contain, or form one equipment run.
        output[index] = candidate_hypothesis_context(
            candidate_boxes[index],
            [candidate_boxes[item] for item in sorted(neighbors)],
        )
    return output


def candidate_hypothesis_context(
    proposal_bbox: tuple[float, float, float, float],
    candidate_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Scalar convenience wrapper for the aligned-run graph contract."""

    legacy = legacy_candidate_hypothesis_context(proposal_bbox, candidate_boxes)
    left, top, right, bottom = proposal_bbox
    width = right - left
    height = bottom - top
    area = width * height
    if area <= 0:
        raise ValueError("candidate hypothesis context requires a positive bbox")
    horizontal_scores: list[float] = []
    vertical_scores: list[float] = []
    for other in candidate_boxes:
        if other == proposal_bbox:
            continue
        other_width = other[2] - other[0]
        other_height = other[3] - other[1]
        other_area = other_width * other_height
        if other_area < area * 0.22 or other_area > area * 4.5:
            continue
        overlap_x = max(0.0, min(right, other[2]) - max(left, other[0]))
        overlap_y = max(0.0, min(bottom, other[3]) - max(top, other[1]))
        if overlap_x * overlap_y > min(area, other_area) * 0.08:
            continue
        gap_x = max(0.0, max(left, other[0]) - min(right, other[2]))
        gap_y = max(0.0, max(top, other[1]) - min(bottom, other[3]))
        horizontal_scale = max(1.0, (width + other_width) * 0.5)
        vertical_scale = max(1.0, (height + other_height) * 0.5)
        horizontal_alignment = overlap_y / max(1.0, min(height, other_height))
        vertical_alignment = overlap_x / max(1.0, min(width, other_width))
        if horizontal_alignment >= 0.55 and gap_x <= horizontal_scale * 0.85:
            horizontal_scores.append(
                horizontal_alignment * math.exp(-gap_x / horizontal_scale)
            )
        if vertical_alignment >= 0.55 and gap_y <= vertical_scale * 0.85:
            vertical_scores.append(
                vertical_alignment * math.exp(-gap_y / vertical_scale)
            )
    horizontal_run = sum(sorted(horizontal_scores)[-2:]) / 2.0
    vertical_run = sum(sorted(vertical_scores)[-2:]) / 2.0
    return np.asarray(
        (
            legacy[0],
            legacy[1],
            min(1.0, horizontal_run),
            min(1.0, vertical_run),
        ),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class LocalElementCropTransform:
    source_bbox: tuple[float, float, float, float]
    input_size: int

    @property
    def side_px(self) -> float:
        return self.source_bbox[2] - self.source_bbox[0]


def focus_candidate_detail_evidence(
    evidence: np.ndarray,
    proposal_bbox: tuple[float, float, float, float],
    transform: LocalElementCropTransform,
    *,
    outside_attenuation: float = 0.08,
    halo_ratio: float = 0.08,
) -> np.ndarray:
    """Keep target ink explicit while retaining context in the other views.

    Architectural symbols are commonly attached to walls, cabinet runs, text,
    or MEP lines.  An unmarked crop asks the classifier to guess which of those
    competing marks the proposal miner intended.  The detail view now carries
    an anti-aliased proposal focus: pixels inside the proposed extent retain
    full evidence, a small halo fades smoothly, and distant context is strongly
    attenuated.  Assembly and room views remain untouched, so the model sees
    both the target and the surrounding building relationship.
    """

    value = np.asarray(evidence, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 4:
        raise ValueError("focused detail evidence must have four channels")
    if not 0.0 <= outside_attenuation < 1.0 or not 0.0 <= halo_ratio <= 0.5:
        raise ValueError("invalid focused detail attenuation")
    crop_left, crop_top, crop_right, crop_bottom = transform.source_bbox
    side = crop_right - crop_left
    if side <= 0:
        raise ValueError("focused detail transform must have positive extent")
    left, top, right, bottom = proposal_bbox
    if right <= left or bottom <= top:
        raise ValueError("focused detail proposal must have positive extent")
    height, width = value.shape[-2:]
    x = crop_left + (np.arange(width, dtype=np.float32) + 0.5) * side / width
    y = crop_top + (np.arange(height, dtype=np.float32) + 0.5) * side / height
    distance_x = np.maximum(np.maximum(left - x, x - right), 0.0)
    distance_y = np.maximum(np.maximum(top - y, y - bottom), 0.0)
    distance = np.hypot(distance_y[:, None], distance_x[None, :])
    halo = max(1.0, min(right - left, bottom - top) * halo_ratio)
    focus = np.clip(1.0 - distance / halo, 0.0, 1.0)
    focus = outside_attenuation + (1.0 - outside_attenuation) * focus
    return (value * focus[None]).astype(np.float32)


def focus_candidate_detail_evidence_batch(
    evidence: np.ndarray,
    proposal_boxes: list[tuple[float, float, float, float]],
    transforms: list[LocalElementCropTransform],
    *,
    outside_attenuation: float = 0.08,
    halo_ratio: float = 0.08,
) -> np.ndarray:
    """Vectorized equivalent of :func:`focus_candidate_detail_evidence`.

    Dense real sheets can contain thousands of native hypotheses.  Creating
    two coordinate arrays and one attenuation mask in Python for every box
    made crop preparation dominate CPU latency.  This implementation keeps
    exactly the same per-hypothesis focus contract while evaluating a complete
    inference batch as one array operation.
    """

    value = np.asarray(evidence, dtype=np.float32)
    if value.ndim != 4 or value.shape[1] != 4:
        raise ValueError("focused detail evidence batch must have four channels")
    if len(value) != len(proposal_boxes) or len(value) != len(transforms):
        raise ValueError("focused detail batch inputs must have equal length")
    if not 0.0 <= outside_attenuation < 1.0 or not 0.0 <= halo_ratio <= 0.5:
        raise ValueError("invalid focused detail attenuation")
    if not proposal_boxes:
        return value.copy()

    boxes = np.asarray(proposal_boxes, dtype=np.float32)
    crop_boxes = np.asarray(
        [transform.source_bbox for transform in transforms],
        dtype=np.float32,
    )
    sides = crop_boxes[:, 2] - crop_boxes[:, 0]
    if np.any(sides <= 0):
        raise ValueError("focused detail transforms must have positive extent")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("focused detail proposals must have positive extent")

    height, width = value.shape[-2:]
    x_unit = (np.arange(width, dtype=np.float32) + 0.5) / width
    y_unit = (np.arange(height, dtype=np.float32) + 0.5) / height
    x = crop_boxes[:, 0, None] + sides[:, None] * x_unit[None, :]
    y = crop_boxes[:, 1, None] + sides[:, None] * y_unit[None, :]
    distance_x = np.maximum(
        np.maximum(boxes[:, 0, None] - x, x - boxes[:, 2, None]),
        0.0,
    )
    distance_y = np.maximum(
        np.maximum(boxes[:, 1, None] - y, y - boxes[:, 3, None]),
        0.0,
    )
    distance = np.hypot(distance_y[:, :, None], distance_x[:, None, :])
    halo = np.maximum(1.0, np.minimum(widths, heights) * halo_ratio)
    focus = np.clip(1.0 - distance / halo[:, None, None], 0.0, 1.0)
    focus = outside_attenuation + (1.0 - outside_attenuation) * focus
    return (value * focus[:, None]).astype(np.float32)


def element_geometry_target(
    element_bbox: tuple[float, float, float, float],
    transform: LocalElementCropTransform,
    *,
    yaw_deg: float,
) -> np.ndarray:
    left, top, right, bottom = element_bbox
    crop_left, crop_top, _, _ = transform.source_bbox
    side = transform.side_px
    center_x = ((left + right) / 2 - crop_left) / side
    center_y = ((top + bottom) / 2 - crop_top) / side
    width = max(1e-6, (right - left) / side)
    height = max(1e-6, (bottom - top) / side)
    yaw = math.radians(yaw_deg)
    return np.asarray(
        (
            center_x - 0.5,
            center_y - 0.5,
            math.log(width),
            math.log(height),
            math.sin(yaw),
            math.cos(yaw),
        ),
        dtype=np.float32,
    )


def decode_element_geometry(
    geometry: np.ndarray,
    transform: LocalElementCropTransform,
) -> tuple[tuple[float, float, float, float], float]:
    value = np.asarray(geometry, dtype=np.float64)
    if value.shape != (6,):
        raise ValueError("local element geometry must have six values")
    crop_left, crop_top, _, _ = transform.source_bbox
    side = transform.side_px
    center_x = crop_left + (float(value[0]) + 0.5) * side
    center_y = crop_top + (float(value[1]) + 0.5) * side
    width = float(np.exp(np.clip(value[2], -8.0, 1.0))) * side
    height = float(np.exp(np.clip(value[3], -8.0, 1.0))) * side
    bbox = (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )
    yaw = math.degrees(math.atan2(float(value[4]), float(value[5])))
    return bbox, yaw


def extract_local_element_evidence(
    image: Image.Image,
    proposal_bbox: tuple[float, float, float, float],
    *,
    input_size: int = 64,
    context_scale: float = 2.0,
    center_jitter: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, LocalElementCropTransform]:
    full_evidence = build_cad_evidence(image.convert("RGB"))
    return extract_local_element_evidence_from_map(
        full_evidence,
        image.size,
        proposal_bbox,
        input_size=input_size,
        context_scale=context_scale,
        center_jitter=center_jitter,
    )


def extract_local_element_evidence_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_bbox: tuple[float, float, float, float],
    *,
    input_size: int = 64,
    context_scale: float = 2.0,
    center_jitter: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, LocalElementCropTransform]:
    """Crop a shared native evidence map without recomputing filters per object."""

    if input_size < 32 or context_scale < 1.0:
        raise ValueError("local crop input_size or context_scale is invalid")
    evidence_map = np.asarray(full_evidence, dtype=np.float32)
    if (
        evidence_map.ndim != 3
        or evidence_map.shape[0] != 4
        or evidence_map.shape[2] != image_size[0]
        or evidence_map.shape[1] != image_size[1]
    ):
        raise ValueError("full_evidence must align with image_size and have four channels")
    left, top, right, bottom = proposal_bbox
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("proposal_bbox must have positive area")
    side = max(8.0, max(width, height) * context_scale)
    center_x = (left + right) / 2 + center_jitter[0] * side
    center_y = (top + bottom) / 2 + center_jitter[1] * side
    crop_bbox = (
        center_x - side / 2,
        center_y - side / 2,
        center_x + side / 2,
        center_y + side / 2,
    )
    integer_left = math.floor(crop_bbox[0])
    integer_top = math.floor(crop_bbox[1])
    integer_right = math.ceil(crop_bbox[2])
    integer_bottom = math.ceil(crop_bbox[3])
    integer_side = max(integer_right - integer_left, integer_bottom - integer_top)
    integer_right = integer_left + integer_side
    integer_bottom = integer_top + integer_side
    # Sample directly into the fixed-size tensor. PIL.crop materializes the
    # complete padded crop before resizing; one malformed or whole-room proposal
    # could therefore allocate a 10k x 10k temporary image for every channel.
    # Fixed-grid interpolation keeps work O(input_size²), independent of bbox size.
    sample_x = (
        integer_left
        + (np.arange(input_size, dtype=np.float32) + 0.5)
        * (integer_side / input_size)
        - 0.5
    )
    sample_y = (
        integer_top
        + (np.arange(input_size, dtype=np.float32) + 0.5)
        * (integer_side / input_size)
        - 0.5
    )
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    coordinates = np.stack((grid_y, grid_x))
    ndimage = _ndimage()
    evidence = np.stack(
        [
            ndimage.map_coordinates(
                channel,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for channel in evidence_map
        ]
    )
    transform = LocalElementCropTransform(
        source_bbox=(
            float(integer_left),
            float(integer_top),
            float(integer_right),
            float(integer_bottom),
        ),
        input_size=input_size,
    )
    return evidence.astype(np.float32), transform


def extract_local_element_pyramid_evidence_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_bbox: tuple[float, float, float, float],
    *,
    input_size: int = 64,
    detail_scale: float = 2.0,
    context_scale: float = 5.5,
    center_jitter: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, LocalElementCropTransform]:
    """Join native symbol detail with a co-centered room/wall context crop.

    Geometry is encoded against the first four detail channels.  The second
    four channels keep the same center but cover enough surrounding plan to
    distinguish visually similar symbols by their wall and room relationships.
    """

    if context_scale <= detail_scale:
        raise ValueError("context_scale must be larger than detail_scale")
    detail, transform = extract_local_element_evidence_from_map(
        full_evidence,
        image_size,
        proposal_bbox,
        input_size=input_size,
        context_scale=detail_scale,
        center_jitter=center_jitter,
    )
    center_ratio = detail_scale / context_scale
    context, _ = extract_local_element_evidence_from_map(
        full_evidence,
        image_size,
        proposal_bbox,
        input_size=input_size,
        context_scale=context_scale,
        center_jitter=(
            center_jitter[0] * center_ratio,
            center_jitter[1] * center_ratio,
        ),
    )
    return np.concatenate((detail, context), axis=0), transform


def _batch_crop_evidence_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_boxes: list[tuple[float, float, float, float]],
    *,
    input_size: int,
    context_scale: float,
) -> tuple[np.ndarray, list[LocalElementCropTransform]]:
    """Vectorized fixed-grid crops for one inference batch."""

    if input_size < 32 or context_scale < 1.0:
        raise ValueError("local crop input_size or context_scale is invalid")
    evidence_map = np.asarray(full_evidence, dtype=np.float32)
    if (
        evidence_map.ndim != 3
        or evidence_map.shape[0] != 4
        or evidence_map.shape[2] != image_size[0]
        or evidence_map.shape[1] != image_size[1]
    ):
        raise ValueError("full_evidence must align with image_size and have four channels")
    if not proposal_boxes:
        return np.zeros((0, 4, input_size, input_size), dtype=np.float32), []
    lefts: list[int] = []
    tops: list[int] = []
    sides: list[int] = []
    transforms: list[LocalElementCropTransform] = []
    for left, top, right, bottom in proposal_boxes:
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise ValueError("proposal_bbox must have positive area")
        side = max(8.0, max(width, height) * context_scale)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        integer_left = math.floor(center_x - side / 2)
        integer_top = math.floor(center_y - side / 2)
        integer_right = math.ceil(center_x + side / 2)
        integer_bottom = math.ceil(center_y + side / 2)
        integer_side = max(integer_right - integer_left, integer_bottom - integer_top)
        lefts.append(integer_left)
        tops.append(integer_top)
        sides.append(integer_side)
        transforms.append(
            LocalElementCropTransform(
                source_bbox=(
                    float(integer_left),
                    float(integer_top),
                    float(integer_left + integer_side),
                    float(integer_top + integer_side),
                ),
                input_size=input_size,
            )
        )
    left_array = np.asarray(lefts, dtype=np.float32)[:, None, None]
    top_array = np.asarray(tops, dtype=np.float32)[:, None, None]
    side_array = np.asarray(sides, dtype=np.float32)[:, None, None]
    unit_x = (np.arange(input_size, dtype=np.float32) + 0.5)[None, None, :]
    unit_y = (np.arange(input_size, dtype=np.float32) + 0.5)[None, :, None]
    grid_x = np.broadcast_to(
        left_array + unit_x * (side_array / input_size) - 0.5,
        (len(proposal_boxes), input_size, input_size),
    )
    grid_y = np.broadcast_to(
        top_array + unit_y * (side_array / input_size) - 0.5,
        (len(proposal_boxes), input_size, input_size),
    )
    coordinates = np.stack((grid_y, grid_x))
    ndimage = _ndimage()
    crops = np.stack(
        [
            ndimage.map_coordinates(
                channel,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for channel in evidence_map
        ],
        axis=1,
    )
    return crops.astype(np.float32), transforms


def extract_local_element_pyramid_batch_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_boxes: list[tuple[float, float, float, float]],
    *,
    input_size: int = 64,
    detail_scale: float = 2.0,
    context_scale: float = 5.5,
) -> tuple[np.ndarray, list[LocalElementCropTransform]]:
    """Create a native-detail/context pyramid for all candidates in one batch."""

    if context_scale <= detail_scale:
        raise ValueError("context_scale must be larger than detail_scale")
    detail, transforms = _batch_crop_evidence_from_map(
        full_evidence,
        image_size,
        proposal_boxes,
        input_size=input_size,
        context_scale=detail_scale,
    )
    context, _ = _batch_crop_evidence_from_map(
        full_evidence,
        image_size,
        proposal_boxes,
        input_size=input_size,
        context_scale=context_scale,
    )
    return np.concatenate((detail, context), axis=1), transforms


def extract_local_element_hierarchy_evidence_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_bbox: tuple[float, float, float, float],
    *,
    input_size: int = 64,
    detail_scale: float = 2.1,
    assembly_scale: float = 6.5,
    room_scale: float = 18.0,
    center_jitter: tuple[float, float] = (0.0, 0.0),
    focus_detail: bool = True,
) -> tuple[np.ndarray, LocalElementCropTransform]:
    """Read an object through object, assembly, and room-level source views."""

    if not 1.0 <= detail_scale < assembly_scale < room_scale:
        raise ValueError("local hierarchy scales must be strictly increasing")
    views = []
    transform = None
    for scale in (detail_scale, assembly_scale, room_scale):
        scale_jitter = detail_scale / scale
        evidence, candidate_transform = extract_local_element_evidence_from_map(
            full_evidence,
            image_size,
            proposal_bbox,
            input_size=input_size,
            context_scale=scale,
            center_jitter=(
                center_jitter[0] * scale_jitter,
                center_jitter[1] * scale_jitter,
            ),
        )
        views.append(evidence)
        if transform is None:
            transform = candidate_transform
    assert transform is not None
    if focus_detail:
        views[0] = focus_candidate_detail_evidence(
            views[0],
            proposal_bbox,
            transform,
        )
    return np.concatenate(views, axis=0), transform


def extract_local_element_hierarchy_batch_from_map(
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    proposal_boxes: list[tuple[float, float, float, float]],
    *,
    input_size: int = 64,
    detail_scale: float = 2.1,
    assembly_scale: float = 6.5,
    room_scale: float = 18.0,
    focus_detail: bool = True,
) -> tuple[np.ndarray, list[LocalElementCropTransform]]:
    """Vectorized object, assembly, and room context tensor."""

    if not 1.0 <= detail_scale < assembly_scale < room_scale:
        raise ValueError("local hierarchy scales must be strictly increasing")
    detail, transforms = _batch_crop_evidence_from_map(
        full_evidence,
        image_size,
        proposal_boxes,
        input_size=input_size,
        context_scale=detail_scale,
    )
    assembly, _ = _batch_crop_evidence_from_map(
        full_evidence,
        image_size,
        proposal_boxes,
        input_size=input_size,
        context_scale=assembly_scale,
    )
    room, _ = _batch_crop_evidence_from_map(
        full_evidence,
        image_size,
        proposal_boxes,
        input_size=input_size,
        context_scale=room_scale,
    )
    if focus_detail:
        detail = focus_candidate_detail_evidence_batch(
            detail,
            proposal_boxes,
            transforms,
        )
    return np.concatenate((detail, assembly, room), axis=1), transforms
