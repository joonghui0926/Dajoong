"""Joint whole-sheet decoder for the Dajoong building program.

Dense probabilities are evidence, not BIM entities.  This decoder recovers one
room partition and one connected wall program first, then admits openings and
equipment into that shared coordinate system.  Accepted wall vectors are
optionally aligned against the native source raster so a small global model does
not force the final geometry to remain at its normalized inference resolution.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .core.model.aec_decode import (
    AecTileProposal,
    PixelLineProposal,
    PixelRoomProposal,
    PixelSymbolProposal,
)
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
)
from .semantic_recognition import (
    SemanticWallVector,
    _mask_outer_loop,
    _polygon_area,
    _refine_wall_vectors_from_raster,
    _simplify_loop,
    _wall_vectors,
)


class GlobalProgramDecodeDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.global-program-decode-diagnostics.v1"
    source_size: tuple[int, int]
    model_size: tuple[int, int]
    room_instance_count: int = Field(ge=0)
    structural_wall_count: int = Field(ge=0)
    rejected_wall_count: int = Field(ge=0)
    element_count: int = Field(ge=0)
    native_wall_refinement_applied: bool
    native_element_refinement_count: int = Field(default=0, ge=0)
    full_sheet_context: bool = True
    release_eligible: bool = False
    release_blockers: list[str] = Field(default_factory=list)


class GlobalProgramDecodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.global-program-decode.v1"
    proposal: AecTileProposal
    diagnostics: GlobalProgramDecodeDiagnostics
    room_semantic_seeds: list[PixelRoomProposal] = Field(default_factory=list)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = np.asarray(value, dtype=np.float32)
    shifted = shifted - shifted.max(axis=0, keepdims=True)
    exponential = np.exp(np.clip(shifted, -30.0, 30.0))
    return exponential / np.maximum(exponential.sum(axis=0, keepdims=True), 1e-9)


def _validate_outputs(
    topology_logits: np.ndarray,
    room_semantic_logits: np.ndarray,
    element_semantic_logits: np.ndarray,
    element_geometry: np.ndarray,
    uncertainty: np.ndarray,
) -> tuple[int, int]:
    topology = np.asarray(topology_logits)
    rooms = np.asarray(room_semantic_logits)
    elements = np.asarray(element_semantic_logits)
    geometry = np.asarray(element_geometry)
    uncertainty_map = np.asarray(uncertainty)
    if topology.ndim != 3 or topology.shape[0] != len(TOPOLOGY_TARGET_CHANNELS):
        raise ValueError("topology_logits do not match the global topology contract")
    height, width = topology.shape[1:]
    expected_spatial = (height, width)
    if rooms.shape != (len(ROOM_PROGRAM_CLASSES), *expected_spatial):
        raise ValueError("room_semantic_logits do not match the room program contract")
    if elements.shape != (len(ELEMENT_PROGRAM_CLASSES), *expected_spatial):
        raise ValueError("element_semantic_logits do not match the element program contract")
    if geometry.shape != (len(ELEMENT_GEOMETRY_CHANNELS), *expected_spatial):
        raise ValueError("element_geometry does not match the geometry contract")
    if uncertainty_map.shape != (3, *expected_spatial):
        raise ValueError("uncertainty must be [3, height, width]")
    return width, height


def _room_instances(
    topology_probability: np.ndarray,
    room_probability: np.ndarray,
    *,
    source_size: tuple[int, int],
    room_threshold: float,
) -> tuple[np.ndarray, list[PixelRoomProposal]]:
    height, width = topology_probability.shape[1:]
    wall = topology_probability[1] >= 0.45
    interior = topology_probability[5] >= room_threshold
    traversable = interior & ~ndimage.binary_dilation(wall, iterations=1)
    labels, count = ndimage.label(
        traversable,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8),
    )
    minimum_area = max(20, round(width * height * 0.00035))
    scale_x = source_size[0] / width
    scale_y = source_size[1] / height
    rooms: list[PixelRoomProposal] = []
    retained_labels = np.zeros_like(labels, dtype=np.int32)
    retained_index = 0
    for component in range(1, count + 1):
        mask = labels == component
        pixel_area = int(mask.sum())
        if pixel_area < minimum_area:
            continue
        class_scores = np.zeros(room_probability.shape[0], dtype=np.float32)
        sample_count = max(4, min(pixel_area, round(pixel_area * 0.015)))
        for class_index in range(1, room_probability.shape[0]):
            values = room_probability[class_index, mask]
            top = np.partition(values, len(values) - sample_count)[-sample_count:]
            class_scores[class_index] = float(top.mean())
        class_scores[0] = -1.0
        class_index = int(class_scores.argmax())
        class_name = ROOM_PROGRAM_CLASSES[class_index]
        loop = _mask_outer_loop(mask)
        if len(loop) < 3:
            continue
        simplified = _simplify_loop(loop, tolerance=1.25)
        polygon = [(x * scale_x, y * scale_y) for x, y in simplified]
        if len(polygon) < 3 or _polygon_area(polygon) < minimum_area * scale_x * scale_y:
            continue
        retained_index += 1
        retained_labels[mask] = retained_index
        confidence = float(class_scores[class_index])
        rooms.append(
            PixelRoomProposal(
                id=f"global:room:{retained_index - 1}",
                name=f"{class_name} {retained_index}",
                room_class=class_name,
                polygon_px=polygon,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                source_ref_ids=["pending"],
                model_version="pending",
                review_required=confidence < 0.72,
            )
        )
    return retained_labels, rooms


def _localized_room_semantic_seeds(
    topology_probability: np.ndarray,
    room_probability: np.ndarray,
    *,
    source_size: tuple[int, int],
) -> list[PixelRoomProposal]:
    """Preserve room-name seeds independently of provisional room geometry."""

    seed_probability = topology_probability[4]
    height, width = seed_probability.shape
    class_index_map = np.argmax(room_probability[1:], axis=0) + 1
    class_probability = np.max(room_probability[1:], axis=0)
    background_probability = room_probability[0]
    support = (
        (seed_probability >= 0.45)
        & (class_probability >= 0.18)
        & (class_probability > background_probability)
    )
    labels, count = ndimage.label(
        support,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8),
    )
    minimum_area = max(2, round(width * height * 0.00004))
    scale_x = source_size[0] / width
    scale_y = source_size[1] / height
    seeds: list[PixelRoomProposal] = []
    for component in range(1, count + 1):
        mask = labels == component
        if int(mask.sum()) < minimum_area:
            continue
        votes = np.bincount(
            class_index_map[mask],
            weights=class_probability[mask] * seed_probability[mask],
            minlength=len(ROOM_PROGRAM_CLASSES),
        )
        class_index = int(np.argmax(votes[1:]) + 1)
        class_mask = mask & (class_index_map == class_index)
        if not class_mask.any():
            continue
        values = class_probability[class_mask]
        sample_count = max(1, min(len(values), round(len(values) * 0.20)))
        confidence = float(
            np.partition(values, len(values) - sample_count)[-sample_count:].mean()
        )
        loop = _mask_outer_loop(mask)
        polygon = _simplify_loop(loop, tolerance=0.75)
        if len(polygon) < 3:
            rows, columns = np.nonzero(mask)
            left = float(columns.min())
            top = float(rows.min())
            right = float(columns.max() + 1)
            bottom = float(rows.max() + 1)
            polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
        class_name = ROOM_PROGRAM_CLASSES[class_index]
        seeds.append(
            PixelRoomProposal(
                id=f"global:room-semantic-seed:{len(seeds)}",
                name=f"{class_name} seed {len(seeds) + 1}",
                room_class=class_name,
                polygon_px=[(x * scale_x, y * scale_y) for x, y in polygon],
                confidence=confidence,
                uncertainty=1.0 - confidence,
                source_ref_ids=["pending"],
                model_version="pending+localized-room-semantic-seed-v2",
                review_required=confidence < 0.72,
            )
        )
    return seeds


def _sample_line(image: np.ndarray, line: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = line
    count = max(5, min(128, round(math.dist((x0, y0), (x1, y1)) * 1.5)))
    xs = np.clip(np.rint(np.linspace(x0, x1, count)).astype(int), 0, image.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y0, y1, count)).astype(int), 0, image.shape[0] - 1)
    return image[ys, xs]


def _structural_support(
    vector: SemanticWallVector,
    room_instances: np.ndarray,
    exterior_probability: np.ndarray,
) -> float:
    x0, y0 = vector.start_px
    x1, y1 = vector.end_px
    length = math.dist((x0, y0), (x1, y1))
    if length <= 1e-6:
        return 0.0
    direction = np.asarray((x1 - x0, y1 - y0), dtype=np.float64) / length
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    count = max(7, min(96, round(length / 2)))
    fractions = np.linspace(0.08, 0.92, count)
    centers = np.asarray((x0, y0))[None, :] + fractions[:, None] * np.asarray(
        (x1 - x0, y1 - y0)
    )[None, :]
    offset = max(2.0, min(8.0, vector.thickness_px * 0.75))
    left = centers - normal[None, :] * offset
    right = centers + normal[None, :] * offset

    def sample_labels(points: np.ndarray) -> np.ndarray:
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, room_instances.shape[1] - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, room_instances.shape[0] - 1)
        return room_instances[ys, xs]

    left_labels = sample_labels(left)
    right_labels = sample_labels(right)
    separated = (left_labels != right_labels) & ((left_labels > 0) | (right_labels > 0))
    exterior = float(_sample_line(exterior_probability, (x0, y0, x1, y1)).mean())
    return max(float(separated.mean()), exterior)


def _close_wall_graph(vectors: list[SemanticWallVector]) -> list[SemanticWallVector]:
    """Extend raster-trimmed wall axes to supported orthogonal junctions."""

    output: list[SemanticWallVector] = []
    for vector in vectors:
        x0, y0 = vector.start_px
        x1, y1 = vector.end_px
        delta_x, delta_y = abs(x1 - x0), abs(y1 - y0)
        horizontal = delta_x >= delta_y
        if min(delta_x, delta_y) > max(delta_x, delta_y) * 0.2:
            output.append(vector)
            continue
        tolerance = max(6.0, vector.thickness_px * 3.0)
        if horizontal:
            coordinate = (y0 + y1) / 2
            start, end = sorted((x0, x1))
            for candidate in vectors:
                cx0, cy0 = candidate.start_px
                cx1, cy1 = candidate.end_px
                if abs(cy1 - cy0) < abs(cx1 - cx0):
                    continue
                crossing_x = (cx0 + cx1) / 2
                candidate_top, candidate_bottom = sorted((cy0, cy1))
                if not (
                    start - tolerance <= crossing_x <= end + tolerance
                    and candidate_top - tolerance <= coordinate <= candidate_bottom + tolerance
                ):
                    continue
                if crossing_x <= start + tolerance:
                    start = min(start, crossing_x)
                if crossing_x >= end - tolerance:
                    end = max(end, crossing_x)
            output.append(
                SemanticWallVector(
                    start_px=(start, coordinate),
                    end_px=(end, coordinate),
                    thickness_px=vector.thickness_px,
                )
            )
        else:
            coordinate = (x0 + x1) / 2
            start, end = sorted((y0, y1))
            for candidate in vectors:
                cx0, cy0 = candidate.start_px
                cx1, cy1 = candidate.end_px
                if abs(cx1 - cx0) < abs(cy1 - cy0):
                    continue
                crossing_y = (cy0 + cy1) / 2
                candidate_left, candidate_right = sorted((cx0, cx1))
                if not (
                    start - tolerance <= crossing_y <= end + tolerance
                    and candidate_left - tolerance <= coordinate <= candidate_right + tolerance
                ):
                    continue
                if crossing_y <= start + tolerance:
                    start = min(start, crossing_y)
                if crossing_y >= end - tolerance:
                    end = max(end, crossing_y)
            output.append(
                SemanticWallVector(
                    start_px=(coordinate, start),
                    end_px=(coordinate, end),
                    thickness_px=vector.thickness_px,
                )
            )
    return output


def _decode_walls(
    topology_probability: np.ndarray,
    room_instances: np.ndarray,
    uncertainty: np.ndarray,
    *,
    source_size: tuple[int, int],
    source_gray: np.ndarray | None,
    source_ref_ids: list[str],
    model_version: str,
    wall_threshold: float,
    structural_support_threshold: float,
) -> tuple[list[PixelLineProposal], int, bool]:
    model_height, model_width = topology_probability.shape[1:]
    wall_mask = topology_probability[1] >= wall_threshold
    vectors = _close_wall_graph(_wall_vectors(wall_mask))
    scale_x = source_size[0] / model_width
    scale_y = source_size[1] / model_height
    scale_thickness = (scale_x + scale_y) / 2
    retained: list[tuple[SemanticWallVector, float, float, float]] = []
    rejected = 0
    for vector in vectors:
        line = (*vector.start_px, *vector.end_px)
        evidence = float(_sample_line(topology_probability[1], line).mean())
        support = _structural_support(vector, room_instances, topology_probability[0])
        line_uncertainty = float(_sample_line(uncertainty[0], line).mean())
        if evidence < wall_threshold or support < structural_support_threshold:
            rejected += 1
            continue
        retained.append((vector, evidence, support, line_uncertainty))

    scaled_vectors = [
        SemanticWallVector(
            start_px=(item[0].start_px[0] * scale_x, item[0].start_px[1] * scale_y),
            end_px=(item[0].end_px[0] * scale_x, item[0].end_px[1] * scale_y),
            thickness_px=max(1.0, item[0].thickness_px * scale_thickness),
        )
        for item in retained
    ]
    refinement_applied = source_gray is not None and bool(scaled_vectors)
    if source_gray is not None:
        gray = np.asarray(source_gray, dtype=np.uint8)
        if gray.shape != (source_size[1], source_size[0]):
            raise ValueError("source_gray must match source_size")
        scaled_vectors = _refine_wall_vectors_from_raster(gray, scaled_vectors)

    output = []
    for index, (vector, metadata) in enumerate(
        zip(scaled_vectors, retained, strict=True)
    ):
        _, evidence, support, line_uncertainty = metadata
        confidence = min(1.0, evidence * 0.65 + support * 0.35)
        output.append(
            PixelLineProposal(
                id=f"global:wall:{index}",
                start_px=vector.start_px,
                end_px=vector.end_px,
                thickness_px=vector.thickness_px,
                confidence=confidence,
                uncertainty=line_uncertainty,
                source_ref_ids=source_ref_ids,
                model_version=model_version,
                review_required=support < 0.55 or line_uncertainty > 0.35,
            )
        )
    return output, rejected, refinement_applied


def _decode_elements(
    element_probability: np.ndarray,
    geometry: np.ndarray,
    uncertainty: np.ndarray,
    *,
    source_size: tuple[int, int],
    source_ref_ids: list[str],
    model_version: str,
    element_threshold: float,
    source_gray: np.ndarray | None,
) -> tuple[list[PixelSymbolProposal], int]:
    height, width = element_probability.shape[1:]
    scale_x = source_size[0] / width
    scale_y = source_size[1] / height
    class_map = element_probability.argmax(axis=0)
    minimum_area = max(2, round(width * height * 0.000025))
    output: list[PixelSymbolProposal] = []
    refinement_count = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_index, class_name in enumerate(ELEMENT_PROGRAM_CLASSES[1:], start=1):
        mask = (class_map == class_index) & (
            element_probability[class_index] >= element_threshold
        )
        labels, count = ndimage.label(mask, structure=structure)
        for component in range(1, count + 1):
            local = labels == component
            if int(local.sum()) < minimum_area:
                continue
            ys, xs = np.nonzero(local)
            weights = element_probability[class_index, ys, xs]
            weight_total = max(float(weights.sum()), 1e-9)
            predicted_x = xs + geometry[0, ys, xs] * width
            predicted_y = ys + geometry[1, ys, xs] * height
            center_x = float(np.dot(predicted_x, weights) / weight_total)
            center_y = float(np.dot(predicted_y, weights) / weight_total)
            box_width = float(np.exp(np.median(geometry[2, ys, xs])) * width)
            box_height = float(np.exp(np.median(geometry[3, ys, xs])) * height)
            component_width = float(xs.max() - xs.min() + 1)
            component_height = float(ys.max() - ys.min() + 1)
            if not 1.0 <= box_width <= width:
                box_width = component_width
            if not 1.0 <= box_height <= height:
                box_height = component_height
            left = max(0.0, center_x - box_width / 2)
            top = max(0.0, center_y - box_height / 2)
            right = min(float(width), center_x + box_width / 2)
            bottom = min(float(height), center_y + box_height / 2)
            source_bbox = (
                left * scale_x,
                top * scale_y,
                right * scale_x,
                bottom * scale_y,
            )
            if source_gray is not None:
                refined_bbox = _refine_element_bbox_from_raster(
                    source_gray,
                    source_bbox,
                )
                if any(
                    abs(refined - original) > 0.5
                    for refined, original in zip(
                        refined_bbox,
                        source_bbox,
                        strict=True,
                    )
                ):
                    refinement_count += 1
                source_bbox = refined_bbox
            confidence = float(weights.mean())
            item_uncertainty = float(uncertainty[2, ys, xs].mean())
            source_center_x = (source_bbox[0] + source_bbox[2]) / 2
            source_center_y = (source_bbox[1] + source_bbox[3]) / 2
            output.append(
                PixelSymbolProposal(
                    id=f"global:element:{len(output)}",
                    symbol_class=class_name,
                    center_px=(source_center_x, source_center_y),
                    bbox_px=source_bbox,
                    confidence=confidence,
                    uncertainty=item_uncertainty,
                    source_ref_ids=source_ref_ids,
                    model_version=model_version,
                    review_required=(
                        confidence < 0.78
                        or item_uncertainty > 0.35
                        or class_name == "unknown"
                    ),
                )
            )
    return output, refinement_count


def _refine_element_bbox_from_raster(
    source_gray: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Snap a coarse global element box to native-resolution symbol ink.

    Whole-sheet semantics decide *what* and approximately *where*.  This local,
    deterministic stage only measures the original raster and refuses large
    moves, preventing 256-pixel inference from becoming final CAD geometry.
    """

    gray = np.asarray(source_gray, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError("source_gray must be a two-dimensional raster")
    left, top, right, bottom = bbox
    box_width = right - left
    box_height = bottom - top
    if box_width < 2 or box_height < 2:
        return bbox
    margin = max(3.0, max(box_width, box_height) * 0.35)
    crop_left = max(0, int(math.floor(left - margin)))
    crop_top = max(0, int(math.floor(top - margin)))
    crop_right = min(gray.shape[1], int(math.ceil(right + margin)))
    crop_bottom = min(gray.shape[0], int(math.ceil(bottom + margin)))
    if crop_right - crop_left < 3 or crop_bottom - crop_top < 3:
        return bbox
    crop = gray[crop_top:crop_bottom, crop_left:crop_right]
    ink = crop < 232
    labels, component_count = ndimage.label(ink, structure=np.ones((3, 3), np.uint8))
    selected: list[tuple[int, int, int, int]] = []
    local_left = left - crop_left
    local_top = top - crop_top
    local_right = right - crop_left
    local_bottom = bottom - crop_top
    minimum_area = max(2, round(box_width * box_height * 0.002))
    for component in range(1, component_count + 1):
        ys, xs = np.nonzero(labels == component)
        if len(xs) < minimum_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        span_x = x1 - x0
        span_y = y1 - y0
        # Long drafting/wall runs are context, not the local object extent.
        if span_x >= crop.shape[1] * 0.88 and span_y <= max(3, crop.shape[0] * 0.12):
            continue
        if span_y >= crop.shape[0] * 0.88 and span_x <= max(3, crop.shape[1] * 0.12):
            continue
        overlaps = not (
            x1 < local_left
            or x0 > local_right
            or y1 < local_top
            or y0 > local_bottom
        )
        if overlaps:
            selected.append((x0, y0, x1, y1))
    if not selected:
        return bbox
    refined = (
        float(crop_left + min(item[0] for item in selected)),
        float(crop_top + min(item[1] for item in selected)),
        float(crop_left + max(item[2] for item in selected)),
        float(crop_top + max(item[3] for item in selected)),
    )
    refined_width = refined[2] - refined[0]
    refined_height = refined[3] - refined[1]
    if not (
        box_width * 0.35 <= refined_width <= box_width * 1.8
        and box_height * 0.35 <= refined_height <= box_height * 1.8
    ):
        return bbox
    original_center = ((left + right) / 2, (top + bottom) / 2)
    refined_center = ((refined[0] + refined[2]) / 2, (refined[1] + refined[3]) / 2)
    if (
        abs(refined_center[0] - original_center[0]) > box_width * 0.35
        or abs(refined_center[1] - original_center[1]) > box_height * 0.35
    ):
        return bbox
    return refined


def decode_global_program(
    *,
    tile_id: str,
    source_ref_ids: list[str],
    model_version: str,
    source_size: tuple[int, int],
    topology_logits: np.ndarray,
    room_semantic_logits: np.ndarray,
    element_semantic_logits: np.ndarray,
    element_geometry: np.ndarray,
    uncertainty: np.ndarray,
    source_gray: np.ndarray | None = None,
    wall_threshold: float = 0.48,
    room_threshold: float = 0.45,
    element_threshold: float = 0.62,
    structural_support_threshold: float = 0.12,
) -> GlobalProgramDecodeResult:
    """Decode a single coherent building program from full-sheet model outputs."""

    if not tile_id or not source_ref_ids or not model_version:
        raise ValueError("tile_id, source_ref_ids, and model_version are required")
    if source_size[0] < 1 or source_size[1] < 1:
        raise ValueError("source_size must be positive")
    model_size = _validate_outputs(
        topology_logits,
        room_semantic_logits,
        element_semantic_logits,
        element_geometry,
        uncertainty,
    )
    topology_probability = _sigmoid(topology_logits)
    room_probability = _softmax(room_semantic_logits)
    element_probability = _softmax(element_semantic_logits)
    uncertainty_map = np.asarray(uncertainty, dtype=np.float32)
    room_instances, rooms = _room_instances(
        topology_probability,
        room_probability,
        source_size=source_size,
        room_threshold=room_threshold,
    )
    room_semantic_seeds = _localized_room_semantic_seeds(
        topology_probability,
        room_probability,
        source_size=source_size,
    )
    rooms = [
        room.model_copy(
            update={
                "source_ref_ids": source_ref_ids,
                "model_version": model_version,
            }
        )
        for room in rooms
    ]
    walls, rejected_walls, refined = _decode_walls(
        topology_probability,
        room_instances,
        uncertainty_map,
        source_size=source_size,
        source_gray=source_gray,
        source_ref_ids=source_ref_ids,
        model_version=model_version,
        wall_threshold=wall_threshold,
        structural_support_threshold=structural_support_threshold,
    )
    elements, refined_element_count = _decode_elements(
        element_probability,
        np.asarray(element_geometry, dtype=np.float32),
        uncertainty_map,
        source_size=source_size,
        source_ref_ids=source_ref_ids,
        model_version=model_version,
        element_threshold=element_threshold,
        source_gray=source_gray,
    )
    proposal = AecTileProposal(
        tile_id=tile_id,
        source_ref_ids=source_ref_ids,
        model_version=model_version,
        wall_segments=walls,
        symbols=elements,
        room_regions=rooms,
        rejected_candidates=rejected_walls,
    ).finalize()
    blockers = ["model_not_commercially_calibrated", "direct_gt_qualification_missing"]
    return GlobalProgramDecodeResult(
        proposal=proposal,
        diagnostics=GlobalProgramDecodeDiagnostics(
            source_size=source_size,
            model_size=model_size,
            room_instance_count=len(rooms),
            structural_wall_count=len(walls),
            rejected_wall_count=rejected_walls,
            element_count=len(elements),
            native_wall_refinement_applied=refined,
            native_element_refinement_count=refined_element_count,
            release_eligible=False,
            release_blockers=blockers,
        ),
        room_semantic_seeds=[
            seed.model_copy(
                update={
                    "source_ref_ids": source_ref_ids,
                    "model_version": (
                        f"{model_version}+localized-room-semantic-seed-v2"
                    ),
                }
            )
            for seed in room_semantic_seeds
        ],
    )
