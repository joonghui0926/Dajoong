"""Recover room instances from the final source-native wall graph.

The learned room output is semantic evidence, not the authority for room
boundaries.  Its training contract is a localized room-label seed while its
decoded polygon comes from a provisional learned wall graph.  Final rooms are
the connected free-space faces of the accepted source-native wall graph.  A
semantic region may name a face only when it belongs unambiguously to that
face; otherwise the face remains reviewable instead of receiving a confident
but false occupancy label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .core.model.aec_decode import PixelRoomProposal
from .semantic_recognition import _mask_outer_loop, _polygon_area, _simplify_loop


def _wall_geometry(wall: Any) -> tuple[np.ndarray, np.ndarray, float, str]:
    start = np.asarray(wall.start_px, dtype=np.float64)
    end = np.asarray(wall.end_px, dtype=np.float64)
    thickness = float(wall.thickness_px or 4.0)
    orientation = "horizontal" if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else "vertical"
    return start, end, thickness, orientation


def _draw_room_barrier(
    walls: list[Any],
    source_size: tuple[int, int],
) -> np.ndarray:
    image = Image.new("1", source_size, 0)
    draw = ImageDraw.Draw(image)
    geometry = [_wall_geometry(wall) for wall in walls]
    for start, end, thickness, _ in geometry:
        draw.line(
            (*start.tolist(), *end.tolist()),
            fill=1,
            width=max(3, round(thickness)),
        )

    # Wall vectors are centerlines and may be split at doors/windows. Close
    # only collinear gaps bounded by two substantial accepted wall segments;
    # arbitrary raster ink never participates here.
    minimum_side = min(source_size)
    for index, left in enumerate(geometry):
        left_start, left_end, left_thickness, left_orientation = left
        left_length = float(np.linalg.norm(left_end - left_start))
        if left_length < minimum_side * 0.08:
            continue
        for right in geometry[index + 1 :]:
            right_start, right_end, right_thickness, right_orientation = right
            if right_orientation != left_orientation:
                continue
            right_length = float(np.linalg.norm(right_end - right_start))
            if right_length < minimum_side * 0.08:
                continue
            if left_orientation == "horizontal":
                coordinate_delta = abs(
                    float((left_start[1] + left_end[1] - right_start[1] - right_end[1]) / 2)
                )
                left_interval = sorted((float(left_start[0]), float(left_end[0])))
                right_interval = sorted((float(right_start[0]), float(right_end[0])))
                coordinate = float(
                    (left_start[1] + left_end[1] + right_start[1] + right_end[1]) / 4
                )
            else:
                coordinate_delta = abs(
                    float((left_start[0] + left_end[0] - right_start[0] - right_end[0]) / 2)
                )
                left_interval = sorted((float(left_start[1]), float(left_end[1])))
                right_interval = sorted((float(right_start[1]), float(right_end[1])))
                coordinate = float(
                    (left_start[0] + left_end[0] + right_start[0] + right_end[0]) / 4
                )
            if coordinate_delta > max(4.0, left_thickness, right_thickness):
                continue
            if left_interval[0] > right_interval[0]:
                left_interval, right_interval = right_interval, left_interval
            gap_start, gap_end = left_interval[1], right_interval[0]
            gap = gap_end - gap_start
            if not 1.0 <= gap <= minimum_side * 0.28:
                continue
            width = max(3, round((left_thickness + right_thickness) / 2))
            if left_orientation == "horizontal":
                draw.line((gap_start, coordinate, gap_end, coordinate), fill=1, width=width)
            else:
                draw.line((coordinate, gap_start, coordinate, gap_end), fill=1, width=width)
    return np.asarray(image, dtype=np.bool_)


def _polygon_mask(
    polygon: list[tuple[float, float]],
    source_size: tuple[int, int],
) -> np.ndarray:
    image = Image.new("1", source_size, 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(image).polygon(polygon, fill=1)
    return np.asarray(image, dtype=np.bool_)


def _room_overlap(
    left: Any,
    right: Any,
    *,
    source_size: tuple[int, int],
) -> tuple[float, float, float]:
    """Return IoU and directional coverages for two room proposals."""

    left_mask = _polygon_mask(left.polygon_px, source_size)
    right_mask = _polygon_mask(right.polygon_px, source_size)
    intersection = int((left_mask & right_mask).sum())
    left_area = int(left_mask.sum())
    right_area = int(right_mask.sum())
    union = left_area + right_area - intersection
    return (
        intersection / max(1, union),
        intersection / max(1, left_area),
        intersection / max(1, right_area),
    )


def merge_topology_and_provisional_rooms(
    topology_rooms: list[PixelRoomProposal],
    provisional_rooms: list[Any],
    *,
    source_size: tuple[int, int],
    source_ref_ids: list[str],
    model_version: str,
) -> list[PixelRoomProposal]:
    """Keep unresolved full-sheet rooms instead of erasing them.

    Wall faces remain the geometry authority whenever they exist.  A partial
    sheet, however, can legitimately contain rooms whose walls continue beyond
    the captured image or selected plan frame.  Such a room cannot become a
    closed graph face.  The previous all-or-nothing replacement silently
    removed those rooms and also stripped room context from their fixtures.

    Unmatched full-sheet room proposals are therefore retained as explicitly
    review-required partial geometry.  They are never promoted to accepted BIM
    spaces and cannot replace an overlapping wall-graph room.
    """

    merged = list(topology_rooms)
    for provisional in provisional_rooms:
        polygon = [
            (float(point[0]), float(point[1])) for point in getattr(provisional, "polygon_px", [])
        ]
        if len(polygon) < 3 or _polygon_area(polygon) <= 0:
            continue
        overlaps_existing = False
        for topology in topology_rooms:
            iou, provisional_coverage, topology_coverage = _room_overlap(
                provisional,
                topology,
                source_size=source_size,
            )
            if iou >= 0.28 or provisional_coverage >= 0.62 or topology_coverage >= 0.62:
                overlaps_existing = True
                break
        if overlaps_existing:
            continue
        confidence = min(float(getattr(provisional, "confidence", 0.0)), 0.71)
        merged.append(
            PixelRoomProposal(
                id=f"partial:room:{len(merged)}",
                name=str(getattr(provisional, "name", "Unresolved partial room")),
                room_class=str(getattr(provisional, "room_class", "unknown")),
                polygon_px=polygon,
                confidence=confidence,
                uncertainty=max(
                    1.0 - confidence,
                    float(getattr(provisional, "uncertainty", 1.0)),
                ),
                source_ref_ids=source_ref_ids,
                model_version=(f"{model_version}+unresolved-partial-room-preservation-v1"),
                review_required=True,
            )
        )
    return sorted(
        merged,
        key=lambda item: (
            min(point[1] for point in item.polygon_px),
            min(point[0] for point in item.polygon_px),
        ),
    )


def _semantic_room_assignments(
    face_masks: list[np.ndarray],
    semantic_rooms: list[Any],
    *,
    source_size: tuple[int, int],
) -> dict[int, tuple[Any, float]]:
    """Assign localized semantic evidence to final wall-graph faces.

    The room student is trained with small class seeds.  Its decoded polygon,
    however, follows a provisional learned wall graph.  Requiring that polygon
    to have near-perfect IoU with a later source-native face discards otherwise
    valid class evidence.  We instead require the semantic region to lie
    predominantly inside exactly one final face.  A large full-region match is
    still accepted through the original IoU path.
    """

    proposals_by_face: dict[int, list[tuple[Any, float]]] = {
        index: [] for index in range(len(face_masks))
    }
    for semantic_room in semantic_rooms:
        confidence = float(semantic_room.confidence)
        if confidence < 0.72:
            continue
        semantic_mask = _polygon_mask(semantic_room.polygon_px, source_size)
        semantic_area = int(semantic_mask.sum())
        if semantic_area == 0:
            continue
        rows, columns = np.nonzero(semantic_mask)
        center_x = int(np.clip(round(float(columns.mean())), 0, source_size[0] - 1))
        center_y = int(np.clip(round(float(rows.mean())), 0, source_size[1] - 1))
        candidates: list[tuple[int, float, float, float, bool]] = []
        for face_index, face_mask in enumerate(face_masks):
            intersection = int((face_mask & semantic_mask).sum())
            if intersection == 0:
                continue
            face_area = int(face_mask.sum())
            semantic_coverage = intersection / semantic_area
            union = face_area + semantic_area - intersection
            iou = intersection / max(1, union)
            center_inside = bool(face_mask[center_y, center_x])
            full_region_match = iou >= 0.78
            localized_seed_match = center_inside and semantic_coverage >= 0.80
            if not (full_region_match or localized_seed_match):
                continue
            score = (
                0.60 * semantic_coverage + 0.25 * float(center_inside) + 0.15 * min(1.0, iou / 0.78)
            )
            candidates.append((face_index, score, semantic_coverage, iou, center_inside))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[1], reverse=True)
        best = candidates[0]
        if len(candidates) > 1 and best[1] - candidates[1][1] < 0.12:
            continue
        match_quality = max(best[2], best[3])
        proposals_by_face[best[0]].append((semantic_room, min(confidence, match_quality)))

    assignments: dict[int, tuple[Any, float]] = {}
    for face_index, proposals in proposals_by_face.items():
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[1], reverse=True)
        best_room, best_score = proposals[0]
        if len(proposals) > 1:
            next_room, next_score = proposals[1]
            different_classes = str(next_room.room_class) != str(best_room.room_class)
            if different_classes and best_score - next_score < 0.12:
                continue
        assignments[face_index] = best_room, best_score
    return assignments


def reconstruct_rooms_from_wall_graph(
    walls: list[Any],
    semantic_rooms: list[Any],
    *,
    source_size: tuple[int, int],
    source_ref_ids: list[str],
    model_version: str,
) -> list[PixelRoomProposal]:
    """Return the bounded faces of the final wall graph in source pixels."""

    if not walls or not source_ref_ids:
        return []
    barrier = _draw_room_barrier(walls, source_size)
    barrier = ndimage.binary_closing(barrier, structure=np.ones((3, 3), dtype=np.bool_))
    enclosed = ndimage.binary_fill_holes(barrier) & ~barrier
    labels, count = ndimage.label(
        enclosed,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8),
    )
    minimum_area = max(256, round(source_size[0] * source_size[1] * 0.004))
    face_masks: list[np.ndarray] = []
    for component in range(1, count + 1):
        mask = labels == component
        area = int(mask.sum())
        if area < minimum_area:
            continue
        loop = _mask_outer_loop(mask)
        polygon = _simplify_loop(loop, tolerance=1.5)
        if len(polygon) < 3 or _polygon_area(polygon) < minimum_area:
            continue
        face_masks.append(mask)

    assignments = _semantic_room_assignments(
        face_masks,
        semantic_rooms,
        source_size=source_size,
    )
    rooms: list[PixelRoomProposal] = []
    for face_index, mask in enumerate(face_masks):
        loop = _mask_outer_loop(mask)
        polygon = _simplify_loop(loop, tolerance=1.5)
        room_class = "unknown"
        confidence = 0.5
        assignment = assignments.get(face_index)
        if assignment is not None:
            matched_room, confidence = assignment
            room_class = str(matched_room.room_class)
        rooms.append(
            PixelRoomProposal(
                id=f"topology:room:{len(rooms)}",
                name=f"{room_class} {len(rooms) + 1}",
                room_class=room_class,
                polygon_px=[(float(x), float(y)) for x, y in polygon],
                confidence=confidence,
                uncertainty=1.0 - confidence,
                source_ref_ids=source_ref_ids,
                model_version=f"{model_version}+wall-face-room-topology-v2",
                review_required=room_class == "unknown" or confidence < 0.88,
            )
        )
    return sorted(
        rooms,
        key=lambda item: (
            min(point[1] for point in item.polygon_px),
            min(point[0] for point in item.polygon_px),
        ),
    )
