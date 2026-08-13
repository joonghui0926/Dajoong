"""Fuse whole-sheet context with native-detail plan-instance recognition.

The complete sheet is always recognized first and remains the coordinate and
provenance authority.  Region passes are evidence refiners only: they may
replace geometry inside a reviewed plan proposal when they materially improve
sampling density, but they can never erase unmatched whole-sheet evidence.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import sha256_json
from .semantic_recognition import (
    SemanticDetection,
    SemanticRecognitionResult,
    SemanticRoom,
    SemanticWallVector,
    _merge_collinear_wall_vectors,
)
from .sheet_layout import SheetLayoutAnalysis, SheetPlanRegion, discover_plan_regions


class SemanticRecognizer(Protocol):
    def recognize(
        self,
        image_path: str | Path,
        *,
        max_side: int = 1024,
    ) -> tuple[SemanticRecognitionResult, np.ndarray]: ...


class SemanticRegionPass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str
    bbox_px: tuple[int, int, int, int]
    crop_path: str
    global_linear_scale: float = Field(gt=0, le=1)
    detail_linear_scale: float = Field(gt=0, le=1)
    scale_gain: float = Field(gt=0)
    trusted_for_geometry: bool
    wall_vector_count: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    room_count: int = Field(ge=0)
    inference_ms: float = Field(ge=0)


class SemanticMultiviewDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.semantic-multiview.v1"
    sheet_id: str
    fusion_version: str = "whole-sheet-native-detail-fusion-v1"
    whole_sheet_preserved: bool = True
    layout: SheetLayoutAnalysis
    region_passes: list[SemanticRegionPass]
    fused_wall_vector_count: int = Field(ge=0)
    fused_detection_count: int = Field(ge=0)
    fused_room_count: int = Field(ge=0)
    review_required: bool = True
    total_ms: float = Field(ge=0)
    content_sha256: str = ""

    def finalize(self) -> SemanticMultiviewDiagnostics:
        payload = self.model_dump(mode="json", exclude={"content_sha256", "total_ms"})
        self.content_sha256 = sha256_json(payload)
        return self


def _inside(point: tuple[float, float], bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def _room_bbox(room: SemanticRoom) -> tuple[int, int, int, int]:
    x = [point[0] for point in room.polygon_px]
    y = [point[1] for point in room.polygon_px]
    return (int(min(x)), int(min(y)), int(max(x)) + 1, int(max(y)) + 1)


def _translate_detection(
    detection: SemanticDetection,
    region: SheetPlanRegion,
) -> SemanticDetection:
    left, top, _, _ = region.bbox_px
    x0, y0, x1, y1 = detection.bbox_px
    return detection.model_copy(
        update={
            "id": f"{region.id}:{detection.id}",
            "bbox_px": (x0 + left, y0 + top, x1 + left, y1 + top),
            "evidence_mode": f"{detection.evidence_mode}+native_plan_detail",
        }
    )


def _translate_room(room: SemanticRoom, region: SheetPlanRegion) -> SemanticRoom:
    left, top, _, _ = region.bbox_px
    return room.model_copy(
        update={
            "id": f"{region.id}:{room.id}",
            "polygon_px": [(x + left, y + top) for x, y in room.polygon_px],
        }
    )


def _translate_wall(
    vector: SemanticWallVector,
    region: SheetPlanRegion,
) -> SemanticWallVector:
    left, top, _, _ = region.bbox_px
    return vector.model_copy(
        update={
            "start_px": (vector.start_px[0] + left, vector.start_px[1] + top),
            "end_px": (vector.end_px[0] + left, vector.end_px[1] + top),
        }
    )


def _fuse_detections(
    global_items: list[SemanticDetection],
    detail_items: list[SemanticDetection],
) -> list[SemanticDetection]:
    """Prefer native geometry for duplicates and preserve every unmatched item."""

    output = list(global_items)
    for detail in sorted(detail_items, key=lambda item: item.confidence, reverse=True):
        best_index: int | None = None
        best_score = 0.0
        detail_center = (
            (detail.bbox_px[0] + detail.bbox_px[2]) / 2,
            (detail.bbox_px[1] + detail.bbox_px[3]) / 2,
        )
        detail_diagonal = max(
            8.0,
            float(np.hypot(
                detail.bbox_px[2] - detail.bbox_px[0],
                detail.bbox_px[3] - detail.bbox_px[1],
            )),
        )
        for index, existing in enumerate(output):
            if existing.symbol_class != detail.symbol_class:
                continue
            existing_center = (
                (existing.bbox_px[0] + existing.bbox_px[2]) / 2,
                (existing.bbox_px[1] + existing.bbox_px[3]) / 2,
            )
            overlap = _bbox_iou(existing.bbox_px, detail.bbox_px)
            distance_score = max(
                0.0,
                1.0 - float(np.hypot(
                    existing_center[0] - detail_center[0],
                    existing_center[1] - detail_center[1],
                )) / (detail_diagonal * 0.65),
            )
            score = max(overlap, distance_score)
            if score > best_score and (overlap >= 0.12 or distance_score >= 0.55):
                best_index = index
                best_score = score
        if best_index is None:
            output.append(detail)
            continue
        existing = output[best_index]
        output[best_index] = detail.model_copy(
            update={
                "confidence": max(existing.confidence, detail.confidence),
                "pixel_area": max(existing.pixel_area, detail.pixel_area),
                "review_required": existing.review_required and detail.review_required,
                "promote_to_bim": existing.promote_to_bim or detail.promote_to_bim,
                "evidence_mode": (
                    f"{existing.evidence_mode}+{detail.evidence_mode}"
                ),
            }
        )
    return output


def _fuse_rooms(
    global_items: list[SemanticRoom],
    detail_items: list[SemanticRoom],
) -> list[SemanticRoom]:
    """Merge geometric duplicates without dropping an unmatched whole-sheet room."""

    output = list(global_items)
    for detail in sorted(detail_items, key=lambda item: item.confidence, reverse=True):
        detail_bbox = _room_bbox(detail)
        best_index: int | None = None
        best_overlap = 0.0
        for index, existing in enumerate(output):
            overlap = _bbox_iou(_room_bbox(existing), detail_bbox)
            if overlap > best_overlap and overlap >= 0.35:
                best_index = index
                best_overlap = overlap
        if best_index is None:
            output.append(detail)
            continue
        existing = output[best_index]
        # Native geometry wins, while a stronger whole-sheet semantic name is
        # retained when the detail pass is explicitly unclassified.
        class_name = detail.class_name
        if class_name == "Unclassified interior" and existing.class_name != class_name:
            class_name = existing.class_name
        output[best_index] = detail.model_copy(
            update={
                "class_name": class_name,
                "confidence": max(existing.confidence, detail.confidence),
                "review_required": existing.review_required and detail.review_required,
            }
        )
    return output


def _wall_midpoint(vector: SemanticWallVector) -> tuple[float, float]:
    return (
        (vector.start_px[0] + vector.end_px[0]) / 2,
        (vector.start_px[1] + vector.end_px[1]) / 2,
    )


def _select_regions(
    layout: SheetLayoutAnalysis,
    *,
    global_scale: float,
    maximum_region_passes: int,
) -> list[SheetPlanRegion]:
    # Multiple plans must each be read. A single plan receives a native-detail
    # pass only when the full sheet was materially reduced.
    if layout.multi_plan_candidate:
        candidates = layout.regions
    elif global_scale < 0.86:
        candidates = layout.regions
    else:
        candidates = []
    # This argument remains for API compatibility and as an operational alert
    # threshold. It must never truncate visible plan instances: the former
    # confidence/area filter plus an eight-pass slice made later or smaller
    # drawings irrecoverable. Ambiguous regions stay review-required instead.
    _ = maximum_region_passes
    return list(candidates)


def recognize_semantic_multiview(
    recognizer: SemanticRecognizer,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    sheet_id: str,
    max_side: int,
    maximum_region_passes: int = 8,
) -> tuple[
    SemanticRecognitionResult,
    np.ndarray,
    SheetLayoutAnalysis,
    SemanticMultiviewDiagnostics,
]:
    """Recognize the whole sheet, then fuse only useful native-detail passes."""

    started = time.perf_counter()
    source_path = Path(image_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    global_result, global_mask = recognizer.recognize(source_path, max_side=max_side)
    source = Image.open(source_path).convert("RGB")
    layout = discover_plan_regions(source, sheet_id=sheet_id)
    global_scale = min(
        global_result.model_input_size[0] / max(1, global_result.source_size[0]),
        global_result.model_input_size[1] / max(1, global_result.source_size[1]),
    )
    selected_regions = _select_regions(
        layout,
        global_scale=global_scale,
        maximum_region_passes=maximum_region_passes,
    )
    region_dir = destination / "00-semantic-regions"
    translated_walls: list[SemanticWallVector] = []
    translated_detections: list[SemanticDetection] = []
    translated_rooms: list[SemanticRoom] = []
    trusted_regions: list[SheetPlanRegion] = []
    region_passes: list[SemanticRegionPass] = []
    fused_mask = np.asarray(global_mask, dtype=np.bool_).copy()
    for index, region in enumerate(selected_regions, start=1):
        region_dir.mkdir(parents=True, exist_ok=True)
        crop = source.crop(region.bbox_px)
        crop_path = region_dir / f"plan-{index:02d}.png"
        crop.save(crop_path, format="PNG", optimize=True)
        detail_result, detail_mask = recognizer.recognize(crop_path, max_side=max_side)
        detail_scale = min(
            detail_result.model_input_size[0] / max(1, detail_result.source_size[0]),
            detail_result.model_input_size[1] / max(1, detail_result.source_size[1]),
        )
        scale_gain = detail_scale / max(global_scale, 1e-9)
        trusted = scale_gain >= 1.12 or layout.multi_plan_candidate
        if trusted:
            trusted_regions.append(region)
            left, top, right, bottom = region.bbox_px
            fused_mask[top:bottom, left:right] = np.asarray(detail_mask, dtype=np.bool_)
            translated_walls.extend(
                _translate_wall(item, region) for item in detail_result.wall_vectors_px
            )
            translated_detections.extend(
                _translate_detection(item, region) for item in detail_result.detections
            )
            translated_rooms.extend(
                _translate_room(item, region) for item in detail_result.rooms
            )
        region_passes.append(
            SemanticRegionPass(
                region_id=region.id,
                bbox_px=region.bbox_px,
                crop_path=str(crop_path),
                global_linear_scale=global_scale,
                detail_linear_scale=detail_scale,
                scale_gain=scale_gain,
                trusted_for_geometry=trusted,
                wall_vector_count=len(detail_result.wall_vectors_px),
                detection_count=len(detail_result.detections),
                room_count=len(detail_result.rooms),
                inference_ms=detail_result.inference_ms,
            )
        )

    # Replace global wall vectors only inside a trusted refined region. Other
    # whole-sheet walls remain untouched, including context between plan views.
    global_walls = [
        vector
        for vector in global_result.wall_vectors_px
        if not any(_inside(_wall_midpoint(vector), region.bbox_px) for region in trusted_regions)
    ]
    fused_walls = _merge_collinear_wall_vectors([*global_walls, *translated_walls])
    fused_detections = _fuse_detections(global_result.detections, translated_detections)
    fused_rooms = _fuse_rooms(global_result.rooms, translated_rooms)
    counts = {
        name: sum(item.class_name == name for item in fused_detections)
        for name in recognizer.icon_classes[1:]  # type: ignore[attr-defined]
    }
    evidence_counts = {
        mode: sum(item.evidence_mode == mode for item in fused_detections)
        for mode in sorted({item.evidence_mode for item in fused_detections})
    }
    room_names = {
        *getattr(recognizer, "room_classes", ()),
        "Unclassified interior",
        *(item.class_name for item in fused_rooms),
    }
    room_counts = {
        name: sum(item.class_name == name for item in fused_rooms)
        for name in sorted(room_names)
        if name not in {"Background", "Outdoor", "Wall", "Railing"}
    }
    fused_result = global_result.model_copy(
        update={
            "decoder_version": f"{global_result.decoder_version}+native-detail-fusion-v1",
            "wall_pixels": int(fused_mask.sum()),
            "wall_centerlines_px": [
                (*vector.start_px, *vector.end_px) for vector in fused_walls
            ],
            "wall_vectors_px": fused_walls,
            "detections": fused_detections,
            "rooms": fused_rooms,
            "counts": counts,
            "evidence_counts": evidence_counts,
            "room_counts": room_counts,
            "inference_ms": round(
                global_result.inference_ms
                + sum(item.inference_ms for item in region_passes),
                3,
            ),
            "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "content_sha256": "",
        }
    ).finalize()
    diagnostics = SemanticMultiviewDiagnostics(
        sheet_id=sheet_id,
        layout=layout,
        region_passes=region_passes,
        fused_wall_vector_count=len(fused_walls),
        fused_detection_count=len(fused_detections),
        fused_room_count=len(fused_rooms),
        total_ms=round((time.perf_counter() - started) * 1000, 3),
    ).finalize()
    return fused_result, fused_mask, layout, diagnostics
