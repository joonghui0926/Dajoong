"""Verified CPU inference for the whole-sheet Dajoong building program."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import sha256_file
from .core.model.cad_evidence import (
    GLOBAL_PROGRAM_INPUT_CONTRACT,
    ORIENTED_EVIDENCE_ROTATION_CONTRACT,
    build_cad_evidence,
    letterbox_cad_evidence,
)
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
)
from .global_program_decode import GlobalProgramDecodeResult, decode_global_program
from .local_element_inference import (
    LocalElementOnnxRecognizer,
    LocalElementRefinementDiagnostics,
)
from .native_opening_candidates import (
    NativeOpeningDiagnostics,
    consolidate_walls_across_openings,
    infer_openings_from_wall_graph,
)
from .native_wall_candidates import (
    NativeWallCandidateDiagnostics,
    UnresolvedNativeWallCandidate,
    mine_native_wall_candidates,
    promote_supported_native_wall_candidates,
    refine_context_walls_with_native_bands,
    translate_native_wall_candidate,
    unresolved_native_wall_candidates,
)
from .perception_forest import (
    SpatialEvidenceGraph,
    build_spatial_evidence_graph_from_proposal,
)
from .room_topology import (
    merge_topology_and_provisional_rooms,
    reconstruct_rooms_from_wall_graph,
)
from .sheet_layout import SheetLayoutAnalysis, SheetPlanRegion, discover_plan_regions


class GlobalProgramRegionPass(BaseModel):
    """One native-detail pass mapped back into whole-sheet coordinates."""

    model_config = ConfigDict(extra="forbid")

    region_id: str
    bbox_px: tuple[int, int, int, int]
    scale_gain: float = Field(gt=0)
    wall_count: int = Field(ge=0)
    room_count: int = Field(ge=0)
    element_count: int = Field(ge=0)
    trusted_for_refinement: bool
    pass_kind: Literal["plan_instance", "native_detail_window"] = "plan_instance"
    context_x: float = Field(default=0.0, ge=0, le=1)
    context_y: float = Field(default=0.0, ge=0, le=1)
    context_width: float = Field(default=1.0, gt=0, le=1)
    context_height: float = Field(default=1.0, gt=0, le=1)
    touches_left: bool = False
    touches_top: bool = False
    touches_right: bool = False
    touches_bottom: bool = False


class GlobalProgramMultiviewDiagnostics(BaseModel):
    """Proof that the sheet and every useful plan instance were both read."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.global-program-multiview.v2"
    layout: SheetLayoutAnalysis
    whole_sheet_preserved: bool = True
    region_passes: list[GlobalProgramRegionPass] = Field(default_factory=list)
    fused_wall_count: int = Field(ge=0)
    fused_room_count: int = Field(ge=0)
    fused_element_count: int = Field(ge=0)
    unresolved_native_candidate_count: int = Field(default=0, ge=0)
    context_rejected_detail_wall_count: int = Field(default=0, ge=0)
    context_rejected_detail_wall_ids: list[str] = Field(default_factory=list)
    native_wall_geometry_refinement_count: int = Field(default=0, ge=0)
    selected_plan_instance_id: str = ""
    native_wall_candidate_diagnostics: NativeWallCandidateDiagnostics | None = None
    native_opening_diagnostics: NativeOpeningDiagnostics | None = None
    unresolved_native_walls: list[UnresolvedNativeWallCandidate] = Field(default_factory=list)


class GlobalProgramInferenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.global-program-inference.v1"
    input_path: str
    input_sha256: str
    model_version: str
    model_sha256: str
    production_authorized: bool
    decode: GlobalProgramDecodeResult
    multiview: GlobalProgramMultiviewDiagnostics | None = None
    local_element_refinement: LocalElementRefinementDiagnostics | None = None
    evidence_graph: SpatialEvidenceGraph
    timings_ms: dict[str, float] = Field(default_factory=dict)


def validate_global_program_manifest(
    payload: dict[str, Any],
    artifact_path: str | Path,
    *,
    require_production: bool,
) -> None:
    artifact = Path(artifact_path).expanduser().resolve()
    if payload.get("schema_version") != "dajoong.global-program-onnx.v1":
        raise ValueError("unsupported global-program ONNX manifest")
    if payload.get("artifact_sha256") != sha256_file(artifact):
        raise ValueError("global-program ONNX artifact hash mismatch")
    if payload.get("input_contract") != GLOBAL_PROGRAM_INPUT_CONTRACT:
        raise ValueError("global-program evidence contract mismatch")
    model_version = str(payload.get("model_version") or "")
    if model_version.startswith("dajoong-global-program-student-v8-") and (
        payload.get("oriented_evidence_rotation_contract") != ORIENTED_EVIDENCE_ROTATION_CONTRACT
    ):
        raise ValueError("global-program oriented-evidence rotation contract mismatch")
    contracts = (
        ("topology_channels", TOPOLOGY_TARGET_CHANNELS),
        ("room_classes", ROOM_PROGRAM_CLASSES),
        ("element_classes", ELEMENT_PROGRAM_CLASSES),
        ("element_geometry_channels", ELEMENT_GEOMETRY_CHANNELS),
    )
    for field, expected in contracts:
        if tuple(str(value) for value in payload.get(field) or []) != expected:
            raise ValueError(f"global-program {field} contract mismatch")
    if require_production and not payload.get("production_authorized", False):
        raise PermissionError("global-program model is not authorized for production")
    input_names = payload.get("input_names")
    if input_names is not None:
        supported_inputs = (
            ["full_sheet_evidence", "crop_context"],
            ["view_evidence", "whole_sheet_evidence", "crop_context"],
        )
        if input_names not in supported_inputs:
            raise ValueError("global-program input names do not match the context contract")
        if payload.get("crop_context_contract") != ("normalized_origin_extent_sheet_edges_v1"):
            raise ValueError("global-program crop context contract mismatch")
        if (
            input_names == ["view_evidence", "whole_sheet_evidence", "crop_context"]
            and payload.get("whole_sheet_context_contract") != "explicit_complete_sheet_evidence_v1"
        ):
            raise ValueError("global-program whole-sheet context contract mismatch")


def _normalized_crop_context(
    bbox_px: tuple[int, int, int, int] | None,
    *,
    frame_bbox_px: tuple[int, int, int, int],
) -> np.ndarray:
    frame_left, frame_top, frame_right, frame_bottom = frame_bbox_px
    frame_width = frame_right - frame_left
    frame_height = frame_bottom - frame_top
    if frame_width < 1 or frame_height < 1:
        raise ValueError("crop context frame must have positive area")
    left, top, right, bottom = bbox_px or frame_bbox_px
    if not (
        frame_left <= left < right <= frame_right and frame_top <= top < bottom <= frame_bottom
    ):
        raise ValueError("crop context lies outside its whole-plan frame")
    return np.asarray(
        [
            [
                (left - frame_left) / frame_width,
                (top - frame_top) / frame_height,
                (right - left) / frame_width,
                (bottom - top) / frame_height,
                float(left == frame_left),
                float(top == frame_top),
                float(right == frame_right),
                float(bottom == frame_bottom),
            ]
        ],
        dtype=np.float32,
    )


def letterbox_evidence(
    evidence: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize a whole sheet without changing architectural angles or proportions."""

    return letterbox_cad_evidence(evidence, target_size)


def _crop_model_outputs(
    outputs: list[np.ndarray],
    content_bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    left, top, right, bottom = content_bbox
    return [output[:, :, top:bottom, left:right] for output in outputs]


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    if intersection <= 0:
        return 0.0
    left_area = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-9, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def _translate_decode(
    decoded: GlobalProgramDecodeResult,
    region: SheetPlanRegion,
) -> GlobalProgramDecodeResult:
    """Translate a crop pass without losing its source-space provenance."""

    return _translate_decode_bbox(
        decoded,
        region_id=region.id,
        bbox_px=region.bbox_px,
    )


def _translate_decode_bbox(
    decoded: GlobalProgramDecodeResult,
    *,
    region_id: str,
    bbox_px: tuple[int, int, int, int],
) -> GlobalProgramDecodeResult:
    """Translate any audited source crop back into whole-sheet coordinates."""

    left, top, _, _ = bbox_px
    proposal = decoded.proposal
    walls = [
        wall.model_copy(
            update={
                "id": f"{region_id}:{wall.id}",
                "start_px": (wall.start_px[0] + left, wall.start_px[1] + top),
                "end_px": (wall.end_px[0] + left, wall.end_px[1] + top),
                "model_version": f"{wall.model_version}+plan-instance-detail",
            }
        )
        for wall in proposal.wall_segments
    ]
    rooms = [
        room.model_copy(
            update={
                "id": f"{region_id}:{room.id}",
                "polygon_px": [(x + left, y + top) for x, y in room.polygon_px],
                "model_version": f"{room.model_version}+plan-instance-detail",
            }
        )
        for room in proposal.room_regions
    ]
    symbols = []
    for symbol in proposal.symbols:
        x0, y0, x1, y1 = symbol.bbox_px
        symbols.append(
            symbol.model_copy(
                update={
                    "id": f"{region_id}:{symbol.id}",
                    "center_px": (symbol.center_px[0] + left, symbol.center_px[1] + top),
                    "bbox_px": (x0 + left, y0 + top, x1 + left, y1 + top),
                    "model_version": f"{symbol.model_version}+plan-instance-detail",
                }
            )
        )
    semantic_seeds = [
        room.model_copy(
            update={
                "id": f"{region_id}:{room.id}",
                "polygon_px": [(x + left, y + top) for x, y in room.polygon_px],
                "model_version": f"{room.model_version}+plan-instance-detail",
            }
        )
        for room in decoded.room_semantic_seeds
    ]
    return decoded.model_copy(
        update={
            "proposal": proposal.model_copy(
                update={
                    "wall_segments": walls,
                    "room_regions": rooms,
                    "symbols": symbols,
                    "model_version": f"{proposal.model_version}+plan-instance-detail",
                    "content_sha256": "",
                }
            ).finalize(),
            "room_semantic_seeds": semantic_seeds,
        }
    )


def _line_distance(left: Any, right: Any) -> float:
    left_mid = (
        (left.start_px[0] + left.end_px[0]) / 2,
        (left.start_px[1] + left.end_px[1]) / 2,
    )
    right_mid = (
        (right.start_px[0] + right.end_px[0]) / 2,
        (right.start_px[1] + right.end_px[1]) / 2,
    )
    left_length = float(
        np.hypot(left.end_px[0] - left.start_px[0], left.end_px[1] - left.start_px[1])
    )
    right_length = float(
        np.hypot(right.end_px[0] - right.start_px[0], right.end_px[1] - right.start_px[1])
    )
    left_angle = float(
        np.arctan2(left.end_px[1] - left.start_px[1], left.end_px[0] - left.start_px[0])
    )
    right_angle = float(
        np.arctan2(right.end_px[1] - right.start_px[1], right.end_px[0] - right.start_px[0])
    )
    angle_delta = abs(np.sin(left_angle - right_angle))
    center_delta = float(np.hypot(left_mid[0] - right_mid[0], left_mid[1] - right_mid[1]))
    length_delta = abs(left_length - right_length) / max(1.0, left_length, right_length)
    return center_delta / max(4.0, left_length, right_length) + angle_delta + length_delta


def _room_bbox(room: Any) -> tuple[float, float, float, float]:
    x = [point[0] for point in room.polygon_px]
    y = [point[1] for point in room.polygon_px]
    return min(x), min(y), max(x), max(y)


def _proposal_inside_plan_region(
    proposal: Any,
    region: SheetPlanRegion | None,
    *,
    endpoint_slack_px: float = 12.0,
) -> bool:
    """Keep every fused entity inside the selected plan coordinate frame.

    Whole-sheet inference is useful context, but it may also decode neighboring
    plans, title blocks or detached structure. Once layout discovery identifies
    one plan, geometry outside that plan must remain page evidence instead of
    becoming part of the selected BIM.
    """

    if region is None:
        return True
    left, top, right, bottom = region.bbox_px

    def inside(point: tuple[float, float]) -> bool:
        return (
            left - endpoint_slack_px <= point[0] <= right + endpoint_slack_px
            and top - endpoint_slack_px <= point[1] <= bottom + endpoint_slack_px
        )

    if hasattr(proposal, "start_px") and hasattr(proposal, "end_px"):
        return inside(proposal.start_px) and inside(proposal.end_px)
    if hasattr(proposal, "polygon_px"):
        return bool(proposal.polygon_px) and all(inside(point) for point in proposal.polygon_px)
    if hasattr(proposal, "bbox_px"):
        x0, y0, x1, y1 = proposal.bbox_px
        return inside((x0, y0)) and inside((x1, y1))
    return False


def _clip_decode_to_plan_region(
    decoded: GlobalProgramDecodeResult,
    region: SheetPlanRegion | None,
) -> GlobalProgramDecodeResult:
    if region is None:
        return decoded
    proposal = decoded.proposal
    walls = [item for item in proposal.wall_segments if _proposal_inside_plan_region(item, region)]
    rooms = [item for item in proposal.room_regions if _proposal_inside_plan_region(item, region)]
    symbols = [item for item in proposal.symbols if _proposal_inside_plan_region(item, region)]
    semantic_seeds = [
        item for item in decoded.room_semantic_seeds if _proposal_inside_plan_region(item, region)
    ]
    removed = (
        len(proposal.wall_segments)
        + len(proposal.room_regions)
        + len(proposal.symbols)
        - len(walls)
        - len(rooms)
        - len(symbols)
    )
    if not removed and len(semantic_seeds) == len(decoded.room_semantic_seeds):
        return decoded
    clipped = proposal.model_copy(
        update={
            "wall_segments": walls,
            "room_regions": rooms,
            "symbols": symbols,
            "rejected_candidates": proposal.rejected_candidates + removed,
            "content_sha256": "",
        }
    ).finalize()
    diagnostics = decoded.diagnostics.model_copy(
        update={
            "structural_wall_count": len(walls),
            "room_instance_count": len(rooms),
            "element_count": len(symbols),
        }
    )
    return decoded.model_copy(
        update={
            "proposal": clipped,
            "diagnostics": diagnostics,
            "room_semantic_seeds": semantic_seeds,
        }
    )


def _consolidate_collinear_walls(walls: list[Any]) -> list[Any]:
    """Join overlapping window fragments in source coordinates.

    The operation is geometry-only and conservative: nearly parallel centerlines
    must share the same physical band and their projected intervals must touch
    or be separated only by an opening-sized gap. Perpendicular furniture edges
    and neighboring parallel walls therefore remain separate.
    """

    output = list(walls)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(output):
            left_start = np.asarray(left.start_px, dtype=np.float64)
            left_end = np.asarray(left.end_px, dtype=np.float64)
            left_vector = left_end - left_start
            left_length = float(np.linalg.norm(left_vector))
            if left_length <= 1e-6:
                continue
            direction = left_vector / left_length
            normal = np.asarray((-direction[1], direction[0]))
            left_thickness = float(left.thickness_px or 4.0)
            for right_index in range(left_index + 1, len(output)):
                right = output[right_index]
                right_start = np.asarray(right.start_px, dtype=np.float64)
                right_end = np.asarray(right.end_px, dtype=np.float64)
                right_vector = right_end - right_start
                right_length = float(np.linalg.norm(right_vector))
                if right_length <= 1e-6:
                    continue
                right_direction = right_vector / right_length
                angle_error = abs(
                    float(direction[0] * right_direction[1] - direction[1] * right_direction[0])
                )
                if angle_error > np.sin(np.deg2rad(7.5)):
                    continue
                right_thickness = float(right.thickness_px or 4.0)
                perpendicular = max(
                    abs(float(np.dot(right_start - left_start, normal))),
                    abs(float(np.dot(right_end - left_start, normal))),
                )
                if perpendicular > max(3.0, 0.65 * max(left_thickness, right_thickness)):
                    continue
                left_interval = sorted((0.0, float(np.dot(left_end - left_start, direction))))
                right_interval = sorted(
                    (
                        float(np.dot(right_start - left_start, direction)),
                        float(np.dot(right_end - left_start, direction)),
                    )
                )
                gap = max(
                    0.0,
                    max(left_interval[0], right_interval[0])
                    - min(left_interval[1], right_interval[1]),
                )
                if gap > max(10.0, 3.5 * max(left_thickness, right_thickness)):
                    continue
                lower = min(left_interval[0], right_interval[0])
                upper = max(left_interval[1], right_interval[1])
                combined_start = left_start + direction * lower
                combined_end = left_start + direction * upper
                merged = left.model_copy(
                    update={
                        "id": min(str(left.id), str(right.id)),
                        "start_px": tuple(float(value) for value in combined_start),
                        "end_px": tuple(float(value) for value in combined_end),
                        "thickness_px": (
                            left_thickness * left_length + right_thickness * right_length
                        )
                        / (left_length + right_length),
                        "confidence": max(left.confidence, right.confidence),
                        "uncertainty": min(left.uncertainty, right.uncertainty),
                        "review_required": (left.review_required or right.review_required),
                        "model_version": f"{left.model_version}+source-stitch",
                    }
                )
                output[left_index] = merged
                output.pop(right_index)
                changed = True
                break
            if changed:
                break
    return output


def _point_segment_distance(point: np.ndarray, wall: Any) -> float:
    start = np.asarray(wall.start_px, dtype=np.float64)
    end = np.asarray(wall.end_px, dtype=np.float64)
    vector = end - start
    denominator = float(np.dot(vector, vector))
    if denominator <= 1e-9:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, vector) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * vector)))


def _detail_wall_has_context(
    wall: Any,
    context_walls: list[Any],
    *,
    source_size: tuple[int, int],
) -> bool:
    """Require a native detail line to belong to the whole-building graph.

    A local crop is deliberately not allowed to decide that a line is a wall.
    Collinear refinement of a known wall is accepted. New topology must land on
    the established graph at both ends; isolated tables, labels and cabinetry
    outlines otherwise remain review evidence instead of becoming BIM walls.
    """

    if not context_walls:
        return False
    start = np.asarray(wall.start_px, dtype=np.float64)
    end = np.asarray(wall.end_px, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < max(12.0, min(source_size) * 0.012):
        return False
    direction = vector / max(length, 1e-9)
    normal = np.asarray((-direction[1], direction[0]))
    wall_thickness = float(wall.thickness_px or 4.0)
    for context in context_walls:
        context_start = np.asarray(context.start_px, dtype=np.float64)
        context_end = np.asarray(context.end_px, dtype=np.float64)
        context_vector = context_end - context_start
        context_length = float(np.linalg.norm(context_vector))
        if context_length <= 1e-9:
            continue
        context_direction = context_vector / context_length
        angle_error = abs(
            float(direction[0] * context_direction[1] - direction[1] * context_direction[0])
        )
        if angle_error > np.sin(np.deg2rad(7.5)):
            continue
        perpendicular = max(
            abs(float(np.dot(context_start - start, normal))),
            abs(float(np.dot(context_end - start, normal))),
        )
        tolerance = max(
            4.0,
            wall_thickness * 0.8,
            float(context.thickness_px or 4.0) * 0.8,
        )
        if perpendicular > tolerance:
            continue
        wall_interval = (0.0, length)
        context_interval = sorted(
            (
                float(np.dot(context_start - start, direction)),
                float(np.dot(context_end - start, direction)),
            )
        )
        overlap = max(
            0.0,
            min(wall_interval[1], context_interval[1]) - max(wall_interval[0], context_interval[0]),
        )
        if overlap >= min(length, context_length) * 0.18:
            return True

    endpoint_tolerance = max(7.0, wall_thickness * 2.2)
    start_supported = (
        min(_point_segment_distance(start, context) for context in context_walls)
        <= endpoint_tolerance
    )
    end_supported = (
        min(_point_segment_distance(end, context) for context in context_walls)
        <= endpoint_tolerance
    )
    return start_supported and end_supported


def _filter_detail_walls_by_whole_building_context(
    context_decode: GlobalProgramDecodeResult,
    detail_decodes: list[GlobalProgramDecodeResult],
    *,
    source_size: tuple[int, int],
) -> tuple[list[GlobalProgramDecodeResult], list[str]]:
    """Turn crop outputs into proposals and keep final authority global."""

    if not detail_decodes:
        return [], []
    candidates = _consolidate_collinear_walls(
        [wall for decoded in detail_decodes for wall in decoded.proposal.wall_segments]
    )
    context = list(context_decode.proposal.wall_segments)
    accepted: list[Any] = []
    rejected: list[str] = []
    # Re-run after each accepted bridge so a long structural chain may connect
    # across adjacent windows, but isolated closed furniture components cannot
    # bootstrap themselves without two anchors in the whole-building graph.
    pending = list(candidates)
    progress = True
    while progress and pending:
        progress = False
        next_pending = []
        for wall in pending:
            if _detail_wall_has_context(
                wall,
                [*context, *accepted],
                source_size=source_size,
            ):
                accepted.append(wall)
                progress = True
            else:
                next_pending.append(wall)
        pending = next_pending
    rejected.extend(str(wall.id) for wall in pending)
    if not accepted:
        return [], rejected
    proposal = (
        detail_decodes[0]
        .proposal.model_copy(
            update={
                "wall_segments": accepted,
                "room_regions": [],
                "symbols": [],
                "rejected_candidates": sum(
                    item.proposal.rejected_candidates for item in detail_decodes
                )
                + len(rejected),
                "content_sha256": "",
            }
        )
        .finalize()
    )
    consolidated = detail_decodes[0].model_copy(update={"proposal": proposal})
    return [consolidated], rejected


def _window_starts(extent: int, window: int, overlap: int) -> list[int]:
    if extent <= window:
        return [0]
    stride = window - overlap
    if stride <= 0:
        raise ValueError("native detail window overlap must be smaller than its size")
    starts = list(range(0, max(1, extent - window + 1), stride))
    final = extent - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def _strip_detail_rooms(decoded: GlobalProgramDecodeResult) -> GlobalProgramDecodeResult:
    """Local windows refine walls/elements; partial rooms never become rooms."""

    proposal = decoded.proposal.model_copy(
        update={"room_regions": [], "content_sha256": ""}
    ).finalize()
    return decoded.model_copy(update={"proposal": proposal, "room_semantic_seeds": []})


def _fuse_proposals(
    global_decode: GlobalProgramDecodeResult,
    region_decodes: list[GlobalProgramDecodeResult],
) -> GlobalProgramDecodeResult:
    """Union global context with native detail; a region pass may not erase evidence."""

    walls = list(global_decode.proposal.wall_segments)
    rooms = list(global_decode.proposal.room_regions)
    symbols = list(global_decode.proposal.symbols)
    rejected = global_decode.proposal.rejected_candidates
    for decoded in region_decodes:
        rejected += decoded.proposal.rejected_candidates
        for detail in decoded.proposal.wall_segments:
            duplicate = next(
                (index for index, item in enumerate(walls) if _line_distance(item, detail) <= 0.24),
                None,
            )
            if duplicate is None:
                walls.append(detail)
            elif detail.confidence >= walls[duplicate].confidence:
                walls[duplicate] = detail
        for detail in decoded.proposal.room_regions:
            duplicate = next(
                (
                    index
                    for index, item in enumerate(rooms)
                    if _bbox_iou(_room_bbox(item), _room_bbox(detail)) >= 0.45
                ),
                None,
            )
            if duplicate is None:
                rooms.append(detail)
            elif detail.confidence >= rooms[duplicate].confidence:
                rooms[duplicate] = detail
        for detail in decoded.proposal.symbols:
            duplicate = next(
                (
                    index
                    for index, item in enumerate(symbols)
                    if item.symbol_class == detail.symbol_class
                    and _bbox_iou(item.bbox_px, detail.bbox_px) >= 0.24
                ),
                None,
            )
            if duplicate is None:
                symbols.append(detail)
            elif detail.confidence >= symbols[duplicate].confidence:
                symbols[duplicate] = detail
    walls = _consolidate_collinear_walls(walls)
    proposal = global_decode.proposal.model_copy(
        update={
            "model_version": f"{global_decode.proposal.model_version}+hierarchical-sheet",
            "wall_segments": walls,
            "room_regions": rooms,
            "symbols": symbols,
            "rejected_candidates": rejected,
            "content_sha256": "",
        }
    ).finalize()
    diagnostics = global_decode.diagnostics.model_copy(
        update={
            "structural_wall_count": len(walls),
            "room_instance_count": len(rooms),
            "element_count": len(symbols),
            "full_sheet_context": True,
        }
    )
    return global_decode.model_copy(update={"proposal": proposal, "diagnostics": diagnostics})


class GlobalProgramOnnxRecognizer:
    """Run the global program student with immutable artifact verification."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threads: int = 1,
        require_production: bool = True,
        local_element_model_path: str | Path | None = None,
        discover_native_candidates: bool | None = None,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        self.model_path = Path(model_path).expanduser().resolve()
        self.manifest_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if not self.model_path.is_file() or not self.manifest_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_global_program_manifest(
            self.manifest,
            self.model_path,
            require_production=require_production,
        )
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - declared runtime dependency.
            raise RuntimeError("Install onnxruntime to run the global program model") from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        expected_outputs = list(self.manifest["output_names"])
        actual_outputs = [output.name for output in self.session.get_outputs()]
        if actual_outputs != expected_outputs:
            raise ValueError("global-program ONNX output names do not match the manifest")
        input_size = self.manifest.get("input_size") or []
        if len(input_size) != 2 or input_size[0] != input_size[1]:
            raise ValueError("global-program manifest requires one square input size")
        self.input_size = int(input_size[0])
        self.input_names = [item.name for item in self.session.get_inputs()]
        if self.input_names not in (
            ["full_sheet_evidence"],
            ["full_sheet_evidence", "crop_context"],
            ["view_evidence", "whole_sheet_evidence", "crop_context"],
        ):
            raise ValueError("global-program ONNX input names are unsupported")
        self.model_version = str(self.manifest["model_version"])
        self.model_sha256 = str(self.manifest["artifact_sha256"])
        self.local_element_recognizer = (
            LocalElementOnnxRecognizer(
                local_element_model_path,
                threads=threads,
                require_production=require_production,
            )
            if local_element_model_path is not None
            else None
        )
        self.discover_native_candidates = (
            local_element_model_path is not None
            if discover_native_candidates is None
            else bool(discover_native_candidates)
        )

    def _run_model(
        self,
        view_evidence: np.ndarray,
        whole_sheet_evidence: np.ndarray,
        crop_context: np.ndarray,
    ) -> list[np.ndarray]:
        if self.input_names[0] == "full_sheet_evidence":
            feed = {"full_sheet_evidence": view_evidence}
        else:
            if whole_sheet_evidence.shape != view_evidence.shape:
                raise ValueError("whole-sheet evidence batch must match the view batch")
            feed = {
                "view_evidence": view_evidence,
                "whole_sheet_evidence": whole_sheet_evidence,
            }
        if "crop_context" in self.input_names:
            context = np.asarray(crop_context, dtype=np.float32)
            if context.shape != (view_evidence.shape[0], 8):
                raise ValueError("crop context batch does not match evidence batch")
            feed["crop_context"] = context
        return self.session.run(list(self.manifest["output_names"]), feed)

    def recognize(
        self,
        image_path: str | Path,
        *,
        sheet_id: str,
        source_ref_ids: list[str],
        selected_plan_instance_id: str = "",
    ) -> GlobalProgramInferenceBundle:
        started = time.perf_counter()
        source_path = Path(image_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        evidence_started = time.perf_counter()
        full_evidence = build_cad_evidence(source)
        model_input, content_bbox = letterbox_evidence(full_evidence, self.input_size)
        evidence_ms = (time.perf_counter() - evidence_started) * 1000
        inference_started = time.perf_counter()
        outputs = self._run_model(
            model_input,
            model_input,
            _normalized_crop_context(
                None,
                frame_bbox_px=(0, 0, source.width, source.height),
            ),
        )
        outputs = _crop_model_outputs(outputs, content_bbox)
        decode_started = time.perf_counter()
        decode = decode_global_program(
            tile_id=sheet_id,
            source_ref_ids=source_ref_ids,
            model_version=self.model_version,
            source_size=source.size,
            topology_logits=outputs[0][0],
            room_semantic_logits=outputs[1][0],
            element_semantic_logits=outputs[2][0],
            element_geometry=outputs[3][0],
            uncertainty=outputs[4][0],
            source_gray=np.asarray(source.convert("L"), dtype=np.uint8),
        )
        whole_sheet_inference_ms = (time.perf_counter() - inference_started) * 1000

        # A whole-sheet pass is the context authority, not the final sampling
        # authority.  Read every credible plan instance again before local object
        # discovery so large sheets and multi-plan pages do not erase small ink.
        layout_started = time.perf_counter()
        layout = discover_plan_regions(source, sheet_id=sheet_id)
        global_scale = min(
            self.input_size / max(1, source.width),
            self.input_size / max(1, source.height),
        )
        selected_regions = (
            layout.regions if layout.multi_plan_candidate or global_scale < 0.86 else []
        )
        # Do not filter low-confidence or small plan proposals here. Layout
        # ambiguity must become a review state, never missing geometry. The
        # whole sheet stays the coordinate authority for every region pass.
        translated_decodes: list[GlobalProgramDecodeResult] = []
        region_passes: list[GlobalProgramRegionPass] = []
        for region in selected_regions:
            crop = source.crop(region.bbox_px)
            region_left, region_top, region_right, region_bottom = region.bbox_px
            crop_evidence = full_evidence[:, region_top:region_bottom, region_left:region_right]
            crop_input, crop_bbox = letterbox_evidence(crop_evidence, self.input_size)
            crop_outputs = self._run_model(
                crop_input,
                crop_input,
                _normalized_crop_context(
                    None,
                    frame_bbox_px=region.bbox_px,
                ),
            )
            crop_outputs = _crop_model_outputs(crop_outputs, crop_bbox)
            region_decode = decode_global_program(
                tile_id=region.id,
                source_ref_ids=source_ref_ids,
                model_version=f"{self.model_version}+plan-instance-detail",
                source_size=crop.size,
                topology_logits=crop_outputs[0][0],
                room_semantic_logits=crop_outputs[1][0],
                element_semantic_logits=crop_outputs[2][0],
                element_geometry=crop_outputs[3][0],
                uncertainty=crop_outputs[4][0],
                source_gray=np.asarray(crop.convert("L"), dtype=np.uint8),
            )
            detail_scale = min(
                self.input_size / max(1, crop.width),
                self.input_size / max(1, crop.height),
            )
            scale_gain = detail_scale / max(global_scale, 1e-9)
            trusted = scale_gain >= 1.12 or layout.multi_plan_candidate
            if trusted:
                translated_decodes.append(_translate_decode(region_decode, region))
            region_passes.append(
                GlobalProgramRegionPass(
                    region_id=region.id,
                    bbox_px=region.bbox_px,
                    scale_gain=scale_gain,
                    wall_count=len(region_decode.proposal.wall_segments),
                    room_count=len(region_decode.proposal.room_regions),
                    element_count=len(region_decode.proposal.symbols),
                    trusted_for_refinement=trusted,
                )
            )
        if layout.multi_plan_candidate:
            region_ids = {item.region_id for item in region_passes}
            if not selected_plan_instance_id:
                # Return the analyzed sheet and its instance list, but never
                # fuse those plans. The pipeline writes this audit artifact and
                # then stops before compilation so Studio can ask for a plan.
                pass
            elif selected_plan_instance_id not in region_ids:
                raise ValueError(
                    "selected plan instance is not present on this sheet: "
                    f"{selected_plan_instance_id}"
                )
            else:
                decode = next(
                    item
                    for item in translated_decodes
                    if item.proposal.tile_id == selected_plan_instance_id
                )
        elif translated_decodes:
            decode = _fuse_proposals(decode, translated_decodes)
        if layout.multi_plan_candidate:
            blockers = sorted(
                set(
                    decode.diagnostics.release_blockers
                    + (
                        ["multi_plan_sheet_requires_instance_selection"]
                        if not selected_plan_instance_id
                        else []
                    )
                )
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        if global_scale < 0.86 and not region_passes:
            blockers = sorted(
                set(decode.diagnostics.release_blockers + ["native_detail_plan_region_not_found"])
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        if layout.region_overflow:
            blockers = sorted(
                set(decode.diagnostics.release_blockers + ["plan_instance_region_overflow"])
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        if any(region.confidence < 0.50 for region in layout.regions):
            blockers = sorted(
                set(
                    decode.diagnostics.release_blockers
                    + ["plan_instance_region_low_confidence"]
                )
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        if layout.unassigned_structural_fraction > 0.025:
            blockers = sorted(
                set(
                    decode.diagnostics.release_blockers
                    + ["unassigned_sheet_structural_evidence"]
                )
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        if layout.source_edge_truncation:
            blockers = sorted(
                set(
                    decode.diagnostics.release_blockers + ["source_drawing_truncated_at_image_edge"]
                )
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        multiview = GlobalProgramMultiviewDiagnostics(
            layout=layout,
            region_passes=region_passes,
            fused_wall_count=len(decode.proposal.wall_segments),
            fused_room_count=len(decode.proposal.room_regions),
            fused_element_count=len(decode.proposal.symbols),
            selected_plan_instance_id=selected_plan_instance_id,
        )
        # Native-resolution discovery must operate on the actual plan, not on
        # title blocks, schedules, dimensions, or the page border.  Explicit
        # selection is authoritative on multi-plan sheets; a single credible
        # region is safe to use automatically.  Ambiguous sheets remain
        # fail-closed above instead of merging independent plans.
        active_plan_region = None
        if selected_plan_instance_id:
            active_plan_region = next(
                item for item in layout.regions if item.id == selected_plan_instance_id
            )
        elif len(layout.regions) == 1:
            # Use the sole discovered geometry frame even when it is small or
            # uncertain. The blocker above prevents silent release, while
            # retaining the frame allows native-detail recovery and review.
            active_plan_region = layout.regions[0]
        # From here onward the selected plan region is the sole geometry frame.
        # The initial whole-sheet pass remains available as context, but detached
        # page structure may not leak into the BIM or host later local elements.
        decode = _clip_decode_to_plan_region(decode, active_plan_region)
        # A second pass over the same downscaled plan is not native detail. Read
        # the complete plan through overlapping source-resolution windows while
        # retaining the whole-sheet result as the context authority. Every
        # window is recorded; none is selected by saliency or silently dropped.
        detail_window_decodes: list[GlobalProgramDecodeResult] = []
        detail_element_decodes: list[GlobalProgramDecodeResult] = []
        if active_plan_region is not None:
            region_left, region_top, region_right, region_bottom = active_plan_region.bbox_px
            region_width = region_right - region_left
            region_height = region_bottom - region_top
            active_plan_evidence = full_evidence[
                :, region_top:region_bottom, region_left:region_right
            ]
            active_plan_input, _ = letterbox_evidence(
                active_plan_evidence,
                self.input_size,
            )
            detail_window_size = max(self.input_size, min(768, self.input_size * 3))
            detail_overlap = max(32, detail_window_size // 4)
            window_index = 0
            for local_top in _window_starts(region_height, detail_window_size, detail_overlap):
                for local_left in _window_starts(region_width, detail_window_size, detail_overlap):
                    bbox_px = (
                        region_left + local_left,
                        region_top + local_top,
                        min(region_right, region_left + local_left + detail_window_size),
                        min(region_bottom, region_top + local_top + detail_window_size),
                    )
                    window_id = f"{active_plan_region.id}:detail-{window_index:04d}"
                    window_index += 1
                    crop = source.crop(bbox_px)
                    crop_evidence = full_evidence[
                        :, bbox_px[1] : bbox_px[3], bbox_px[0] : bbox_px[2]
                    ]
                    crop_input, crop_bbox = letterbox_evidence(crop_evidence, self.input_size)
                    crop_outputs = self._run_model(
                        crop_input,
                        active_plan_input,
                        _normalized_crop_context(
                            bbox_px,
                            frame_bbox_px=active_plan_region.bbox_px,
                        ),
                    )
                    crop_outputs = _crop_model_outputs(crop_outputs, crop_bbox)
                    window_decode = decode_global_program(
                        tile_id=window_id,
                        source_ref_ids=source_ref_ids,
                        model_version=f"{self.model_version}+native-detail-window",
                        source_size=crop.size,
                        topology_logits=crop_outputs[0][0],
                        room_semantic_logits=crop_outputs[1][0],
                        element_semantic_logits=crop_outputs[2][0],
                        element_geometry=crop_outputs[3][0],
                        uncertainty=crop_outputs[4][0],
                        source_gray=np.asarray(crop.convert("L"), dtype=np.uint8),
                    )
                    window_element_count = len(window_decode.proposal.symbols)
                    if window_element_count and self.local_element_recognizer is not None:
                        candidate_symbols = [
                            symbol.model_copy(update={"review_required": True})
                            for symbol in window_decode.proposal.symbols
                        ]
                        element_only_decode = window_decode.model_copy(
                            update={
                                "proposal": window_decode.proposal.model_copy(
                                    update={
                                        "wall_segments": [],
                                        "room_regions": [],
                                        "symbols": candidate_symbols,
                                        "content_sha256": "",
                                    }
                                ).finalize()
                            }
                        )
                        detail_element_decodes.append(
                            _translate_decode_bbox(
                                element_only_decode,
                                region_id=window_id,
                                bbox_px=bbox_px,
                            )
                        )
                    window_decode = _strip_detail_rooms(window_decode)
                    # Local windows recover structural topology. Their dense
                    # element head is not an instance detector: the same long
                    # opening can produce many overlapping boxes in adjacent
                    # windows. Native candidate + local-classifier inference
                    # below is the sole source-resolution element authority.
                    window_decode = window_decode.model_copy(
                        update={
                            "proposal": window_decode.proposal.model_copy(
                                update={"symbols": [], "content_sha256": ""}
                            ).finalize()
                        }
                    )
                    translated = _translate_decode_bbox(
                        window_decode,
                        region_id=window_id,
                        bbox_px=bbox_px,
                    )
                    detail_window_decodes.append(translated)
                    region_passes.append(
                        GlobalProgramRegionPass(
                            region_id=window_id,
                            bbox_px=bbox_px,
                            scale_gain=min(
                                self.input_size / max(1, crop.width),
                                self.input_size / max(1, crop.height),
                            )
                            / max(global_scale, 1e-9),
                            wall_count=len(window_decode.proposal.wall_segments),
                            room_count=0,
                            element_count=window_element_count,
                            trusted_for_refinement=True,
                            pass_kind="native_detail_window",
                            context_x=local_left / max(1, region_width),
                            context_y=local_top / max(1, region_height),
                            context_width=crop.width / max(1, region_width),
                            context_height=crop.height / max(1, region_height),
                            touches_left=local_left == 0,
                            touches_top=local_top == 0,
                            touches_right=bbox_px[2] == region_right,
                            touches_bottom=bbox_px[3] == region_bottom,
                        )
                    )
        context_rejected_detail_wall_ids: list[str] = []
        if detail_window_decodes:
            context_detail_decodes, context_rejected_detail_wall_ids = (
                _filter_detail_walls_by_whole_building_context(
                    decode,
                    detail_window_decodes,
                    source_size=source.size,
                )
            )
            if context_detail_decodes:
                decode = _fuse_proposals(decode, context_detail_decodes)
        # Dense window semantics are proposals, never final BIM instances. They
        # restore objects that native connected-component mining cannot invent
        # (open cabinets, composite sanitary fixtures, furniture assemblies).
        # The local objectness head must confirm every one before compilation.
        if detail_element_decodes and self.local_element_recognizer is not None:
            decode = _fuse_proposals(decode, detail_element_decodes)
        multiview = multiview.model_copy(
            update={
                "region_passes": region_passes,
                "fused_wall_count": len(decode.proposal.wall_segments),
                "fused_room_count": len(decode.proposal.room_regions),
                "fused_element_count": len(decode.proposal.symbols),
                "context_rejected_detail_wall_count": len(context_rejected_detail_wall_ids),
                "context_rejected_detail_wall_ids": context_rejected_detail_wall_ids,
            }
        )
        multiview_ms = (time.perf_counter() - layout_started) * 1000
        # Local symbols are intentionally deferred until the complete wall graph
        # and its enclosed room faces have been reconstructed.  The previous
        # order supplied provisional, often fragmented structure to the local
        # recognizer even though its contract calls the inputs host/room context.
        # That made a nominally whole-sheet model behave like an isolated crop
        # classifier.  Structural geometry now has authority over local identity.
        local_diagnostics = None
        native_wall_source = source
        native_wall_offset = (0.0, 0.0)
        if active_plan_region is not None:
            native_wall_source = source.crop(active_plan_region.bbox_px)
            native_wall_offset = (
                float(active_plan_region.bbox_px[0]),
                float(active_plan_region.bbox_px[1]),
            )
        native_wall_candidates, native_wall_diagnostics = mine_native_wall_candidates(
            native_wall_source
        )
        if native_wall_offset != (0.0, 0.0):
            native_wall_candidates = [
                translate_native_wall_candidate(item, offset_px=native_wall_offset)
                for item in native_wall_candidates
            ]
        refined_walls, native_wall_refinement_count = refine_context_walls_with_native_bands(
            decode.proposal.wall_segments,
            native_wall_candidates,
        )
        if native_wall_refinement_count:
            decode = decode.model_copy(
                update={
                    "proposal": decode.proposal.model_copy(
                        update={
                            "wall_segments": refined_walls,
                            "content_sha256": "",
                        }
                    ).finalize()
                }
            )
        promoted_native_walls, _ = promote_supported_native_wall_candidates(
            native_wall_candidates,
            decode.proposal.wall_segments,
            source_ref_ids=source_ref_ids,
            source_size=source.size,
        )
        if promoted_native_walls:
            native_wall_proposal = decode.proposal.model_copy(
                update={
                    "wall_segments": promoted_native_walls,
                    "room_regions": [],
                    "symbols": [],
                    "content_sha256": "",
                }
            ).finalize()
            native_wall_decode = decode.model_copy(update={"proposal": native_wall_proposal})
            decode = _fuse_proposals(decode, [native_wall_decode])
        provisional_rooms = list(decode.proposal.room_regions)
        topology_rooms = reconstruct_rooms_from_wall_graph(
            decode.proposal.wall_segments,
            decode.room_semantic_seeds or decode.proposal.room_regions,
            source_size=source.size,
            source_ref_ids=source_ref_ids,
            model_version=decode.proposal.model_version,
        )
        resolved_rooms = merge_topology_and_provisional_rooms(
            topology_rooms,
            provisional_rooms,
            source_size=source.size,
            source_ref_ids=source_ref_ids,
            model_version=decode.proposal.model_version,
        )
        if resolved_rooms:
            decode = decode.model_copy(
                update={
                    "proposal": decode.proposal.model_copy(
                        update={
                            "room_regions": resolved_rooms,
                            "content_sha256": "",
                        }
                    ).finalize(),
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"room_instance_count": len(resolved_rooms)}
                    ),
                }
            )
        native_openings, native_opening_diagnostics = infer_openings_from_wall_graph(
            source,
            decode.proposal.wall_segments,
            model_version=decode.proposal.model_version,
        )
        consolidated_walls = consolidate_walls_across_openings(
            decode.proposal.wall_segments,
            native_openings,
            image_size=source.size,
        )
        # Openings are owned by the final wall graph.  Remove free-standing
        # opening classifications before local fixture inference; otherwise a
        # crop model can invent a door or window with no corresponding wall gap.
        non_opening_symbols = [
            item for item in decode.proposal.symbols if item.symbol_class not in {"door", "window"}
        ]
        decode = decode.model_copy(
            update={
                "proposal": decode.proposal.model_copy(
                    update={
                        "symbols": [*non_opening_symbols, *native_openings],
                        "wall_segments": consolidated_walls,
                        "content_sha256": "",
                    }
                ).finalize()
            }
        )
        if self.local_element_recognizer is not None:
            refined_symbols, local_diagnostics = self.local_element_recognizer.refine(
                source,
                decode.proposal.symbols,
                discover_candidates=self.discover_native_candidates,
                source_ref_ids=source_ref_ids,
                discovery_region_px=(
                    active_plan_region.bbox_px if active_plan_region is not None else None
                ),
                full_evidence=full_evidence,
                host_walls=decode.proposal.wall_segments,
                room_regions=decode.proposal.room_regions,
            )
            refined_symbols = [
                item for item in refined_symbols if item.symbol_class not in {"door", "window"}
            ]
            refined_symbols = [*refined_symbols, *native_openings]
            refined_proposal = decode.proposal.model_copy(
                update={"symbols": refined_symbols}
            ).finalize()
            decode = decode.model_copy(update={"proposal": refined_proposal})
            multiview = multiview.model_copy(
                update={
                    "fused_element_count": len(refined_symbols),
                    "unresolved_native_candidate_count": len(
                        local_diagnostics.unresolved_discovered
                    ),
                }
            )
            if local_diagnostics.unresolved_discovered:
                blockers = sorted(
                    set(
                        decode.diagnostics.release_blockers
                        + ["unresolved_native_element_candidates"]
                    )
                )
                decode = decode.model_copy(
                    update={
                        "diagnostics": decode.diagnostics.model_copy(
                            update={"release_blockers": blockers}
                        )
                    }
                )
        unresolved_walls = unresolved_native_wall_candidates(
            native_wall_candidates,
            decode.proposal.wall_segments,
        )
        multiview = multiview.model_copy(
            update={
                "native_wall_candidate_diagnostics": native_wall_diagnostics,
                "native_opening_diagnostics": native_opening_diagnostics,
                "unresolved_native_walls": unresolved_walls,
                "native_wall_geometry_refinement_count": (native_wall_refinement_count),
            }
        )
        if unresolved_walls:
            blockers = sorted(
                set(decode.diagnostics.release_blockers + ["unresolved_native_wall_candidates"])
            )
            decode = decode.model_copy(
                update={
                    "diagnostics": decode.diagnostics.model_copy(
                        update={"release_blockers": blockers}
                    )
                }
            )
        evidence_graph = build_spatial_evidence_graph_from_proposal(
            decode.proposal,
            source_size=source.size,
            full_sheet_context=True,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000
        return GlobalProgramInferenceBundle(
            input_path=str(source_path),
            input_sha256=sha256_file(source_path),
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            production_authorized=bool(self.manifest.get("production_authorized", False)),
            decode=decode,
            multiview=multiview,
            local_element_refinement=local_diagnostics,
            evidence_graph=evidence_graph,
            timings_ms={
                "evidence": round(evidence_ms, 3),
                "onnx": round(whole_sheet_inference_ms, 3),
                "hierarchical_multiview": round(multiview_ms, 3),
                "decode_and_graph": round(decode_ms, 3),
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        )
