"""Build local-element calibration data from direct source-pixel review.

This module is intentionally separate from synthetic pretraining.  A model that
only sees clean generated glyphs cannot learn the real background distribution
of architectural sheets: text, dimensions, wall joins, hatches, and fragments
outnumber actual object symbols.  Here native candidates on a reviewed sheet
are assigned from direct annotations as one complete-object positive per
fixture, an explicit background fragment, or an ignored ambiguous overlap.
Evaluation-only drawings remain sealed from commercial training and their
hashes are recorded as mandatory evaluation exclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from .core.hashing import sha256_file, sha256_json
from .core.model.cad_evidence import build_cad_evidence, letterbox_cad_evidence
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
)
from .core.model.local_element_student import (
    LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_FEATURES,
)
from .ground_truth import (
    assert_commercial_training_eligible,
    compile_benchmark_graph_from_manifest,
    validate_ground_truth_manifest,
)
from .local_element_candidates import mine_native_element_candidates
from .local_element_crops import (
    CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LOCAL_ELEMENT_INPUT_CHANNELS,
    candidate_hypothesis_context,
    element_geometry_target,
    extract_local_element_hierarchy_evidence_from_map,
    normalized_candidate_context,
    semantic_element_context,
)

DirectCorpusPurpose = Literal["research_calibration", "production_training"]
CandidateSupervisionState = Literal["positive", "background", "ignore"]

DIRECT_CANDIDATE_SUPERVISION_CONTRACT = (
    "one_best_iou_050_positive_fragments_background_ambiguous_ignore_v3"
)


def _bbox_overlap_metrics(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return IoU, target coverage, and candidate coverage."""

    tx0, ty0, tx1, ty1 = target
    cx0, cy0, cx1, cy1 = candidate
    intersection = max(0.0, min(tx1, cx1) - max(tx0, cx0)) * max(
        0.0, min(ty1, cy1) - max(ty0, cy0)
    )
    target_area = max(1e-6, (tx1 - tx0) * (ty1 - ty0))
    candidate_area = max(1e-6, (cx1 - cx0) * (cy1 - cy0))
    union = target_area + candidate_area - intersection
    return (
        intersection / max(union, 1e-6),
        intersection / target_area,
        intersection / candidate_area,
    )


def _candidate_supervision_state(
    candidate: tuple[float, float, float, float],
    targets: list[tuple[float, float, float, float]],
) -> CandidateSupervisionState:
    """Classify an unselected proposal without inventing a semantic label.

    IoU 0.50 is the same complete-object boundary used by geometry evaluation.
    Alternate complete proposals and partial overlaps are ignored: calling them
    background would teach the objectness head to reject a valid object.  Only
    clear fragments, multi-object envelopes, and unrelated proposals become
    background examples.
    """

    metrics = [_bbox_overlap_metrics(target, candidate) for target in targets]
    if any(iou >= 0.50 for iou, _, _ in metrics):
        return "ignore"
    if any(
        candidate_coverage >= 0.70 and target_coverage <= 0.42
        for _, target_coverage, candidate_coverage in metrics
    ):
        return "background"
    if any(
        target_coverage >= 0.70 and candidate_coverage <= 0.35
        for _, target_coverage, candidate_coverage in metrics
    ):
        return "background"
    if max((iou for iou, _, _ in metrics), default=0.0) <= 0.08:
        return "background"
    return "ignore"


def _one_best_candidate_per_target(
    candidates: list[tuple[float, float, float, float]],
    targets: list[tuple[float, float, float, float]],
) -> dict[int, tuple[int, float]]:
    """Greedily form a deterministic one-to-one complete-object assignment."""

    edges = sorted(
        (
            (iou, target_index, candidate_index)
            for target_index, target in enumerate(targets)
            for candidate_index, candidate in enumerate(candidates)
            if (iou := _bbox_overlap_metrics(target, candidate)[0]) >= 0.50
        ),
        key=lambda edge: (-edge[0], edge[1], edge[2]),
    )
    assigned_targets: set[int] = set()
    assigned_candidates: dict[int, tuple[int, float]] = {}
    for iou, target_index, candidate_index in edges:
        if (
            target_index in assigned_targets
            or candidate_index in assigned_candidates
        ):
            continue
        assigned_targets.add(target_index)
        assigned_candidates[candidate_index] = (target_index, iou)
    return assigned_candidates


def _evenly_spaced_indices(indices: list[int], maximum: int) -> list[int]:
    """Retain negatives across the full candidate ledger, not its first region."""

    if maximum <= 0:
        return []
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return [indices[int(position)] for position in positions]


def _polygon_bbox(entity: dict[str, Any]) -> tuple[float, float, float, float]:
    points = entity.get("polygon") or []
    if len(points) < 3:
        raise ValueError(f"entity {entity.get('id', '<unknown>')} has no polygon")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _wall_thickness_px(entity: dict[str, Any]) -> float:
    polygon = entity.get("polygon") or []
    if len(polygon) >= 3:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return max(2.0, min(width, height))
    return 8.0


def _canonical_fixture_class(value: object) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    aliases = {
        "appliance": "electrical_appliance",
        "cabinet": "base_cabinet",
        "storage": "base_cabinet",
        "watertap": "water_tap",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ELEMENT_PROGRAM_CLASSES else "unknown"


def _write_sample(
    *,
    cursor: int,
    label_name: str,
    proposal_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float] | None,
    yaw_deg: float,
    full_evidence: np.ndarray,
    image_size: tuple[int, int],
    input_size: int,
    semantic_rooms: list[tuple[str, list[tuple[float, float]]]],
    semantic_walls: list[
        tuple[tuple[float, float], tuple[float, float], float]
    ],
    candidate_boxes: list[tuple[float, float, float, float]],
    evidence: np.memmap,
    labels: np.memmap,
    geometry: np.memmap,
    geometry_valid: np.memmap,
    candidate_context: np.memmap,
    sample_indices: np.memmap,
    sample_index: int,
) -> None:
    crop_evidence, transform = extract_local_element_hierarchy_evidence_from_map(
        full_evidence,
        image_size,
        proposal_bbox,
        input_size=input_size,
        detail_scale=2.0,
        assembly_scale=6.2,
        room_scale=17.0,
    )
    evidence[cursor] = np.uint8(np.clip(crop_evidence, 0.0, 1.0) * 255)
    labels[cursor] = ELEMENT_PROGRAM_CLASSES.index(label_name)
    if target_bbox is None:
        geometry[cursor] = 0
        geometry_valid[cursor] = 0
    else:
        geometry[cursor] = element_geometry_target(
            target_bbox,
            transform,
            yaw_deg=yaw_deg,
        )
        geometry_valid[cursor] = 1
    candidate_context[cursor] = np.concatenate(
        (
            normalized_candidate_context(
                proposal_bbox,
                image_size=image_size,
                letterbox_size=input_size,
            ),
            semantic_element_context(
                proposal_bbox,
                image_size=image_size,
                rooms=semantic_rooms,
                walls=semantic_walls,
            ),
            candidate_hypothesis_context(proposal_bbox, candidate_boxes),
        )
    )
    sample_indices[cursor] = sample_index


def build_direct_local_element_corpus(
    ground_truth_root: str | Path,
    output_root: str | Path,
    *,
    purpose: DirectCorpusPurpose,
    input_size: int = 64,
    proposal_aligned_positive: bool = True,
    maximum_hard_negatives_per_sheet: int = 512,
) -> dict[str, Any]:
    """Create a fail-closed real-sheet corpus without pseudo-labels.

    Positive classes and geometry come only from the direct source-pixel
    manifest.  The native miner contributes crop locations, never labels.
    Complete-object positives, explicit background, and ambiguous ignored
    proposals share the same geometry contract as evaluation.
    """

    if purpose not in {"research_calibration", "production_training"}:
        raise ValueError("invalid direct local-element corpus purpose")
    if input_size < 32 or maximum_hard_negatives_per_sheet < 0:
        raise ValueError("invalid direct local-element corpus options")

    source_root = Path(ground_truth_root).expanduser().resolve()
    packets: list[dict[str, Any]] = []
    for manifest_path in sorted(source_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_ground_truth_manifest(manifest, source_root=manifest_path.parent)
        if purpose == "production_training":
            assert_commercial_training_eligible(manifest)
        graph = compile_benchmark_graph_from_manifest(manifest)
        source = manifest["source"]
        image_path = Path(str(source["image_path"]))
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        packets.append(
            {
                "manifest_path": manifest_path,
                "manifest": manifest,
                "graph": graph,
                "image_path": image_path.resolve(),
            }
        )
    if not packets:
        raise ValueError("no valid direct source-pixel annotation packets found")

    prepared: list[dict[str, Any]] = []
    total = 0
    for packet in packets:
        graph = packet["graph"]
        with Image.open(packet["image_path"]) as opened:
            image = opened.convert("RGB")
        candidates, diagnostics = mine_native_element_candidates(
            image,
            source_ref_ids=[str(packet["manifest"]["source"]["image_sha256"])],
        )
        fixture_boxes = [_polygon_bbox(entity) for entity in graph["fixtures"]]
        candidate_boxes = [candidate.bbox_px for candidate in candidates]
        aligned_candidate_indices = (
            _one_best_candidate_per_target(candidate_boxes, fixture_boxes)
            if proposal_aligned_positive
            else {}
        )
        ignored_candidate_indices = [
            index
            for index, candidate in enumerate(candidates)
            if index not in aligned_candidate_indices
            and _candidate_supervision_state(candidate.bbox_px, fixture_boxes)
            == "ignore"
        ]
        ignored_set = set(ignored_candidate_indices)
        background_candidate_indices = [
            index
            for index in range(len(candidates))
            if index not in aligned_candidate_indices and index not in ignored_set
        ]
        hard_negative_indices = _evenly_spaced_indices(
            background_candidate_indices,
            maximum_hard_negatives_per_sheet,
        )
        positive_count = len(fixture_boxes) + len(aligned_candidate_indices)
        item_count = positive_count + len(hard_negative_indices)
        total += item_count
        prepared.append(
            {
                **packet,
                "image": image,
                "candidates": candidates,
                "candidate_diagnostics": diagnostics,
                "fixture_boxes": fixture_boxes,
                "aligned_candidate_indices": aligned_candidate_indices,
                "ignored_candidate_indices": ignored_candidate_indices,
                "background_candidate_count": len(background_candidate_indices),
                "hard_negative_indices": hard_negative_indices,
                "item_count": item_count,
            }
        )

    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    evidence_path = destination / "evidence.npy"
    labels_path = destination / "labels.npy"
    geometry_path = destination / "geometry.npy"
    geometry_valid_path = destination / "geometry-valid.npy"
    whole_evidence_path = destination / "whole-sheet-evidence.npy"
    candidate_context_path = destination / "candidate-context.npy"
    sample_indices_path = destination / "sample-indices.npy"
    evidence = np.lib.format.open_memmap(
        evidence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, LOCAL_ELEMENT_INPUT_CHANNELS, input_size, input_size),
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int16, shape=(total,)
    )
    geometry = np.lib.format.open_memmap(
        geometry_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, len(ELEMENT_GEOMETRY_CHANNELS)),
    )
    geometry_valid = np.lib.format.open_memmap(
        geometry_valid_path, mode="w+", dtype=np.uint8, shape=(total,)
    )
    whole_evidence = np.lib.format.open_memmap(
        whole_evidence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(prepared), 4, input_size, input_size),
    )
    candidate_context = np.lib.format.open_memmap(
        candidate_context_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, LOCAL_ELEMENT_CONTEXT_FEATURES),
    )
    sample_indices = np.lib.format.open_memmap(
        sample_indices_path, mode="w+", dtype=np.int32, shape=(total,)
    )

    class_counts = {name: 0 for name in ELEMENT_PROGRAM_CLASSES}
    records: list[dict[str, Any]] = []
    cursor = 0
    for sample_index, packet in enumerate(prepared):
        image = packet["image"]
        graph = packet["graph"]
        full_evidence = build_cad_evidence(image)
        normalized_whole, _ = letterbox_cad_evidence(full_evidence, input_size)
        whole_evidence[sample_index] = np.uint8(
            np.clip(normalized_whole[0], 0.0, 1.0) * 255
        )
        semantic_rooms = [
            (
                str(room.get("room_type") or "unknown"),
                [tuple(map(float, point)) for point in room["polygon"]],
            )
            for room in graph["rooms"]
        ]
        semantic_walls = [
            (
                tuple(map(float, wall["from"])),
                tuple(map(float, wall["to"])),
                _wall_thickness_px(wall),
            )
            for wall in graph["walls"]
        ]
        candidate_boxes = [candidate.bbox_px for candidate in packet["candidates"]]
        start = cursor
        for fixture, target_bbox in zip(
            graph["fixtures"], packet["fixture_boxes"], strict=True
        ):
            label_name = _canonical_fixture_class(fixture.get("fixture_type"))
            _write_sample(
                cursor=cursor,
                label_name=label_name,
                proposal_bbox=target_bbox,
                target_bbox=target_bbox,
                yaw_deg=float(fixture.get("yaw_deg") or 0.0),
                full_evidence=full_evidence,
                image_size=image.size,
                input_size=input_size,
                semantic_rooms=semantic_rooms,
                semantic_walls=semantic_walls,
                candidate_boxes=candidate_boxes,
                evidence=evidence,
                labels=labels,
                geometry=geometry,
                geometry_valid=geometry_valid,
                candidate_context=candidate_context,
                sample_indices=sample_indices,
                sample_index=sample_index,
            )
            class_counts[label_name] += 1
            cursor += 1
        for candidate_index, (target_index, _) in packet[
            "aligned_candidate_indices"
        ].items():
            fixture = graph["fixtures"][target_index]
            target_bbox = packet["fixture_boxes"][target_index]
            proposal_bbox = packet["candidates"][candidate_index].bbox_px
            label_name = _canonical_fixture_class(fixture.get("fixture_type"))
            _write_sample(
                cursor=cursor,
                label_name=label_name,
                proposal_bbox=proposal_bbox,
                target_bbox=target_bbox,
                yaw_deg=float(fixture.get("yaw_deg") or 0.0),
                full_evidence=full_evidence,
                image_size=image.size,
                input_size=input_size,
                semantic_rooms=semantic_rooms,
                semantic_walls=semantic_walls,
                candidate_boxes=candidate_boxes,
                evidence=evidence,
                labels=labels,
                geometry=geometry,
                geometry_valid=geometry_valid,
                candidate_context=candidate_context,
                sample_indices=sample_indices,
                sample_index=sample_index,
            )
            class_counts[label_name] += 1
            cursor += 1
        for candidate_index in packet["hard_negative_indices"]:
            proposal_bbox = packet["candidates"][candidate_index].bbox_px
            _write_sample(
                cursor=cursor,
                label_name="background",
                proposal_bbox=proposal_bbox,
                target_bbox=None,
                yaw_deg=0.0,
                full_evidence=full_evidence,
                image_size=image.size,
                input_size=input_size,
                semantic_rooms=semantic_rooms,
                semantic_walls=semantic_walls,
                candidate_boxes=candidate_boxes,
                evidence=evidence,
                labels=labels,
                geometry=geometry,
                geometry_valid=geometry_valid,
                candidate_context=candidate_context,
                sample_indices=sample_indices,
                sample_index=sample_index,
            )
            class_counts["background"] += 1
            cursor += 1
        source = packet["manifest"]["source"]
        records.append(
            {
                "sheet_id": source["sheet_id"],
                "source_image_sha256": source["image_sha256"],
                "source_license_scope": source["license_scope"],
                "annotation_manifest_sha256": sha256_file(packet["manifest_path"]),
                "start_index": start,
                "item_count": cursor - start,
                "direct_fixture_count": len(graph["fixtures"]),
                "native_candidate_count": len(packet["candidates"]),
                "aligned_native_positive_count": len(
                    packet["aligned_candidate_indices"]
                ),
                "ignored_ambiguous_candidate_count": len(
                    packet["ignored_candidate_indices"]
                ),
                "explicit_background_candidate_count": packet[
                    "background_candidate_count"
                ],
                "hard_negative_count": len(packet["hard_negative_indices"]),
            }
        )
    if cursor != total:
        raise AssertionError("direct local-element corpus allocation mismatch")
    for array in (
        evidence,
        labels,
        geometry,
        geometry_valid,
        whole_evidence,
        candidate_context,
        sample_indices,
    ):
        array.flush()

    research_only = purpose == "research_calibration"
    manifest = {
        "schema_version": "dajoong.direct-local-element-corpus.v1",
        "role": (
            "direct_real_research_calibration_only"
            if research_only
            else "direct_real_commercial_training"
        ),
        "ground_truth_policy": "direct_visual_source_annotation_only",
        "label_origin": "direct_source_pixel_manifest_only",
        "candidate_role": "crop_proposal_only_never_semantic_label",
        "candidate_supervision_contract": DIRECT_CANDIDATE_SUPERVISION_CONTRACT,
        "production_training_eligible": not research_only,
        "evaluation_eligible": False,
        "evaluation_exclusion_source_sha256": [
            record["source_image_sha256"] for record in records
        ],
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "objectness_label_contract": "background_zero_foreground_one_v1",
        "class_label_contract": "foreground_taxonomy_conditional_on_objectness_v1",
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "input_size": input_size,
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "whole_sheet_input_channels": 4,
        "candidate_context_features": LOCAL_ELEMENT_CONTEXT_FEATURES,
        "proposal_aligned_positive": proposal_aligned_positive,
        "maximum_hard_negatives_per_sheet": maximum_hard_negatives_per_sheet,
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "sample_count": len(records),
        "item_count": total,
        "class_counts": class_counts,
        "files": {
            "evidence": {"path": evidence_path.name, "sha256": sha256_file(evidence_path)},
            "labels": {"path": labels_path.name, "sha256": sha256_file(labels_path)},
            "geometry": {"path": geometry_path.name, "sha256": sha256_file(geometry_path)},
            "geometry_valid": {
                "path": geometry_valid_path.name,
                "sha256": sha256_file(geometry_valid_path),
            },
            "whole_evidence": {
                "path": whole_evidence_path.name,
                "sha256": sha256_file(whole_evidence_path),
            },
            "candidate_context": {
                "path": candidate_context_path.name,
                "sha256": sha256_file(candidate_context_path),
            },
            "sample_indices": {
                "path": sample_indices_path.name,
                "sha256": sha256_file(sample_indices_path),
            },
        },
        "records": records,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
