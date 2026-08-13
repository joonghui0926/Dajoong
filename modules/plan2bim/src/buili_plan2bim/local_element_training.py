"""Synthetic-only corpus and trainer for the native-detail element student."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core.hashing import sha256_file, sha256_json
from .core.model.cad_evidence import (
    ORIENTED_EVIDENCE_ROTATION_CONTRACT,
    build_cad_evidence,
    letterbox_cad_evidence,
)
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
)
from .core.model.local_element_student import (
    ELEMENT_CLASS_FAMILY_INDICES,
    ELEMENT_FAMILY_CLASSES,
    ELEMENT_FAMILY_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_FEATURES,
    DajoongLocalElementStudent,
    LocalElementStudentConfig,
    LocalElementStudentOnnxAdapter,
)
from .direct_local_element_corpus import DIRECT_CANDIDATE_SUPERVISION_CONTRACT
from .global_topology_training import _balanced_class_weights, _split_records
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
from .synthetic_pretraining import (
    FIXTURE_HOSTING_CONTRACT,
    SyntheticPretrainingSample,
    assert_synthetic_pretraining_only,
)
from .training_augmentation import (
    deterministic_quadrant,
    rotate_element_geometry,
    rotate_normalized_bbox_context,
    rotate_oriented_evidence,
)

CANDIDATE_ALIGNMENT_CONTRACT = "mutual_coverage_072_iou_055_or_truth_v1"

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:  # pragma: no cover - runtime install intentionally omits torch.
    torch = None
    nn = None
    functional = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None


def _opening_bbox(
    center: tuple[float, float],
    width: float,
    orientation: str,
) -> tuple[float, float, float, float]:
    half_width = width / 2
    half_depth = max(3.0, width * 0.08)
    if orientation == "horizontal":
        return (
            center[0] - half_width,
            center[1] - half_depth,
            center[0] + half_width,
            center[1] + half_depth,
        )
    return (
        center[0] - half_depth,
        center[1] - half_width,
        center[0] + half_depth,
        center[1] + half_width,
    )


def _overlap_fraction(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    area = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    return intersection / area


def _mutual_overlap_coverage(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Reject negatives that cover a truth box or are covered by one.

    Candidate components can be much larger or much smaller than a labeled
    object.  A one-sided denominator previously mislabeled large components
    containing a real object as background.
    """

    return max(_overlap_fraction(left, right), _overlap_fraction(right, left))


def _candidate_aligned_positive_bbox(
    target: tuple[float, float, float, float],
    candidates: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Use the same proposal geometry at training time that inference sees."""

    if not candidates:
        return target

    best = max(candidates, key=lambda candidate: _candidate_alignment_score(target, candidate))
    return best if _candidate_alignment_score(target, best) >= 0.0 else target


def _candidate_alignment_score(
    target: tuple[float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> float:
    """Return a strict whole-object alignment score, or ``-1`` for fragments.

    The candidate miner is allowed to over-cover ink, but a positive training
    crop must explain one complete annotated object in both directions.  A
    small stroke inside an object and a large envelope containing several
    objects are therefore background hypotheses, not silent omissions.
    """

    target_area = max(1e-9, (target[2] - target[0]) * (target[3] - target[1]))
    candidate_area = max(
        1e-9,
        (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
    )
    area_ratio = candidate_area / target_area
    if not 0.18 <= area_ratio <= 5.5:
        return -1.0
    intersection_width = max(
        0.0,
        min(candidate[2], target[2]) - max(candidate[0], target[0]),
    )
    intersection_height = max(
        0.0,
        min(candidate[3], target[3]) - max(candidate[1], target[1]),
    )
    intersection = intersection_width * intersection_height
    intersection_over_target = intersection / target_area
    intersection_over_candidate = intersection / candidate_area
    union = candidate_area + target_area - intersection
    iou = intersection / max(union, 1e-9)
    if (
        intersection_over_target < 0.72
        or intersection_over_candidate < 0.72
        or iou < 0.55
    ):
        return -1.0
    return iou + min(intersection_over_target, intersection_over_candidate)


def _fragment_negative_candidate_bboxes(
    target: tuple[float, float, float, float],
    candidates: list[tuple[float, float, float, float]],
    other_objects: list[tuple[float, float, float, float]],
    *,
    limit: int,
) -> list[tuple[float, float, float, float]]:
    """Return proposal fragments that must not become standalone BIM objects.

    The native miner intentionally retains both whole-object hypotheses and the
    disconnected strokes that produced them. Earlier corpora trained only the
    best whole-object proposal and silently omitted its sibling fragments. That
    made duplicate boxes unavoidable at inference because a fragment had never
    been shown to the classifier as background.
    """

    if limit < 0:
        raise ValueError("fragment-negative limit cannot be negative")
    target_area = max(1e-9, (target[2] - target[0]) * (target[3] - target[1]))
    ranked: list[tuple[float, tuple[float, float, float, float]]] = []
    for candidate in candidates:
        candidate_area = max(
            1e-9,
            (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
        )
        intersection_width = max(
            0.0,
            min(candidate[2], target[2]) - max(candidate[0], target[0]),
        )
        intersection_height = max(
            0.0,
            min(candidate[3], target[3]) - max(candidate[1], target[1]),
        )
        intersection = intersection_width * intersection_height
        target_coverage = intersection / target_area
        candidate_coverage = intersection / candidate_area
        area_ratio = candidate_area / target_area
        if not (
            candidate_coverage >= 0.65
            and 0.04 <= target_coverage <= 0.62
            and area_ratio <= 0.72
        ):
            continue
        if any(
            _mutual_overlap_coverage(candidate, other) >= 0.18
            for other in other_objects
        ):
            continue
        fragment_score = candidate_coverage + target_coverage - area_ratio * 0.15
        ranked.append((fragment_score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked[:limit]]


def _negative_bbox(
    rng: random.Random,
    image_size: tuple[int, int],
    occupied: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    width, height = image_size
    for _ in range(48):
        side = rng.uniform(12.0, 42.0)
        center_x = rng.uniform(side, max(side, width - side))
        center_y = rng.uniform(side, max(side, height - side))
        candidate = (
            center_x - side / 2,
            center_y - side / 2,
            center_x + side / 2,
            center_y + side / 2,
        )
        if all(_mutual_overlap_coverage(candidate, item) < 0.08 for item in occupied):
            return candidate
    return (2.0, 2.0, 14.0, 14.0)


def build_synthetic_local_element_corpus(
    source_corpus_root: str | Path,
    output_root: str | Path,
    *,
    input_size: int = 64,
    negatives_per_sheet: int = 2,
    hard_negatives_per_sheet: int = 0,
    proposal_aligned_positives_per_object: int = 0,
    fragment_negatives_per_object: int = 0,
    seed: int = 26_081_104,
) -> dict[str, Any]:
    if (
        input_size < 32
        or negatives_per_sheet < 0
        or hard_negatives_per_sheet < 0
        or proposal_aligned_positives_per_object not in {0, 1}
        or fragment_negatives_per_object < 0
    ):
        raise ValueError("invalid local element corpus options")
    source_root = Path(source_corpus_root).expanduser().resolve()
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    assert_synthetic_pretraining_only(source_manifest)
    annotations = sorted((source_root / "annotations").glob("*.json"))
    samples: list[tuple[Path, SyntheticPretrainingSample]] = []
    source_generator_versions: set[str] = set()
    source_canvas_profiles: set[str] = set()
    fixture_hosting_contracts: set[str] = set()
    total = 0
    for annotation in annotations:
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        assert_synthetic_pretraining_only(payload)
        sample = SyntheticPretrainingSample.model_validate(payload)
        samples.append((annotation, sample))
        source_generator_versions.add(sample.generator_version)
        source_canvas_profiles.add(sample.canvas_profile)
        fixture_hosting_contracts.add(sample.fixture_hosting_contract)
        total += (
            (len(sample.openings) + len(sample.fixtures))
            * (
                1
                + proposal_aligned_positives_per_object
                + fragment_negatives_per_object
            )
            + negatives_per_sheet
            + hard_negatives_per_sheet
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
        labels_path,
        mode="w+",
        dtype=np.int16,
        shape=(total,),
    )
    geometry = np.lib.format.open_memmap(
        geometry_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, len(ELEMENT_GEOMETRY_CHANNELS)),
    )
    geometry_valid = np.lib.format.open_memmap(
        geometry_valid_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total,),
    )
    whole_evidence = np.lib.format.open_memmap(
        whole_evidence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(samples), 4, input_size, input_size),
    )
    candidate_context = np.lib.format.open_memmap(
        candidate_context_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, LOCAL_ELEMENT_CONTEXT_FEATURES),
    )
    sample_indices = np.lib.format.open_memmap(
        sample_indices_path,
        mode="w+",
        dtype=np.int32,
        shape=(total,),
    )
    class_counts = {name: 0 for name in ELEMENT_PROGRAM_CLASSES}
    records: list[dict[str, Any]] = []
    cursor = 0
    for sample_index, (annotation, sample) in enumerate(samples):
        rng = random.Random(f"{seed}/{sample.sample_id}")
        with Image.open(source_root / sample.image_path) as opened:
            image = opened.convert("RGB")
        full_evidence = build_cad_evidence(image)
        normalized_whole, _ = letterbox_cad_evidence(full_evidence, input_size)
        whole_evidence[sample_index] = np.uint8(
            np.clip(normalized_whole[0], 0.0, 1.0) * 255
        )
        objects: list[tuple[str, tuple[float, float, float, float], float]] = []
        for opening in sample.openings:
            objects.append(
                (
                    opening.kind,
                    _opening_bbox(
                        opening.center_px,
                        opening.width_px,
                        opening.orientation,
                    ),
                    0.0 if opening.orientation == "horizontal" else 90.0,
                )
            )
        objects.extend(
            (fixture.fixture_type, fixture.bbox_px, fixture.yaw_deg)
            for fixture in sample.fixtures
        )
        candidates, _ = mine_native_element_candidates(
            image,
            source_ref_ids=[sample.image_sha256],
        )
        candidate_boxes = [item.bbox_px for item in candidates]
        semantic_rooms = [
            (room.room_class, list(room.polygon_px)) for room in sample.rooms
        ]
        semantic_walls = [
            (start, end, 8.0) for start, end in sample.walls
        ]

        def candidate_building_context(
            candidate_bbox: tuple[float, float, float, float],
            *,
            image_size: tuple[int, int] = image.size,
            rooms: list[tuple[str, list[tuple[float, float]]]] = semantic_rooms,
            walls: list[
                tuple[tuple[float, float], tuple[float, float], float]
            ] = semantic_walls,
            hierarchy_boxes: list[
                tuple[float, float, float, float]
            ] = candidate_boxes,
        ) -> np.ndarray:
            return np.concatenate(
                (
                    normalized_candidate_context(
                        candidate_bbox,
                        image_size=image_size,
                        letterbox_size=input_size,
                    ),
                    semantic_element_context(
                        candidate_bbox,
                        image_size=image_size,
                        rooms=rooms,
                        walls=walls,
                    ),
                    candidate_hypothesis_context(
                        candidate_bbox,
                        hierarchy_boxes,
                    ),
                )
            )
        start = cursor
        occupied = [item[1] for item in objects]
        for object_index, (class_name, bbox, yaw_deg) in enumerate(objects):
            if class_name not in ELEMENT_PROGRAM_CLASSES:
                class_name = "unknown"
            proposal_boxes = [bbox]
            if proposal_aligned_positives_per_object:
                proposal_boxes.append(
                    _candidate_aligned_positive_bbox(bbox, candidate_boxes)
                )
            for proposal_bbox in proposal_boxes:
                detail_scale = rng.uniform(1.65, 2.55)
                crop_evidence, transform = extract_local_element_hierarchy_evidence_from_map(
                    full_evidence,
                    image.size,
                    proposal_bbox,
                    input_size=input_size,
                    detail_scale=detail_scale,
                    assembly_scale=detail_scale * rng.uniform(2.8, 3.4),
                    room_scale=detail_scale * rng.uniform(7.5, 9.5),
                    center_jitter=(
                        rng.uniform(-0.12, 0.12),
                        rng.uniform(-0.12, 0.12),
                    ),
                )
                evidence[cursor] = np.uint8(
                    np.clip(crop_evidence, 0.0, 1.0) * 255
                )
                labels[cursor] = ELEMENT_PROGRAM_CLASSES.index(class_name)
                geometry[cursor] = element_geometry_target(
                    bbox,
                    transform,
                    yaw_deg=yaw_deg,
                )
                geometry_valid[cursor] = 1
                candidate_context[cursor] = candidate_building_context(proposal_bbox)
                sample_indices[cursor] = sample_index
                class_counts[class_name] += 1
                cursor += 1
            fragment_boxes = _fragment_negative_candidate_bboxes(
                bbox,
                candidate_boxes,
                [
                    other_bbox
                    for other_index, (_, other_bbox, _) in enumerate(objects)
                    if other_index != object_index
                ],
                limit=fragment_negatives_per_object,
            )
            while len(fragment_boxes) < fragment_negatives_per_object:
                fragment_boxes.append(_negative_bbox(rng, image.size, occupied))
            for fragment_bbox in fragment_boxes:
                detail_scale = rng.uniform(1.7, 2.5)
                crop_evidence, _ = extract_local_element_hierarchy_evidence_from_map(
                    full_evidence,
                    image.size,
                    fragment_bbox,
                    input_size=input_size,
                    detail_scale=detail_scale,
                    assembly_scale=detail_scale * rng.uniform(2.8, 3.4),
                    room_scale=detail_scale * rng.uniform(7.5, 9.5),
                    center_jitter=(
                        rng.uniform(-0.10, 0.10),
                        rng.uniform(-0.10, 0.10),
                    ),
                )
                evidence[cursor] = np.uint8(
                    np.clip(crop_evidence, 0.0, 1.0) * 255
                )
                labels[cursor] = 0
                geometry[cursor] = 0
                geometry_valid[cursor] = 0
                candidate_context[cursor] = candidate_building_context(fragment_bbox)
                sample_indices[cursor] = sample_index
                class_counts["background"] += 1
                cursor += 1
        for _ in range(negatives_per_sheet):
            bbox = _negative_bbox(rng, image.size, occupied)
            detail_scale = rng.uniform(1.6, 2.4)
            crop_evidence, _ = extract_local_element_hierarchy_evidence_from_map(
                full_evidence,
                image.size,
                bbox,
                input_size=input_size,
                detail_scale=detail_scale,
                assembly_scale=detail_scale * rng.uniform(2.8, 3.4),
                room_scale=detail_scale * rng.uniform(7.5, 9.5),
                center_jitter=(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)),
            )
            evidence[cursor] = np.uint8(np.clip(crop_evidence, 0.0, 1.0) * 255)
            labels[cursor] = 0
            geometry[cursor] = 0
            geometry_valid[cursor] = 0
            candidate_context[cursor] = candidate_building_context(bbox)
            sample_indices[cursor] = sample_index
            class_counts["background"] += 1
            cursor += 1
        if hard_negatives_per_sheet:
            hard_boxes = [
                item.bbox_px
                for item in candidates
                if all(
                    _mutual_overlap_coverage(item.bbox_px, occupied_box) < 0.05
                    for occupied_box in occupied
                )
            ]
            rng.shuffle(hard_boxes)
            for index in range(hard_negatives_per_sheet):
                bbox = (
                    hard_boxes[index]
                    if index < len(hard_boxes)
                    else _negative_bbox(rng, image.size, occupied)
                )
                detail_scale = rng.uniform(1.7, 2.5)
                crop_evidence, _ = extract_local_element_hierarchy_evidence_from_map(
                    full_evidence,
                    image.size,
                    bbox,
                    input_size=input_size,
                    detail_scale=detail_scale,
                    assembly_scale=detail_scale * rng.uniform(2.8, 3.4),
                    room_scale=detail_scale * rng.uniform(7.5, 9.5),
                    center_jitter=(
                        rng.uniform(-0.12, 0.12),
                        rng.uniform(-0.12, 0.12),
                    ),
                )
                evidence[cursor] = np.uint8(np.clip(crop_evidence, 0.0, 1.0) * 255)
                labels[cursor] = 0
                geometry[cursor] = 0
                geometry_valid[cursor] = 0
                candidate_context[cursor] = candidate_building_context(bbox)
                sample_indices[cursor] = sample_index
                class_counts["background"] += 1
                cursor += 1
        records.append(
            {
                "sample_id": sample.sample_id,
                "annotation_sha256": sha256_file(annotation),
                "start_index": start,
                "item_count": cursor - start,
            }
        )
    if cursor != total:
        raise AssertionError("local element corpus allocation mismatch")
    evidence.flush()
    labels.flush()
    geometry.flush()
    geometry_valid.flush()
    whole_evidence.flush()
    candidate_context.flush()
    sample_indices.flush()
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.synthetic-local-element-corpus.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "source_corpus_sha256": sha256_file(source_manifest_path),
        "source_generator_versions": sorted(source_generator_versions),
        "source_canvas_profiles": sorted(source_canvas_profiles),
        "fixture_hosting_contracts": sorted(fixture_hosting_contracts),
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "objectness_label_contract": "background_zero_foreground_one_v1",
        "class_label_contract": "foreground_taxonomy_conditional_on_objectness_v1",
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "input_size": input_size,
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "random_negatives_per_sheet": negatives_per_sheet,
        "hard_negatives_per_sheet": hard_negatives_per_sheet,
        "proposal_aligned_positives_per_object": (
            proposal_aligned_positives_per_object
        ),
        "fragment_negatives_per_object": fragment_negatives_per_object,
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
    assert_synthetic_pretraining_only(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


@dataclass(frozen=True)
class LocalElementTrainOptions:
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 26_081_105
    device: str = "auto"
    validation_fraction: float = 0.1
    # Loss weighting corrects imbalance without repeatedly presenting the same
    # rare glyphs. Replacement sampling overfit those glyphs and hurt common-class
    # calibration in the 2,000-sheet synthetic holdout.
    balanced_sampling: bool = False
    quadrant_augmentation: bool = True
    # Room labels are weak context, not program truth. Open-plan kitchens,
    # shared utility spaces, and multi-use rooms regularly contain equipment
    # whose semantic family differs from the printed room name.
    semantic_context_dropout: float = 0.75
    background_weight: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer options")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half")
        if self.background_weight <= 0:
            raise ValueError("background_weight must be positive")
        if not 0.0 <= self.semantic_context_dropout < 1.0:
            raise ValueError("semantic_context_dropout must be in [0, 1)")


if torch is not None:

    class SyntheticLocalElementDataset(Dataset):
        def __init__(
            self,
            corpus_root: str | Path,
            *,
            split: str,
            validation_fraction: float,
            corpus_role: str = "synthetic_pretrain_only",
            quadrant_augmentation: bool = False,
            augmentation_seed: int = 26_081_105,
            semantic_context_dropout: float = 0.0,
        ) -> None:
            allowed_roles = {
                "synthetic_pretrain_only",
                "direct_real_research_calibration_only",
            }
            if corpus_role not in allowed_roles:
                raise ValueError("unsupported local element corpus role")
            allowed_splits = (
                {"train", "validation"}
                if corpus_role == "synthetic_pretrain_only"
                else {"all"}
            )
            if split not in allowed_splits:
                raise ValueError(
                    "synthetic corpora require train/validation; "
                    "direct research corpora require the explicit all split"
                )
            self.root = Path(corpus_root).expanduser().resolve()
            self.manifest = json.loads(
                (self.root / "manifest.json").read_text(encoding="utf-8")
            )
            if self.manifest.get("role") != corpus_role:
                raise ValueError("local element corpus role mismatch")
            if corpus_role == "synthetic_pretrain_only":
                assert_synthetic_pretraining_only(self.manifest)
            else:
                if self.manifest.get("ground_truth_policy") != (
                    "direct_visual_source_annotation_only"
                ):
                    raise ValueError("direct corpus ground-truth policy mismatch")
                if self.manifest.get("label_origin") != (
                    "direct_source_pixel_manifest_only"
                ):
                    raise ValueError("direct corpus label origin mismatch")
                if self.manifest.get("candidate_role") != (
                    "crop_proposal_only_never_semantic_label"
                ):
                    raise ValueError("direct corpus candidate role mismatch")
                if self.manifest.get("production_training_eligible") is not False:
                    raise ValueError("research corpus cannot be production eligible")
                if self.manifest.get("evaluation_eligible") is not False:
                    raise ValueError("research calibration corpus cannot be evaluation")
                if not self.manifest.get("evaluation_exclusion_source_sha256"):
                    raise ValueError("direct corpus must record evaluation exclusions")
            if self.manifest.get("input_contract") != LOCAL_ELEMENT_EVIDENCE_CONTRACT:
                raise ValueError(
                    "local element evidence contract mismatch: "
                    f"expected {LOCAL_ELEMENT_EVIDENCE_CONTRACT!r}"
                )
            if corpus_role == "synthetic_pretrain_only":
                if set(self.manifest.get("source_canvas_profiles") or ()) != {
                    "square",
                    "portrait",
                    "landscape",
                }:
                    raise ValueError(
                        "local training corpus must cover square, portrait, and "
                        "landscape sheets"
                    )
                if set(self.manifest.get("fixture_hosting_contracts") or ()) != {
                    FIXTURE_HOSTING_CONTRACT
                }:
                    raise ValueError(
                        "local training corpus fixture hosting contract mismatch"
                    )
            if self.manifest.get("local_view_contract") != (
                "native_detail_assembly_room_v1"
            ):
                raise ValueError("local element hierarchy contract mismatch")
            if self.manifest.get("objectness_label_contract") != (
                "background_zero_foreground_one_v1"
            ):
                raise ValueError("local element objectness label contract mismatch")
            if corpus_role == "synthetic_pretrain_only" and (
                self.manifest.get("candidate_alignment_contract")
                != CANDIDATE_ALIGNMENT_CONTRACT
            ):
                raise ValueError("local candidate alignment contract mismatch")
            if self.manifest.get("candidate_hypothesis_context_contract") != (
                CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
            ):
                raise ValueError("local proposal graph context contract mismatch")
            if self.manifest.get("candidate_context_contract") != (
                LOCAL_ELEMENT_CONTEXT_CONTRACT
            ):
                raise ValueError("local candidate context contract mismatch")
            if tuple(self.manifest["classes"]) != ELEMENT_PROGRAM_CLASSES:
                raise ValueError("local element class contract mismatch")
            if split == "all":
                records = list(self.manifest["records"])
            else:
                training, validation = _split_records(
                    list(self.manifest["records"]),
                    validation_fraction,
                )
                records = validation if split == "validation" else training
            self.indices = [
                index
                for record in records
                for index in range(
                    int(record["start_index"]),
                    int(record["start_index"]) + int(record["item_count"]),
                )
            ]
            self.quadrant_augmentation = (
                split != "validation" and quadrant_augmentation
            )
            self.augmentation_seed = augmentation_seed
            self.semantic_context_dropout = semantic_context_dropout
            files = self.manifest["files"]
            self.evidence = np.load(self.root / files["evidence"]["path"], mmap_mode="r")
            self.labels = np.load(self.root / files["labels"]["path"], mmap_mode="r")
            self.geometry = np.load(self.root / files["geometry"]["path"], mmap_mode="r")
            self.geometry_valid = np.load(
                self.root / files["geometry_valid"]["path"],
                mmap_mode="r",
            )
            self.whole_evidence = np.load(
                self.root / files["whole_evidence"]["path"], mmap_mode="r"
            )
            self.candidate_context = np.load(
                self.root / files["candidate_context"]["path"], mmap_mode="r"
            )
            self.sample_indices = np.load(
                self.root / files["sample_indices"]["path"], mmap_mode="r"
            )

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.indices[index]
            evidence = np.asarray(self.evidence[item], dtype=np.float32) / 255.0
            geometry = np.asarray(self.geometry[item], dtype=np.float32).copy()
            whole_sheet_evidence = (
                np.asarray(
                    self.whole_evidence[int(self.sample_indices[item])],
                    dtype=np.float32,
                )
                / 255.0
            )
            candidate_context = np.asarray(
                self.candidate_context[item], dtype=np.float32
            ).copy()
            if self.quadrant_augmentation:
                quadrants = deterministic_quadrant(
                    item,
                    seed=self.augmentation_seed,
                )
                evidence = rotate_oriented_evidence(evidence, quadrants)
                geometry = rotate_element_geometry(
                    geometry,
                    quadrants,
                    spatial=False,
                )
                whole_sheet_evidence = rotate_oriented_evidence(
                    whole_sheet_evidence,
                    quadrants,
                )
                candidate_context = rotate_normalized_bbox_context(
                    candidate_context,
                    quadrants,
                )
            # Room names are useful but optional evidence.  Open-plan and
            # mixed-use drawings frequently assign a fixture to an adjacent
            # broad room label.  Drop the room one-hot deterministically so
            # the student cannot use synthetic room taxonomy as a shortcut.
            semantic_rng = random.Random(
                f"{self.augmentation_seed}/{item}/semantic-context"
            )
            if semantic_rng.random() < self.semantic_context_dropout:
                room_start = 4
                room_stop = room_start + len(ROOM_PROGRAM_CLASSES)
                candidate_context[room_start:room_stop] = 0.0
            return {
                "evidence": torch.from_numpy(evidence.copy()),
                "whole_sheet_evidence": torch.from_numpy(
                    whole_sheet_evidence.copy()
                ),
                "candidate_context": torch.from_numpy(candidate_context.copy()),
                "label": torch.tensor(int(self.labels[item]), dtype=torch.long),
                "geometry": torch.from_numpy(geometry.copy()),
                "geometry_valid": torch.tensor(
                    float(self.geometry_valid[item]),
                    dtype=torch.float32,
                ),
            }

        def class_counts(self) -> list[int]:
            values = np.asarray(self.labels[self.indices], dtype=np.int64)
            return [
                int(value)
                for value in np.bincount(
                    values,
                    minlength=len(ELEMENT_PROGRAM_CLASSES),
                )[: len(ELEMENT_PROGRAM_CLASSES)]
            ]

        def sampling_weights(self) -> list[float]:
            """Match proposal reality while keeping rare foreground learnable.

            The retired sampler assigned equal probability to background and to
            each of 47 foreground classes.  Background therefore occupied about
            two percent of training batches despite exceeding 95 percent of the
            native proposal stream on a hard real sheet.  Reserve half of each
            epoch for background and distribute the other half across supported
            foreground classes.
            """

            counts = self.class_counts()
            foreground_classes = sum(count > 0 for count in counts[1:])
            return [
                (
                    0.5 / max(1, counts[0])
                    if int(self.labels[item]) == 0
                    else 0.5
                    / max(1, foreground_classes)
                    / max(1, counts[int(self.labels[item])])
                )
                for item in self.indices
            ]


    class LocalElementCriterion(nn.Module):
        def __init__(self, class_weights: list[float]) -> None:
            super().__init__()
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )

        def forward(self, output: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
            foreground = target["label"] > 0
            if foreground.any():
                # The taxonomy is conditional on object existence; background
                # never teaches a random foreground family through this loss.
                classification = functional.cross_entropy(
                    output["class_logits"][foreground, 1:],
                    target["label"][foreground] - 1,
                    weight=self.class_weights[1:],
                )
            else:
                classification = output["class_logits"].sum() * 0.0
                family = output["family_logits"].sum() * 0.0
            if foreground.any():
                family_lookup = torch.as_tensor(
                    ELEMENT_CLASS_FAMILY_INDICES,
                    dtype=torch.long,
                    device=target["label"].device,
                )
                family_target = family_lookup[target["label"][foreground] - 1]
                family = functional.cross_entropy(
                    output["family_logits"][foreground],
                    family_target,
                )
            objectness_target = foreground.float().unsqueeze(1)
            objectness = functional.binary_cross_entropy(
                output["objectness"],
                objectness_target,
            )
            valid = target["geometry_valid"].unsqueeze(1)
            geometry = (
                functional.smooth_l1_loss(
                    output["geometry"] * valid,
                    target["geometry"] * valid,
                    reduction="sum",
                )
                / (valid.sum() * len(ELEMENT_GEOMETRY_CHANNELS)).clamp_min(1.0)
            )
            with torch.no_grad():
                conditional = functional.softmax(
                    output["class_logits"][:, 1:], dim=1
                )
                foreground_label = (target["label"] - 1).clamp_min(0).unsqueeze(1)
                class_correct_probability = conditional.gather(
                    1, foreground_label
                )
                correct_probability = torch.where(
                    objectness_target.bool(),
                    output["objectness"] * class_correct_probability,
                    1.0 - output["objectness"],
                )
                risk_target = 1.0 - correct_probability
            uncertainty = functional.binary_cross_entropy(
                output["uncertainty"],
                risk_target,
            )
            total = (
                classification
                + family * 0.45
                + objectness
                + geometry * 0.5
                + uncertainty * 0.05
            )
            return {
                "total": total,
                "classification": classification,
                "family": family,
                "objectness": objectness,
                "geometry": geometry,
                "uncertainty": uncertainty,
            }


    def _joint_foreground_prediction(output: dict[str, Any]) -> Any:
        fine_probability = functional.softmax(output["class_logits"][:, 1:], dim=1)
        # The fine head is already normalized across every foreground class.
        # Multiplying it by a second family softmax double-counted taxonomy and
        # pushed valid real symbols below the acceptance threshold.  The family
        # head remains an auxiliary representation loss, not a second prior.
        return fine_probability.argmax(dim=1) + 1


    def _joint_foreground_confidence(output: dict[str, Any]) -> Any:
        fine_probability = functional.softmax(output["class_logits"][:, 1:], dim=1)
        return fine_probability.max(dim=1).values

else:  # pragma: no cover

    class SyntheticLocalElementDataset:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")


def _run_local_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: str,
    optimizer: Any | None,
    progress_label: str = "",
    progress_interval: int = 100,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    confusion = (
        None
        if training
        else torch.zeros(
            (len(ELEMENT_PROGRAM_CLASSES), len(ELEMENT_PROGRAM_CLASSES)),
            dtype=torch.int64,
        )
    )
    batches = 0
    for batch in loader:
        evidence = batch["evidence"].to(device)
        whole_sheet_evidence = batch["whole_sheet_evidence"].to(device)
        candidate_context = batch["candidate_context"].to(device)
        target = {
            "label": batch["label"].to(device),
            "geometry": batch["geometry"].to(device),
            "geometry_valid": batch["geometry_valid"].to(device),
        }
        with torch.set_grad_enabled(training):
            output = model(evidence, whole_sheet_evidence, candidate_context)
            losses = criterion(output, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if confusion is not None:
            conditional_prediction = _joint_foreground_prediction(output)
            prediction = torch.where(
                (output["objectness"][:, 0] >= 0.5)
                & (_joint_foreground_confidence(output) >= 0.25),
                conditional_prediction,
                torch.zeros_like(conditional_prediction),
            ).detach().cpu()
            truth = target["label"].detach().cpu()
            count = len(ELEMENT_PROGRAM_CLASSES)
            confusion += torch.bincount(
                truth * count + prediction,
                minlength=count * count,
            ).reshape(count, count)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if progress_label and batches % progress_interval == 0:
            print(
                f"{progress_label}: {batches}/{len(loader)} batches",
                flush=True,
            )
    metrics = {name: value / max(1, batches) for name, value in totals.items()}
    if confusion is not None:
        _, macro_f1, micro_f1 = _local_classification_metrics(confusion)
        metrics["foreground_macro_f1"] = macro_f1
        metrics["foreground_micro_f1"] = micro_f1
    return metrics


def _local_classification_metrics(
    confusion: Any,
) -> tuple[list[dict[str, Any]], float, float]:
    rows = []
    foreground_tp = foreground_fp = foreground_fn = 0
    for index, class_name in enumerate(ELEMENT_PROGRAM_CLASSES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        support = int(confusion[index, :].sum())
        denominator = 2 * true_positive + false_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        rows.append(
            {
                "class_name": class_name,
                "support_items": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": (
                    0.0
                    if precision_denominator == 0
                    else true_positive / precision_denominator
                ),
                "recall": (
                    0.0
                    if recall_denominator == 0
                    else true_positive / recall_denominator
                ),
                "f1": 0.0 if denominator == 0 else 2 * true_positive / denominator,
            }
        )
        if index > 0:
            foreground_tp += true_positive
            foreground_fp += false_positive
            foreground_fn += false_negative
    supported = [row["f1"] for row in rows[1:] if row["support_items"] > 0]
    micro_denominator = 2 * foreground_tp + foreground_fp + foreground_fn
    return (
        rows,
        sum(supported) / max(1, len(supported)),
        0.0
        if micro_denominator == 0
        else 2 * foreground_tp / micro_denominator,
    )


def _warm_start_local_model(
    model: Any,
    payload: dict[str, Any],
    target_config: LocalElementStudentConfig,
) -> dict[str, int]:
    """Transfer the detail/context encoder while expanding the BIM taxonomy."""

    source_config = dict(payload.get("config") or {})
    target_payload = target_config.to_dict()
    compatible_fields = (
        "input_size",
        "stem_width",
        "geometry_channels",
    )
    if any(source_config.get(key) != target_payload.get(key) for key in compatible_fields):
        raise ValueError("initial checkpoint local architecture is incompatible")
    source_state = payload["state_dict"]
    target_state = model.state_dict()
    transferred_tensors = 0
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
            transferred_tensors += 1
        semantic_name = (
            f"semantic_encoder.{name.removeprefix('encoder.')}"
            if name.startswith("encoder.")
            else ""
        )
        if (
            semantic_name
            and semantic_name in target_state
            and target_state[semantic_name].shape == value.shape
        ):
            target_state[semantic_name] = value
            transferred_tensors += 1
        structure_whole_name = (
            f"structure_whole_sheet_encoder.{name.removeprefix('whole_sheet_encoder.')}"
            if name.startswith("whole_sheet_encoder.")
            else ""
        )
        if (
            structure_whole_name
            and structure_whole_name in target_state
            and target_state[structure_whole_name].shape == value.shape
        ):
            target_state[structure_whole_name] = value
            transferred_tensors += 1
        structure_whole_projection_name = ""
        if name.startswith("whole_sheet_global_projection."):
            structure_whole_projection_name = name.replace(
                "whole_sheet_global_projection.",
                "structure_whole_global_projection.",
                1,
            )
        elif name.startswith("whole_sheet_location_projection."):
            structure_whole_projection_name = name.replace(
                "whole_sheet_location_projection.",
                "structure_whole_location_projection.",
                1,
            )
        if (
            structure_whole_projection_name
            and structure_whole_projection_name in target_state
            and target_state[structure_whole_projection_name].shape == value.shape
        ):
            target_state[structure_whole_projection_name] = value
            transferred_tensors += 1
    # V7 saw detail + one context view (8 channels). V8 keeps those weights,
    # then initializes assembly and room views from the learned context filters.
    # This is a contract-aware expansion, not a silent shape mismatch.
    first_conv = "encoder.0.0.weight"
    semantic_first_conv = "semantic_encoder.0.0.weight"
    source_channels = int(source_config.get("input_channels") or 0)
    if (
        source_channels == 8
        and target_config.input_channels == 12
        and first_conv in source_state
        and first_conv in target_state
        and source_state[first_conv].shape[1] == 8
        and target_state[first_conv].shape[1] == 12
    ):
        target_state[first_conv][:, :4] = source_state[first_conv][:, :4]
        target_state[first_conv][:, 4:8] = source_state[first_conv][:, 4:8]
        target_state[first_conv][:, 8:12] = source_state[first_conv][:, 4:8]
        transferred_tensors += 1
        if semantic_first_conv in target_state:
            target_state[semantic_first_conv][:, :4] = source_state[first_conv][:, :4]
            target_state[semantic_first_conv][:, 4:8] = source_state[first_conv][:, 4:8]
            target_state[semantic_first_conv][:, 8:12] = source_state[first_conv][:, 4:8]
            transferred_tensors += 1
    source_classes = tuple(source_config.get("classes") or ())
    shared_classes = set(source_classes) & set(target_config.classes)
    for suffix in ("weight", "bias"):
        name = f"class_head.{suffix}"
        if name not in source_state or name not in target_state:
            continue
        for class_name in shared_classes:
            target_index = target_config.classes.index(class_name)
            source_index = source_classes.index(class_name)
            target_state[name][target_index] = source_state[name][source_index]
    model.load_state_dict(target_state)
    return {
        "transferred_tensor_count": transferred_tensors,
        "transferred_element_class_count": len(shared_classes),
    }


def train_local_element_student(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    options: LocalElementTrainOptions | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    options = options or LocalElementTrainOptions()
    options.validate()
    corpus_manifest = json.loads(
        (Path(corpus_root).expanduser().resolve() / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert_synthetic_pretraining_only(corpus_manifest)
    missing_classes = sorted(
        class_name
        for class_name in ELEMENT_PROGRAM_CLASSES
        if class_name != "unknown"
        and int(corpus_manifest.get("class_counts", {}).get(class_name, 0)) <= 0
    )
    if missing_classes:
        raise ValueError(
            "local element training corpus is missing taxonomy classes: "
            + ", ".join(missing_classes)
        )
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = SyntheticLocalElementDataset(
        corpus_root,
        split="train",
        validation_fraction=options.validation_fraction,
        quadrant_augmentation=options.quadrant_augmentation,
        augmentation_seed=options.seed,
        semantic_context_dropout=options.semantic_context_dropout,
    )
    validation_data = SyntheticLocalElementDataset(
        corpus_root,
        split="validation",
        validation_fraction=options.validation_fraction,
        quadrant_augmentation=False,
        augmentation_seed=options.seed,
        semantic_context_dropout=0.0,
    )
    sampler = None
    if options.balanced_sampling:
        if WeightedRandomSampler is None:
            raise RuntimeError("balanced local sampling requires torch")
        sampler = WeightedRandomSampler(
            train_data.sampling_weights(),
            num_samples=len(train_data),
            replacement=True,
            generator=torch.Generator().manual_seed(options.seed),
        )
    train_loader = DataLoader(
        train_data,
        batch_size=options.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
    )
    counts = train_data.class_counts()
    class_weights = _balanced_class_weights(
        counts,
        ELEMENT_PROGRAM_CLASSES,
        background_weight=options.background_weight,
    )
    config = LocalElementStudentConfig()
    model = DajoongLocalElementStudent(config).to(device)
    initialization: dict[str, Any] = {
        "initial_checkpoint_sha256": "",
        "initial_checkpoint_epoch": 0,
        "transferred_tensor_count": 0,
        "transferred_element_class_count": 0,
    }
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint).expanduser().resolve()
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        if initial_payload.get("role") != "synthetic_pretrain_only":
            raise ValueError("initial checkpoint must be synthetic pretraining only")
        if initial_payload.get("oriented_evidence_rotation_contract") != (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ):
            raise ValueError(
                "initial checkpoint oriented-evidence rotation contract mismatch"
            )
        if initial_payload.get("candidate_alignment_contract") != (
            CANDIDATE_ALIGNMENT_CONTRACT
        ):
            raise ValueError("initial checkpoint candidate alignment contract mismatch")
        if initial_payload.get("candidate_hypothesis_context_contract") != (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ):
            raise ValueError("initial checkpoint proposal graph context mismatch")
        if initial_payload.get("input_contract") != LOCAL_ELEMENT_EVIDENCE_CONTRACT:
            raise ValueError("initial checkpoint local evidence contract mismatch")
        if initial_payload.get("candidate_context_contract") != (
            LOCAL_ELEMENT_CONTEXT_CONTRACT
        ):
            raise ValueError("initial checkpoint candidate context contract mismatch")
        transfer = _warm_start_local_model(model, initial_payload, config)
        initialization = {
            "initial_checkpoint_sha256": sha256_file(initial_path),
            "initial_checkpoint_epoch": int(initial_payload.get("epoch") or 0),
            **transfer,
        }
    criterion = LocalElementCriterion(class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best.pt"
    best_validation = float("inf")
    best_validation_f1 = -1.0
    history = []
    started = time.time()
    for epoch in range(options.epochs):
        train_metrics = _run_local_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            progress_label=f"epoch {epoch + 1}/{options.epochs} train",
        )
        with torch.inference_mode():
            validation_metrics = _run_local_epoch(
                model,
                validation_loader,
                criterion,
                device=device,
                optimizer=None,
                progress_label=f"epoch {epoch + 1}/{options.epochs} validation",
            )
        history.append(
            {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics}
        )
        validation_f1 = validation_metrics["foreground_macro_f1"]
        improves_f1 = validation_f1 > best_validation_f1 + 1e-9
        ties_f1_with_lower_loss = (
            abs(validation_f1 - best_validation_f1) <= 1e-9
            and validation_metrics["total"] < best_validation
        )
        if improves_f1 or ties_f1_with_lower_loss:
            best_validation = validation_metrics["total"]
            best_validation_f1 = validation_f1
            torch.save(
                {
                    "schema_version": "dajoong.local-element-checkpoint.v1",
                    "role": "synthetic_pretrain_only",
                    "real_drawing_ground_truth": False,
                    "evaluation_eligible": False,
                    "production_authorized": False,
                    "oriented_evidence_rotation_contract": (
                        ORIENTED_EVIDENCE_ROTATION_CONTRACT
                    ),
                    "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
                    "candidate_hypothesis_context_contract": (
                        CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
                    ),
                    "perception_authority_contract": (
                        LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
                    ),
                    "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                    "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
                    "config": config.to_dict(),
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": best_validation,
                    "validation_foreground_macro_f1": best_validation_f1,
                },
                checkpoint_path,
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = model.cpu().eval()
    model.load_state_dict(payload["state_dict"])
    onnx_path = destination / "local-element-student.onnx"
    output_names = [
        "class_logits",
        "family_logits",
        "objectness",
        "geometry",
        "uncertainty",
    ]
    torch.onnx.export(
        LocalElementStudentOnnxAdapter(model),
        (
            torch.zeros(1, config.input_channels, config.input_size, config.input_size),
            torch.zeros(
                1,
                config.whole_sheet_input_channels,
                config.input_size,
                config.input_size,
            ),
            torch.zeros(
                1,
                config.candidate_context_features,
                dtype=torch.float32,
            ),
        ),
        onnx_path,
        input_names=[
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        output_names=output_names,
        dynamic_axes={
            "element_crop_evidence": {0: "batch"},
            "whole_sheet_evidence": {0: "batch"},
            "candidate_context": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "artifact": onnx_path.name,
        "artifact_sha256": sha256_file(onnx_path),
        "model_version": config.model_version,
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "input_name": "element_crop_evidence",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_size": config.input_size,
        "input_channels": config.input_channels,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "whole_sheet_input_channels": config.whole_sheet_input_channels,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": config.candidate_context_features,
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "output_names": output_names,
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "family_classes": list(ELEMENT_FAMILY_CLASSES),
        "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
        "family_contract": ELEMENT_FAMILY_CONTRACT,
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
    }
    manifest["content_sha256"] = sha256_json(manifest)
    onnx_manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    onnx_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report: dict[str, Any] = {
        "schema_version": "dajoong.local-element-pretraining.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "model_config": config.to_dict(),
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "train_options": asdict(options),
        **initialization,
        "parameter_count": model.parameter_count(),
        "train_items": len(train_data),
        "validation_items": len(validation_data),
        "class_counts": dict(zip(ELEMENT_PROGRAM_CLASSES, counts, strict=True)),
        "class_weights": dict(
            zip(ELEMENT_PROGRAM_CLASSES, class_weights, strict=True)
        ),
        "best_validation_loss": best_validation,
        "best_validation_foreground_macro_f1": best_validation_f1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
        "duration_seconds": time.time() - started,
        "history": history,
    }
    report["content_sha256"] = sha256_json(report)
    assert_synthetic_pretraining_only(report)
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def calibrate_local_element_student_from_direct_research(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    initial_checkpoint: str | Path,
    options: LocalElementTrainOptions | None = None,
) -> dict[str, Any]:
    """Fit a synthetic-pretrained student to direct real-sheet annotations.

    This path is deliberately not an evaluator and never produces a production
    artifact. Every source used here is written into the checkpoint exclusion
    ledger so the same sheet cannot later be reported as held-out evidence.
    """

    if torch is None or DataLoader is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    options = options or LocalElementTrainOptions(
        epochs=8,
        batch_size=128,
        learning_rate=1e-4,
        validation_fraction=0.1,
    )
    options.validate()
    root = Path(corpus_root).expanduser().resolve()
    corpus_manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    expected_role = "direct_real_research_calibration_only"
    if corpus_manifest.get("role") != expected_role:
        raise ValueError("direct research calibrator requires a research-only corpus")
    if corpus_manifest.get("candidate_supervision_contract") != (
        DIRECT_CANDIDATE_SUPERVISION_CONTRACT
    ):
        raise ValueError("direct candidate supervision contract mismatch")
    exclusions = sorted(
        {str(value) for value in corpus_manifest["evaluation_exclusion_source_sha256"]}
    )
    if not exclusions:
        raise ValueError("direct research calibration requires source exclusions")

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = SyntheticLocalElementDataset(
        root,
        split="all",
        validation_fraction=options.validation_fraction,
        corpus_role=expected_role,
        quadrant_augmentation=options.quadrant_augmentation,
        augmentation_seed=options.seed,
        semantic_context_dropout=options.semantic_context_dropout,
    )
    sampler = None
    if options.balanced_sampling:
        if WeightedRandomSampler is None:
            raise RuntimeError("balanced local sampling requires torch")
        sampler = WeightedRandomSampler(
            data.sampling_weights(),
            num_samples=len(data),
            replacement=True,
            generator=torch.Generator().manual_seed(options.seed),
        )
    loader = DataLoader(
        data,
        batch_size=options.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        generator=torch.Generator().manual_seed(options.seed),
    )
    fit_loader = DataLoader(
        data,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
    )
    counts = data.class_counts()
    class_weights = _balanced_class_weights(
        counts,
        ELEMENT_PROGRAM_CLASSES,
        background_weight=options.background_weight,
    )
    base_config = LocalElementStudentConfig()
    config = LocalElementStudentConfig(
        model_version=f"{base_config.model_version}-direct-research-calibration-v1"
    )
    model = DajoongLocalElementStudent(config).to(device)
    initial_path = Path(initial_checkpoint).expanduser().resolve()
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial_payload.get("role") != "synthetic_pretrain_only":
        raise ValueError("research calibration must start from synthetic pretraining")
    required_contracts = {
        "oriented_evidence_rotation_contract": ORIENTED_EVIDENCE_ROTATION_CONTRACT,
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
    }
    for name, expected in required_contracts.items():
        if initial_payload.get(name) != expected:
            raise ValueError(f"initial checkpoint {name} mismatch")
    transfer = _warm_start_local_model(model, initial_payload, config)
    criterion = LocalElementCriterion(class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    history = []
    started = time.time()
    for epoch in range(options.epochs):
        metrics = _run_local_epoch(
            model,
            loader,
            criterion,
            device=device,
            optimizer=optimizer,
            progress_label=f"direct calibration {epoch + 1}/{options.epochs}",
        )
        history.append({"epoch": epoch + 1, "calibration_train": metrics})
    with torch.inference_mode():
        fit_metrics = _run_local_epoch(
            model,
            fit_loader,
            criterion,
            device=device,
            optimizer=None,
        )

    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "calibrated-research-only.pt"
    checkpoint_payload = {
        "schema_version": "dajoong.local-element-checkpoint.v1",
        "role": expected_role,
        "real_drawing_ground_truth": True,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "selection_scope": "in_sample_research_calibration_only_not_evaluation",
        "training_source_sha256s": exclusions,
        "evaluation_exclusion_source_sha256": exclusions,
        "oriented_evidence_rotation_contract": ORIENTED_EVIDENCE_ROTATION_CONTRACT,
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_supervision_contract": DIRECT_CANDIDATE_SUPERVISION_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "epoch": options.epochs,
        "calibration_fit_loss": fit_metrics["total"],
        "calibration_fit_foreground_macro_f1": fit_metrics[
            "foreground_macro_f1"
        ],
    }
    torch.save(checkpoint_payload, checkpoint_path)

    model = model.cpu().eval()
    onnx_path = destination / "local-element-student.onnx"
    output_names = [
        "class_logits",
        "family_logits",
        "objectness",
        "geometry",
        "uncertainty",
    ]
    torch.onnx.export(
        LocalElementStudentOnnxAdapter(model),
        (
            torch.zeros(1, config.input_channels, config.input_size, config.input_size),
            torch.zeros(
                1,
                config.whole_sheet_input_channels,
                config.input_size,
                config.input_size,
            ),
            torch.zeros(1, config.candidate_context_features, dtype=torch.float32),
        ),
        onnx_path,
        input_names=[
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        output_names=output_names,
        dynamic_axes={
            "element_crop_evidence": {0: "batch"},
            "whole_sheet_evidence": {0: "batch"},
            "candidate_context": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "role": expected_role,
        "real_drawing_ground_truth": True,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "selection_scope": "in_sample_research_calibration_only_not_evaluation",
        "training_source_sha256s": exclusions,
        "evaluation_exclusion_source_sha256": exclusions,
        "artifact": onnx_path.name,
        "artifact_sha256": sha256_file(onnx_path),
        "model_version": config.model_version,
        "oriented_evidence_rotation_contract": ORIENTED_EVIDENCE_ROTATION_CONTRACT,
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_supervision_contract": DIRECT_CANDIDATE_SUPERVISION_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "input_name": "element_crop_evidence",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_size": config.input_size,
        "input_channels": config.input_channels,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "whole_sheet_input_channels": config.whole_sheet_input_channels,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": config.candidate_context_features,
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "output_names": output_names,
        "objectness_contract": (
            "binary_object_existence_before_conditional_taxonomy_v1"
        ),
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "family_classes": list(ELEMENT_FAMILY_CLASSES),
        "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
        "family_contract": ELEMENT_FAMILY_CONTRACT,
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": options.epochs,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    onnx_path.with_suffix(onnx_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report: dict[str, Any] = {
        "schema_version": "dajoong.direct-local-element-calibration.v1",
        "role": expected_role,
        "real_drawing_ground_truth": True,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "selection_scope": "in_sample_research_calibration_only_not_evaluation",
        "training_source_sha256s": exclusions,
        "evaluation_exclusion_source_sha256": exclusions,
        "source_corpus_sha256": sha256_file(root / "manifest.json"),
        "initial_checkpoint_sha256": sha256_file(initial_path),
        **transfer,
        "train_options": asdict(options),
        "parameter_count": model.parameter_count(),
        "calibration_items": len(data),
        "class_counts": dict(zip(ELEMENT_PROGRAM_CLASSES, counts, strict=True)),
        "calibration_fit_metrics_not_evaluation": fit_metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
        "duration_seconds": time.time() - started,
        "history": history,
    }
    report["content_sha256"] = sha256_json(report)
    (destination / "calibration-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def export_synthetic_local_element_checkpoint(
    checkpoint_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Export a completed local checkpoint without retraining or relabeling it."""

    if torch is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("role") != "synthetic_pretrain_only":
        raise ValueError("only synthetic pretraining checkpoints may use this exporter")
    if payload.get("oriented_evidence_rotation_contract") != (
        ORIENTED_EVIDENCE_ROTATION_CONTRACT
    ):
        raise ValueError("checkpoint oriented-evidence rotation contract mismatch")
    if payload.get("candidate_alignment_contract") != CANDIDATE_ALIGNMENT_CONTRACT:
        raise ValueError("checkpoint candidate alignment contract mismatch")
    if payload.get("candidate_hypothesis_context_contract") != (
        CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
    ):
        raise ValueError("checkpoint proposal graph context mismatch")
    if payload.get("perception_authority_contract") != (
        LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
    ):
        raise ValueError("checkpoint dual-authority contract mismatch")
    config = LocalElementStudentConfig()
    if payload.get("config") != config.to_dict():
        raise ValueError("local element checkpoint contract mismatch")
    model = DajoongLocalElementStudent(config).cpu().eval()
    model.load_state_dict(payload["state_dict"])
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    onnx_path = destination / "local-element-student.onnx"
    output_names = [
        "class_logits",
        "family_logits",
        "objectness",
        "geometry",
        "uncertainty",
    ]
    torch.onnx.export(
        LocalElementStudentOnnxAdapter(model),
        (
            torch.zeros(1, config.input_channels, config.input_size, config.input_size),
            torch.zeros(
                1,
                config.whole_sheet_input_channels,
                config.input_size,
                config.input_size,
            ),
            torch.zeros(
                1,
                config.candidate_context_features,
                dtype=torch.float32,
            ),
        ),
        onnx_path,
        input_names=[
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        output_names=output_names,
        dynamic_axes={
            "element_crop_evidence": {0: "batch"},
            "whole_sheet_evidence": {0: "batch"},
            "candidate_context": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "artifact": onnx_path.name,
        "artifact_sha256": sha256_file(onnx_path),
        "model_version": config.model_version,
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "candidate_alignment_contract": CANDIDATE_ALIGNMENT_CONTRACT,
        "candidate_hypothesis_context_contract": (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        ),
        "perception_authority_contract": LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
        "input_name": "element_crop_evidence",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_size": config.input_size,
        "input_channels": config.input_channels,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "whole_sheet_input_channels": config.whole_sheet_input_channels,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": config.candidate_context_features,
        "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "output_names": output_names,
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "family_classes": list(ELEMENT_FAMILY_CLASSES),
        "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
        "family_contract": ELEMENT_FAMILY_CONTRACT,
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_validation_foreground_macro_f1": float(
            payload.get("validation_foreground_macro_f1") or 0.0
        ),
        "promotion_requirements": [
            "commercial-rights direct whole-sheet visual labels",
            "fixed untouched real-drawing qualification holdout",
            "immutable paired global/local artifact registration",
        ],
    }
    manifest["content_sha256"] = sha256_json(manifest)
    manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def evaluate_local_element_checkpoint(
    checkpoint_path: str | Path,
    corpus_root: str | Path,
    *,
    batch_size: int = 256,
    validation_fraction: float = 0.1,
    device: str = "cpu",
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("role") != "synthetic_pretrain_only":
        raise ValueError("local element evaluator accepts synthetic checkpoints only")
    if payload.get("oriented_evidence_rotation_contract") != (
        ORIENTED_EVIDENCE_ROTATION_CONTRACT
    ):
        raise ValueError("checkpoint oriented-evidence rotation contract mismatch")
    if payload.get("candidate_alignment_contract") != CANDIDATE_ALIGNMENT_CONTRACT:
        raise ValueError("checkpoint candidate alignment contract mismatch")
    if payload.get("candidate_hypothesis_context_contract") != (
        CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
    ):
        raise ValueError("checkpoint proposal graph context mismatch")
    if payload.get("perception_authority_contract") != (
        LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
    ):
        raise ValueError("checkpoint dual-authority contract mismatch")
    config = LocalElementStudentConfig()
    if payload.get("config") != config.to_dict():
        raise ValueError("local element checkpoint contract mismatch")
    dataset = SyntheticLocalElementDataset(
        corpus_root,
        split="validation",
        validation_fraction=validation_fraction,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = DajoongLocalElementStudent(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    confusion = torch.zeros(
        (len(ELEMENT_PROGRAM_CLASSES), len(ELEMENT_PROGRAM_CLASSES)),
        dtype=torch.int64,
    )
    geometry_error = 0.0
    geometry_values = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["evidence"].to(device),
                batch["whole_sheet_evidence"].to(device),
                batch["candidate_context"].to(device),
            )
            conditional_prediction = _joint_foreground_prediction(output)
            prediction = torch.where(
                (output["objectness"][:, 0] >= 0.5)
                & (_joint_foreground_confidence(output) >= 0.25),
                conditional_prediction,
                torch.zeros_like(conditional_prediction),
            ).cpu()
            target = batch["label"]
            count = len(ELEMENT_PROGRAM_CLASSES)
            confusion += torch.bincount(
                target * count + prediction,
                minlength=count * count,
            ).reshape(count, count)
            valid = batch["geometry_valid"].unsqueeze(1).to(device)
            geometry_error += float(
                ((output["geometry"] - batch["geometry"].to(device)).abs() * valid)
                .sum()
                .cpu()
            )
            geometry_values += int(valid.sum().cpu()) * len(ELEMENT_GEOMETRY_CHANNELS)
    rows, foreground_macro_f1, foreground_micro_f1 = (
        _local_classification_metrics(confusion)
    )
    report: dict[str, Any] = {
        "schema_version": "dajoong.synthetic-local-element-evaluation.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_items": len(dataset),
        "class_metrics": rows,
        "foreground_macro_f1": foreground_macro_f1,
        "foreground_micro_f1": foreground_micro_f1,
        "confusion_matrix": confusion.tolist(),
        "geometry_mae": geometry_error / max(1, geometry_values),
        "evaluation_duration_seconds": time.perf_counter() - started,
    }
    report["content_sha256"] = sha256_json(report)
    assert_synthetic_pretraining_only(report)
    return report
