"""Evaluation of the exported production pipeline against direct visual ground truth.

This module intentionally evaluates the final metric plan graph, not a local neural
head.  Public drawings remain evaluation-only and every report records that it is
not a production accuracy claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .core.hashing import sha256_file
from .core.model.global_topology_student import ELEMENT_PROGRAM_CLASSES
from .ground_truth import (
    assert_benchmark_graph_geometry,
    assert_evaluation_content_profile,
    assert_manifest_graph_correspondence,
    validate_ground_truth_manifest,
)


def _safe_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _binary_scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    true_positive = int(np.logical_and(prediction, target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    union = true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "iou": true_positive / max(1, union),
    }


def _draw_lines(
    size: tuple[int, int],
    entities: Iterable[dict[str, Any]],
    *,
    coordinate_scale: float,
    width: int | None = None,
    use_entity_width: bool = False,
) -> np.ndarray:
    canvas = Image.new("1", size, 0)
    draw = ImageDraw.Draw(canvas)
    for entity in entities:
        start = entity.get("from")
        end = entity.get("to")
        if not start or not end:
            continue
        line_width = width or 1
        if use_entity_width:
            line_width = max(1, round(float(entity.get("thickness_m") or 0.12) * coordinate_scale))
        draw.line(
            (
                float(start[0]) * coordinate_scale,
                float(start[1]) * coordinate_scale,
                float(end[0]) * coordinate_scale,
                float(end[1]) * coordinate_scale,
            ),
            fill=1,
            width=line_width,
        )
    return np.asarray(canvas, dtype=np.bool_)


def _draw_polygons(
    size: tuple[int, int],
    entities: Iterable[dict[str, Any]],
    *,
    coordinate_scale: float = 1.0,
) -> np.ndarray:
    canvas = Image.new("1", size, 0)
    draw = ImageDraw.Draw(canvas)
    for entity in entities:
        polygon = entity.get("polygon")
        if not polygon or len(polygon) < 3:
            continue
        draw.polygon(
            [(float(x) * coordinate_scale, float(y) * coordinate_scale) for x, y in polygon],
            fill=1,
        )
    return np.asarray(canvas, dtype=np.bool_)


def _tolerant_line_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance_px: float,
) -> dict[str, float | int]:
    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    if not prediction_count or not target_count:
        precision = 0.0 if prediction_count else float(not target_count)
        recall = 0.0 if target_count else float(not prediction_count)
        return {
            "prediction_pixels": prediction_count,
            "target_pixels": target_count,
            "precision": precision,
            "recall": recall,
            "f1": _safe_f1(precision, recall),
        }
    distance_to_target = ndimage.distance_transform_edt(~target)
    distance_to_prediction = ndimage.distance_transform_edt(~prediction)
    precision = float((distance_to_target[prediction] <= tolerance_px).mean())
    recall = float((distance_to_prediction[target] <= tolerance_px).mean())
    return {
        "prediction_pixels": prediction_count,
        "target_pixels": target_count,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
    }


def _center(entity: dict[str, Any], *, coordinate_scale: float) -> tuple[float, float]:
    center = entity.get("center_m") or entity.get("center_px")
    if center:
        return float(center[0]) * coordinate_scale, float(center[1]) * coordinate_scale
    polygon = entity.get("polygon") or []
    if polygon:
        return (
            float(np.mean([point[0] for point in polygon])) * coordinate_scale,
            float(np.mean([point[1] for point in polygon])) * coordinate_scale,
        )
    raise ValueError(f"entity {entity.get('id', '<unknown>')} has no center evidence")


def _entity_type(entity: dict[str, Any], *, kind: str) -> str:
    if kind == "opening":
        value = entity.get("type") or entity.get("opening_type") or "unknown"
    else:
        value = (
            entity.get("fixture_type")
            or entity.get("symbol_class")
            or entity.get("type")
            or entity.get("family_id")
            or "unknown"
        )
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if kind == "opening":
        return normalized
    # Compiler family IDs carry a display namespace that the direct annotation
    # taxonomy does not.  Strip it before compact spelling normalization so
    # ``residential-water-tap`` and ``water_tap`` are one measured class.
    for prefix in ("residential-", "generic-"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    compact = normalized.replace("-", "")
    if "electricalappliance" in normalized or "electrical-appliance" in normalized:
        subtype = {
            "refrigerator": "refrigerator",
            "integratedstove": "stove",
            "stove": "stove",
            "dishwasher": "dishwasher",
            "washingmachine": "washingmachine",
            "tumbledryer": "tumbledryer",
        }
        for token, canonical in subtype.items():
            if token in compact:
                return canonical
        return "appliance"
    if "sink" in normalized:
        return "sink"
    if "toilet" in normalized or normalized == "wc":
        return "toilet"
    if "saunabench" in normalized or normalized.endswith("-bench"):
        return "bench"
    if "closet" in normalized or "cabinet" in normalized or "wardrobe" in normalized:
        return "storage"
    if "shower" in normalized:
        return "shower"
    if "column" in normalized:
        return "column"
    if "stair" in normalized:
        return "stairs"
    if compact == "coatrack":
        return "coatrack"
    if compact == "watertap":
        return "watertap"
    if compact == "jacuzzi":
        return "jacuzzi"
    if compact == "woodstove":
        return "woodstove"
    if compact == "fireplacecorner":
        return "fireplacecorner"
    if compact == "placeforfireplacecorner":
        return "placeforfireplacecorner"
    if compact == "placeforfireplace":
        return "placeforfireplace"
    if compact in {"housing", "misc"}:
        return compact
    furniture = {
        "bed": "bed",
        "sofa": "sofa",
        "armchair": "armchair",
        "chair": "chair",
        "diningtable": "diningtable",
        "coffeetable": "coffeetable",
        "desk": "desk",
        "bench": "bench",
        "refrigerator": "refrigerator",
        "stove": "stove",
        "dishwasher": "dishwasher",
        "washingmachine": "washingmachine",
        "tumbledryer": "tumbledryer",
    }
    if compact in furniture:
        return furniture[compact]
    return normalized


def _type_compatible(left: str, right: str, *, kind: str) -> bool:
    if left == right:
        return True
    if kind == "opening":
        return {left, right} <= {"door", "window", "opening"} and "opening" in {left, right}
    return left == right


def _entity_plan_extents(
    entity: dict[str, Any],
    *,
    coordinate_scale: float,
    kind: str,
) -> tuple[float, ...] | None:
    """Return comparable plan dimensions without trusting a display mesh."""

    polygon = entity.get("polygon") or []
    if polygon:
        xs = [float(point[0]) * coordinate_scale for point in polygon]
        ys = [float(point[1]) * coordinate_scale for point in polygon]
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        if kind == "opening":
            return (max(width, depth),)
        return tuple(sorted((width, depth), reverse=True))
    bbox_px = entity.get("bbox_px")
    if bbox_px and len(bbox_px) >= 4:
        width = (float(bbox_px[2]) - float(bbox_px[0])) * coordinate_scale
        depth = (float(bbox_px[3]) - float(bbox_px[1])) * coordinate_scale
        if kind == "opening":
            return (max(width, depth),)
        return tuple(sorted((width, depth), reverse=True))
    if kind == "opening" and entity.get("width_m") is not None:
        return (float(entity["width_m"]) * coordinate_scale,)
    size = entity.get("size_m") or entity.get("geometry_scale_xyz")
    if size and len(size) >= 2:
        return tuple(
            sorted(
                (
                    float(size[0]) * coordinate_scale,
                    float(size[1]) * coordinate_scale,
                ),
                reverse=True,
            )
        )
    return None


def _dimension_similarity(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    similarities = [
        min(left_value, right_value) / max(left_value, right_value, 1e-9)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    return float(min(similarities))


def _entity_scores(
    prediction: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    prediction_scale: float,
    kind: str,
    maximum_distance_px: float,
    minimum_dimension_similarity: float | None = None,
) -> dict[str, Any]:
    if not prediction or not target:
        matches = 0
        precision = matches / max(1, len(prediction))
        recall = matches / max(1, len(target))
        target_counts: dict[str, int] = {}
        prediction_counts: dict[str, int] = {}
        for entity in target:
            name = _entity_type(entity, kind=kind)
            target_counts[name] = target_counts.get(name, 0) + 1
        for entity in prediction:
            name = _entity_type(entity, kind=kind)
            prediction_counts[name] = prediction_counts.get(name, 0) + 1
        per_class = {}
        for name in sorted(set(target_counts) | set(prediction_counts)):
            precision = 0.0 if prediction_counts.get(name, 0) else float(
                not target_counts.get(name, 0)
            )
            recall = 0.0 if target_counts.get(name, 0) else float(
                not prediction_counts.get(name, 0)
            )
            per_class[name] = {
                "matches": 0,
                "prediction_count": prediction_counts.get(name, 0),
                "target_count": target_counts.get(name, 0),
                "precision": precision,
                "recall": recall,
                "f1": _safe_f1(precision, recall),
            }
        return {
            "matches": matches,
            "prediction_count": len(prediction),
            "target_count": len(target),
            "precision": precision,
            "recall": recall,
            "f1": _safe_f1(precision, recall),
            "matched_pairs": [],
            "per_class": per_class,
        }
    cost = np.full((len(target), len(prediction)), 1e9, dtype=np.float64)
    for target_index, target_entity in enumerate(target):
        target_center = _center(target_entity, coordinate_scale=1.0)
        target_type = _entity_type(target_entity, kind=kind)
        for prediction_index, prediction_entity in enumerate(prediction):
            prediction_center = _center(prediction_entity, coordinate_scale=prediction_scale)
            prediction_type = _entity_type(prediction_entity, kind=kind)
            if not _type_compatible(target_type, prediction_type, kind=kind):
                continue
            distance = math.dist(target_center, prediction_center)
            target_extent = _entity_plan_extents(
                target_entity,
                coordinate_scale=1.0,
                kind=kind,
            )
            prediction_extent = _entity_plan_extents(
                prediction_entity,
                coordinate_scale=prediction_scale,
                kind=kind,
            )
            dimension_similarity = _dimension_similarity(
                target_extent,
                prediction_extent,
            )
            if (
                minimum_dimension_similarity is not None
                and (
                    dimension_similarity is None
                    or dimension_similarity < minimum_dimension_similarity
                )
            ):
                continue
            if distance <= maximum_distance_px:
                size_penalty = (
                    0.0
                    if dimension_similarity is None
                    else (1.0 - dimension_similarity) * maximum_distance_px
                )
                cost[target_index, prediction_index] = distance + size_penalty
    rows, columns = linear_sum_assignment(cost)
    pairs = [
        {
            "target_id": target[row].get("id"),
            "prediction_id": prediction[column].get("id"),
            "distance_px": math.dist(
                _center(target[row], coordinate_scale=1.0),
                _center(prediction[column], coordinate_scale=prediction_scale),
            ),
            "dimension_similarity": _dimension_similarity(
                _entity_plan_extents(
                    target[row],
                    coordinate_scale=1.0,
                    kind=kind,
                ),
                _entity_plan_extents(
                    prediction[column],
                    coordinate_scale=prediction_scale,
                    kind=kind,
                ),
            ),
            "class_name": _entity_type(target[row], kind=kind),
        }
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] < 1e8
    ]
    matches = len(pairs)
    precision = matches / max(1, len(prediction))
    recall = matches / max(1, len(target))
    target_counts: dict[str, int] = {}
    prediction_counts: dict[str, int] = {}
    matched_counts: dict[str, int] = {}
    for entity in target:
        name = _entity_type(entity, kind=kind)
        target_counts[name] = target_counts.get(name, 0) + 1
    for entity in prediction:
        name = _entity_type(entity, kind=kind)
        prediction_counts[name] = prediction_counts.get(name, 0) + 1
    for pair in pairs:
        name = str(pair["class_name"])
        matched_counts[name] = matched_counts.get(name, 0) + 1
    per_class = {}
    for name in sorted(set(target_counts) | set(prediction_counts)):
        class_matches = matched_counts.get(name, 0)
        class_precision = class_matches / max(1, prediction_counts.get(name, 0))
        class_recall = class_matches / max(1, target_counts.get(name, 0))
        per_class[name] = {
            "matches": class_matches,
            "prediction_count": prediction_counts.get(name, 0),
            "target_count": target_counts.get(name, 0),
            "precision": class_precision,
            "recall": class_recall,
            "f1": _safe_f1(class_precision, class_recall),
        }
    return {
        "matches": matches,
        "prediction_count": len(prediction),
        "target_count": len(target),
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "matched_pairs": pairs,
        "per_class": per_class,
    }


def _room_type(entity: dict[str, Any]) -> str:
    value = (
        entity.get("room_class")
        or entity.get("room_type")
        or entity.get("occupancy")
        or entity.get("name")
        or "unknown"
    )
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    # Product room names append a display ordinal (for example ``bathroom 2``).
    # The evaluator must compare the semantic occupancy, not treat each display
    # instance as a separate class.
    if normalized.rsplit("-", 1)[-1].isdigit():
        normalized = normalized.rsplit("-", 1)[0]
    aliases = {
        "livingroom": "living",
        "living-room": "living",
        "entry-lobby": "hallway",
        "entry": "hallway",
        "corridor": "hallway",
        "technicalroom": "mechanical",
        "technical-room": "mechanical",
        "utility-laundry": "utility",
        "laundry": "utility",
        "closet-walkin": "storage",
        "walk-in-closet": "storage",
        "terrace": "outdoor",
        "balcony": "outdoor",
    }
    compact = normalized.replace("-", "")
    return aliases.get(normalized, aliases.get(compact, normalized))


def _room_class_scores(
    size: tuple[int, int],
    prediction: list[dict[str, Any]],
    target: list[dict[str, Any]],
    *,
    prediction_scale: float,
) -> dict[str, dict[str, float | int]]:
    output = {}
    classes = sorted(
        {_room_type(item) for item in prediction} | {_room_type(item) for item in target}
    )
    for class_name in classes:
        prediction_mask = _draw_polygons(
            size,
            [item for item in prediction if _room_type(item) == class_name],
            coordinate_scale=prediction_scale,
        )
        target_mask = _draw_polygons(
            size,
            [item for item in target if _room_type(item) == class_name],
        )
        output[class_name] = _binary_scores(prediction_mask, target_mask)
    return output


def _aggregate_entity_classes(
    sheets: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for sheet in sheets:
        for class_name, score in sheet[metric_name]["per_class"].items():
            row = totals.setdefault(
                class_name,
                {"matches": 0, "prediction_count": 0, "target_count": 0},
            )
            for field in row:
                row[field] += int(score[field])
    output = {}
    for class_name, row in totals.items():
        precision = row["matches"] / max(1, row["prediction_count"])
        recall = row["matches"] / max(1, row["target_count"])
        output[class_name] = {
            **row,
            "precision": precision,
            "recall": recall,
            "f1": _safe_f1(precision, recall),
        }
    return output


def _aggregate_binary_classes(
    sheets: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for sheet in sheets:
        for class_name, score in sheet[metric_name].items():
            row = totals.setdefault(
                class_name,
                {"true_positive": 0, "false_positive": 0, "false_negative": 0},
            )
            for field in row:
                row[field] += int(score[field])
    output = {}
    for class_name, row in totals.items():
        precision = row["true_positive"] / max(
            1, row["true_positive"] + row["false_positive"]
        )
        recall = row["true_positive"] / max(
            1, row["true_positive"] + row["false_negative"]
        )
        output[class_name] = {
            **row,
            "precision": precision,
            "recall": recall,
            "f1": _safe_f1(precision, recall),
        }
    return output


def _wall_overlay(
    source_path: Path,
    target_centerline: np.ndarray,
    prediction_centerline: np.ndarray,
    output_path: Path,
) -> None:
    source = Image.open(source_path).convert("RGB")
    image = np.asarray(source, dtype=np.uint8).copy()
    target_near_prediction = ndimage.distance_transform_edt(~prediction_centerline) <= 8
    prediction_near_target = ndimage.distance_transform_edt(~target_centerline) <= 8
    target_missing = target_centerline & ~target_near_prediction
    prediction_extra = prediction_centerline & ~prediction_near_target
    matched = (target_centerline & target_near_prediction) | (
        prediction_centerline & prediction_near_target
    )
    for mask, color in (
        (ndimage.binary_dilation(target_missing, iterations=2), (220, 56, 54)),
        (ndimage.binary_dilation(prediction_extra, iterations=2), (36, 128, 89)),
        (ndimage.binary_dilation(matched, iterations=1), (214, 165, 44)),
    ):
        image[mask] = np.asarray(color, dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def evaluate_exported_pipeline(
    *,
    ground_truth_root: str | Path,
    prediction_root: str | Path,
    output_root: str | Path,
    pixels_per_meter: float,
    wall_tolerances_px: tuple[int, ...] = (4, 8, 16),
    required_content_profile: str = "full_editable_bim",
) -> dict[str, Any]:
    ground_truth_root = Path(ground_truth_root).resolve()
    prediction_root = Path(prediction_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    evaluated_source_hashes: set[str] = set()
    for sheet_dir in sorted(path for path in ground_truth_root.iterdir() if path.is_dir()):
        manifest_path = sheet_dir / "manifest.json"
        target_path = sheet_dir / "benchmark-graph.json"
        prediction_path = prediction_root / sheet_dir.name / "03-plan-graph.json"
        if (
            not manifest_path.is_file()
            or not target_path.is_file()
            or not prediction_path.is_file()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = json.loads(target_path.read_text(encoding="utf-8"))
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        source_path = Path(manifest["source"]["image_path"])
        validate_ground_truth_manifest(manifest)
        assert_manifest_graph_correspondence(manifest, target)
        assert_evaluation_content_profile(
            manifest,
            required_profile=required_content_profile,
        )
        expected_hash = manifest["source"]["image_sha256"]
        if expected_hash in evaluated_source_hashes:
            raise ValueError(
                "the same reviewed source is present more than once under different "
                f"sample names: {expected_hash}"
            )
        evaluated_source_hashes.add(expected_hash)
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"source hash mismatch for {sheet_dir.name}")
        assert_prediction_source_correspondence(
            prediction,
            expected_source_sha256=expected_hash,
            sheet_id=sheet_dir.name,
        )
        assert_prediction_scale_correspondence(
            prediction,
            expected_pixels_per_meter=pixels_per_meter,
            sheet_id=sheet_dir.name,
        )
        assert_prediction_not_trained_on_source(
            prediction,
            expected_source_sha256=expected_hash,
            sheet_id=sheet_dir.name,
        )
        if not manifest["visual_review"].get("whole_sheet_reviewed"):
            raise ValueError(f"ground truth is not whole-sheet reviewed: {sheet_dir.name}")
        if not manifest["omission_scan"].get("completed"):
            raise ValueError(f"ground truth omission scan is incomplete: {sheet_dir.name}")
        size = (int(manifest["source"]["width_px"]), int(manifest["source"]["height_px"]))
        assert_benchmark_graph_geometry(
            target,
            image_size=size,
            reviewed_plan_bbox_px=manifest["source"].get("reviewed_plan_bbox_px"),
        )
        target_centerline = _draw_lines(size, target.get("walls", []), coordinate_scale=1.0)
        prediction_centerline = _draw_lines(
            size,
            prediction.get("walls", []),
            coordinate_scale=pixels_per_meter,
        )
        target_footprint = _draw_polygons(size, target.get("walls", []))
        prediction_footprint = _draw_lines(
            size,
            prediction.get("walls", []),
            coordinate_scale=pixels_per_meter,
            use_entity_width=True,
        )
        target_rooms = _draw_polygons(size, target.get("rooms", []))
        prediction_rooms = _draw_polygons(
            size,
            prediction.get("rooms", []),
            coordinate_scale=pixels_per_meter,
        )
        wall_centerline = {
            str(tolerance): _tolerant_line_scores(
                prediction_centerline,
                target_centerline,
                tolerance,
            )
            for tolerance in wall_tolerances_px
        }
        overlay_path = output_root / sheet_dir.name / "wall-diagnostic.png"
        _wall_overlay(source_path, target_centerline, prediction_centerline, overlay_path)
        sheet = {
            "sheet_id": sheet_dir.name,
            "source_sha256": expected_hash,
            "prediction_graph_sha256": sha256_file(prediction_path),
            "evaluation_license_scope": manifest["source"].get("license_scope"),
            "wall_centerline_by_tolerance_px": wall_centerline,
            "wall_footprint": _binary_scores(prediction_footprint, target_footprint),
            "room_union": _binary_scores(prediction_rooms, target_rooms),
            "room_classes": _room_class_scores(
                size,
                prediction.get("rooms", []),
                target.get("rooms", []),
                prediction_scale=pixels_per_meter,
            ),
            "openings": _entity_scores(
                prediction.get("openings", []),
                target.get("openings", []),
                prediction_scale=pixels_per_meter,
                kind="opening",
                maximum_distance_px=24.0,
            ),
            "fixtures": _entity_scores(
                prediction.get("fixtures", []),
                target.get("fixtures", []),
                prediction_scale=pixels_per_meter,
                kind="fixture",
                maximum_distance_px=32.0,
            ),
            "openings_geometry_aware": _entity_scores(
                prediction.get("openings", []),
                target.get("openings", []),
                prediction_scale=pixels_per_meter,
                kind="opening",
                maximum_distance_px=24.0,
                minimum_dimension_similarity=0.75,
            ),
            "fixtures_geometry_aware": _entity_scores(
                prediction.get("fixtures", []),
                target.get("fixtures", []),
                prediction_scale=pixels_per_meter,
                kind="fixture",
                maximum_distance_px=32.0,
                minimum_dimension_similarity=0.75,
            ),
            "wall_count": {
                "prediction": len(prediction.get("walls", [])),
                "target": len(target.get("walls", [])),
            },
            "diagnostic_overlay": str(overlay_path),
        }
        sheets.append(sheet)
    if not sheets:
        raise ValueError("no matching evaluated sheets")

    def macro(path: tuple[str, ...]) -> float:
        values = []
        for sheet in sheets:
            value: Any = sheet
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    room_class_scores = _aggregate_binary_classes(sheets, "room_classes")
    opening_class_scores = _aggregate_entity_classes(
        sheets, "openings_geometry_aware"
    )
    fixture_class_scores = _aggregate_entity_classes(
        sheets, "fixtures_geometry_aware"
    )
    expected_fixture_classes = sorted(
        {
            _entity_type({"fixture_type": name}, kind="fixture")
            for name in ELEMENT_PROGRAM_CLASSES
            if name not in {"background", "door", "window", "unknown"}
        }
    )
    observed_fixture_targets = sorted(
        name
        for name, value in fixture_class_scores.items()
        if value["target_count"] > 0
    )
    aggregate = {
        "sheet_count": len(sheets),
        "wall_centerline_macro_f1": {
            str(tolerance): macro(("wall_centerline_by_tolerance_px", str(tolerance), "f1"))
            for tolerance in wall_tolerances_px
        },
        "wall_footprint_macro_f1": macro(("wall_footprint", "f1")),
        "wall_footprint_macro_iou": macro(("wall_footprint", "iou")),
        "room_union_macro_f1": macro(("room_union", "f1")),
        "opening_entity_macro_f1": macro(("openings", "f1")),
        "fixture_entity_macro_f1": macro(("fixtures", "f1")),
        "opening_geometry_aware_macro_f1": macro(
            ("openings_geometry_aware", "f1")
        ),
        "fixture_geometry_aware_macro_f1": macro(
            ("fixtures_geometry_aware", "f1")
        ),
        "room_class_scores": room_class_scores,
        "opening_class_scores": opening_class_scores,
        "fixture_class_scores": fixture_class_scores,
        "minimum_observed_room_class_f1": min(
            (float(value["f1"]) for value in room_class_scores.values()),
            default=0.0,
        ),
        "minimum_observed_opening_class_f1": min(
            (float(value["f1"]) for value in opening_class_scores.values()),
            default=0.0,
        ),
        "minimum_observed_fixture_class_f1": min(
            (float(value["f1"]) for value in fixture_class_scores.values()),
            default=0.0,
        ),
        "fixture_taxonomy_target_coverage": {
            "expected_classes": expected_fixture_classes,
            "observed_target_classes": observed_fixture_targets,
            "missing_target_classes": sorted(
                set(expected_fixture_classes) - set(observed_fixture_targets)
            ),
            "coverage": len(observed_fixture_targets)
            / max(1, len(expected_fixture_classes)),
        },
    }
    report = {
        "schema_version": "dajoong.exported-pipeline-evaluation.v1",
        "evaluation_target": "final_exported_metric_plan_graph",
        "ground_truth_policy": "direct_visual_source_annotation_only",
        "content_profile": required_content_profile,
        "production_accuracy_claim": False,
        "reason_not_production_claim": (
            "The public CubiCasa drawings are a small non-commercial evaluation-only corpus."
        ),
        "pixels_per_meter": pixels_per_meter,
        "fixed_wall_tolerances_px": list(wall_tolerances_px),
        "aggregate": aggregate,
        "sheets": sheets,
        "ground_truth_integrity": {
            "passed": True,
            "validated_sheet_count": len(sheets),
            "source_hashes_verified": True,
            "whole_sheet_omission_scans_verified": True,
            "geometry_bounds_verified": True,
            "content_profile_verified": required_content_profile,
        },
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prediction_source_hashes(prediction: Mapping[str, Any]) -> set[str]:
    """Collect only explicit SHA-256 source identities from an exported graph."""

    values: list[object] = []
    for source in prediction.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        values.extend((source.get("source_hash"), source.get("sha256")))
    provenance = prediction.get("provenance")
    if isinstance(provenance, Mapping):
        values.extend(
            (
                provenance.get("source_image_sha256"),
                provenance.get("input_sha256"),
            )
        )
    pipeline = prediction.get("pipeline")
    if isinstance(pipeline, Mapping):
        values.append(pipeline.get("input_sha256"))
    return {
        value.lower()
        for raw in values
        if isinstance(raw, str)
        for value in [raw.strip()]
        if len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    }


def assert_prediction_source_correspondence(
    prediction: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    sheet_id: str,
) -> None:
    """Fail closed when a same-named prediction came from another drawing."""

    observed = prediction_source_hashes(prediction)
    expected = expected_source_sha256.lower()
    if expected not in observed:
        detail = ", ".join(sorted(observed)) if observed else "no source hash"
        raise ValueError(
            f"prediction/source mismatch for {sheet_id}: expected {expected}, observed {detail}"
        )


def assert_prediction_not_trained_on_source(
    prediction: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    sheet_id: str,
) -> None:
    """Reject in-sample evaluation using a model's exclusion ledger."""

    pipeline = prediction.get("pipeline")
    exclusions = (
        pipeline.get("model_training_source_exclusions", [])
        if isinstance(pipeline, Mapping)
        else []
    )
    normalized = {
        value.strip().lower()
        for value in exclusions
        if isinstance(value, str) and len(value.strip()) == 64
    }
    expected = expected_source_sha256.lower()
    if expected in normalized:
        raise ValueError(
            f"evaluation leakage for {sheet_id}: source {expected} was used to "
            "train or calibrate a model in this prediction"
        )


def assert_prediction_scale_correspondence(
    prediction: Mapping[str, Any],
    *,
    expected_pixels_per_meter: float,
    sheet_id: str,
) -> None:
    """Reject metric comparisons whose source-pixel calibration changed."""

    pipeline = prediction.get("pipeline")
    scale = pipeline.get("metric_scale", {}) if isinstance(pipeline, Mapping) else {}
    if not isinstance(scale, Mapping) or (
        scale.get("contract") != "source_pixels_to_metric_bim_v1"
    ):
        raise ValueError(
            f"prediction/scale contract missing for {sheet_id}; metric evaluation "
            "cannot infer calibration from geometry"
        )
    observed = float(scale.get("pixels_per_meter") or 0.0)
    if not math.isclose(
        observed,
        float(expected_pixels_per_meter),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"prediction/scale mismatch for {sheet_id}: graph uses {observed:g} "
            f"px/m but evaluation requested {expected_pixels_per_meter:g} px/m"
        )
