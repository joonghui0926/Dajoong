"""Fail-closed validation for directly reviewed Plan2BIM ground truth."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


class GroundTruthPolicyError(ValueError):
    """Raised when an annotation cannot be treated as ground truth."""


BENCHMARK_ENTITY_GROUPS = ("rooms", "walls", "openings", "fixtures", "routes")
ENTITY_KIND_TO_GROUP = {
    "room": "rooms",
    "wall": "walls",
    "opening": "openings",
    "fixture": "fixtures",
    "route": "routes",
}


ALLOWED_LICENSE_SCOPES = {
    "commercial_train",
    "research_eval_only",
    "internal_eval_only",
}
ALLOWED_COMMERCIAL_PERMISSION_BASES = {
    "dajoong_owned",
    "explicit_commercial_training_license",
    "written_permission",
}
REQUIRED_VISUAL_PASSES = {
    "full_sheet",
    "walls",
    "openings",
    "rooms",
    "fixtures",
}
FULL_EDITABLE_BIM_VISUAL_PASSES = {"furniture", "typed_appliances"}
FULL_EDITABLE_BIM_REQUIRED_CONTENT = {
    "walls",
    "openings",
    "room_regions",
    "fixed_fixtures",
    "movable_furniture",
    "drawn_appliances",
    "drawn_cabinet_modules",
}
ALLOWED_CONTENT_PROFILES = {"structural_core", "full_editable_bim"}
PROHIBITED_METHOD_TERMS = {
    "auto",
    "generated",
    "model",
    "prediction",
    "pseudo",
    "teacher",
}
DIRECT_GEOMETRY_ORIGIN = "independent_source_pixel_manual_authoring"
DIRECT_ENTITY_EVIDENCE_KIND = "native_source_pixels"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GroundTruthPolicyError(message)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(coordinate) for coordinate in point) else None


def _entity_points(entity: Mapping[str, Any]) -> list[tuple[float, float]]:
    polygon = entity.get("polygon")
    if isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes)):
        return [point for value in polygon if (point := _point(value)) is not None]
    points = entity.get("points_m") or entity.get("points_px")
    if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        return [point for value in points if (point := _point(value)) is not None]
    line = entity.get("line_px") or entity.get("line_m")
    if isinstance(line, Sequence) and not isinstance(line, (str, bytes)):
        return [point for value in line if (point := _point(value)) is not None]
    output = []
    for field in ("from", "to", "center_m", "center_px"):
        if (point := _point(entity.get(field))) is not None:
            output.append(point)
    return output


def compile_benchmark_graph_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the metric target only from source-pixel annotation events.

    Keeping a separately authored ``benchmark-graph.json`` allowed a valid-looking
    manifest to sit beside coordinates copied from an SVG or a model prediction.
    The manifest is now the sole geometry source; the graph is a deterministic
    metric projection of those recorded annotation events.
    """

    validate_ground_truth_manifest(manifest)
    source = _as_mapping(manifest.get("source"), "source")
    groups: dict[str, list[dict[str, Any]]] = {
        group: [] for group in BENCHMARK_ENTITY_GROUPS
    }
    for raw_entity in manifest.get("entities", []):
        entity = _as_mapping(raw_entity, "entity")
        entity_kind = str(entity["entity_kind"])
        group = ENTITY_KIND_TO_GROUP[entity_kind]
        geometry = deepcopy(dict(_as_mapping(entity["geometry"], "geometry")))
        metric_entity = {
            "id": str(entity["entity_id"]),
            "annotation_event_id": str(entity["annotation_event_id"]),
            "evidence_bbox_px": list(entity["evidence_bbox_px"]),
            **geometry,
        }
        groups[group].append(metric_entity)
    for entities in groups.values():
        entities.sort(key=lambda entity: entity["id"])
    return {
        "schema_version": "dajoong.manual-benchmark-graph.v3",
        "sheet_id": str(source.get("sheet_id", "")),
        "source_image_sha256": str(source["image_sha256"]),
        "annotation_session_id": str(
            _as_mapping(manifest["visual_review"], "visual_review")[
                "annotation_session_id"
            ]
        ),
        "levels": [{"id": "level-1", "name": "Level 1", "elevation_m": 0.0}],
        **groups,
        "vertical_connections": [],
    }


def assert_manifest_graph_correspondence(
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> None:
    """Reject metric targets that are not the exact manifest compilation."""

    expected = compile_benchmark_graph_from_manifest(manifest)
    actual = deepcopy(dict(graph))
    for group in BENCHMARK_ENTITY_GROUPS:
        values = actual.get(group, [])
        if isinstance(values, list):
            values.sort(key=lambda entity: str(entity.get("id", "")))
    _require(
        actual == expected,
        "benchmark graph must be compiled exactly from source-pixel annotation events",
    )


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, (*points[1:], points[0]), strict=True)
        )
        / 2
    )


def audit_benchmark_graph_geometry(
    graph: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    reviewed_plan_bbox_px: Sequence[float] | None = None,
    bounds_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    """Return every source-coordinate defect that invalidates an F1 target.

    A target outside the reviewed raster or with zero-area geometry cannot be
    detected from the supplied image.  Keeping it in the denominator trains and
    rewards coordinate corruption, so benchmark consumers must fail closed.
    """

    width, height = image_size
    issues: list[dict[str, Any]] = []
    if width <= 0 or height <= 0:
        return [{"code": "invalid_image_size", "image_size": list(image_size)}]

    def add(code: str, **details: Any) -> None:
        issues.append({"code": code, **details})

    def outside(point: tuple[float, float]) -> bool:
        return not (
            -bounds_tolerance_px <= point[0] <= width + bounds_tolerance_px
            and -bounds_tolerance_px <= point[1] <= height + bounds_tolerance_px
        )

    if reviewed_plan_bbox_px is not None:
        values = list(reviewed_plan_bbox_px)
        if len(values) != 4:
            add("invalid_reviewed_plan_bbox", value=values)
        else:
            try:
                left, top, right, bottom = (float(value) for value in values)
            except (TypeError, ValueError):
                add("invalid_reviewed_plan_bbox", value=values)
            else:
                if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                    add("invalid_reviewed_plan_bbox", value=values)
                elif right <= left or bottom <= top:
                    add("degenerate_reviewed_plan_bbox", value=values)
                elif outside((left, top)) or outside((right, bottom)):
                    add(
                        "reviewed_plan_bbox_outside_source",
                        value=values,
                        image_size=[width, height],
                    )

    for group in BENCHMARK_ENTITY_GROUPS:
        entities = graph.get(group, [])
        if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
            add("invalid_entity_group", group=group)
            continue
        for index, raw_entity in enumerate(entities):
            if not isinstance(raw_entity, Mapping):
                add("invalid_entity", group=group, index=index)
                continue
            entity_id = str(raw_entity.get("id") or f"{group}:{index}")
            points = _entity_points(raw_entity)
            if not points:
                add("missing_geometry", group=group, index=index, entity_id=entity_id)
                continue
            if group in {"rooms", "openings", "fixtures"}:
                polygon = raw_entity.get("polygon")
                if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)):
                    add("missing_polygon", group=group, index=index, entity_id=entity_id)
                elif len(points) < 3 or _polygon_area(points) <= 0.25:
                    add(
                        "degenerate_polygon",
                        group=group,
                        index=index,
                        entity_id=entity_id,
                        points=[list(point) for point in points],
                    )
            if group == "walls" and len(points) < 2:
                add("degenerate_wall", group=group, index=index, entity_id=entity_id)
            invalid_points = [list(point) for point in points if outside(point)]
            if invalid_points:
                add(
                    "entity_outside_source",
                    group=group,
                    index=index,
                    entity_id=entity_id,
                    invalid_points=invalid_points,
                    image_size=[width, height],
                )
    return issues


def assert_benchmark_graph_geometry(
    graph: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    reviewed_plan_bbox_px: Sequence[float] | None = None,
) -> None:
    issues = audit_benchmark_graph_geometry(
        graph,
        image_size=image_size,
        reviewed_plan_bbox_px=reviewed_plan_bbox_px,
    )
    if issues:
        first = issues[0]
        raise GroundTruthPolicyError(
            "benchmark graph is not aligned to the reviewed source raster: "
            f"{first['code']} ({len(issues)} total issue(s))"
        )


def validate_ground_truth_manifest(
    manifest: Mapping[str, Any],
    *,
    source_root: str | Path | None = None,
) -> None:
    """Validate that a manifest represents direct source-image annotation.

    This validator deliberately does not infer intent. Missing provenance or review
    evidence rejects the manifest instead of silently treating candidates as truth.
    """

    _require(
        manifest.get("schema_version") == "dajoong.manual-ground-truth.v2",
        "schema_version must be dajoong.manual-ground-truth.v2",
    )

    source = _as_mapping(manifest.get("source"), "source")
    review = _as_mapping(manifest.get("visual_review"), "visual_review")

    method = str(review.get("annotation_method", "")).strip().lower()
    _require(
        method == "direct_visual_source_annotation",
        "ground truth must be direct visual source annotation",
    )
    _require(
        not any(term in method for term in PROHIBITED_METHOD_TERMS),
        "automated, generated, model or pseudo-label methods cannot be ground truth",
    )
    _require(review.get("whole_sheet_reviewed") is True, "the whole source sheet must be reviewed")
    _require(
        review.get("candidate_output_role") == "review_aid_only_not_ground_truth",
        "candidate output must be declared as review aid only",
    )
    _require(bool(str(review.get("annotator", "")).strip()), "annotator is required")
    _require(bool(str(review.get("reviewed_on", "")).strip()), "reviewed_on is required")
    # A prose label saying "direct visual" is not provenance.  The retired
    # benchmark adapter copied SVG/candidate geometry and then wrote that label,
    # which allowed derived coordinates to enter an F1 denominator.  New
    # packets must explicitly record that geometry was authored independently
    # against source pixels.  Legacy manifests therefore fail closed.
    _require(
        review.get("geometry_origin") == DIRECT_GEOMETRY_ORIGIN,
        "ground-truth geometry must be independently authored from source pixels",
    )
    _require(
        bool(str(review.get("annotation_session_id", "")).strip()),
        "a source-pixel annotation_session_id is required",
    )

    passes = review.get("review_passes")
    _require(
        isinstance(passes, Sequence) and not isinstance(passes, str), "review_passes must be a list"
    )
    missing_passes = REQUIRED_VISUAL_PASSES - {str(item) for item in passes}
    _require(not missing_passes, f"missing visual review passes: {sorted(missing_passes)}")

    target_contract = _as_mapping(manifest.get("target_contract"), "target_contract")
    content_profile = str(target_contract.get("content_profile", "")).strip()
    _require(
        content_profile in ALLOWED_CONTENT_PROFILES,
        "target_contract.content_profile must be structural_core or full_editable_bim",
    )
    if content_profile == "full_editable_bim":
        missing_product_passes = FULL_EDITABLE_BIM_VISUAL_PASSES - {
            str(item) for item in passes
        }
        _require(
            not missing_product_passes,
            "full_editable_bim ground truth is missing visual review passes: "
            f"{sorted(missing_product_passes)}",
        )
        excluded = {str(item) for item in target_contract.get("excluded", [])}
        _require(
            "movable_furniture" not in excluded,
            "full_editable_bim ground truth cannot exclude movable_furniture",
        )
        included = {str(item) for item in target_contract.get("included", [])}
        _require(
            "all_visible_editable_bim_elements" in included
            or not (FULL_EDITABLE_BIM_REQUIRED_CONTENT - included),
            "full_editable_bim ground truth must explicitly include every editable "
            "content family",
        )

    scope = str(source.get("license_scope", ""))
    _require(scope in ALLOWED_LICENSE_SCOPES, f"invalid source license_scope: {scope!r}")
    _require(bool(str(source.get("image_sha256", "")).strip()), "source image_sha256 is required")
    bounds = source.get("reviewed_plan_bbox_px")
    _require(
        isinstance(bounds, Sequence) and not isinstance(bounds, str) and len(bounds) == 4,
        "reviewed_plan_bbox_px must contain four coordinates",
    )

    image_path_value = str(source.get("image_path", "")).strip()
    _require(bool(image_path_value), "source image_path is required")
    image_path = Path(image_path_value)
    if not image_path.is_absolute() and source_root is not None:
        image_path = Path(source_root) / image_path
    _require(image_path.is_file(), f"source image does not exist: {image_path}")
    _require(
        _sha256(image_path) == source["image_sha256"],
        "source image hash does not match the visually reviewed source",
    )
    try:
        width_px = int(source.get("width_px"))
        height_px = int(source.get("height_px"))
    except (TypeError, ValueError):
        raise GroundTruthPolicyError("source width_px and height_px are required") from None
    _require(width_px > 0 and height_px > 0, "source dimensions must be positive")
    with Image.open(image_path) as source_image:
        _require(
            source_image.size == (width_px, height_px),
            "source dimensions do not match the visually reviewed image",
        )

    try:
        reviewed_left, reviewed_top, reviewed_right, reviewed_bottom = (
            float(value) for value in bounds
        )
    except (TypeError, ValueError):
        raise GroundTruthPolicyError("reviewed_plan_bbox_px must be numeric") from None
    _require(
        0 <= reviewed_left < reviewed_right <= width_px
        and 0 <= reviewed_top < reviewed_bottom <= height_px,
        "reviewed_plan_bbox_px must stay inside the source image",
    )

    entities = manifest.get("entities")
    _require(
        isinstance(entities, Sequence) and not isinstance(entities, str), "entities must be a list"
    )
    entity_ids: set[str] = set()
    annotation_event_ids: set[str] = set()
    for index, entity_value in enumerate(entities):
        entity = _as_mapping(entity_value, f"entities[{index}]")
        _require(
            entity.get("directly_annotated") is True,
            f"entities[{index}] was not directly annotated",
        )
        _require(
            entity.get("evidence_kind") == DIRECT_ENTITY_EVIDENCE_KIND,
            f"entities[{index}] must cite native source pixels as its evidence",
        )
        _require(
            bool(str(entity.get("annotation_event_id", "")).strip()),
            f"entities[{index}] annotation_event_id is required",
        )
        _require(
            bool(str(entity.get("entity_kind", "")).strip()),
            f"entities[{index}] entity_kind is required",
        )
        _require(
            str(entity.get("entity_kind")) in ENTITY_KIND_TO_GROUP,
            f"entities[{index}] has an unsupported entity_kind",
        )
        _require(
            bool(str(entity.get("entity_id", "")).strip()),
            f"entities[{index}] entity_id is required",
        )
        entity_id = str(entity["entity_id"])
        annotation_event_id = str(entity["annotation_event_id"])
        _require(entity_id not in entity_ids, f"duplicate entity_id: {entity_id}")
        _require(
            annotation_event_id not in annotation_event_ids,
            f"duplicate annotation_event_id: {annotation_event_id}",
        )
        entity_ids.add(entity_id)
        annotation_event_ids.add(annotation_event_id)
        evidence = entity.get("evidence_bbox_px")
        _require(
            isinstance(evidence, Sequence) and not isinstance(evidence, str) and len(evidence) == 4,
            f"entities[{index}] evidence_bbox_px must contain four coordinates",
        )
        try:
            left, top, right, bottom = (float(value) for value in evidence)
        except (TypeError, ValueError):
            raise GroundTruthPolicyError(
                f"entities[{index}] evidence_bbox_px must be numeric"
            ) from None
        _require(
            0 <= left < right <= width_px and 0 <= top < bottom <= height_px,
            f"entities[{index}] evidence_bbox_px must stay inside the source image",
        )
        geometry = _as_mapping(entity.get("geometry"), f"entities[{index}].geometry")
        geometry_points = _entity_points(geometry)
        _require(geometry_points, f"entities[{index}] geometry has no source-pixel points")
        _require(
            all(
                left - 1 <= x <= right + 1 and top - 1 <= y <= bottom + 1
                for x, y in geometry_points
            ),
            f"entities[{index}] geometry must stay inside its evidence_bbox_px",
        )
        entity_kind = str(entity["entity_kind"])
        if entity_kind == "wall":
            _require(
                _point(geometry.get("from")) is not None
                and _point(geometry.get("to")) is not None,
                f"entities[{index}] wall requires from and to source-pixel points",
            )
        if entity_kind in {"room", "opening", "fixture"}:
            polygon = geometry.get("polygon")
            _require(
                isinstance(polygon, Sequence)
                and not isinstance(polygon, (str, bytes))
                and _polygon_area(_entity_points({"polygon": polygon})) > 0.25,
                f"entities[{index}] {entity_kind} requires a non-degenerate polygon",
            )
        semantic_field = {
            "room": "room_type",
            "opening": "type",
            "fixture": "fixture_type",
        }.get(entity_kind)
        if semantic_field:
            _require(
                bool(str(geometry.get(semantic_field, "")).strip()),
                f"entities[{index}] {entity_kind} requires {semantic_field}",
            )

    omission_scan = _as_mapping(manifest.get("omission_scan"), "omission_scan")
    _require(omission_scan.get("completed") is True, "whole-plan omission scan is required")
    _require(
        omission_scan.get("coverage") == "entire_reviewed_plan_bbox",
        "omission scan must cover the entire reviewed plan",
    )
    _require(
        isinstance(omission_scan.get("unresolved_findings"), Sequence)
        and not isinstance(omission_scan.get("unresolved_findings"), (str, bytes)),
        "omission scan must record unresolved_findings",
    )
    if content_profile == "full_editable_bim":
        for field in (
            "visible_movable_furniture_count",
            "visible_typed_appliance_count",
            "visible_fixed_fixture_count",
            "visible_cabinet_module_count",
        ):
            _require(
                field in omission_scan
                and isinstance(omission_scan[field], int)
                and omission_scan[field] >= 0,
                f"full_editable_bim omission scan requires non-negative {field}",
            )
        audited_fixture_total = sum(
            int(omission_scan[field])
            for field in (
                "visible_movable_furniture_count",
                "visible_typed_appliance_count",
                "visible_fixed_fixture_count",
                "visible_cabinet_module_count",
            )
        )
        annotated_fixture_total = sum(
            str(entity["entity_kind"]) == "fixture" for entity in entities
        )
        _require(
            annotated_fixture_total == audited_fixture_total,
            "full_editable_bim fixture count does not match the manual omission scan",
        )


def assert_evaluation_content_profile(
    manifest: Mapping[str, Any],
    *,
    required_profile: str,
) -> None:
    """Prevent a narrow benchmark denominator from supporting a broad claim."""

    _require(
        required_profile in ALLOWED_CONTENT_PROFILES,
        f"unknown required evaluation content profile: {required_profile}",
    )
    target_contract = _as_mapping(manifest.get("target_contract"), "target_contract")
    actual = str(target_contract.get("content_profile", "")).strip()
    _require(
        actual == required_profile,
        f"evaluation requires content profile {required_profile!r}, got {actual!r}",
    )


def assert_commercial_training_eligible(manifest: Mapping[str, Any]) -> None:
    """Reject an otherwise valid annotation if it is not commercially trainable."""

    source = _as_mapping(manifest.get("source"), "source")
    _require(
        source.get("license_scope") == "commercial_train",
        "only commercial_train ground truth may enter the production training split",
    )
    provenance = _as_mapping(source.get("license_provenance"), "source.license_provenance")
    _require(
        provenance.get("permission_basis") in ALLOWED_COMMERCIAL_PERMISSION_BASES,
        "commercial training requires a verified permission basis",
    )
    _require(
        provenance.get("commercial_training_allowed") is True,
        "commercial training permission must be explicit",
    )
    _require(
        provenance.get("derivative_model_commercial_use_allowed") is True,
        "commercial use of the trained model must be allowed",
    )
    for field in ("rights_holder", "evidence_ref", "verified_by", "verified_on"):
        _require(
            bool(str(provenance.get(field, "")).strip()),
            f"source.license_provenance.{field} is required",
        )
    _require(
        bool(str(source.get("collection_group_id", "")).strip()),
        "source.collection_group_id is required to prevent train/test leakage",
    )
