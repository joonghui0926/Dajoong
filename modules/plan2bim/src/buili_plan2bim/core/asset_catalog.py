"""Offline, license-audited BIM family resolution for the conversion hot path.

Recognition decides *what* and *where*.  This module only resolves a matching
geometry family after the semantic decision exists.  It never calls a network
service and never turns an unknown class into a guessed high-detail object.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .cad_families import FAMILY_MANIFESTS, parametric_family_parts
from .hashing import sha256_json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_ROOT = PACKAGE_ROOT / "assets" / "curated-v1"
ALLOWED_LICENSES = frozenset({"by", "cc0"})

# Exact semantic class to asset family mapping.  Generic appliances deliberately
# have no mapping: a high-detail refrigerator is still wrong when the source only
# establishes "appliance".
ASSET_FAMILY_BY_BIM_FAMILY = {
    "residential-toilet": "toilet",
    "residential-sink": "sink",
    "residential-closet": "wardrobe",
    "residential-bed": "bed",
    "residential-sofa": "sofa",
    "residential-armchair": "armchair",
    "residential-chair": "chair",
    "residential-dining-table": "dining_table",
    "residential-coffee-table": "coffee_table",
    "residential-dishwasher": "dishwasher",
    "residential-washing-machine": "washing_machine",
    "residential-refrigerator": "refrigerator",
    "residential-stove": "stove",
    "residential-shower-head": "shower_head",
    "residential-shower-enclosure": "shower_enclosure",
    "residential-bench": "bench",
    "residential-base-cabinet": "base_cabinet",
    "residential-wall-cabinet": "wall_cabinet",
    "residential-desk": "dining_table",
    "residential-tumble-dryer": "washing_machine",
}

NATIVE_ASSET_VARIANTS: dict[str, tuple[float, float, float]] = {
    "compact": (0.72, 0.86, 1.08),
    "standard": (1.0, 1.0, 1.0),
    "wide": (1.34, 0.82, 0.92),
}

OUTDOOR_ROOM_TERMS = frozenset({"balcony", "deck", "garden", "outdoor", "patio", "terrace"})
ROOM_ASSET_TERMS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"bunk"}), frozenset({"bed", "bedroom", "dorm", "kids", "sleep"})),
    (frozenset({"bar", "counter"}), frozenset({"dining", "kitchen", "restaurant"})),
    (frozenset({"desk", "desktop", "office"}), frozenset({"office", "study", "work"})),
    (frozenset({"dining"}), frozenset({"dining", "kitchen", "restaurant"})),
    (frozenset({"closet", "wardrobe"}), frozenset({"bed", "bedroom", "closet", "dressing"})),
    (frozenset({"couch", "sofa"}), frozenset({"family", "living", "lounge"})),
)


def _mesh_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    return sha256_json(
        {
            "vertices": np.asarray(vertices, dtype=np.float32).round(7).tolist(),
            "faces": np.asarray(faces, dtype=np.int32).tolist(),
        }
    )


def _box_component(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hx, hy, hz = (max(float(value), 1e-5) / 2 for value in size)
    cx, cy, cz = center
    vertices = np.asarray(
        [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(faces), 1))
    return vertices, faces, colors


def _combine_components(
    components: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for component_vertices, component_faces, component_colors in components:
        vertices.append(component_vertices)
        faces.append(component_faces + offset)
        colors.append(component_colors)
        offset += len(component_vertices)
    return np.vstack(vertices), np.vstack(faces), np.vstack(colors)


def _native_family_mesh(
    family_id: str,
    size: tuple[float, float, float],
    discipline: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if family_id not in FAMILY_MANIFESTS:
        return None
    palette = {
        "architectural": (139, 117, 89),
        "electrical": (63, 79, 83),
        "mechanical": (88, 117, 111),
        "plumbing": (205, 209, 204),
        "fire": (154, 89, 74),
    }
    color = palette.get(discipline, palette["architectural"])
    return _combine_components(
        _box_component(
            (part.center[0], part.center[1], part.center[2] + part.size[2] / 2),
            part.size,
            color,
        )
        for part in parametric_family_parts(family_id, size)
    )


def _normalize_and_fit(
    vertices: np.ndarray,
    target_size: tuple[float, float, float],
) -> tuple[np.ndarray, bool, tuple[float, float, float]]:
    """Fit an authored mesh to the evidence envelope after best-axis alignment.

    Candidate choice is semantic first.  Exact XYZ fitting then preserves the
    measured drawing footprint; the scale vector remains recorded for audit.
    """

    adapted = np.asarray(vertices, dtype=np.float64).copy()
    minimum = adapted.min(axis=0)
    maximum = adapted.max(axis=0)
    extents = maximum - minimum
    if np.any(extents <= 1e-8) or not np.isfinite(adapted).all():
        raise ValueError("catalog mesh is degenerate")
    target_ratio = target_size[0] / max(target_size[1], 1e-8)
    direct_ratio = extents[0] / max(extents[1], 1e-8)
    swapped_ratio = extents[1] / max(extents[0], 1e-8)
    swapped = abs(math.log(target_ratio / swapped_ratio)) < abs(
        math.log(target_ratio / direct_ratio)
    )
    if swapped:
        adapted[:, [0, 1]] = adapted[:, [1, 0]]
        minimum = adapted.min(axis=0)
        maximum = adapted.max(axis=0)
        extents = maximum - minimum
    adapted[:, 0] -= (minimum[0] + maximum[0]) / 2
    adapted[:, 1] -= (minimum[1] + maximum[1]) / 2
    adapted[:, 2] -= minimum[2]
    scale = tuple(float(value) for value in np.asarray(target_size) / extents)
    adapted *= np.asarray(scale)
    return adapted.astype(np.float32), swapped, scale


def _normalize_to_unit_envelope(
    vertices: np.ndarray,
    target_size: tuple[float, float, float],
) -> tuple[np.ndarray, bool, tuple[float, float, float]]:
    adapted = np.asarray(vertices, dtype=np.float64).copy()
    minimum = adapted.min(axis=0)
    maximum = adapted.max(axis=0)
    extents = maximum - minimum
    if np.any(extents <= 1e-8) or not np.isfinite(adapted).all():
        raise ValueError("catalog mesh is degenerate")
    target_ratio = target_size[0] / max(target_size[1], 1e-8)
    direct_ratio = extents[0] / max(extents[1], 1e-8)
    swapped_ratio = extents[1] / max(extents[0], 1e-8)
    swapped = abs(math.log(target_ratio / swapped_ratio)) < abs(
        math.log(target_ratio / direct_ratio)
    )
    normalized, extents = _normalize_fixed_orientation(adapted, swapped=swapped)
    source_scale = tuple(float(value) for value in np.asarray(target_size) / extents)
    return normalized, swapped, source_scale


def _best_axis_swap(
    extents: tuple[float, float, float] | np.ndarray,
    target_size: tuple[float, float, float],
) -> bool:
    target_ratio = target_size[0] / max(target_size[1], 1e-8)
    direct_ratio = float(extents[0]) / max(float(extents[1]), 1e-8)
    swapped_ratio = float(extents[1]) / max(float(extents[0]), 1e-8)
    return abs(math.log(target_ratio / swapped_ratio)) < abs(
        math.log(target_ratio / direct_ratio)
    )


def _normalize_fixed_orientation(
    vertices: np.ndarray,
    *,
    swapped: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one explicit XY orientation to a content-addressable unit envelope."""

    adapted = np.asarray(vertices, dtype=np.float64).copy()
    if swapped:
        adapted[:, [0, 1]] = adapted[:, [1, 0]]
    minimum = adapted.min(axis=0)
    maximum = adapted.max(axis=0)
    extents = maximum - minimum
    if np.any(extents <= 1e-8) or not np.isfinite(adapted).all():
        raise ValueError("catalog mesh is degenerate")
    adapted[:, 0] -= (minimum[0] + maximum[0]) / 2
    adapted[:, 1] -= (minimum[1] + maximum[1]) / 2
    adapted[:, 2] -= minimum[2]
    adapted /= extents
    return adapted.astype(np.float32), extents


@lru_cache(maxsize=8)
def _catalog_cached(
    catalog_root_value: str,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], ...]]]:
    catalog_root = Path(catalog_root_value)
    path = catalog_root / "catalog.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "dajoong.curated-bim-family-catalog.v1":
        raise ValueError("unsupported curated family catalog")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog.get("entries", []):
        for family in entry.get("families", []):
            grouped.setdefault(str(family), []).append(entry)
    entries = {
        family: tuple(sorted(candidates, key=lambda item: str(item.get("uid") or "")))
        for family, candidates in grouped.items()
    }
    return catalog, entries


def _catalog(
    catalog_root: Path,
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], ...]]]:
    return _catalog_cached(str(catalog_root.resolve()))


@lru_cache(maxsize=512)
def _load_catalog_mesh(path_value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path_value) as mesh:
        vertices = np.asarray(mesh["vertices"], dtype=np.float32)
        faces = np.asarray(mesh["faces"], dtype=np.int32)
        colors = np.asarray(mesh["face_colors"], dtype=np.uint8)[:, :3]
    vertices.setflags(write=False)
    faces.setflags(write=False)
    colors.setflags(write=False)
    return vertices, faces, colors


@lru_cache(maxsize=512)
def _normalized_catalog_asset(
    path_value: str,
    swapped: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, tuple[float, float, float]]:
    vertices, faces, colors = _load_catalog_mesh(path_value)
    normalized, extents = _normalize_fixed_orientation(vertices, swapped=swapped)
    mesh_sha256 = _mesh_sha256(normalized, faces)
    normalized.setflags(write=False)
    return normalized, faces, colors, mesh_sha256, tuple(float(value) for value in extents)


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in "".join(
            character.lower() if character.isalnum() else " " for character in str(value or "")
        ).split()
        if token
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared),
    )
    return math.dist(point, (start[0] + fraction * dx, start[1] + fraction * dy))


def _wall_endpoints(
    wall: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Read both the current PlanGraph keys and the pre-v2 compatibility keys."""

    start_values = wall.get("from") or wall.get("start_m") or ()
    end_values = wall.get("to") or wall.get("end_m") or ()
    if len(start_values) < 2 or len(end_values) < 2:
        return None
    return (
        (float(start_values[0]), float(start_values[1])),
        (float(end_values[0]), float(end_values[1])),
    )


def _asset_context_index(graph: dict[str, Any]) -> dict[str, Any]:
    """Build a small spatial index once so contextual matching stays linear."""

    cell_size_m = 2.5
    fixtures_by_cell: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for fixture in graph.get("fixtures") or []:
        center = fixture.get("center_m") or ()
        if len(center) < 2:
            continue
        key = (
            str(fixture.get("level_id") or ""),
            math.floor(float(center[0]) / cell_size_m),
            math.floor(float(center[1]) / cell_size_m),
        )
        fixtures_by_cell.setdefault(key, []).append(fixture)
    walls_by_level: dict[str, list[dict[str, Any]]] = {}
    for wall in graph.get("walls") or []:
        walls_by_level.setdefault(str(wall.get("level_id") or ""), []).append(wall)
    return {
        "cell_size_m": cell_size_m,
        "fixtures_by_cell": fixtures_by_cell,
        "rooms_by_id": {
            str(room.get("id") or ""): room for room in graph.get("rooms") or []
        },
        "walls_by_level": walls_by_level,
    }


def _asset_context(
    graph: dict[str, Any],
    fixture: dict[str, Any],
    index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index = index or _asset_context_index(graph)
    room_id = str(fixture.get("room_id") or "")
    room = index["rooms_by_id"].get(room_id, {})
    room_label = " ".join(
        value for value in (str(room.get("name") or ""), str(room.get("occupancy") or "")) if value
    )
    room_tokens = _tokens(room_label)
    center_values = fixture.get("center_m") or ()
    wall_distance = math.inf
    nearest_wall_angle_deg: float | None = None
    neighbor_tokens: set[str] = set()
    neighbor_families: list[str] = []
    if len(center_values) >= 2:
        center = (float(center_values[0]), float(center_values[1]))
        level_id = str(fixture.get("level_id") or "")
        for wall in index["walls_by_level"].get(level_id, ()):
            endpoints = _wall_endpoints(wall)
            if endpoints is None:
                continue
            start, end = endpoints
            distance = _point_segment_distance(center, start, end)
            if distance < wall_distance:
                wall_distance = distance
                nearest_wall_angle_deg = math.degrees(
                    math.atan2(end[1] - start[1], end[0] - start[0])
                )
        cell_size_m = float(index["cell_size_m"])
        cell_x = math.floor(center[0] / cell_size_m)
        cell_y = math.floor(center[1] / cell_size_m)
        neighbors: list[tuple[float, str]] = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                key = (level_id, cell_x + offset_x, cell_y + offset_y)
                for candidate in index["fixtures_by_cell"].get(key, ()):
                    if candidate is fixture:
                        continue
                    candidate_center = candidate.get("center_m") or ()
                    if len(candidate_center) < 2:
                        continue
                    distance = math.dist(
                        center,
                        (float(candidate_center[0]), float(candidate_center[1])),
                    )
                    if distance <= cell_size_m:
                        family = str(
                            candidate.get("family_id") or candidate.get("type") or ""
                        )
                        if family:
                            neighbors.append((distance, family))
        for _, family in sorted(neighbors)[:8]:
            neighbor_families.append(family)
            neighbor_tokens.update(_tokens(family))
    mounting = str(fixture.get("mounting") or "").lower()
    discipline = str(fixture.get("discipline") or "").lower()
    installation = "room_interior"
    if mounting in {"ceiling", "ceiling_hosted"}:
        installation = "ceiling_hosted"
    elif mounting in {"wall", "wall_hosted"}:
        installation = "wall_hosted"
    elif discipline in {"plumbing", "mechanical", "electrical", "fire"} and math.isfinite(
        wall_distance
    ) and wall_distance <= 0.45:
        installation = "service_wall"
    elif math.isfinite(wall_distance) and wall_distance <= 0.35:
        installation = "wall_adjacent"
    elif room_tokens & OUTDOOR_ROOM_TERMS:
        installation = "outdoor"
    return {
        "room_id": room_id,
        "room_label": room_label,
        "room_tokens": room_tokens,
        "nearest_wall_m": None if not math.isfinite(wall_distance) else wall_distance,
        "nearest_wall_angle_deg": nearest_wall_angle_deg,
        "installation": installation,
        "nearby_families": tuple(neighbor_families),
        "nearby_tokens": frozenset(neighbor_tokens),
    }


def _context_affinity(
    entry: dict[str, Any],
    fixture: dict[str, Any],
    context: dict[str, Any],
) -> float:
    name_tokens = _tokens(entry.get("name"))
    room_tokens = context["room_tokens"]
    nearby_tokens = context.get("nearby_tokens") or frozenset()
    scores: list[float] = []
    outdoor_asset = bool(name_tokens & frozenset({"garden", "outdoor", "park", "patio"}))
    outdoor_room = bool(room_tokens & OUTDOOR_ROOM_TERMS)
    if outdoor_asset:
        scores.append(1.0 if outdoor_room else 0.05)
    elif outdoor_room:
        scores.append(0.60)
    for asset_terms, compatible_rooms in ROOM_ASSET_TERMS:
        if name_tokens & asset_terms:
            scores.append(1.0 if room_tokens & compatible_rooms else 0.20)
    if "bunk" in name_tokens:
        size = fixture.get("size_m") or ()
        scores.append(1.0 if len(size) >= 3 and float(size[2]) >= 1.20 else 0.10)
    if "set" in name_tokens and int(fixture.get("observed_count") or 1) <= 1:
        scores.append(0.25)
    if "dining" in name_tokens:
        dining_context = bool(
            room_tokens & frozenset({"dining", "kitchen", "breakfast"})
            or nearby_tokens & frozenset({"chair", "stool"})
        )
        scores.append(1.0 if dining_context else 0.18)
    if name_tokens & frozenset({"desktop", "game", "adjustable"}):
        office_context = bool(
            room_tokens & frozenset({"office", "study", "work", "classroom"})
            or nearby_tokens & frozenset({"computer", "monitor", "desk"})
        )
        scores.append(1.0 if office_context else 0.32)
    if "double" in name_tokens and "sink" in name_tokens:
        size = fixture.get("size_m") or ()
        scores.append(1.0 if len(size) >= 1 and float(size[0]) >= 1.10 else 0.15)
    if name_tokens & frozenset({"kitchen", "dish", "drainer", "hood"}):
        kitchen_context = bool(room_tokens & frozenset({"kitchen", "galley"}))
        scores.append(1.0 if kitchen_context else 0.08)
    if "park" in name_tokens:
        scores.append(1.0 if context.get("installation") == "outdoor" else 0.12)
    return sum(scores) / len(scores) if scores else 0.68


def _shape_similarity(entry: dict[str, Any], target_size: tuple[float, float, float]) -> float:
    extents = tuple(float(value) for value in (entry.get("normalized_extents") or ()))
    if len(extents) != 3 or any(value <= 1e-8 for value in extents):
        return 0.5
    errors = []
    for source in (extents, (extents[1], extents[0], extents[2])):
        scale_logs = [
            math.log(target / authored)
            for target, authored in zip(target_size, source, strict=True)
        ]
        mean = sum(scale_logs) / 3
        errors.append(math.sqrt(sum((value - mean) ** 2 for value in scale_logs) / 3))
    return math.exp(-min(errors))


def _native_variant_ranking(
    target_size: tuple[float, float, float],
) -> tuple[tuple[str, float], ...]:
    ranked = []
    for variant, authored in NATIVE_ASSET_VARIANTS.items():
        ranked.append(
            (
                variant,
                _shape_similarity({"normalized_extents": authored}, target_size),
            )
        )
    return tuple(sorted(ranked, key=lambda item: (-item[1], item[0])))


def _rank_catalog_entries(
    candidates: tuple[dict[str, Any], ...],
    fixture: dict[str, Any],
    target_size: tuple[float, float, float],
    context: dict[str, Any],
) -> tuple[tuple[dict[str, Any], dict[str, float]], ...]:
    if not candidates:
        return ()
    preferred_uid = str(fixture.get("preferred_asset_uid") or "")
    target_ratio = target_size[0] / max(target_size[1], 1e-8)
    ranked: list[tuple[dict[str, Any], dict[str, float]]] = []
    for entry in candidates:
        authored_ratio = max(float(entry.get("horizontal_aspect") or 1.0), 1e-8)
        ratio_error = min(
            abs(math.log(target_ratio / authored_ratio)),
            abs(math.log(target_ratio * authored_ratio)),
        )
        footprint = math.exp(-ratio_error)
        shape = _shape_similarity(entry, target_size)
        contextual = _context_affinity(entry, fixture, context)
        faces = max(0, int(entry.get("face_count") or 0))
        detail = min(1.0, math.log1p(faces) / math.log1p(20_000))
        score = 0.40 * footprint + 0.32 * shape + 0.22 * contextual + 0.06 * detail
        if preferred_uid and str(entry.get("uid")) == preferred_uid:
            score = 1.0
        ranked.append(
            (
                entry,
                {
                    "score": score,
                    "footprint": footprint,
                    "shape": shape,
                    "context": contextual,
                    "detail": detail,
                },
            )
        )
    return tuple(
        sorted(ranked, key=lambda item: (-item[1]["score"], str(item[0].get("uid") or "")))
    )


def resolved_fixture_geometry(
    graph: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    """Return inline legacy geometry or a shared converter-owned asset definition."""

    if fixture.get("mesh_vertices") and fixture.get("mesh_faces"):
        return fixture
    reference = str(fixture.get("geometry_ref") or "")
    definition = (graph.get("family_assets") or {}).get(reference)
    if not isinstance(definition, dict):
        return {}
    if definition.get("normalized_to_unit_envelope") is not True:
        return definition
    vertices = np.asarray(definition.get("mesh_vertices") or [], dtype=np.float64)
    scale = np.asarray(
        fixture.get("geometry_scale_xyz") or fixture.get("size_m") or [],
        dtype=np.float64,
    )
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or scale.shape != (3,):
        return {}
    return {
        **definition,
        "mesh_vertices": (vertices * scale).round(7).tolist(),
    }


def audit_fixture_geometry(graph: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    resolved = 0
    normalized_asset_audits: dict[str, tuple[bool, str]] = {}
    for fixture in graph.get("fixtures") or []:
        status = str(fixture.get("geometry_status") or "semantic_marker")
        if status not in {"licensed_api_asset", "native_bim_parametric"}:
            continue
        resolved += 1
        fixture_id = str(fixture.get("id") or "")
        reference = str(fixture.get("geometry_ref") or "")
        definition = (graph.get("family_assets") or {}).get(reference)
        if isinstance(definition, dict) and definition.get("normalized_to_unit_envelope") is True:
            if reference not in normalized_asset_audits:
                vertices = np.asarray(definition.get("mesh_vertices") or [], dtype=np.float64)
                faces = np.asarray(definition.get("mesh_faces") or [], dtype=np.int64)
                reason = ""
                if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
                    reason = "invalid_vertices"
                elif faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
                    reason = "invalid_faces"
                elif (
                    not np.isfinite(vertices).all()
                    or faces.min() < 0
                    or faces.max() >= len(vertices)
                ):
                    reason = "mesh_integrity"
                else:
                    unit_extents = vertices.max(axis=0) - vertices.min(axis=0)
                    if not np.allclose(unit_extents, (1, 1, 1), rtol=0, atol=1e-5):
                        reason = "asset_not_unit_normalized"
                    elif abs(float(vertices[:, 2].min())) > 1e-5:
                        reason = "mesh_not_base_normalized"
                normalized_asset_audits[reference] = (not reason, reason)
            passed, reason = normalized_asset_audits[reference]
            if not passed:
                failures.append({"id": fixture_id, "reason": reason})
                continue
            scale = np.asarray(
                fixture.get("geometry_scale_xyz") or fixture.get("size_m") or (),
                dtype=np.float64,
            )
            target = np.asarray(fixture.get("size_m") or (), dtype=np.float64)
            if (
                scale.shape != (3,)
                or target.shape != (3,)
                or not np.allclose(scale, target, rtol=0, atol=1e-5)
            ):
                failures.append(
                    {
                        "id": fixture_id,
                        "reason": "mesh_does_not_match_evidence_envelope",
                        "geometry_scale_xyz": scale.round(7).tolist(),
                        "evidence_size_m": target.round(7).tolist(),
                    }
                )
            if (
                status == "licensed_api_asset"
                and str(fixture.get("asset_license")) not in ALLOWED_LICENSES
            ):
                failures.append({"id": fixture_id, "reason": "asset_license_not_allowed"})
            continue
        geometry = resolved_fixture_geometry(graph, fixture)
        vertices = np.asarray(geometry.get("mesh_vertices") or [], dtype=np.float64)
        faces = np.asarray(geometry.get("mesh_faces") or [], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
            failures.append({"id": fixture_id, "reason": "invalid_vertices"})
            continue
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
            failures.append({"id": fixture_id, "reason": "invalid_faces"})
            continue
        if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
            failures.append({"id": fixture_id, "reason": "mesh_integrity"})
            continue
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        target = np.asarray(fixture.get("size_m") or (), dtype=np.float64)
        if target.shape != (3,) or not np.allclose(extents, target, rtol=0, atol=1e-5):
            failures.append(
                {
                    "id": fixture_id,
                    "reason": "mesh_does_not_match_evidence_envelope",
                    "mesh_extents_m": extents.round(7).tolist(),
                    "evidence_size_m": target.round(7).tolist(),
                }
            )
        if abs(float(vertices[:, 2].min())) > 1e-5:
            failures.append({"id": fixture_id, "reason": "mesh_not_base_normalized"})
        if (
            status == "licensed_api_asset"
            and str(fixture.get("asset_license")) not in ALLOWED_LICENSES
        ):
            failures.append({"id": fixture_id, "reason": "asset_license_not_allowed"})
    unresolved = [
        str(item.get("id") or "")
        for item in graph.get("fixtures") or []
        if str(item.get("geometry_status") or "semantic_marker") == "semantic_marker"
    ]
    report = {
        "schema_version": "dajoong.fixture-geometry-audit.v1",
        "resolved_fixture_count": resolved,
        "unresolved_fixture_count": len(unresolved),
        "unresolved_fixture_ids": unresolved,
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    report["content_sha256"] = sha256_json(report)
    return report


@lru_cache(maxsize=4)
def _family_asset_library_cached(catalog_root_value: str) -> tuple[dict[str, Any], ...]:
    root = Path(catalog_root_value)
    catalog, _ = _catalog(root)
    records: list[dict[str, Any]] = []
    for entry in catalog.get("entries", []):
        license_slug = str(entry.get("license") or "").lower()
        if license_slug not in ALLOWED_LICENSES:
            raise ValueError(f"asset {entry.get('uid')} has a non-allowlisted license")
        mesh_path = str((root / str(entry["mesh_path"])).resolve())
        for orientation, swapped in (("direct", False), ("quarter_turn", True)):
            vertices, mesh_faces, mesh_colors, mesh_sha256, _ = _normalized_catalog_asset(
                mesh_path, swapped
            )
            records.append(
                {
                    "asset_id": f"licensed:{entry['uid']}:{orientation}",
                    "geometry_ref": f"mesh:{mesh_sha256}",
                    "families": [str(value) for value in entry.get("families", [])],
                    "orientation": orientation,
                    "geometry_status": "licensed_api_asset",
                    "asset_uid": entry["uid"],
                    "name": entry.get("name", ""),
                    "author": entry.get("author", ""),
                    "license": license_slug,
                    "source_uri": entry.get("source_uri", ""),
                    "definition": {
                        "schema_version": "dajoong.family-asset.v1",
                        "geometry_status": "licensed_api_asset",
                        "asset_uid": entry["uid"],
                        "asset_mesh_sha256": mesh_sha256,
                        "normalized_to_unit_envelope": True,
                        "mesh_vertices": vertices.round(7).tolist(),
                        "mesh_faces": mesh_faces.tolist(),
                        "mesh_face_colors": mesh_colors.tolist(),
                    },
                }
            )

    for family_id, manifest in sorted(FAMILY_MANIFESTS.items()):
        discipline = str(manifest["discipline"])
        for variant, canonical_size in NATIVE_ASSET_VARIANTS.items():
            generated = _native_family_mesh(family_id, canonical_size, discipline)
            if generated is None:  # pragma: no cover - guarded by FAMILY_MANIFESTS
                continue
            vertices, faces, colors = generated
            vertices, _, _ = _normalize_and_fit(vertices, (1.0, 1.0, 1.0))
            mesh_sha256 = _mesh_sha256(vertices, faces)
            records.append(
                {
                    "asset_id": f"native:{family_id}:{variant}",
                    "geometry_ref": f"mesh:{mesh_sha256}",
                    "families": [family_id],
                    "orientation": variant,
                    "geometry_status": "native_bim_parametric",
                    "family_id": family_id,
                    "name": f"{family_id.replace('-', ' ').title()} / {variant}",
                    "author": "Dajoong",
                    "license": "proprietary",
                    "source_uri": "",
                    "definition": {
                        "schema_version": "dajoong.family-asset.v1",
                        "geometry_status": "native_bim_parametric",
                        "family_id": family_id,
                        "variant": variant,
                        "asset_mesh_sha256": mesh_sha256,
                        "normalized_to_unit_envelope": True,
                        "mesh_vertices": vertices.round(7).tolist(),
                        "mesh_faces": faces.tolist(),
                        "mesh_face_colors": colors.tolist(),
                    },
                }
            )
    return tuple(records)


def family_asset_library(
    catalog_root: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return server-side asset records; geometry stays out of client bundles."""

    root = Path(catalog_root or DEFAULT_CATALOG_ROOT).resolve()
    return _family_asset_library_cached(str(root))


def family_asset_definitions(
    catalog_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Index the lazy asset library by its content-addressed geometry reference."""

    return {
        str(record["geometry_ref"]): record["definition"]
        for record in family_asset_library(catalog_root)
    }


def attach_family_assets(
    graph: dict[str, Any],
    catalog_root: str | Path | None = None,
) -> dict[str, Any]:
    """Attach local geometry and provenance to every supported fixture in-place."""

    root = Path(catalog_root or DEFAULT_CATALOG_ROOT).resolve()
    catalog, entries = _catalog(root)
    graph["family_assets"] = {}
    shared_assets: dict[str, dict[str, Any]] = graph["family_assets"]
    licensed = native = unresolved = ambiguous = 0
    context_index = _asset_context_index(graph)
    for fixture in graph.get("fixtures") or []:
        family_id = str(fixture.get("family_id") or fixture.get("type") or "")
        discipline = str(fixture.get("discipline") or "architectural")
        target_size = tuple(float(value) for value in (fixture.get("size_m") or ())[:3])
        if len(target_size) != 3 or any(value <= 0 for value in target_size):
            fixture["geometry_status"] = "semantic_marker"
            unresolved += 1
            continue

        asset_family = ASSET_FAMILY_BY_BIM_FAMILY.get(family_id)
        candidates = entries.get(asset_family or "", ())
        selection_started = time.perf_counter_ns()
        context = _asset_context(graph, fixture, context_index)
        ranking = _rank_catalog_entries(candidates, fixture, target_size, context)
        selection_elapsed_us = (time.perf_counter_ns() - selection_started) / 1_000
        entry = ranking[0][0] if ranking else None
        score_components = ranking[0][1] if ranking else {"score": 0.0}
        selection_score = score_components["score"]
        if entry is not None:
            selection_margin = selection_score - (
                ranking[1][1]["score"] if len(ranking) > 1 else 0.0
            )
            selection_review_required = selection_score < 0.72 or selection_margin < 0.025
            license_slug = str(entry.get("license") or "").lower()
            if license_slug not in ALLOWED_LICENSES:
                raise ValueError(f"asset {entry.get('uid')} has a non-allowlisted license")
            mesh_path = str((root / str(entry["mesh_path"])).resolve())
            catalog_extents = tuple(
                float(value) for value in (entry.get("normalized_extents") or ())
            )
            if len(catalog_extents) != 3:
                mesh_vertices, _, _ = _load_catalog_mesh(mesh_path)
                catalog_extents = tuple(
                    float(value)
                    for value in (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0))
                )
            swapped = _best_axis_swap(catalog_extents, target_size)
            vertices, faces, colors, mesh_sha256, oriented_extents = _normalized_catalog_asset(
                mesh_path, swapped
            )
            scale = tuple(
                float(value) for value in np.asarray(target_size) / oriented_extents
            )
            geometry_ref = f"mesh:{mesh_sha256}"
            if geometry_ref not in shared_assets:
                shared_assets[geometry_ref] = {
                    "schema_version": "dajoong.family-asset.v1",
                    "geometry_status": "licensed_api_asset",
                    "asset_uid": entry["uid"],
                    "asset_mesh_sha256": mesh_sha256,
                    "normalized_to_unit_envelope": True,
                    "mesh_vertices": vertices.round(7).tolist(),
                    "mesh_faces": faces.tolist(),
                    "mesh_face_colors": colors.tolist(),
                }
            fixture.update(
                {
                    "geometry_status": "licensed_api_asset",
                    "asset_provider": entry.get("provider", catalog.get("provider", "")),
                    "asset_uid": entry["uid"],
                    "asset_name": entry.get("name", ""),
                    "asset_author": entry.get("author", ""),
                    "asset_license": license_slug,
                    "asset_source_uri": entry.get("source_uri", ""),
                    "asset_sha256": entry.get("source_sha256", ""),
                    "asset_mesh_sha256": mesh_sha256,
                    "geometry_ref": geometry_ref,
                    "geometry_scale_xyz": list(target_size),
                    "asset_axis_swapped": swapped,
                    "asset_scale_xyz": [round(value, 8) for value in scale],
                    "asset_fit_policy": "exact_evidence_envelope_xyz_v1",
                    "asset_candidate_count": len(candidates),
                    "asset_selection_score": round(selection_score, 8),
                    "asset_selection_margin": round(selection_margin, 8),
                    "asset_selection_review_required": selection_review_required,
                    "asset_selection_elapsed_us": round(selection_elapsed_us, 3),
                    "asset_selection_policy": "dajoong-context-shape-ranker-v1",
                    "asset_selection_context": {
                        "room_id": context["room_id"],
                        "room_label": context["room_label"],
                        "installation": context["installation"],
                        "nearest_wall_m": (
                            None
                            if context["nearest_wall_m"] is None
                            else round(float(context["nearest_wall_m"]), 4)
                        ),
                        "nearest_wall_angle_deg": (
                            None
                            if context["nearest_wall_angle_deg"] is None
                            else round(float(context["nearest_wall_angle_deg"]), 2)
                        ),
                        "nearby_families": list(context["nearby_families"]),
                    },
                    "asset_selection_components": {
                        key: round(float(value), 8) for key, value in score_components.items()
                    },
                    "asset_selection_alternates": [
                        {
                            "asset_uid": str(candidate.get("uid") or ""),
                            "asset_name": str(candidate.get("name") or ""),
                            "score": round(float(components["score"]), 8),
                        }
                        for candidate, components in ranking[1:4]
                    ],
                }
            )
            if selection_review_required:
                ambiguous += 1
            licensed += 1
            continue

        native_ranking = _native_variant_ranking(target_size)
        variant, variant_score = native_ranking[0]
        native_margin = variant_score - native_ranking[1][1]
        native_review_required = variant_score < 0.72 or native_margin < 0.025
        native_selection_elapsed_us = (time.perf_counter_ns() - selection_started) / 1_000
        generated = _native_family_mesh(
            family_id, NATIVE_ASSET_VARIANTS[variant], discipline
        )
        if generated is None:
            fixture["geometry_status"] = "semantic_marker"
            unresolved += 1
            continue
        vertices, faces, colors = generated
        vertices, _, _ = _normalize_and_fit(vertices, (1.0, 1.0, 1.0))
        mesh_sha256 = _mesh_sha256(vertices, faces)
        geometry_ref = f"mesh:{mesh_sha256}"
        if geometry_ref not in shared_assets:
            shared_assets[geometry_ref] = {
                "schema_version": "dajoong.family-asset.v1",
                "geometry_status": "native_bim_parametric",
                "family_id": family_id,
                "asset_mesh_sha256": mesh_sha256,
                "normalized_to_unit_envelope": True,
                "mesh_vertices": vertices.round(7).tolist(),
                "mesh_faces": faces.tolist(),
                "mesh_face_colors": colors.tolist(),
            }
        fixture.update(
            {
                "geometry_status": "native_bim_parametric",
                "native_generator": "dajoong-evidence-sized-family-v2",
                "native_variant": variant,
                "asset_mesh_sha256": mesh_sha256,
                "geometry_ref": geometry_ref,
                "geometry_scale_xyz": list(target_size),
                "asset_candidate_count": len(native_ranking),
                "asset_selection_score": round(variant_score, 8),
                "asset_selection_margin": round(native_margin, 8),
                "asset_selection_review_required": native_review_required,
                "asset_selection_elapsed_us": round(native_selection_elapsed_us, 3),
                "asset_selection_policy": "dajoong-context-shape-ranker-v1",
                "asset_selection_context": {
                    "room_id": context["room_id"],
                    "room_label": context["room_label"],
                    "installation": context["installation"],
                    "nearest_wall_m": (
                        None
                        if context["nearest_wall_m"] is None
                        else round(float(context["nearest_wall_m"]), 4)
                    ),
                    "nearest_wall_angle_deg": (
                        None
                        if context["nearest_wall_angle_deg"] is None
                        else round(float(context["nearest_wall_angle_deg"]), 2)
                    ),
                    "nearby_families": list(context["nearby_families"]),
                },
                "asset_selection_alternates": [
                    {"native_variant": name, "score": round(score, 8)}
                    for name, score in native_ranking[1:]
                ],
            }
        )
        if native_review_required:
            ambiguous += 1
        native += 1

    audit = audit_fixture_geometry(graph)
    graph["asset_audit"] = audit
    graph.setdefault("pipeline", {})["family_resolution"] = {
        "schema_version": catalog["schema_version"],
        "catalog_entry_count": int(catalog.get("entry_count") or 0),
        "network_on_conversion_path": False,
        "licensed_asset_count": licensed,
        "native_parametric_count": native,
        "ambiguous_asset_choice_count": ambiguous,
        "unresolved_count": unresolved,
        "shared_geometry_count": len(shared_assets),
        "audit_sha256": audit["content_sha256"],
    }
    if unresolved or ambiguous or not audit["passed"]:
        graph.setdefault("confidence", {})["review_required"] = True
        graph.setdefault("pipeline", {})["review_required"] = True
    return graph
