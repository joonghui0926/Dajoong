from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .hashing import sha256_json

Severity = Literal["info", "warning", "error"]
Point2D = tuple[float, float]
DISTRIBUTED_FIXTURE_KINDS = {
    "cable_tray",
    "duct_trunk",
    "fire_riser",
    "hydronic_riser",
    "plumbing_riser",
    "railing",
    "stair",
}
INTEGRATED_APPLIANCE_MARKERS = {
    "dishwasher",
    "refrigerator",
    "stove",
    "washing_machine",
    "washing-machine",
    "washingmachine",
    "tumble_dryer",
    "tumble-dryer",
    "tumbledryer",
}
CASEWORK_MARKERS = {"base_cabinet", "base-cabinet", "cabinet", "closet", "casework"}


class PlanGraphViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str
    entity_ids: list[str] = Field(default_factory=list)
    source_ref_ids: list[str] = Field(default_factory=list)
    remediation: str = ""


class PlanGraphCertificate(BaseModel):
    """Content-addressed proof that a PlanGraph passed deterministic release gates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.plan-graph-certificate.v1"
    source_content_sha256: str
    release_allowed: bool
    review_required: bool
    checked_invariants: int = Field(ge=0)
    passed_invariants: int = Field(ge=0)
    violations: list[PlanGraphViolation]
    unsupported_features: list[str] = Field(default_factory=list)
    content_sha256: str


def _id(entity: dict[str, Any], fallback: str) -> str:
    return str(entity.get("id") or entity.get("source_entity_id") or fallback)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _point(value: Any) -> Point2D | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if not _finite(value[0]) or not _finite(value[1]):
        return None
    return float(value[0]), float(value[1])


def _cross(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _polygon_area(points: list[Point2D]) -> float:
    return 0.5 * sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(points, points[1:] + points[:1], strict=True)
    )


def _on_segment(point: Point2D, start: Point2D, end: Point2D, tolerance: float) -> bool:
    if abs(_cross(start, end, point)) > tolerance:
        return False
    return (
        min(start[0], end[0]) - tolerance <= point[0] <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance <= point[1] <= max(start[1], end[1]) + tolerance
    )


def _proper_intersection(
    a: Point2D,
    b: Point2D,
    c: Point2D,
    d: Point2D,
    tolerance: float,
) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)
    return (ab_c > tolerance and ab_d < -tolerance or ab_c < -tolerance and ab_d > tolerance) and (
        cd_a > tolerance and cd_b < -tolerance or cd_a < -tolerance and cd_b > tolerance
    )


def _self_intersects(points: list[Point2D], tolerance: float) -> bool:
    count = len(points)
    for left in range(count):
        a, b = points[left], points[(left + 1) % count]
        for right in range(left + 1, count):
            if right in {left, (left + 1) % count} or (right + 1) % count == left:
                continue
            c, d = points[right], points[(right + 1) % count]
            if _proper_intersection(a, b, c, d, tolerance):
                return True
    return False


def _point_boundary_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq),
    )
    projection = start[0] + fraction * dx, start[1] + fraction * dy
    return math.dist(point, projection)


def _distance_to_polygon(point: Point2D, polygon: list[Point2D]) -> float:
    return min(
        _point_boundary_distance(point, start, end)
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    )


def _project_offset(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-18:
        return 0.0
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length


def _inside_polygon(point: Point2D, polygon: list[Point2D], tolerance: float) -> bool:
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if _point_boundary_distance(point, start, end) <= tolerance:
            return True
    inside = False
    x, y = point
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (start[1] > y) == (end[1] > y):
            continue
        crossing_x = (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
        if x < crossing_x:
            inside = not inside
    return inside


def _source_ids(entity: dict[str, Any]) -> list[str]:
    return [str(item) for item in entity.get("source_ref_ids") or []]


def _duplicates(values: Iterable[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if key and count > 1)


def _fixture_plan_envelope(entity: dict[str, Any]) -> tuple[float, float, float, float] | None:
    center = _point(entity.get("center_m"))
    size = entity.get("size_m") or []
    if center is None or not isinstance(size, (list, tuple)) or len(size) < 2:
        return None
    if not _finite(size[0]) or not _finite(size[1]):
        return None
    width = float(size[0])
    depth = float(size[1])
    yaw = math.radians(float(entity.get("yaw_deg") or 0.0))
    world_width = abs(math.cos(yaw)) * width + abs(math.sin(yaw)) * depth
    world_depth = abs(math.sin(yaw)) * width + abs(math.cos(yaw)) * depth
    return (
        center[0] - world_width / 2,
        center[1] - world_depth / 2,
        center[0] + world_width / 2,
        center[1] + world_depth / 2,
    )


def _fixture_semantic_text(entity: dict[str, Any]) -> str:
    return " ".join(
        str(entity.get(key) or "").lower()
        for key in ("type", "family_id", "source_class")
    ).replace(" ", "_")


class PlanGraphVerifier:
    """Fail-closed geometry and provenance checks for arbitrary drawing outputs.

    The verifier does not guess missing geometry. A model may propose any PlanGraph,
    but only a graph that satisfies these independent invariants can be released as BIM.
    """

    def __init__(self, *, tolerance_m: float = 0.015) -> None:
        if tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive")
        self.tolerance_m = tolerance_m

    def verify(
        self,
        graph: dict[str, Any],
        *,
        permit_review_required: bool = False,
    ) -> PlanGraphCertificate:
        violations: list[PlanGraphViolation] = []
        checked = 0
        unsupported = [str(item) for item in graph.get("unsupported_features") or []]

        def check(condition: bool, violation: PlanGraphViolation | None = None) -> bool:
            nonlocal checked
            checked += 1
            if not condition and violation is not None:
                violations.append(violation)
            return condition

        check(
            graph.get("schema_version") == "buili.plan-graph.v2",
            PlanGraphViolation(
                code="UNSUPPORTED_PLAN_GRAPH_SCHEMA",
                severity="error",
                message="Only the versioned Buili PlanGraph v2 contract is releaseable.",
                remediation="Migrate the extraction result to buili.plan-graph.v2.",
            ),
        )
        rooms = list(graph.get("rooms") or [])
        walls = list(graph.get("walls") or [])
        openings = list(graph.get("openings") or [])
        fixtures = list(graph.get("fixtures") or [])
        routes = list(graph.get("routes") or [])
        vertical_connections = list(graph.get("vertical_connections") or [])
        constraints = list(graph.get("constraints") or [])
        dimensions = list(graph.get("dimensions") or [])
        sources = list(graph.get("sources") or [])
        check(
            bool(sources),
            PlanGraphViolation(
                code="SOURCE_EVIDENCE_MISSING",
                severity="error",
                message="The compiled graph has no source evidence.",
                remediation="Attach the drawing hash, sheet id, and evidence region before export.",
            ),
        )

        source_map = {str(source.get("source_ref_id") or ""): source for source in sources}
        source_duplicates = _duplicates(
            source_map_key
            for source_map_key in [str(source.get("source_ref_id") or "") for source in sources]
        )
        check(
            not source_duplicates,
            PlanGraphViolation(
                code="DUPLICATE_SOURCE_ID",
                severity="error",
                message="Source evidence identifiers are not unique.",
                source_ref_ids=source_duplicates,
            ),
        )
        for source_id, source in source_map.items():
            digest = str(source.get("source_hash") or "")
            check(
                bool(source_id)
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest),
                PlanGraphViolation(
                    code="INVALID_SOURCE_PROVENANCE",
                    severity="error",
                    message="Every source requires a stable id and lowercase SHA-256 digest.",
                    source_ref_ids=[source_id] if source_id else [],
                ),
            )

        entity_groups = (
            ("room", rooms),
            ("wall", walls),
            ("opening", openings),
            ("fixture", fixtures),
            ("route", routes),
            ("vertical", vertical_connections),
            ("constraint", constraints),
            ("dimension", dimensions),
        )
        all_ids: list[str] = []
        for prefix, entities in entity_groups:
            all_ids.extend(
                _id(entity, f"{prefix}-{index}") for index, entity in enumerate(entities)
            )
        duplicate_entities = _duplicates(all_ids)
        check(
            not duplicate_entities,
            PlanGraphViolation(
                code="DUPLICATE_ENTITY_ID",
                severity="error",
                message="BIM entity identifiers must be globally unique.",
                entity_ids=duplicate_entities,
            ),
        )

        room_polygons: dict[str, list[Point2D]] = {}
        for index, room in enumerate(rooms):
            room_id = _id(room, f"room-{index}")
            raw_polygon = room.get("polygon") or []
            polygon = [point for value in raw_polygon if (point := _point(value)) is not None]
            valid_points = len(polygon) == len(raw_polygon) and len(polygon) >= 3
            check(
                valid_points,
                PlanGraphViolation(
                    code="INVALID_ROOM_POLYGON",
                    severity="error",
                    message="A room boundary must contain at least three finite metric vertices.",
                    entity_ids=[room_id],
                    source_ref_ids=_source_ids(room),
                ),
            )
            if not valid_points:
                continue
            room_polygons[room_id] = polygon
            check(
                abs(_polygon_area(polygon)) > self.tolerance_m**2,
                PlanGraphViolation(
                    code="ZERO_AREA_ROOM",
                    severity="error",
                    message="A room polygon has no usable enclosed area.",
                    entity_ids=[room_id],
                    source_ref_ids=_source_ids(room),
                ),
            )
            check(
                not _self_intersects(polygon, self.tolerance_m**2),
                PlanGraphViolation(
                    code="SELF_INTERSECTING_ROOM",
                    severity="error",
                    message="A room cycle crosses itself and cannot define a BIM space.",
                    entity_ids=[room_id],
                    source_ref_ids=_source_ids(room),
                ),
            )
            self._check_evidence(room, room_id, source_map, check)

        wall_map: dict[str, tuple[Point2D, Point2D, float]] = {}
        for index, wall in enumerate(walls):
            wall_id = _id(wall, f"wall-{index}")
            start = _point(wall.get("from") or wall.get("from_"))
            end = _point(wall.get("to"))
            thickness = wall.get("thickness_m")
            height = wall.get("height_m")
            valid = (
                start is not None
                and end is not None
                and math.dist(start, end) > self.tolerance_m
                and _finite(thickness)
                and float(thickness) > 0
                and _finite(height)
                and float(height) > 0
            )
            check(
                valid,
                PlanGraphViolation(
                    code="INVALID_WALL_GEOMETRY",
                    severity="error",
                    message="A wall requires a non-zero metric segment, height, and thickness.",
                    entity_ids=[wall_id],
                    source_ref_ids=_source_ids(wall),
                ),
            )
            if valid and start is not None and end is not None:
                wall_map[wall_id] = start, end, float(thickness)
            self._check_evidence(wall, wall_id, source_map, check)

        opening_intervals: dict[str, list[tuple[float, float, str]]] = {}
        for index, opening in enumerate(openings):
            opening_id = _id(opening, f"opening-{index}")
            wall_id = str(opening.get("wall_id") or "")
            width = opening.get("width_m")
            height = opening.get("height_m")
            x_m = opening.get("x_m")
            sill = opening.get("sill_height_m", 0.0)
            wall_geometry = wall_map.get(wall_id)
            center = _point(opening.get("center_m"))
            valid_dimensions = (
                _finite(width)
                and float(width) > 0
                and _finite(height)
                and float(height) > 0
                and _finite(sill)
                and float(sill) >= 0
                and (_finite(x_m) or center is not None)
            )
            check(
                wall_geometry is not None,
                PlanGraphViolation(
                    code="ORPHAN_OPENING",
                    severity="error",
                    message="A door or window references no compiled wall.",
                    entity_ids=[opening_id, wall_id] if wall_id else [opening_id],
                    source_ref_ids=_source_ids(opening),
                ),
            )
            check(
                valid_dimensions,
                PlanGraphViolation(
                    code="INVALID_OPENING_DIMENSIONS",
                    severity="error",
                    message=(
                        "An opening requires finite positive width and height and a wall offset."
                    ),
                    entity_ids=[opening_id],
                    source_ref_ids=_source_ids(opening),
                ),
            )
            opening_type = str(opening.get("type") or "").lower()
            if opening_type == "door":
                operation_type = str(opening.get("operation_type") or "unknown")
                handing = str(opening.get("handing") or "unknown")
                swing_side = str(opening.get("swing_side") or "unknown")
                resolved_operation = (
                    operation_type
                    in {"single_swing", "double_swing", "sliding", "folding", "fixed"}
                    and handing in {"start", "end", "double"}
                    and swing_side in {"positive", "negative", "both", "none"}
                )
                check(
                    resolved_operation,
                    PlanGraphViolation(
                        code="UNRESOLVED_DOOR_OPERATION",
                        severity="error",
                        message=(
                            "A door requires an explicit operation, hinge handing, and "
                            "host-wall-relative swing side before release."
                        ),
                        entity_ids=[opening_id],
                        source_ref_ids=_source_ids(opening),
                        remediation=(
                            "Confirm the door operation in Studio or provide calibrated "
                            "door-swing evidence from the source drawing."
                        ),
                    ),
                )
            if wall_geometry is not None and valid_dimensions:
                start, end, thickness = wall_geometry
                wall_length = math.dist(start, end)
                half_width = float(width) / 2
                center_offset = float(x_m) if _finite(x_m) else _project_offset(center, start, end)
                fits = (
                    half_width - self.tolerance_m
                    <= center_offset
                    <= wall_length - half_width + self.tolerance_m
                )
                check(
                    fits,
                    PlanGraphViolation(
                        code="OPENING_OUTSIDE_WALL",
                        severity="error",
                        message="An opening extends beyond its supporting wall segment.",
                        entity_ids=[opening_id, wall_id],
                        source_ref_ids=_source_ids(opening),
                    ),
                )
                if center is not None:
                    check(
                        _point_boundary_distance(center, start, end)
                        <= max(thickness, self.tolerance_m) * 1.5,
                        PlanGraphViolation(
                            code="OPENING_CENTER_OFF_WALL",
                            severity="error",
                            message="The opening center contradicts the referenced wall geometry.",
                            entity_ids=[opening_id, wall_id],
                            source_ref_ids=_source_ids(opening),
                        ),
                    )
                opening_intervals.setdefault(wall_id, []).append(
                    (center_offset - half_width, center_offset + half_width, opening_id)
                )
            self._check_evidence(opening, opening_id, source_map, check)

        for wall_id, intervals in opening_intervals.items():
            ordered = sorted(intervals)
            for left, right in zip(ordered, ordered[1:], strict=False):
                check(
                    left[1] <= right[0] + self.tolerance_m,
                    PlanGraphViolation(
                        code="OVERLAPPING_OPENINGS",
                        severity="error",
                        message="Two openings occupy the same span of a wall.",
                        entity_ids=[left[2], right[2], wall_id],
                    ),
                )

        for index, constraint in enumerate(constraints):
            constraint_id = _id(constraint, f"constraint-{index}")
            constraint_type = str(constraint.get("type") or "")
            references = list(constraint.get("references") or [])
            valid_references = (
                constraint_type == "coincident"
                and len(references) >= 2
                and all(
                    str(reference.get("collection") or "") == "walls"
                    and str(reference.get("entity_id") or "") in wall_map
                    and str(reference.get("handle") or "") in {"from", "to"}
                    for reference in references
                )
            )
            check(
                valid_references,
                PlanGraphViolation(
                    code="INVALID_GEOMETRIC_CONSTRAINT",
                    severity="error",
                    message=(
                        "A coincident constraint requires at least two valid wall "
                        "endpoint references."
                    ),
                    entity_ids=[constraint_id],
                ),
            )
            if valid_references:
                points = [
                    wall_map[str(reference["entity_id"])][
                        0 if str(reference["handle"]) == "from" else 1
                    ]
                    for reference in references
                ]
                check(
                    all(math.dist(points[0], point) <= self.tolerance_m for point in points[1:]),
                    PlanGraphViolation(
                        code="BROKEN_COINCIDENT_CONSTRAINT",
                        severity="error",
                        message="Constrained wall endpoints no longer share one metric coordinate.",
                        entity_ids=[constraint_id, *[str(ref["entity_id"]) for ref in references]],
                    ),
                )

        for index, dimension in enumerate(dimensions):
            dimension_id = _id(dimension, f"dimension-{index}")
            start = _point(dimension.get("from"))
            end = _point(dimension.get("to"))
            check(
                start is not None
                and end is not None
                and math.dist(start, end) > self.tolerance_m,
                PlanGraphViolation(
                    code="INVALID_DIMENSION_GEOMETRY",
                    severity="error",
                    message="A dimension requires two distinct finite metric points.",
                    entity_ids=[dimension_id],
                ),
            )

        room_ids = set(room_polygons)
        for index, fixture in enumerate(fixtures):
            fixture_id = _id(fixture, f"fixture-{index}")
            room_id = str(fixture.get("room_id") or "")
            center = _point(fixture.get("center_m"))
            check(
                not room_id or room_id in room_ids,
                PlanGraphViolation(
                    code="ORPHAN_FIXTURE_ROOM",
                    severity="error",
                    message="A fixture references a room that is not present.",
                    entity_ids=[fixture_id, room_id] if room_id else [fixture_id],
                    source_ref_ids=_source_ids(fixture),
                ),
            )
            if room_id in room_polygons and center is not None:
                polygon = room_polygons[room_id]
                inside = _inside_polygon(center, polygon, self.tolerance_m)
                distance = 0.0 if inside else _distance_to_polygon(center, polygon)
                fixture_type = str(fixture.get("type") or "")
                severity: Severity = (
                    "warning"
                    if fixture_type in DISTRIBUTED_FIXTURE_KINDS or distance <= 0.30
                    else "error"
                )
                check(
                    inside,
                    PlanGraphViolation(
                        code="FIXTURE_OUTSIDE_ASSIGNED_ROOM",
                        severity=severity,
                        message=(
                            "A fixture anchor lies outside its assigned room cycle; "
                            "distributed systems may cross room boundaries but require review."
                        ),
                        entity_ids=[fixture_id, room_id],
                        source_ref_ids=_source_ids(fixture),
                    ),
                )
            required = fixture.get("required_count", 0)
            observed = fixture.get("observed_count", 0)
            check(
                isinstance(required, int)
                and required >= 0
                and isinstance(observed, int)
                and observed >= 0,
                PlanGraphViolation(
                    code="INVALID_FIXTURE_COUNT",
                    severity="error",
                    message=(
                        "Fixture requirement and observation counts must be non-negative integers."
                    ),
                    entity_ids=[fixture_id],
                ),
            )
            self._check_evidence(fixture, fixture_id, source_map, check)

        # Integrated appliances replace, rather than occupy the same solid as,
        # floor-mounted casework.  Countertop sinks and cooktops are excluded:
        # their work plane is intentionally above a cabinet volume.
        for left_index, left in enumerate(fixtures):
            left_text = _fixture_semantic_text(left)
            left_is_casework = any(marker in left_text for marker in CASEWORK_MARKERS)
            for right in fixtures[left_index + 1 :]:
                right_text = _fixture_semantic_text(right)
                right_is_casework = any(
                    marker in right_text for marker in CASEWORK_MARKERS
                )
                left_is_appliance = any(
                    marker in left_text for marker in INTEGRATED_APPLIANCE_MARKERS
                )
                right_is_appliance = any(
                    marker in right_text for marker in INTEGRATED_APPLIANCE_MARKERS
                )
                if left_is_casework and right_is_appliance:
                    casework, appliance = left, right
                elif right_is_casework and left_is_appliance:
                    casework, appliance = right, left
                else:
                    continue
                if str(left.get("level_id")) != str(right.get("level_id")):
                    continue
                left_box = _fixture_plan_envelope(casework)
                if left_box is None:
                    continue
                left_base = float(casework.get("base_elevation_m") or 0.0)
                left_size = casework.get("size_m") or []
                if len(left_size) < 3 or not _finite(left_size[2]):
                    continue
                left_top = left_base + float(left_size[2])
                appliance_box = _fixture_plan_envelope(appliance)
                if appliance_box is None:
                    continue
                appliance_size = appliance.get("size_m") or []
                if len(appliance_size) < 3 or not _finite(appliance_size[2]):
                    continue
                appliance_base = float(appliance.get("base_elevation_m") or 0.0)
                appliance_top = appliance_base + float(appliance_size[2])
                vertical_overlap = min(left_top, appliance_top) - max(
                    left_base, appliance_base
                )
                intersection_width = max(
                    0.0,
                    min(left_box[2], appliance_box[2])
                    - max(left_box[0], appliance_box[0]),
                )
                intersection_depth = max(
                    0.0,
                    min(left_box[3], appliance_box[3])
                    - max(left_box[1], appliance_box[1]),
                )
                appliance_area = max(
                    1e-6,
                    (appliance_box[2] - appliance_box[0])
                    * (appliance_box[3] - appliance_box[1]),
                )
                occupied_fraction = intersection_width * intersection_depth / appliance_area
                check(
                    not (vertical_overlap > self.tolerance_m and occupied_fraction >= 0.72),
                    PlanGraphViolation(
                        code="INTEGRATED_APPLIANCE_CASEWORK_COLLISION",
                        severity="error",
                        message=(
                            "An integrated appliance occupies the same solid volume as casework."
                        ),
                        entity_ids=[_id(casework, "casework"), _id(appliance, "appliance")],
                        source_ref_ids=sorted(
                            set(_source_ids(casework) + _source_ids(appliance))
                        ),
                        remediation=(
                            "Split the cabinet run at the appliance bay before export."
                        ),
                    ),
                )

        for index, route in enumerate(routes):
            route_id = _id(route, f"route-{index}")
            raw_points = route.get("points_m") or []
            points_valid = len(raw_points) >= 2 and all(
                isinstance(point, (list, tuple))
                and len(point) == 3
                and all(_finite(coordinate) for coordinate in point)
                for point in raw_points
            )
            section = route.get("section_m") or []
            section_valid = (
                isinstance(section, (list, tuple))
                and len(section) == 2
                and all(_finite(value) and float(value) > 0 for value in section)
            )
            check(
                points_valid and section_valid,
                PlanGraphViolation(
                    code="INVALID_SYSTEM_ROUTE",
                    severity="error",
                    message=(
                        "A system route requires two or more finite 3D points "
                        "and a positive section."
                    ),
                    entity_ids=[route_id],
                    source_ref_ids=_source_ids(route),
                ),
            )
            if points_valid:
                zero_segments = any(
                    math.dist(left, right) <= self.tolerance_m
                    for left, right in zip(raw_points, raw_points[1:], strict=False)
                )
                check(
                    not zero_segments,
                    PlanGraphViolation(
                        code="ZERO_LENGTH_ROUTE_SEGMENT",
                        severity="error",
                        message="A MEP route contains a zero-length segment.",
                        entity_ids=[route_id],
                        source_ref_ids=_source_ids(route),
                    ),
                )
            self._check_evidence(route, route_id, source_map, check)

        self._verify_level_links(graph, source_map, check)
        if unsupported:
            violations.append(
                PlanGraphViolation(
                    code="UNSUPPORTED_DRAWING_FEATURE",
                    severity="error",
                    message=(
                        "The drawing contains features outside the production compiler contract."
                    ),
                    entity_ids=unsupported,
                    remediation=(
                        "Route only the listed regions to review; do not publish guessed geometry."
                    ),
                )
            )

        qualification = graph.get("qualification")
        if isinstance(qualification, dict):
            check(
                bool(qualification.get("production_release_eligible", False)),
                PlanGraphViolation(
                    code="MODEL_NOT_QUALIFIED_FOR_DRAWING_CLASS",
                    severity="error",
                    message=(
                        "The exact model pair has not passed the sealed benchmark gate "
                        "for this drawing complexity class and required BIM claims."
                    ),
                    remediation=(
                        "Complete review or qualify the exact model artifacts on the "
                        "required sealed benchmark cohort before release."
                    ),
                ),
            )

        declared_review = bool(
            (graph.get("confidence") or {}).get("review_required")
            or (graph.get("pipeline") or {}).get("review_required")
        )
        review_required = (
            declared_review
            or bool(unsupported)
            or any(violation.severity == "error" for violation in violations)
        )
        release_allowed = not any(violation.severity == "error" for violation in violations)
        if review_required and not permit_review_required:
            release_allowed = False
        source_hash = sha256_json(graph)
        payload = {
            "source_content_sha256": source_hash,
            "release_allowed": release_allowed,
            "review_required": review_required,
            "checked_invariants": checked,
            "violations": [violation.model_dump(mode="json") for violation in violations],
            "unsupported_features": unsupported,
        }
        return PlanGraphCertificate(
            source_content_sha256=source_hash,
            release_allowed=release_allowed,
            review_required=review_required,
            checked_invariants=checked,
            passed_invariants=max(0, checked - len(violations)),
            violations=violations,
            unsupported_features=unsupported,
            content_sha256=sha256_json(payload),
        )

    def _check_evidence(
        self,
        entity: dict[str, Any],
        entity_id: str,
        source_map: dict[str, dict[str, Any]],
        check: Any,
    ) -> None:
        refs = _source_ids(entity)
        confidence = entity.get("confidence")
        check(
            bool(refs) and not (set(refs) - set(source_map)),
            PlanGraphViolation(
                code="ENTITY_EVIDENCE_MISSING",
                severity="error",
                message="Every persisted BIM entity must resolve to drawing evidence.",
                entity_ids=[entity_id],
                source_ref_ids=refs,
            ),
        )
        check(
            _finite(confidence) and 0 <= float(confidence) <= 1,
            PlanGraphViolation(
                code="ENTITY_CONFIDENCE_MISSING",
                severity="error",
                message="Every persisted BIM entity must retain calibrated confidence.",
                entity_ids=[entity_id],
                source_ref_ids=refs,
            ),
        )

    def _verify_level_links(
        self,
        graph: dict[str, Any],
        source_map: dict[str, dict[str, Any]],
        check: Any,
    ) -> None:
        levels = list(graph.get("levels") or [])
        if not levels:
            return
        level_ids = {str(level.get("id") or "") for level in levels}
        elevations: dict[str, float] = {}
        nominal_heights: dict[str, float] = {}
        for level in levels:
            level_id = str(level.get("id") or "")
            valid = bool(level_id) and _finite(level.get("elevation_m"))
            check(
                valid,
                PlanGraphViolation(
                    code="INVALID_LEVEL",
                    severity="error",
                    message="Every level requires an id and finite elevation.",
                    entity_ids=[level_id] if level_id else [],
                ),
            )
            if valid:
                elevations[level_id] = float(level["elevation_m"])
            height = level.get("nominal_height_m")
            valid_height = _finite(height) and float(height) > 0
            check(
                valid_height,
                PlanGraphViolation(
                    code="INVALID_LEVEL_HEIGHT",
                    severity="error",
                    message="Every level requires a finite positive nominal height.",
                    entity_ids=[level_id] if level_id else [],
                    source_ref_ids=_source_ids(level),
                    remediation="Set a measured floor-to-ceiling height for this level.",
                ),
            )
            if valid_height and level_id:
                nominal_heights[level_id] = float(height)
            self._check_evidence(level, level_id or "level", source_map, check)
        check(
            len(level_ids) == len(levels) and len(set(elevations.values())) == len(elevations),
            PlanGraphViolation(
                code="DUPLICATE_LEVEL",
                severity="error",
                message="Level identifiers and elevations must be unique.",
                entity_ids=sorted(level_ids),
            ),
        )
        if len(levels) > 1:
            for group_name in ("rooms", "walls", "openings", "fixtures", "routes"):
                for index, entity in enumerate(graph.get(group_name) or []):
                    entity_id = _id(entity, f"{group_name}-{index}")
                    level_id = str(entity.get("level_id") or "")
                    check(
                        level_id in level_ids,
                        PlanGraphViolation(
                            code="ENTITY_LEVEL_MISSING",
                            severity="error",
                            message=(
                                "Every entity in a multi-level drawing set must name its level."
                            ),
                            entity_ids=[entity_id, level_id] if level_id else [entity_id],
                            source_ref_ids=_source_ids(entity),
                        ),
                    )
        ordered_ids = [
            level_id for level_id, _ in sorted(elevations.items(), key=lambda item: item[1])
        ]
        next_level_by_id = {
            lower: upper
            for lower, upper in zip(ordered_ids, ordered_ids[1:], strict=False)
        }
        for lower, upper in next_level_by_id.items():
            rise = elevations[upper] - elevations[lower]
            nominal_height = nominal_heights.get(lower)
            if nominal_height is not None:
                check(
                    nominal_height <= rise + self.tolerance_m,
                    PlanGraphViolation(
                        code="LEVEL_VOLUME_OVERLAP",
                        severity="error",
                        message=(
                            "A level's nominal height crosses the elevation of the next level."
                        ),
                        entity_ids=[lower, upper],
                        remediation=(
                            "Correct the level elevation or nominal height before building export."
                        ),
                    ),
                )
            for wall in graph.get("walls") or []:
                if str(wall.get("level_id") or "") != lower:
                    continue
                wall_height = wall.get("height_m")
                if not _finite(wall_height):
                    continue
                wall_id = _id(wall, "wall")
                check(
                    float(wall_height) <= rise + self.tolerance_m,
                    PlanGraphViolation(
                        code="WALL_CROSSES_NEXT_LEVEL",
                        severity="error",
                        message="A wall extends through the next building level.",
                        entity_ids=[wall_id, lower, upper],
                        source_ref_ids=_source_ids(wall),
                        remediation=(
                            "Reduce the wall height or correct the adjacent level elevation."
                        ),
                    ),
                )
            for fixture in graph.get("fixtures") or []:
                if str(fixture.get("level_id") or "") != lower:
                    continue
                fixture_size = fixture.get("size_m") or []
                if (
                    not isinstance(fixture_size, (list, tuple))
                    or len(fixture_size) < 3
                    or not _finite(fixture_size[2])
                    or not _finite(fixture.get("base_elevation_m", 0.0))
                ):
                    continue
                fixture_top = float(fixture.get("base_elevation_m", 0.0)) + float(
                    fixture_size[2]
                )
                fixture_id = _id(fixture, "fixture")
                check(
                    fixture_top <= rise + self.tolerance_m,
                    PlanGraphViolation(
                        code="FIXTURE_CROSSES_NEXT_LEVEL",
                        severity="error",
                        message="An installed object extends through the next building level.",
                        entity_ids=[fixture_id, lower, upper],
                        source_ref_ids=_source_ids(fixture),
                        remediation=(
                            "Correct the object's base elevation or height before building export."
                        ),
                    ),
                )
        adjacency = {
            frozenset((left, right))
            for left, right in zip(ordered_ids, ordered_ids[1:], strict=False)
        }
        level_points: dict[str, list[Point2D]] = {level_id: [] for level_id in level_ids}
        for wall in graph.get("walls") or []:
            level_id = str(wall.get("level_id") or "")
            for value in (wall.get("from"), wall.get("to")):
                if level_id in level_points and (point := _point(value)) is not None:
                    level_points[level_id].append(point)
        for room in graph.get("rooms") or []:
            level_id = str(room.get("level_id") or "")
            for value in room.get("polygon") or []:
                if level_id in level_points and (point := _point(value)) is not None:
                    level_points[level_id].append(point)
        level_bounds = {
            level_id: (
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            )
            for level_id, points in level_points.items()
            if points
        }
        vertical_connections = list(graph.get("vertical_connections") or [])
        connection_signatures: dict[
            tuple[str, str, str, int, int, int, int, int], list[str]
        ] = {}
        shaft_connections: dict[str, list[dict[str, Any]]] = {}
        for connection in vertical_connections:
            connection_id = _id(connection, "vertical-connection")
            from_level = str(connection.get("from_level_id") or "")
            to_level = str(connection.get("to_level_id") or "")
            valid_levels = (
                from_level in level_ids and to_level in level_ids and from_level != to_level
            )
            check(
                valid_levels,
                PlanGraphViolation(
                    code="IMPOSSIBLE_VERTICAL_CONNECTION",
                    severity="error",
                    message=(
                        "A stair, ramp, lift, or riser targets a nonexistent or identical level."
                    ),
                    entity_ids=[connection_id, from_level, to_level],
                    source_ref_ids=_source_ids(connection),
                ),
            )
            if valid_levels:
                check(
                    elevations[to_level] > elevations[from_level],
                    PlanGraphViolation(
                        code="REVERSED_VERTICAL_CONNECTION",
                        severity="error",
                        message="A vertical connection must run from a lower to a higher level.",
                        entity_ids=[connection_id, from_level, to_level],
                        source_ref_ids=_source_ids(connection),
                        remediation=(
                            "Swap the connection endpoints or correct the level elevations."
                        ),
                    ),
                )
            kind = str(connection.get("type") or connection.get("kind") or "stair")
            if valid_levels and kind in {"stair", "ramp", "escalator"}:
                check(
                    frozenset((from_level, to_level)) in adjacency,
                    PlanGraphViolation(
                        code="SKIPPED_LEVEL_CONNECTION",
                        severity="error",
                        message="A stair-like connection may only join adjacent levels.",
                        entity_ids=[connection_id, from_level, to_level],
                        source_ref_ids=_source_ids(connection),
                    ),
                )
            center = _point(connection.get("center_m"))
            footprint = connection.get("footprint_m") or []
            valid_geometry = (
                center is not None
                and isinstance(footprint, (list, tuple))
                and len(footprint) == 2
                and all(_finite(value) and float(value) > 0 for value in footprint)
            )
            check(
                valid_geometry,
                PlanGraphViolation(
                    code="INVALID_VERTICAL_CONNECTION_GEOMETRY",
                    severity="error",
                    message=(
                        "A vertical connection requires a finite center and positive footprint."
                    ),
                    entity_ids=[connection_id],
                    source_ref_ids=_source_ids(connection),
                ),
            )
            if valid_geometry and center is not None:
                signature = (
                    kind,
                    from_level,
                    to_level,
                    round(center[0] / self.tolerance_m),
                    round(center[1] / self.tolerance_m),
                    round(float(footprint[0]) / self.tolerance_m),
                    round(float(footprint[1]) / self.tolerance_m),
                    round(float(connection.get("yaw_deg") or 0.0) / 0.5),
                )
                connection_signatures.setdefault(signature, []).append(connection_id)
            shaft_id = str(connection.get("shaft_id") or "")
            if shaft_id:
                shaft_connections.setdefault(shaft_id, []).append(connection)
            if valid_levels and valid_geometry and center is not None:
                half_width = float(footprint[0]) / 2
                half_depth = float(footprint[1]) / 2
                for level_id in (from_level, to_level):
                    bounds = level_bounds.get(level_id)
                    if bounds is None:
                        continue
                    inside = (
                        center[0] - half_width >= bounds[0] - self.tolerance_m
                        and center[1] - half_depth >= bounds[1] - self.tolerance_m
                        and center[0] + half_width <= bounds[2] + self.tolerance_m
                        and center[1] + half_depth <= bounds[3] + self.tolerance_m
                    )
                    check(
                        inside,
                        PlanGraphViolation(
                            code="VERTICAL_CONNECTION_OUTSIDE_LEVEL",
                            severity="error",
                            message=(
                                "A stair, lift, or riser footprint falls outside a connected level."
                            ),
                            entity_ids=[connection_id, level_id],
                            source_ref_ids=_source_ids(connection),
                        ),
                    )
            self._check_evidence(connection, connection_id, source_map, check)

        for duplicate_ids in connection_signatures.values():
            check(
                len(duplicate_ids) == 1,
                PlanGraphViolation(
                    code="DUPLICATE_VERTICAL_CONNECTION",
                    severity="error",
                    message="Two vertical connections occupy the same level pair and footprint.",
                    entity_ids=duplicate_ids,
                    remediation="Keep one connection or move it to its measured shaft location.",
                ),
            )

        ordered_index = {level_id: index for index, level_id in enumerate(ordered_ids)}
        for shaft_id, connections in shaft_connections.items():
            centers = [
                point
                for connection in connections
                if (point := _point(connection.get("center_m"))) is not None
            ]
            footprints = [
                tuple(float(value) for value in connection.get("footprint_m")[:2])
                for connection in connections
                if isinstance(connection.get("footprint_m"), (list, tuple))
                and len(connection.get("footprint_m")) == 2
                and all(_finite(value) for value in connection.get("footprint_m"))
            ]
            connection_ids = [_id(connection, "vertical-connection") for connection in connections]
            check(
                len(centers) == len(connections)
                and all(
                    math.dist(centers[0], center) <= self.tolerance_m
                    for center in centers[1:]
                ),
                PlanGraphViolation(
                    code="MISALIGNED_VERTICAL_SHAFT",
                    severity="error",
                    message="Segments assigned to one shaft do not share a vertical centerline.",
                    entity_ids=[shaft_id, *connection_ids],
                    remediation="Align the shaft segments in plan before building export.",
                ),
            )
            check(
                len(footprints) == len(connections)
                and all(
                    max(
                        abs(left - right)
                        for left, right in zip(footprints[0], footprint, strict=True)
                    )
                    <= self.tolerance_m
                    for footprint in footprints[1:]
                ),
                PlanGraphViolation(
                    code="INCONSISTENT_VERTICAL_SHAFT_FOOTPRINT",
                    severity="error",
                    message="Segments assigned to one shaft use inconsistent footprints.",
                    entity_ids=[shaft_id, *connection_ids],
                    remediation="Use one measured footprint for the complete shaft run.",
                ),
            )
            covered_steps: set[tuple[int, int]] = set()
            touched_indices: set[int] = set()
            for connection in connections:
                start = ordered_index.get(str(connection.get("from_level_id") or ""))
                end = ordered_index.get(str(connection.get("to_level_id") or ""))
                if start is None or end is None or end <= start:
                    continue
                touched_indices.update(range(start, end + 1))
                covered_steps.update((index, index + 1) for index in range(start, end))
            if touched_indices:
                expected_steps = {
                    (index, index + 1)
                    for index in range(min(touched_indices), max(touched_indices))
                }
                check(
                    expected_steps <= covered_steps,
                    PlanGraphViolation(
                        code="DISCONTINUOUS_VERTICAL_SHAFT",
                        severity="error",
                        message="A named vertical shaft has a missing segment between levels.",
                        entity_ids=[shaft_id, *connection_ids],
                        remediation="Add the measured missing segment or split unrelated shafts.",
                    ),
                )


def assert_releaseable(graph: dict[str, Any]) -> PlanGraphCertificate:
    certificate = PlanGraphVerifier().verify(graph)
    if not certificate.release_allowed:
        codes = (
            ", ".join(violation.code for violation in certificate.violations) or "REVIEW_REQUIRED"
        )
        raise ValueError(f"PlanGraph release blocked by deterministic verifier: {codes}")
    return certificate
