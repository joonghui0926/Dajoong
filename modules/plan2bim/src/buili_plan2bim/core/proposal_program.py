from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .bim_program import (
    BimProgram,
    FixtureInstruction,
    LevelInstruction,
    OpeningInstruction,
    ProgramEvidence,
    RoomInstruction,
    RoomSide,
    WallInstruction,
)
from .cad_families import FAMILY_MANIFESTS, approved_family_asset_sha256
from .model.aec_decode import AecTileProposal, PixelLineProposal
from .model.evidence_coverage import EvidenceCoverageCertificate

Point2D = tuple[float, float]
ReviewState = Literal["accepted", "review_required", "rejected"]


class MetricLevelContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=160)
    level_id: str = Field(min_length=1, max_length=160)
    level_name: str = Field(min_length=1, max_length=300)
    elevation_m: float
    nominal_height_m: float = Field(gt=0)
    pixels_per_meter: float = Field(gt=0)
    tile_origin_px: Point2D = (0.0, 0.0)
    wall_thickness_m: float = Field(default=0.12, gt=0)
    evidence: ProgramEvidence
    independent_evidence_groups: int = Field(default=1, ge=1)


class UnresolvedSymbol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    symbol_class: str
    reason: str
    source_ref_ids: list[str]


class ProgramBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.proposal-program-build.v1"
    program: BimProgram
    unresolved_symbols: list[UnresolvedSymbol]
    independent_evidence_groups: int
    coverage_release_allowed: bool
    coverage_certificate_sha256: str
    auto_accept_eligible: bool


def _cluster_points(
    lines: list[PixelLineProposal], tolerance_px: float
) -> tuple[list[Point2D], list[tuple[int, int]]]:
    points: list[Point2D] = []

    def node_index(point: Point2D) -> int:
        candidates = [
            (math.dist(existing, point), index)
            for index, existing in enumerate(points)
            if math.dist(existing, point) <= tolerance_px
        ]
        if not candidates:
            points.append(point)
            return len(points) - 1
        return min(candidates)[1]

    edges = [(node_index(line.start_px), node_index(line.end_px)) for line in lines]
    return points, edges


def _bounded_faces(
    points: list[Point2D],
    edges: list[tuple[int, int]],
) -> list[list[tuple[int, int, int]]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    edge_index: dict[frozenset[int], int] = {}
    for index, (left, right) in enumerate(edges):
        if left == right:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)
        edge_index[frozenset((left, right))] = index
    for node, neighbors in adjacency.items():
        neighbors.sort(
            key=lambda other: math.atan2(
                points[other][1] - points[node][1], points[other][0] - points[node][0]
            )
        )

    visited: set[tuple[int, int]] = set()
    faces: list[list[tuple[int, int, int]]] = []
    directed_edges = [(left, right) for left, right in edges] + [
        (right, left) for left, right in edges
    ]
    for first_left, first_right in directed_edges:
        if (first_left, first_right) in visited:
            continue
        face: list[tuple[int, int, int]] = []
        left, right = first_left, first_right
        for _ in range(max(4, len(edges) * 2 + 2)):
            if (left, right) in visited:
                break
            visited.add((left, right))
            face.append((left, right, edge_index[frozenset((left, right))]))
            neighbors = adjacency[right]
            if left not in neighbors:
                face = []
                break
            reverse_index = neighbors.index(left)
            next_node = neighbors[(reverse_index - 1) % len(neighbors)]
            left, right = right, next_node
            if (left, right) == (first_left, first_right):
                break
        if not face or (left, right) != (first_left, first_right):
            continue
        polygon = [points[item[0]] for item in face]
        signed_area = (
            sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
            / 2
        )
        if signed_area > 4.0 and len(face) >= 3:
            faces.append(face)
    return faces


def _metric_point(context: MetricLevelContext, point_px: Point2D) -> Point2D:
    return (
        (point_px[0] + context.tile_origin_px[0]) / context.pixels_per_meter,
        (point_px[1] + context.tile_origin_px[1]) / context.pixels_per_meter,
    )


def _projection_on_line(point: Point2D, start: Point2D, end: Point2D) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return 0.0, math.inf
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    nearest = start[0] + fraction * dx, start[1] + fraction * dy
    return fraction, math.dist(point, nearest)


def _point_in_polygon(point: Point2D, polygon: list[Point2D]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


def _fixture_spec(symbol_class: str) -> tuple[str, str, tuple[float, float, float], float]:
    mapping = {
        "light": ("light-fixture", "electrical", (0.6, 0.6, 0.08), 2.85),
        "electrical_panel": ("electrical-panel", "electrical", (0.55, 0.18, 1.0), 0.9),
        "receptacle": ("receptacle", "electrical", (0.08, 0.04, 0.12), 0.3),
        "hvac_terminal": ("supply-air-terminal", "mechanical", (0.6, 0.6, 0.2), 2.75),
        "sprinkler": ("sprinkler", "fire", (0.08, 0.08, 0.08), 2.9),
        "plumbing_fixture": ("generic-plumbing-fixture", "plumbing", (0.7, 0.5, 0.8), 0.0),
        "riser": ("generic-riser", "mechanical", (0.25, 0.25, 3.0), 0.0),
        "closet": ("residential-closet", "architectural", (1.2, 0.6, 2.2), 0.0),
        "electrical_appliance": (
            "residential-electrical-appliance",
            "electrical",
            (0.65, 0.65, 0.9),
            0.0,
        ),
        "toilet": ("residential-toilet", "plumbing", (0.7, 0.4, 0.75), 0.0),
        "sink": ("residential-sink", "plumbing", (0.65, 0.5, 0.9), 0.0),
        "sauna_bench": ("residential-bench", "architectural", (1.2, 0.5, 0.5), 0.0),
        "fireplace": ("residential-fireplace", "architectural", (1.0, 0.45, 1.2), 0.0),
        "bathtub": ("residential-bathtub", "plumbing", (1.7, 0.75, 0.6), 0.0),
        "chimney": ("residential-chimney", "architectural", (0.6, 0.6, 3.0), 0.0),
    }
    return mapping.get(
        symbol_class, (f"unknown-{symbol_class}", "architectural", (0.2, 0.2, 0.2), 0.0)
    )


def _evidence_sized_fixture(
    default_size: tuple[float, float, float],
    bbox_px: tuple[float, float, float, float],
    pixels_per_meter: float,
) -> tuple[tuple[float, float, float], float]:
    """Preserve the observed footprint while bounding obvious annotation spill."""

    observed_width = abs(float(bbox_px[2]) - float(bbox_px[0])) / pixels_per_meter
    observed_depth = abs(float(bbox_px[3]) - float(bbox_px[1])) / pixels_per_meter
    lower_width = max(0.04, default_size[0] * 0.45)
    upper_width = max(lower_width, default_size[0] * 2.2)
    lower_depth = max(0.04, default_size[1] * 0.45)
    upper_depth = max(lower_depth, default_size[1] * 2.2)
    width = min(upper_width, max(lower_width, observed_width))
    depth = min(upper_depth, max(lower_depth, observed_depth))
    yaw_deg = 0.0
    if observed_depth > observed_width * 1.25 and default_size[0] > default_size[1] * 1.15:
        width, depth = depth, width
        yaw_deg = 90.0
    return (width, depth, default_size[2]), yaw_deg


def build_program_from_tile_proposal(
    proposal: AecTileProposal,
    context: MetricLevelContext,
    *,
    coverage_certificate: EvidenceCoverageCertificate | None = None,
    snap_tolerance_px: float = 4.0,
    opening_attachment_tolerance_m: float = 0.35,
) -> ProgramBuildResult:
    if context.evidence.id not in proposal.source_ref_ids:
        raise ValueError("metric context evidence must be present in the tile proposal")
    if coverage_certificate is not None and coverage_certificate.tile_id != proposal.tile_id:
        raise ValueError("coverage certificate must refer to the same tile proposal")
    coverage_release_allowed = bool(
        coverage_certificate is not None and coverage_certificate.release_allowed
    )
    auto_accept = context.independent_evidence_groups >= 2 and coverage_release_allowed
    default_review: ReviewState = "accepted" if auto_accept else "review_required"
    points, edge_nodes = _cluster_points(proposal.wall_segments, snap_tolerance_px)
    faces = _bounded_faces(points, edge_nodes)
    walls: list[WallInstruction] = []
    for index, segment in enumerate(proposal.wall_segments):
        review_state: ReviewState = (
            "accepted" if auto_accept and not segment.review_required else "review_required"
        )
        walls.append(
            WallInstruction(
                id=f"{context.level_id}:wall:{index}",
                level_id=context.level_id,
                start_m=_metric_point(context, points[edge_nodes[index][0]]),
                end_m=_metric_point(context, points[edge_nodes[index][1]]),
                thickness_m=context.wall_thickness_m,
                height_m=context.nominal_height_m,
                wall_type="unknown",
                material="",
                confidence=segment.confidence,
                uncertainty=segment.uncertainty,
                source_ref_ids=segment.source_ref_ids,
                model_version=proposal.model_version,
                review_state=review_state,
            )
        )

    rooms: list[RoomInstruction] = []
    room_polygons: dict[str, list[Point2D]] = {}
    for face_index, face in enumerate(faces):
        room_id = f"{context.level_id}:room:{face_index}"
        sides = []
        polygon = []
        confidences = []
        uncertainties = []
        for left, right, wall_index in face:
            stored_left, stored_right = edge_nodes[wall_index]
            sides.append(
                RoomSide(
                    wall_id=walls[wall_index].id,
                    reversed=not (stored_left == left and stored_right == right),
                )
            )
            polygon.append(_metric_point(context, points[left]))
            confidences.append(walls[wall_index].confidence)
            uncertainties.append(walls[wall_index].uncertainty)
        room_polygons[room_id] = polygon
        rooms.append(
            RoomInstruction(
                id=room_id,
                level_id=context.level_id,
                name=f"Unlabeled room {face_index + 1}",
                occupancy="",
                sides=sides,
                confidence=min(confidences),
                uncertainty=max(uncertainties),
                source_ref_ids=proposal.source_ref_ids,
                model_version=proposal.model_version,
                review_state=default_review,
            )
        )

    if proposal.room_regions:
        rooms = []
        room_polygons = {}
        for index, region in enumerate(proposal.room_regions):
            room_id = f"{context.level_id}:room:{index}"
            polygon = [_metric_point(context, point) for point in region.polygon_px]
            room_polygons[room_id] = polygon
            rooms.append(
                RoomInstruction(
                    id=room_id,
                    level_id=context.level_id,
                    name=region.name,
                    occupancy=region.room_class,
                    polygon_m=polygon,
                    confidence=region.confidence,
                    uncertainty=region.uncertainty,
                    source_ref_ids=region.source_ref_ids,
                    model_version=region.model_version,
                    review_state=(
                        "accepted"
                        if auto_accept and not region.review_required
                        else "review_required"
                    ),
                )
            )

    openings: list[OpeningInstruction] = []
    fixtures: list[FixtureInstruction] = []
    unresolved: list[UnresolvedSymbol] = []
    for symbol in proposal.symbols:
        center_m = _metric_point(context, symbol.center_px)
        if symbol.symbol_class in {"door", "window"}:
            nearest: tuple[float, int, float] | None = None
            for wall_index, wall in enumerate(walls):
                wall_length = math.dist(wall.start_m, wall.end_m)
                if wall_length < 0.3:
                    continue
                fraction, distance = _projection_on_line(center_m, wall.start_m, wall.end_m)
                if 0 <= fraction <= 1 and (nearest is None or distance < nearest[0]):
                    nearest = distance, wall_index, fraction
            if nearest is None or nearest[0] > opening_attachment_tolerance_m:
                unresolved.append(
                    UnresolvedSymbol(
                        proposal_id=symbol.id,
                        symbol_class=symbol.symbol_class,
                        reason="no-wall-within-opening-attachment-tolerance",
                        source_ref_ids=symbol.source_ref_ids,
                    )
                )
                continue
            _, wall_index, fraction = nearest
            wall = walls[wall_index]
            wall_length = math.dist(wall.start_m, wall.end_m)
            observed_width = (
                max(
                    abs(symbol.bbox_px[2] - symbol.bbox_px[0]),
                    abs(symbol.bbox_px[3] - symbol.bbox_px[1]),
                )
                / context.pixels_per_meter
            )
            maximum_width = wall_length - 0.04
            if maximum_width < 0.2:
                unresolved.append(
                    UnresolvedSymbol(
                        proposal_id=symbol.id,
                        symbol_class=symbol.symbol_class,
                        reason="host-wall-too-short-for-opening",
                        source_ref_ids=symbol.source_ref_ids,
                    )
                )
                continue
            width = min(max(0.2, observed_width), maximum_width)
            offset = min(
                max(width / 2, fraction * wall_length),
                wall_length - width / 2,
            )
            openings.append(
                OpeningInstruction(
                    id=f"{context.level_id}:opening:{len(openings)}",
                    level_id=context.level_id,
                    kind=symbol.symbol_class,
                    wall_id=wall.id,
                    offset_m=offset,
                    width_m=width,
                    height_m=2.1 if symbol.symbol_class == "door" else 1.2,
                    sill_height_m=0.0 if symbol.symbol_class == "door" else 0.9,
                    family_id=f"generic-{symbol.symbol_class}",
                    confidence=symbol.confidence,
                    uncertainty=symbol.uncertainty,
                    source_ref_ids=symbol.source_ref_ids,
                    model_version=proposal.model_version,
                    review_state=(
                        "accepted"
                        if auto_accept and not symbol.review_required
                        else "review_required"
                    ),
                )
            )
            continue
        if symbol.symbol_class == "stair":
            unresolved.append(
                UnresolvedSymbol(
                    proposal_id=symbol.id,
                    symbol_class=symbol.symbol_class,
                    reason="adjacent-level-arrival-evidence-required",
                    source_ref_ids=symbol.source_ref_ids,
                )
            )
            continue
        family_id, discipline, default_size_m, base_elevation_m = _fixture_spec(symbol.symbol_class)
        size_m, yaw_deg = _evidence_sized_fixture(
            default_size_m,
            symbol.bbox_px,
            context.pixels_per_meter,
        )
        room_id = next(
            (
                candidate_id
                for candidate_id, polygon in room_polygons.items()
                if _point_in_polygon(center_m, polygon)
            ),
            "",
        )
        fixtures.append(
            FixtureInstruction(
                id=f"{context.level_id}:fixture:{len(fixtures)}",
                level_id=context.level_id,
                family_id=family_id,
                discipline=discipline,  # type: ignore[arg-type]
                center_m=center_m,
                base_elevation_m=base_elevation_m,
                size_m=size_m,
                yaw_deg=yaw_deg,
                room_id=room_id,
                geometry_status=(
                    "approved_family" if family_id in FAMILY_MANIFESTS else "semantic_marker"
                ),
                asset_sha256=approved_family_asset_sha256(family_id),
                confidence=symbol.confidence,
                uncertainty=symbol.uncertainty,
                source_ref_ids=symbol.source_ref_ids,
                model_version=proposal.model_version,
                review_state="review_required",
            )
        )

    level = LevelInstruction(
        id=context.level_id,
        name=context.level_name,
        elevation_m=context.elevation_m,
        nominal_height_m=context.nominal_height_m,
        confidence=min((wall.confidence for wall in walls), default=0.0),
        uncertainty=max((wall.uncertainty for wall in walls), default=1.0),
        source_ref_ids=proposal.source_ref_ids,
        model_version=proposal.model_version,
        review_state=default_review,
    )
    program = BimProgram(
        program_id=f"{context.project_id}:{context.level_id}:{proposal.tile_id}",
        project_id=context.project_id,
        evidence=[context.evidence],
        levels=[level],
        walls=walls,
        rooms=rooms,
        openings=openings,
        fixtures=fixtures,
        compiler_version="dajoong-bim-compiler-0.1",
    ).finalize()
    return ProgramBuildResult(
        program=program,
        unresolved_symbols=unresolved,
        independent_evidence_groups=context.independent_evidence_groups,
        coverage_release_allowed=coverage_release_allowed,
        coverage_certificate_sha256=(
            coverage_certificate.content_sha256 if coverage_certificate is not None else ""
        ),
        auto_accept_eligible=auto_accept and not unresolved,
    )
