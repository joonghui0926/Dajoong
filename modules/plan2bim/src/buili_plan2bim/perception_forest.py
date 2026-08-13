"""Global-first evidence graph for floor-plan reconstruction.

Method v1 promoted isolated dense-pixel classes directly into BIM entities.  This
module makes the full drawing topology the primary representation.  Perception
experts contribute evidence nodes; deterministic relations expose whether an
opening has a host wall and whether an object belongs to a recovered interior.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import sha256_json
from .core.model.aec_decode import (
    AecTileProposal,
    PixelLineProposal,
    PixelRoomProposal,
    PixelSymbolProposal,
)
from .semantic_recognition import SemanticRecognitionResult

EvidenceKind = Literal["wall", "room", "opening", "fixture"]
RelationKind = Literal[
    "host_candidate",
    "inside_room",
    "joins_wall",
    "bounds_room",
]


class SpatialEvidenceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    label: str
    expert_id: str
    confidence: float = Field(ge=0, le=1)
    review_required: bool
    promoted: bool
    bbox_px: tuple[float, float, float, float] | None = None
    line_px: tuple[float, float, float, float] | None = None
    polygon_px: list[tuple[float, float]] | None = None
    thickness_px: float | None = Field(default=None, gt=0)
    compiled_entities: list[CompiledEntityEvidence] = Field(default_factory=list)
    asset_resolution: AssetResolutionEvidence | None = None


class CompiledEntityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    collection: Literal["walls", "rooms", "openings", "fixtures"]
    review_state: str
    family_id: str = ""
    host_wall_id: str = ""
    room_id: str = ""


class AssetResolutionEvidence(BaseModel):
    """Bounded asset audit metadata; mesh payloads deliberately stay server-side."""

    model_config = ConfigDict(extra="forbid")

    geometry_status: str
    family_id: str
    geometry_ref: str = ""
    asset_uid: str = ""
    asset_provider: str = ""
    asset_license: str = ""
    asset_name: str = ""
    selection_policy: str = ""
    selection_score: float | None = None
    selection_margin: float | None = None
    selection_review_required: bool = True
    candidate_count: int = Field(default=0, ge=0)
    selection_elapsed_us: float = Field(default=0.0, ge=0)
    selection_context: dict[str, Any] = Field(default_factory=dict)
    selection_components: dict[str, float] = Field(default_factory=dict)
    selection_alternates: list[dict[str, Any]] = Field(default_factory=list)


class SpatialEvidenceRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: RelationKind
    source_id: str
    target_id: str
    confidence: float = Field(ge=0, le=1)
    distance_px: float = Field(ge=0)


class EvidenceCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_sheet_context: bool
    expert_count: int = Field(ge=1)
    node_count_by_kind: dict[str, int]
    promoted_count_by_kind: dict[str, int]
    review_count_by_kind: dict[str, int]
    unhosted_opening_count: int = Field(ge=0)
    unassigned_fixture_count: int = Field(ge=0)
    unclassified_room_count: int = Field(ge=0)
    independent_expert_consensus_available: bool
    compiled_count_by_kind: dict[str, int] = Field(default_factory=dict)
    promoted_without_compiled_entity_count: int = Field(default=0, ge=0)


class TopologyIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_component_count: int = Field(ge=0)
    largest_wall_component_ratio: float = Field(ge=0, le=1)
    dangling_wall_endpoint_count: int = Field(ge=0)
    unsupported_room_boundary_count: int = Field(ge=0)
    promoted_elements_with_required_relation: int = Field(ge=0)
    promoted_elements_requiring_relation: int = Field(ge=0)


class SpatialEvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.spatial-evidence-graph.v2"
    method_version: str = "dajoong-forest-reconstruction-v2"
    source_size: tuple[int, int]
    source_ref_ids: list[str] = Field(min_length=1)
    expert_ids: list[str] = Field(min_length=1)
    nodes: list[SpatialEvidenceNode]
    relations: list[SpatialEvidenceRelation]
    coverage: EvidenceCoverage
    topology_integrity: TopologyIntegrity
    release_ready: bool
    release_blockers: list[str]
    content_sha256: str = ""

    def finalize(self) -> SpatialEvidenceGraph:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


class ForestPerceptionBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    proposal: AecTileProposal
    evidence_graph: SpatialEvidenceGraph


def _point_segment_distance(
    point: tuple[float, float],
    line: tuple[float, float, float, float],
) -> float:
    x0, y0, x1, y1 = line
    dx, dy = x1 - x0, y1 - y0
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, (x0, y0))
    fraction = ((point[0] - x0) * dx + (point[1] - y0) * dy) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    nearest = x0 + fraction * dx, y0 + fraction * dy
    return math.dist(point, nearest)


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
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


def _drawing_scale_floor(
    source_size: tuple[int, int],
    *,
    fraction_of_diagonal: float,
    minimum_px: float,
) -> float:
    """Return a source-scale tolerance instead of a fixed resize-dependent pixel value."""

    width, height = source_size
    return max(minimum_px, math.hypot(width, height) * fraction_of_diagonal)


def _wall_join_tolerance(
    left: SpatialEvidenceNode,
    right: SpatialEvidenceNode,
    source_size: tuple[int, int],
) -> float:
    return max(
        _drawing_scale_floor(
            source_size,
            fraction_of_diagonal=0.0015,
            minimum_px=2.0,
        ),
        float(left.thickness_px or 1.0),
        float(right.thickness_px or 1.0),
    )


def _wall_relation_tolerance(
    wall: SpatialEvidenceNode,
    source_size: tuple[int, int],
    *,
    thickness_multiplier: float,
) -> float:
    return max(
        _drawing_scale_floor(
            source_size,
            fraction_of_diagonal=0.003,
            minimum_px=3.0,
        ),
        float(wall.thickness_px or 1.0) * thickness_multiplier,
    )


def _wall_topology_integrity(
    walls: list[SpatialEvidenceNode],
    rooms: list[SpatialEvidenceNode],
    source_size: tuple[int, int],
) -> tuple[int, float, int, int]:
    if not walls:
        return 0, 0.0, 0, len(rooms)
    parent = list(range(len(walls)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left_wall in enumerate(walls):
        assert left_wall.line_px is not None
        left_points = (left_wall.line_px[:2], left_wall.line_px[2:])
        for right_index in range(left_index + 1, len(walls)):
            right_wall = walls[right_index]
            assert right_wall.line_px is not None
            tolerance = _wall_join_tolerance(left_wall, right_wall, source_size)
            right_points = (right_wall.line_px[:2], right_wall.line_px[2:])
            connected = any(
                _point_segment_distance(point, right_wall.line_px) <= tolerance
                for point in left_points
            ) or any(
                _point_segment_distance(point, left_wall.line_px) <= tolerance
                for point in right_points
            )
            if connected:
                union(left_index, right_index)
    component_sizes: dict[int, int] = {}
    for index in range(len(walls)):
        root = find(index)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    dangling = 0
    for wall_index, wall in enumerate(walls):
        assert wall.line_px is not None
        for endpoint in (wall.line_px[:2], wall.line_px[2:]):
            connected = False
            for candidate_index, candidate in enumerate(walls):
                if candidate_index == wall_index or candidate.line_px is None:
                    continue
                tolerance = _wall_join_tolerance(wall, candidate, source_size)
                if _point_segment_distance(endpoint, candidate.line_px) <= tolerance:
                    connected = True
                    break
            if not connected:
                dangling += 1

    unsupported_rooms = 0
    for room in rooms:
        polygon = room.polygon_px or []
        boundary_samples = []
        for start, end in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
            boundary_samples.extend(
                (
                    start,
                    ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
                )
            )
        supported = sum(
            min(
                _point_segment_distance(point, wall.line_px)
                for wall in walls
                if wall.line_px is not None
            )
            <= _wall_relation_tolerance(
                min(
                    walls,
                    key=lambda wall: _point_segment_distance(point, wall.line_px),
                ),
                source_size,
                thickness_multiplier=1.5,
            )
            for point in boundary_samples
        )
        if boundary_samples and supported / len(boundary_samples) < 0.6:
            unsupported_rooms += 1
    return (
        len(component_sizes),
        max(component_sizes.values()) / len(walls),
        dangling,
        unsupported_rooms,
    )


def _spatial_relations(
    nodes: list[SpatialEvidenceNode],
    source_size: tuple[int, int],
) -> list[SpatialEvidenceRelation]:
    """Materialize the building relationships that isolated detections cannot express."""

    walls = [node for node in nodes if node.kind == "wall" and node.line_px is not None]
    rooms = [node for node in nodes if node.kind == "room" and node.polygon_px]
    relations: list[SpatialEvidenceRelation] = []
    for left_index, left_wall in enumerate(walls):
        assert left_wall.line_px is not None
        left_endpoints = (left_wall.line_px[:2], left_wall.line_px[2:])
        for right_wall in walls[left_index + 1 :]:
            assert right_wall.line_px is not None
            right_endpoints = (right_wall.line_px[:2], right_wall.line_px[2:])
            distance = min(
                _point_segment_distance(point, right_wall.line_px)
                for point in left_endpoints
            )
            distance = min(
                distance,
                *(
                    _point_segment_distance(point, left_wall.line_px)
                    for point in right_endpoints
                ),
            )
            tolerance = _wall_join_tolerance(left_wall, right_wall, source_size)
            if distance <= tolerance:
                relations.append(
                    SpatialEvidenceRelation(
                        id=f"forest:joins:{left_wall.id}:{right_wall.id}",
                        kind="joins_wall",
                        source_id=left_wall.id,
                        target_id=right_wall.id,
                        confidence=max(0.0, 1.0 - distance / tolerance),
                        distance_px=distance,
                    )
                )

    for room in rooms:
        polygon = room.polygon_px or []
        if len(polygon) < 3:
            continue
        boundary_samples = [
            (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
            )
            for start, end in zip(polygon, [*polygon[1:], polygon[0]], strict=True)
        ]
        linked_walls: set[str] = set()
        for point in boundary_samples:
            if not walls:
                break
            distance, wall = min(
                (
                    (_point_segment_distance(point, candidate.line_px), candidate)
                    for candidate in walls
                    if candidate.line_px is not None
                ),
                key=lambda item: item[0],
            )
            tolerance = _wall_relation_tolerance(
                wall,
                source_size,
                thickness_multiplier=1.5,
            )
            if distance <= tolerance and wall.id not in linked_walls:
                linked_walls.add(wall.id)
                relations.append(
                    SpatialEvidenceRelation(
                        id=f"forest:bounds:{room.id}:{wall.id}",
                        kind="bounds_room",
                        source_id=room.id,
                        target_id=wall.id,
                        confidence=max(0.0, 1.0 - distance / tolerance),
                        distance_px=distance,
                    )
                )

    for node in nodes:
        if node.bbox_px is None:
            continue
        left, top, right, bottom = node.bbox_px
        center = (left + right) / 2, (top + bottom) / 2
        if node.kind == "opening" and walls:
            distance, wall = min(
                (
                    (_point_segment_distance(center, candidate.line_px), candidate)
                    for candidate in walls
                    if candidate.line_px is not None
                ),
                key=lambda item: item[0],
            )
            tolerance = _wall_relation_tolerance(
                wall,
                source_size,
                thickness_multiplier=1.25,
            )
            if distance <= tolerance:
                relations.append(
                    SpatialEvidenceRelation(
                        id=f"forest:host:{node.id}",
                        kind="host_candidate",
                        source_id=node.id,
                        target_id=wall.id,
                        confidence=max(0.0, 1.0 - distance / tolerance),
                        distance_px=distance,
                    )
                )
        elif node.kind == "fixture":
            containing = next(
                (
                    room
                    for room in rooms
                    if room.polygon_px and _point_in_polygon(center, room.polygon_px)
                ),
                None,
            )
            if containing is not None:
                relations.append(
                    SpatialEvidenceRelation(
                        id=f"forest:inside:{node.id}",
                        kind="inside_room",
                        source_id=node.id,
                        target_id=containing.id,
                        confidence=min(node.confidence, containing.confidence),
                        distance_px=0.0,
                    )
                )
    return relations


def _finalize_spatial_evidence_graph(
    nodes: list[SpatialEvidenceNode],
    *,
    source_size: tuple[int, int],
    source_ref_ids: list[str],
    expert_ids: list[str],
    full_sheet_context: bool,
) -> SpatialEvidenceGraph:
    walls = [node for node in nodes if node.kind == "wall" and node.line_px is not None]
    rooms = [node for node in nodes if node.kind == "room" and node.polygon_px]
    relations = _spatial_relations(nodes, source_size)
    kinds: tuple[EvidenceKind, ...] = ("wall", "room", "opening", "fixture")
    node_count = {kind: sum(node.kind == kind for node in nodes) for kind in kinds}
    promoted_count = {
        kind: sum(node.kind == kind and node.promoted for node in nodes) for kind in kinds
    }
    review_count = {
        kind: sum(node.kind == kind and node.review_required for node in nodes)
        for kind in kinds
    }
    hosted = {relation.source_id for relation in relations if relation.kind == "host_candidate"}
    assigned = {relation.source_id for relation in relations if relation.kind == "inside_room"}
    bounded = {relation.source_id for relation in relations if relation.kind == "bounds_room"}
    unhosted = sum(
        node.kind == "opening" and node.promoted and node.id not in hosted for node in nodes
    )
    unassigned = sum(
        node.kind == "fixture" and node.promoted and node.id not in assigned for node in nodes
    )
    unclassified = sum(
        node.kind == "room" and node.label == "Unclassified interior" for node in nodes
    )
    (
        wall_component_count,
        largest_wall_component_ratio,
        dangling_wall_endpoint_count,
        unsupported_room_boundary_count,
    ) = _wall_topology_integrity(walls, rooms, source_size)
    unsupported_room_boundary_count = max(
        unsupported_room_boundary_count,
        sum(node.kind == "room" and node.promoted and node.id not in bounded for node in nodes),
    )
    requiring_relation_ids = {
        node.id
        for node in nodes
        if node.promoted and node.kind in {"opening", "fixture"}
    }
    requiring_relation = len(requiring_relation_ids)
    related = len((hosted | assigned) & requiring_relation_ids)
    blockers = ["independent_expert_consensus_unavailable"]
    if unhosted:
        blockers.append("promoted_opening_without_host_candidate")
    if unassigned:
        blockers.append("promoted_fixture_without_room_assignment")
    if unclassified:
        blockers.append("unclassified_enclosed_interior")
    if walls and largest_wall_component_ratio < 0.7:
        blockers.append("fragmented_global_wall_topology")
    if unsupported_room_boundary_count:
        blockers.append("room_boundary_not_supported_by_wall_topology")
    return SpatialEvidenceGraph(
        source_size=source_size,
        source_ref_ids=source_ref_ids,
        expert_ids=expert_ids,
        nodes=nodes,
        relations=relations,
        coverage=EvidenceCoverage(
            full_sheet_context=full_sheet_context,
            expert_count=len(expert_ids),
            node_count_by_kind=node_count,
            promoted_count_by_kind=promoted_count,
            review_count_by_kind=review_count,
            unhosted_opening_count=unhosted,
            unassigned_fixture_count=unassigned,
            unclassified_room_count=unclassified,
            independent_expert_consensus_available=False,
        ),
        topology_integrity=TopologyIntegrity(
            wall_component_count=wall_component_count,
            largest_wall_component_ratio=largest_wall_component_ratio,
            dangling_wall_endpoint_count=dangling_wall_endpoint_count,
            unsupported_room_boundary_count=unsupported_room_boundary_count,
            promoted_elements_with_required_relation=related,
            promoted_elements_requiring_relation=requiring_relation,
        ),
        release_ready=False,
        release_blockers=blockers,
    ).finalize()


def build_spatial_evidence_graph(
    recognition: SemanticRecognitionResult,
    *,
    source_ref_ids: list[str],
) -> SpatialEvidenceGraph:
    expert_id = f"{recognition.model_version}+{recognition.decoder_version}"
    nodes: list[SpatialEvidenceNode] = []
    for index, wall in enumerate(recognition.wall_vectors_px):
        nodes.append(
            SpatialEvidenceNode(
                id=f"semantic:wall:{index}",
                kind="wall",
                label="wall",
                expert_id=expert_id,
                confidence=0.82,
                review_required=True,
                promoted=True,
                line_px=(*wall.start_px, *wall.end_px),
                thickness_px=wall.thickness_px,
            )
        )
    for room in recognition.rooms:
        nodes.append(
            SpatialEvidenceNode(
                id=room.id,
                kind="room",
                label=room.class_name,
                expert_id=expert_id,
                confidence=room.confidence,
                review_required=room.review_required,
                promoted=True,
                polygon_px=room.polygon_px,
            )
        )
    for detection in recognition.detections:
        kind: EvidenceKind = (
            "opening" if detection.symbol_class in {"door", "window"} else "fixture"
        )
        nodes.append(
            SpatialEvidenceNode(
                id=detection.id,
                kind=kind,
                label=detection.symbol_class,
                expert_id=expert_id,
                confidence=detection.confidence,
                review_required=detection.review_required,
                promoted=detection.promote_to_bim,
                bbox_px=tuple(float(value) for value in detection.bbox_px),
            )
        )

    return _finalize_spatial_evidence_graph(
        nodes,
        source_size=recognition.source_size,
        source_ref_ids=source_ref_ids,
        expert_ids=[expert_id],
        full_sheet_context=recognition.model_input_size[0] > 0,
    )


def build_spatial_evidence_graph_from_proposal(
    proposal: AecTileProposal,
    *,
    source_size: tuple[int, int],
    full_sheet_context: bool,
) -> SpatialEvidenceGraph:
    """Build the same authoritative relation graph from a global-program proposal."""

    nodes: list[SpatialEvidenceNode] = []
    for wall in proposal.wall_segments:
        nodes.append(
            SpatialEvidenceNode(
                id=wall.id,
                kind="wall",
                label="wall",
                expert_id=wall.model_version,
                confidence=wall.confidence,
                review_required=wall.review_required,
                promoted=True,
                line_px=(*wall.start_px, *wall.end_px),
                thickness_px=wall.thickness_px,
            )
        )
    for room in proposal.room_regions:
        nodes.append(
            SpatialEvidenceNode(
                id=room.id,
                kind="room",
                label=room.room_class,
                expert_id=room.model_version,
                confidence=room.confidence,
                review_required=room.review_required,
                promoted=True,
                polygon_px=room.polygon_px,
            )
        )
    for symbol in proposal.symbols:
        kind: EvidenceKind = (
            "opening" if symbol.symbol_class in {"door", "window"} else "fixture"
        )
        nodes.append(
            SpatialEvidenceNode(
                id=symbol.id,
                kind=kind,
                label=symbol.symbol_class,
                expert_id=symbol.model_version,
                confidence=symbol.confidence,
                review_required=symbol.review_required,
                promoted=True,
                bbox_px=symbol.bbox_px,
            )
        )
    expert_ids = sorted({node.expert_id for node in nodes}) or [proposal.model_version]
    return _finalize_spatial_evidence_graph(
        nodes,
        source_size=source_size,
        source_ref_ids=proposal.source_ref_ids,
        expert_ids=expert_ids,
        full_sheet_context=full_sheet_context,
    )


def attach_compiled_graph_evidence(
    evidence_graph: SpatialEvidenceGraph,
    plan_graph: dict[str, Any],
) -> SpatialEvidenceGraph:
    """Join perception, compiled BIM entities, and server-selected assets.

    The compiler keeps stable proposal ids in ``source_entity_id``.  This join
    makes missing promotion visible instead of letting an isolated detector or
    a plausible-looking mesh hide a broken building program.
    """

    graph = evidence_graph.model_copy(deep=True)
    nodes = {node.id: node for node in graph.nodes}
    collection_kind: dict[str, EvidenceKind] = {
        "walls": "wall",
        "rooms": "room",
        "openings": "opening",
        "fixtures": "fixture",
    }
    for collection, kind in collection_kind.items():
        for entity in plan_graph.get(collection) or []:
            source_id = str(entity.get("source_entity_id") or "")
            node = nodes.get(source_id)
            if node is None or node.kind != kind:
                continue
            node.compiled_entities.append(
                CompiledEntityEvidence(
                    entity_id=str(entity.get("id") or ""),
                    collection=collection,  # type: ignore[arg-type]
                    review_state=str(entity.get("review_state") or "review_required"),
                    family_id=str(entity.get("family_id") or ""),
                    host_wall_id=str(entity.get("wall_id") or ""),
                    room_id=str(entity.get("room_id") or ""),
                )
            )
            if collection == "fixtures":
                raw_components = entity.get("asset_selection_components")
                components = raw_components if isinstance(raw_components, dict) else {}
                raw_context = entity.get("asset_selection_context")
                context = raw_context if isinstance(raw_context, dict) else {}
                raw_alternates = entity.get("asset_selection_alternates")
                alternates = raw_alternates if isinstance(raw_alternates, list) else []
                node.asset_resolution = AssetResolutionEvidence(
                    geometry_status=str(entity.get("geometry_status") or "semantic_marker"),
                    family_id=str(entity.get("family_id") or entity.get("type") or ""),
                    geometry_ref=str(entity.get("geometry_ref") or ""),
                    asset_uid=str(entity.get("asset_uid") or ""),
                    asset_provider=str(entity.get("asset_provider") or ""),
                    asset_license=str(entity.get("asset_license") or ""),
                    asset_name=str(entity.get("asset_name") or ""),
                    selection_policy=str(entity.get("asset_selection_policy") or ""),
                    selection_score=(
                        float(entity["asset_selection_score"])
                        if entity.get("asset_selection_score") is not None
                        else None
                    ),
                    selection_margin=(
                        float(entity["asset_selection_margin"])
                        if entity.get("asset_selection_margin") is not None
                        else None
                    ),
                    selection_review_required=bool(
                        entity.get("asset_selection_review_required", True)
                    ),
                    candidate_count=int(entity.get("asset_candidate_count") or 0),
                    selection_elapsed_us=float(
                        entity.get("asset_selection_elapsed_us") or 0.0
                    ),
                    selection_context=dict(context),
                    selection_components={
                        str(key): float(value) for key, value in components.items()
                    },
                    selection_alternates=[
                        dict(value) for value in alternates if isinstance(value, dict)
                    ],
                )

    kinds: tuple[EvidenceKind, ...] = ("wall", "room", "opening", "fixture")
    graph.coverage.compiled_count_by_kind = {
        kind: sum(
            node.kind == kind and bool(node.compiled_entities) for node in graph.nodes
        )
        for kind in kinds
    }
    missing = sum(
        node.promoted and not node.compiled_entities for node in graph.nodes
    )
    graph.coverage.promoted_without_compiled_entity_count = missing
    if missing and "promoted_evidence_not_compiled" not in graph.release_blockers:
        graph.release_blockers.append("promoted_evidence_not_compiled")
    graph.release_blockers = sorted(set(graph.release_blockers))
    graph.release_ready = graph.release_ready and missing == 0
    return graph.finalize()


def proposal_from_spatial_evidence_graph(
    evidence_graph: SpatialEvidenceGraph,
    *,
    tile_id: str,
    source_ref_ids: list[str],
    rejected_candidates: int = 0,
) -> AecTileProposal:
    """Materialize the one compiler proposal from the global evidence graph.

    No parallel detector-to-BIM path is allowed here.  Specialists add nodes to
    the graph; host and containment relations determine review state; only this
    graph is translated into the deterministic metric compiler contract.
    """

    related_sources = {relation.source_id for relation in evidence_graph.relations}
    walls: list[PixelLineProposal] = []
    rooms: list[PixelRoomProposal] = []
    symbols: list[PixelSymbolProposal] = []
    for node in evidence_graph.nodes:
        uncertainty = 1.0 - node.confidence
        if node.kind == "wall" and node.promoted and node.line_px is not None:
            walls.append(
                PixelLineProposal(
                    id=node.id,
                    start_px=node.line_px[:2],
                    end_px=node.line_px[2:],
                    thickness_px=node.thickness_px,
                    confidence=node.confidence,
                    uncertainty=uncertainty,
                    source_ref_ids=source_ref_ids,
                    model_version=node.expert_id,
                    review_required=node.review_required,
                )
            )
        elif node.kind == "room" and node.promoted and node.polygon_px:
            rooms.append(
                PixelRoomProposal(
                    id=node.id,
                    name=node.label,
                    room_class=node.label,
                    polygon_px=node.polygon_px,
                    confidence=node.confidence,
                    uncertainty=uncertainty,
                    source_ref_ids=source_ref_ids,
                    model_version=node.expert_id,
                    review_required=node.review_required,
                )
            )
        elif node.kind in {"opening", "fixture"} and node.promoted and node.bbox_px:
            left, top, right, bottom = node.bbox_px
            symbols.append(
                PixelSymbolProposal(
                    id=node.id,
                    symbol_class=node.label,
                    center_px=((left + right) / 2, (top + bottom) / 2),
                    bbox_px=node.bbox_px,
                    confidence=node.confidence,
                    uncertainty=uncertainty,
                    source_ref_ids=source_ref_ids,
                    model_version=node.expert_id,
                    review_required=(
                        node.review_required or node.id not in related_sources
                    ),
                )
            )
    return AecTileProposal(
        tile_id=tile_id,
        source_ref_ids=source_ref_ids,
        model_version=evidence_graph.method_version,
        wall_segments=walls,
        symbols=symbols,
        room_regions=rooms,
        rejected_candidates=rejected_candidates,
    ).finalize()


def build_forest_perception_bundle(
    base_proposal: AecTileProposal,
    recognition: SemanticRecognitionResult,
    *,
    source_ref_ids: list[str],
) -> ForestPerceptionBundle:
    """Fuse current experts through one global graph before BIM compilation."""

    evidence_graph = build_spatial_evidence_graph(
        recognition,
        source_ref_ids=source_ref_ids,
    )
    proposal = proposal_from_spatial_evidence_graph(
        evidence_graph,
        tile_id=base_proposal.tile_id,
        source_ref_ids=source_ref_ids,
        rejected_candidates=base_proposal.rejected_candidates,
    )
    return ForestPerceptionBundle(proposal=proposal, evidence_graph=evidence_graph)
