from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import sha256_json
from .plan_graph_verification import PlanGraphVerifier

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
ReviewState = Literal["accepted", "review_required", "rejected"]
Discipline = Literal["architectural", "structural", "electrical", "mechanical", "plumbing", "fire"]


def _finite_vector(value: tuple[float, ...]) -> tuple[float, ...]:
    if any(not math.isfinite(item) for item in value):
        raise ValueError("BIM program vectors must be finite")
    return value


class ProgramEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    uri: str = Field(min_length=1, max_length=4096)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(ge=1)
    region: tuple[float, float, float, float] | None = None
    source_kind: Literal[
        "vector_pdf",
        "raster_pdf",
        "raster_image",
        "schedule",
        "section",
        "field_scan",
    ]
    extractor: str = Field(min_length=1, max_length=160)
    model_version: str = Field(min_length=1, max_length=160)

    @field_validator("region")
    @classmethod
    def valid_region(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        _finite_vector(value)
        if value[2] < value[0] or value[3] < value[1]:
            raise ValueError("evidence region maxima must be >= minima")
        return value


class ProgramEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    proposal_id: str = Field(default="", max_length=160)
    level_id: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=160)
    review_state: ReviewState = "review_required"


class LevelInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=300)
    elevation_m: float
    nominal_height_m: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=160)
    review_state: ReviewState = "review_required"

    @field_validator("elevation_m")
    @classmethod
    def finite_elevation(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("level elevation must be finite")
        return value


class WallInstruction(ProgramEntity):
    start_m: Point2D
    end_m: Point2D
    thickness_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    wall_type: Literal["exterior", "interior", "shaft", "curtain", "unknown"] = "unknown"
    material: str = ""

    @field_validator("start_m", "end_m")
    @classmethod
    def finite_points(cls, value: Point2D) -> Point2D:
        return _finite_vector(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def nonzero(self) -> WallInstruction:
        if math.dist(self.start_m, self.end_m) <= 1e-6:
            raise ValueError("wall endpoints must be distinct")
        return self


class RoomSide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_id: str = Field(min_length=1, max_length=160)
    reversed: bool = False
    start_offset_m: float | None = Field(default=None, ge=0)
    end_offset_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_partial_wall_interval(self) -> RoomSide:
        if (self.start_offset_m is None) != (self.end_offset_m is None):
            raise ValueError("room side offsets must either both be set or both be omitted")
        if (
            self.start_offset_m is not None
            and self.end_offset_m is not None
            and self.end_offset_m <= self.start_offset_m
        ):
            raise ValueError("room side end offset must be greater than its start offset")
        return self


class RoomInstruction(ProgramEntity):
    name: str = Field(min_length=1, max_length=300)
    sides: list[RoomSide] = Field(default_factory=list, max_length=10_000)
    polygon_m: list[Point2D] = Field(default_factory=list, max_length=100_000)
    occupancy: str = ""

    @model_validator(mode="after")
    def geometry_exists(self) -> RoomInstruction:
        if len(self.sides) < 3 and len(self.polygon_m) < 3:
            raise ValueError("room requires either three wall sides or a polygon")
        for point in self.polygon_m:
            _finite_vector(point)
        return self


class OpeningInstruction(ProgramEntity):
    kind: Literal["door", "window", "opening"]
    wall_id: str = Field(min_length=1, max_length=160)
    offset_m: float = Field(ge=0)
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    sill_height_m: float = Field(default=0.0, ge=0)
    family_id: str = ""
    operation_type: Literal[
        "single_swing", "double_swing", "sliding", "folding", "fixed", "unknown"
    ] = "unknown"
    handing: Literal["start", "end", "double", "unknown"] = "unknown"
    swing_side: Literal["positive", "negative", "both", "none", "unknown"] = "unknown"


class FixtureInstruction(ProgramEntity):
    family_id: str = Field(min_length=1, max_length=160)
    discipline: Discipline
    center_m: Point2D
    base_elevation_m: float = 0.0
    size_m: Point3D
    yaw_deg: float = Field(default=0.0, ge=-180, le=180)
    room_id: str = ""
    geometry_status: Literal[
        "evidence_sized",
        "approved_family",
        "semantic_marker",
        "licensed_api_asset",
        "native_bim_parametric",
    ]
    material: str = ""
    asset_sha256: str = ""

    @field_validator("center_m", "size_m")
    @classmethod
    def finite_geometry(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _finite_vector(value)

    @field_validator("size_m")
    @classmethod
    def positive_size(cls, value: Point3D) -> Point3D:
        if any(item <= 0 for item in value):
            raise ValueError("fixture sizes must be positive")
        return value


class RouteInstruction(ProgramEntity):
    system_id: str = Field(min_length=1, max_length=160)
    discipline: Literal["electrical", "mechanical", "plumbing", "fire"]
    kind: str = Field(min_length=1, max_length=160)
    points_m: list[Point3D] = Field(min_length=2, max_length=100_000)
    section_m: tuple[float, float] = (0.05, 0.05)
    material: str = ""

    @field_validator("points_m")
    @classmethod
    def finite_route(cls, value: list[Point3D]) -> list[Point3D]:
        for point in value:
            _finite_vector(point)
        return value

    @field_validator("section_m")
    @classmethod
    def positive_section(cls, value: tuple[float, float]) -> tuple[float, float]:
        _finite_vector(value)
        if any(item <= 0 for item in value):
            raise ValueError("route section must be positive")
        return value


class VerticalConnectionInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    kind: Literal["stair", "ramp", "escalator", "elevator", "riser"]
    from_level_id: str = Field(min_length=1, max_length=160)
    to_level_id: str = Field(min_length=1, max_length=160)
    center_m: Point2D
    footprint_m: tuple[float, float]
    yaw_deg: float = Field(default=0.0, ge=-180, le=180)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=160)
    review_state: ReviewState = "review_required"

    @field_validator("center_m", "footprint_m")
    @classmethod
    def finite_geometry(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        return _finite_vector(value)

    @field_validator("footprint_m")
    @classmethod
    def positive_footprint(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(item <= 0 for item in value):
            raise ValueError("vertical connection footprint must be positive")
        return value


class BimProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.bim-program.v1"
    program_id: str = Field(min_length=1, max_length=160)
    project_id: str = Field(min_length=1, max_length=160)
    evidence: list[ProgramEvidence] = Field(min_length=1, max_length=1_000_000)
    levels: list[LevelInstruction] = Field(min_length=1, max_length=10_000)
    walls: list[WallInstruction] = Field(default_factory=list, max_length=1_000_000)
    rooms: list[RoomInstruction] = Field(default_factory=list, max_length=100_000)
    openings: list[OpeningInstruction] = Field(default_factory=list, max_length=1_000_000)
    fixtures: list[FixtureInstruction] = Field(default_factory=list, max_length=2_000_000)
    routes: list[RouteInstruction] = Field(default_factory=list, max_length=1_000_000)
    vertical_connections: list[VerticalConnectionInstruction] = Field(
        default_factory=list, max_length=100_000
    )
    compiler_version: str = "dajoong-bim-compiler-0.1"
    content_sha256: str = ""

    @model_validator(mode="after")
    def references_exist(self) -> BimProgram:
        evidence_ids = {item.id for item in self.evidence}
        level_ids = {item.id for item in self.levels}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence ids must be unique")
        if len(level_ids) != len(self.levels):
            raise ValueError("level ids must be unique")
        entities: list[Any] = [
            *self.walls,
            *self.rooms,
            *self.openings,
            *self.fixtures,
            *self.routes,
        ]
        ids = [entity.id for entity in entities]
        ids.extend(connection.id for connection in self.vertical_connections)
        if len(ids) != len(set(ids)):
            raise ValueError("BIM program entity ids must be globally unique")
        for entity in entities:
            if entity.level_id not in level_ids:
                raise ValueError(f"entity {entity.id!r} references unknown level")
            missing = set(entity.source_ref_ids) - evidence_ids
            if missing:
                raise ValueError(f"entity {entity.id!r} references unknown evidence {missing}")
        walls = {wall.id: wall for wall in self.walls}
        rooms = {room.id: room for room in self.rooms}
        for room in self.rooms:
            for side in room.sides:
                if side.wall_id not in walls:
                    raise ValueError(f"room {room.id!r} references unknown wall {side.wall_id!r}")
                if walls[side.wall_id].level_id != room.level_id:
                    raise ValueError(f"room {room.id!r} references a wall on another level")
        for opening in self.openings:
            if opening.wall_id not in walls:
                raise ValueError(f"opening {opening.id!r} references unknown wall")
            if walls[opening.wall_id].level_id != opening.level_id:
                raise ValueError(f"opening {opening.id!r} references a wall on another level")
        for fixture in self.fixtures:
            if fixture.room_id and fixture.room_id not in rooms:
                raise ValueError(f"fixture {fixture.id!r} references unknown room")
        for connection in self.vertical_connections:
            if connection.from_level_id not in level_ids or connection.to_level_id not in level_ids:
                raise ValueError(f"vertical connection {connection.id!r} references unknown level")
            missing = set(connection.source_ref_ids) - evidence_ids
            if missing:
                raise ValueError(
                    f"vertical connection {connection.id!r} references unknown evidence {missing}"
                )
        return self

    def finalize(self) -> BimProgram:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


class BimProgramCompiler:
    """Compile neural proposals into a deterministic, proof-carrying PlanGraph."""

    def __init__(self, *, snap_tolerance_m: float = 0.03, quantization_m: float = 0.001) -> None:
        if snap_tolerance_m <= 0 or quantization_m <= 0:
            raise ValueError("compiler tolerances must be positive")
        self.snap_tolerance_m = snap_tolerance_m
        self.quantization_m = quantization_m

    def compile(self, program: BimProgram) -> dict[str, Any]:
        if not program.content_sha256:
            program = program.model_copy(deep=True).finalize()
        snapped = self._snap_walls(program.walls)
        wall_map = {wall["id"]: wall for wall in snapped}
        unsupported: list[str] = []
        rooms = []
        for room in program.rooms:
            polygon: list[list[float]] = [
                [self._quantize(point[0]), self._quantize(point[1])] for point in room.polygon_m
            ]
            previous_end: Point2D | None = None
            first_start: Point2D | None = None
            for side in room.sides if not room.polygon_m else []:
                wall = wall_map[side.wall_id]
                wall_start = tuple(wall["from"])
                wall_end = tuple(wall["to"])
                wall_length = math.dist(wall_start, wall_end)
                lower = side.start_offset_m if side.start_offset_m is not None else 0.0
                upper = side.end_offset_m if side.end_offset_m is not None else wall_length
                if lower < 0 or upper > wall_length + self.snap_tolerance_m:
                    unsupported.append(f"room-side-outside-wall:{room.id}:{side.wall_id}")
                lower = max(0.0, min(wall_length, lower))
                upper = max(0.0, min(wall_length, upper))
                direction = (
                    (wall_end[0] - wall_start[0]) / wall_length,
                    (wall_end[1] - wall_start[1]) / wall_length,
                )
                interval_start = (
                    self._quantize(wall_start[0] + direction[0] * lower),
                    self._quantize(wall_start[1] + direction[1] * lower),
                )
                interval_end = (
                    self._quantize(wall_start[0] + direction[0] * upper),
                    self._quantize(wall_start[1] + direction[1] * upper),
                )
                start, end = (
                    (interval_end, interval_start)
                    if side.reversed
                    else (interval_start, interval_end)
                )
                if (
                    previous_end is not None
                    and math.dist(previous_end, start) > self.snap_tolerance_m
                ):
                    unsupported.append(f"room-cycle-gap:{room.id}:{side.wall_id}")
                if first_start is None:
                    first_start = start
                polygon.append([float(start[0]), float(start[1])])
                previous_end = end
            if first_start is not None and previous_end is not None:
                if math.dist(previous_end, first_start) > self.snap_tolerance_m:
                    unsupported.append(f"room-cycle-open:{room.id}")
            rooms.append(
                {
                    "id": room.id,
                    "level_id": room.level_id,
                    "name": room.name,
                    "polygon": polygon,
                    "occupancy": room.occupancy,
                    **self._evidence_fields(room),
                }
            )
        openings = []
        for opening in program.openings:
            wall = wall_map[opening.wall_id]
            start, end = tuple(wall["from"]), tuple(wall["to"])
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            center = [
                self._quantize(start[0] + dx / length * opening.offset_m),
                self._quantize(start[1] + dy / length * opening.offset_m),
            ]
            openings.append(
                {
                    "id": opening.id,
                    "source_entity_id": opening.proposal_id or opening.id,
                    "level_id": opening.level_id,
                    "type": opening.kind,
                    "wall_id": opening.wall_id,
                    "x_m": self._quantize(opening.offset_m),
                    "center_m": center,
                    "width_m": opening.width_m,
                    "height_m": opening.height_m,
                    "sill_height_m": opening.sill_height_m,
                    "family_id": opening.family_id,
                    "operation_type": opening.operation_type,
                    "handing": opening.handing,
                    "swing_side": opening.swing_side,
                    **self._evidence_fields(opening),
                }
            )
        fixtures = [
            {
                "id": fixture.id,
                "source_entity_id": fixture.proposal_id or fixture.id,
                "level_id": fixture.level_id,
                "type": fixture.family_id,
                "family_id": fixture.family_id,
                "discipline": fixture.discipline,
                "room_id": fixture.room_id,
                "center_m": [self._quantize(value) for value in fixture.center_m],
                "base_elevation_m": self._quantize(fixture.base_elevation_m),
                "size_m": [self._quantize(value) for value in fixture.size_m],
                "yaw_deg": fixture.yaw_deg,
                "geometry_status": fixture.geometry_status,
                "material": fixture.material,
                "asset_sha256": fixture.asset_sha256,
                "required_count": 1,
                "observed_count": 1,
                **self._evidence_fields(fixture),
            }
            for fixture in program.fixtures
        ]
        routes = [
            {
                "id": route.id,
                "level_id": route.level_id,
                "system_id": route.system_id,
                "discipline": route.discipline,
                "type": route.kind,
                "points_m": [
                    [self._quantize(coordinate) for coordinate in point] for point in route.points_m
                ],
                "section_m": list(route.section_m),
                "material": route.material,
                **self._evidence_fields(route),
            }
            for route in program.routes
        ]
        levels = [
            {
                "id": level.id,
                "name": level.name,
                "elevation_m": self._quantize(level.elevation_m),
                "nominal_height_m": self._quantize(level.nominal_height_m),
                **self._evidence_fields(level),
            }
            for level in program.levels
        ]
        connections = [
            {
                "id": connection.id,
                "type": connection.kind,
                "from_level_id": connection.from_level_id,
                "to_level_id": connection.to_level_id,
                "center_m": [self._quantize(value) for value in connection.center_m],
                "footprint_m": [self._quantize(value) for value in connection.footprint_m],
                "yaw_deg": connection.yaw_deg,
                **self._evidence_fields(connection),
            }
            for connection in program.vertical_connections
        ]
        sources = [
            {
                "source_ref_id": evidence.id,
                "source_hash": evidence.sha256,
                "uri": evidence.uri,
                "page": evidence.page_number,
                "bbox": list(evidence.region or ()),
                "source_type": evidence.source_kind,
                "source_strength": "strong",
                "extractor": evidence.extractor,
                "model_version": evidence.model_version,
            }
            for evidence in program.evidence
        ]
        all_entities: list[Any] = [
            *program.levels,
            *program.walls,
            *program.rooms,
            *program.openings,
            *program.fixtures,
            *program.routes,
            *program.vertical_connections,
        ]
        confidences = [entity.confidence for entity in all_entities]
        review_required = bool(unsupported) or any(
            entity.review_state != "accepted" or entity.uncertainty > 0.25
            for entity in all_entities
        )
        graph: dict[str, Any] = {
            "schema_version": "buili.plan-graph.v2",
            "project_id": program.project_id,
            "sheet_id": "multi-sheet-program",
            "scale": {
                "px_per_meter": 1.0,
                "source": "metric_bim_program",
                "confidence": min(confidences, default=0.0),
            },
            "levels": levels,
            "rooms": rooms,
            "walls": snapped,
            "openings": openings,
            "fixtures": fixtures,
            "routes": routes,
            "vertical_connections": connections,
            "constraints": [],
            "dimensions": [],
            "sources": sources,
            "unsupported_features": sorted(set(unsupported)),
            "extraction": {
                "method": "dajoong_e3_bim_program_compiler",
                "program_id": program.program_id,
                "program_sha256": program.content_sha256,
                "compiler_version": program.compiler_version,
            },
            "provenance": {
                "source_hash": program.content_sha256,
                "source_revision_state": "program_compiled",
            },
            "confidence": {
                "overall": statistics.fmean(confidences) if confidences else 0.0,
                "geometry": statistics.fmean(
                    entity.confidence for entity in [*program.walls, *program.rooms]
                )
                if program.walls or program.rooms
                else 0.0,
                "semantics": statistics.fmean(confidences) if confidences else 0.0,
                "scale": min(confidences, default=0.0),
                "traceability": 1.0,
                "review_required": review_required,
                "method": "proof_carrying_program_quality_gate",
            },
            "warnings": [],
            "pipeline": {
                "contract_version": "buili.plan-graph.v2",
                "pipeline_version": program.compiler_version,
                "deterministic": True,
                "program_sha256": program.content_sha256,
                "review_required": review_required,
            },
        }
        certificate = PlanGraphVerifier().verify(graph)
        graph["verification"] = certificate.model_dump(mode="json")
        graph["pipeline"]["release_allowed"] = certificate.release_allowed
        graph["pipeline"]["certificate_sha256"] = certificate.content_sha256
        graph["pipeline"]["content_sha256"] = sha256_json(
            {key: value for key, value in graph.items() if key != "verification"}
        )
        return graph

    def _snap_walls(self, walls: list[WallInstruction]) -> list[dict[str, Any]]:
        by_level: dict[str, list[WallInstruction]] = defaultdict(list)
        for wall in walls:
            by_level[wall.level_id].append(wall)
        output = []
        for level_id in sorted(by_level):
            level_walls = sorted(by_level[level_id], key=lambda wall: wall.id)
            endpoints: list[tuple[Point2D, float]] = []
            for wall in level_walls:
                endpoints.extend(((wall.start_m, wall.confidence), (wall.end_m, wall.confidence)))
            union = _UnionFind(len(endpoints))
            for left in range(len(endpoints)):
                for right in range(left + 1, len(endpoints)):
                    if math.dist(endpoints[left][0], endpoints[right][0]) <= self.snap_tolerance_m:
                        union.union(left, right)
            clusters: dict[int, list[int]] = defaultdict(list)
            for index in range(len(endpoints)):
                clusters[union.find(index)].append(index)
            snapped_points: dict[int, Point2D] = {}
            for members in clusters.values():
                weight = sum(max(endpoints[index][1], 1e-6) for index in members)
                x = sum(
                    endpoints[index][0][0] * max(endpoints[index][1], 1e-6) for index in members
                )
                y = sum(
                    endpoints[index][0][1] * max(endpoints[index][1], 1e-6) for index in members
                )
                point = self._quantize(x / weight), self._quantize(y / weight)
                for index in members:
                    snapped_points[index] = point
            for index, wall in enumerate(level_walls):
                output.append(
                    {
                        "id": wall.id,
                        "level_id": wall.level_id,
                        "room_id": "",
                        "from": list(snapped_points[index * 2]),
                        "to": list(snapped_points[index * 2 + 1]),
                        "thickness_m": wall.thickness_m,
                        "height_m": wall.height_m,
                        "wall_type": wall.wall_type,
                        "material": wall.material,
                        **self._evidence_fields(wall),
                    }
                )
        return output

    def _quantize(self, value: float) -> float:
        return round(round(float(value) / self.quantization_m) * self.quantization_m, 9)

    @staticmethod
    def _evidence_fields(entity: Any) -> dict[str, Any]:
        fields = {
            "confidence": entity.confidence,
            "uncertainty": entity.uncertainty,
            "source_ref_ids": list(entity.source_ref_ids),
            "model_version": entity.model_version,
            "review_state": entity.review_state,
        }
        proposal_id = str(getattr(entity, "proposal_id", ""))
        if proposal_id:
            fields["source_entity_id"] = proposal_id
        return fields
