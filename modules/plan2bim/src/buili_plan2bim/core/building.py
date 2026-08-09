from __future__ import annotations

import copy
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hashing import sha256_json
from .plan_graph_verification import PlanGraphVerifier


class BuildingLevelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=300)
    elevation_m: float
    nominal_height_m: float = Field(default=3.0, gt=0)
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    rotation_deg: float = Field(default=0.0, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_finite_placement(self) -> BuildingLevelSpec:
        values = (self.elevation_m, self.x_offset_m, self.y_offset_m, self.rotation_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("building level placement values must be finite")
        return self


class BuildingVerticalConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    type: str = Field(pattern=r"^(stair|ramp|escalator|elevator|riser)$")
    from_level_id: str = Field(min_length=1, max_length=160)
    to_level_id: str = Field(min_length=1, max_length=160)
    shaft_id: str = Field(default="", max_length=160)
    center_m: tuple[float, float]
    footprint_m: tuple[float, float]
    yaw_deg: float = Field(default=0.0, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_geometry(self) -> BuildingVerticalConnection:
        values = (*self.center_m, *self.footprint_m, self.yaw_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vertical connection geometry must be finite")
        if any(value <= 0 for value in self.footprint_m):
            raise ValueError("vertical connection footprint must be positive")
        if self.from_level_id == self.to_level_id:
            raise ValueError("vertical connection must join two different levels")
        return self


class BuildingAssemblyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1, max_length=160)
    levels: list[BuildingLevelSpec] = Field(min_length=1, max_length=10_000)
    vertical_connections: list[BuildingVerticalConnection] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology(self) -> BuildingAssemblyConfig:
        ids = [level.level_id for level in self.levels]
        if len(ids) != len(set(ids)):
            raise ValueError("building level ids must be unique")
        elevations = [level.elevation_m for level in self.levels]
        if len(elevations) != len(set(elevations)):
            raise ValueError("building levels must have unique elevations")
        level_map = {level.level_id: level for level in self.levels}
        connection_ids = [connection.id for connection in self.vertical_connections]
        if len(connection_ids) != len(set(connection_ids)):
            raise ValueError("vertical connection ids must be unique")
        ordered_ids = [
            level.level_id for level in sorted(self.levels, key=lambda item: item.elevation_m)
        ]
        adjacent = {
            (lower, upper) for lower, upper in zip(ordered_ids, ordered_ids[1:], strict=False)
        }
        for connection in self.vertical_connections:
            if connection.from_level_id not in level_map or connection.to_level_id not in level_map:
                raise ValueError(
                    f"vertical connection {connection.id!r} references an unknown level"
                )
            if (
                level_map[connection.to_level_id].elevation_m
                <= level_map[connection.from_level_id].elevation_m
            ):
                raise ValueError(
                    f"vertical connection {connection.id!r} must move to a higher level"
                )
            if (
                connection.type in {"stair", "ramp", "escalator"}
                and (
                    connection.from_level_id,
                    connection.to_level_id,
                )
                not in adjacent
            ):
                raise ValueError(f"vertical connection {connection.id!r} must join adjacent levels")
        return self


def _transform_point(point: Any, spec: BuildingLevelSpec) -> list[float]:
    x, y = float(point[0]), float(point[1])
    angle = math.radians(spec.rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        x * cosine - y * sine + spec.x_offset_m,
        x * sine + y * cosine + spec.y_offset_m,
    ]


def assemble_building_graph(
    level_graphs: dict[str, dict[str, Any]],
    config: BuildingAssemblyConfig,
) -> dict[str, Any]:
    """Assemble reviewed single-level graphs without inventing vertical topology."""

    requested = {level.level_id for level in config.levels}
    if set(level_graphs) != requested:
        raise ValueError(
            f"level graph keys must exactly match assembly levels: expected {sorted(requested)}"
        )
    output: dict[str, Any] = {
        "schema_version": "buili.plan-graph.v2",
        "project_id": config.project_id,
        "sheet_id": "multi-level-building",
        "scale": {"px_per_meter": 1.0, "source": "assembled_metric_graphs", "confidence": 0.0},
        "levels": [],
        "rooms": [],
        "walls": [],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "constraints": [],
        "dimensions": [],
        "sources": [],
        "unsupported_features": [],
        "warnings": [],
        "extraction": {"method": "buili_multilevel_assembly_v1"},
        "provenance": {"source_revision_state": "assembled_review_graphs"},
    }
    review_required = False
    confidence_values: list[float] = []
    source_ids: list[str] = []
    for spec in sorted(config.levels, key=lambda item: item.elevation_m):
        graph = copy.deepcopy(level_graphs[spec.level_id])
        graph_levels = list(graph.get("levels") or [])
        if len(graph_levels) != 1:
            raise ValueError(f"level graph {spec.level_id!r} must contain exactly one level")
        source_map: dict[str, str] = {}
        for source in graph.get("sources") or []:
            original = str(source.get("source_ref_id") or "")
            namespaced = f"{spec.level_id}:{original}"
            source_map[original] = namespaced
            source["source_ref_id"] = namespaced
            output["sources"].append(source)
            source_ids.append(namespaced)
        id_maps: dict[str, dict[str, str]] = {}
        for collection in ("walls", "rooms", "openings", "fixtures", "routes", "dimensions"):
            id_maps[collection] = {
                str(entity.get("id") or entity.get("source_entity_id")): (
                    f"{spec.level_id}:{entity.get('id') or entity.get('source_entity_id')}"
                )
                for entity in graph.get(collection) or []
            }
        level_entity = graph_levels[0]
        level_entity["id"] = spec.level_id
        level_entity["name"] = spec.name
        level_entity["elevation_m"] = spec.elevation_m
        level_entity["nominal_height_m"] = spec.nominal_height_m
        level_entity["source_ref_ids"] = [
            source_map.get(str(item), str(item))
            for item in level_entity.get("source_ref_ids") or []
        ]
        output["levels"].append(level_entity)

        for collection in ("walls", "rooms", "openings", "fixtures", "routes", "dimensions"):
            for entity in graph.get(collection) or []:
                original_id = str(entity.get("id") or entity.get("source_entity_id"))
                entity["id"] = id_maps[collection][original_id]
                if entity.get("source_entity_id"):
                    entity["source_entity_id"] = entity["id"]
                entity["level_id"] = spec.level_id
                entity["source_ref_ids"] = [
                    source_map.get(str(item), str(item))
                    for item in entity.get("source_ref_ids") or []
                ]
                if collection == "walls":
                    entity["from"] = _transform_point(entity["from"], spec)
                    entity["to"] = _transform_point(entity["to"], spec)
                    if entity.get("footprint_m"):
                        entity["footprint_m"] = [
                            _transform_point(point, spec) for point in entity["footprint_m"]
                        ]
                elif collection == "rooms":
                    entity["polygon"] = [
                        _transform_point(point, spec) for point in entity.get("polygon") or []
                    ]
                elif collection == "openings":
                    entity["wall_id"] = id_maps["walls"][str(entity["wall_id"])]
                    entity["center_m"] = _transform_point(entity["center_m"], spec)
                elif collection == "fixtures":
                    if entity.get("room_id"):
                        entity["room_id"] = id_maps["rooms"].get(str(entity["room_id"]), "")
                    entity["center_m"] = _transform_point(entity["center_m"], spec)
                    entity["yaw_deg"] = float(entity.get("yaw_deg") or 0.0) + spec.rotation_deg
                elif collection == "routes":
                    entity["points_m"] = [
                        [*_transform_point(point, spec), float(point[2])]
                        for point in entity.get("points_m") or []
                    ]
                elif collection == "dimensions":
                    entity["from"] = _transform_point(entity["from"], spec)
                    entity["to"] = _transform_point(entity["to"], spec)
                output[collection].append(entity)
                confidence_values.append(float(entity.get("confidence") or 0.0))
                review_required = (
                    review_required
                    or str(entity.get("review_state") or "review_required") != "accepted"
                )
        for constraint in graph.get("constraints") or []:
            original_id = str(constraint.get("id") or "constraint")
            constraint["id"] = f"{spec.level_id}:{original_id}"
            constraint["level_id"] = spec.level_id
            constraint["references"] = [
                {
                    **reference,
                    "entity_id": id_maps["walls"].get(
                        str(reference.get("entity_id") or ""),
                        str(reference.get("entity_id") or ""),
                    ),
                }
                for reference in constraint.get("references") or []
            ]
            constraint["source_ref_ids"] = [
                source_map.get(str(item), str(item))
                for item in constraint.get("source_ref_ids") or []
            ]
            output["constraints"].append(constraint)
        output["unsupported_features"].extend(graph.get("unsupported_features") or [])
        output["warnings"].extend(graph.get("warnings") or [])

    level_source_ids: dict[str, list[str]] = {
        level_id: [
            str(source.get("source_ref_id") or "")
            for source in output["sources"]
            if str(source.get("source_ref_id") or "").startswith(f"{level_id}:")
        ]
        for level_id in requested
    }
    for connection in config.vertical_connections:
        connection_sources = list(
            dict.fromkeys(
                [
                    *level_source_ids.get(connection.from_level_id, []),
                    *level_source_ids.get(connection.to_level_id, []),
                ]
            )
        )
        output["vertical_connections"].append(
            {
                **connection.model_dump(mode="json"),
                "confidence": 1.0,
                "uncertainty": 0.0,
                "source_ref_ids": connection_sources,
                "model_version": "human-confirmed-building-assembly-v1",
                "review_state": "accepted",
            }
        )
    output["unsupported_features"] = sorted(set(output["unsupported_features"]))
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    output["scale"]["confidence"] = confidence
    output["confidence"] = {
        "overall": confidence,
        "geometry": confidence,
        "semantics": confidence,
        "scale": confidence,
        "traceability": 1.0,
        "review_required": review_required,
        "method": "assembled_level_quality_gate",
    }
    output["pipeline"] = {
        "contract_version": "buili.plan-graph.v2",
        "pipeline_version": "buili-building-assembly-1.0",
        "deterministic": True,
        "review_required": review_required,
    }
    level_qualifications = {
        level_id: copy.deepcopy(graph.get("qualification"))
        for level_id, graph in level_graphs.items()
        if isinstance(graph.get("qualification"), dict)
    }
    if level_qualifications:
        eligible = len(level_qualifications) == len(level_graphs) and all(
            bool(item.get("production_release_eligible", False))
            for item in level_qualifications.values()
        )
        output["qualification"] = {
            "schema_version": "dajoong.building-model-qualification.v1",
            "level_qualifications": level_qualifications,
            "production_release_eligible": eligible,
            "review_required": not eligible,
            "review_reasons": sorted(
                {
                    str(reason)
                    for item in level_qualifications.values()
                    for reason in item.get("review_reasons", [])
                }
                | ({"one_or_more_levels_missing_qualification"}
                   if len(level_qualifications) != len(level_graphs) else set())
            ),
        }
        output["pipeline"]["review_required"] = not eligible or review_required
        output["confidence"]["review_required"] = not eligible or review_required
    output["provenance"]["source_hash"] = sha256_json(
        {
            level_id: graph.get("pipeline", {}).get("content_sha256", "")
            for level_id, graph in level_graphs.items()
        }
    )
    certificate = PlanGraphVerifier().verify(output)
    output["verification"] = certificate.model_dump(mode="json")
    output["pipeline"]["release_allowed"] = certificate.release_allowed
    output["pipeline"]["certificate_sha256"] = certificate.content_sha256
    output["pipeline"]["content_sha256"] = sha256_json(
        {key: value for key, value in output.items() if key != "verification"}
    )
    return output
