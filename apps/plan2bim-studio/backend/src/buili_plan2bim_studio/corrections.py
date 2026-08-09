from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, Literal

from buili_plan2bim.core.hashing import sha256_json
from buili_plan2bim.core.plan_graph_verification import PlanGraphVerifier
from pydantic import BaseModel, ConfigDict, Field, model_validator

CollectionName = Literal[
    "levels",
    "walls",
    "rooms",
    "openings",
    "fixtures",
    "routes",
    "vertical_connections",
    "constraints",
    "dimensions",
]
ActionName = Literal["add", "update", "delete", "accept"]

EDITABLE_FIELDS: dict[str, set[str]] = {
    "levels": {"name", "elevation_m", "nominal_height_m"},
    "walls": {
        "from",
        "to",
        "thickness_m",
        "height_m",
        "wall_type",
        "material",
        "room_id",
    },
    "rooms": {"name", "polygon", "occupancy"},
    "openings": {
        "type",
        "wall_id",
        "center_m",
        "x_m",
        "width_m",
        "height_m",
        "sill_height_m",
        "family_id",
        "operation_type",
        "handing",
        "swing_side",
    },
    "fixtures": {
        "type",
        "family_id",
        "discipline",
        "room_id",
        "center_m",
        "base_elevation_m",
        "size_m",
        "yaw_deg",
        "material",
        "asset_sha256",
        "geometry_status",
        "observed_count",
        "required_count",
    },
    "routes": {
        "kind",
        "system_id",
        "discipline",
        "points_m",
        "section_m",
        "material",
    },
    "vertical_connections": {
        "type",
        "kind",
        "from_level_id",
        "to_level_id",
        "center_m",
        "footprint_m",
        "yaw_deg",
    },
    "constraints": {"type", "references", "value_m"},
    "dimensions": {"type", "name", "from", "to"},
}

ADD_ONLY_FIELDS = {
    "source_ref_ids",
    "source_entity_id",
    "copied_from_entity_id",
}


class GraphCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    action: ActionName
    collection: CollectionName
    entity_id: str = Field(min_length=1, max_length=300)
    changes: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="manual_review", min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_changes(self) -> GraphCorrection:
        allowed = EDITABLE_FIELDS[self.collection]
        add_only = ADD_ONLY_FIELDS if self.action == "add" else set()
        unknown = set(self.changes) - allowed - add_only - {"id", "level_id"}
        if unknown:
            raise ValueError(f"unsupported fields for {self.collection}: {sorted(unknown)}")
        if self.action in {"delete", "accept"} and self.changes:
            raise ValueError(f"{self.action} operations cannot include changes")
        if self.action in {"add", "update"} and not self.changes:
            raise ValueError(f"{self.action} operations require changes")
        return self


class GraphCorrectionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["buili.plan2bim-corrections.v1"] = "buili.plan2bim-corrections.v1"
    expected_graph_sha256: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(default="studio-user", min_length=1, max_length=200)
    operations: list[GraphCorrection] = Field(default_factory=list, max_length=20_000)


def graph_content_hash(graph: dict[str, Any]) -> str:
    payload = copy.deepcopy(graph)
    pipeline = payload.get("pipeline")
    if isinstance(pipeline, dict):
        pipeline.pop("content_sha256", None)
    return sha256_json(payload)


def _find_entity(items: list[dict[str, Any]], entity_id: str) -> tuple[int, dict[str, Any]]:
    for index, item in enumerate(items):
        if item.get("id") == entity_id:
            return index, item
    raise KeyError(f"entity not found: {entity_id}")


def _check_references(graph: dict[str, Any]) -> None:
    level_ids = {str(item.get("id")) for item in graph.get("levels", [])}
    wall_ids = {str(item.get("id")) for item in graph.get("walls", [])}
    room_ids = {str(item.get("id")) for item in graph.get("rooms", [])}
    for collection in (
        "walls",
        "rooms",
        "openings",
        "fixtures",
        "routes",
        "constraints",
        "dimensions",
    ):
        for item in graph.get(collection, []):
            level_id = str(item.get("level_id", ""))
            if level_id and level_id not in level_ids:
                raise ValueError(f"{item.get('id')} references missing level {level_id}")
    for opening in graph.get("openings", []):
        if opening.get("wall_id") not in wall_ids:
            raise ValueError(
                f"{opening.get('id')} references missing wall {opening.get('wall_id')}"
            )
    for fixture in graph.get("fixtures", []):
        room_id = str(fixture.get("room_id", ""))
        if room_id and room_id not in room_ids:
            raise ValueError(f"{fixture.get('id')} references missing room {room_id}")
    for connection in graph.get("vertical_connections", []):
        for key in ("from_level_id", "to_level_id"):
            level_id = str(connection.get(key, ""))
            if level_id not in level_ids:
                raise ValueError(f"{connection.get('id')} references missing level {level_id}")
    for constraint in graph.get("constraints", []):
        for reference in constraint.get("references", []):
            if reference.get("collection") != "walls":
                raise ValueError(
                    f"{constraint.get('id')} has unsupported constraint reference collection"
                )
            wall_id = str(reference.get("entity_id", ""))
            if wall_id not in wall_ids:
                raise ValueError(
                    f"{constraint.get('id')} references missing wall {wall_id}"
                )


def _accepted_entity(entity: dict[str, Any], reviewer: str, correction_id: str) -> None:
    entity["confidence"] = 1.0
    entity["uncertainty"] = 0.0
    entity["review_state"] = "accepted"
    entity["model_version"] = "human-correction"
    entity["reviewed_by"] = reviewer
    entity["correction_id"] = correction_id


def apply_graph_corrections(
    graph: dict[str, Any],
    correction_set: GraphCorrectionSet,
) -> dict[str, Any]:
    current_hash = graph_content_hash(graph)
    if correction_set.expected_graph_sha256 != current_hash:
        raise ValueError(
            "stale correction set: expected "
            f"{correction_set.expected_graph_sha256}, current {current_hash}"
        )
    corrected = copy.deepcopy(graph)
    log: list[dict[str, Any]] = list(corrected.get("correction_log", []))
    seen_operation_ids = {str(item.get("id")) for item in log}
    for operation in correction_set.operations:
        if operation.id in seen_operation_ids:
            raise ValueError(f"duplicate correction id: {operation.id}")
        collection = corrected.setdefault(operation.collection, [])
        if not isinstance(collection, list):
            raise ValueError(f"graph collection is not a list: {operation.collection}")
        if operation.action == "add":
            if any(item.get("id") == operation.entity_id for item in collection):
                raise ValueError(f"entity already exists: {operation.entity_id}")
            entity = {"id": operation.entity_id, **copy.deepcopy(operation.changes)}
            _accepted_entity(entity, correction_set.reviewer, operation.id)
            collection.append(entity)
        else:
            index, entity = _find_entity(collection, operation.entity_id)
            if operation.action == "delete":
                del collection[index]
            else:
                if operation.action == "update":
                    entity.update(copy.deepcopy(operation.changes))
                _accepted_entity(entity, correction_set.reviewer, operation.id)
        log.append(
            {
                "id": operation.id,
                "action": operation.action,
                "collection": operation.collection,
                "entity_id": operation.entity_id,
                "reason": operation.reason,
                "reviewer": correction_set.reviewer,
                "changes": copy.deepcopy(operation.changes),
            }
        )
        seen_operation_ids.add(operation.id)
    corrected["correction_log"] = log
    _check_references(corrected)
    verification = PlanGraphVerifier().verify(corrected)
    corrected["verification"] = verification.model_dump(mode="json")
    pipeline = corrected.setdefault("pipeline", {})
    pipeline["review_required"] = verification.review_required
    pipeline["release_allowed"] = verification.release_allowed
    pipeline["content_sha256"] = graph_content_hash(corrected)
    return corrected


def correction_summary(operations: Iterable[GraphCorrection]) -> dict[str, int]:
    summary = {"add": 0, "update": 0, "delete": 0, "accept": 0}
    for operation in operations:
        summary[operation.action] += 1
    return summary
