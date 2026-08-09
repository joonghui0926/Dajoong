from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.spatial import Delaunay

from .cad_families import FAMILY_MANIFESTS, parametric_family_parts
from .hashing import sha256_file, sha256_json

JSON_CHUNK = b"JSON"
BIN_CHUNK = b"BIN\x00"


def _pad4(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing = (previous[0] - current[0]) * (point[1] - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if point[0] < crossing:
                inside = not inside
        previous = current
    return inside


def _triangulate(polygon: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    if len(polygon) == 3:
        return [(0, 1, 2)]
    if len(polygon) < 3:
        return []
    try:
        candidates = Delaunay(polygon).simplices.tolist()
    except Exception:
        return [(0, index, index + 1) for index in range(1, len(polygon) - 1)]
    output = []
    for triangle in candidates:
        centroid = (
            sum(polygon[index][0] for index in triangle) / 3,
            sum(polygon[index][1] for index in triangle) / 3,
        )
        if _point_in_polygon(centroid, polygon):
            output.append(tuple(int(index) for index in triangle))
    return output


def _box(
    vertices: list[tuple[float, float, float]],
    indices: list[int],
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_rad: float = 0.0,
) -> None:
    cx, cy, cz = center
    hx, hy, hz = (max(1e-5, value) / 2 for value in size)
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    start = len(vertices)
    for x, y, z in (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ):
        vertices.append((cx + x * cosine - y * sine, cy + x * sine + y * cosine, cz + z))
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    )
    indices.extend(start + item for face in faces for item in face)


class _GlbBuilder:
    def __init__(self) -> None:
        self.binary = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []

    def material(
        self,
        name: str,
        color: tuple[float, float, float, float],
        *,
        roughness: float = 0.72,
    ) -> int:
        material: dict[str, Any] = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(color),
                "metallicFactor": 0.0,
                "roughnessFactor": roughness,
            },
            "doubleSided": True,
        }
        if color[3] < 1:
            material["alphaMode"] = "BLEND"
        self.materials.append(material)
        return len(self.materials) - 1

    def node(
        self,
        name: str,
        vertices: list[tuple[float, float, float]],
        indices: list[int],
        material: int,
        extras: dict[str, Any],
    ) -> int:
        if not vertices or not indices:
            raise ValueError(f"GLB node {name!r} has no geometry")
        position_blob = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
        index_blob = b"".join(struct.pack("<I", index) for index in indices)
        position_offset = len(self.binary)
        self.binary.extend(_pad4(position_blob, b"\x00"))
        index_offset = len(self.binary)
        self.binary.extend(_pad4(index_blob, b"\x00"))

        position_view = len(self.buffer_views)
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": len(position_blob),
                "target": 34962,
            }
        )
        index_view = len(self.buffer_views)
        self.buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_blob),
                "target": 34963,
            }
        )
        position_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": [min(item[axis] for item in vertices) for axis in range(3)],
                "max": [max(item[axis] for item in vertices) for axis in range(3)],
            }
        )
        index_accessor = len(self.accessors)
        self.accessors.append(
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": len(indices),
                "type": "SCALAR",
                "min": [min(indices)],
                "max": [max(indices)],
            }
        )
        mesh_index = len(self.meshes)
        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "indices": index_accessor,
                        "material": material,
                    }
                ],
            }
        )
        self.nodes.append({"name": name, "mesh": mesh_index, "extras": extras})
        return len(self.nodes) - 1

    def write(self, path: Path, *, scene_extras: dict[str, Any]) -> None:
        binary = _pad4(bytes(self.binary), b"\x00")
        document = {
            "asset": {
                "version": "2.0",
                "generator": "Dajoong Plan2BIM editable viewer derivative v1",
                "extras": {
                    "canonicalAuthoringFormat": "IFC4.3",
                    "upAxis": "Z",
                    "units": "metre",
                },
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Dajoong BIM",
                    "nodes": list(range(len(self.nodes))),
                    "extras": scene_extras,
                }
            ],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(binary)}],
        }
        json_blob = _pad4(
            json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            b" ",
        )
        total = 12 + 8 + len(json_blob) + 8 + len(binary)
        payload = b"".join(
            (
                struct.pack("<4sII", b"glTF", 2, total),
                struct.pack("<I4s", len(json_blob), JSON_CHUNK),
                json_blob,
                struct.pack("<I4s", len(binary), BIN_CHUNK),
                binary,
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(path.suffix + ".tmp")
        staging.write_bytes(payload)
        staging.replace(path)


def _extras(collection: str, entity: dict[str, Any], editable: list[str]) -> dict[str, Any]:
    return {
        "buili": {
            "collection": collection,
            "entityId": str(entity.get("id") or entity.get("source_entity_id")),
            "levelId": str(entity.get("level_id") or ""),
            "familyId": str(entity.get("family_id") or ""),
            "reviewState": str(entity.get("review_state") or "review_required"),
            "confidence": float(entity.get("confidence") or 0.0),
            "sourceRefIds": list(entity.get("source_ref_ids") or []),
            "editableFields": editable,
            "properties": {
                key: entity[key]
                for key in editable
                if key in entity
            },
        }
    }


def export_editable_glb(graph: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Create a colored, selectable GLB derivative with one node per BIM element."""

    destination = Path(path).expanduser().resolve()
    builder = _GlbBuilder()
    materials = {
        "wall": builder.material("Warm white wall", (0.91, 0.91, 0.87, 1.0)),
        "floor": builder.material("Natural floor", (0.58, 0.48, 0.34, 1.0)),
        "door": builder.material("Timber door", (0.35, 0.22, 0.14, 1.0)),
        "window": builder.material("Glazing", (0.30, 0.49, 0.46, 0.58), roughness=0.25),
        "architectural": builder.material("Architectural object", (0.44, 0.40, 0.32, 1.0)),
        "electrical": builder.material("Electrical", (0.83, 0.68, 0.32, 1.0)),
        "mechanical": builder.material("Mechanical", (0.30, 0.47, 0.43, 1.0)),
        "plumbing": builder.material("Plumbing", (0.76, 0.79, 0.76, 1.0)),
        "fire": builder.material("Fire protection", (0.58, 0.24, 0.20, 1.0)),
        "vertical": builder.material("Vertical circulation", (0.10, 0.52, 0.66, 1.0)),
        "vertical_glass": builder.material(
            "Lift enclosure", (0.23, 0.57, 0.65, 0.42), roughness=0.22
        ),
    }
    levels = {
        str(item["id"]): float(item.get("elevation_m") or 0.0) for item in graph.get("levels") or []
    }
    walls = {str(item["id"]): item for item in graph.get("walls") or []}
    openings_by_wall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for opening in graph.get("openings") or []:
        openings_by_wall[str(opening.get("wall_id"))].append(opening)

    for room in graph.get("rooms") or []:
        polygon = [(float(point[0]), float(point[1])) for point in room.get("polygon") or []]
        triangles = _triangulate(polygon)
        if not triangles:
            continue
        elevation = levels.get(str(room.get("level_id")), 0.0) + 0.015
        vertices = [(x, y, elevation) for x, y in polygon]
        indices = [index for triangle in triangles for index in triangle]
        builder.node(
            f"room::{room['id']}",
            vertices,
            indices,
            materials["floor"],
            _extras("rooms", room, ["name", "occupancy", "polygon", "floor_finish_type"]),
        )

    for wall_id, wall in walls.items():
        start = tuple(float(value) for value in wall["from"][:2])
        end = tuple(float(value) for value in wall["to"][:2])
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        yaw = math.atan2(dy, dx)
        ux, uy = dx / length, dy / length
        thickness = float(wall.get("thickness_m") or 0.12)
        height = float(wall.get("height_m") or 3.0)
        elevation = levels.get(str(wall.get("level_id")), 0.0)
        attached = []
        for opening in openings_by_wall.get(wall_id, []):
            offset = float(opening.get("x_m") or 0.0)
            width = float(opening.get("width_m") or 0.9)
            attached.append(
                (max(0.0, offset - width / 2), min(length, offset + width / 2), opening)
            )
        attached.sort(key=lambda item: item[0])
        vertices: list[tuple[float, float, float]] = []
        indices: list[int] = []
        cursor = 0.0
        for lower, upper, opening in attached:
            if lower > cursor:
                segment_start, segment_end = cursor, lower
                center_offset = (segment_start + segment_end) / 2
                _box(
                    vertices,
                    indices,
                    center=(
                        start[0] + ux * center_offset,
                        start[1] + uy * center_offset,
                        elevation + height / 2,
                    ),
                    size=(segment_end - segment_start, thickness, height),
                    yaw_rad=yaw,
                )
            sill = float(opening.get("sill_height_m") or 0.0)
            opening_height = min(height, float(opening.get("height_m") or 2.1))
            center_offset = (lower + upper) / 2
            if sill > 1e-4:
                _box(
                    vertices,
                    indices,
                    center=(
                        start[0] + ux * center_offset,
                        start[1] + uy * center_offset,
                        elevation + sill / 2,
                    ),
                    size=(upper - lower, thickness, sill),
                    yaw_rad=yaw,
                )
            top_base = sill + opening_height
            if top_base < height - 1e-4:
                _box(
                    vertices,
                    indices,
                    center=(
                        start[0] + ux * center_offset,
                        start[1] + uy * center_offset,
                        elevation + (top_base + height) / 2,
                    ),
                    size=(upper - lower, thickness, height - top_base),
                    yaw_rad=yaw,
                )
            cursor = max(cursor, upper)
        if cursor < length:
            segment_start, segment_end = cursor, length
            center_offset = (segment_start + segment_end) / 2
            _box(
                vertices,
                indices,
                center=(
                    start[0] + ux * center_offset,
                    start[1] + uy * center_offset,
                    elevation + height / 2,
                ),
                size=(segment_end - segment_start, thickness, height),
                yaw_rad=yaw,
            )
        if vertices:
            builder.node(
                f"wall::{wall_id}",
                vertices,
                indices,
                materials["wall"],
                _extras(
                    "walls",
                    wall,
                    ["from", "to", "thickness_m", "height_m", "wall_type", "material"],
                ),
            )

    for opening in graph.get("openings") or []:
        wall = walls.get(str(opening.get("wall_id")))
        if wall is None:
            continue
        start = tuple(float(value) for value in wall["from"][:2])
        end = tuple(float(value) for value in wall["to"][:2])
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        yaw = math.atan2(dy, dx)
        offset = float(opening.get("x_m") or 0.0)
        width = float(opening.get("width_m") or 0.9)
        height = float(opening.get("height_m") or 2.1)
        sill = float(opening.get("sill_height_m") or 0.0)
        elevation = levels.get(str(opening.get("level_id")), 0.0)
        center = (start[0] + ux * offset, start[1] + uy * offset)
        kind = "window" if str(opening.get("type")) == "window" else "door"
        vertices, indices = [], []
        if kind == "door":
            _box(
                vertices,
                indices,
                center=(center[0], center[1], elevation + height / 2),
                size=(
                    width * 0.94,
                    max(0.035, float(wall.get("thickness_m") or 0.12) * 0.22),
                    height * 0.97,
                ),
                yaw_rad=yaw,
            )
        else:
            frame = max(0.035, min(width, height) * 0.07)
            depth = max(0.035, float(wall.get("thickness_m") or 0.12) * 0.35)
            for x_offset, part_width, z_offset, part_height in (
                (-width / 2 + frame / 2, frame, 0.0, height),
                (width / 2 - frame / 2, frame, 0.0, height),
                (0.0, width - 2 * frame, -height / 2 + frame / 2, frame),
                (0.0, width - 2 * frame, height / 2 - frame / 2, frame),
            ):
                _box(
                    vertices,
                    indices,
                    center=(
                        center[0] + ux * x_offset,
                        center[1] + uy * x_offset,
                        elevation + sill + height / 2 + z_offset,
                    ),
                    size=(part_width, depth, part_height),
                    yaw_rad=yaw,
                )
        builder.node(
            f"{kind}::{opening['id']}",
            vertices,
            indices,
            materials[kind],
            _extras(
                "openings",
                opening,
                [
                    "type",
                    "wall_id",
                    "x_m",
                    "width_m",
                    "height_m",
                    "sill_height_m",
                    "family_id",
                    "operation_type",
                    "handing",
                    "swing_side",
                ],
            ),
        )

    for fixture in graph.get("fixtures") or []:
        family_id = str(fixture.get("family_id") or fixture.get("type") or "unknown")
        discipline = str(fixture.get("discipline") or "architectural")
        if discipline not in materials:
            discipline = "architectural"
        center = tuple(float(value) for value in (fixture.get("center_m") or (0, 0))[:2])
        size = tuple(float(value) for value in (fixture.get("size_m") or (0.2, 0.2, 0.2))[:3])
        elevation = levels.get(str(fixture.get("level_id")), 0.0) + float(
            fixture.get("base_elevation_m") or 0.0
        )
        yaw = math.radians(float(fixture.get("yaw_deg") or 0.0))
        vertices, indices = [], []
        if family_id in FAMILY_MANIFESTS:
            for part in parametric_family_parts(family_id, size):
                local_x, local_y, local_z = part.center
                cosine, sine = math.cos(yaw), math.sin(yaw)
                _box(
                    vertices,
                    indices,
                    center=(
                        center[0] + local_x * cosine - local_y * sine,
                        center[1] + local_x * sine + local_y * cosine,
                        elevation + local_z + part.size[2] / 2,
                    ),
                    size=part.size,
                    yaw_rad=yaw,
                )
        else:
            _box(
                vertices,
                indices,
                center=(center[0], center[1], elevation + size[2] / 2),
                size=size,
                yaw_rad=yaw,
            )
        builder.node(
            f"fixture::{fixture['id']}::{family_id}",
            vertices,
            indices,
            materials[discipline],
            _extras(
                "fixtures",
                fixture,
                ["family_id", "room_id", "center_m", "size_m", "yaw_deg", "material"],
            ),
        )

    for connection in graph.get("vertical_connections") or []:
        connection_id = str(connection.get("id") or "vertical")
        kind = str(connection.get("type") or connection.get("kind") or "stair")
        from_level = str(connection.get("from_level_id") or "")
        to_level = str(connection.get("to_level_id") or "")
        if from_level not in levels or to_level not in levels:
            continue
        lower, upper = levels[from_level], levels[to_level]
        rise = upper - lower
        if rise <= 1e-6:
            continue
        center_value = connection.get("center_m") or (0.0, 0.0)
        center = float(center_value[0]), float(center_value[1])
        footprint_value = connection.get("footprint_m") or (1.2, 3.0)
        footprint = max(0.1, float(footprint_value[0])), max(0.1, float(footprint_value[1]))
        yaw = math.radians(float(connection.get("yaw_deg") or 0.0))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        vertices: list[tuple[float, float, float]] = []
        indices: list[int] = []
        if kind in {"stair", "escalator", "ramp"}:
            step_count = max(3, min(40, round(rise / 0.18)))
            tread = footprint[1] / step_count
            for step in range(step_count):
                local_y = -footprint[1] / 2 + tread * (step + 0.5)
                step_height = rise * (step + 1) / step_count
                _box(
                    vertices,
                    indices,
                    center=(
                        center[0] - local_y * sine,
                        center[1] + local_y * cosine,
                        lower + step_height / 2,
                    ),
                    size=(footprint[0], tread, step_height),
                    yaw_rad=yaw,
                )
        else:
            _box(
                vertices,
                indices,
                center=(center[0], center[1], lower + rise / 2),
                size=(footprint[0], footprint[1], rise),
                yaw_rad=yaw,
            )
        render_entity = {**connection, "level_id": from_level}
        builder.node(
            f"vertical::{connection_id}::{kind}",
            vertices,
            indices,
            materials["vertical_glass" if kind == "elevator" else "vertical"],
            _extras(
                "vertical_connections",
                render_entity,
                [
                    "type",
                    "from_level_id",
                    "to_level_id",
                    "shaft_id",
                    "center_m",
                    "footprint_m",
                    "yaw_deg",
                ],
            ),
        )

    graph_hash = str((graph.get("pipeline") or {}).get("content_sha256") or sha256_json(graph))
    builder.write(
        destination,
        scene_extras={
            "planGraphSha256": graph_hash,
            "viewerDerivative": True,
            "editableInProduct": True,
            "canonicalBim": destination.with_suffix(".ifc").name,
        },
    )
    manifest = {
        "schema_version": "buili.editable-glb-manifest.v1",
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "nodes": len(builder.nodes),
        "meshes": len(builder.meshes),
        "materials": len(builder.materials),
        "plan_graph_sha256": graph_hash,
        "canonical_authoring_format": "IFC4.3",
        "scope": "colored selectable web derivative; IFC remains canonical BIM",
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return manifest
