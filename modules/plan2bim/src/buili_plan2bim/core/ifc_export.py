from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cad_families import (
    approved_family_asset_sha256,
    parametric_family_parts,
)
from .plan_graph_verification import PlanGraphCertificate, PlanGraphVerifier

IFC_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def _ifc_guid(namespace: str) -> str:
    """Return a deterministic 22-character IFC-compatible compressed UUID."""
    value = uuid.uuid5(uuid.NAMESPACE_URL, f"https://dajoong.ai/bim/{namespace}").int
    encoded = []
    for _ in range(22):
        encoded.append(IFC_ALPHABET[value & 0x3F])
        value >>= 6
    return "".join(reversed(encoded))


def _string(value: Any) -> str:
    if value is None or value == "":
        return "$"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _real(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("IFC numeric values must be finite")
    rendered = f"{numeric:.9f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered += "."
    return rendered


def _tuple(values: Iterable[Any]) -> str:
    return "(" + ",".join(str(value) for value in values) + ")"


@dataclass
class StepWriter:
    entities: list[str] = field(default_factory=list)

    def add(self, entity: str, *arguments: Any) -> int:
        self.entities.append(f"{entity.upper()}({_tuple(arguments)[1:-1]})")
        return len(self.entities)

    @staticmethod
    def ref(index: int) -> str:
        return f"#{index}"

    def render(self, *, filename: str) -> str:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        data = "\n".join(
            f"#{index}={entity};" for index, entity in enumerate(self.entities, start=1)
        )
        return (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('ViewDefinition [ReferenceView_V1.2]'),'2;1');\n"
            f"FILE_NAME({_string(filename)},{_string(now)},('Dajoong'),('Dajoong'),"
            "'Dajoong Spatial Compiler','Dajoong Spatial Compiler','');\n"
            "FILE_SCHEMA(('IFC4'));\n"
            "ENDSEC;\n"
            "DATA;\n"
            f"{data}\n"
            "ENDSEC;\n"
            "END-ISO-10303-21;\n"
        )


@dataclass(frozen=True)
class IfcStyle:
    name: str
    rgb: tuple[float, float, float]
    transparency: float = 0.0


STYLES = {
    "wall": IfcStyle("White painted architectural walls", (0.95, 0.95, 0.93)),
    "room": IfcStyle("Spatial volumes", (0.58, 0.69, 0.62), 0.82),
    "floor_oak": IfcStyle("Natural oak floor", (0.79, 0.61, 0.38)),
    "floor_porcelain": IfcStyle("Porcelain tile floor", (0.85, 0.84, 0.81)),
    "floor_sauna_wood": IfcStyle("Sauna timber floor", (0.72, 0.55, 0.36)),
    "floor_deck": IfcStyle("Exterior deck floor", (0.65, 0.53, 0.38)),
    "floor_service": IfcStyle("Service-room floor", (0.76, 0.78, 0.76)),
    "floor_default": IfcStyle("Unspecified finish floor", (0.79, 0.77, 0.72)),
    "door": IfcStyle("Doors", (0.39, 0.48, 0.42)),
    "window": IfcStyle("Windows", (0.36, 0.58, 0.55), 0.45),
    "electrical": IfcStyle("Electrical", (0.49, 0.57, 0.39)),
    "mechanical": IfcStyle("Mechanical", (0.28, 0.48, 0.44)),
    "plumbing": IfcStyle("Plumbing", (0.36, 0.49, 0.46)),
    "fire": IfcStyle("Fire protection", (0.55, 0.34, 0.31)),
    "structural": IfcStyle("Structural", (0.39, 0.43, 0.41)),
    "fixture": IfcStyle("Architectural fixtures", (0.60, 0.62, 0.57)),
}


def _discipline(kind: str) -> str:
    lowered = kind.lower()
    if any(token in lowered for token in ("receptacle", "outlet", "switch", "light", "panel")):
        return "electrical"
    if any(token in lowered for token in ("duct", "diffuser", "ahu", "mechanical")):
        return "mechanical"
    if any(token in lowered for token in ("plumb", "water", "hydronic", "drain", "sink")):
        return "plumbing"
    if any(token in lowered for token in ("fire", "sprinkler", "smoke")):
        return "fire"
    if any(token in lowered for token in ("column", "beam", "structural")):
        return "structural"
    return "fixture"


def _point2(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected a metric 2D point, received {value!r}")
    return float(value[0]), float(value[1])


def _wall_offset(
    opening: dict[str, Any],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    if isinstance(opening.get("x_m"), (int, float)):
        return float(opening["x_m"])
    center = _point2(opening["center_m"])
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    return ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length


def _opening_family_mesh(
    kind: str, width: float, depth: float, height: float
) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
    """Build a framed door/window family in opening-local coordinates."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    colors: list[list[int]] = []

    def add_box(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[int, int, int],
    ) -> None:
        center_x, center_y, center_z = center
        half_x, half_y, half_z = (value / 2 for value in size)
        offset = len(vertices)
        vertices.extend(
            [
                [center_x - half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y - half_y, center_z - half_z],
                [center_x + half_x, center_y + half_y, center_z - half_z],
                [center_x - half_x, center_y + half_y, center_z - half_z],
                [center_x - half_x, center_y - half_y, center_z + half_z],
                [center_x + half_x, center_y - half_y, center_z + half_z],
                [center_x + half_x, center_y + half_y, center_z + half_z],
                [center_x - half_x, center_y + half_y, center_z + half_z],
            ]
        )
        box_faces = (
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
        )
        faces.extend([[offset + index for index in face] for face in box_faces])
        colors.extend([list(color)] * len(box_faces))

    frame = min(0.085, max(0.045, min(width, height) * 0.065))
    frame_depth = min(max(depth * 0.62, 0.055), 0.12)
    inner_width = max(0.05, width - frame * 2)
    inner_height = max(0.05, height - frame * 2)
    frame_color = (67, 77, 70)
    add_box((-width / 2 + frame / 2, 0.0, height / 2), (frame, frame_depth, height), frame_color)
    add_box((width / 2 - frame / 2, 0.0, height / 2), (frame, frame_depth, height), frame_color)
    add_box((0.0, 0.0, height - frame / 2), (inner_width, frame_depth, frame), frame_color)

    if kind == "window":
        add_box((0.0, 0.0, frame / 2), (inner_width, frame_depth, frame), frame_color)
        glass_depth = min(0.024, max(0.012, depth * 0.12))
        add_box(
            (0.0, 0.0, height / 2),
            (inner_width, glass_depth, inner_height),
            (145, 186, 180),
        )
        if width >= 1.35:
            mullion = min(0.055, frame * 0.72)
            add_box((0.0, 0.0, height / 2), (mullion, frame_depth, inner_height), frame_color)
    else:
        leaf_depth = min(0.05, max(0.035, depth * 0.22))
        leaf_height = max(0.05, height - frame - 0.015)
        leaf_color = (123, 99, 70)
        add_box(
            (0.0, 0.0, leaf_height / 2 + 0.01),
            (inner_width, leaf_depth, leaf_height),
            leaf_color,
        )
        panel_depth = 0.012
        panel_width = inner_width * 0.72
        for panel_z in (height * 0.32, height * 0.69):
            add_box(
                (0.0, -leaf_depth / 2 - panel_depth / 2, panel_z),
                (panel_width, panel_depth, height * 0.24),
                (105, 82, 58),
            )
        add_box(
            (inner_width * 0.37, -leaf_depth / 2 - 0.025, height * 0.49),
            (0.045, 0.05, 0.045),
            (169, 151, 105),
        )
    return vertices, faces, colors


def _door_operation_type(opening: dict[str, Any]) -> str:
    operation = str(opening.get("operation_type") or "unknown")
    handing = str(opening.get("handing") or "unknown")
    suffix = "LEFT" if handing == "start" else "RIGHT"
    if operation == "single_swing":
        return f".SINGLE_SWING_{suffix}."
    if operation == "double_swing":
        return ".DOUBLE_DOOR_SINGLE_SWING."
    if operation == "sliding":
        return f".SLIDING_TO_{suffix}."
    if operation == "folding":
        return f".FOLDING_TO_{suffix}."
    return ".NOTDEFINED."


class IfcModelBuilder:
    """Dependency-free IFC4 writer for the verified, parametric PlanGraph subset."""

    def __init__(
        self,
        graph: dict[str, Any],
        certificate: PlanGraphCertificate,
        *,
        draft: bool,
    ) -> None:
        self.graph = graph
        self.certificate = certificate
        self.draft = draft
        self.step = StepWriter()
        self.styles: dict[str, int] = {}
        self.contained_products: dict[str, list[int]] = defaultdict(list)
        self.owner_history = 0
        self.context = 0
        self.storey_placements: dict[str, int] = {}
        self.storeys: dict[str, int] = {}
        self.level_elevations: dict[str, float] = {}
        self.default_level_id = "level-1"
        self.wall_products: dict[
            str, tuple[int, tuple[float, float], tuple[float, float], str]
        ] = {}

    def build(self) -> StepWriter:
        self._project()
        self._materials()
        self._rooms()
        self._walls()
        self._openings()
        self._fixtures()
        self._routes()
        self._vertical_connections()
        for level_id, products in self.contained_products.items():
            if not products:
                continue
            self.step.add(
                "IFCRELCONTAINEDINSPATIALSTRUCTURE",
                _string(_ifc_guid(f"containment/{level_id}")),
                self.step.ref(self.owner_history),
                _string(f"{level_id} containment"),
                "$",
                _tuple(self.step.ref(product) for product in products),
                self.step.ref(self.storeys[level_id]),
            )
        return self.step

    def _project(self) -> None:
        person = self.step.add("IFCPERSON", "$", "$", _string("Dajoong"), "$", "$", "$", "$", "$")
        organization = self.step.add(
            "IFCORGANIZATION", "$", _string("Dajoong"), _string("Verified BIM Compiler"), "$", "$"
        )
        person_org = self.step.add(
            "IFCPERSONANDORGANIZATION", self.step.ref(person), self.step.ref(organization), "$"
        )
        application = self.step.add(
            "IFCAPPLICATION",
            self.step.ref(organization),
            _string("0.2.0"),
            _string("Dajoong Verified BIM Compiler"),
            _string("DAJOONG_BIM"),
        )
        self.owner_history = self.step.add(
            "IFCOWNERHISTORY",
            self.step.ref(person_org),
            self.step.ref(application),
            "$",
            ".ADDED.",
            "$",
            "$",
            "$",
            str(int(time.time())),
        )
        length_unit = self.step.add("IFCSIUNIT", "*", ".LENGTHUNIT.", "$", ".METRE.")
        area_unit = self.step.add("IFCSIUNIT", "*", ".AREAUNIT.", "$", ".SQUARE_METRE.")
        volume_unit = self.step.add("IFCSIUNIT", "*", ".VOLUMEUNIT.", "$", ".CUBIC_METRE.")
        units = self.step.add(
            "IFCUNITASSIGNMENT",
            _tuple(self.step.ref(unit) for unit in (length_unit, area_unit, volume_unit)),
        )
        world = self._axis3((0.0, 0.0, 0.0))
        self.context = self.step.add(
            "IFCGEOMETRICREPRESENTATIONCONTEXT",
            "$",
            _string("Model"),
            "3",
            "1.E-05",
            self.step.ref(world),
            "$",
        )
        project_name = str(self.graph.get("project_id") or "Dajoong BIM")
        status = "DRAFT / NOT FOR CONSTRUCTION" if self.draft else "VERIFIED RELEASE"
        project = self.step.add(
            "IFCPROJECT",
            _string(_ifc_guid(f"project/{project_name}")),
            self.step.ref(self.owner_history),
            _string(project_name),
            _string(status),
            "$",
            "$",
            "$",
            _tuple([self.step.ref(self.context)]),
            self.step.ref(units),
        )
        site_placement = self._local_placement(None, (0.0, 0.0, 0.0))
        site = self.step.add(
            "IFCSITE",
            _string(_ifc_guid(f"site/{project_name}")),
            self.step.ref(self.owner_history),
            _string("Site"),
            "$",
            "$",
            self.step.ref(site_placement),
            "$",
            "$",
            ".ELEMENT.",
            "$",
            "$",
            "$",
            "$",
            "$",
        )
        building_placement = self._local_placement(site_placement, (0.0, 0.0, 0.0))
        building = self.step.add(
            "IFCBUILDING",
            _string(_ifc_guid(f"building/{project_name}")),
            self.step.ref(self.owner_history),
            _string("Building"),
            "$",
            "$",
            self.step.ref(building_placement),
            "$",
            "$",
            ".ELEMENT.",
            "$",
            "$",
            "$",
        )
        levels = list(self.graph.get("levels") or [])
        if not levels:
            levels = [
                {
                    "id": "level-1",
                    "name": "Level 1",
                    "elevation_m": 0.0,
                    "nominal_height_m": 3.0,
                }
            ]
        self.default_level_id = str(levels[0].get("id") or "level-1")
        for index, level in enumerate(levels):
            level_id = str(level.get("id") or f"level-{index + 1}")
            elevation = float(level.get("elevation_m") or 0.0)
            placement = self._local_placement(building_placement, (0.0, 0.0, elevation))
            storey = self.step.add(
                "IFCBUILDINGSTOREY",
                _string(_ifc_guid(f"storey/{project_name}/{level_id}")),
                self.step.ref(self.owner_history),
                _string(level.get("name") or level_id),
                _string("Evidence-grounded building storey"),
                "$",
                self.step.ref(placement),
                "$",
                _string(level_id),
                ".ELEMENT.",
                _real(elevation),
            )
            self.storey_placements[level_id] = placement
            self.storeys[level_id] = storey
            self.level_elevations[level_id] = elevation
        self.step.add(
            "IFCRELAGGREGATES",
            _string(_ifc_guid("aggregate/project-site")),
            self.step.ref(self.owner_history),
            "$",
            "$",
            self.step.ref(project),
            _tuple([self.step.ref(site)]),
        )
        self.step.add(
            "IFCRELAGGREGATES",
            _string(_ifc_guid("aggregate/site-building")),
            self.step.ref(self.owner_history),
            "$",
            "$",
            self.step.ref(site),
            _tuple([self.step.ref(building)]),
        )
        self.step.add(
            "IFCRELAGGREGATES",
            _string(_ifc_guid("aggregate/building-storeys")),
            self.step.ref(self.owner_history),
            "$",
            "$",
            self.step.ref(building),
            _tuple(self.step.ref(storey) for storey in self.storeys.values()),
        )
        self._properties(
            project,
            "Pset_DajoongVerification",
            {
                "CertificateSha256": self.certificate.content_sha256,
                "SourceContentSha256": self.certificate.source_content_sha256,
                "ReleaseAllowed": str(self.certificate.release_allowed),
                "ReviewRequired": str(self.certificate.review_required),
                "ArtifactStatus": status,
            },
        )

    def _materials(self) -> None:
        for key, style in STYLES.items():
            colour = self.step.add(
                "IFCCOLOURRGB",
                _string(style.name),
                *(_real(component) for component in style.rgb),
            )
            shading = self.step.add(
                "IFCSURFACESTYLESHADING",
                self.step.ref(colour),
                _real(style.transparency),
            )
            self.styles[key] = self.step.add(
                "IFCSURFACESTYLE",
                _string(style.name),
                ".BOTH.",
                _tuple([self.step.ref(shading)]),
            )

    def _axis2(self, point: tuple[float, float]) -> int:
        location = self.step.add("IFCCARTESIANPOINT", _tuple(_real(value) for value in point))
        return self.step.add("IFCAXIS2PLACEMENT2D", self.step.ref(location), "$")

    def _axis3(
        self,
        point: tuple[float, float, float],
        direction: tuple[float, float, float] | None = None,
    ) -> int:
        location = self.step.add("IFCCARTESIANPOINT", _tuple(_real(value) for value in point))
        axis = self.step.add("IFCDIRECTION", _tuple([_real(0), _real(0), _real(1)]))
        ref = "$"
        if direction is not None:
            ref_direction = self.step.add(
                "IFCDIRECTION", _tuple(_real(value) for value in direction)
            )
            ref = self.step.ref(ref_direction)
        return self.step.add(
            "IFCAXIS2PLACEMENT3D", self.step.ref(location), self.step.ref(axis), ref
        )

    def _local_placement(
        self,
        parent: int | None,
        point: tuple[float, float, float],
        direction: tuple[float, float, float] | None = None,
    ) -> int:
        axis = self._axis3(point, direction)
        return self.step.add(
            "IFCLOCALPLACEMENT",
            self.step.ref(parent) if parent else "$",
            self.step.ref(axis),
        )

    def _level_id(self, entity: dict[str, Any]) -> str:
        level_id = str(entity.get("level_id") or self.default_level_id)
        if level_id not in self.storey_placements:
            raise ValueError(f"Entity refers to unknown IFC storey {level_id!r}")
        return level_id

    def _rectangle_representation(
        self,
        size: tuple[float, float, float],
        style: str,
    ) -> int:
        profile_placement = self._axis2((0.0, 0.0))
        profile = self.step.add(
            "IFCRECTANGLEPROFILEDEF",
            ".AREA.",
            "$",
            self.step.ref(profile_placement),
            _real(size[0]),
            _real(size[1]),
        )
        solid_placement = self._axis3((0.0, 0.0, 0.0))
        extrusion = self.step.add("IFCDIRECTION", _tuple([_real(0), _real(0), _real(1)]))
        solid = self.step.add(
            "IFCEXTRUDEDAREASOLID",
            self.step.ref(profile),
            self.step.ref(solid_placement),
            self.step.ref(extrusion),
            _real(size[2]),
        )
        self.step.add(
            "IFCSTYLEDITEM",
            self.step.ref(solid),
            _tuple([self.step.ref(self.styles[style])]),
            "$",
        )
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("SweptSolid"),
            _tuple([self.step.ref(solid)]),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _polygon_representation(
        self,
        polygon: list[tuple[float, float]],
        height: float,
        style: str,
    ) -> int:
        points = [
            self.step.add("IFCCARTESIANPOINT", _tuple(_real(value) for value in point))
            for point in polygon
        ]
        points.append(points[0])
        polyline = self.step.add("IFCPOLYLINE", _tuple(self.step.ref(point) for point in points))
        profile = self.step.add(
            "IFCARBITRARYCLOSEDPROFILEDEF", ".AREA.", "$", self.step.ref(polyline)
        )
        placement = self._axis3((0.0, 0.0, 0.0))
        direction = self.step.add("IFCDIRECTION", _tuple([_real(0), _real(0), _real(1)]))
        solid = self.step.add(
            "IFCEXTRUDEDAREASOLID",
            self.step.ref(profile),
            self.step.ref(placement),
            self.step.ref(direction),
            _real(height),
        )
        self.step.add(
            "IFCSTYLEDITEM",
            self.step.ref(solid),
            _tuple([self.step.ref(self.styles[style])]),
            "$",
        )
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("SweptSolid"),
            _tuple([self.step.ref(solid)]),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _mesh_representation(
        self,
        vertices: list[Any],
        faces: list[Any],
        style: str,
        face_colors: list[Any] | None = None,
    ) -> int:
        """Write a validated local mesh as an IFC4 tessellated face set."""

        if len(vertices) < 4 or len(vertices) > 1_000_000:
            raise ValueError("Licensed mesh requires 4..1,000,000 vertices")
        if len(faces) < 4 or len(faces) > 2_000_000:
            raise ValueError("Licensed mesh requires 4..2,000,000 triangular faces")
        rendered_vertices = []
        for vertex in vertices:
            if not isinstance(vertex, (list, tuple)) or len(vertex) != 3:
                raise ValueError("Licensed mesh vertex must contain exactly three coordinates")
            rendered_vertices.append(_tuple(_real(float(value)) for value in vertex))
        rendered_faces = []
        vertex_count = len(vertices)
        for face in faces:
            if not isinstance(face, (list, tuple)) or len(face) != 3:
                raise ValueError("Licensed mesh face must contain exactly three indices")
            indices = [int(value) for value in face]
            if len(set(indices)) != 3 or min(indices) < 0 or max(indices) >= vertex_count:
                raise ValueError("Licensed mesh contains an invalid triangular face")
            # IFC indices are one-based.
            rendered_faces.append(_tuple(index + 1 for index in indices))
        point_list = self.step.add(
            "IFCCARTESIANPOINTLIST3D",
            _tuple(rendered_vertices),
        )
        face_set = self.step.add(
            "IFCTRIANGULATEDFACESET",
            self.step.ref(point_list),
            "$",
            ".F.",
            _tuple(rendered_faces),
            "$",
        )
        indexed_color_written = False
        if face_colors and len(face_colors) == len(faces):
            palette: dict[tuple[int, int, int], int] = {}
            color_indices = []
            for raw_color in face_colors:
                if not isinstance(raw_color, (list, tuple)) or len(raw_color) < 3:
                    raise ValueError("Licensed mesh face color must contain RGB values")
                values = tuple(max(0, min(255, int(value))) for value in raw_color[:3])
                # Four-bit channel quantization keeps IFC files compact while
                # retaining the source material palette.
                quantized = tuple(round(value / 17) * 17 for value in values)
                if quantized not in palette:
                    palette[quantized] = len(palette) + 1
                color_indices.append(palette[quantized])
            if palette:
                ordered = sorted(palette, key=palette.get)
                color_list = self.step.add(
                    "IFCCOLOURRGBLIST",
                    _tuple(_tuple(_real(channel / 255) for channel in color) for color in ordered),
                )
                self.step.add(
                    "IFCINDEXEDCOLOURMAP",
                    self.step.ref(face_set),
                    "$",
                    self.step.ref(color_list),
                    _tuple(color_indices),
                )
                indexed_color_written = True
        if not indexed_color_written:
            self.step.add(
                "IFCSTYLEDITEM",
                self.step.ref(face_set),
                _tuple([self.step.ref(self.styles[style])]),
                "$",
            )
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("Tessellation"),
            _tuple([self.step.ref(face_set)]),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _parametric_family_representation(
        self,
        family_id: str,
        size: tuple[float, float, float],
        style: str,
    ) -> int:
        solids = []
        for part in parametric_family_parts(family_id, size):
            profile_placement = self._axis2((0.0, 0.0))
            profile = self.step.add(
                "IFCRECTANGLEPROFILEDEF",
                ".AREA.",
                "$",
                self.step.ref(profile_placement),
                _real(part.size[0]),
                _real(part.size[1]),
            )
            solid_placement = self._axis3(part.center)
            extrusion = self.step.add("IFCDIRECTION", _tuple([_real(0), _real(0), _real(1)]))
            solid = self.step.add(
                "IFCEXTRUDEDAREASOLID",
                self.step.ref(profile),
                self.step.ref(solid_placement),
                self.step.ref(extrusion),
                _real(part.size[2]),
            )
            self.step.add(
                "IFCSTYLEDITEM",
                self.step.ref(solid),
                _tuple([self.step.ref(self.styles[style])]),
                "$",
            )
            solids.append(solid)
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("SweptSolid"),
            _tuple(self.step.ref(solid) for solid in solids),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _swept_disk_representation(
        self,
        points: list[tuple[float, float, float]],
        radius: float,
        style: str,
    ) -> int:
        point_entities = [
            self.step.add("IFCCARTESIANPOINT", _tuple(_real(value) for value in point))
            for point in points
        ]
        directrix = self.step.add(
            "IFCPOLYLINE", _tuple(self.step.ref(point) for point in point_entities)
        )
        solid = self.step.add(
            "IFCSWEPTDISKSOLID",
            self.step.ref(directrix),
            _real(radius),
            "$",
            "$",
            "$",
        )
        self.step.add(
            "IFCSTYLEDITEM",
            self.step.ref(solid),
            _tuple([self.step.ref(self.styles[style])]),
            "$",
        )
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("SweptSolid"),
            _tuple([self.step.ref(solid)]),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _stair_representation(
        self,
        footprint: tuple[float, float],
        rise: float,
        style: str = "structural",
    ) -> int:
        step_count = min(32, max(2, math.ceil(rise / 0.18)))
        step_depth = footprint[1] / step_count
        solids = []
        for index in range(step_count):
            profile_placement = self._axis2((0.0, 0.0))
            profile = self.step.add(
                "IFCRECTANGLEPROFILEDEF",
                ".AREA.",
                "$",
                self.step.ref(profile_placement),
                _real(footprint[0]),
                _real(step_depth),
            )
            center_y = -footprint[1] / 2 + (index + 0.5) * step_depth
            solid_placement = self._axis3((0.0, center_y, 0.0))
            extrusion = self.step.add("IFCDIRECTION", _tuple([_real(0), _real(0), _real(1)]))
            solid = self.step.add(
                "IFCEXTRUDEDAREASOLID",
                self.step.ref(profile),
                self.step.ref(solid_placement),
                self.step.ref(extrusion),
                _real(rise * (index + 1) / step_count),
            )
            self.step.add(
                "IFCSTYLEDITEM",
                self.step.ref(solid),
                _tuple([self.step.ref(self.styles[style])]),
                "$",
            )
            solids.append(solid)
        shape = self.step.add(
            "IFCSHAPEREPRESENTATION",
            self.step.ref(self.context),
            _string("Body"),
            _string("SweptSolid"),
            _tuple(self.step.ref(solid) for solid in solids),
        )
        return self.step.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", _tuple([self.step.ref(shape)]))

    def _rooms(self) -> None:
        for index, room in enumerate(self.graph.get("rooms") or []):
            room_id = str(room.get("id") or f"room-{index}")
            level_id = self._level_id(room)
            polygon = [_point2(point) for point in room["polygon"]]
            placement = self._local_placement(self.storey_placements[level_id], (0.0, 0.0, 0.0))
            finish = str(room.get("floor_finish_type") or "default")
            style = f"floor_{finish}"
            if style not in self.styles:
                style = "floor_default"
            representation = self._polygon_representation(polygon, 0.03, style)
            product = self.step.add(
                "IFCSPACE",
                _string(_ifc_guid(f"room/{room_id}")),
                self.step.ref(self.owner_history),
                _string(room.get("name") or room_id),
                _string("Verified room cycle"),
                "$",
                self.step.ref(placement),
                self.step.ref(representation),
                "$",
                ".ELEMENT.",
                ".INTERNAL.",
                "$",
            )
            self.contained_products[level_id].append(product)
            self._entity_properties(product, room_id, room)

    def _walls(self) -> None:
        for index, wall in enumerate(self.graph.get("walls") or []):
            wall_id = str(wall.get("id") or f"wall-{index}")
            level_id = self._level_id(wall)
            start = _point2(wall.get("from") or wall.get("from_"))
            end = _point2(wall["to"])
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            midpoint = (start[0] + dx / 2, start[1] + dy / 2, 0.0)
            direction = (dx / length, dy / length, 0.0)
            footprint = wall.get("footprint_m") or []
            if len(footprint) >= 3:
                placement = self._local_placement(self.storey_placements[level_id], (0.0, 0.0, 0.0))
                representation = self._polygon_representation(
                    [_point2(point) for point in footprint], float(wall["height_m"]), "wall"
                )
            else:
                placement = self._local_placement(
                    self.storey_placements[level_id], midpoint, direction
                )
                representation = self._rectangle_representation(
                    (length, float(wall["thickness_m"]), float(wall["height_m"])), "wall"
                )
            product = self.step.add(
                "IFCWALL",
                _string(_ifc_guid(f"wall/{wall_id}")),
                self.step.ref(self.owner_history),
                _string(wall_id),
                _string("Evidence-grounded wall"),
                "$",
                self.step.ref(placement),
                self.step.ref(representation),
                _string(wall_id),
                ".STANDARD.",
            )
            self.wall_products[wall_id] = product, start, end, level_id
            self.contained_products[level_id].append(product)
            self._entity_properties(product, wall_id, wall)

    def _openings(self) -> None:
        for index, opening in enumerate(self.graph.get("openings") or []):
            opening_id = str(
                opening.get("source_entity_id") or opening.get("id") or f"opening-{index}"
            )
            wall_id = str(opening["wall_id"])
            wall_product, start, end, wall_level_id = self.wall_products[wall_id]
            level_id = self._level_id(opening)
            if level_id != wall_level_id:
                raise ValueError(f"Opening {opening_id!r} and host wall are on different storeys")
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            direction = (dx / length, dy / length, 0.0)
            offset = _wall_offset(opening, start, end)
            center = (
                start[0] + direction[0] * offset,
                start[1] + direction[1] * offset,
                float(opening.get("sill_height_m", 0.0)),
            )
            width = float(opening["width_m"])
            height = float(opening["height_m"])
            wall = next(item for item in self.graph["walls"] if str(item.get("id")) == wall_id)
            depth = float(wall["thickness_m"]) * 1.1
            opening_placement = self._local_placement(
                self.storey_placements[level_id], center, direction
            )
            opening_representation = self._rectangle_representation((width, depth, height), "wall")
            opening_product = self.step.add(
                "IFCOPENINGELEMENT",
                _string(_ifc_guid(f"opening-void/{opening_id}")),
                self.step.ref(self.owner_history),
                _string(f"Void {opening_id}"),
                "$",
                "$",
                self.step.ref(opening_placement),
                self.step.ref(opening_representation),
                _string(opening_id),
                ".OPENING.",
            )
            self.step.add(
                "IFCRELVOIDSELEMENT",
                _string(_ifc_guid(f"void/{wall_id}/{opening_id}")),
                self.step.ref(self.owner_history),
                "$",
                "$",
                self.step.ref(wall_product),
                self.step.ref(opening_product),
            )
            kind = "window" if str(opening.get("type")) == "window" else "door"
            family_vertices, family_faces, family_colors = _opening_family_mesh(
                kind, width, depth, height
            )
            product_representation = self._mesh_representation(
                family_vertices,
                family_faces,
                kind,
                family_colors,
            )
            if kind == "window":
                product = self.step.add(
                    "IFCWINDOW",
                    _string(_ifc_guid(f"window/{opening_id}")),
                    self.step.ref(self.owner_history),
                    _string(opening_id),
                    _string("Evidence-grounded window"),
                    "$",
                    self.step.ref(opening_placement),
                    self.step.ref(product_representation),
                    _string(opening_id),
                    _real(height),
                    _real(width),
                    ".WINDOW.",
                    ".NOTDEFINED.",
                    "$",
                )
            else:
                product = self.step.add(
                    "IFCDOOR",
                    _string(_ifc_guid(f"door/{opening_id}")),
                    self.step.ref(self.owner_history),
                    _string(opening_id),
                    _string("Evidence-grounded door"),
                    "$",
                    self.step.ref(opening_placement),
                    self.step.ref(product_representation),
                    _string(opening_id),
                    _real(height),
                    _real(width),
                    ".DOOR.",
                    _door_operation_type(opening),
                    "$",
                )
            self.step.add(
                "IFCRELFILLSELEMENT",
                _string(_ifc_guid(f"fill/{opening_id}")),
                self.step.ref(self.owner_history),
                "$",
                "$",
                self.step.ref(opening_product),
                self.step.ref(product),
            )
            self.contained_products[level_id].append(product)
            self._entity_properties(product, opening_id, opening)

    def _fixtures(self) -> None:
        for index, fixture in enumerate(self.graph.get("fixtures") or []):
            fixture_id = str(
                fixture.get("source_entity_id") or fixture.get("id") or f"fixture-{index}"
            )
            level_id = self._level_id(fixture)
            center = _point2(fixture.get("center_m") or (0.0, 0.0))
            kind = str(fixture.get("family_id") or fixture.get("type") or "fixture")
            discipline = str(fixture.get("discipline") or _discipline(kind))
            if discipline not in self.styles:
                discipline = _discipline(kind)
            size_value = fixture.get("size_m") or (0.2, 0.2, 0.2)
            size = tuple(max(1e-5, float(value)) for value in size_value[:3])
            if len(size) != 3:
                raise ValueError(f"Fixture {fixture_id!r} requires a 3D size")
            base_elevation = float(fixture.get("base_elevation_m") or 0.0)
            yaw = math.radians(float(fixture.get("yaw_deg") or 0.0))
            direction = (math.cos(yaw), math.sin(yaw), 0.0)
            placement = self._local_placement(
                self.storey_placements[level_id],
                (center[0], center[1], base_elevation),
                direction,
            )
            footprint = fixture.get("footprint_local_m") or []
            geometry_status = str(fixture.get("geometry_status") or "semantic_marker")
            asset_sha256 = str(fixture.get("asset_sha256") or "")
            if geometry_status == "approved_family":
                expected_asset_sha256 = approved_family_asset_sha256(kind)
                if not expected_asset_sha256 or asset_sha256 != expected_asset_sha256:
                    raise ValueError(
                        f"Fixture {fixture_id!r} has an invalid approved-family asset hash"
                    )
                representation = self._parametric_family_representation(kind, size, discipline)
            elif geometry_status == "licensed_api_asset":
                required_asset_fields = (
                    "asset_uid",
                    "asset_provider",
                    "asset_license",
                    "asset_source_uri",
                    "asset_sha256",
                    "asset_mesh_sha256",
                )
                missing = [field for field in required_asset_fields if not fixture.get(field)]
                if missing:
                    raise ValueError(
                        f"Fixture {fixture_id!r} is missing licensed asset provenance: "
                        + ", ".join(missing)
                    )
                if str(fixture["asset_license"]).lower() not in {"cc0", "by"}:
                    raise ValueError(f"Fixture {fixture_id!r} uses a non-allowlisted asset license")
                representation = self._mesh_representation(
                    list(fixture.get("mesh_vertices") or []),
                    list(fixture.get("mesh_faces") or []),
                    discipline,
                    list(fixture.get("mesh_face_colors") or []),
                )
            elif geometry_status == "native_bim_parametric":
                if not fixture.get("native_generator") or not fixture.get("asset_mesh_sha256"):
                    raise ValueError(
                        f"Fixture {fixture_id!r} is missing native BIM generator provenance"
                    )
                representation = self._mesh_representation(
                    list(fixture.get("mesh_vertices") or []),
                    list(fixture.get("mesh_faces") or []),
                    discipline,
                    list(fixture.get("mesh_face_colors") or []),
                )
            else:
                representation = (
                    self._polygon_representation(
                        [_point2(point) for point in footprint], size[2], discipline
                    )
                    if len(footprint) >= 3
                    else self._rectangle_representation(size, discipline)
                )
            lowered = kind.lower()
            if any(token in lowered for token in ("light", "led", "luminaire")):
                ifc_entity = "IFCLIGHTFIXTURE"
            elif any(token in lowered for token in ("panel", "distribution")):
                ifc_entity = "IFCELECTRICDISTRIBUTIONBOARD"
            elif any(token in lowered for token in ("receptacle", "outlet")):
                ifc_entity = "IFCOUTLET"
            elif any(token in lowered for token in ("diffuser", "air-terminal", "hvac-terminal")):
                ifc_entity = "IFCAIRTERMINAL"
            elif any(token in lowered for token in ("sprinkler", "fire")):
                ifc_entity = "IFCFIRESUPPRESSIONTERMINAL"
            elif any(token in lowered for token in ("sink", "toilet", "plumbing")):
                ifc_entity = "IFCSANITARYTERMINAL"
            else:
                ifc_entity = "IFCBUILDINGELEMENTPROXY"
            product = self.step.add(
                ifc_entity,
                _string(_ifc_guid(f"fixture/{fixture_id}")),
                self.step.ref(self.owner_history),
                _string(fixture_id),
                _string(f"Evidence-grounded {kind} ({geometry_status})"),
                _string(kind),
                self.step.ref(placement),
                self.step.ref(representation),
                _string(fixture_id),
                ".NOTDEFINED." if ifc_entity != "IFCBUILDINGELEMENTPROXY" else ".USERDEFINED.",
            )
            self.contained_products[level_id].append(product)
            self._entity_properties(
                product,
                fixture_id,
                fixture,
                extra={
                    "Discipline": discipline,
                    "GeometryStatus": geometry_status,
                    "FamilyId": kind,
                    "AssetSha256": asset_sha256,
                    "AssetMeshSha256": fixture.get("asset_mesh_sha256", ""),
                    "AssetProvider": fixture.get("asset_provider", ""),
                    "AssetUid": fixture.get("asset_uid", ""),
                    "AssetName": fixture.get("asset_name", ""),
                    "AssetAuthor": fixture.get("asset_author", ""),
                    "AssetLicense": fixture.get("asset_license", ""),
                    "AssetSourceUri": fixture.get("asset_source_uri", ""),
                    "NativeGenerator": fixture.get("native_generator", ""),
                    "AssetAxisSwapped": fixture.get("asset_axis_swapped", ""),
                    "SemanticResolution": fixture.get("semantic_resolution", ""),
                    "MeshVertexCount": len(fixture.get("mesh_vertices") or []),
                    "MeshFaceCount": len(fixture.get("mesh_faces") or []),
                    "ColorEncoding": (
                        "IfcIndexedColourMap"
                        if fixture.get("mesh_face_colors")
                        else "discipline_surface_style"
                    ),
                    "Material": fixture.get("material", ""),
                },
            )

    def _routes(self) -> None:
        systems: dict[str, list[int]] = defaultdict(list)
        for index, route in enumerate(self.graph.get("routes") or []):
            route_id = str(route.get("id") or f"route-{index}")
            level_id = self._level_id(route)
            raw_points = route.get("points_m") or []
            if len(raw_points) < 2:
                raise ValueError(f"Route {route_id!r} requires at least two 3D points")
            points = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) != 3:
                    raise ValueError(f"Route {route_id!r} has an invalid 3D point")
                points.append(tuple(float(value) for value in point))
            discipline = str(route.get("discipline") or _discipline(str(route.get("type"))))
            if discipline not in self.styles:
                discipline = _discipline(str(route.get("type") or "route"))
            section = route.get("section_m") or (0.05, 0.05)
            radius = max(0.005, max(float(section[0]), float(section[1])) / 2)
            representation = self._swept_disk_representation(points, radius, discipline)
            placement = self._local_placement(self.storey_placements[level_id], (0.0, 0.0, 0.0))
            if discipline == "mechanical":
                ifc_entity = "IFCDUCTSEGMENT"
            elif discipline in {"plumbing", "fire"}:
                ifc_entity = "IFCPIPESEGMENT"
            else:
                ifc_entity = "IFCCABLECARRIERSEGMENT"
            product = self.step.add(
                ifc_entity,
                _string(_ifc_guid(f"route/{route_id}")),
                self.step.ref(self.owner_history),
                _string(route_id),
                _string("Evidence-grounded routed system segment"),
                _string(route.get("type") or "route"),
                self.step.ref(placement),
                self.step.ref(representation),
                _string(route_id),
                ".NOTDEFINED.",
            )
            self.contained_products[level_id].append(product)
            system_id = str(route.get("system_id") or f"{level_id}:{discipline}")
            systems[system_id].append(product)
            self._entity_properties(
                product,
                route_id,
                route,
                extra={
                    "SystemId": system_id,
                    "Discipline": discipline,
                    "SectionM": ",".join(_real(float(value)) for value in section),
                    "Material": route.get("material", ""),
                },
            )
        for system_id, products in systems.items():
            system = self.step.add(
                "IFCSYSTEM",
                _string(_ifc_guid(f"system/{system_id}")),
                self.step.ref(self.owner_history),
                _string(system_id),
                _string("Evidence-grounded distribution system"),
                _string(system_id),
            )
            self.step.add(
                "IFCRELASSIGNSTOGROUP",
                _string(_ifc_guid(f"system-members/{system_id}")),
                self.step.ref(self.owner_history),
                "$",
                "$",
                _tuple(self.step.ref(product) for product in products),
                "$",
                self.step.ref(system),
            )

    def _vertical_connections(self) -> None:
        for index, connection in enumerate(self.graph.get("vertical_connections") or []):
            connection_id = str(connection.get("id") or f"vertical-{index}")
            from_level = str(connection["from_level_id"])
            to_level = str(connection["to_level_id"])
            if from_level not in self.storey_placements or to_level not in self.storey_placements:
                raise ValueError(f"Vertical connection {connection_id!r} refers to unknown storey")
            rise = self.level_elevations[to_level] - self.level_elevations[from_level]
            if rise <= 0:
                raise ValueError(
                    f"Vertical connection {connection_id!r} must rise to a level above"
                )
            center = _point2(connection.get("center_m") or (0.0, 0.0))
            footprint_value = connection.get("footprint_m") or (1.2, 3.0)
            footprint = float(footprint_value[0]), float(footprint_value[1])
            yaw = math.radians(float(connection.get("yaw_deg") or 0.0))
            direction = math.cos(yaw), math.sin(yaw), 0.0
            kind = str(connection.get("type") or connection.get("kind") or "stair")
            placement = self._local_placement(
                self.storey_placements[from_level],
                (center[0], center[1], 0.0),
                direction,
            )
            representation = (
                self._stair_representation(footprint, rise)
                if kind in {"stair", "ramp", "escalator"}
                else self._rectangle_representation(
                    (footprint[0], footprint[1], rise), "structural"
                )
            )
            common = (
                _string(_ifc_guid(f"vertical/{connection_id}")),
                self.step.ref(self.owner_history),
                _string(connection_id),
                _string(f"Evidence-grounded {kind} from {from_level} to {to_level}"),
                _string(kind),
                self.step.ref(placement),
                self.step.ref(representation),
                _string(connection_id),
            )
            if kind == "stair":
                product = self.step.add("IFCSTAIR", *common, ".STRAIGHT_RUN_STAIR.")
            elif kind == "ramp":
                product = self.step.add("IFCRAMP", *common, ".STRAIGHT_RUN_RAMP.")
            elif kind in {"elevator", "escalator"}:
                product = self.step.add(
                    "IFCTRANSPORTELEMENT",
                    *common,
                    ".ELEVATOR." if kind == "elevator" else ".ESCALATOR.",
                    "$",
                    "$",
                )
            else:
                product = self.step.add("IFCBUILDINGELEMENTPROXY", *common, ".USERDEFINED.")
            self.contained_products[from_level].append(product)
            self._entity_properties(
                product,
                connection_id,
                connection,
                extra={
                    "FromLevelId": from_level,
                    "ToLevelId": to_level,
                    "ShaftId": str(connection.get("shaft_id") or ""),
                    "RiseM": rise,
                    "FootprintM": f"{_real(footprint[0])},{_real(footprint[1])}",
                },
            )

    def _entity_properties(
        self,
        product: int,
        entity_id: str,
        entity: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        source_refs = ",".join(str(value) for value in entity.get("source_ref_ids") or [])
        properties = {
            "EntityId": entity_id,
            "SourceRefIds": source_refs,
            "Confidence": entity.get("confidence", ""),
            "ReviewState": "review_required" if self.draft else "accepted",
            "CertificateSha256": self.certificate.content_sha256,
            **(extra or {}),
        }
        self._properties(product, "Pset_DajoongEvidence", properties)

    def _properties(self, product: int, name: str, values: dict[str, Any]) -> None:
        properties = []
        for key, value in values.items():
            if value is None or value == "":
                nominal = "$"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                nominal = f"IFCREAL({_real(float(value))})"
            else:
                nominal = f"IFCTEXT({_string(value)})"
            properties.append(
                self.step.add("IFCPROPERTYSINGLEVALUE", _string(key), "$", nominal, "$")
            )
        property_set = self.step.add(
            "IFCPROPERTYSET",
            _string(_ifc_guid(f"property-set/{product}/{name}")),
            self.step.ref(self.owner_history),
            _string(name),
            "$",
            _tuple(self.step.ref(prop) for prop in properties),
        )
        self.step.add(
            "IFCRELDEFINESBYPROPERTIES",
            _string(_ifc_guid(f"property-relation/{product}/{name}")),
            self.step.ref(self.owner_history),
            "$",
            "$",
            _tuple([self.step.ref(product)]),
            self.step.ref(property_set),
        )


def export_ifc(
    graph: dict[str, Any],
    output_path: Path,
    *,
    allow_draft: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    certificate = PlanGraphVerifier().verify(graph)
    if not certificate.release_allowed and not allow_draft:
        codes = sorted(
            {
                violation.code
                for violation in certificate.violations
                if violation.severity == "error"
            }
        )
        if not codes:
            codes = ["REVIEW_REQUIRED"]
        raise ValueError("IFC export blocked by fail-closed verifier: " + ", ".join(codes))
    builder = IfcModelBuilder(
        graph,
        certificate,
        draft=not certificate.release_allowed,
    )
    writer = builder.build()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_suffix(output_path.suffix + ".tmp")
    staging.write_text(writer.render(filename=output_path.name), encoding="utf-8", newline="\n")
    staging.replace(output_path)
    certificate_path = output_path.with_suffix(output_path.suffix + ".certificate.json")
    certificate_path.write_text(certificate.model_dump_json(indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "certificate": str(certificate_path),
        "bytes": output_path.stat().st_size,
        "entities": len(writer.entities),
        "releaseAllowed": certificate.release_allowed,
        "reviewRequired": certificate.review_required,
        "exportSeconds": round(time.perf_counter() - started, 6),
        "certificateSha256": certificate.content_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help=(
            "Export an explicitly marked NOT FOR CONSTRUCTION draft when verification "
            "blocks release."
        ),
    )
    args = parser.parse_args()
    graph = json.loads(args.plan_graph.read_text(encoding="utf-8"))
    print(json.dumps(export_ifc(graph, args.output, allow_draft=args.allow_draft), indent=2))


if __name__ == "__main__":
    main()
