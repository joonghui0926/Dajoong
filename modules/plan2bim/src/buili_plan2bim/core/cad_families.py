from __future__ import annotations

from typing import Literal, NamedTuple

from .hashing import sha256_json

Discipline = Literal["architectural", "electrical", "mechanical", "plumbing", "fire"]


class CadPart(NamedTuple):
    center: tuple[float, float, float]
    size: tuple[float, float, float]


FAMILY_MANIFESTS: dict[str, dict[str, object]] = {
    "heat-pump": {
        "version": "1.0.0",
        "discipline": "mechanical",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "housing+service-panel+dual-top-rails",
    },
    "supply-air-terminal": {
        "version": "1.0.0",
        "discipline": "mechanical",
        "ifc_class": "IfcAirTerminal",
        "recipe": "full-face-plenum+orthogonal-diffuser-blades",
    },
    "return-air-terminal": {
        "version": "1.0.0",
        "discipline": "mechanical",
        "ifc_class": "IfcAirTerminal",
        "recipe": "full-face-plenum+parallel-return-grille",
    },
    "thermostat": {
        "version": "1.0.0",
        "discipline": "mechanical",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "backplate+controller-face+display",
    },
    "electrical-panel": {
        "version": "1.0.0",
        "discipline": "electrical",
        "ifc_class": "IfcElectricDistributionBoard",
        "recipe": "cabinet+door+handle",
    },
    "light-fixture": {
        "version": "1.0.0",
        "discipline": "electrical",
        "ifc_class": "IfcLightFixture",
        "recipe": "housing+recessed-lens",
    },
    "receptacle": {
        "version": "1.0.0",
        "discipline": "electrical",
        "ifc_class": "IfcOutlet",
        "recipe": "plate+dual-slot-pairs",
    },
    "sprinkler": {
        "version": "1.0.0",
        "discipline": "fire",
        "ifc_class": "IfcFireSuppressionTerminal",
        "recipe": "escutcheon+stem+deflector-cross",
    },
    "residential-closet": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcFurniture",
        "recipe": "cabinet+double-door+toe-kick",
    },
    "residential-electrical-appliance": {
        "version": "1.0.0",
        "discipline": "electrical",
        "ifc_class": "IfcElectricAppliance",
        "recipe": "appliance-carcass+door+control-strip",
    },
    "residential-toilet": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "cistern+seat+bowl+base",
    },
    "residential-sink": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "counter+rimed-basin+pedestal+tap",
    },
    "residential-bench": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcFurniture",
        "recipe": "seat+four-legs",
    },
    "residential-fireplace": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "hearth+side-jambs+lintel+firebox",
    },
    "residential-bathtub": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "tub-shell+inner-well+rim",
    },
    "residential-chimney": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcChimney",
        "recipe": "shaft+cap",
    },
    "residential-base-cabinet": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcFurniture",
        "recipe": "toe-kick+carcass+counter+paired-doors",
    },
    "residential-wall-cabinet": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcFurniture",
        "recipe": "carcass+paired-doors+lower-trim",
    },
    "residential-shower-enclosure": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "tray+glass-panels+corner-posts+shower-head",
    },
    "structural-column": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcColumn",
        "recipe": "base+shaft+capital",
    },
    "residential-stair": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcStairFlight",
        "recipe": "seven-treads+riser-flight",
    },
    "generic-equipment-housing": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "plinth+housing+service-panel+top-cap",
    },
    "residential-coat-rack": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcFurniture",
        "recipe": "back-rail+five-hooks",
    },
    "residential-water-tap": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "base+stem+spout+handle",
    },
    "residential-jacuzzi": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "tub-bottom+four-rim-rails+jet-console",
    },
    "residential-wood-stove": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcEnergyConversionDevice",
        "recipe": "four-legs+firebox+door+flue",
    },
    "residential-corner-fireplace": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "corner-hearth+dual-jamb+lintel+firebox",
    },
    "fireplace-provision": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "reserved-hearth+corner-markers",
    },
    "corner-fireplace-provision": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "reserved-corner-hearth+dual-markers",
    },
    "generic-fixture": {
        "version": "1.0.0",
        "discipline": "architectural",
        "ifc_class": "IfcBuildingElementProxy",
        "recipe": "pedestal+cross-body+top-cap",
    },
    "generic-plumbing-fixture": {
        "version": "1.0.0",
        "discipline": "plumbing",
        "ifc_class": "IfcSanitaryTerminal",
        "recipe": "pedestal+basin+tap",
    },
    "generic-riser": {
        "version": "1.0.0",
        "discipline": "mechanical",
        "ifc_class": "IfcFlowSegment",
        "recipe": "vertical-shaft+top-cap",
    },
}


def approved_family_asset_sha256(family_id: str) -> str:
    manifest = FAMILY_MANIFESTS.get(family_id)
    if manifest is None:
        return ""
    return sha256_json({"family_id": family_id, **manifest})


def approved_family_discipline(family_id: str) -> Discipline | None:
    manifest = FAMILY_MANIFESTS.get(family_id)
    return manifest["discipline"] if manifest else None  # type: ignore[return-value]


def _part(
    x: float,
    y: float,
    z: float,
    width: float,
    depth: float,
    height: float,
) -> CadPart:
    return CadPart(
        center=(x, y, z),
        size=(max(width, 1e-5), max(depth, 1e-5), max(height, 1e-5)),
    )


def parametric_family_parts(family_id: str, size: tuple[float, float, float]) -> list[CadPart]:
    """Return deterministic local solids with a footprint-preserving base."""

    if family_id not in FAMILY_MANIFESTS:
        raise KeyError(f"unapproved CAD family: {family_id}")
    width, depth, height = (max(float(value), 1e-4) for value in size)
    base = min(height * 0.18, max(0.015, min(width, depth) * 0.12))
    parts = [_part(0, 0, 0, width, depth, base)]
    if family_id == "heat-pump":
        body_height = max(base, height * 0.72)
        parts.extend(
            [
                _part(0, 0, base, width * 0.94, depth * 0.92, body_height),
                _part(
                    0,
                    -depth * 0.47,
                    base + body_height * 0.25,
                    width * 0.76,
                    depth * 0.05,
                    body_height * 0.48,
                ),
                _part(
                    -width * 0.28,
                    0,
                    base + body_height,
                    width * 0.18,
                    depth * 0.75,
                    height * 0.08,
                ),
                _part(
                    width * 0.28,
                    0,
                    base + body_height,
                    width * 0.18,
                    depth * 0.75,
                    height * 0.08,
                ),
            ]
        )
    elif family_id in {"supply-air-terminal", "return-air-terminal"}:
        blade_height = max(0.008, height * 0.14)
        if family_id == "supply-air-terminal":
            parts.extend(
                [
                    _part(0, 0, base, width * 0.10, depth * 0.86, blade_height),
                    _part(0, 0, base, width * 0.86, depth * 0.10, blade_height),
                    _part(-width * 0.27, 0, base, width * 0.05, depth * 0.78, blade_height),
                    _part(width * 0.27, 0, base, width * 0.05, depth * 0.78, blade_height),
                ]
            )
        else:
            parts.extend(
                _part(0, depth * offset, base, width * 0.86, depth * 0.055, blade_height)
                for offset in (-0.30, -0.15, 0.0, 0.15, 0.30)
            )
    elif family_id == "thermostat":
        parts.extend(
            [
                _part(0, 0, base, width * 0.88, depth * 0.82, height * 0.58),
                _part(
                    0,
                    -depth * 0.43,
                    base + height * 0.18,
                    width * 0.48,
                    depth * 0.04,
                    height * 0.22,
                ),
            ]
        )
    elif family_id == "electrical-panel":
        parts.extend(
            [
                _part(0, 0, base, width * 0.94, depth * 0.90, height * 0.78),
                _part(
                    0,
                    -depth * 0.47,
                    base + height * 0.12,
                    width * 0.82,
                    depth * 0.04,
                    height * 0.58,
                ),
                _part(
                    width * 0.30,
                    -depth * 0.47,
                    base + height * 0.36,
                    width * 0.04,
                    depth * 0.03,
                    height * 0.16,
                ),
            ]
        )
    elif family_id == "light-fixture":
        parts.append(_part(0, 0, base, width * 0.82, depth * 0.82, max(0.012, height * 0.18)))
    elif family_id == "receptacle":
        slot_width = max(0.006, width * 0.08)
        slot_depth = max(0.006, depth * 0.16)
        for x_offset in (-0.22, 0.22):
            parts.extend(
                [
                    _part(
                        width * x_offset,
                        -depth * 0.30,
                        base,
                        slot_width,
                        slot_depth,
                        height * 0.20,
                    ),
                    _part(
                        width * x_offset,
                        depth * 0.05,
                        base,
                        slot_width,
                        slot_depth,
                        height * 0.20,
                    ),
                ]
            )
    elif family_id == "sprinkler":
        parts.extend(
            [
                _part(0, 0, base, width * 0.16, depth * 0.16, height * 0.62),
                _part(0, 0, base + height * 0.56, width * 0.72, depth * 0.08, height * 0.08),
                _part(0, 0, base + height * 0.56, width * 0.08, depth * 0.72, height * 0.08),
            ]
        )
    elif family_id == "residential-closet":
        parts.extend(
            [
                _part(0, 0, base, width, depth, height * 0.92),
                _part(
                    -width * 0.245,
                    -depth * 0.505,
                    base + height * 0.04,
                    width * 0.48,
                    depth * 0.025,
                    height * 0.82,
                ),
                _part(
                    width * 0.245,
                    -depth * 0.505,
                    base + height * 0.04,
                    width * 0.48,
                    depth * 0.025,
                    height * 0.82,
                ),
                _part(0, -depth * 0.18, 0, width * 0.9, depth * 0.55, height * 0.06),
            ]
        )
    elif family_id == "residential-electrical-appliance":
        parts.extend(
            [
                _part(0, 0, base, width * 0.96, depth * 0.96, height * 0.9),
                _part(
                    0,
                    -depth * 0.49,
                    base + height * 0.08,
                    width * 0.84,
                    depth * 0.025,
                    height * 0.66,
                ),
                _part(
                    0, -depth * 0.5, base + height * 0.78, width * 0.84, depth * 0.02, height * 0.09
                ),
            ]
        )
    elif family_id == "residential-toilet":
        parts.extend(
            [
                _part(0, depth * 0.3, base, width * 0.82, depth * 0.34, height * 0.68),
                _part(0, -depth * 0.08, base, width * 0.92, depth * 0.58, height * 0.38),
                _part(0, -depth * 0.05, base + height * 0.36, width, depth * 0.62, height * 0.08),
                _part(0, depth * 0.02, 0, width * 0.48, depth * 0.35, height * 0.2),
            ]
        )
    elif family_id == "residential-sink":
        parts.extend(
            [
                _part(0, 0, base + height * 0.72, width, depth, height * 0.1),
                _part(0, 0, base + height * 0.58, width * 0.72, depth * 0.62, height * 0.18),
                _part(0, depth * 0.2, base, width * 0.3, depth * 0.3, height * 0.62),
                _part(
                    0, depth * 0.36, base + height * 0.8, width * 0.08, depth * 0.08, height * 0.18
                ),
            ]
        )
    elif family_id == "residential-bench":
        parts.append(_part(0, 0, base + height * 0.72, width, depth, height * 0.14))
        for x_offset in (-0.42, 0.42):
            for y_offset in (-0.34, 0.34):
                parts.append(
                    _part(
                        width * x_offset,
                        depth * y_offset,
                        base,
                        width * 0.1,
                        depth * 0.1,
                        height * 0.72,
                    )
                )
    elif family_id == "residential-fireplace":
        jamb = width * 0.18
        parts.extend(
            [
                _part(0, -depth * 0.08, 0, width, depth, height * 0.12),
                _part(-width * 0.41, 0, base, jamb, depth * 0.9, height * 0.72),
                _part(width * 0.41, 0, base, jamb, depth * 0.9, height * 0.72),
                _part(0, 0, base + height * 0.72, width, depth * 0.9, height * 0.2),
            ]
        )
    elif family_id == "residential-bathtub":
        rim = max(width, depth) * 0.06
        parts.extend(
            [
                _part(0, -depth * 0.46, base, width, rim, height * 0.9),
                _part(0, depth * 0.46, base, width, rim, height * 0.9),
                _part(-width * 0.46, 0, base, rim, depth * 0.84, height * 0.9),
                _part(width * 0.46, 0, base, rim, depth * 0.84, height * 0.9),
                _part(0, 0, 0, width * 0.9, depth * 0.78, height * 0.12),
            ]
        )
    elif family_id == "residential-chimney":
        parts.extend(
            [
                _part(0, 0, base, width * 0.82, depth * 0.82, height * 0.9),
                _part(0, 0, base + height * 0.9, width, depth, height * 0.1),
            ]
        )
    elif family_id in {"residential-base-cabinet", "residential-wall-cabinet"}:
        carcass_height = height * 0.82
        parts.extend(
            [
                _part(0, 0, base, width * 0.96, depth * 0.94, carcass_height),
                _part(0, 0, base + carcass_height, width, depth, height * 0.08),
                _part(
                    -width * 0.245,
                    -depth * 0.485,
                    base + height * 0.43,
                    width * 0.47,
                    depth * 0.03,
                    height * 0.72,
                ),
                _part(
                    width * 0.245,
                    -depth * 0.485,
                    base + height * 0.43,
                    width * 0.47,
                    depth * 0.03,
                    height * 0.72,
                ),
            ]
        )
    elif family_id == "residential-shower-enclosure":
        panel = max(0.012, min(width, depth) * 0.025)
        post = max(panel * 2.5, 0.025)
        parts.extend(
            [
                _part(0, 0, base, width, depth, max(base, height * 0.035)),
                _part(-width * 0.49, 0, base, panel, depth, height * 0.92),
                _part(0, depth * 0.49, base, width, panel, height * 0.92),
                _part(-width * 0.49, depth * 0.49, base, post, post, height),
                _part(width * 0.49, depth * 0.49, base, post, post, height),
                _part(
                    width * 0.34,
                    depth * 0.42,
                    height * 0.78,
                    width * 0.14,
                    depth * 0.08,
                    height * 0.07,
                ),
            ]
        )
    elif family_id == "structural-column":
        parts.extend(
            [
                _part(0, 0, base, width * 0.78, depth * 0.78, height * 0.82),
                _part(0, 0, height * 0.86, width, depth, height * 0.08),
            ]
        )
    elif family_id == "residential-stair":
        step_count = 7
        tread_depth = depth / step_count
        for index in range(step_count):
            step_height = height * (index + 1) / step_count
            parts.append(
                _part(
                    0,
                    -depth / 2 + tread_depth * (index + 0.5),
                    base,
                    width,
                    tread_depth,
                    step_height,
                )
            )
    elif family_id == "generic-equipment-housing":
        parts.extend(
            [
                _part(0, 0, base, width * 0.94, depth * 0.92, height * 0.76),
                _part(
                    0,
                    -depth * 0.47,
                    base + height * 0.18,
                    width * 0.62,
                    depth * 0.05,
                    height * 0.42,
                ),
                _part(0, 0, base + height * 0.78, width, depth, height * 0.08),
            ]
        )
    elif family_id == "residential-coat-rack":
        parts.append(_part(0, 0, base + height * 0.56, width, depth * 0.28, height * 0.10))
        parts.extend(
            _part(
                width * offset,
                -depth * 0.28,
                base + height * 0.45,
                width * 0.035,
                depth * 0.48,
                height * 0.22,
            )
            for offset in (-0.4, -0.2, 0.0, 0.2, 0.4)
        )
    elif family_id == "residential-water-tap":
        parts.extend(
            [
                _part(0, 0, base, width * 0.22, depth * 0.22, height * 0.72),
                _part(
                    0,
                    -depth * 0.22,
                    base + height * 0.62,
                    width * 0.22,
                    depth * 0.52,
                    height * 0.10,
                ),
                _part(
                    -width * 0.22,
                    0,
                    base + height * 0.42,
                    width * 0.42,
                    depth * 0.10,
                    height * 0.08,
                ),
            ]
        )
    elif family_id == "residential-jacuzzi":
        rim_height = max(base, height * 0.20)
        parts.extend(
            [
                _part(0, 0, base, width * 0.82, depth * 0.78, height * 0.18),
                _part(0, -depth * 0.45, base + height * 0.22, width, depth * 0.10, rim_height),
                _part(0, depth * 0.45, base + height * 0.22, width, depth * 0.10, rim_height),
                _part(
                    -width * 0.45, 0, base + height * 0.22, width * 0.10, depth * 0.80, rim_height
                ),
                _part(
                    width * 0.45, 0, base + height * 0.22, width * 0.10, depth * 0.80, rim_height
                ),
                _part(
                    0, depth * 0.36, base + height * 0.42, width * 0.32, depth * 0.12, height * 0.08
                ),
            ]
        )
    elif family_id == "residential-wood-stove":
        parts.extend(
            [
                _part(
                    -width * 0.34, -depth * 0.34, base, width * 0.10, depth * 0.10, height * 0.20
                ),
                _part(width * 0.34, -depth * 0.34, base, width * 0.10, depth * 0.10, height * 0.20),
                _part(-width * 0.34, depth * 0.34, base, width * 0.10, depth * 0.10, height * 0.20),
                _part(width * 0.34, depth * 0.34, base, width * 0.10, depth * 0.10, height * 0.20),
                _part(0, 0, base + height * 0.18, width * 0.88, depth * 0.88, height * 0.55),
                _part(
                    0,
                    -depth * 0.46,
                    base + height * 0.28,
                    width * 0.58,
                    depth * 0.04,
                    height * 0.30,
                ),
                _part(0, 0, base + height * 0.73, width * 0.24, depth * 0.24, height * 0.27),
            ]
        )
    elif family_id == "residential-corner-fireplace":
        parts.extend(
            [
                _part(-width * 0.30, 0, base, width * 0.24, depth, height * 0.78),
                _part(0, -depth * 0.30, base, width, depth * 0.24, height * 0.78),
                _part(
                    -width * 0.12,
                    -depth * 0.12,
                    base + height * 0.70,
                    width * 0.70,
                    depth * 0.70,
                    height * 0.10,
                ),
                _part(
                    -width * 0.10, -depth * 0.10, base, width * 0.55, depth * 0.55, height * 0.08
                ),
            ]
        )
    elif family_id in {"fireplace-provision", "corner-fireplace-provision"}:
        parts.append(_part(0, 0, base, width, depth, max(base, height * 0.25)))
        if family_id == "fireplace-provision":
            parts.extend(
                [
                    _part(-width * 0.46, 0, base, width * 0.08, depth, height * 0.72),
                    _part(width * 0.46, 0, base, width * 0.08, depth, height * 0.72),
                ]
            )
        else:
            parts.extend(
                [
                    _part(-width * 0.46, 0, base, width * 0.08, depth, height * 0.72),
                    _part(0, -depth * 0.46, base, width, depth * 0.08, height * 0.72),
                ]
            )
    elif family_id == "generic-fixture":
        parts.extend(
            [
                _part(0, 0, base, width * 0.24, depth * 0.24, height * 0.72),
                _part(0, 0, base + height * 0.40, width * 0.82, depth * 0.16, height * 0.12),
                _part(0, 0, base + height * 0.40, width * 0.16, depth * 0.82, height * 0.12),
                _part(0, 0, base + height * 0.74, width * 0.52, depth * 0.52, height * 0.08),
            ]
        )
    elif family_id == "generic-plumbing-fixture":
        parts.extend(
            [
                _part(0, 0, base, width * 0.42, depth * 0.42, height * 0.58),
                _part(0, 0, height * 0.56, width, depth, height * 0.16),
                _part(
                    0,
                    depth * 0.2,
                    height * 0.74,
                    width * 0.08,
                    depth * 0.08,
                    height * 0.18,
                ),
            ]
        )
    elif family_id == "generic-riser":
        parts.extend(
            [
                _part(0, 0, base, width * 0.72, depth * 0.72, height * 0.94),
                _part(0, 0, height * 0.94, width, depth, height * 0.06),
            ]
        )
    return parts
