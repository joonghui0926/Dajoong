from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from buili_plan2bim.core.asset_catalog import (
    attach_family_assets,
    audit_fixture_geometry,
    family_asset_definitions,
    family_asset_library,
    resolved_fixture_geometry,
)
from buili_plan2bim.core.glb_export import export_editable_glb


def _fixture(
    fixture_id: str,
    family_id: str,
    size_m: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "id": fixture_id,
        "level_id": "L1",
        "room_id": "room-1",
        "type": family_id,
        "family_id": family_id,
        "discipline": "plumbing" if "toilet" in family_id else "electrical",
        "center_m": [2.0, 2.0],
        "size_m": list(size_m),
        "yaw_deg": 0.0,
        "base_elevation_m": 0.0,
        "confidence": 0.99,
        "source_ref_ids": ["source-1"],
        "review_state": "review_required",
    }


def _graph() -> dict[str, object]:
    return {
        "schema_version": "buili.plan-graph.v2",
        "project_id": "asset-test",
        "levels": [{"id": "L1", "elevation_m": 0.0}],
        "rooms": [],
        "walls": [],
        "openings": [],
        "fixtures": [
            _fixture("toilet-1", "residential-toilet", (0.72, 0.46, 0.78)),
            _fixture(
                "appliance-1",
                "residential-electrical-appliance",
                (0.62, 0.64, 0.88),
            ),
            _fixture("unknown-1", "unresolved-symbol", (0.5, 0.4, 0.6)),
        ],
        "routes": [],
        "vertical_connections": [],
        "pipeline": {"content_sha256": "a" * 64},
        "confidence": {"review_required": False},
    }


def _glb_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    return json.loads(payload[20 : 20 + json_length].rstrip(b" \x00"))


def test_family_resolution_is_offline_evidence_sized_and_fail_closed() -> None:
    graph = _graph()

    attach_family_assets(graph)

    toilet, appliance, unknown = graph["fixtures"]
    assert toilet["geometry_status"] == "licensed_api_asset"
    assert toilet["asset_provider"]
    assert toilet["asset_license"] in {"by", "cc0"}
    assert toilet["asset_candidate_count"] >= 1
    assert 0 < toilet["asset_selection_score"] <= 1
    assert "mesh_vertices" not in toilet
    toilet_geometry = resolved_fixture_geometry(graph, toilet)
    toilet_vertices = np.asarray(toilet_geometry["mesh_vertices"], dtype=np.float64)
    assert np.allclose(
        toilet_vertices.max(axis=0) - toilet_vertices.min(axis=0),
        toilet["size_m"],
        rtol=0,
        atol=1e-5,
    )

    # A generic appliance remains generic.  Geometry may aid coordination, but
    # no refrigerator/dishwasher asset identity is fabricated from weak evidence.
    assert appliance["geometry_status"] == "native_bim_parametric"
    assert "asset_uid" not in appliance
    assert appliance["native_variant"] in {"compact", "standard", "wide"}
    assert appliance["asset_selection_policy"] == "dajoong-context-shape-ranker-v1"
    assert appliance["asset_candidate_count"] == 3
    assert unknown["geometry_status"] == "semantic_marker"
    assert "mesh_vertices" not in unknown
    assert graph["pipeline"]["family_resolution"]["network_on_conversion_path"] is False
    assert graph["pipeline"]["family_resolution"]["shared_geometry_count"] == 2
    assert graph["confidence"]["review_required"] is True

    audit = audit_fixture_geometry(graph)
    assert audit["passed"] is True
    assert audit["resolved_fixture_count"] == 2
    assert audit["unresolved_fixture_ids"] == ["unknown-1"]


def test_glb_uses_embedded_family_mesh_and_vertex_colors(tmp_path: Path) -> None:
    graph = _graph()
    graph["fixtures"] = graph["fixtures"][:1]
    attach_family_assets(graph)

    path = tmp_path / "asset.glb"
    export_editable_glb(graph, path)
    document = _glb_json(path)
    node = next(
        item for item in document["nodes"] if item["name"].startswith("fixture::toilet-1")
    )
    primitive = document["meshes"][node["mesh"]]["primitives"][0]

    assert "COLOR_0" in primitive["attributes"]
    assert node["extras"]["buili"]["properties"]["geometry_status"] == (
        "licensed_api_asset"
    )


def test_server_asset_library_exceeds_one_hundred_without_network() -> None:
    records = family_asset_library()
    definitions = family_asset_definitions()

    assert len(records) >= 100
    assert len({item["asset_id"] for item in records}) == len(records)
    assert len(definitions) >= 100
    assert all(str(item["geometry_ref"]).startswith("mesh:") for item in records)
    assert all(item["definition"]["normalized_to_unit_envelope"] is True for item in records)
    assert {item["geometry_status"] for item in records} == {
        "licensed_api_asset",
        "native_bim_parametric",
    }


def test_context_ranker_chooses_room_and_shape_appropriate_variant_quickly() -> None:
    graph = _graph()
    graph["rooms"] = [
        {"id": "kids-room", "name": "Kids Bedroom", "occupancy": "Bedroom"}
    ]
    graph["fixtures"] = [
        {
            **_fixture("bed-1", "residential-bed", (1.9, 1.05, 1.49)),
            "room_id": "kids-room",
            "discipline": "architectural",
        }
    ]

    attach_family_assets(graph)

    bed = graph["fixtures"][0]
    assert "Bunk" in bed["asset_name"]
    assert bed["asset_selection_policy"] == "dajoong-context-shape-ranker-v1"
    assert bed["asset_selection_context"]["room_label"] == "Kids Bedroom Bedroom"
    assert bed["asset_selection_components"]["context"] == 1.0
    assert bed["asset_selection_margin"] > 0
    assert bed["asset_selection_review_required"] is False
    assert len(bed["asset_selection_alternates"]) == 3
    assert bed["asset_selection_elapsed_us"] < 5_000


def test_context_ranker_reads_current_wall_keys_and_nearby_families() -> None:
    graph = _graph()
    graph["rooms"] = [
        {"id": "dining-room", "name": "Dining Room", "occupancy": "Dining"}
    ]
    graph["walls"] = [
        {
            "id": "wall-1",
            "level_id": "L1",
            "from": [0.0, 0.0],
            "to": [5.0, 0.0],
            "height_m": 2.7,
            "thickness_m": 0.15,
        }
    ]
    graph["fixtures"] = [
        {
            **_fixture("table-1", "residential-dining-table", (1.8, 0.9, 0.75)),
            "room_id": "dining-room",
            "center_m": [2.0, 0.2],
            "discipline": "architectural",
        },
        {
            **_fixture("chair-1", "residential-chair", (0.45, 0.45, 0.9)),
            "room_id": "dining-room",
            "center_m": [2.0, 1.1],
            "discipline": "architectural",
        },
    ]

    attach_family_assets(graph)

    table = graph["fixtures"][0]
    context = table["asset_selection_context"]
    assert context["installation"] == "wall_adjacent"
    assert context["nearest_wall_m"] == 0.2
    assert context["nearest_wall_angle_deg"] == 0.0
    assert "residential-chair" in context["nearby_families"]
    assert table["asset_selection_elapsed_us"] < 5_000

    office_graph = _graph()
    office_graph["rooms"] = [
        {"id": "office", "name": "Private Office", "occupancy": "Office"}
    ]
    office_graph["fixtures"] = [
        {
            **_fixture("office-table", "residential-dining-table", (1.4, 0.7, 0.75)),
            "room_id": "office",
            "center_m": [2.0, 2.0],
            "discipline": "architectural",
        },
        {
            **_fixture("office-chair", "residential-chair", (0.45, 0.45, 0.9)),
            "room_id": "office",
            "center_m": [2.0, 2.8],
            "discipline": "architectural",
        },
    ]
    attach_family_assets(office_graph)

    office_table = office_graph["fixtures"][0]
    assert office_table["asset_name"] != table["asset_name"]
    assert "Adjustable" in office_table["asset_name"]
