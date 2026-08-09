from __future__ import annotations

import copy

import pytest

from buili_plan2bim_studio.corrections import (
    GraphCorrection,
    GraphCorrectionSet,
    apply_graph_corrections,
    graph_content_hash,
)


def _graph() -> dict[str, object]:
    return {
        "schema_version": "buili.plan-graph.v2",
        "levels": [{"id": "L1", "name": "Level 1"}],
        "walls": [
            {
                "id": "L1:wall:1",
                "level_id": "L1",
                "from": [0.0, 0.0],
                "to": [4.0, 0.0],
                "height_m": 3.0,
                "thickness_m": 0.12,
                "confidence": 0.7,
                "uncertainty": 0.3,
                "review_state": "review_required",
            }
        ],
        "rooms": [],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "sources": [],
        "unsupported_features": [],
        "pipeline": {"content_sha256": "old"},
    }


def test_update_is_audited_and_does_not_mutate_source() -> None:
    source = _graph()
    original = copy.deepcopy(source)
    corrections = GraphCorrectionSet(
        expected_graph_sha256=graph_content_hash(source),
        reviewer="Paul Cho",
        operations=[
            GraphCorrection(
                id="edit-1",
                action="update",
                collection="walls",
                entity_id="L1:wall:1",
                changes={"height_m": 2.8, "thickness_m": 0.15},
            )
        ],
    )
    corrected = apply_graph_corrections(source, corrections)
    assert source == original
    assert corrected["walls"][0]["height_m"] == 2.8
    assert corrected["walls"][0]["review_state"] == "accepted"
    assert corrected["correction_log"][0]["reviewer"] == "Paul Cho"


def test_stale_patch_is_rejected() -> None:
    with pytest.raises(ValueError, match="stale correction set"):
        apply_graph_corrections(
            _graph(),
            GraphCorrectionSet(expected_graph_sha256="f" * 64, operations=[]),
        )


def test_constraint_and_dimension_corrections_round_trip() -> None:
    source = _graph()
    corrections = GraphCorrectionSet(
        expected_graph_sha256=graph_content_hash(source),
        operations=[
            GraphCorrection(
                id="add-constraint",
                action="add",
                collection="constraints",
                entity_id="L1:constraint:1",
                changes={
                    "level_id": "L1",
                    "type": "coincident",
                    "references": [
                        {"collection": "walls", "entity_id": "L1:wall:1", "handle": "from"},
                        {"collection": "walls", "entity_id": "L1:wall:1", "handle": "from"},
                    ],
                },
            ),
            GraphCorrection(
                id="add-dimension",
                action="add",
                collection="dimensions",
                entity_id="L1:dimension:1",
                changes={
                    "level_id": "L1",
                    "type": "aligned",
                    "name": "Driving dimension",
                    "from": [0.0, 0.0],
                    "to": [4.0, 0.0],
                },
            ),
        ],
    )
    corrected = apply_graph_corrections(source, corrections)
    assert corrected["constraints"][0]["type"] == "coincident"
    assert corrected["dimensions"][0]["name"] == "Driving dimension"


def test_copied_fixture_preserves_semantics_and_provenance() -> None:
    source = _graph()
    corrections = GraphCorrectionSet(
        expected_graph_sha256=graph_content_hash(source),
        operations=[
            GraphCorrection(
                id="copy-fixture",
                action="add",
                collection="fixtures",
                entity_id="L1:fixture:copy:1",
                changes={
                    "level_id": "L1",
                    "type": "air-terminal",
                    "family_id": "supply-diffuser-600",
                    "discipline": "mechanical",
                    "center_m": [2.0, 1.5],
                    "base_elevation_m": 2.7,
                    "size_m": [0.6, 0.6, 0.15],
                    "yaw_deg": 0.0,
                    "material": "powder-coated-steel",
                    "asset_sha256": "a" * 64,
                    "geometry_status": "validated",
                    "required_count": 1,
                    "observed_count": 1,
                    "source_ref_ids": ["source-L1"],
                    "copied_from_entity_id": "L1:fixture:original",
                },
            )
        ],
    )
    corrected = apply_graph_corrections(source, corrections)
    fixture = corrected["fixtures"][0]
    assert fixture["copied_from_entity_id"] == "L1:fixture:original"
    assert fixture["asset_sha256"] == "a" * 64
    assert fixture["review_state"] == "accepted"
