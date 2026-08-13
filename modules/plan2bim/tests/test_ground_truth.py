from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from buili_plan2bim.ground_truth import (
    GroundTruthPolicyError,
    assert_benchmark_graph_geometry,
    assert_commercial_training_eligible,
    assert_evaluation_content_profile,
    assert_manifest_graph_correspondence,
    audit_benchmark_graph_geometry,
    compile_benchmark_graph_from_manifest,
    validate_ground_truth_manifest,
)


def _write_source(path):
    Image.new("L", (64, 64), 255).save(path)


def _manifest(image_path, *, license_scope: str = "commercial_train"):
    return {
        "schema_version": "dajoong.manual-ground-truth.v2",
        "source": {
            "image_path": str(image_path),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "width_px": 64,
            "height_px": 64,
            "reviewed_plan_bbox_px": [0, 0, 64, 64],
            "license_scope": license_scope,
            "collection_group_id": "owned-building-001",
            "license_provenance": {
                "permission_basis": "dajoong_owned",
                "commercial_training_allowed": True,
                "derivative_model_commercial_use_allowed": True,
                "rights_holder": "Dajoong",
                "evidence_ref": "internal://rights/owned-building-001",
                "verified_by": "Dajoong dataset steward",
                "verified_on": "2026-08-10",
            },
        },
        "visual_review": {
            "annotation_method": "direct_visual_source_annotation",
            "geometry_origin": "independent_source_pixel_manual_authoring",
            "annotation_session_id": "session-001",
            "whole_sheet_reviewed": True,
            "candidate_output_role": "review_aid_only_not_ground_truth",
            "annotator": "OpenAI Codex direct source-image review",
            "reviewed_on": "2026-08-10",
            "review_passes": ["full_sheet", "walls", "openings", "rooms", "fixtures"],
        },
        "target_contract": {
            "content_profile": "structural_core",
            "included": ["walls", "openings", "rooms", "installed_equipment"],
            "excluded": ["movable_furniture"],
        },
        "entities": [
            {
                "entity_id": "door-001",
                "entity_kind": "opening",
                "directly_annotated": True,
                "evidence_kind": "native_source_pixels",
                "annotation_event_id": "event-door-001",
                "evidence_bbox_px": [8, 8, 24, 24],
                "geometry": {
                    "type": "door",
                    "polygon": [[8, 14], [24, 14], [24, 18], [8, 18]],
                },
            }
        ],
        "omission_scan": {
            "completed": True,
            "coverage": "entire_reviewed_plan_bbox",
            "unresolved_findings": [],
        },
    }


def test_accepts_direct_whole_sheet_annotation(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)

    validate_ground_truth_manifest(manifest)
    assert_commercial_training_eligible(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annotation_method", "model_generated_pseudo_labels"),
        ("whole_sheet_reviewed", False),
        ("candidate_output_role", "ground_truth"),
    ],
)
def test_rejects_automatic_or_unreviewed_labels(tmp_path, field, value):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    manifest["visual_review"][field] = value

    with pytest.raises(GroundTruthPolicyError):
        validate_ground_truth_manifest(manifest)


def test_rejects_manifest_that_only_claims_direct_review_but_has_no_source_pixel_authorship(
    tmp_path,
):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    del manifest["visual_review"]["geometry_origin"]

    with pytest.raises(GroundTruthPolicyError, match="independently authored"):
        validate_ground_truth_manifest(manifest)


def test_rejects_entity_without_native_pixel_annotation_event(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    del manifest["entities"][0]["annotation_event_id"]

    with pytest.raises(GroundTruthPolicyError, match="annotation_event_id"):
        validate_ground_truth_manifest(manifest)


def test_metric_graph_is_compiled_from_source_pixel_annotation_events(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)

    graph = compile_benchmark_graph_from_manifest(manifest)

    assert graph["source_image_sha256"] == manifest["source"]["image_sha256"]
    assert graph["annotation_session_id"] == "session-001"
    assert graph["openings"] == [
        {
            "id": "door-001",
            "annotation_event_id": "event-door-001",
            "evidence_bbox_px": [8, 8, 24, 24],
            "type": "door",
            "polygon": [[8, 14], [24, 14], [24, 18], [8, 18]],
        }
    ]
    assert_manifest_graph_correspondence(manifest, graph)


def test_rejects_separately_changed_metric_graph(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    graph = compile_benchmark_graph_from_manifest(manifest)
    graph["openings"][0]["polygon"] = [[30, 30], [40, 30], [40, 34], [30, 34]]

    with pytest.raises(GroundTruthPolicyError, match="compiled exactly"):
        assert_manifest_graph_correspondence(manifest, graph)


def test_rejects_changed_source_after_review(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    source.write_bytes(b"changed after annotation")

    with pytest.raises(GroundTruthPolicyError, match="hash"):
        validate_ground_truth_manifest(manifest)


def test_research_only_annotations_cannot_train_commercial_model(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source, license_scope="research_eval_only")

    validate_ground_truth_manifest(manifest)
    with pytest.raises(GroundTruthPolicyError, match="commercial_train"):
        assert_commercial_training_eligible(manifest)


def test_product_ground_truth_requires_furniture_and_typed_appliance_passes(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    manifest["target_contract"] = {
        "content_profile": "full_editable_bim",
        "included": ["all_visible_editable_bim_elements"],
        "excluded": ["text", "hatches"],
    }

    with pytest.raises(GroundTruthPolicyError, match="furniture"):
        validate_ground_truth_manifest(manifest)

    manifest["visual_review"]["review_passes"].extend(
        ["furniture", "typed_appliances"]
    )
    manifest["omission_scan"].update(
        {
            "visible_movable_furniture_count": 0,
            "visible_typed_appliance_count": 0,
            "visible_fixed_fixture_count": 0,
            "visible_cabinet_module_count": 0,
        }
    )
    validate_ground_truth_manifest(manifest)


def test_product_ground_truth_rejects_incomplete_omission_inventory(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    manifest["target_contract"] = {
        "content_profile": "full_editable_bim",
        "included": ["all_visible_editable_bim_elements"],
        "excluded": ["text", "hatches"],
    }
    manifest["visual_review"]["review_passes"].extend(
        ["furniture", "typed_appliances"]
    )

    with pytest.raises(GroundTruthPolicyError, match="omission scan requires"):
        validate_ground_truth_manifest(manifest)

    manifest["omission_scan"].update(
        {
            "visible_movable_furniture_count": 1,
            "visible_typed_appliance_count": 0,
            "visible_fixed_fixture_count": 0,
            "visible_cabinet_module_count": 0,
        }
    )
    with pytest.raises(GroundTruthPolicyError, match="fixture count"):
        validate_ground_truth_manifest(manifest)


def test_product_ground_truth_cannot_hide_furniture_from_denominator(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    manifest["target_contract"] = {
        "content_profile": "full_editable_bim",
        "included": ["all_visible_editable_bim_elements"],
        "excluded": ["movable_furniture"],
    }
    manifest["visual_review"]["review_passes"].extend(
        ["furniture", "typed_appliances"]
    )

    with pytest.raises(GroundTruthPolicyError, match="cannot exclude"):
        validate_ground_truth_manifest(manifest)


def test_structural_benchmark_cannot_support_full_product_f1_claim(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)

    validate_ground_truth_manifest(manifest)
    with pytest.raises(GroundTruthPolicyError, match="full_editable_bim"):
        assert_evaluation_content_profile(
            manifest,
            required_profile="full_editable_bim",
        )


def test_commercial_scope_without_rights_evidence_cannot_train(tmp_path):
    source = tmp_path / "source.png"
    _write_source(source)
    manifest = _manifest(source)
    del manifest["source"]["license_provenance"]

    validate_ground_truth_manifest(manifest)
    with pytest.raises(GroundTruthPolicyError, match="license_provenance"):
        assert_commercial_training_eligible(manifest)


def test_benchmark_geometry_rejects_out_of_source_and_zero_area_targets():
    graph = {
        "rooms": [],
        "walls": [],
        "openings": [],
        "fixtures": [
            {
                "id": "outside-column",
                "polygon": [[10, 110], [20, 110], [20, 120], [10, 120]],
            },
            {
                "id": "flat-misc",
                "polygon": [[10, 20], [30, 20], [30, 20], [10, 20]],
            },
        ],
        "routes": [],
    }

    issues = audit_benchmark_graph_geometry(graph, image_size=(100, 100))

    assert {issue["code"] for issue in issues} == {
        "entity_outside_source",
        "degenerate_polygon",
    }
    with pytest.raises(GroundTruthPolicyError, match="not aligned"):
        assert_benchmark_graph_geometry(graph, image_size=(100, 100))


def test_benchmark_geometry_accepts_source_aligned_targets():
    graph = {
        "rooms": [{"id": "room", "polygon": [[0, 0], [90, 0], [90, 90], [0, 90]]}],
        "walls": [{"id": "wall", "from": [0, 0], "to": [90, 0]}],
        "openings": [
            {"id": "door", "polygon": [[20, 0], [30, 0], [30, 3], [20, 3]]}
        ],
        "fixtures": [
            {"id": "sink", "polygon": [[20, 20], [30, 20], [30, 30], [20, 30]]}
        ],
        "routes": [],
    }

    assert audit_benchmark_graph_geometry(
        graph,
        image_size=(100, 100),
        reviewed_plan_bbox_px=[0, 0, 100, 100],
    ) == []
