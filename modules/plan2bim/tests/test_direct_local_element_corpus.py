from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from buili_plan2bim.direct_local_element_corpus import (
    DIRECT_CANDIDATE_SUPERVISION_CONTRACT,
    _candidate_supervision_state,
    _evenly_spaced_indices,
    build_direct_local_element_corpus,
)
from buili_plan2bim.ground_truth import GroundTruthPolicyError


def _write_packet(root: Path, *, license_scope: str) -> None:
    sheet = root / "sheet-001"
    sheet.mkdir(parents=True)
    image_path = sheet / "source.png"
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 120, 120), outline="black", width=5)
    draw.rectangle((40, 48, 72, 76), outline="black", width=3)
    draw.ellipse((48, 55, 64, 69), outline="black", width=2)
    draw.text((82, 42), "A101", fill="black")
    image.save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "dajoong.manual-ground-truth.v2",
        "source": {
            "sheet_id": "sheet-001",
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "width_px": 128,
            "height_px": 128,
            "reviewed_plan_bbox_px": [0, 0, 128, 128],
            "license_scope": license_scope,
        },
        "visual_review": {
            "annotation_method": "direct_visual_source_annotation",
            "geometry_origin": "independent_source_pixel_manual_authoring",
            "annotation_session_id": "test-direct-session",
            "annotator": "test human",
            "reviewed_on": "2026-08-12",
            "whole_sheet_reviewed": True,
            "native_resolution_reviewed": True,
            "candidate_output_role": "review_aid_only_not_ground_truth",
            "review_passes": [
                "full_sheet",
                "walls",
                "openings",
                "rooms",
                "fixtures",
                "furniture",
                "typed_appliances",
            ],
        },
        "target_contract": {
            "content_profile": "full_editable_bim",
            "included": ["all_visible_editable_bim_elements"],
            "excluded": ["text", "dimensions", "hatches"],
        },
        "entities": [
            {
                "entity_id": "fixture-001",
                "entity_kind": "fixture",
                "directly_annotated": True,
                "evidence_kind": "native_source_pixels",
                "annotation_event_id": "fixture-event-001",
                "evidence_bbox_px": [40, 48, 72, 76],
                "geometry": {
                    "fixture_type": "sink",
                    "polygon": [[40, 48], [72, 48], [72, 76], [40, 76]],
                },
            }
        ],
        "omission_scan": {
            "completed": True,
            "coverage": "entire_reviewed_plan_bbox",
            "unresolved_findings": [],
            "visible_movable_furniture_count": 0,
            "visible_typed_appliance_count": 0,
            "visible_fixed_fixture_count": 1,
            "visible_cabinet_module_count": 0,
        },
    }
    (sheet / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_direct_research_corpus_keeps_truth_and_real_hard_negatives(tmp_path: Path) -> None:
    source = tmp_path / "gt"
    _write_packet(source, license_scope="research_eval_only")
    manifest = build_direct_local_element_corpus(
        source,
        tmp_path / "corpus",
        purpose="research_calibration",
        maximum_hard_negatives_per_sheet=32,
    )
    assert manifest["role"] == "direct_real_research_calibration_only"
    assert manifest["production_training_eligible"] is False
    assert manifest["class_counts"]["sink"] >= 2
    assert manifest["class_counts"]["background"] > 0
    assert (
        manifest["candidate_supervision_contract"]
        == DIRECT_CANDIDATE_SUPERVISION_CONTRACT
    )
    assert manifest["records"][0]["direct_fixture_count"] == 1
    assert manifest["records"][0]["hard_negative_count"] > 0
    assert (
        manifest["records"][0]["aligned_native_positive_count"]
        + manifest["records"][0]["ignored_ambiguous_candidate_count"]
        + manifest["records"][0]["explicit_background_candidate_count"]
        == manifest["records"][0]["native_candidate_count"]
    )
    assert (
        manifest["records"][0]["hard_negative_count"]
        <= manifest["records"][0]["explicit_background_candidate_count"]
    )
    assert (
        manifest["records"][0]["aligned_native_positive_count"]
        + manifest["records"][0]["hard_negative_count"]
        <= manifest["records"][0]["native_candidate_count"]
    )
    labels = np.load(tmp_path / "corpus" / "labels.npy")
    assert labels.shape == (manifest["item_count"],)
    assert manifest["evaluation_exclusion_source_sha256"]


def test_research_only_packet_cannot_enter_production_training(tmp_path: Path) -> None:
    source = tmp_path / "gt"
    _write_packet(source, license_scope="research_eval_only")
    try:
        build_direct_local_element_corpus(
            source,
            tmp_path / "corpus",
            purpose="production_training",
        )
    except GroundTruthPolicyError as error:
        assert "commercial_train" in str(error)
    else:  # pragma: no cover - explicit fail-close assertion.
        raise AssertionError("research-only drawing entered production training")


def test_direct_candidate_supervision_keeps_ambiguous_overlap_out_of_background() -> None:
    targets = [(20.0, 20.0, 60.0, 60.0)]
    assert _candidate_supervision_state((20.0, 20.0, 60.0, 60.0), targets) == (
        "ignore"
    )
    assert _candidate_supervision_state((25.0, 25.0, 40.0, 40.0), targets) == (
        "background"
    )
    assert _candidate_supervision_state((0.0, 0.0, 100.0, 100.0), targets) == (
        "background"
    )
    assert _candidate_supervision_state((10.0, 20.0, 50.0, 60.0), targets) == (
        "ignore"
    )
    assert _candidate_supervision_state((80.0, 80.0, 100.0, 100.0), targets) == (
        "background"
    )


def test_direct_negative_sampling_covers_the_full_candidate_ledger() -> None:
    selected = _evenly_spaced_indices(list(range(100)), 5)
    assert selected == [0, 24, 49, 74, 99]
