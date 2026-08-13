from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from buili_plan2bim.ground_truth import GroundTruthPolicyError
from buili_plan2bim.training_corpus import build_commercial_training_index


def _write_manifest(root: Path, name: str, group: str, *, scope: str = "commercial_train") -> Path:
    source = root / f"{name}.png"
    shade = 250 - (sum(name.encode("utf-8")) % 32)
    Image.new("L", (100, 100), shade).save(source)
    manifest = {
        "schema_version": "dajoong.manual-ground-truth.v2",
        "source": {
            "image_path": str(source),
            "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "width_px": 100,
            "height_px": 100,
            "reviewed_plan_bbox_px": [0, 0, 100, 100],
            "license_scope": scope,
            "collection_group_id": group,
            "license_provenance": {
                "permission_basis": "dajoong_owned",
                "commercial_training_allowed": True,
                "derivative_model_commercial_use_allowed": True,
                "rights_holder": "Dajoong",
                "evidence_ref": f"internal://rights/{group}",
                "verified_by": "dataset steward",
                "verified_on": "2026-08-10",
            },
        },
        "visual_review": {
            "annotation_method": "direct_visual_source_annotation",
            "geometry_origin": "independent_source_pixel_manual_authoring",
            "annotation_session_id": f"session-{name}",
            "whole_sheet_reviewed": True,
            "candidate_output_role": "review_aid_only_not_ground_truth",
            "annotator": "direct visual reviewer",
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
                "entity_id": f"wall-{name}",
                "entity_kind": "wall",
                "directly_annotated": True,
                "evidence_kind": "native_source_pixels",
                "annotation_event_id": f"event-wall-{name}",
                "evidence_bbox_px": [1, 1, 99, 10],
                "geometry": {"from": [1, 5], "to": [99, 5]},
            }
        ],
        "omission_scan": {
            "completed": True,
            "coverage": "entire_reviewed_plan_bbox",
            "unresolved_findings": [],
        },
    }
    path = root / f"{name}.manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_index_keeps_a_collection_group_in_one_stable_split(tmp_path: Path) -> None:
    first = _write_manifest(tmp_path, "a", "building-01")
    second = _write_manifest(tmp_path, "b", "building-01")

    index = build_commercial_training_index([first, second])

    assert index["sample_count"] == 2
    assert len({record["split"] for record in index["records"]}) == 1
    assert index["content_sha256"]


def test_index_rejects_evaluation_only_source(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "a", "building-01", scope="research_eval_only")

    with pytest.raises(GroundTruthPolicyError, match="commercial_train"):
        build_commercial_training_index([manifest])


def test_index_rejects_duplicate_source_pixels(tmp_path: Path) -> None:
    first = _write_manifest(tmp_path, "a", "building-01")
    duplicate = tmp_path / "duplicate.manifest.json"
    duplicate.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(GroundTruthPolicyError, match="duplicate source"):
        build_commercial_training_index([first, duplicate])
