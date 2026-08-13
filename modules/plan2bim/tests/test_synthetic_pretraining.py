from __future__ import annotations

import json

from buili_plan2bim.synthetic_pretraining import (
    _GENERAL_FIXTURE_TYPES,
    assert_synthetic_pretraining_only,
    audit_synthetic_pretraining_corpus,
    generate_synthetic_pretraining_corpus,
    generate_synthetic_pretraining_sample,
)


def test_synthetic_pretraining_is_never_ground_truth(tmp_path) -> None:
    sample = generate_synthetic_pretraining_sample(
        tmp_path,
        sample_index=0,
        seed=42,
        size=256,
    )

    assert sample.role == "synthetic_pretrain_only"
    assert sample.real_drawing_ground_truth is False
    assert sample.evaluation_eligible is False
    assert sample.rooms
    assert sample.walls
    assert sample.openings
    assert sample.building_footprint_px
    assert any(opening.kind == "window" for opening in sample.openings)
    assert sample.image_path == "images/synthetic-0000000.png"
    assert sample.generator_version == (
        "dajoong-inverse-compiler-generator-v10-host-aware"
    )
    assert all(fixture.yaw_deg == 0.0 for fixture in sample.fixtures)


def test_synthetic_corpus_is_deterministic_and_auditable(tmp_path) -> None:
    first = generate_synthetic_pretraining_corpus(tmp_path / "first", count=3, seed=7)
    second = generate_synthetic_pretraining_corpus(tmp_path / "second", count=3, seed=7)

    assert first["sample_sha256"] == second["sample_sha256"]
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    assert manifest["sample_count"] == 3
    assert manifest["real_drawing_ground_truth"] is False
    assert manifest["evaluation_eligible"] is False
    audit = audit_synthetic_pretraining_corpus(
        tmp_path / "first",
        require_complete_taxonomy=False,
    )
    assert audit["passed"] is True
    assert audit["overlapping_sample_count"] == 0
    assert audit["invalid_bbox_count"] == 0


def test_synthetic_audit_rejects_contradictory_yaw(tmp_path) -> None:
    root = tmp_path / "corpus"
    generate_synthetic_pretraining_corpus(root, count=3, seed=17)
    annotation_path = sorted((root / "annotations").glob("*.json"))[0]
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["fixtures"][0]["yaw_deg"] = 17.0
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    audit = audit_synthetic_pretraining_corpus(
        root,
        require_complete_taxonomy=False,
    )

    assert audit["passed"] is False
    assert any("contradictory_source_yaw" in error for error in audit["errors"])


def test_synthetic_guard_fails_closed() -> None:
    unsafe = {
        "role": "training_ground_truth",
        "real_drawing_ground_truth": True,
        "evaluation_eligible": True,
    }

    try:
        assert_synthetic_pretraining_only(unsafe)
    except ValueError as error:
        assert "cannot enter ground-truth" in str(error)
    else:
        raise AssertionError("unsafe synthetic supervision was accepted")


def test_synthetic_program_exercises_broad_editable_element_vocabulary(tmp_path) -> None:
    fixture_types = set()
    for index in range(24):
        sample = generate_synthetic_pretraining_sample(
            tmp_path,
            sample_index=index,
            seed=1000 + index,
            size=256,
        )
        fixture_types.update(fixture.fixture_type for fixture in sample.fixtures)

    assert set(_GENERAL_FIXTURE_TYPES) <= fixture_types


def test_synthetic_program_covers_the_complete_room_contract(tmp_path) -> None:
    room_classes = set()
    for index in range(24):
        sample = generate_synthetic_pretraining_sample(
            tmp_path,
            sample_index=index,
            seed=2000 + index,
            size=256,
        )
        room_classes.update(room.room_class for room in sample.rooms)

    assert {
        "living",
        "bedroom",
        "kitchen",
        "bathroom",
        "storage",
        "office",
        "mechanical",
        "garage",
        "utility",
        "outdoor",
        "other",
        "corridor",
    } <= room_classes
