from __future__ import annotations

import json

import numpy as np

from buili_plan2bim.core.model.cad_evidence import GLOBAL_PROGRAM_INPUT_CONTRACT
from buili_plan2bim.core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
    GlobalTopologyStudentConfig,
)
from buili_plan2bim.synthetic_pretraining import generate_synthetic_pretraining_sample
from buili_plan2bim.topology_supervision import (
    build_synthetic_topology_target,
    build_synthetic_topology_target_corpus,
)


def test_global_topology_contract_requires_whole_sheet_token() -> None:
    config = GlobalTopologyStudentConfig()

    config.validate()
    assert config.context_grids[0] == 1
    assert config.output_channels == TOPOLOGY_TARGET_CHANNELS
    assert config.room_classes == ROOM_PROGRAM_CLASSES
    assert config.element_classes == ELEMENT_PROGRAM_CLASSES
    assert config.element_geometry_channels == ELEMENT_GEOMETRY_CHANNELS
    assert len(ELEMENT_PROGRAM_CLASSES) >= 20
    assert {
        "housing",
        "coat_rack",
        "water_tap",
        "jacuzzi",
        "wood_stove",
        "fireplace_corner",
        "place_for_fireplace",
        "place_for_fireplace_corner",
        "misc",
        "bed",
        "sofa",
        "armchair",
        "chair",
        "dining_table",
        "coffee_table",
        "desk",
        "bench",
        "refrigerator",
        "stove",
        "dishwasher",
        "washing_machine",
        "tumble_dryer",
    } <= set(ELEMENT_PROGRAM_CLASSES)


def test_synthetic_topology_target_is_sealed_from_ground_truth(tmp_path) -> None:
    generate_synthetic_pretraining_sample(
        tmp_path / "corpus",
        sample_index=0,
        seed=42,
        size=256,
    )
    annotation = tmp_path / "corpus" / "annotations" / "synthetic-0000000.json"
    output = tmp_path / "targets" / "sample.npz"

    manifest = build_synthetic_topology_target(annotation, output, target_size=128)
    array = np.load(output)

    assert array["targets"].shape == (len(TOPOLOGY_TARGET_CHANNELS), 128, 128)
    assert array["channel_names"].tolist() == list(TOPOLOGY_TARGET_CHANNELS)
    assert array["room_semantics"].shape == (128, 128)
    assert array["room_classes"].tolist() == list(ROOM_PROGRAM_CLASSES)
    assert array["element_semantics"].shape == (128, 128)
    assert array["element_classes"].tolist() == list(ELEMENT_PROGRAM_CLASSES)
    assert array["element_geometry"].shape == (
        len(ELEMENT_GEOMETRY_CHANNELS),
        128,
        128,
    )
    assert array["element_geometry_valid"].sum() > 0
    assert manifest["real_drawing_ground_truth"] is False
    assert manifest["evaluation_eligible"] is False
    assert manifest["room_semantic_contract"] == "localized_label_seed_v1"
    assert manifest["input_contract"] == GLOBAL_PROGRAM_INPUT_CONTRACT
    assert not np.array_equal(array["targets"][0], array["targets"][1])
    written = json.loads(output.with_suffix(".npz.json").read_text(encoding="utf-8"))
    assert written["content_sha256"] == manifest["content_sha256"]


def test_portrait_source_and_all_targets_share_one_letterbox_frame(tmp_path) -> None:
    sample = generate_synthetic_pretraining_sample(
        tmp_path / "corpus",
        sample_index=0,
        seed=42,
        size=320,
        canvas_profile="portrait",
    )
    annotation = tmp_path / "corpus" / "annotations" / "synthetic-0000000.json"
    output = tmp_path / "targets" / "portrait.npz"

    manifest = build_synthetic_topology_target(annotation, output, target_size=160)
    arrays = np.load(output)
    left, top, right, bottom = manifest["content_bbox"]

    assert sample.image_size_px is not None
    assert sample.image_size_px[0] < sample.image_size_px[1]
    assert top == 0 and bottom == 160
    assert left > 0 and right < 160
    assert np.all(arrays["targets"][:, :, :left] == 0)
    assert np.all(arrays["targets"][:, :, right:] == 0)
    assert np.all(arrays["room_semantics"][:, :left] == 0)
    assert np.all(arrays["element_semantics"][:, right:] == 0)


def test_synthetic_target_corpus_preserves_pretrain_only_role(tmp_path) -> None:
    from buili_plan2bim.synthetic_pretraining import generate_synthetic_pretraining_corpus

    corpus_root = tmp_path / "corpus"
    generate_synthetic_pretraining_corpus(corpus_root, count=3, seed=11)

    manifest = build_synthetic_topology_target_corpus(
        corpus_root,
        tmp_path / "targets",
        target_size=96,
    )

    assert manifest["sample_count"] == 3
    assert manifest["role"] == "synthetic_pretrain_only"
    assert manifest["real_drawing_ground_truth"] is False
    assert manifest["evaluation_eligible"] is False
    assert sum(manifest["room_pixel_counts"].values()) == 3 * 96 * 96
    assert sum(manifest["element_pixel_counts"].values()) == 3 * 96 * 96
    assert all("room_pixel_counts" in record for record in manifest["records"])
    assert all("element_pixel_counts" in record for record in manifest["records"])
