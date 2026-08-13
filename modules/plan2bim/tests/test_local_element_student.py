from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from buili_plan2bim.core.hashing import sha256_file
from buili_plan2bim.core.model.aec_decode import PixelSymbolProposal
from buili_plan2bim.core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
)
from buili_plan2bim.core.model.local_element_student import (
    ELEMENT_CLASS_FAMILY_INDICES,
    ELEMENT_FAMILY_CLASSES,
    LEGACY_ELEMENT_FAMILY_CONTRACT,
    LEGACY_LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_FEATURES,
    LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
    LOCAL_ELEMENT_STRUCTURE_CONTEXT_FEATURES,
    LocalElementStudentConfig,
)
from buili_plan2bim.local_element_candidates import (
    _flat_close_rect,
    _flat_open_rect,
    _paired_outline_boxes,
    candidate_ledger_iou_recall,
    candidate_ledger_recall,
    mine_native_element_candidates,
)


from buili_plan2bim.local_element_crops import (
    CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
    LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS,
    LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LOCAL_ELEMENT_INPUT_CHANNELS,
    candidate_hypothesis_context,
    candidate_hypothesis_contexts,
    decode_element_geometry,
    element_geometry_target,
    extract_local_element_evidence,
    extract_local_element_hierarchy_batch_from_map,
    extract_local_element_hierarchy_evidence_from_map,
    extract_local_element_pyramid_batch_from_map,
    extract_local_element_pyramid_evidence_from_map,
    focus_candidate_detail_evidence,
    focus_candidate_detail_evidence_batch,
    normalized_candidate_context,
    semantic_element_context,
)
from buili_plan2bim.local_element_inference import (
    _native_geometry_choice,
    _preserve_structural_candidate,
    validate_local_element_manifest,
)
from buili_plan2bim.local_element_training import (
    _candidate_aligned_positive_bbox,
    _candidate_alignment_score,
    _fragment_negative_candidate_bboxes,
    _local_classification_metrics,
    build_synthetic_local_element_corpus,
)
from buili_plan2bim.synthetic_pretraining import generate_synthetic_pretraining_corpus


@pytest.mark.parametrize(
    "shape",
    [(1, 2), (2, 1), (2, 4), (3, 3), (4, 6), (7, 5), (12, 1), (1, 14)],
)
def test_separable_morphology_preserves_rectangular_scipy_contract(
    shape: tuple[int, int],
) -> None:
    """The speed path may not move, add, or remove proposal pixels."""

    scipy = pytest.importorskip("scipy.ndimage")
    generator = np.random.default_rng(20260812)
    mask = generator.random((41, 48)) > 0.68
    structure = np.ones(shape, dtype=np.uint8)

    assert np.array_equal(
        _flat_open_rect(mask, shape),
        scipy.binary_opening(mask, structure=structure),
    )
    assert np.array_equal(
        _flat_close_rect(mask, shape),
        scipy.binary_closing(mask, structure=structure),
    )


def test_local_element_contract_is_small_and_complete() -> None:
    config = LocalElementStudentConfig()

    config.validate()
    assert config.input_size == 64
    assert config.input_channels == LOCAL_ELEMENT_INPUT_CHANNELS
    assert config.classes == ELEMENT_PROGRAM_CLASSES
    assert config.geometry_channels == ELEMENT_GEOMETRY_CHANNELS
    assert config.candidate_context_features == LOCAL_ELEMENT_CONTEXT_FEATURES


def test_local_element_student_separates_structure_and_semantic_authority() -> None:
    torch = pytest.importorskip("torch")
    from buili_plan2bim.core.model.local_element_student import (
        DajoongLocalElementStudent,
    )

    model = DajoongLocalElementStudent().eval()
    batch = torch.zeros(2, 12, 64, 64)
    whole = torch.zeros(2, 4, 64, 64)
    context = torch.zeros(2, LOCAL_ELEMENT_CONTEXT_FEATURES)
    output = model(batch, whole, context)

    assert model.semantic_encoder is not model.encoder
    assert model.structure_whole_sheet_encoder is not model.whole_sheet_encoder
    assert output["class_logits"].shape == (2, len(ELEMENT_PROGRAM_CLASSES))
    assert output["objectness"].shape == (2, 1)
    assert output["geometry"].shape == (2, len(ELEMENT_GEOMETRY_CHANNELS))
    assert model.uncertainty_head.in_features == model.objectness_head.in_features * 2
    assert LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT.endswith("_v2")

    context_with_other_room = context.clone()
    context_with_other_room[:, 4:17] = 1.0
    structural_context = model._structure_context(context)
    other_structural_context = model._structure_context(context_with_other_room)
    assert structural_context.shape == (2, LOCAL_ELEMENT_STRUCTURE_CONTEXT_FEATURES)
    assert torch.equal(structural_context, other_structural_context)


def test_candidate_hypothesis_context_sees_parent_and_child_extents() -> None:
    boxes = [
        (10.0, 10.0, 30.0, 30.0),
        (8.0, 8.0, 40.0, 40.0),
        (14.0, 14.0, 20.0, 20.0),
    ]

    middle = candidate_hypothesis_context(boxes[0], boxes)

    assert middle.shape == (4,)
    assert middle[0] > 0
    assert middle[1] > 0
    vectorized = candidate_hypothesis_contexts(boxes, block_size=2)
    assert vectorized.shape == (3, 4)
    assert np.allclose(vectorized[0], middle)


def test_candidate_hypothesis_context_sees_aligned_equipment_runs() -> None:
    boxes = [
        (10.0, 10.0, 30.0, 30.0),
        (31.0, 10.0, 51.0, 30.0),
        (52.0, 10.0, 72.0, 30.0),
        (120.0, 80.0, 140.0, 100.0),
    ]

    vectorized = candidate_hypothesis_contexts(boxes)

    assert vectorized.shape == (4, 4)
    assert vectorized[1, 2] > 0.8
    assert vectorized[1, 3] == 0.0
    assert vectorized[3, 2] == 0.0
    assert candidate_hypothesis_context(boxes[1], boxes).tolist() == pytest.approx(
        vectorized[1].tolist()
    )
    assert CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT.endswith("equipment_runs_v2")


def test_spatial_candidate_graph_preserves_scalar_contract_on_large_sheet() -> None:
    # A large real sheet can produce thousands of native hypotheses.  The
    # vectorized entry point must preserve the exact scalar relation contract
    # without allocating a complete candidate-by-candidate tensor.
    boxes = [
        (
            float((index % 160) * 18),
            float((index // 160) * 18),
            float((index % 160) * 18 + 12),
            float((index // 160) * 18 + 12),
        )
        for index in range(20_000)
    ]
    # Add a parent, child, and two aligned neighbors around one audited target.
    target_index = len(boxes)
    boxes.extend(
        [
            (400.0, 400.0, 420.0, 420.0),
            (396.0, 396.0, 428.0, 428.0),
            (405.0, 405.0, 412.0, 412.0),
            (424.0, 400.0, 444.0, 420.0),
            (376.0, 400.0, 396.0, 420.0),
        ]
    )

    actual = candidate_hypothesis_contexts(boxes)[target_index]
    expected = candidate_hypothesis_context(boxes[target_index], boxes)

    assert actual.tolist() == pytest.approx(expected.tolist())


def test_native_outline_assembly_recovers_one_object_from_separate_strokes() -> None:
    foreground = np.zeros((500, 600), dtype=np.bool_)
    foreground[140:143, 220:300] = True
    foreground[197:200, 220:300] = True
    foreground[140:200, 220:223] = True
    foreground[140:200, 297:300] = True

    boxes = _paired_outline_boxes(foreground)

    target = (220.0, 140.0, 300.0, 200.0)
    assert max(
        candidate_ledger_iou_recall(
            [
                PixelSymbolProposal(
                    id=f"candidate-{index}",
                    symbol_class="unknown",
                    center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                    bbox_px=box,
                    confidence=0,
                    uncertainty=1,
                    source_ref_ids=["source"],
                    model_version="test",
                    review_required=True,
                )
                for index, box in enumerate(boxes)
            ],
            [target],
            minimum_iou=0.80,
        )["recall"],
        0,
    ) == 1.0


def test_candidate_context_matches_non_square_whole_plan_letterbox() -> None:
    # A 100 x 200 sheet occupies x=16..48 in a 64-square model input. The
    # source center is therefore model x=.5, while a full-width source box is
    # only half of the model width after aspect-preserving letterboxing.
    value = normalized_candidate_context(
        (25.0, 50.0, 75.0, 150.0),
        image_size=(100, 200),
        letterbox_size=64,
    )

    assert value.tolist() == pytest.approx([0.375, 0.25, 0.25, 0.5])


def test_semantic_element_context_carries_room_and_wall_relationships() -> None:
    value = semantic_element_context(
        (45.0, 45.0, 55.0, 55.0),
        image_size=(100, 100),
        rooms=[("kitchen", [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])],
        walls=[((0.0, 50.0), (100.0, 50.0), 4.0)],
    )

    assert value.shape == (16,)
    assert value[ROOM_PROGRAM_CLASSES.index("kitchen")] == 1.0
    assert value[-3] == 1.0
    assert value[-1] == 1.0


def test_local_element_contract_rejects_class_reordering() -> None:
    with pytest.raises(ValueError, match="classes"):
        LocalElementStudentConfig(classes=tuple(reversed(ELEMENT_PROGRAM_CLASSES))).validate()


def test_local_macro_f1_does_not_hide_a_missed_rare_class() -> None:
    count = len(ELEMENT_PROGRAM_CLASSES)
    confusion = np.zeros((count, count), dtype=np.int64)
    confusion[0, 0] = 100
    confusion[1, 1] = 20
    confusion[2, 1] = 5

    rows, macro_f1, micro_f1 = _local_classification_metrics(confusion)

    assert rows[1]["f1"] == pytest.approx(8 / 9)
    assert rows[2]["f1"] == 0.0
    assert macro_f1 == pytest.approx(4 / 9)
    assert micro_f1 == pytest.approx(0.8)


def test_local_element_crop_geometry_round_trips_source_coordinates() -> None:
    image = Image.new("RGB", (200, 120), "white")
    bbox = (72.0, 28.0, 112.0, 48.0)
    evidence, transform = extract_local_element_evidence(
        image,
        bbox,
        context_scale=2.2,
        center_jitter=(0.1, -0.05),
    )
    target = element_geometry_target(bbox, transform, yaw_deg=90.0)
    recovered, yaw = decode_element_geometry(target, transform)

    assert evidence.shape == (4, 64, 64)
    assert np.allclose(recovered, bbox, atol=1e-5)
    assert yaw == pytest.approx(90.0)


def test_local_element_pyramid_keeps_detail_geometry_and_adds_context() -> None:
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 220, 140), outline="black", width=5)
    draw.rectangle((92, 54, 126, 78), outline="black", width=2)
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence

    bbox = (92.0, 54.0, 126.0, 78.0)
    evidence, transform = extract_local_element_pyramid_evidence_from_map(
        build_cad_evidence(image),
        image.size,
        bbox,
        detail_scale=2.0,
        context_scale=6.0,
    )
    recovered, _ = decode_element_geometry(
        element_geometry_target(bbox, transform, yaw_deg=0.0),
        transform,
    )

    assert evidence.shape == (LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS, 64, 64)
    assert not np.array_equal(evidence[:4], evidence[4:])
    assert np.allclose(recovered, bbox, atol=1e-5)


def test_local_crop_work_is_bounded_for_a_huge_context_box() -> None:
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence

    image = Image.new("RGB", (128, 96), "white")
    evidence, transform = extract_local_element_pyramid_evidence_from_map(
        build_cad_evidence(image),
        image.size,
        (-20_000.0, -15_000.0, 20_000.0, 15_000.0),
        input_size=64,
        detail_scale=2.0,
        context_scale=5.5,
    )

    assert evidence.shape == (LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS, 64, 64)
    assert transform.side_px == 80_000.0


def test_batched_local_crops_match_scalar_runtime_crops() -> None:
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence

    image = Image.new("RGB", (220, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 20, 200, 130), outline="black", width=4)
    evidence_map = build_cad_evidence(image)
    boxes = [(24.0, 30.0, 54.0, 60.0), (130.0, 70.0, 188.0, 110.0)]
    batch, transforms = extract_local_element_pyramid_batch_from_map(
        evidence_map,
        image.size,
        boxes,
        input_size=64,
        detail_scale=2.1,
        context_scale=5.5,
    )

    assert batch.shape == (2, LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS, 64, 64)
    for index, box in enumerate(boxes):
        scalar, transform = extract_local_element_pyramid_evidence_from_map(
            evidence_map,
            image.size,
            box,
            input_size=64,
            detail_scale=2.1,
            context_scale=5.5,
        )
        assert np.allclose(batch[index], scalar, atol=1e-6)
        assert transforms[index] == transform


def test_hierarchical_local_crops_keep_detail_assembly_and_room_views() -> None:
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence

    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 345, 225), outline="black", width=5)
    draw.rectangle((130, 78, 164, 104), outline="black", width=2)
    bbox = (130.0, 78.0, 164.0, 104.0)
    evidence_map = build_cad_evidence(image)
    scalar, transform = extract_local_element_hierarchy_evidence_from_map(
        evidence_map,
        image.size,
        bbox,
    )
    batch, transforms = extract_local_element_hierarchy_batch_from_map(
        evidence_map,
        image.size,
        [bbox],
    )

    assert scalar.shape == (LOCAL_ELEMENT_INPUT_CHANNELS, 64, 64)
    assert np.allclose(batch[0], scalar, atol=1e-6)
    assert transforms == [transform]
    assert not np.array_equal(scalar[:4], scalar[4:8])
    assert not np.array_equal(scalar[4:8], scalar[8:12])


def test_focused_detail_marks_the_candidate_without_erasing_context() -> None:
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence

    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.line((12, 96, 244, 96), fill="black", width=5)
    draw.rectangle((104, 78, 136, 112), outline="black", width=2)
    bbox = (104.0, 78.0, 136.0, 112.0)
    evidence_map = build_cad_evidence(image)
    focused, _ = extract_local_element_hierarchy_evidence_from_map(
        evidence_map,
        image.size,
        bbox,
        focus_detail=True,
    )
    legacy, _ = extract_local_element_hierarchy_evidence_from_map(
        evidence_map,
        image.size,
        bbox,
        focus_detail=False,
    )

    assert focused[:4].sum() < legacy[:4].sum()
    assert np.allclose(focused[:, 28:36, 28:36], legacy[:, 28:36, 28:36])
    assert np.array_equal(focused[4:], legacy[4:])


def test_local_element_manifest_fails_closed_before_production(tmp_path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"not-an-onnx-but-hashable")
    manifest = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "artifact_sha256": sha256_file(artifact),
        "input_contract": LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "whole_sheet_input_channels": 4,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": 20,
        "candidate_context_contract": "normalized_bbox_room_and_wall_relations_v2",
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "output_names": [
            "class_logits",
            "objectness",
            "geometry",
            "uncertainty",
        ],
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "production_authorized": False,
    }

    validate_local_element_manifest(manifest, artifact, require_production=False)
    with pytest.raises(PermissionError, match="not authorized"):
        validate_local_element_manifest(manifest, artifact, require_production=True)


def test_axis_consistent_local_manifest_requires_rotation_contract(tmp_path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"hashable")
    manifest = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "artifact_sha256": sha256_file(artifact),
        "model_version": "dajoong-local-element-student-v10-coherent-proposals",
        "input_contract": LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "whole_sheet_input_channels": 4,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": 20,
        "candidate_context_contract": "normalized_bbox_room_and_wall_relations_v2",
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "output_names": [
            "class_logits",
            "objectness",
            "geometry",
            "uncertainty",
        ],
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "production_authorized": False,
    }

    with pytest.raises(ValueError, match="rotation contract mismatch"):
        validate_local_element_manifest(manifest, artifact, require_production=False)
    manifest["oriented_evidence_rotation_contract"] = (
        "c4_spatial_rotate_swap_axis_channels_on_odd_quadrants_v1"
    )
    with pytest.raises(ValueError, match="candidate alignment contract mismatch"):
        validate_local_element_manifest(manifest, artifact, require_production=False)
    manifest["candidate_alignment_contract"] = (
        "mutual_coverage_072_iou_055_or_truth_v1"
    )
    validate_local_element_manifest(manifest, artifact, require_production=False)


def test_relational_hierarchy_manifest_requires_graph_and_family_contracts(tmp_path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"hashable-v11")
    manifest = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "artifact_sha256": sha256_file(artifact),
        "model_version": "dajoong-local-element-student-v11-relational-hierarchy",
        "oriented_evidence_rotation_contract": (
            "c4_spatial_rotate_swap_axis_channels_on_odd_quadrants_v1"
        ),
        "candidate_alignment_contract": "mutual_coverage_072_iou_055_or_truth_v1",
        "candidate_hypothesis_context_contract": (
            "nested_proposal_graph_counts_and_extent_ratios_v1"
        ),
        "input_contract": LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "whole_sheet_input_channels": 4,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": LOCAL_ELEMENT_CONTEXT_FEATURES,
        "candidate_context_contract": LEGACY_LOCAL_ELEMENT_CONTEXT_CONTRACT,
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "output_names": [
            "class_logits",
            "family_logits",
            "objectness",
            "geometry",
            "uncertainty",
        ],
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "family_classes": list(ELEMENT_FAMILY_CLASSES),
        "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
        "family_contract": LEGACY_ELEMENT_FAMILY_CONTRACT,
        "production_authorized": False,
    }

    validate_local_element_manifest(manifest, artifact, require_production=False)
    manifest.pop("family_contract")
    with pytest.raises(ValueError, match="family output contract"):
        validate_local_element_manifest(manifest, artifact, require_production=False)


def test_v16_manifest_requires_independent_perception_authorities(tmp_path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"hashable-v16")
    manifest = {
        "schema_version": "dajoong.local-element-onnx.v1",
        "artifact_sha256": sha256_file(artifact),
        "model_version": LocalElementStudentConfig().model_version,
        "oriented_evidence_rotation_contract": (
            "c4_spatial_rotate_swap_axis_channels_on_odd_quadrants_v1"
        ),
        "candidate_alignment_contract": "mutual_coverage_072_iou_055_or_truth_v1",
        "candidate_hypothesis_context_contract": CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
        "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        "local_view_contract": "native_detail_assembly_room_v1",
        "input_names": [
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ],
        "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
        "whole_sheet_input_channels": 4,
        "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
        "candidate_context_features": LOCAL_ELEMENT_CONTEXT_FEATURES,
        "candidate_context_contract": (
            "letterbox_aligned_bbox_room_wall_and_equipment_run_relations_v5"
        ),
        "classes": list(ELEMENT_PROGRAM_CLASSES),
        "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "output_names": [
            "class_logits",
            "family_logits",
            "objectness",
            "geometry",
            "uncertainty",
        ],
        "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
        "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
        "family_classes": list(ELEMENT_FAMILY_CLASSES),
        "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
        "family_contract": "foreground_family_auxiliary_consistency_v2",
        "production_authorized": False,
    }

    with pytest.raises(ValueError, match="dual-authority"):
        validate_local_element_manifest(manifest, artifact, require_production=False)
    manifest["perception_authority_contract"] = LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
    validate_local_element_manifest(manifest, artifact, require_production=False)


def test_local_element_corpus_is_sheet_split_and_class_sealed(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_pretraining_corpus(source, count=3, seed=71)

    manifest = build_synthetic_local_element_corpus(
        source,
        tmp_path / "crops",
        input_size=64,
        negatives_per_sheet=1,
    )
    evidence = np.load(tmp_path / "crops" / "evidence.npy", mmap_mode="r")
    whole_evidence = np.load(
        tmp_path / "crops" / "whole-sheet-evidence.npy", mmap_mode="r"
    )
    candidate_context = np.load(
        tmp_path / "crops" / "candidate-context.npy", mmap_mode="r"
    )

    assert manifest["sample_count"] == 3
    assert manifest["input_contract"] == LOCAL_ELEMENT_EVIDENCE_CONTRACT
    assert manifest["item_count"] == evidence.shape[0]
    assert evidence.shape[1:] == (LOCAL_ELEMENT_INPUT_CHANNELS, 64, 64)
    assert whole_evidence.shape == (3, 4, 64, 64)
    assert candidate_context.shape == (
        manifest["item_count"],
        LOCAL_ELEMENT_CONTEXT_FEATURES,
    )
    assert np.all((candidate_context >= 0) & (candidate_context <= 1))
    assert sum(manifest["class_counts"].values()) == manifest["item_count"]
    assert manifest["class_counts"]["background"] == 3
    assert set(manifest["source_canvas_profiles"]) == {
        "square",
        "portrait",
        "landscape",
    }


def test_native_candidate_ledger_keeps_small_symbols_after_page_lines() -> None:
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 225, 165), outline="black", width=7)
    draw.line((15, 90, 225, 90), fill="black", width=7)
    draw.rectangle((96, 48, 122, 68), outline="black", width=2)
    draw.line((100, 52, 118, 64), fill="black", width=1)

    proposals, diagnostics = mine_native_element_candidates(
        image,
        source_ref_ids=["source"],
    )

    assert diagnostics.foreground_pixels > 0
    assert diagnostics.page_line_pixels > 0
    assert diagnostics.candidate_count == len(proposals)
    assert any(
        proposal.bbox_px[0] <= 109 <= proposal.bbox_px[2]
        and proposal.bbox_px[1] <= 58 <= proposal.bbox_px[3]
        for proposal in proposals
    )


def test_vectorized_detail_focus_matches_single_candidate_contract() -> None:
    rng = np.random.default_rng(107)
    evidence = rng.random((3, 4, 64, 64), dtype=np.float32)
    boxes = [
        (10.0, 12.0, 28.0, 34.0),
        (42.0, 18.0, 68.0, 40.0),
        (75.0, 61.0, 91.0, 87.0),
    ]
    from buili_plan2bim.local_element_crops import LocalElementCropTransform

    transforms = [
        LocalElementCropTransform((0.0, 0.0, 80.0, 80.0), 64),
        LocalElementCropTransform((24.0, 0.0, 88.0, 64.0), 64),
        LocalElementCropTransform((55.0, 45.0, 111.0, 101.0), 64),
    ]

    expected = np.stack(
        [
            focus_candidate_detail_evidence(value, box, transform)
            for value, box, transform in zip(
                evidence,
                boxes,
                transforms,
                strict=True,
            )
        ]
    )
    actual = focus_candidate_detail_evidence_batch(evidence, boxes, transforms)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-6)


def test_vectorized_crop_batch_matches_individual_source_sampling() -> None:
    image = Image.new("RGB", (173, 109), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((11, 9, 157, 96), outline="black", width=5)
    draw.ellipse((44, 31, 67, 58), outline="black", width=2)
    from buili_plan2bim.core.model.cad_evidence import build_cad_evidence
    from buili_plan2bim.local_element_crops import extract_local_element_evidence_from_map

    evidence_map = build_cad_evidence(image)
    boxes = [
        (39.0, 27.0, 72.0, 62.0),
        (98.0, 41.0, 126.0, 73.0),
    ]
    actual, transforms = extract_local_element_pyramid_batch_from_map(
        evidence_map,
        image.size,
        boxes,
        input_size=64,
        detail_scale=2.0,
        context_scale=5.5,
    )
    detail_expected = []
    context_expected = []
    for box in boxes:
        detail, _ = extract_local_element_evidence_from_map(
            evidence_map,
            image.size,
            box,
            input_size=64,
            context_scale=2.0,
        )
        context, _ = extract_local_element_evidence_from_map(
            evidence_map,
            image.size,
            box,
            input_size=64,
            context_scale=5.5,
        )
        detail_expected.append(detail)
        context_expected.append(context)
    expected = np.concatenate(
        (np.stack(detail_expected), np.stack(context_expected)),
        axis=1,
    )

    assert len(transforms) == len(boxes)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-6)


def test_strict_candidate_recall_does_not_count_a_large_context_crop_as_geometry() -> None:
    target = (40.0, 40.0, 60.0, 60.0)
    context = PixelSymbolProposal(
        id="context",
        symbol_class="unknown",
        center_px=(50.0, 50.0),
        bbox_px=(10.0, 10.0, 90.0, 90.0),
        confidence=0.0,
        uncertainty=1.0,
        source_ref_ids=["source"],
        model_version="test",
        review_required=True,
    )

    loose = candidate_ledger_recall([context], [target])
    strict = candidate_ledger_iou_recall([context], [target])

    assert loose["recall"] == 1.0
    assert strict["recall"] == 0.0
    assert strict["median_best_iou"] == pytest.approx(0.0625)


def test_native_geometry_keeps_source_envelope_when_regression_disagrees() -> None:
    proposal = (10.0, 10.0, 50.0, 50.0)
    incompatible = (24.0, 24.0, 44.0, 44.0)
    compatible = (11.0, 11.0, 49.0, 49.0)
    low_risk_but_weak = (5.0, 17.0, 55.0, 43.0)

    preserved, preserved_changed = _native_geometry_choice(
        proposal,
        incompatible,
        model_risk=0.35,
        native_candidate=True,
    )
    refined, refined_changed = _native_geometry_choice(
        proposal,
        compatible,
        model_risk=0.35,
        native_candidate=True,
    )
    global_refined, global_changed = _native_geometry_choice(
        proposal,
        incompatible,
        model_risk=0.35,
        native_candidate=False,
    )
    weak_refined, weak_changed = _native_geometry_choice(
        proposal,
        low_risk_but_weak,
        model_risk=0.05,
        native_candidate=True,
    )

    assert preserved == proposal
    assert preserved_changed is False
    assert refined == compatible
    assert refined_changed is True
    assert global_refined == incompatible
    assert global_changed is True
    assert weak_refined == proposal
    assert weak_changed is False


def test_strong_structure_survives_uncertain_fine_taxonomy() -> None:
    assert _preserve_structural_candidate(
        requires_confirmation=True,
        accepted_class=False,
        set_deferred=False,
        objectness=0.91,
        threshold=0.78,
    )
    assert not _preserve_structural_candidate(
        requires_confirmation=True,
        accepted_class=False,
        set_deferred=False,
        objectness=0.52,
        threshold=0.78,
    )
    assert not _preserve_structural_candidate(
        requires_confirmation=False,
        accepted_class=False,
        set_deferred=False,
        objectness=0.91,
        threshold=0.78,
    )


def test_native_geometry_allows_low_risk_source_preserving_object_completion() -> None:
    fragment = (20.0, 30.0, 40.0, 50.0)
    complete_object = (10.0, 20.0, 55.0, 60.0)

    refined, changed = _native_geometry_choice(
        fragment,
        complete_object,
        model_risk=0.02,
        native_candidate=True,
    )
    rejected, rejected_changed = _native_geometry_choice(
        fragment,
        complete_object,
        model_risk=0.12,
        native_candidate=True,
    )

    assert refined == complete_object
    assert changed is True
    assert rejected == fragment
    assert rejected_changed is False


def test_native_candidate_ledger_adds_a_whole_object_hypothesis_for_stroke_groups() -> None:
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    # Four disconnected strokes form one compact appliance-like symbol.
    for left, top in ((132, 82), (145, 82), (132, 95), (145, 95)):
        draw.rectangle((left, top, left + 10, top + 10), outline="black", width=2)

    proposals, _ = mine_native_element_candidates(image, source_ref_ids=["source"])

    assert any(
        proposal.bbox_px[0] <= 132
        and proposal.bbox_px[1] <= 82
        and proposal.bbox_px[2] >= 155
        and proposal.bbox_px[3] >= 105
        for proposal in proposals
    )


def test_native_candidate_ledger_keeps_a_bounded_linear_fixture() -> None:
    image = Image.new("RGB", (260, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 248, 208), outline="black", width=7)
    draw.line((92, 70, 92, 132), fill="black", width=2)

    proposals, _ = mine_native_element_candidates(image, source_ref_ids=["source"])

    assert any(
        proposal.bbox_px[0] <= 89
        and proposal.bbox_px[1] <= 70
        and proposal.bbox_px[2] >= 95
        and proposal.bbox_px[3] >= 132
        and proposal.bbox_px[2] - proposal.bbox_px[0] <= 20
        for proposal in proposals
    )


def test_local_corpus_can_add_candidate_aligned_hard_negatives(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_pretraining_corpus(source, count=2, seed=91)

    manifest = build_synthetic_local_element_corpus(
        source,
        tmp_path / "crops",
        negatives_per_sheet=1,
        hard_negatives_per_sheet=3,
    )

    assert manifest["hard_negatives_per_sheet"] == 3
    assert manifest["input_contract"] == LOCAL_ELEMENT_EVIDENCE_CONTRACT
    assert manifest["class_counts"]["background"] == 8


def test_fragment_candidates_are_explicit_background_siblings() -> None:
    target = (40.0, 40.0, 80.0, 80.0)
    whole = (38.0, 38.0, 82.0, 82.0)
    fragment = (42.0, 42.0, 56.0, 56.0)
    neighbor = (72.0, 44.0, 92.0, 64.0)

    selected = _fragment_negative_candidate_bboxes(
        target,
        [whole, fragment, neighbor],
        [(70.0, 40.0, 94.0, 68.0)],
        limit=3,
    )

    assert selected == [fragment]


def test_candidate_aligned_positive_rejects_fragments_and_assemblies() -> None:
    target = (40.0, 40.0, 80.0, 80.0)
    fragment = (40.0, 40.0, 80.0, 55.0)
    assembly = (20.0, 20.0, 100.0, 100.0)

    assert _candidate_aligned_positive_bbox(target, [fragment, assembly]) == target
    assert _candidate_alignment_score(target, fragment) < 0
    assert _candidate_alignment_score(target, assembly) < 0


def test_candidate_aligned_positive_accepts_the_same_complete_object() -> None:
    target = (40.0, 40.0, 80.0, 80.0)
    matched = (38.0, 39.0, 81.0, 82.0)

    assert _candidate_aligned_positive_bbox(target, [matched]) == matched


def test_local_corpus_can_train_object_fragments_as_background(tmp_path) -> None:
    source = tmp_path / "source"
    generate_synthetic_pretraining_corpus(source, count=1, seed=92)

    baseline = build_synthetic_local_element_corpus(
        source,
        tmp_path / "baseline-crops",
        negatives_per_sheet=1,
    )
    manifest = build_synthetic_local_element_corpus(
        source,
        tmp_path / "fragment-crops",
        negatives_per_sheet=1,
        fragment_negatives_per_object=2,
    )
    positive_count = sum(
        count
        for class_name, count in baseline["class_counts"].items()
        if class_name != "background"
    )

    assert manifest["fragment_negatives_per_object"] == 2
    assert manifest["class_counts"]["background"] == 1 + positive_count * 2


def test_candidate_ledger_recall_exposes_pre_classifier_misses() -> None:
    from buili_plan2bim.core.model.aec_decode import PixelSymbolProposal

    candidate = PixelSymbolProposal(
        id="candidate",
        symbol_class="unknown",
        center_px=(15.0, 15.0),
        bbox_px=(8.0, 8.0, 22.0, 22.0),
        confidence=0.0,
        uncertainty=1.0,
        source_ref_ids=["source"],
        model_version="test",
        review_required=True,
    )

    report = candidate_ledger_recall(
        [candidate],
        [(10.0, 10.0, 20.0, 20.0), (40.0, 40.0, 50.0, 50.0)],
    )

    assert report["recall"] == 0.5
    assert report["missed_target_indices"] == [1]
