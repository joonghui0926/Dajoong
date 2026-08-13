from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from buili_plan2bim.core.hashing import sha256_file
from buili_plan2bim.core.model.aec_decode import (
    AecTileProposal,
    PixelLineProposal,
    PixelRoomProposal,
    PixelSymbolProposal,
)
from buili_plan2bim.core.model.cad_evidence import (
    ORIENTED_EVIDENCE_ROTATION_CONTRACT,
)
from buili_plan2bim.core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
)
from buili_plan2bim.global_program_decode import (
    GlobalProgramDecodeDiagnostics,
    GlobalProgramDecodeResult,
)
from buili_plan2bim.global_program_inference import (
    GlobalProgramMultiviewDiagnostics,
    _clip_decode_to_plan_region,
    _filter_detail_walls_by_whole_building_context,
    _fuse_proposals,
    _translate_decode,
    letterbox_evidence,
    validate_global_program_manifest,
)
from buili_plan2bim.local_element_inference import (
    UnresolvedNativeElementCandidate,
    _element_has_required_host,
)
from buili_plan2bim.sheet_layout import SheetPlanRegion


def _manifest(artifact: Path) -> dict[str, object]:
    return {
        "schema_version": "dajoong.global-program-onnx.v1",
        "artifact_sha256": sha256_file(artifact),
        "input_contract": "cad_global_oriented_letterbox_v2",
        "topology_channels": list(TOPOLOGY_TARGET_CHANNELS),
        "room_classes": list(ROOM_PROGRAM_CLASSES),
        "element_classes": list(ELEMENT_PROGRAM_CLASSES),
        "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "production_authorized": False,
    }


def test_global_program_manifest_is_hash_and_contract_locked(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"immutable-model")
    payload = _manifest(artifact)

    validate_global_program_manifest(payload, artifact, require_production=False)
    with pytest.raises(PermissionError, match="not authorized"):
        validate_global_program_manifest(payload, artifact, require_production=True)

    payload["element_classes"] = ["background", "box"]
    with pytest.raises(ValueError, match="element_classes"):
        validate_global_program_manifest(payload, artifact, require_production=False)


def test_contextual_manifest_rejects_missing_crop_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"immutable-model")
    payload = _manifest(artifact)
    payload["input_names"] = ["full_sheet_evidence", "crop_context"]

    with pytest.raises(ValueError, match="crop context"):
        validate_global_program_manifest(payload, artifact, require_production=False)


def test_dual_view_manifest_requires_explicit_whole_sheet_contract(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"immutable-model")
    payload = _manifest(artifact)
    payload["input_names"] = [
        "view_evidence",
        "whole_sheet_evidence",
        "crop_context",
    ]
    payload["crop_context_contract"] = "normalized_origin_extent_sheet_edges_v1"

    with pytest.raises(ValueError, match="whole-sheet context"):
        validate_global_program_manifest(payload, artifact, require_production=False)

    payload["whole_sheet_context_contract"] = "explicit_complete_sheet_evidence_v1"
    validate_global_program_manifest(payload, artifact, require_production=False)


def test_axis_consistent_manifest_requires_rotation_contract(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"immutable-model")
    payload = _manifest(artifact)
    payload["model_version"] = (
        "dajoong-global-program-student-v8-axis-consistent-context"
    )

    with pytest.raises(ValueError, match="rotation contract"):
        validate_global_program_manifest(payload, artifact, require_production=False)

    payload["oriented_evidence_rotation_contract"] = (
        ORIENTED_EVIDENCE_ROTATION_CONTRACT
    )
    validate_global_program_manifest(payload, artifact, require_production=False)


def test_whole_sheet_letterbox_preserves_rectangular_aspect_ratio() -> None:
    evidence = np.ones((4, 100, 400), dtype=np.float32)

    tensor, content_bbox = letterbox_evidence(evidence, 256)

    assert tensor.shape == (1, 4, 256, 256)
    left, top, right, bottom = content_bbox
    assert (right - left, bottom - top) == (256, 64)
    assert np.all(tensor[0, :, top:bottom, left:right] == 1)
    assert np.all(tensor[0, :, :top] == 0)


def _decode(prefix: str, offset: float = 0.0) -> GlobalProgramDecodeResult:
    proposal = AecTileProposal(
        tile_id=prefix,
        source_ref_ids=["source"],
        model_version="test",
        wall_segments=[
            PixelLineProposal(
                id=f"{prefix}:wall",
                start_px=(10.0 + offset, 10.0),
                end_px=(90.0 + offset, 10.0),
                thickness_px=4.0,
                confidence=0.8,
                uncertainty=0.2,
                source_ref_ids=["source"],
                model_version="test",
                review_required=False,
            )
        ],
        symbols=[
            PixelSymbolProposal(
                id=f"{prefix}:bed",
                symbol_class="bed",
                center_px=(40.0 + offset, 40.0),
                bbox_px=(30.0 + offset, 30.0, 50.0 + offset, 50.0),
                confidence=0.9,
                uncertainty=0.1,
                source_ref_ids=["source"],
                model_version="test",
                review_required=False,
            )
        ],
        room_regions=[
            PixelRoomProposal(
                id=f"{prefix}:room",
                name="Bedroom",
                room_class="Bedroom",
                polygon_px=[
                    (10.0 + offset, 10.0),
                    (90.0 + offset, 10.0),
                    (90.0 + offset, 90.0),
                    (10.0 + offset, 90.0),
                ],
                confidence=0.9,
                uncertainty=0.1,
                source_ref_ids=["source"],
                model_version="test",
                review_required=False,
            )
        ],
        rejected_candidates=0,
    ).finalize()
    return GlobalProgramDecodeResult(
        proposal=proposal,
        diagnostics=GlobalProgramDecodeDiagnostics(
            source_size=(100, 100),
            model_size=(64, 64),
            room_instance_count=1,
            structural_wall_count=1,
            rejected_wall_count=0,
            element_count=1,
            native_wall_refinement_applied=True,
            release_blockers=["test"],
        ),
    )


def test_region_decode_maps_back_to_whole_sheet_coordinates() -> None:
    region = SheetPlanRegion(
        id="sheet:plan-02",
        bbox_px=(500, 100, 600, 200),
        crop_to_sheet_transform=((1, 0, 500), (0, 1, 100), (0, 0, 1)),
        sheet_area_fraction=0.2,
        structural_pixel_count=100,
        structural_density=0.1,
        confidence=0.9,
    )

    translated = _translate_decode(_decode("detail"), region)

    assert translated.proposal.wall_segments[0].start_px == (510.0, 110.0)
    assert translated.proposal.symbols[0].bbox_px == (530.0, 130.0, 550.0, 150.0)
    assert translated.proposal.room_regions[0].polygon_px[0] == (510.0, 110.0)


def test_selected_plan_region_rejects_neighboring_page_geometry() -> None:
    decoded = _decode("whole-sheet")
    outside_wall = decoded.proposal.wall_segments[0].model_copy(
        update={"id": "outside", "start_px": (10.0, 150.0), "end_px": (90.0, 150.0)}
    )
    proposal = decoded.proposal.model_copy(
        update={
            "wall_segments": [*decoded.proposal.wall_segments, outside_wall],
            "content_sha256": "",
        }
    ).finalize()
    decoded = decoded.model_copy(update={"proposal": proposal})
    region = SheetPlanRegion(
        id="sheet:plan-01",
        bbox_px=(0, 0, 100, 100),
        crop_to_sheet_transform=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        sheet_area_fraction=0.5,
        structural_pixel_count=100,
        structural_density=0.1,
        confidence=0.9,
    )

    clipped = _clip_decode_to_plan_region(decoded, region)

    assert [wall.id for wall in clipped.proposal.wall_segments] == ["whole-sheet:wall"]
    assert clipped.proposal.rejected_candidates == decoded.proposal.rejected_candidates + 1


def test_hierarchical_fusion_preserves_unmatched_whole_sheet_evidence() -> None:
    global_decode = _decode("global")
    detail_decode = _decode("detail", offset=200.0)

    fused = _fuse_proposals(global_decode, [detail_decode])

    assert len(fused.proposal.wall_segments) == 2
    assert len(fused.proposal.room_regions) == 2
    assert len(fused.proposal.symbols) == 2
    assert "hierarchical-sheet" in fused.proposal.model_version


def test_hierarchical_fusion_stitches_overlapping_source_wall_fragments() -> None:
    global_decode = _decode("global")
    detail_decode = _decode("detail")
    detail_wall = detail_decode.proposal.wall_segments[0].model_copy(
        update={"start_px": (48.0, 10.5), "end_px": (90.0, 10.5)}
    )
    detail_decode = detail_decode.model_copy(
        update={
            "proposal": detail_decode.proposal.model_copy(
                update={"wall_segments": [detail_wall], "content_sha256": ""}
            ).finalize()
        }
    )

    fused = _fuse_proposals(global_decode, [detail_decode])

    assert len(fused.proposal.wall_segments) == 1
    wall = fused.proposal.wall_segments[0]
    assert wall.start_px[0] <= 10.0
    assert wall.end_px[0] >= 90.0
    assert "source-stitch" in wall.model_version


def test_detail_wall_needs_whole_building_graph_support() -> None:
    context = _decode("context")
    isolated = _decode("isolated")
    isolated_wall = isolated.proposal.wall_segments[0].model_copy(
        update={"start_px": (25.0, 45.0), "end_px": (75.0, 45.0)}
    )
    isolated = isolated.model_copy(
        update={
            "proposal": isolated.proposal.model_copy(
                update={"wall_segments": [isolated_wall], "content_sha256": ""}
            ).finalize()
        }
    )

    accepted, rejected = _filter_detail_walls_by_whole_building_context(
        context,
        [isolated],
        source_size=(100, 100),
    )

    assert accepted == []
    assert rejected == [isolated_wall.id]


def test_detail_partition_anchored_at_both_ends_is_retained() -> None:
    context = _decode("context")
    second_wall = context.proposal.wall_segments[0].model_copy(
        update={"id": "context:bottom", "start_px": (10.0, 90.0), "end_px": (90.0, 90.0)}
    )
    context = context.model_copy(
        update={
            "proposal": context.proposal.model_copy(
                update={
                    "wall_segments": [*context.proposal.wall_segments, second_wall],
                    "content_sha256": "",
                }
            ).finalize()
        }
    )
    partition = _decode("partition")
    partition_wall = partition.proposal.wall_segments[0].model_copy(
        update={"start_px": (50.0, 10.0), "end_px": (50.0, 90.0)}
    )
    partition = partition.model_copy(
        update={
            "proposal": partition.proposal.model_copy(
                update={"wall_segments": [partition_wall], "content_sha256": ""}
            ).finalize()
        }
    )

    accepted, rejected = _filter_detail_walls_by_whole_building_context(
        context,
        [partition],
        source_size=(100, 100),
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].proposal.wall_segments == [partition_wall]


def test_multiview_contract_exposes_instance_selection_and_unresolved_ink() -> None:
    from buili_plan2bim.sheet_layout import SheetLayoutAnalysis

    diagnostics = GlobalProgramMultiviewDiagnostics(
        layout=SheetLayoutAnalysis(
            sheet_id="sheet",
            image_size_px=(1000, 600),
            regions=[],
            multi_plan_candidate=False,
            unassigned_structural_fraction=0.0,
        ).finalize(),
        fused_wall_count=3,
        fused_room_count=1,
        fused_element_count=2,
        unresolved_native_candidate_count=4,
        selected_plan_instance_id="sheet:plan-01",
    )
    unresolved = UnresolvedNativeElementCandidate(
        candidate_id="candidate-1",
        bbox_px=(10, 20, 30, 40),
        proposed_class="chair",
        confidence=0.42,
        reason="below_discovery_threshold",
    )

    assert diagnostics.unresolved_native_candidate_count == 4
    assert diagnostics.selected_plan_instance_id == "sheet:plan-01"
    assert unresolved.bbox_px == (10.0, 20.0, 30.0, 40.0)


def test_detail_pass_contract_carries_whole_plan_position() -> None:
    from buili_plan2bim.global_program_inference import GlobalProgramRegionPass

    detail = GlobalProgramRegionPass(
        region_id="plan:detail-3",
        bbox_px=(400, 0, 600, 200),
        scale_gain=2.0,
        wall_count=4,
        room_count=0,
        element_count=0,
        trusted_for_refinement=True,
        pass_kind="native_detail_window",
        context_x=0.5,
        context_y=0.0,
        context_width=0.25,
        context_height=0.4,
        touches_top=True,
    )

    assert detail.context_x == 0.5
    assert detail.context_width == 0.25
    assert detail.touches_top is True


def test_opening_requires_alignment_with_a_wall_host() -> None:
    opening = _decode("opening").proposal.symbols[0].model_copy(
        update={
            "symbol_class": "window",
            "center_px": (40.0, 10.0),
            "bbox_px": (30.0, 7.0, 50.0, 13.0),
        }
    )
    wall = _decode("wall").proposal.wall_segments[0]

    assert _element_has_required_host(opening, [wall]) is True
    assert _element_has_required_host(
        opening.model_copy(
            update={"center_px": (40.0, 50.0), "bbox_px": (30.0, 47.0, 50.0, 53.0)}
        ),
        [wall],
    ) is False
