from __future__ import annotations

from buili_plan2bim.core.asset_catalog import attach_family_assets
from buili_plan2bim.core.bim_program import BimProgramCompiler, ProgramEvidence
from buili_plan2bim.core.model.aec_decode import AecTileProposal, PixelLineProposal
from buili_plan2bim.core.proposal_program import (
    MetricLevelContext,
    build_program_from_tile_proposal,
)
from buili_plan2bim.perception_forest import (
    attach_compiled_graph_evidence,
    build_forest_perception_bundle,
    build_spatial_evidence_graph,
)
from buili_plan2bim.semantic_recognition import (
    SemanticDetection,
    SemanticRecognitionResult,
    SemanticRoom,
    SemanticWallVector,
)


def _recognition() -> SemanticRecognitionResult:
    return SemanticRecognitionResult(
        input_path="plan.png",
        input_sha256="a" * 64,
        model_version="semantic-test-v1",
        model_sha256="b" * 64,
        license_scope="test",
        production_authorized=False,
        source_size=(200, 120),
        model_input_size=(200, 120),
        wall_pixels=2_000,
        wall_vectors_px=[
            SemanticWallVector(
                start_px=(10, 40),
                end_px=(190, 40),
                thickness_px=8,
            )
        ],
        detections=[
            SemanticDetection(
                id="door-1",
                class_name="Door",
                symbol_class="door",
                bbox_px=(80, 34, 110, 46),
                confidence=0.9,
                pixel_area=360,
                review_required=False,
                promote_to_bim=True,
            ),
            SemanticDetection(
                id="sink-1",
                class_name="Sink",
                symbol_class="sink",
                bbox_px=(60, 60, 75, 75),
                confidence=0.88,
                pixel_area=225,
                review_required=False,
                promote_to_bim=True,
            ),
        ],
        rooms=[
            SemanticRoom(
                id="room-1",
                class_name="Kitchen",
                polygon_px=[(10, 45), (190, 45), (190, 110), (10, 110)],
                confidence=0.85,
                pixel_area=11_700,
                review_required=False,
            )
        ],
        counts={"Door": 1, "Sink": 1},
        inference_ms=1,
        total_ms=2,
    ).finalize()


def test_global_evidence_graph_links_elements_to_structure() -> None:
    graph = build_spatial_evidence_graph(_recognition(), source_ref_ids=["source-1"])

    relations = {(item.kind, item.source_id, item.target_id) for item in graph.relations}
    assert ("host_candidate", "door-1", "semantic:wall:0") in relations
    assert ("inside_room", "sink-1", "room-1") in relations
    assert graph.coverage.unhosted_opening_count == 0
    assert graph.coverage.unassigned_fixture_count == 0
    assert graph.topology_integrity.wall_component_count >= 1
    assert graph.topology_integrity.largest_wall_component_ratio > 0
    assert graph.topology_integrity.promoted_elements_requiring_relation == 2
    assert graph.topology_integrity.promoted_elements_with_required_relation == 2


def test_single_expert_graph_fails_closed_for_release() -> None:
    graph = build_spatial_evidence_graph(_recognition(), source_ref_ids=["source-1"])

    assert graph.release_ready is False
    assert graph.coverage.independent_expert_consensus_available is False
    assert "independent_expert_consensus_unavailable" in graph.release_blockers


def test_evidence_survives_compilation_and_server_asset_resolution() -> None:
    recognition = _recognition()
    base = AecTileProposal(
        tile_id="full-sheet",
        source_ref_ids=["source-1"],
        model_version="base-test",
        wall_segments=[],
        symbols=[],
        rejected_candidates=0,
    ).finalize()
    forest = build_forest_perception_bundle(
        base,
        recognition,
        source_ref_ids=["source-1"],
    )
    evidence = ProgramEvidence(
        id="source-1",
        uri="https://example.invalid/plan.png",
        sha256="a" * 64,
        page_number=1,
        source_kind="raster_image",
        extractor="test",
        model_version="test",
    )
    build = build_program_from_tile_proposal(
        forest.proposal,
        MetricLevelContext(
            project_id="test",
            level_id="L1",
            level_name="Level 1",
            pixels_per_meter=100,
            elevation_m=0,
            nominal_height_m=3,
            evidence=evidence,
        ),
    )
    plan_graph = BimProgramCompiler().compile(build.program)
    attach_family_assets(plan_graph)
    graph = attach_compiled_graph_evidence(forest.evidence_graph, plan_graph)

    by_id = {node.id: node for node in graph.nodes}
    assert by_id["semantic:wall:0"].compiled_entities[0].collection == "walls"
    assert by_id["room-1"].compiled_entities[0].collection == "rooms"
    assert by_id["door-1"].compiled_entities[0].host_wall_id == "L1:wall:0"
    assert by_id["sink-1"].compiled_entities[0].collection == "fixtures"
    assert by_id["sink-1"].asset_resolution is not None
    assert by_id["sink-1"].asset_resolution.geometry_ref.startswith("mesh:")
    assert "mesh_vertices" not in by_id["sink-1"].model_dump_json()
    assert graph.coverage.promoted_without_compiled_entity_count == 0
    assert graph.coverage.compiled_count_by_kind == {
        "wall": 1,
        "room": 1,
        "opening": 1,
        "fixture": 1,
    }


def test_global_graph_is_the_only_proposal_source() -> None:
    base = AecTileProposal(
        tile_id="full-sheet",
        source_ref_ids=["source-1"],
        model_version="legacy-local-path",
        wall_segments=[
            PixelLineProposal(
                id="legacy:wall:must-not-survive",
                start_px=(0, 0),
                end_px=(1, 0),
                confidence=1,
                uncertainty=0,
                source_ref_ids=["source-1"],
                model_version="legacy",
                review_required=False,
            )
        ],
        symbols=[],
        rejected_candidates=0,
    ).finalize()

    forest = build_forest_perception_bundle(
        base,
        _recognition(),
        source_ref_ids=["source-1"],
    )

    wall_ids = {wall.id for wall in forest.proposal.wall_segments}
    assert wall_ids == {"semantic:wall:0"}
    assert "legacy:wall:must-not-survive" not in wall_ids
    assert forest.proposal.model_version == "dajoong-forest-reconstruction-v2"
