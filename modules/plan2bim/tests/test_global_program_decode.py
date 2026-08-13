from __future__ import annotations

import numpy as np

from buili_plan2bim.core.model.global_topology_student import (
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
)
from buili_plan2bim.global_program_decode import (
    _refine_element_bbox_from_raster,
    decode_global_program,
)
from buili_plan2bim.perception_forest import build_spatial_evidence_graph_from_proposal


def _synthetic_outputs() -> tuple[np.ndarray, ...]:
    height = width = 64
    topology = np.full((6, height, width), -8.0, dtype=np.float32)
    topology[1, 8:11, 8:56] = 8
    topology[1, 53:56, 8:56] = 8
    topology[1, 8:56, 8:11] = 8
    topology[1, 8:56, 53:56] = 8
    topology[0] = topology[1]
    topology[5, 11:53, 11:53] = 8
    topology[4, 30:34, 30:34] = 8

    rooms = np.full(
        (len(ROOM_PROGRAM_CLASSES), height, width),
        -5.0,
        dtype=np.float32,
    )
    rooms[0] = 2
    rooms[ROOM_PROGRAM_CLASSES.index("living"), 11:53, 11:53] = 8

    elements = np.full(
        (len(ELEMENT_PROGRAM_CLASSES), height, width),
        -5.0,
        dtype=np.float32,
    )
    elements[0] = 2
    elements[ELEMENT_PROGRAM_CLASSES.index("door"), 7:12, 28:36] = 8
    elements[ELEMENT_PROGRAM_CLASSES.index("sink"), 28:34, 28:35] = 8
    geometry = np.zeros((6, height, width), dtype=np.float32)
    for y_slice, x_slice in (
        (slice(7, 12), slice(28, 36)),
        (slice(28, 34), slice(28, 35)),
    ):
        yy, xx = np.mgrid[y_slice, x_slice]
        center_x = (xx.min() + xx.max() + 1) / 2
        center_y = (yy.min() + yy.max() + 1) / 2
        geometry[0, y_slice, x_slice] = (center_x - xx) / width
        geometry[1, y_slice, x_slice] = (center_y - yy) / height
        geometry[2, y_slice, x_slice] = np.log((xx.max() - xx.min() + 1) / width)
        geometry[3, y_slice, x_slice] = np.log((yy.max() - yy.min() + 1) / height)
        geometry[5, y_slice, x_slice] = 1
    uncertainty = np.full((3, height, width), 0.1, dtype=np.float32)
    return topology, rooms, elements, geometry, uncertainty


def test_global_program_decoder_recovers_one_shared_building_program() -> None:
    topology, rooms, elements, geometry, uncertainty = _synthetic_outputs()

    result = decode_global_program(
        tile_id="sheet:global",
        source_ref_ids=["source:1"],
        model_version="global-program-test",
        source_size=(640, 640),
        topology_logits=topology,
        room_semantic_logits=rooms,
        element_semantic_logits=elements,
        element_geometry=geometry,
        uncertainty=uncertainty,
    )

    assert result.diagnostics.full_sheet_context is True
    assert result.diagnostics.release_eligible is False
    assert len(result.proposal.wall_segments) == 4
    assert [room.room_class for room in result.proposal.room_regions] == ["living"]
    assert {symbol.symbol_class for symbol in result.proposal.symbols} == {"door", "sink"}
    assert min(point[0] for point in result.proposal.room_regions[0].polygon_px) >= 100
    assert all(item.source_ref_ids == ["source:1"] for item in result.proposal.symbols)

    graph = build_spatial_evidence_graph_from_proposal(
        result.proposal,
        source_size=(640, 640),
        full_sheet_context=True,
    )
    relation_kinds = {relation.kind for relation in graph.relations}
    assert {"joins_wall", "bounds_room", "host_candidate", "inside_room"}.issubset(
        relation_kinds
    )
    assert graph.topology_integrity.wall_component_count == 1
    assert graph.coverage.unhosted_opening_count == 0
    assert graph.coverage.unassigned_fixture_count == 0


def test_global_program_decoder_rejects_contract_mismatch() -> None:
    topology, rooms, elements, geometry, uncertainty = _synthetic_outputs()

    try:
        decode_global_program(
            tile_id="sheet:global",
            source_ref_ids=["source:1"],
            model_version="global-program-test",
            source_size=(640, 640),
            topology_logits=topology[:5],
            room_semantic_logits=rooms,
            element_semantic_logits=elements,
            element_geometry=geometry,
            uncertainty=uncertainty,
        )
    except ValueError as error:
        assert "topology_logits" in str(error)
    else:  # pragma: no cover
        raise AssertionError("contract mismatch should fail closed")


def test_room_semantics_can_be_a_localized_label_seed() -> None:
    topology, rooms, elements, geometry, uncertainty = _synthetic_outputs()
    rooms[:] = -5.0
    rooms[0] = 2.0
    rooms[ROOM_PROGRAM_CLASSES.index("living"), 30:35, 30:35] = 10.0

    result = decode_global_program(
        tile_id="sheet:localized-label",
        source_ref_ids=["source:1"],
        model_version="global-program-test",
        source_size=(640, 640),
        topology_logits=topology,
        room_semantic_logits=rooms,
        element_semantic_logits=elements,
        element_geometry=geometry,
        uncertainty=uncertainty,
    )

    assert [room.room_class for room in result.proposal.room_regions] == ["living"]
    assert [room.room_class for room in result.room_semantic_seeds] == ["living"]


def test_multiple_room_seeds_survive_one_provisional_room_component() -> None:
    topology, rooms, elements, geometry, uncertainty = _synthetic_outputs()
    topology[4] = -8.0
    topology[4, 18:23, 18:23] = 8.0
    topology[4, 40:45, 40:45] = 8.0
    rooms[:] = -5.0
    rooms[0] = 2.0
    rooms[ROOM_PROGRAM_CLASSES.index("living"), 18:23, 18:23] = 10.0
    rooms[ROOM_PROGRAM_CLASSES.index("bathroom"), 40:45, 40:45] = 10.0

    result = decode_global_program(
        tile_id="sheet:two-labels-one-provisional-room",
        source_ref_ids=["source:1"],
        model_version="global-program-test",
        source_size=(640, 640),
        topology_logits=topology,
        room_semantic_logits=rooms,
        element_semantic_logits=elements,
        element_geometry=geometry,
        uncertainty=uncertainty,
    )

    assert len(result.proposal.room_regions) == 1
    assert {seed.room_class for seed in result.room_semantic_seeds} == {
        "living",
        "bathroom",
    }


def test_native_element_refinement_measures_symbol_without_absorbing_wall_runs() -> None:
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[30, 10:90] = 0
    gray[42, 39:63] = 0
    gray[59, 39:63] = 0
    gray[42:60, 39] = 0
    gray[42:60, 62] = 0

    refined = _refine_element_bbox_from_raster(gray, (32.0, 36.0, 70.0, 66.0))

    assert refined == (39.0, 42.0, 63.0, 60.0)
