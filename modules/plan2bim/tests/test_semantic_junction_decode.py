from __future__ import annotations

import numpy as np

from buili_plan2bim.semantic_junction_decode import (
    JunctionDetection,
    decode_icon_junctions,
    decode_opening_junctions,
)
from buili_plan2bim.semantic_recognition import (
    SemanticDetection,
    SemanticWallVector,
    _merge_collinear_wall_vectors,
    _merge_junction_detections,
    _recover_unclassified_interior_rooms,
)


def test_four_corner_heatmaps_recover_one_typed_fixture() -> None:
    heatmaps = np.zeros((21, 100, 120), dtype=np.float32)
    heatmaps[17, 20, 30] = 0.92
    heatmaps[18, 20, 70] = 0.91
    heatmaps[19, 60, 30] = 0.90
    heatmaps[20, 60, 70] = 0.93
    icons = np.full((11, 100, 120), 0.01, dtype=np.float32)
    icons[0] = 0.1
    icons[6, 20:61, 30:71] = 0.88

    detections = decode_icon_junctions(
        heatmaps,
        icons,
        source_size=(240, 200),
    )

    assert len(detections) == 1
    assert detections[0].class_index == 6
    assert detections[0].bbox_px == (60, 40, 140, 120)
    assert detections[0].evidence_mode == "four_corner_heatmap"


def test_reciprocal_opening_endpoints_require_wall_context() -> None:
    heatmaps = np.zeros((21, 100, 120), dtype=np.float32)
    heatmaps[13, 50, 20] = 0.93
    heatmaps[14, 50, 80] = 0.92
    icons = np.full((11, 100, 120), 0.01, dtype=np.float32)
    icons[1, 46:55, 20:81] = 0.86
    wall_mask = np.zeros((100, 120), dtype=np.bool_)
    wall_mask[46:55, 10:100] = True

    detections = decode_opening_junctions(
        heatmaps,
        icons,
        wall_mask,
        source_size=(120, 100),
    )

    assert len(detections) == 1
    assert detections[0].class_index == 1
    assert detections[0].evidence_mode == "reciprocal_opening_endpoints"

    without_wall = decode_opening_junctions(
        heatmaps,
        icons,
        np.zeros_like(wall_mask),
        source_size=(120, 100),
    )
    assert without_wall == []


def test_near_collinear_four_corners_remain_review_only() -> None:
    detections = _merge_junction_detections(
        [],
        [
            JunctionDetection(
                class_index=4,
                bbox_px=(100, 200, 160, 203),
                confidence=0.91,
                pixel_area=180,
                evidence_mode="four_corner_heatmap",
            )
        ],
        icon_classes=(
            "Background",
            "Window",
            "Door",
            "Closet",
            "Electrical appliance",
            "Toilet",
            "Sink",
            "Sauna bench",
            "Fireplace",
            "Bathtub",
            "Chimney",
        ),
    )

    assert len(detections) == 1
    assert detections[0].review_required is True
    assert detections[0].promote_to_bim is False


def test_complete_four_corner_geometry_can_corroborate_segmentation() -> None:
    detections = _merge_junction_detections(
        [
            SemanticDetection(
                id="semantic:3:1",
                class_name="Closet",
                symbol_class="closet",
                bbox_px=(100, 200, 160, 260),
                confidence=0.61,
                pixel_area=1800,
                review_required=True,
                promote_to_bim=False,
            )
        ],
        [
            JunctionDetection(
                class_index=3,
                bbox_px=(101, 199, 161, 261),
                confidence=0.65,
                pixel_area=3720,
                evidence_mode="four_corner_heatmap",
            )
        ],
        icon_classes=(
            "Background",
            "Window",
            "Door",
            "Closet",
            "Electrical appliance",
            "Toilet",
            "Sink",
            "Sauna bench",
            "Fireplace",
            "Bathtub",
            "Chimney",
        ),
    )

    assert len(detections) == 1
    assert detections[0].evidence_mode == "segmentation_component+four_corner_heatmap"
    assert detections[0].review_required is False
    assert detections[0].promote_to_bim is True


def test_fragmented_collinear_wall_runs_merge_across_opening_gap() -> None:
    vectors = [
        SemanticWallVector(start_px=(10, 40), end_px=(90, 40), thickness_px=20),
        SemanticWallVector(start_px=(112, 42), end_px=(210, 42), thickness_px=22),
    ]

    merged = _merge_collinear_wall_vectors(vectors)

    assert len(merged) == 1
    assert merged[0].start_px[0] == 10
    assert merged[0].end_px[0] == 210
    assert 40 <= merged[0].start_px[1] <= 42


def test_nearby_parallel_walls_do_not_merge() -> None:
    vectors = [
        SemanticWallVector(start_px=(10, 40), end_px=(90, 40), thickness_px=8),
        SemanticWallVector(start_px=(15, 47), end_px=(95, 47), thickness_px=8),
    ]

    assert len(_merge_collinear_wall_vectors(vectors)) == 2


def test_large_enclosed_unclassified_interior_is_preserved() -> None:
    walls = [
        SemanticWallVector(start_px=(10, 10), end_px=(90, 10), thickness_px=6),
        SemanticWallVector(start_px=(90, 10), end_px=(90, 90), thickness_px=6),
        SemanticWallVector(start_px=(90, 90), end_px=(10, 90), thickness_px=6),
        SemanticWallVector(start_px=(10, 90), end_px=(10, 10), thickness_px=6),
    ]

    rooms = _recover_unclassified_interior_rooms([], walls, source_size=(100, 100))

    assert len(rooms) == 1
    assert rooms[0].class_name == "Unclassified interior"
    assert rooms[0].pixel_area > 5_000
    assert rooms[0].review_required is True


def test_open_wall_does_not_invent_unclassified_room() -> None:
    walls = [
        SemanticWallVector(start_px=(10, 10), end_px=(90, 10), thickness_px=6),
        SemanticWallVector(start_px=(90, 10), end_px=(90, 90), thickness_px=6),
        SemanticWallVector(start_px=(10, 90), end_px=(10, 10), thickness_px=6),
    ]

    assert _recover_unclassified_interior_rooms([], walls, source_size=(100, 100)) == []
