from __future__ import annotations

import numpy as np

from buili_plan2bim.training_augmentation import (
    crop_dense_training_example,
    detail_crop_context,
    deterministic_detail_crop,
    deterministic_quadrant,
    rotate_dense_training_example,
    rotate_element_geometry,
    rotate_normalized_bbox_context,
    rotate_oriented_evidence,
    rotate_spatial_bbox,
)


def test_local_geometry_rotates_with_pixels_without_label_contradiction() -> None:
    horizontal = np.asarray((0.2, -0.1, np.log(0.4), np.log(0.1), 0.0, 1.0))

    vertical = rotate_element_geometry(horizontal, 1, spatial=False)

    assert np.allclose(vertical[:2], (-0.1, -0.2))
    assert np.allclose(vertical[2:4], (np.log(0.1), np.log(0.4)))
    assert np.allclose(vertical[4:], (1.0, 0.0))
    assert np.allclose(
        rotate_element_geometry(horizontal, 4, spatial=False),
        horizontal,
    )


def test_dense_rotation_keeps_all_supervision_channels_aligned() -> None:
    evidence = np.zeros((4, 5, 5), dtype=np.float32)
    topology = np.zeros((6, 5, 5), dtype=np.float32)
    room = np.zeros((5, 5), dtype=np.int64)
    element = np.zeros((5, 5), dtype=np.int64)
    geometry = np.zeros((6, 5, 5), dtype=np.float32)
    valid = np.zeros((5, 5), dtype=np.float32)
    evidence[:, 1, 3] = 1
    topology[:, 1, 3] = 1
    room[1, 3] = 4
    element[1, 3] = 8
    geometry[:, 1, 3] = (0.2, -0.1, np.log(0.4), np.log(0.1), 0.0, 1.0)
    valid[1, 3] = 1

    rotated = rotate_dense_training_example(
        evidence=evidence,
        topology=topology,
        room_semantics=room,
        element_semantics=element,
        element_geometry=geometry,
        element_geometry_valid=valid,
        quadrants=1,
    )

    location = np.argwhere(rotated["element_geometry_valid"] == 1)[0]
    y, x = int(location[0]), int(location[1])
    assert rotated["evidence"][:, y, x].min() == 1
    assert rotated["topology"][:, y, x].min() == 1
    assert rotated["room_semantics"][y, x] == 4
    assert rotated["element_semantics"][y, x] == 8
    assert np.allclose(
        rotated["element_geometry"][:, y, x],
        (-0.1, -0.2, np.log(0.1), np.log(0.4), 1.0, 0.0),
    )


def test_quarter_turn_swaps_oriented_evidence_channel_roles() -> None:
    evidence = np.zeros((4, 5, 5), dtype=np.float32)
    topology = np.zeros((6, 5, 5), dtype=np.float32)
    room = np.zeros((5, 5), dtype=np.int64)
    element = np.zeros((5, 5), dtype=np.int64)
    geometry = np.zeros((6, 5, 5), dtype=np.float32)
    valid = np.zeros((5, 5), dtype=np.float32)
    evidence[1, 2, 1:4] = 1.0

    rotated = rotate_dense_training_example(
        evidence=evidence,
        topology=topology,
        room_semantics=room,
        element_semantics=element,
        element_geometry=geometry,
        element_geometry_valid=valid,
        quadrants=1,
    )

    assert not rotated["evidence"][1].any()
    assert rotated["evidence"][2, 1:4, 2].min() == 1.0


def test_deterministic_quadrant_is_stable_and_bounded() -> None:
    first = deterministic_quadrant("sheet-001", seed=71)

    assert first == deterministic_quadrant("sheet-001", seed=71)
    assert 0 <= first <= 3


def test_detail_crop_is_stable_and_inside_the_sheet() -> None:
    crop = deterministic_detail_crop("sheet-002", seed=71, size=(192, 160))

    assert crop == deterministic_detail_crop("sheet-002", seed=71, size=(192, 160))
    if crop is not None:
        left, top, right, bottom = crop
        assert 0 <= left < right <= 192
        assert 0 <= top < bottom <= 160


def test_detail_crop_context_preserves_location_in_the_whole_sheet() -> None:
    context = detail_crop_context((20, 10, 60, 50), size=(100, 80))

    assert np.allclose(context[:4], (0.2, 0.125, 0.4, 0.5))
    assert np.allclose(context[4:], (0, 0, 0, 0))
    assert np.allclose(
        detail_crop_context(None, size=(100, 80)),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )


def test_detail_crop_context_uses_drawing_frame_not_letterbox_padding() -> None:
    context = detail_crop_context(
        (25, 20, 75, 60),
        size=(100, 100),
        frame_bbox=(0, 20, 100, 80),
    )

    assert np.allclose(context, (0.25, 0, 0.5, 2 / 3, 0, 1, 0, 0))


def test_content_bbox_rotates_with_the_dense_training_frame() -> None:
    assert rotate_spatial_bbox(
        (20, 0, 80, 100),
        size=(100, 100),
        quadrants=1,
    ) == (0, 20, 100, 80)


def test_normalized_candidate_context_rotates_without_float_boundary_failure() -> None:
    context = np.asarray((0.08, 0.12, 0.24, 0.16), dtype=np.float32)

    rotated = rotate_normalized_bbox_context(context, 1)

    assert np.allclose(rotated, (0.12, 0.68, 0.16, 0.24), atol=1e-6)
    assert np.allclose(
        rotate_normalized_bbox_context(context, 4),
        context,
        atol=1e-6,
    )


def test_hierarchical_oriented_evidence_swaps_axis_roles_in_every_view() -> None:
    evidence = np.zeros((12, 3, 3), dtype=np.float32)
    for offset in (0, 4, 8):
        evidence[offset + 1, 0, 1] = offset + 1
        evidence[offset + 2, 2, 1] = offset + 2

    spatial_only = np.rot90(evidence, 1, axes=(-2, -1))
    rotated = rotate_oriented_evidence(evidence, 1)

    for offset in (0, 4, 8):
        assert np.array_equal(rotated[offset], spatial_only[offset])
        assert np.array_equal(rotated[offset + 1], spatial_only[offset + 2])
        assert np.array_equal(rotated[offset + 2], spatial_only[offset + 1])
        assert np.array_equal(rotated[offset + 3], spatial_only[offset + 3])


def test_oriented_evidence_rejects_an_incomplete_channel_group() -> None:
    with np.testing.assert_raises_regex(ValueError, "4\\*n"):
        rotate_oriented_evidence(np.zeros((6, 3, 3), dtype=np.float32), 1)


def test_dense_detail_crop_keeps_semantics_and_geometry_aligned() -> None:
    evidence = np.zeros((4, 8, 8), dtype=np.float32)
    topology = np.zeros((6, 8, 8), dtype=np.float32)
    room = np.zeros((8, 8), dtype=np.int64)
    element = np.zeros((8, 8), dtype=np.int64)
    geometry = np.zeros((6, 8, 8), dtype=np.float32)
    valid = np.zeros((8, 8), dtype=np.float32)
    evidence[:, 3:5, 3:5] = 1
    topology[:, 3:5, 3:5] = 1
    room[3:5, 3:5] = 4
    element[3:5, 3:5] = 8
    yy, xx = np.indices((8, 8), dtype=np.float32)
    geometry[0] = (4 - xx) / 8
    geometry[1] = (4 - yy) / 8
    geometry[2] = np.log(0.25)
    geometry[3] = np.log(0.25)
    geometry[5] = 1
    valid[3:5, 3:5] = 1
    geometry *= valid[None, ...]

    cropped = crop_dense_training_example(
        evidence=evidence,
        topology=topology,
        room_semantics=room,
        element_semantics=element,
        element_geometry=geometry,
        element_geometry_valid=valid,
        bbox=(2, 2, 6, 6),
    )

    assert cropped["evidence"].shape == evidence.shape
    assert set(np.unique(cropped["room_semantics"])) == {0, 4}
    assert set(np.unique(cropped["element_semantics"])) == {0, 8}
    active = cropped["element_geometry_valid"] > 0.5
    assert active.any()
    assert np.allclose(cropped["element_geometry"][2, active], np.log(0.5))


def test_dense_detail_crop_drops_instances_cut_by_the_window() -> None:
    evidence = np.zeros((4, 8, 8), dtype=np.float32)
    topology = np.zeros((6, 8, 8), dtype=np.float32)
    room = np.zeros((8, 8), dtype=np.int64)
    element = np.zeros((8, 8), dtype=np.int64)
    geometry = np.zeros((6, 8, 8), dtype=np.float32)
    valid = np.zeros((8, 8), dtype=np.float32)
    element[2:6, 1:5] = 8
    valid[2:6, 1:5] = 1
    yy, xx = np.indices((8, 8), dtype=np.float32)
    geometry[0] = (3 - xx) / 8
    geometry[1] = (4 - yy) / 8
    geometry[2] = np.log(4 / 8)
    geometry[3] = np.log(4 / 8)
    geometry[5] = 1
    geometry *= valid[None, ...]

    cropped = crop_dense_training_example(
        evidence=evidence,
        topology=topology,
        room_semantics=room,
        element_semantics=element,
        element_geometry=geometry,
        element_geometry_valid=valid,
        bbox=(2, 1, 7, 7),
    )

    assert not cropped["element_geometry_valid"].any()
    assert not cropped["element_semantics"].any()
