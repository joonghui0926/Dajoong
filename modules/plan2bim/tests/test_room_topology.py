from __future__ import annotations

from buili_plan2bim.core.model.aec_decode import PixelLineProposal, PixelRoomProposal
from buili_plan2bim.room_topology import (
    merge_topology_and_provisional_rooms,
    reconstruct_rooms_from_wall_graph,
)


def _wall(
    wall_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
) -> PixelLineProposal:
    return PixelLineProposal(
        id=wall_id,
        start_px=start,
        end_px=end,
        thickness_px=8,
        confidence=1,
        uncertainty=0,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )


def test_wall_faces_recover_two_rooms_and_reject_bad_semantic_name() -> None:
    walls = [
        _wall("top", (10, 10), (190, 10)),
        _wall("bottom", (10, 110), (190, 110)),
        _wall("left", (10, 10), (10, 110)),
        _wall("right", (190, 10), (190, 110)),
        _wall("middle", (100, 10), (100, 110)),
    ]
    false_semantic = PixelRoomProposal(
        id="bad",
        name="outdoor 1",
        room_class="outdoor",
        polygon_px=[(15, 15), (185, 15), (185, 105), (15, 105)],
        confidence=0.99,
        uncertainty=0.01,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    rooms = reconstruct_rooms_from_wall_graph(
        walls,
        [false_semantic],
        source_size=(200, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 2
    assert {room.room_class for room in rooms} == {"unknown"}
    assert all(room.review_required for room in rooms)


def test_wall_face_accepts_one_to_one_high_confidence_semantics() -> None:
    walls = [
        _wall("top", (10, 10), (100, 10)),
        _wall("bottom", (10, 110), (100, 110)),
        _wall("left", (10, 10), (10, 110)),
        _wall("right", (100, 10), (100, 110)),
    ]
    semantic = PixelRoomProposal(
        id="good",
        name="bathroom 1",
        room_class="bathroom",
        polygon_px=[(14, 14), (96, 14), (96, 106), (14, 106)],
        confidence=0.96,
        uncertainty=0.04,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    rooms = reconstruct_rooms_from_wall_graph(
        walls,
        [semantic],
        source_size=(120, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 1
    assert rooms[0].room_class == "bathroom"


def test_wall_face_accepts_localized_room_label_seed() -> None:
    walls = [
        _wall("top", (10, 10), (100, 10)),
        _wall("bottom", (10, 110), (100, 110)),
        _wall("left", (10, 10), (10, 110)),
        _wall("right", (100, 10), (100, 110)),
    ]
    seed = PixelRoomProposal(
        id="seed",
        name="bathroom seed",
        room_class="bathroom",
        polygon_px=[(48, 52), (62, 52), (62, 68), (48, 68)],
        confidence=0.96,
        uncertainty=0.04,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    rooms = reconstruct_rooms_from_wall_graph(
        walls,
        [seed],
        source_size=(120, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 1
    assert rooms[0].room_class == "bathroom"
    assert rooms[0].confidence == 0.96


def test_room_label_seed_spanning_two_faces_is_rejected() -> None:
    walls = [
        _wall("top", (10, 10), (190, 10)),
        _wall("bottom", (10, 110), (190, 110)),
        _wall("left", (10, 10), (10, 110)),
        _wall("right", (190, 10), (190, 110)),
        _wall("middle", (100, 10), (100, 110)),
    ]
    ambiguous_seed = PixelRoomProposal(
        id="ambiguous",
        name="bathroom seed",
        room_class="bathroom",
        polygon_px=[(88, 48), (112, 48), (112, 72), (88, 72)],
        confidence=0.98,
        uncertainty=0.02,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    rooms = reconstruct_rooms_from_wall_graph(
        walls,
        [ambiguous_seed],
        source_size=(200, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 2
    assert {room.room_class for room in rooms} == {"unknown"}


def test_conflicting_room_label_seeds_remain_reviewable() -> None:
    walls = [
        _wall("top", (10, 10), (100, 10)),
        _wall("bottom", (10, 110), (100, 110)),
        _wall("left", (10, 10), (10, 110)),
        _wall("right", (100, 10), (100, 110)),
    ]

    def semantic(room_class: str, confidence: float) -> PixelRoomProposal:
        return PixelRoomProposal(
            id=room_class,
            name=f"{room_class} seed",
            room_class=room_class,
            polygon_px=[(45, 50), (65, 50), (65, 70), (45, 70)],
            confidence=confidence,
            uncertainty=1.0 - confidence,
            source_ref_ids=["source"],
            model_version="test",
            review_required=False,
        )

    rooms = reconstruct_rooms_from_wall_graph(
        walls,
        [semantic("bathroom", 0.96), semantic("bedroom", 0.93)],
        source_size=(120, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 1
    assert rooms[0].room_class == "unknown"
    assert rooms[0].review_required


def test_partial_sheet_room_survives_closed_face_reconstruction() -> None:
    topology = PixelRoomProposal(
        id="topology-room",
        name="bedroom 1",
        room_class="bedroom",
        polygon_px=[(10, 10), (90, 10), (90, 90), (10, 90)],
        confidence=0.96,
        uncertainty=0.04,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )
    partial = PixelRoomProposal(
        id="provisional-boundary-room",
        name="living 2",
        room_class="living",
        polygon_px=[(105, 10), (195, 10), (195, 120), (105, 120)],
        confidence=0.93,
        uncertainty=0.07,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    rooms = merge_topology_and_provisional_rooms(
        [topology],
        [partial],
        source_size=(200, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert len(rooms) == 2
    preserved = next(room for room in rooms if room.id.startswith("partial:"))
    assert preserved.room_class == "living"
    assert preserved.review_required
    assert preserved.confidence == 0.71
    assert "unresolved-partial-room-preservation-v1" in preserved.model_version


def test_overlapping_provisional_room_does_not_duplicate_topology_room() -> None:
    topology = PixelRoomProposal(
        id="topology-room",
        name="bathroom 1",
        room_class="bathroom",
        polygon_px=[(10, 10), (100, 10), (100, 100), (10, 100)],
        confidence=0.97,
        uncertainty=0.03,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )
    provisional = topology.model_copy(
        update={
            "id": "provisional-room",
            "polygon_px": [(14, 14), (96, 14), (96, 96), (14, 96)],
        }
    )

    rooms = merge_topology_and_provisional_rooms(
        [topology],
        [provisional],
        source_size=(120, 120),
        source_ref_ids=["source"],
        model_version="test",
    )

    assert [room.id for room in rooms] == ["topology-room"]
