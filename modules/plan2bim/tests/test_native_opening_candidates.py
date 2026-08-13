from PIL import Image, ImageDraw

from buili_plan2bim.core.model.aec_decode import PixelLineProposal, PixelSymbolProposal
from buili_plan2bim.native_opening_candidates import (
    consolidate_walls_across_openings,
    infer_openings_from_wall_graph,
)


def _wall() -> PixelLineProposal:
    return PixelLineProposal(
        id="wall",
        start_px=(10.0, 50.0),
        end_px=(190.0, 50.0),
        thickness_px=12.0,
        confidence=0.95,
        uncertainty=0.05,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )


def test_wall_gap_with_swing_is_a_hosted_door() -> None:
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 44, 74, 56), fill="black")
    draw.rectangle((125, 44, 190, 56), fill="black")
    draw.line((75, 44, 75, 95), fill="black", width=2)
    draw.arc((75, 44, 177, 146), 180, 270, fill="black", width=2)

    openings, diagnostics = infer_openings_from_wall_graph(
        image,
        [_wall()],
        model_version="test",
    )

    assert len(openings) == 1
    assert openings[0].symbol_class == "door"
    assert 74 <= openings[0].bbox_px[0] <= 77
    assert 123 <= openings[0].bbox_px[2] <= 126
    assert diagnostics.door_count == 1


def test_plain_wall_gap_is_a_hosted_window() -> None:
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 44, 74, 56), fill="black")
    draw.rectangle((125, 44, 190, 56), fill="black")
    draw.line((75, 48, 125, 48), fill="black", width=1)
    draw.line((75, 52, 125, 52), fill="black", width=1)

    openings, diagnostics = infer_openings_from_wall_graph(
        image,
        [_wall()],
        model_version="test",
    )

    assert len(openings) == 1
    assert openings[0].symbol_class == "window"
    assert diagnostics.window_count == 1


def test_host_wall_fragments_are_joined_only_across_detected_opening() -> None:
    left = _wall().model_copy(update={"id": "left", "end_px": (74.0, 50.0)})
    right = _wall().model_copy(update={"id": "right", "start_px": (125.0, 50.0)})
    opening = PixelSymbolProposal(
        id="opening",
        symbol_class="door",
        center_px=(99.5, 50.0),
        bbox_px=(75.0, 44.0, 125.0, 56.0),
        confidence=0.95,
        uncertainty=0.05,
        source_ref_ids=["source"],
        model_version="test",
        review_required=True,
    )

    joined = consolidate_walls_across_openings(
        [left, right],
        [opening],
        image_size=(200, 300),
    )

    assert len(joined) == 1
    assert joined[0].start_px == (10.0, 50.0)
    assert joined[0].end_px == (190.0, 50.0)
