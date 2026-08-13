from __future__ import annotations

from PIL import Image, ImageDraw

from buili_plan2bim.core.model.aec_decode import PixelLineProposal
from buili_plan2bim.native_wall_candidates import (
    mine_native_wall_candidates,
    promote_supported_native_wall_candidates,
    refine_context_walls_with_native_bands,
    unresolved_native_wall_candidates,
)


def test_native_wall_ledger_finds_paired_wall_edges() -> None:
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 390, 270), outline="black", width=5)
    draw.line((150, 30, 150, 270), fill="black", width=5)
    draw.line((250, 30, 250, 270), fill="black", width=5)

    candidates, diagnostics = mine_native_wall_candidates(image)

    assert diagnostics.candidate_count == len(candidates)
    assert diagnostics.foreground_pixels > 0
    assert any(item.orientation == "vertical" for item in candidates)
    assert any(item.orientation == "horizontal" for item in candidates)


def test_native_wall_ledger_preserves_thick_joined_envelope_bands() -> None:
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 25, 390, 275), outline="black", width=28)

    candidates, diagnostics = mine_native_wall_candidates(image)

    assert diagnostics.adaptive_maximum_thickness_px >= 28
    horizontal = [item for item in candidates if item.orientation == "horizontal"]
    vertical = [item for item in candidates if item.orientation == "vertical"]
    assert any(abs(item.start_px[1] - 38.5) <= 3 and item.thickness_px >= 24 for item in horizontal)
    assert any(abs(item.start_px[0] - 43.5) <= 3 and item.thickness_px >= 24 for item in vertical)


def test_native_band_geometry_refines_context_without_inventing_topology() -> None:
    context = [
        PixelLineProposal(
            id="learned",
            start_px=(35, 38),
            end_px=(380, 38),
            thickness_px=12,
            confidence=0.99,
            uncertainty=0.01,
            source_ref_ids=["source"],
            model_version="test",
            review_required=False,
        )
    ]
    image = Image.new("RGB", (420, 300), "white")
    ImageDraw.Draw(image).rectangle((30, 25, 390, 55), fill="black")
    candidates, _ = mine_native_wall_candidates(image)

    refined, count = refine_context_walls_with_native_bands(context, candidates)

    assert count == 1
    assert refined[0].start_px[0] <= 31
    assert refined[0].end_px[0] >= 389
    assert refined[0].thickness_px >= 28
    assert "native-band-centerline" in refined[0].model_version


def test_native_wall_ledger_exposes_unexplained_run() -> None:
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 390, 270), outline="black", width=5)
    candidates, _ = mine_native_wall_candidates(image)
    one_wall = PixelLineProposal(
        id="known",
        start_px=(30, 30),
        end_px=(390, 30),
        thickness_px=5,
        confidence=1,
        uncertainty=0,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    unresolved = unresolved_native_wall_candidates(candidates, [one_wall])

    assert unresolved
    assert any(abs(item.start_px[0] - item.end_px[0]) < 1 for item in unresolved)


def test_native_wall_promotion_does_not_duplicate_aligned_or_isolated_lines() -> None:
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 390, 270), outline="black", width=5)
    draw.line((150, 30, 150, 270), fill="black", width=5)
    draw.line((270, 100, 330, 100), fill="black", width=3)
    candidates, _ = mine_native_wall_candidates(image)
    context = [
        PixelLineProposal(
            id="known",
            start_px=(30, 30),
            end_px=(390, 30),
            thickness_px=5,
            confidence=1,
            uncertainty=0,
            source_ref_ids=["source"],
            model_version="test",
            review_required=False,
        )
    ]

    promoted, rejected = promote_supported_native_wall_candidates(
        candidates,
        context,
        source_ref_ids=["source"],
    )

    assert promoted == []
    assert rejected


def test_connected_native_wall_network_bootstraps_an_empty_global_graph() -> None:
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    # Draw both physical faces of each wall. A filled line is deliberately
    # ineligible for source-only bootstrap because it has no paired-edge proof.
    draw.rectangle((25, 20, 395, 280), outline="black", width=3)
    draw.rectangle((33, 28, 387, 272), outline="black", width=3)
    draw.line((196, 28, 196, 272), fill="black", width=3)
    draw.line((204, 28, 204, 272), fill="black", width=3)
    candidates, _ = mine_native_wall_candidates(image)

    promoted, _ = promote_supported_native_wall_candidates(
        candidates,
        [],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert len(promoted) >= 3
    assert {"horizontal", "vertical"} == {
        (
            "horizontal"
            if abs(item.end_px[0] - item.start_px[0]) >= abs(item.end_px[1] - item.start_px[1])
            else "vertical"
        )
        for item in promoted
    }
    assert all(item.review_required for item in promoted)
    assert all("network-bootstrap" in item.model_version for item in promoted)


def test_small_paired_furniture_box_cannot_bootstrap_topology() -> None:
    image = Image.new("RGB", (420, 300), "white")
    ImageDraw.Draw(image).rectangle((160, 120, 260, 180), outline="black", width=10)
    candidates, _ = mine_native_wall_candidates(image)

    promoted, rejected = promote_supported_native_wall_candidates(
        candidates,
        [],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert promoted == []
    assert rejected


def test_all_large_disconnected_structural_networks_bootstrap() -> None:
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    # Two building wings separated by a deliberately blank circulation gap.
    # Both are full structural networks and neither may disappear merely
    # because the other contains more native wall bands.
    for bounds in ((30, 25, 390, 575), (510, 25, 870, 575)):
        left, top, right, bottom = bounds
        draw.rectangle(bounds, outline="black", width=3)
        draw.rectangle((left + 8, top + 8, right - 8, bottom - 8), outline="black", width=3)
        middle = (left + right) // 2
        draw.line((middle - 4, top + 8, middle - 4, bottom - 8), fill="black", width=3)
        draw.line((middle + 4, top + 8, middle + 4, bottom - 8), fill="black", width=3)
    candidates, _ = mine_native_wall_candidates(image)

    promoted, _ = promote_supported_native_wall_candidates(
        candidates,
        [],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert any(item.start_px[0] < 400 for item in promoted)
    assert any(item.start_px[0] > 500 for item in promoted)


def test_opening_sized_gap_keeps_large_wall_network_connected() -> None:
    image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    # The right vertical wall stops short of the upper horizontal by 72 px,
    # representing a large opening. The remaining network spans the sheet and
    # should survive as one structural component.
    for offset in (0, 8):
        draw.line((30, 30 + offset, 500, 30 + offset), fill="black", width=3)
        draw.line((30 + offset, 30, 30 + offset, 570), fill="black", width=3)
        draw.line((508 + offset, 110, 508 + offset, 570), fill="black", width=3)
        draw.line((30, 562 + offset, 516, 562 + offset), fill="black", width=3)
    candidates, _ = mine_native_wall_candidates(image)

    promoted, _ = promote_supported_native_wall_candidates(
        candidates,
        [],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert len(promoted) >= 4


def test_page_scale_solid_wall_bands_can_bootstrap() -> None:
    image = Image.new("RGB", (500, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 25, 465, 875), outline="black", width=22)

    candidates, _ = mine_native_wall_candidates(image)
    promoted, _ = promote_supported_native_wall_candidates(
        candidates,
        [],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert len(promoted) >= 4
    assert all(item.review_required for item in promoted)


def test_one_existing_wall_does_not_disable_whole_graph_bootstrap() -> None:
    image = Image.new("RGB", (500, 900), "white")
    ImageDraw.Draw(image).rectangle((35, 25, 465, 875), outline="black", width=22)
    candidates, _ = mine_native_wall_candidates(image)
    one_model_wall = PixelLineProposal(
        id="partial-model-wall",
        start_px=(35.0, 36.0),
        end_px=(465.0, 36.0),
        thickness_px=22.0,
        confidence=0.95,
        uncertainty=0.05,
        source_ref_ids=["source"],
        model_version="test",
        review_required=False,
    )

    promoted, _ = promote_supported_native_wall_candidates(
        candidates,
        [one_model_wall],
        source_ref_ids=["source"],
        source_size=image.size,
    )

    assert any(abs(item.start_px[0] - item.end_px[0]) < 1 for item in promoted)
