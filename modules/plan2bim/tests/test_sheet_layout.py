from __future__ import annotations

from PIL import Image, ImageDraw

from buili_plan2bim.sheet_layout import (
    _attach_structural_satellites,
    discover_plan_regions,
)
from buili_plan2bim.semantic_multiview import _select_regions


def _draw_plan(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline="black", width=5)
    middle_x = (left + right) // 2
    middle_y = (top + bottom) // 2
    draw.line((middle_x, top, middle_x, bottom), fill="black", width=4)
    draw.line((left, middle_y, right, middle_y), fill="black", width=4)
    draw.rectangle((left + 18, top + 18, middle_x - 18, middle_y - 18), outline="black", width=3)


def test_discovers_two_side_by_side_plan_instances() -> None:
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 992, 592), outline="black", width=3)
    draw.rectangle((15, 510, 985, 585), outline="black", width=3)
    for x in (300, 620, 830):
        draw.line((x, 510, x, 585), fill="black", width=2)
    _draw_plan(draw, (80, 70, 440, 470))
    _draw_plan(draw, (560, 70, 920, 470))

    result = discover_plan_regions(image, sheet_id="A-101")

    assert result.multi_plan_candidate is True
    assert len(result.regions) == 2
    assert result.regions[0].bbox_px[2] < result.regions[1].bbox_px[0]
    assert all(region.review_required for region in result.regions)
    assert result.content_sha256


def test_preserves_single_plan_as_one_instance() -> None:
    image = Image.new("RGB", (700, 700), "white")
    draw = ImageDraw.Draw(image)
    _draw_plan(draw, (80, 80, 620, 620))

    result = discover_plan_regions(image, sheet_id="A-102")

    assert result.multi_plan_candidate is False
    assert len(result.regions) == 1
    assert result.regions[0].crop_to_sheet_transform[0][2] > 0


def test_dimension_strip_does_not_become_a_second_plan() -> None:
    image = Image.new("RGB", (1000, 800), "white")
    draw = ImageDraw.Draw(image)
    _draw_plan(draw, (160, 80, 900, 720))
    # Its area is above the legacy 1.8% seed threshold, but it has no room-scale
    # extent on the horizontal axis and touches the sheet edge like furniture.
    draw.rectangle((0, 160, 55, 700), outline="black", width=4)
    for y in range(180, 690, 35):
        draw.line((0, y, 55, y), fill="black", width=3)

    result = discover_plan_regions(image, sheet_id="A-102-dimension-strip")

    assert result.multi_plan_candidate is False
    assert len(result.regions) == 1


def test_long_interior_exterior_walls_are_not_removed_as_page_furniture() -> None:
    image = Image.new("RGB", (700, 700), "white")
    draw = ImageDraw.Draw(image)
    # The four building walls each span more than 72% of the sheet. The retired
    # furniture cleanup erased all four before plan-instance discovery.
    draw.rectangle((70, 70, 630, 630), outline="black", width=12)
    draw.line((70, 350, 630, 350), fill="black", width=10)
    draw.line((350, 350, 350, 630), fill="black", width=10)

    result = discover_plan_regions(image, sheet_id="A-102-envelope")

    assert len(result.regions) == 1
    left, top, right, bottom = result.regions[0].bbox_px
    assert left <= 70 and top <= 70 and right >= 631 and bottom >= 631


def test_title_block_and_page_border_do_not_join_four_narrow_plans() -> None:
    image = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((38, 38, 1562, 1062), outline="black", width=3)
    title_top = 930
    draw.rectangle((38, title_top, 1562, 1062), outline="black", width=3)
    for x in (390, 650, 910, 1240):
        draw.line((x, title_top, x, 1062), fill="black", width=3)
    draw.line((38, 1000, 1562, 1000), fill="black", width=2)
    plan_boxes = (
        (130, 240, 330, 820),
        (470, 220, 670, 800),
        (930, 230, 1130, 810),
        (1270, 240, 1470, 820),
    )
    for box in plan_boxes:
        _draw_plan(draw, box)

    result = discover_plan_regions(image, sheet_id="A-103")

    assert result.multi_plan_candidate is True
    assert len(result.regions) == 4
    regions_left_to_right = sorted(result.regions, key=lambda region: region.bbox_px[0])
    assert all(
        left.bbox_px[2] < right.bbox_px[0]
        for left, right in zip(
            regions_left_to_right,
            regions_left_to_right[1:],
        )
    )
    for region, expected in zip(regions_left_to_right, plan_boxes, strict=True):
        left, top, right, bottom = region.bbox_px
        expected_left, expected_top, expected_right, expected_bottom = expected
        assert left <= expected_left < expected_right <= right
        assert top <= expected_top < expected_bottom <= bottom
        assert bottom < title_top


def test_satellite_walls_expand_one_plan_without_joining_plan_seeds() -> None:
    regions = _attach_structural_satellites(
        [(300, 80, 700, 520), (800, 80, 1180, 520)],
        [
            (270, 180, 292, 480),
            (170, 210, 245, 245),
            (745, 180, 755, 480),
        ],
        maximum_gap=30,
    )

    assert regions[0] == (170, 80, 700, 520)
    assert regions[1] == (800, 80, 1180, 520)


def test_marks_a_plan_that_continues_through_source_edge_as_truncated() -> None:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    _draw_plan(draw, (80, 80, 720, 590))
    for x in range(120, 700, 40):
        draw.line((x, 590, x, 599), fill="black", width=4)

    result = discover_plan_regions(image, sheet_id="A-104")

    assert result.source_edge_truncation is True
    assert result.source_edge_structural_pixels["bottom"] > 0


def test_region_thresholds_never_truncate_discovered_plan_instances() -> None:
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)
    _draw_plan(draw, (80, 70, 440, 470))
    _draw_plan(draw, (560, 70, 920, 470))

    result = discover_plan_regions(
        image,
        sheet_id="A-105",
        maximum_regions=1,
    )

    assert result.region_overflow is True
    assert result.discovered_region_count == 2
    assert len(result.regions) == 2
    assert len(
        _select_regions(result, global_scale=0.2, maximum_region_passes=1)
    ) == 2
