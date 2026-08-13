"""Whole-sheet plan-instance discovery without assuming one plan per page.

This stage is proposal-only.  It preserves full-sheet coordinates and never
turns its own boxes into ground truth.  A downstream reviewer or a separately
qualified layout model must confirm the regions before production release.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .core.hashing import sha256_json

_DETECTOR_VERSION = "dajoong-whole-sheet-plan-instance-v4-preserve-building-envelope"


class SheetPlanRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    bbox_px: tuple[int, int, int, int]
    crop_to_sheet_transform: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    sheet_area_fraction: float = Field(gt=0, le=1)
    structural_pixel_count: int = Field(ge=0)
    structural_density: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    review_required: bool = True


class SheetLayoutAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.sheet-layout-analysis.v1"
    detector_version: str = _DETECTOR_VERSION
    sheet_id: str
    image_size_px: tuple[int, int]
    regions: list[SheetPlanRegion]
    multi_plan_candidate: bool
    unassigned_structural_fraction: float = Field(ge=0, le=1)
    region_overflow: bool = False
    discovered_region_count: int = Field(default=0, ge=0)
    source_edge_truncation: bool = False
    source_edge_structural_pixels: dict[str, int] = Field(default_factory=dict)
    review_required: bool = True
    content_sha256: str = ""

    def finalize(self) -> SheetLayoutAnalysis:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def _bbox_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int]:
    horizontal = max(0, max(left[0], right[0]) - min(left[2], right[2]))
    vertical = max(0, max(left[1], right[1]) - min(left[3], right[3]))
    return horizontal, vertical


def _bbox_union(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of true values in a one-dimensional mask."""

    padded = np.concatenate(
        (
            np.zeros(1, dtype=np.int8),
            np.asarray(mask, dtype=np.int8),
            np.zeros(1, dtype=np.int8),
        )
    )
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _remove_sheet_furniture(structural: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Remove page borders and ruled title blocks before region grouping.

    Page furniture is detected only from strokes that span most of the full
    sheet.  That distinction is important: a long wall inside one plan remains
    valid evidence, while a border or title-block rule can otherwise join every
    view on the page into one false full-sheet component.
    """

    cleaned = structural.copy()
    height, width = cleaned.shape
    minimum_side = min(width, height)
    row_coverage = structural.sum(axis=1)
    column_coverage = structural.sum(axis=0)
    long_row_runs = _true_runs(row_coverage >= width * 0.72)
    long_column_runs = _true_runs(column_coverage >= height * 0.72)

    title_block_top: int | None = None
    for top, bottom in long_row_runs:
        remaining_fraction = (height - top) / max(1, height)
        if top < height * 0.72 or not 0.04 <= remaining_fraction <= 0.28:
            continue
        band = structural[top:, :]
        band_height = max(1, band.shape[0])
        vertical_rules = _true_runs(band.sum(axis=0) >= band_height * 0.30)
        internal_vertical_rules = [
            run
            for run in vertical_rules
            if run[0] > width * 0.01 and run[1] < width * 0.99
        ]
        horizontal_rules = _true_runs(band.sum(axis=1) >= width * 0.18)
        # A ruled title block has a near-page-width top rule, several internal
        # column dividers, and more than one horizontal rule below it.  Requiring
        # all three prevents a single long exterior wall from truncating a plan.
        if len(internal_vertical_rules) >= 3 and len(horizontal_rules) >= 2:
            title_block_top = top
            break

    if title_block_top is not None:
        cleaned[title_block_top:, :] = False

    line_padding = max(2, round(minimum_side * 0.0015))
    # A long stroke is not page furniture merely because it spans most of the
    # raster. Exterior walls are expected to do exactly that. The retired rule
    # erased every >=72%-wide/height stroke and therefore removed the building
    # envelope before inference. Only strokes physically at the source-image
    # edge are safe to discard without understanding the drawing.
    source_edge_band = max(3, round(minimum_side * 0.006))
    page_row_runs = [
        (top, bottom)
        for top, bottom in long_row_runs
        if top <= source_edge_band or bottom >= height - source_edge_band
    ]
    page_column_runs = [
        (left, right)
        for left, right in long_column_runs
        if left <= source_edge_band or right >= width - source_edge_band
    ]
    # Some scanned sheets place the page frame a few percent inward. Remove it
    # only when all four strokes make a connected rectangle. This preserves a
    # near-edge building envelope whose apparent fourth side is actually a
    # detached neighboring-plan fragment (CUBI-014).
    frame_band_x = max(source_edge_band, round(width * 0.05))
    frame_band_y = max(source_edge_band, round(height * 0.05))
    top_candidates = [run for run in long_row_runs if run[0] <= frame_band_y]
    bottom_candidates = [run for run in long_row_runs if run[1] >= height - frame_band_y]
    left_candidates = [run for run in long_column_runs if run[0] <= frame_band_x]
    right_candidates = [run for run in long_column_runs if run[1] >= width - frame_band_x]
    corner_radius = max(3, round(minimum_side * 0.008))

    def corner_connected(x: int, y: int) -> bool:
        return bool(
            structural[
                max(0, y - corner_radius) : min(height, y + corner_radius + 1),
                max(0, x - corner_radius) : min(width, x + corner_radius + 1),
            ].any()
        )

    frame_found = False
    for top_run in top_candidates:
        for bottom_run in bottom_candidates:
            for left_run in left_candidates:
                for right_run in right_candidates:
                    x_left = (left_run[0] + left_run[1] - 1) // 2
                    x_right = (right_run[0] + right_run[1] - 1) // 2
                    y_top = (top_run[0] + top_run[1] - 1) // 2
                    y_bottom = (bottom_run[0] + bottom_run[1] - 1) // 2
                    if not all(
                        corner_connected(x, y)
                        for x, y in (
                            (x_left, y_top),
                            (x_right, y_top),
                            (x_left, y_bottom),
                            (x_right, y_bottom),
                        )
                    ):
                        continue
                    page_row_runs.extend((top_run, bottom_run))
                    page_column_runs.extend((left_run, right_run))
                    frame_found = True
                    break
                if frame_found:
                    break
            if frame_found:
                break
        if frame_found:
            break
    page_row_runs = sorted(set(page_row_runs))
    page_column_runs = sorted(set(page_column_runs))
    for top, bottom in page_row_runs:
        cleaned[
            max(0, top - line_padding) : min(height, bottom + line_padding),
            :,
        ] = False
    for left, right in page_column_runs:
        cleaned[
            :,
            max(0, left - line_padding) : min(width, right + line_padding),
        ] = False
    return cleaned, title_block_top


def _merge_nearby_regions(
    boxes: list[tuple[int, int, int, int]],
    *,
    maximum_gap: int,
) -> list[tuple[int, int, int, int]]:
    output = list(boxes)
    changed = True
    while changed:
        changed = False
        for left_index, right_index in combinations(range(len(output)), 2):
            horizontal_gap, vertical_gap = _bbox_gap(
                output[left_index],
                output[right_index],
            )
            # Merge broken parts of the same plan when they overlap strongly on
            # one axis.  Do not bridge a deliberate whitespace gutter between
            # side-by-side or stacked plan views.
            left = output[left_index]
            right = output[right_index]
            overlap_x = max(0, min(left[2], right[2]) - max(left[0], right[0]))
            overlap_y = max(0, min(left[3], right[3]) - max(left[1], right[1]))
            minimum_width = max(1, min(left[2] - left[0], right[2] - right[0]))
            minimum_height = max(1, min(left[3] - left[1], right[3] - right[1]))
            strongly_aligned = (
                overlap_x / minimum_width >= 0.45
                or overlap_y / minimum_height >= 0.45
            )
            if strongly_aligned and max(horizontal_gap, vertical_gap) <= maximum_gap:
                output[left_index] = _bbox_union(left, right)
                output.pop(right_index)
                changed = True
                break
        if changed:
            continue
    return output


def _attach_structural_satellites(
    seed_boxes: list[tuple[int, int, int, int]],
    satellite_boxes: list[tuple[int, int, int, int]],
    *,
    maximum_gap: int,
) -> list[tuple[int, int, int, int]]:
    """Grow a plan through nearby wall fragments without merging plan seeds.

    Raster gaps at doors, balcony thresholds, and low-contrast wall junctions
    often split one building into a large component plus several narrow wall
    components.  The previous detector discarded every component below the
    minimum plan area, permanently cropping part of the building.  Satellites
    may expand their nearest aligned seed, but can never join two already
    credible plan instances.
    """

    regions = list(seed_boxes)
    remaining = list(satellite_boxes)
    changed = True
    while changed and remaining:
        changed = False
        next_remaining = []
        for satellite in remaining:
            matches: list[tuple[int, int, int]] = []
            for index, region in enumerate(regions):
                horizontal_gap, vertical_gap = _bbox_gap(region, satellite)
                overlap_x = max(
                    0,
                    min(region[2], satellite[2]) - max(region[0], satellite[0]),
                )
                overlap_y = max(
                    0,
                    min(region[3], satellite[3]) - max(region[1], satellite[1]),
                )
                satellite_width = max(1, satellite[2] - satellite[0])
                satellite_height = max(1, satellite[3] - satellite[1])
                aligned = (
                    overlap_x / satellite_width >= 0.35
                    or overlap_y / satellite_height >= 0.35
                )
                gap = max(horizontal_gap, vertical_gap)
                if aligned and gap <= maximum_gap:
                    matches.append((gap, index, -(overlap_x + overlap_y)))
            if not matches:
                next_remaining.append(satellite)
                continue
            matches.sort()
            best_gap, best_index, best_overlap = matches[0]
            if len(matches) > 1:
                second_gap, _, second_overlap = matches[1]
                # A fragment equally compatible with two plan seeds is an
                # ambiguous bridge, not permission to join those plans.
                if (best_gap, best_overlap) == (second_gap, second_overlap):
                    next_remaining.append(satellite)
                    continue
            regions[best_index] = _bbox_union(regions[best_index], satellite)
            changed = True
        remaining = next_remaining
    return regions


def discover_plan_regions(
    image: Image.Image,
    *,
    sheet_id: str,
    minimum_sheet_area_fraction: float = 0.006,
    maximum_regions: int = 16,
) -> SheetLayoutAnalysis:
    """Propose every floor-plan view while retaining sheet coordinates.

    Long horizontal and vertical strokes establish structural evidence.  The
    detector groups disconnected wall fragments conservatively and leaves every
    proposal review-required; title blocks and detail views are not silently
    promoted to plan instances. ``maximum_regions`` is an audit threshold, not
    a truncation limit: every discovered region remains available to the
    downstream recognizers and an overflow forces review.
    """

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    minimum_side = min(width, height)
    # Work on stroke edges rather than dark fill.  Raster floor plans often use
    # large filled roof, hatch, shadow, or landscape regions; binary opening on
    # raw darkness would make those fills one giant component spanning the page.
    gray_float = gray.astype(np.float32)
    gradient_x = ndimage.sobel(gray_float, axis=1, mode="nearest")
    gradient_y = ndimage.sobel(gray_float, axis=0, mode="nearest")
    edge = np.hypot(gradient_x, gradient_y) >= 72.0
    edge = ndimage.binary_closing(
        edge,
        structure=np.ones((3, 3), dtype=np.bool_),
    )
    line_span = max(7, round(minimum_side * 0.012))
    horizontal = ndimage.binary_opening(
        edge,
        structure=np.ones((1, line_span), dtype=np.bool_),
    )
    vertical = ndimage.binary_opening(
        edge,
        structure=np.ones((line_span, 1), dtype=np.bool_),
    )
    structural, _title_block_top = _remove_sheet_furniture(horizontal | vertical)
    edge_guard = max(5, round(minimum_side * 0.012))
    edge_structural_pixels = {
        "top": int(structural[:edge_guard, :].sum()),
        "bottom": int(structural[-edge_guard:, :].sum()),
        "left": int(structural[:, :edge_guard].sum()),
        "right": int(structural[:, -edge_guard:].sum()),
    }
    # Page furniture was removed already. Dense structural evidence continuing
    # through a source edge means the drawing itself is cropped; unseen rooms
    # and walls must not be hallucinated into a supposedly complete BIM.
    edge_truncation_threshold = max(96, round(minimum_side * 0.45))
    source_edge_truncation = any(
        count >= edge_truncation_threshold
        for count in edge_structural_pixels.values()
    )
    # Keep this narrow: genuine plans are sometimes placed close to the sheet
    # edge.  Full-length border strokes have already been removed above.
    edge_margin = max(6, round(minimum_side * 0.010))
    structural[:edge_margin, :] = False
    structural[-edge_margin:, :] = False
    structural[:, :edge_margin] = False
    structural[:, -edge_margin:] = False
    # Connect local wall fragments without allowing page-wide furniture or
    # datum lines to bridge neighboring plan views.
    connection_radius = max(3, round(minimum_side * 0.0015))
    connected = ndimage.binary_dilation(
        structural,
        structure=np.ones((3, 3), dtype=np.bool_),
        iterations=connection_radius,
    )
    labels, component_count = ndimage.label(
        connected,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sheet_area = max(1, width * height)
    raw_boxes: list[tuple[int, int, int, int]] = []
    satellite_boxes: list[tuple[int, int, int, int]] = []
    # ``labels == component_index`` rescanned the complete sheet for every
    # fragment (quadratic-looking work on dense construction sheets). SciPy's
    # object slices recover the exact same connected-component extents in one
    # pass, keeping whole-sheet recall without a latency penalty.
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        y_slice, x_slice = slices
        left = int(x_slice.start)
        top = int(y_slice.start)
        right = int(x_slice.stop)
        bottom = int(y_slice.stop)
        component_width = right - left
        component_height = bottom - top
        box_area = (right - left) * (bottom - top)
        structural_pixels = int(structural[top:bottom, left:right].sum())
        # A title block is usually a very wide, shallow ruled table.  Keep this
        # exclusion explicit and conservative; genuine panoramic plans still
        # survive unless they span nearly the full sheet width.
        if component_width / width >= 0.72 and component_height / height <= 0.34:
            continue
        if component_height / height >= 0.72 and component_width / width <= 0.16:
            continue
        if structural_pixels < max(48, round(minimum_side * 0.35)):
            continue
        box = (left, top, right, bottom)
        # A plan seed needs meaningful extent on both sheet axes.  Area alone
        # promoted long dimension strings and ruled side bands as independent
        # plans (for example an 82 px-wide strip beside a 1524 px source).  Keep
        # narrow evidence as an attachable satellite when it is internal, but
        # never force a false multi-plan selection dialog from page furniture.
        edge_strip_guard = max(8, round(minimum_side * 0.04))
        touches_source_edge = (
            left <= edge_strip_guard
            or top <= edge_strip_guard
            or right >= width - edge_strip_guard
            or bottom >= height - edge_strip_guard
        )
        extreme_edge_strip = touches_source_edge and (
            (
                component_width / width < 0.08
                and component_height / height > 0.40
            )
            or (
                component_height / height < 0.08
                and component_width / width > 0.40
            )
        )
        plan_seed_shape = (
            component_width / width >= 0.05
            and component_height / height >= 0.05
            and not extreme_edge_strip
        )
        if box_area / sheet_area < minimum_sheet_area_fraction or not plan_seed_shape:
            # Edge-touching fragments commonly belong to a neighboring plan
            # cropped out of the source. They cannot safely expand this plan.
            satellite_edge_guard = max(8, round(minimum_side * 0.04))
            if (
                left <= satellite_edge_guard
                or top <= satellite_edge_guard
                or right >= width - satellite_edge_guard
                or bottom >= height - satellite_edge_guard
            ):
                continue
            satellite_boxes.append(box)
            continue
        raw_boxes.append(box)
    merged = _merge_nearby_regions(
        raw_boxes,
        maximum_gap=max(4, round(minimum_side * 0.015)),
    )
    merged = _attach_structural_satellites(
        merged,
        satellite_boxes,
        maximum_gap=max(8, round(minimum_side * 0.04)),
    )
    # Satellite growth can reveal that two seed components were merely the
    # disconnected wings of one plan. Re-run the conservative aligned merge;
    # whitespace-separated plan instances still remain independent.
    merged = _merge_nearby_regions(
        merged,
        maximum_gap=max(4, round(minimum_side * 0.015)),
    )
    margin = max(4, round(minimum_side * 0.012))
    expanded = [
        (
            max(0, left - margin),
            max(0, top - margin),
            min(width, right + margin),
            min(height, bottom + margin),
        )
        for left, top, right, bottom in merged
    ]
    expanded.sort(
        key=lambda box: (
            -((box[2] - box[0]) * (box[3] - box[1])),
            box[1],
            box[0],
        )
    )
    discovered_region_count = len(expanded)
    region_overflow = discovered_region_count > maximum_regions
    regions: list[SheetPlanRegion] = []
    assigned = np.zeros_like(structural, dtype=np.bool_)
    for index, (left, top, right, bottom) in enumerate(expanded, start=1):
        box_area = max(1, (right - left) * (bottom - top))
        structural_pixels = int(structural[top:bottom, left:right].sum())
        density = structural_pixels / box_area
        area_fraction = box_area / sheet_area
        size_score = min(1.0, area_fraction / 0.18)
        density_score = min(1.0, density / 0.035)
        confidence = 0.35 + 0.35 * size_score + 0.30 * density_score
        regions.append(
            SheetPlanRegion(
                id=f"{sheet_id}:plan-{index:02d}",
                bbox_px=(left, top, right, bottom),
                crop_to_sheet_transform=(
                    (1.0, 0.0, float(left)),
                    (0.0, 1.0, float(top)),
                    (0.0, 0.0, 1.0),
                ),
                sheet_area_fraction=area_fraction,
                structural_pixel_count=structural_pixels,
                structural_density=density,
                confidence=min(1.0, confidence),
            )
        )
        assigned[top:bottom, left:right] = True
    structural_total = int(structural.sum())
    unassigned = int((structural & ~assigned).sum())
    unassigned_fraction = unassigned / max(1, structural_total)
    return SheetLayoutAnalysis(
        sheet_id=sheet_id,
        image_size_px=(width, height),
        regions=regions,
        multi_plan_candidate=len(regions) > 1,
        unassigned_structural_fraction=unassigned_fraction,
        region_overflow=region_overflow,
        discovered_region_count=discovered_region_count,
        source_edge_truncation=source_edge_truncation,
        source_edge_structural_pixels=edge_structural_pixels,
    ).finalize()


def sheet_layout_manifest(analysis: SheetLayoutAnalysis) -> dict[str, Any]:
    """Return a JSON-ready manifest without discarding coordinate transforms."""

    return analysis.model_dump(mode="json")
