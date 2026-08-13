"""Exhaustive native-resolution proposal ledger for small AEC elements.

The whole-sheet model provides context but can lose tiny symbols when normalized.
This miner visits every residual ink component after removing only page-scale lines.
It never assigns a semantic class and never silently truncates the ledger; the tiny
local ONNX specialist decides whether each candidate is an editable element.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.model.aec_decode import PixelSymbolProposal
from .core.model.cad_evidence import _ndimage, raster_ink


class NativeElementCandidateDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.native-element-candidates.v1"
    image_size: tuple[int, int]
    foreground_threshold: float = Field(ge=0, le=1)
    foreground_pixels: int = Field(ge=0)
    page_line_pixels: int = Field(ge=0)
    residual_component_count: int = Field(ge=0)
    assembled_outline_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(ge=0)
    maximum_candidates: int = Field(ge=1)
    capacity_exceeded: bool = False


def _flat_erode_axis(mask: np.ndarray, *, size: int, axis: int) -> np.ndarray:
    """Exact flat binary erosion without constructing a long 2D kernel."""

    if size <= 1:
        return mask
    ndimage: Any = _ndimage()
    return np.asarray(
        ndimage.minimum_filter1d(
            mask,
            size=size,
            axis=axis,
            mode="constant",
            cval=0,
            origin=0,
        ),
        dtype=np.bool_,
    )


def _flat_dilate_axis(mask: np.ndarray, *, size: int, axis: int) -> np.ndarray:
    """Exact flat binary dilation with SciPy's even-kernel origin contract."""

    if size <= 1:
        return mask
    ndimage: Any = _ndimage()
    return np.asarray(
        ndimage.maximum_filter1d(
            mask,
            size=size,
            axis=axis,
            mode="constant",
            cval=0,
            origin=-1 if size % 2 == 0 else 0,
        ),
        dtype=np.bool_,
    )


def _flat_erode_rect(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = _flat_erode_axis(mask, size=shape[0], axis=0)
    return _flat_erode_axis(output, size=shape[1], axis=1)


def _flat_dilate_rect(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = _flat_dilate_axis(mask, size=shape[0], axis=0)
    return _flat_dilate_axis(output, size=shape[1], axis=1)


def _flat_open_rect(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return _flat_dilate_rect(_flat_erode_rect(mask, shape), shape)


def _flat_close_rect(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return _flat_erode_rect(_flat_dilate_rect(mask, shape), shape)


def _otsu_threshold(values: np.ndarray) -> float:
    clipped = np.uint8(np.clip(values, 0.0, 1.0) * 255)
    histogram = np.bincount(clipped.ravel(), minlength=256).astype(np.float64)
    probability = histogram / max(1.0, histogram.sum())
    cumulative = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256))
    total_mean = cumulative_mean[-1]
    denominator = cumulative * (1.0 - cumulative)
    variance = np.zeros_like(denominator)
    valid = denominator > 1e-12
    variance[valid] = (
        (total_mean * cumulative[valid] - cumulative_mean[valid]) ** 2
        / denominator[valid]
    )
    return float(np.argmax(variance)) / 255.0


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-9, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


def _deduplicate_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    image_size: tuple[int, int] | None = None,
) -> list[tuple[float, float, float, float]]:
    # Keep the proposal hierarchy.  A tight symbol envelope and a slightly
    # larger host-aware envelope can overlap heavily while representing very
    # different geometry targets.  The previous 0.72 suppression discarded
    # the tight box almost every time because boxes were sorted largest-first.
    ordered = sorted(
        boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
    )
    output: list[tuple[float, float, float, float]] = []
    center_buckets: dict[tuple[int, int], list[int]] = {}
    if image_size is None:
        bucket_size = 64.0
    else:
        bucket_size = max(8.0, min(image_size) * 0.012)
    for box in ordered:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        bucket = (int(center_x // bucket_size), int(center_y // bucket_size))
        nearby_indices = [
            index
            for delta_y in (-1, 0, 1)
            for delta_x in (-1, 0, 1)
            for index in center_buckets.get(
                (bucket[0] + delta_x, bucket[1] + delta_y), []
            )
        ]
        if any(_bbox_iou(box, output[index]) >= 0.94 for index in nearby_indices):
            continue
        output_index = len(output)
        output.append(box)
        center_buckets.setdefault(bucket, []).append(output_index)
    return sorted(output, key=lambda box: (box[1], box[0], box[3], box[2]))


def _append_envelope_family(
    boxes: list[tuple[float, float, float, float]],
    *,
    x_start: int,
    y_start: int,
    x_stop: int,
    y_stop: int,
    width: int,
    height: int,
) -> None:
    """Add tight-to-context envelopes for one native connected structure.

    A CAD symbol's ink is not its semantic extent.  Thin line symbols need a
    little expansion, while closed outlines already describe the target well.
    Keeping a small hierarchy lets the local specialist choose instead of
    forcing every candidate through one destructive 45% margin.
    """

    component_width = x_stop - x_start
    component_height = y_stop - y_start
    minimum_side = min(component_width, component_height)
    # Two materially different extents are enough: the exact ink envelope and
    # one semantic-outline hypothesis.  The intermediate 10% box multiplied
    # runtime work but added no independent view on the audited hard sheets.
    margins = (
        0.0,
        max(2.0, minimum_side * 0.24),
        max(3.0, minimum_side * 0.32),
    )
    for margin in margins:
        boxes.append(
            (
                max(0.0, x_start - margin),
                max(0.0, y_start - margin),
                min(float(width), x_stop + margin),
                min(float(height), y_stop + margin),
            )
        )


def _enclosed_outline_boxes(
    foreground: np.ndarray,
    *,
    minimum_pixels: int,
) -> list[tuple[float, float, float, float]]:
    """Recover whole closed symbols from their negative-space interiors.

    Furniture, sanitary fixtures, appliances, and built-ins are commonly
    drawn as hollow outlines.  Removing long lines first cuts those outlines
    into fragments.  Here we independently find bounded white regions in the
    native sheet, then expand through the surrounding ink boundary.  Large
    room interiors are rejected by page-relative size and span limits.
    """

    ndimage: Any = _ndimage()
    height, width = foreground.shape
    page_area = max(1, width * height)
    # Seal one- or two-pixel scan gaps without joining nearby objects.
    sealed_ink = _flat_close_rect(foreground, (3, 3))
    free_labels, _ = ndimage.label(
        ~sealed_ink,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_pixels = np.bincount(free_labels.ravel())
    output: list[tuple[float, float, float, float]] = []
    maximum_hole_area = max(96, round(page_area * 0.008))
    maximum_span_x = max(24, round(width * 0.16))
    maximum_span_y = max(24, round(height * 0.16))
    for component, slices in enumerate(ndimage.find_objects(free_labels), start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        # Exterior/background and open room regions touch the page frame.
        if (
            x_slice.start <= 0
            or y_slice.start <= 0
            or x_slice.stop >= width
            or y_slice.stop >= height
        ):
            continue
        pixels = int(component_pixels[component])
        component_width = x_slice.stop - x_slice.start
        component_height = y_slice.stop - y_slice.start
        if not max(4, minimum_pixels) <= pixels <= maximum_hole_area:
            continue
        if min(component_width, component_height) < 2:
            continue
        if component_width > maximum_span_x or component_height > maximum_span_y:
            continue
        aspect = max(component_width, component_height) / max(
            1, min(component_width, component_height)
        )
        if aspect > 18:
            continue
        # Include the outline itself, not only its white interior.
        boundary = max(2.0, min(component_width, component_height) * 0.08)
        output.append(
            (
                max(0.0, x_slice.start - boundary),
                max(0.0, y_slice.start - boundary),
                min(float(width), x_slice.stop + boundary),
                min(float(height), y_slice.stop + boundary),
            )
        )
    return output


def _bounded_linear_boxes(
    foreground: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """Keep mid-length thin equipment lines that wall removal would erase.

    Shower screens, counters, rails, and narrow built-ins can be geometrically
    indistinguishable from a short wall in a local crop.  The exhaustive ledger
    therefore keeps bounded line hypotheses before page-scale line removal.  It
    deliberately assigns no class; text strokes and dimension lines remain
    explicit hard negatives for the local specialist.
    """

    ndimage: Any = _ndimage()
    height, width = foreground.shape
    output: list[tuple[float, float, float, float]] = []
    for horizontal_orientation in (True, False):
        axis_extent = width if horizontal_orientation else height
        cross_extent = height if horizontal_orientation else width
        maximum_length = max(18, round(axis_extent * 0.30))
        maximum_thickness = max(5, round(cross_extent * 0.018))
        for span_ratio in (0.018, 0.03, 0.05):
            span = max(9, round(axis_extent * span_ratio))
            shape = (1, span) if horizontal_orientation else (span, 1)
            supported = _flat_open_rect(foreground, shape)
            labels, _ = ndimage.label(
                supported,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            for slices in ndimage.find_objects(labels):
                if slices is None:
                    continue
                y_slice, x_slice = slices
                box_width = x_slice.stop - x_slice.start
                box_height = y_slice.stop - y_slice.start
                major = box_width if horizontal_orientation else box_height
                minor = box_height if horizontal_orientation else box_width
                if not span <= major <= maximum_length:
                    continue
                if not 1 <= minor <= maximum_thickness:
                    continue
                # Include the semantic thickness around the detected centerline
                # and seal tiny joins at either endpoint.
                cross_margin = max(2.5, minor * 2.0)
                end_margin = max(1.0, minor * 0.5)
                x_margin = end_margin if horizontal_orientation else cross_margin
                y_margin = cross_margin if horizontal_orientation else end_margin
                output.append(
                    (
                        max(0.0, x_slice.start - x_margin),
                        max(0.0, y_slice.start - y_margin),
                        min(float(width), x_slice.stop + x_margin),
                        min(float(height), y_slice.stop + y_margin),
                    )
                )
    return output


def _paired_outline_boxes(
    foreground: np.ndarray,
    *,
    maximum_hypotheses: int = 1400,
) -> list[tuple[float, float, float, float]]:
    """Assemble whole rectangular objects from separated parallel strokes.

    Beds, tables, casework, and benches are often drawn as four independent
    strokes.  A connected-component miner sees the ink but not the object.
    This source-only stage pairs strongly aligned opposite edges and requires
    support at the remaining boundary.  It adds full-envelope hypotheses; it
    assigns no class and therefore cannot manufacture semantic ground truth.
    """

    if maximum_hypotheses < 1:
        return []
    ndimage: Any = _ndimage()
    height, width = foreground.shape

    def line_segments(horizontal: bool) -> list[tuple[int, int, int, int]]:
        axis = width if horizontal else height
        cross_axis = height if horizontal else width
        minimum_span = max(7, round(axis * 0.006))
        shape = (1, minimum_span) if horizontal else (minimum_span, 1)
        supported = _flat_open_rect(foreground, shape)
        labels, _ = ndimage.label(
            supported,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        maximum_thickness = max(7, round(cross_axis * 0.008))
        output: list[tuple[int, int, int, int]] = []
        for slices in ndimage.find_objects(labels):
            if slices is None:
                continue
            y_slice, x_slice = slices
            box_width = x_slice.stop - x_slice.start
            box_height = y_slice.stop - y_slice.start
            major = box_width if horizontal else box_height
            minor = box_height if horizontal else box_width
            if major >= minimum_span and minor <= maximum_thickness:
                output.append(
                    (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)
                )
        return output

    hypotheses: list[
        tuple[float, tuple[float, float, float, float]]
    ] = []
    maximum_width = width * 0.18
    maximum_height = height * 0.18
    for lines, horizontal in (
        (line_segments(True), True),
        (line_segments(False), False),
    ):
        for left_index, left in enumerate(lines):
            for right in lines[left_index + 1 :]:
                if horizontal:
                    separation = abs(
                        (right[1] + right[3] - left[1] - left[3]) / 2
                    )
                    overlap = max(
                        0, min(left[2], right[2]) - max(left[0], right[0])
                    )
                    shorter = min(left[2] - left[0], right[2] - right[0])
                    if not 5 <= separation <= maximum_height:
                        continue
                else:
                    separation = abs(
                        (right[0] + right[2] - left[0] - left[2]) / 2
                    )
                    overlap = max(
                        0, min(left[3], right[3]) - max(left[1], right[1])
                    )
                    shorter = min(left[3] - left[1], right[3] - right[1])
                    if not 5 <= separation <= maximum_width:
                        continue
                alignment = overlap / max(1, shorter)
                if alignment < 0.90:
                    continue
                x_start = min(left[0], right[0])
                y_start = min(left[1], right[1])
                x_stop = max(left[2], right[2])
                y_stop = max(left[3], right[3])
                box_width = x_stop - x_start
                box_height = y_stop - y_start
                if box_width > maximum_width or box_height > maximum_height:
                    continue
                boundary_band = max(2, round(min(box_width, box_height) * 0.08))
                if horizontal:
                    first_support = foreground[
                        y_start:y_stop,
                        max(0, x_start - boundary_band) : min(
                            width, x_start + boundary_band + 1
                        ),
                    ].mean()
                    second_support = foreground[
                        y_start:y_stop,
                        max(0, x_stop - boundary_band) : min(
                            width, x_stop + boundary_band + 1
                        ),
                    ].mean()
                else:
                    first_support = foreground[
                        max(0, y_start - boundary_band) : min(
                            height, y_start + boundary_band + 1
                        ),
                        x_start:x_stop,
                    ].mean()
                    second_support = foreground[
                        max(0, y_stop - boundary_band) : min(
                            height, y_stop + boundary_band + 1
                        ),
                        x_start:x_stop,
                    ].mean()
                boundary_support = float((first_support + second_support) / 2)
                if boundary_support < 0.28:
                    continue
                score = boundary_support + alignment * 0.25
                hypotheses.append(
                    (
                        score,
                        (
                            float(x_start),
                            float(y_start),
                            float(x_stop),
                            float(y_stop),
                        ),
                    )
                )
    # This is a bounded semantic-envelope hypothesis family, not the exhaustive
    # residual ledger.  Preserve the strongest source-supported closures and
    # let the local objectness head reject text/table false positives.
    hypotheses.sort(
        key=lambda item: (
            -item[0],
            (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
            item[1],
        )
    )
    return [box for _, box in hypotheses[:maximum_hypotheses]]


def candidate_ledger_recall(
    candidates: list[PixelSymbolProposal],
    target_boxes: list[tuple[float, float, float, float]],
    *,
    minimum_intersection_over_target: float = 0.35,
) -> dict[str, Any]:
    """Measure the proposal-stage recall ceiling before semantic classification.

    Intersection-over-target is intentional: a candidate may include context around
    a tiny symbol and still preserve all evidence needed by the local classifier.
    This diagnostic prevents a classifier score from hiding objects that were never
    proposed at native resolution.
    """

    if not 0 < minimum_intersection_over_target <= 1:
        raise ValueError("minimum_intersection_over_target must be in (0, 1]")
    matched = 0
    missed_indices: list[int] = []
    for target_index, target in enumerate(target_boxes):
        target_area = max(1e-9, (target[2] - target[0]) * (target[3] - target[1]))
        best = 0.0
        for candidate in candidates:
            left = max(target[0], candidate.bbox_px[0])
            top = max(target[1], candidate.bbox_px[1])
            right = min(target[2], candidate.bbox_px[2])
            bottom = min(target[3], candidate.bbox_px[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            best = max(best, intersection / target_area)
        if best >= minimum_intersection_over_target:
            matched += 1
        else:
            missed_indices.append(target_index)
    total = len(target_boxes)
    return {
        "target_count": total,
        "matched_target_count": matched,
        "missed_target_count": total - matched,
        "recall": 1.0 if total == 0 else matched / total,
        "missed_target_indices": missed_indices,
        "minimum_intersection_over_target": minimum_intersection_over_target,
    }


def candidate_ledger_iou_recall(
    candidates: list[PixelSymbolProposal],
    target_boxes: list[tuple[float, float, float, float]],
    *,
    minimum_iou: float = 0.5,
) -> dict[str, Any]:
    """Measure whether a proposal actually matches an object's full extent.

    This is deliberately stricter than :func:`candidate_ledger_recall`.
    Intersection-over-target answers whether any crop contains the evidence;
    IoU answers whether the proposal geometry itself is usable.  Both are
    reported so a large context crop can never masquerade as a correct object.
    """

    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    matched = 0
    missed_indices: list[int] = []
    best_ious: list[float] = []
    for target_index, target in enumerate(target_boxes):
        best = max(
            (_bbox_iou(target, candidate.bbox_px) for candidate in candidates),
            default=0.0,
        )
        best_ious.append(best)
        if best >= minimum_iou:
            matched += 1
        else:
            missed_indices.append(target_index)
    total = len(target_boxes)
    return {
        "target_count": total,
        "matched_target_count": matched,
        "missed_target_count": total - matched,
        "recall": 1.0 if total == 0 else matched / total,
        "missed_target_indices": missed_indices,
        "minimum_iou": minimum_iou,
        "median_best_iou": 1.0 if not best_ious else float(np.median(best_ious)),
    }


def mine_native_element_candidates(
    image: Image.Image,
    *,
    source_ref_ids: list[str],
    maximum_candidates: int = 32768,
) -> tuple[list[PixelSymbolProposal], NativeElementCandidateDiagnostics]:
    """Return an auditable, exhaustive ledger of native residual components."""

    if not source_ref_ids:
        raise ValueError("source_ref_ids cannot be empty")
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    ink = raster_ink(image)
    height, width = ink.shape
    threshold = min(0.72, max(0.20, _otsu_threshold(ink)))
    foreground = ink >= threshold
    ndimage: Any = _ndimage()
    horizontal_span = max(21, round(width * 0.12))
    vertical_span = max(21, round(height * 0.12))
    horizontal = _flat_open_rect(foreground, (1, horizontal_span))
    vertical = _flat_open_rect(foreground, (vertical_span, 1))
    # Preserve SciPy's default 2D cross footprint here.  This is intentionally
    # not the rectangular helper used below: changing the footprint would
    # change which native marks survive the page-line subtraction.
    page_lines = ndimage.binary_dilation(horizontal | vertical, iterations=1)
    page_area = max(1, width * height)
    minimum_pixels = max(3, round(page_area * 0.000002))
    maximum_pixels = max(64, round(page_area * 0.018))
    boxes: list[tuple[float, float, float, float]] = []
    component_count = 0
    # Closed-outline evidence is deliberately independent from wall removal.
    # It gives the ledger whole-object hypotheses for symbols that the
    # residual-component path necessarily fragments.
    boxes.extend(
        _enclosed_outline_boxes(
            foreground,
            minimum_pixels=minimum_pixels,
        )
    )
    boxes.extend(_bounded_linear_boxes(foreground))
    # Use several conservative line-removal scales. Long wall-attached cabinets
    # disappear when only one page-line span is used, while compact plumbing and
    # electrical symbols benefit from stronger removal. The union keeps both.
    # Three separated line-removal scales cover short, medium and page-scale
    # structure.  Four near-duplicate scales multiplied the same strokes into
    # thousands of candidates without adding a new semantic view.
    for span_ratio in (0.05, 0.12, 0.22):
        scale_horizontal = _flat_open_rect(
            foreground,
            (1, max(11, round(width * span_ratio))),
        )
        scale_vertical = _flat_open_rect(
            foreground,
            (max(11, round(height * span_ratio)), 1),
        )
        scale_lines = ndimage.binary_dilation(
            scale_horizontal | scale_vertical,
            iterations=1,
        )
        residual = _flat_close_rect(foreground & ~scale_lines, (3, 3))
        grouping_span = max(5, round(min(width, height) * 0.018))
        # Symbols are usually made of several disconnected strokes. Horizontal
        # and vertical closings preserve cabinet runs, but alone they leave a
        # toilet, appliance, or chair as many unrelated candidates. Two bounded
        # isotropic scales add whole-object hypotheses without replacing the
        # exhaustive component ledger.
        compact_grouping_spans = sorted(
            {
                max(3, round(min(width, height) * 0.004)),
                max(7, round(min(width, height) * 0.009)),
            }
        )
        component_masks = (
            residual,
            _flat_close_rect(residual, (grouping_span, 1)),
            _flat_close_rect(residual, (1, grouping_span)),
            *(
                _flat_close_rect(residual, (span, span))
                for span in compact_grouping_spans
            ),
        )
        for component_mask in component_masks:
            labels, scale_component_count = ndimage.label(
                component_mask,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            component_count += int(scale_component_count)
            # Count every component once over the full native mask.  Building a
            # temporary boolean crop for every label made dense, real sheets
            # spend most of their time repeatedly scanning the same pixels.
            # bincount preserves the exact candidate set while removing that
            # quadratic-looking inner-loop work.
            component_pixels = np.bincount(labels.ravel())
            for component, slices in enumerate(ndimage.find_objects(labels), start=1):
                if slices is None:
                    continue
                pixels = int(component_pixels[component])
                if not minimum_pixels <= pixels <= maximum_pixels * 2:
                    continue
                y_slice, x_slice = slices
                component_width = x_slice.stop - x_slice.start
                component_height = y_slice.stop - y_slice.start
                if min(component_width, component_height) < 2:
                    continue
                aspect = max(component_width, component_height) / min(
                    component_width, component_height
                )
                if aspect > 32:
                    continue
                _append_envelope_family(
                    boxes,
                    x_start=x_slice.start,
                    y_start=y_slice.start,
                    x_stop=x_slice.stop,
                    y_stop=y_slice.stop,
                    width=width,
                    height=height,
                )
    base_boxes = _deduplicate_boxes(boxes, image_size=(width, height))
    # The limit is a budget for optional assembly hypotheses, never a recall
    # gate.  Earlier versions raised here and therefore made every mark after a
    # complexity-dependent boundary invisible to the product.  Preserve the
    # complete native ledger and expose the over-budget state in diagnostics;
    # batched inference and the spatial proposal graph keep this bounded in
    # memory without deleting source evidence.
    capacity_exceeded = len(base_boxes) > maximum_candidates
    # Pair edges from a slightly stricter ink map than the exhaustive ledger.
    # These are optional full-object hypotheses, so they may use only the
    # capacity left after every native residual component has been preserved.
    outline_budget = max(0, maximum_candidates - len(base_boxes))
    assembled_outlines = _paired_outline_boxes(
        ink >= max(threshold, 0.35),
        maximum_hypotheses=outline_budget,
    )
    boxes = _deduplicate_boxes(
        base_boxes + assembled_outlines,
        image_size=(width, height),
    )
    proposals = [
        PixelSymbolProposal(
            id=f"native-candidate:{index:05d}",
            symbol_class="unknown",
            center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            bbox_px=box,
            confidence=0.0,
            uncertainty=1.0,
            source_ref_ids=source_ref_ids,
            model_version="dajoong-native-candidate-ledger-v1",
            review_required=True,
        )
        for index, box in enumerate(boxes)
    ]
    return proposals, NativeElementCandidateDiagnostics(
        image_size=(width, height),
        foreground_threshold=threshold,
        foreground_pixels=int(foreground.sum()),
        page_line_pixels=int(page_lines.sum()),
        residual_component_count=int(component_count),
        assembled_outline_count=len(assembled_outlines),
        candidate_count=len(proposals),
        maximum_candidates=maximum_candidates,
        capacity_exceeded=capacity_exceeded,
    )
