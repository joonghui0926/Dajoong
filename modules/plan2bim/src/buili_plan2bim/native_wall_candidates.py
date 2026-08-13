"""Native-resolution wall-evidence ledger.

The global program supplies building context, but normalization can erase a thin
or short wall before decoding.  This module independently enumerates long native
ink runs.  It does not name them as walls; unmatched runs remain auditable review
candidates so a missed global proposal can never disappear silently.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.model.cad_evidence import _ndimage, raster_ink
from .core.model.fourier_wall import FourierWallPrior, detect_fourier_wall_prior


class NativeWallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = Field(gt=0)
    orientation: Literal["horizontal", "vertical"]
    evidence_mode: Literal["paired_edges", "single_band"]
    ink_support: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class NativeWallCandidateDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.native-wall-candidates.v2-adaptive-band-scale"
    image_size: tuple[int, int]
    foreground_threshold: float = Field(ge=0, le=1)
    foreground_pixels: int = Field(ge=0)
    raw_run_count: int = Field(ge=0)
    paired_run_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    maximum_candidates: int = Field(ge=1)
    adaptive_maximum_thickness_px: int = Field(ge=1)
    adaptive_maximum_edge_separation_px: float = Field(gt=0)
    fourier_prior: FourierWallPrior


class UnresolvedNativeWallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    reason: str = "no_global_wall_explains_native_run"


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
    variance[valid] = (total_mean * cumulative[valid] - cumulative_mean[valid]) ** 2 / denominator[
        valid
    ]
    return float(np.argmax(variance)) / 255.0


def _length(item: NativeWallCandidate) -> float:
    return math.dist(item.start_px, item.end_px)


def _overlap_fraction(
    left: NativeWallCandidate,
    right: NativeWallCandidate,
) -> float:
    axis = 0 if left.orientation == "horizontal" else 1
    left_extent = sorted((left.start_px[axis], left.end_px[axis]))
    right_extent = sorted((right.start_px[axis], right.end_px[axis]))
    overlap = max(0.0, min(left_extent[1], right_extent[1]) - max(left_extent[0], right_extent[0]))
    return overlap / max(1.0, min(_length(left), _length(right)))


def _overlap_length(
    left: NativeWallCandidate,
    right: NativeWallCandidate,
) -> float:
    if left.orientation != right.orientation:
        return 0.0
    axis = 0 if left.orientation == "horizontal" else 1
    left_extent = sorted((left.start_px[axis], left.end_px[axis]))
    right_extent = sorted((right.start_px[axis], right.end_px[axis]))
    return max(
        0.0,
        min(left_extent[1], right_extent[1]) - max(left_extent[0], right_extent[0]),
    )


def _perpendicular_distance(
    left: NativeWallCandidate,
    right: NativeWallCandidate,
) -> float:
    axis = 1 if left.orientation == "horizontal" else 0
    left_coordinate = (left.start_px[axis] + left.end_px[axis]) / 2
    right_coordinate = (right.start_px[axis] + right.end_px[axis]) / 2
    return abs(left_coordinate - right_coordinate)


def _merge_duplicate_runs(
    candidates: list[NativeWallCandidate],
) -> list[NativeWallCandidate]:
    output: list[NativeWallCandidate] = []
    for candidate in sorted(candidates, key=_length, reverse=True):
        duplicate_index = next(
            (
                index
                for index, item in enumerate(output)
                if item.orientation == candidate.orientation
                and min(_length(item), _length(candidate))
                / max(1.0, _length(item), _length(candidate))
                >= 0.65
                and _overlap_fraction(item, candidate) >= 0.78
                and _perpendicular_distance(item, candidate)
                <= max(2.5, item.thickness_px, candidate.thickness_px)
            ),
            None,
        )
        if duplicate_index is None:
            output.append(candidate)
            continue
        duplicate = output[duplicate_index]
        # A one-pixel dimension/border line is often one pixel longer than the
        # physical wall band beside it. Length-only deduplication therefore
        # retained the annotation and threw away the wall. Prefer the source
        # candidate with stronger ink support/confidence and a credible band
        # thickness when their extents describe the same physical run.
        existing_score = (
            duplicate.confidence
            + min(0.08, duplicate.thickness_px * 0.004)
            + (0.02 if duplicate.evidence_mode == "paired_edges" else 0.0)
        )
        candidate_score = (
            candidate.confidence
            + min(0.08, candidate.thickness_px * 0.004)
            + (0.02 if candidate.evidence_mode == "paired_edges" else 0.0)
        )
        if candidate_score > existing_score:
            output[duplicate_index] = candidate
    return sorted(output, key=lambda item: (item.start_px[1], item.start_px[0], item.id))


def _extract_runs(
    foreground: np.ndarray,
    ink: np.ndarray,
    *,
    orientation: Literal["horizontal", "vertical"],
    span: int,
    minimum_length: int,
    maximum_thickness: int,
) -> list[NativeWallCandidate]:
    """Extract oriented bands without merging an entire wall graph at corners.

    Connected-component labeling turns a rectangular building envelope into a
    single component. Its bounding box then looks too thick and was discarded.
    Scanline tracking keeps joined walls as four independent physical bands.
    """

    ndimage = _ndimage()
    structure = (
        np.ones((1, span), dtype=np.bool_)
        if orientation == "horizontal"
        else np.ones((span, 1), dtype=np.bool_)
    )
    opened = ndimage.binary_opening(foreground, structure=structure)
    scan_image = opened if orientation == "horizontal" else opened.T
    groups: list[dict[str, object]] = []
    active: list[int] = []
    for scan_index, scanline in enumerate(scan_image):
        transitions = np.diff(np.pad(scanline.astype(np.int8), (1, 1), constant_values=0))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        intervals = [
            (int(lower), int(upper - 1))
            for lower, upper in zip(starts, stops, strict=True)
            if upper - lower >= minimum_length
        ]
        available = {
            group_index
            for group_index in active
            if scan_index - int(groups[group_index]["last_scan"]) == 1
        }
        next_active: list[int] = []
        for lower, upper in intervals:
            best: tuple[float, int] | None = None
            for group_index in available:
                group = groups[group_index]
                prior_lower = float(np.median(group["lowers"]))
                prior_upper = float(np.median(group["uppers"]))
                prior_length = prior_upper - prior_lower + 1.0
                current_length = upper - lower + 1.0
                length_ratio = min(prior_length, current_length) / max(prior_length, current_length)
                if length_ratio < 0.55:
                    # At an L/T junction a long horizontal wall becomes the
                    # width of a vertical wall on the next row. Treat that as
                    # a new band instead of growing one giant component.
                    continue
                overlap = max(0.0, min(upper, prior_upper) - max(lower, prior_lower))
                overlap_fraction = overlap / max(
                    1.0,
                    min(upper - lower, prior_upper - prior_lower),
                )
                if overlap_fraction < 0.62:
                    continue
                endpoint_shift = abs(lower - prior_lower) + abs(upper - prior_upper)
                score = overlap_fraction - endpoint_shift / max(1.0, upper - lower) * 0.08
                if best is None or score > best[0]:
                    best = (score, group_index)
            if best is None:
                groups.append(
                    {
                        "first_scan": scan_index,
                        "last_scan": scan_index,
                        "lowers": [lower],
                        "uppers": [upper],
                    }
                )
                group_index = len(groups) - 1
            else:
                group_index = best[1]
                group = groups[group_index]
                group["last_scan"] = scan_index
                group["lowers"].append(lower)
                group["uppers"].append(upper)
                available.remove(group_index)
            next_active.append(group_index)
        active = next_active

    runs: list[NativeWallCandidate] = []
    for component, group in enumerate(groups, start=1):
        first_scan = int(group["first_scan"])
        last_scan = int(group["last_scan"])
        lower = int(round(float(np.median(group["lowers"]))))
        upper = int(round(float(np.median(group["uppers"]))))
        length = upper - lower + 1
        thickness = last_scan - first_scan + 1
        if length < minimum_length or not 1 <= thickness <= maximum_thickness:
            continue
        if orientation == "horizontal":
            region = opened[first_scan : last_scan + 1, lower : upper + 1]
            region_ink = ink[first_scan : last_scan + 1, lower : upper + 1]
        else:
            region = opened[lower : upper + 1, first_scan : last_scan + 1]
            region_ink = ink[lower : upper + 1, first_scan : last_scan + 1]
        support = float(region_ink[region].mean()) if region.any() else 0.0
        if orientation == "horizontal":
            coordinate = (first_scan + last_scan) / 2
            start = (float(lower), float(coordinate))
            end = (float(upper), float(coordinate))
        else:
            coordinate = (first_scan + last_scan) / 2
            start = (float(coordinate), float(lower))
            end = (float(coordinate), float(upper))
        runs.append(
            NativeWallCandidate(
                id=f"native-wall-run:{orientation}:{span}:{component}",
                start_px=start,
                end_px=end,
                thickness_px=float(thickness),
                orientation=orientation,
                evidence_mode="single_band",
                ink_support=min(1.0, support),
                confidence=min(0.88, 0.42 + 0.36 * support + 0.10 * min(1.0, length / (span * 3))),
            )
        )
    return runs


def _pair_edges(
    runs: list[NativeWallCandidate],
    *,
    maximum_separation: float,
) -> tuple[list[NativeWallCandidate], set[str]]:
    paired: list[NativeWallCandidate] = []
    consumed: set[str] = set()
    for index, left in enumerate(runs):
        best: tuple[float, NativeWallCandidate] | None = None
        for right in runs[index + 1 :]:
            if right.orientation != left.orientation:
                continue
            length_ratio = min(_length(left), _length(right)) / max(
                1.0, _length(left), _length(right)
            )
            if length_ratio < 0.72:
                # A short cabinet/fixture edge running beside a long wall is
                # not the second face of that wall.
                continue
            separation = _perpendicular_distance(left, right)
            if not 2.0 <= separation <= maximum_separation:
                continue
            overlap = _overlap_fraction(left, right)
            if overlap < 0.62:
                continue
            score = overlap - separation / max(1.0, maximum_separation) * 0.08
            if best is None or score > best[0]:
                best = score, right
        if best is None:
            continue
        right = best[1]
        axis = 0 if left.orientation == "horizontal" else 1
        lower = max(
            min(left.start_px[axis], left.end_px[axis]),
            min(right.start_px[axis], right.end_px[axis]),
        )
        upper = min(
            max(left.start_px[axis], left.end_px[axis]),
            max(right.start_px[axis], right.end_px[axis]),
        )
        if upper <= lower:
            continue
        if left.orientation == "horizontal":
            coordinate = (
                left.start_px[1] + left.end_px[1] + right.start_px[1] + right.end_px[1]
            ) / 4
            start, end = (lower, coordinate), (upper, coordinate)
        else:
            coordinate = (
                left.start_px[0] + left.end_px[0] + right.start_px[0] + right.end_px[0]
            ) / 4
            start, end = (coordinate, lower), (coordinate, upper)
        confidence = min(
            0.98, 0.58 + 0.20 * best[0] + 0.20 * min(left.ink_support, right.ink_support)
        )
        paired.append(
            NativeWallCandidate(
                id=f"native-wall-pair:{len(paired):05d}",
                start_px=start,
                end_px=end,
                thickness_px=_perpendicular_distance(left, right),
                orientation=left.orientation,
                evidence_mode="paired_edges",
                ink_support=min(left.ink_support, right.ink_support),
                confidence=confidence,
            )
        )
        consumed.update((left.id, right.id))
    return paired, consumed


def mine_native_wall_candidates(
    image: Image.Image,
    *,
    maximum_candidates: int = 16_384,
) -> tuple[list[NativeWallCandidate], NativeWallCandidateDiagnostics]:
    """Enumerate native line evidence without silently converting it to walls."""

    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive")
    ink = raster_ink(image)
    height, width = ink.shape
    minimum_side = min(width, height)
    threshold = min(0.78, max(0.22, _otsu_threshold(ink)))
    foreground = ink >= threshold
    minimum_length = max(12, round(minimum_side * 0.016))
    # The previous fixed 3.2% cap silently excluded the dominant wall bands in
    # compact or heavily scanned plans.  Estimate the drafting scale from the
    # source itself: twice the foreground distance is the local stroke width.
    # A high percentile retains structural bands while the global wall graph
    # still decides whether a band is allowed to become BIM topology.
    distance = _ndimage().distance_transform_edt(foreground)
    positive_distance = distance[foreground]
    estimated_thickness = (
        2.0 * float(np.percentile(positive_distance, 99.5)) + 1.0 if positive_distance.size else 1.0
    )
    maximum_thickness = max(
        8,
        round(minimum_side * 0.032),
        min(round(minimum_side * 0.12), math.ceil(estimated_thickness)),
    )
    spans = sorted(
        {
            minimum_length,
            max(minimum_length, round(minimum_side * 0.032)),
            max(minimum_length, round(minimum_side * 0.065)),
        }
    )
    raw: list[NativeWallCandidate] = []
    for span in spans:
        for orientation in ("horizontal", "vertical"):
            raw.extend(
                _extract_runs(
                    foreground,
                    ink,
                    orientation=orientation,
                    span=span,
                    minimum_length=minimum_length,
                    maximum_thickness=maximum_thickness,
                )
            )
    runs = _merge_duplicate_runs(raw)
    maximum_edge_separation = max(
        8.0,
        minimum_side * 0.022,
        maximum_thickness * 1.05,
    )
    paired, consumed = _pair_edges(
        runs,
        maximum_separation=maximum_edge_separation,
    )
    unpaired = [
        item.model_copy(update={"id": f"native-wall-single:{index:05d}"})
        for index, item in enumerate(runs)
        if item.id not in consumed and _length(item) >= minimum_length * 1.8
    ]
    candidates = _merge_duplicate_runs([*paired, *unpaired])
    if len(candidates) > maximum_candidates:
        raise RuntimeError(
            "native wall candidate ledger exceeds its audited capacity: "
            f"{len(candidates)} > {maximum_candidates}"
        )
    return candidates, NativeWallCandidateDiagnostics(
        image_size=(width, height),
        foreground_threshold=threshold,
        foreground_pixels=int(foreground.sum()),
        raw_run_count=len(raw),
        paired_run_count=len(paired),
        candidate_count=len(candidates),
        maximum_candidates=maximum_candidates,
        adaptive_maximum_thickness_px=maximum_thickness,
        adaptive_maximum_edge_separation_px=maximum_edge_separation,
        fourier_prior=detect_fourier_wall_prior(ink),
    )


def refine_context_walls_with_native_bands(
    context_walls: list[object],
    candidates: list[NativeWallCandidate],
) -> tuple[list[object], int]:
    """Move learned wall geometry onto source-native structural bands.

    The learned whole-sheet pass owns topology.  Native evidence owns the
    centerline, extent and thickness only when it covers most of an existing
    wall and occupies the same physical band.  This ordering prevents a thin
    annotation or furniture edge from replacing a real wall while correcting
    the downsampled model's systematic one-edge bias.
    """

    refined: list[object] = []
    refinement_count = 0
    for wall in context_walls:
        start = wall.start_px
        end = wall.end_px
        delta_x = abs(float(end[0]) - float(start[0]))
        delta_y = abs(float(end[1]) - float(start[1]))
        orientation = "horizontal" if delta_x >= delta_y else "vertical"
        wall_length = _length_like(wall)
        wall_thickness = float(getattr(wall, "thickness_px", 0.0) or 4.0)
        best: tuple[float, NativeWallCandidate] | None = None
        for candidate in candidates:
            if candidate.orientation != orientation:
                continue
            candidate_length = _length(candidate)
            if not 0.65 <= candidate_length / max(1.0, wall_length) <= 1.55:
                continue
            if candidate.thickness_px < max(2.0, wall_thickness * 0.25):
                continue
            if orientation == "horizontal":
                wall_extent = sorted((float(start[0]), float(end[0])))
                native_extent = sorted((candidate.start_px[0], candidate.end_px[0]))
            else:
                wall_extent = sorted((float(start[1]), float(end[1])))
                native_extent = sorted((candidate.start_px[1], candidate.end_px[1]))
            overlap = max(
                0.0,
                min(wall_extent[1], native_extent[1]) - max(wall_extent[0], native_extent[0]),
            )
            wall_coverage = overlap / max(1.0, wall_length)
            if wall_coverage < 0.62:
                continue
            perpendicular = _perpendicular_distance(candidate, _wall_candidate(wall))
            tolerance = max(
                5.0,
                candidate.thickness_px * 0.65,
                wall_thickness * 0.65,
            )
            if perpendicular > tolerance:
                continue
            score = (
                wall_coverage * 0.55
                + min(1.0, candidate_length / max(1.0, wall_length)) * 0.15
                + candidate.confidence * 0.20
                + (1.0 - perpendicular / max(tolerance, 1e-6)) * 0.10
            )
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            refined.append(wall)
            continue
        candidate = best[1]
        refined.append(
            wall.model_copy(
                update={
                    "start_px": candidate.start_px,
                    "end_px": candidate.end_px,
                    "thickness_px": candidate.thickness_px,
                    "review_required": bool(getattr(wall, "review_required", False))
                    or candidate.evidence_mode != "paired_edges",
                    "model_version": (f"{wall.model_version}+native-band-centerline-v1"),
                }
            )
        )
        refinement_count += 1
    return refined, refinement_count


def _wall_candidate(wall: object) -> NativeWallCandidate:
    """Adapt a wall to the small geometry interface used above."""

    start = tuple(float(value) for value in wall.start_px)
    end = tuple(float(value) for value in wall.end_px)
    orientation = "horizontal" if abs(end[0] - start[0]) >= abs(end[1] - start[1]) else "vertical"
    return NativeWallCandidate(
        id="context-wall",
        start_px=start,
        end_px=end,
        thickness_px=float(getattr(wall, "thickness_px", 0.0) or 4.0),
        orientation=orientation,
        evidence_mode="single_band",
        ink_support=1.0,
        confidence=float(getattr(wall, "confidence", 1.0)),
    )


def translate_native_wall_candidate(
    candidate: NativeWallCandidate,
    *,
    offset_px: tuple[float, float],
) -> NativeWallCandidate:
    offset_x, offset_y = offset_px
    return candidate.model_copy(
        update={
            "start_px": (candidate.start_px[0] + offset_x, candidate.start_px[1] + offset_y),
            "end_px": (candidate.end_px[0] + offset_x, candidate.end_px[1] + offset_y),
        }
    )


def unresolved_native_wall_candidates(
    candidates: list[NativeWallCandidate],
    walls: list[object],
    *,
    minimum_overlap_fraction: float = 0.55,
) -> list[UnresolvedNativeWallCandidate]:
    """Return native runs that no emitted wall explains in source coordinates."""

    unresolved: list[UnresolvedNativeWallCandidate] = []
    for candidate in candidates:
        explained = False
        candidate_length = _length(candidate)
        for wall in walls:
            start = wall.start_px
            end = wall.end_px
            delta_x = abs(float(end[0]) - float(start[0]))
            delta_y = abs(float(end[1]) - float(start[1]))
            orientation = "horizontal" if delta_x >= delta_y else "vertical"
            if orientation != candidate.orientation:
                continue
            if orientation == "horizontal":
                perpendicular = abs(
                    (candidate.start_px[1] + candidate.end_px[1]) / 2
                    - (float(start[1]) + float(end[1])) / 2
                )
                candidate_extent = sorted((candidate.start_px[0], candidate.end_px[0]))
                wall_extent = sorted((float(start[0]), float(end[0])))
            else:
                perpendicular = abs(
                    (candidate.start_px[0] + candidate.end_px[0]) / 2
                    - (float(start[0]) + float(end[0])) / 2
                )
                candidate_extent = sorted((candidate.start_px[1], candidate.end_px[1]))
                wall_extent = sorted((float(start[1]), float(end[1])))
            tolerance = max(
                5.0,
                candidate.thickness_px * 1.5,
                float(getattr(wall, "thickness_px", 0.0) or 0.0) * 1.5,
            )
            overlap = max(
                0.0,
                min(candidate_extent[1], wall_extent[1]) - max(candidate_extent[0], wall_extent[0]),
            )
            if (
                perpendicular <= tolerance
                and overlap / max(1.0, candidate_length) >= minimum_overlap_fraction
            ):
                explained = True
                break
        if not explained:
            unresolved.append(
                UnresolvedNativeWallCandidate(
                    candidate_id=candidate.id,
                    start_px=candidate.start_px,
                    end_px=candidate.end_px,
                    thickness_px=candidate.thickness_px,
                    confidence=candidate.confidence,
                )
            )
    return unresolved


def _length_like(wall: object) -> float:
    return math.dist(wall.start_px, wall.end_px)


def _point_to_line_distance(point: tuple[float, float], wall: object) -> float:
    start = wall.start_px
    end = wall.end_px
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 1e-9:
        return math.dist(point, start)
    fraction = (
        (point[0] - float(start[0])) * delta_x + (point[1] - float(start[1])) * delta_y
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    nearest = (
        float(start[0]) + fraction * delta_x,
        float(start[1]) + fraction * delta_y,
    )
    return math.dist(point, nearest)


def _native_lines_connect(
    left: NativeWallCandidate,
    right: NativeWallCandidate,
    *,
    maximum_structural_gap: float = 0.0,
) -> bool:
    """Return whether two source-native bands form one structural junction."""

    tolerance = max(6.0, left.thickness_px, right.thickness_px) * 1.25
    if left.orientation != right.orientation:
        horizontal = left if left.orientation == "horizontal" else right
        vertical = right if left.orientation == "horizontal" else left
        horizontal_x = sorted((horizontal.start_px[0], horizontal.end_px[0]))
        vertical_y = sorted((vertical.start_px[1], vertical.end_px[1]))
        crossing_x = (vertical.start_px[0] + vertical.end_px[0]) / 2
        crossing_y = (horizontal.start_px[1] + horizontal.end_px[1]) / 2
        directly_connected = (
            horizontal_x[0] - tolerance <= crossing_x <= horizontal_x[1] + tolerance
            and vertical_y[0] - tolerance <= crossing_y <= vertical_y[1] + tolerance
        )
        if directly_connected or maximum_structural_gap <= 0:
            return directly_connected
        horizontal_gap = max(
            0.0,
            horizontal_x[0] - crossing_x,
            crossing_x - horizontal_x[1],
        )
        vertical_gap = max(
            0.0,
            vertical_y[0] - crossing_y,
            crossing_y - vertical_y[1],
        )
        # A door/opening can interrupt one leg of a perpendicular wall
        # junction.  Allow that single-axis gap, but never join two lines that
        # are both spatially detached.  Component-scale gates below remain the
        # final authority, so a local furniture arrangement cannot seed walls.
        return (
            min(horizontal_gap, vertical_gap) <= tolerance
            and max(horizontal_gap, vertical_gap) <= maximum_structural_gap
        )
    if _perpendicular_distance(left, right) > tolerance:
        return False
    axis = 0 if left.orientation == "horizontal" else 1
    left_extent = sorted((left.start_px[axis], left.end_px[axis]))
    right_extent = sorted((right.start_px[axis], right.end_px[axis]))
    gap = max(
        0.0,
        max(left_extent[0], right_extent[0]) - min(left_extent[1], right_extent[1]),
    )
    return gap <= tolerance


def _bootstrap_native_wall_network(
    candidates: list[NativeWallCandidate],
    *,
    source_size: tuple[int, int] | None,
    minimum_confidence: float,
) -> list[NativeWallCandidate]:
    """Find a large, connected paired-edge network when the model emits no wall.

    This is intentionally stricter than candidate mining. Paired physical bands
    participate directly. A single filled band may participate only when it is
    page-scale, thick and strongly supported; that covers scans where a heavy
    wall prints as one solid stripe instead of two resolved faces. The network
    must still contain both axes and span a meaningful part of the full drawing,
    so isolated furniture, dimensions and compact rectangles cannot seed it.
    """

    if source_size is None:
        maximum_x = max(
            (max(item.start_px[0], item.end_px[0]) for item in candidates),
            default=0.0,
        )
        maximum_y = max(
            (max(item.start_px[1], item.end_px[1]) for item in candidates),
            default=0.0,
        )
        source_size = (max(1, math.ceil(maximum_x)), max(1, math.ceil(maximum_y)))
    minimum_side = float(min(source_size))
    eligible: list[NativeWallCandidate] = []
    for item in candidates:
        candidate_length = _length(item)
        paired_evidence = (
            item.evidence_mode == "paired_edges"
            and item.confidence >= max(0.80, minimum_confidence)
            and item.ink_support >= 0.40
            and candidate_length
            >= max(24.0, item.thickness_px * 7.0, minimum_side * 0.055)
        )
        page_scale_solid_band = (
            item.evidence_mode == "single_band"
            and item.confidence >= max(0.74, minimum_confidence)
            and item.ink_support >= 0.60
            and item.thickness_px >= max(7.0, minimum_side * 0.012)
            and candidate_length >= max(item.thickness_px * 10.0, minimum_side * 0.34)
        )
        if paired_evidence or page_scale_solid_band:
            eligible.append(item)
    if len(eligible) < 3:
        return []

    adjacency: list[set[int]] = [set() for _ in eligible]
    for left_index, left in enumerate(eligible):
        for right_index in range(left_index + 1, len(eligible)):
            if _native_lines_connect(
                left,
                eligible[right_index],
                maximum_structural_gap=minimum_side * 0.16,
            ):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    structural_components: list[list[NativeWallCandidate]] = []
    visited: set[int] = set()
    for first in range(len(eligible)):
        if first in visited:
            continue
        stack = [first]
        component_indices: list[int] = []
        visited.add(first)
        while stack:
            index = stack.pop()
            component_indices.append(index)
            for neighbor in adjacency[index]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component = [eligible[index] for index in component_indices]
        if len(component) < 3 or {item.orientation for item in component} != {
            "horizontal",
            "vertical",
        }:
            continue
        left = min(min(item.start_px[0], item.end_px[0]) for item in component)
        right = max(max(item.start_px[0], item.end_px[0]) for item in component)
        top = min(min(item.start_px[1], item.end_px[1]) for item in component)
        bottom = max(max(item.start_px[1], item.end_px[1]) for item in component)
        longest = max(_length(item) for item in component)
        # A real sheet can contain several disconnected structural islands:
        # door gaps, cropped corridors and scan damage often split one building
        # into separate native components.  Requiring substantial span on both
        # axes keeps compact furniture boxes out without throwing every island
        # away merely because it is not the single largest component.
        if (
            right - left < minimum_side * 0.32
            or bottom - top < minimum_side * 0.32
            or longest < minimum_side * 0.32
        ):
            continue
        if sum(_length(item) for item in component) < minimum_side * 0.75:
            continue
        structural_components.append(component)
    selected = [item for component in structural_components for item in component]
    return sorted(selected, key=lambda item: (item.start_px[1], item.start_px[0], item.id))


def _context_wall_graph_is_structurally_sufficient(
    walls: list[object],
    *,
    source_size: tuple[int, int] | None,
) -> bool:
    """Judge the whole graph, not whether one model wall happened to exist."""

    axes = [_wall_axis_like(wall) for wall in walls]
    axes = [axis for axis in axes if axis is not None]
    if len(axes) < 3 or {axis[0] for axis in axes} != {"horizontal", "vertical"}:
        return False
    if source_size is None:
        return True
    width, height = source_size
    minimum_side = float(min(source_size))
    left = min(min(float(wall.start_px[0]), float(wall.end_px[0])) for wall in walls)
    right = max(max(float(wall.start_px[0]), float(wall.end_px[0])) for wall in walls)
    top = min(min(float(wall.start_px[1]), float(wall.end_px[1])) for wall in walls)
    bottom = max(max(float(wall.start_px[1]), float(wall.end_px[1])) for wall in walls)
    total_length = sum(_length_like(wall) for wall in walls)
    return (
        right - left >= min(width * 0.22, minimum_side * 0.45)
        and bottom - top >= min(height * 0.22, minimum_side * 0.45)
        and total_length >= minimum_side * 0.70
    )


def _wall_axis_like(wall: object) -> tuple[str, float] | None:
    start = tuple(float(value) for value in wall.start_px)
    end = tuple(float(value) for value in wall.end_px)
    delta_x = abs(end[0] - start[0])
    delta_y = abs(end[1] - start[1])
    if max(delta_x, delta_y) <= 1:
        return None
    return ("horizontal" if delta_x >= delta_y else "vertical", max(delta_x, delta_y))


def _anchored_wall_assembly_candidates(
    candidates: list[NativeWallCandidate],
    context_walls: list[object],
    *,
    source_size: tuple[int, int] | None,
    minimum_confidence: float,
) -> list[NativeWallCandidate]:
    """Bridge opening-separated source bands only when both ends hit the graph."""

    if not context_walls:
        return []
    minimum_side = float(min(source_size)) if source_size is not None else 512.0
    eligible = [
        item
        for item in candidates
        if item.confidence >= minimum_confidence
        and item.ink_support >= 0.52
        and _length(item) >= max(18.0, item.thickness_px * 2.2)
    ]
    groups: list[list[NativeWallCandidate]] = []
    for candidate in sorted(
        eligible,
        key=lambda item: (
            item.orientation,
            (item.start_px[1] + item.end_px[1]) / 2
            if item.orientation == "horizontal"
            else (item.start_px[0] + item.end_px[0]) / 2,
        ),
    ):
        axis = (
            (candidate.start_px[1] + candidate.end_px[1]) / 2
            if candidate.orientation == "horizontal"
            else (candidate.start_px[0] + candidate.end_px[0]) / 2
        )
        match = next(
            (
                group
                for group in groups
                if group[0].orientation == candidate.orientation
                and abs(
                    axis
                    - float(
                        np.median(
                            [
                                (item.start_px[1] + item.end_px[1]) / 2
                                if item.orientation == "horizontal"
                                else (item.start_px[0] + item.end_px[0]) / 2
                                for item in group
                            ]
                        )
                    )
                )
                <= max(4.0, candidate.thickness_px * 0.65)
            ),
            None,
        )
        if match is None:
            groups.append([candidate])
        else:
            match.append(candidate)

    output: list[NativeWallCandidate] = []
    for index, group in enumerate(groups):
        if len(group) < 2:
            continue
        axis_index = 0 if group[0].orientation == "horizontal" else 1
        intervals = sorted(
            (
                min(item.start_px[axis_index], item.end_px[axis_index]),
                max(item.start_px[axis_index], item.end_px[axis_index]),
            )
            for item in group
        )
        start = intervals[0][0]
        end = max(item[1] for item in intervals)
        span = end - start
        if span < minimum_side * 0.45:
            continue
        merged: list[list[float]] = []
        for interval_start, interval_end in intervals:
            if not merged or interval_start > merged[-1][1]:
                merged.append([interval_start, interval_end])
            else:
                merged[-1][1] = max(merged[-1][1], interval_end)
        evidence_length = sum(item_end - item_start for item_start, item_end in merged)
        if evidence_length / max(1.0, span) < 0.20:
            continue
        axis = float(
            np.median(
                [
                    (item.start_px[1] + item.end_px[1]) / 2
                    if item.orientation == "horizontal"
                    else (item.start_px[0] + item.end_px[0]) / 2
                    for item in group
                ]
            )
        )
        thickness = float(np.median([item.thickness_px for item in group]))
        if group[0].orientation == "horizontal":
            start_px = (start, axis)
            end_px = (end, axis)
        else:
            start_px = (axis, start)
            end_px = (axis, end)
        if thickness <= 3.0:
            nearby_parallel_face = False
            for wall in context_walls:
                wall_axis = _wall_axis_like(wall)
                if wall_axis is None or wall_axis[0] != group[0].orientation:
                    continue
                if group[0].orientation == "horizontal":
                    parallel_distance = abs(
                        axis - (float(wall.start_px[1]) + float(wall.end_px[1])) / 2
                    )
                else:
                    parallel_distance = abs(
                        axis - (float(wall.start_px[0]) + float(wall.end_px[0])) / 2
                    )
                if parallel_distance <= minimum_side * 0.025:
                    nearby_parallel_face = True
                    break
            if nearby_parallel_face:
                continue
        endpoint_tolerance = max(8.0, thickness * 2.0, minimum_side * 0.025)
        if not (
            any(
                _point_to_line_distance(start_px, wall) <= endpoint_tolerance
                for wall in context_walls
            )
            and any(
                _point_to_line_distance(end_px, wall) <= endpoint_tolerance
                for wall in context_walls
            )
        ):
            continue
        output.append(
            NativeWallCandidate(
                id=f"native-wall-anchored-assembly:{index:05d}",
                start_px=start_px,
                end_px=end_px,
                thickness_px=thickness,
                orientation=group[0].orientation,
                evidence_mode="paired_edges",
                ink_support=min(1.0, evidence_length / max(1.0, span)),
                confidence=min(item.confidence for item in group),
            )
        )
    return output


def promote_supported_native_wall_candidates(
    candidates: list[NativeWallCandidate],
    context_walls: list[object],
    *,
    source_ref_ids: list[str],
    minimum_confidence: float = 0.72,
    source_size: tuple[int, int] | None = None,
) -> tuple[list[object], list[NativeWallCandidate]]:
    """Refine known topology with high-resolution evidence, fail-closed.

    Native runs used to be collected and then discarded after writing warnings.
    A run is now usable only when it aligns to an existing wall or is a paired
    edge whose two endpoints attach to the established wall graph. Isolated
    furniture and annotation lines remain unresolved evidence.
    """

    from .core.model.aec_decode import PixelLineProposal

    if not source_ref_ids:
        raise ValueError("source_ref_ids cannot be empty")
    promoted: list[PixelLineProposal] = []
    rejected: list[NativeWallCandidate] = []
    accepted_context = list(context_walls)
    bootstrapped = (
        _bootstrap_native_wall_network(
            candidates,
            source_size=source_size,
            minimum_confidence=minimum_confidence,
        )
        if not _context_wall_graph_is_structurally_sufficient(
            accepted_context,
            source_size=source_size,
        )
        else []
    )
    bootstrap_ids = {item.id for item in bootstrapped}
    for candidate in bootstrapped:
        wall = PixelLineProposal(
            id=f"promoted:{candidate.id}",
            start_px=candidate.start_px,
            end_px=candidate.end_px,
            thickness_px=candidate.thickness_px,
            confidence=candidate.confidence,
            uncertainty=1.0 - candidate.confidence,
            source_ref_ids=source_ref_ids,
            model_version="dajoong-native-wall-network-bootstrap-v1",
            review_required=True,
        )
        promoted.append(wall)
        accepted_context.append(wall)
    anchored_assemblies = _anchored_wall_assembly_candidates(
        candidates,
        accepted_context,
        source_size=source_size,
        minimum_confidence=minimum_confidence,
    )
    for candidate in anchored_assemblies:
        candidate_length = _length(candidate)
        already_explained = any(
            _wall_axis_like(wall) is not None
            and _wall_axis_like(wall)[0] == candidate.orientation
            and _perpendicular_distance(candidate, _wall_candidate(wall))
            <= max(5.0, candidate.thickness_px)
            and (
                _overlap_length(candidate, _wall_candidate(wall))
                / max(1.0, candidate_length)
                >= 0.85
            )
            for wall in accepted_context
        )
        if already_explained:
            continue
        wall = PixelLineProposal(
            id=f"promoted:{candidate.id}",
            start_px=candidate.start_px,
            end_px=candidate.end_px,
            thickness_px=candidate.thickness_px,
            confidence=candidate.confidence,
            uncertainty=1.0 - candidate.confidence,
            source_ref_ids=source_ref_ids,
            model_version="dajoong-native-wall-anchored-assembly-v1",
            review_required=True,
        )
        promoted.append(wall)
        accepted_context.append(wall)
    pending = sorted(
        (item for item in candidates if item.id not in bootstrap_ids),
        key=_length,
        reverse=True,
    )
    changed = True
    while changed and pending:
        changed = False
        next_pending: list[NativeWallCandidate] = []
        for candidate in pending:
            if candidate.confidence < minimum_confidence:
                rejected.append(candidate)
                continue
            candidate_length = _length(candidate)
            aligned = False
            for wall in accepted_context:
                start = wall.start_px
                end = wall.end_px
                orientation = (
                    "horizontal"
                    if abs(float(end[0]) - float(start[0])) >= abs(float(end[1]) - float(start[1]))
                    else "vertical"
                )
                if orientation != candidate.orientation:
                    continue
                if candidate.orientation == "horizontal":
                    perpendicular = abs(
                        (candidate.start_px[1] + candidate.end_px[1]) / 2
                        - (float(start[1]) + float(end[1])) / 2
                    )
                    native_extent = sorted((candidate.start_px[0], candidate.end_px[0]))
                    wall_extent = sorted((float(start[0]), float(end[0])))
                else:
                    perpendicular = abs(
                        (candidate.start_px[0] + candidate.end_px[0]) / 2
                        - (float(start[0]) + float(end[0])) / 2
                    )
                    native_extent = sorted((candidate.start_px[1], candidate.end_px[1]))
                    wall_extent = sorted((float(start[1]), float(end[1])))
                overlap = max(
                    0.0,
                    min(native_extent[1], wall_extent[1]) - max(native_extent[0], wall_extent[0]),
                )
                tolerance = max(
                    4.0,
                    candidate.thickness_px * 1.2,
                    float(getattr(wall, "thickness_px", 0.0) or 0.0) * 1.2,
                )
                if (
                    perpendicular <= tolerance
                    and overlap / max(1.0, min(candidate_length, _length_like(wall))) >= 0.28
                ):
                    aligned = True
                    break
            endpoint_tolerance = max(6.0, candidate.thickness_px * 1.8)
            start_attached = any(
                _point_to_line_distance(candidate.start_px, wall) <= endpoint_tolerance
                for wall in accepted_context
            )
            end_attached = any(
                _point_to_line_distance(candidate.end_px, wall) <= endpoint_tolerance
                for wall in accepted_context
            )
            connector = (
                start_attached
                and end_attached
                and (
                    candidate.evidence_mode == "paired_edges"
                    or (candidate.thickness_px >= 4.0 and candidate.ink_support >= 0.55)
                )
                # An actual new wall is longer than its own thickness. Compact
                # near-square paired boxes are fixtures/cabinetry, not topology.
                and candidate_length >= max(24.0, candidate.thickness_px * 7.0)
            )
            if aligned:
                # Collinear native evidence has already served its only valid
                # role in refine_context_walls_with_native_bands(). Emitting it
                # again creates duplicate fragments and converts cabinetry or
                # dimensions into extra BIM walls.
                continue
            if not connector:
                next_pending.append(candidate)
                continue
            wall = PixelLineProposal(
                id=f"promoted:{candidate.id}",
                start_px=candidate.start_px,
                end_px=candidate.end_px,
                thickness_px=candidate.thickness_px,
                confidence=candidate.confidence,
                uncertainty=1.0 - candidate.confidence,
                source_ref_ids=source_ref_ids,
                model_version="dajoong-native-wall-promotion-v1",
                review_required=candidate.evidence_mode != "paired_edges",
            )
            promoted.append(wall)
            accepted_context.append(wall)
            changed = True
        pending = next_pending
    rejected.extend(pending)
    return promoted, rejected
