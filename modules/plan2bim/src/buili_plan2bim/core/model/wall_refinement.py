from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from .aec_decode import AecTileProposal, PixelLineProposal
from .fourier_wall import FourierWallPrior, wall_segment_angle_deg

REFINER_VERSION = "dajoong-wall-reproject-0.1"
Point2D = tuple[float, float]


class WallRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    render_width_px: int = Field(default=3, ge=1, le=15)
    support_half_width_px: int = Field(default=4, ge=1, le=12)
    maximum_global_shift_px: int = Field(default=8, ge=0, le=64)
    phase_max_dimension_px: int = Field(default=256, ge=64, le=2048)
    minimum_phase_peak_ratio: float = Field(default=1.02, ge=1)
    minimum_global_gain: float = Field(default=0.01, ge=0, le=1)
    endpoint_cluster_radius_px: float = Field(default=2.5, ge=0, le=12)
    local_search_radius_px: int = Field(default=2, ge=0, le=8)
    local_passes: int = Field(default=2, ge=0, le=5)
    minimum_local_gain: float = Field(default=0.002, ge=0, le=1)
    minimum_segment_support: float = Field(default=0.68, ge=0, le=1)
    orientation_regularization: float = Field(default=0.04, ge=0, le=0.25)
    normal_equation_regularization: float = Field(default=0.08, gt=0, le=1)


class SegmentRefinementEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    before_support: float = Field(ge=0, le=1)
    after_support: float = Field(ge=0, le=1)
    residual_px: float = Field(ge=0)
    review_required: bool


class WallRefinementAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.wall-refinement-audit.v1"
    refiner_version: str = REFINER_VERSION
    tile_id: str
    phase_shift_px: tuple[int, int]
    phase_peak_ratio: float = Field(ge=0)
    phase_shift_applied: bool
    before_mean_support: float = Field(ge=0, le=1)
    after_mean_support: float = Field(ge=0, le=1)
    candidate_evaluations: int = Field(ge=0)
    elapsed_ns: int = Field(ge=0)
    segments: list[SegmentRefinementEvidence]


class WallRefinementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: AecTileProposal
    audit: WallRefinementAudit


def render_wall_mask(
    shape: tuple[int, int],
    segments: list[PixelLineProposal],
    *,
    width_px: int = 3,
) -> np.ndarray:
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for segment in segments:
        draw.line((*segment.start_px, *segment.end_px), fill=255, width=width_px)
    return np.asarray(image, dtype=np.float32) / 255.0


def estimate_phase_translation(
    reference_ink: np.ndarray,
    moving_mask: np.ndarray,
    *,
    maximum_shift_px: int,
) -> tuple[tuple[int, int], float]:
    reference = np.asarray(reference_ink, dtype=np.float32)
    moving = np.asarray(moving_mask, dtype=np.float32)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("reference_ink and moving_mask must be equally sized 2-D arrays")
    if float(np.ptp(reference)) <= 1e-8 or float(moving.max()) <= 1e-8:
        return (0, 0), 0.0
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    reference_frequency = np.fft.fft2((reference - reference.mean()) * window)
    moving_frequency = np.fft.fft2((moving - moving.mean()) * window)
    cross_power = reference_frequency * np.conj(moving_frequency)
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, 1e-12)
    correlation = np.abs(np.fft.ifft2(cross_power))

    height, width = reference.shape
    y_indices = np.arange(height)
    x_indices = np.arange(width)
    signed_y = np.where(y_indices <= height // 2, y_indices, y_indices - height)
    signed_x = np.where(x_indices <= width // 2, x_indices, x_indices - width)
    allowed = (np.abs(signed_y[:, None]) <= maximum_shift_px) & (
        np.abs(signed_x[None, :]) <= maximum_shift_px
    )
    restricted = np.where(allowed, correlation, -np.inf)
    peak_flat = int(np.argmax(restricted))
    peak_y, peak_x = np.unravel_index(peak_flat, restricted.shape)
    peak = float(restricted[peak_y, peak_x])
    excluded = restricted.copy()
    for delta_y in range(-1, 2):
        for delta_x in range(-1, 2):
            excluded[(peak_y + delta_y) % height, (peak_x + delta_x) % width] = -np.inf
    second = float(np.max(excluded))
    ratio = peak / max(second, 1e-12) if np.isfinite(second) else 0.0
    return (int(signed_x[peak_x]), int(signed_y[peak_y])), ratio


def _thumbnail_phase_translation(
    reference_ink: np.ndarray,
    moving_mask: np.ndarray,
    *,
    maximum_shift_px: int,
    maximum_dimension_px: int,
) -> tuple[tuple[int, int], float]:
    nonzero_y, nonzero_x = np.nonzero(moving_mask >= 0.05)
    if len(nonzero_x) == 0:
        return (0, 0), 0.0
    margin = maximum_shift_px + 4
    left = max(0, int(nonzero_x.min()) - margin)
    right = min(moving_mask.shape[1], int(nonzero_x.max()) + margin + 1)
    top = max(0, int(nonzero_y.min()) - margin)
    bottom = min(moving_mask.shape[0], int(nonzero_y.max()) + margin + 1)
    reference_crop = reference_ink[top:bottom, left:right]
    moving_crop = moving_mask[top:bottom, left:right]
    scale = min(1.0, maximum_dimension_px / max(reference_crop.shape))
    if scale < 1.0:
        resized_width = max(16, round(reference_crop.shape[1] * scale))
        resized_height = max(16, round(reference_crop.shape[0] * scale))
        size = resized_width, resized_height
        reference_crop = np.asarray(
            Image.fromarray(reference_crop.astype(np.float32), mode="F").resize(
                size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        moving_crop = np.asarray(
            Image.fromarray(moving_crop.astype(np.float32), mode="F").resize(
                size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
    scaled_maximum = max(1, math.ceil(maximum_shift_px * scale))
    shift, ratio = estimate_phase_translation(
        reference_crop, moving_crop, maximum_shift_px=scaled_maximum
    )
    restored = (
        int(np.clip(round(shift[0] / scale), -maximum_shift_px, maximum_shift_px)),
        int(np.clip(round(shift[1] / scale), -maximum_shift_px, maximum_shift_px)),
    )
    return restored, ratio


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    weight_x = x - x0
    weight_y = y - y0
    return (
        image[y0, x0] * (1 - weight_x) * (1 - weight_y)
        + image[y0, x1] * weight_x * (1 - weight_y)
        + image[y1, x0] * (1 - weight_x) * weight_y
        + image[y1, x1] * weight_x * weight_y
    )


def _segment_measurement(
    image: np.ndarray,
    start: Point2D,
    end: Point2D,
    *,
    half_width: int,
) -> tuple[float, float, float, float]:
    length = math.dist(start, end)
    if length < 2:
        return 0.0, float(half_width), 0.0, 0.0
    sample_count = max(8, math.ceil(length))
    along = np.linspace(0.0, 1.0, sample_count)
    base_x = start[0] + (end[0] - start[0]) * along
    base_y = start[1] + (end[1] - start[1]) * along
    normal_x = -(end[1] - start[1]) / length
    normal_y = (end[0] - start[0]) / length
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    x = base_x[None, :] + offsets[:, None] * normal_x
    y = base_y[None, :] + offsets[:, None] * normal_y
    values = _bilinear(image, x, y)
    maximum = values.max(axis=0)
    coverage = float(np.mean(maximum >= 0.45))
    continuity = float(np.quantile(maximum, 0.2))

    def signed_offset(section: np.ndarray) -> float:
        profile = np.quantile(section, 0.5, axis=1)
        profile = np.maximum(0.0, profile - float(profile.min()))
        mass = float(profile.sum())
        if mass <= 0.05:
            return 0.0
        return float(np.dot(profile, offsets) / mass)

    endpoint_window = max(4, min(sample_count // 3, 32))
    start_offset = signed_offset(values[:, :endpoint_window])
    end_offset = signed_offset(values[:, -endpoint_window:])
    if float(values.sum()) > 0.05:
        residual = float((abs(start_offset) + abs(end_offset)) / 2)
        symmetry = max(0.0, 1.0 - residual / max(1, half_width))
    else:
        residual = float(half_width)
        symmetry = 0.0
    score = float(np.clip(0.50 * coverage + 0.35 * continuity + 0.15 * symmetry, 0, 1))
    return score, residual, start_offset, end_offset


def _segment_support(
    image: np.ndarray,
    start: Point2D,
    end: Point2D,
    *,
    half_width: int,
) -> tuple[float, float]:
    score, residual, _, _ = _segment_measurement(image, start, end, half_width=half_width)
    return score, residual


def _cluster_endpoints(
    segments: list[PixelLineProposal], radius: float
) -> tuple[list[Point2D], list[tuple[int, int]]]:
    endpoints = [point for segment in segments for point in (segment.start_px, segment.end_px)]
    parents = list(range(len(endpoints)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            if math.dist(endpoints[left], endpoints[right]) <= radius:
                union(left, right)
    groups: dict[int, list[Point2D]] = defaultdict(list)
    for index, point in enumerate(endpoints):
        groups[root(index)].append(point)
    ordered_roots = sorted(
        groups, key=lambda item: min(endpoints.index(point) for point in groups[item])
    )
    root_to_node = {item: index for index, item in enumerate(ordered_roots)}
    nodes = [
        (
            float(np.mean([point[0] for point in groups[item]])),
            float(np.mean([point[1] for point in groups[item]])),
        )
        for item in ordered_roots
    ]
    segment_nodes = [
        (root_to_node[root(2 * index)], root_to_node[root(2 * index + 1)])
        for index in range(len(segments))
    ]
    return nodes, segment_nodes


def refine_wall_proposal(
    proposal: AecTileProposal,
    reference_ink: np.ndarray,
    *,
    config: WallRefinementConfig | None = None,
    fourier_prior: FourierWallPrior | None = None,
) -> WallRefinementResult:
    started = time.perf_counter_ns()
    config = config or WallRefinementConfig()
    image = np.asarray(reference_ink, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("reference_ink must be a finite 2-D array")
    if image.size == 0:
        raise ValueError("reference_ink cannot be empty")
    dynamic_range = float(np.ptp(image))
    if dynamic_range > 1e-8:
        image = (image - float(image.min())) / dynamic_range
    else:
        image = np.zeros_like(image)
    if not proposal.wall_segments:
        audit = WallRefinementAudit(
            tile_id=proposal.tile_id,
            phase_shift_px=(0, 0),
            phase_peak_ratio=0,
            phase_shift_applied=False,
            before_mean_support=0,
            after_mean_support=0,
            candidate_evaluations=0,
            elapsed_ns=time.perf_counter_ns() - started,
            segments=[],
        )
        return WallRefinementResult(proposal=proposal, audit=audit)

    nodes, segment_nodes = _cluster_endpoints(
        proposal.wall_segments, config.endpoint_cluster_radius_px
    )
    before_scores = [
        _segment_support(
            image,
            nodes[start_index],
            nodes[end_index],
            half_width=config.support_half_width_px,
        )[0]
        for start_index, end_index in segment_nodes
    ]
    before_mean = float(np.mean(before_scores))
    rendered = render_wall_mask(
        image.shape, proposal.wall_segments, width_px=config.render_width_px
    )
    phase_shift, phase_ratio = _thumbnail_phase_translation(
        image,
        rendered,
        maximum_shift_px=config.maximum_global_shift_px,
        maximum_dimension_px=config.phase_max_dimension_px,
    )
    shifted_nodes = [(point[0] + phase_shift[0], point[1] + phase_shift[1]) for point in nodes]
    shifted_scores = [
        _segment_support(
            image,
            shifted_nodes[start_index],
            shifted_nodes[end_index],
            half_width=config.support_half_width_px,
        )[0]
        for start_index, end_index in segment_nodes
    ]
    phase_applied = (
        phase_ratio >= config.minimum_phase_peak_ratio
        and float(np.mean(shifted_scores)) >= before_mean + config.minimum_global_gain
    )
    if phase_applied:
        nodes = shifted_nodes

    base_nodes = list(nodes)
    incident: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for segment_index, (start_index, end_index) in enumerate(segment_nodes):
        incident[start_index].append((segment_index, True))
        incident[end_index].append((segment_index, False))
    evaluations = 0

    def graph_measurements(
        candidate_nodes: list[Point2D],
    ) -> list[tuple[float, float, float, float]]:
        nonlocal evaluations
        output = []
        for start_index, end_index in segment_nodes:
            start, end = candidate_nodes[start_index], candidate_nodes[end_index]
            measurement = _segment_measurement(
                image, start, end, half_width=config.support_half_width_px
            )
            score = measurement[0]
            if fourier_prior is not None and fourier_prior.orientations:
                alignment = fourier_prior.alignment(
                    wall_segment_angle_deg(start, end), tolerance_deg=10
                )
                score = (
                    score * (1 - config.orientation_regularization)
                    + alignment * config.orientation_regularization
                )
                measurement = (score, *measurement[1:])
            output.append(measurement)
            evaluations += 1
        return output

    # The thumbnail phase estimate is deliberately coarse. Keep enough local
    # correction radius to absorb its quantization error even when it was useful.
    maximum_node_shift = config.local_search_radius_px + config.maximum_global_shift_px
    height, width = image.shape
    for _ in range(config.local_passes):
        current_measurements = graph_measurements(nodes)
        proposed_nodes = list(nodes)
        for node_index, references in incident.items():
            matrix_rows = []
            targets = []
            weights = []
            for segment_index, is_start in references:
                start_index, end_index = segment_nodes[segment_index]
                start, end = nodes[start_index], nodes[end_index]
                length = math.dist(start, end)
                if length <= 1e-6:
                    continue
                normal = (-(end[1] - start[1]) / length, (end[0] - start[0]) / length)
                measurement = current_measurements[segment_index]
                matrix_rows.append(normal)
                targets.append(measurement[2] if is_start else measurement[3])
                weights.append(max(0.05, measurement[0]))
            if not matrix_rows:
                continue
            matrix = np.asarray(matrix_rows, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            weight = np.asarray(weights, dtype=np.float64)
            lhs = (
                matrix.T @ (matrix * weight[:, None])
                + np.eye(2) * config.normal_equation_regularization
            )
            rhs = matrix.T @ (target * weight)
            delta = np.linalg.solve(lhs, rhs)
            candidate = np.asarray(nodes[node_index]) + delta
            displacement = candidate - np.asarray(base_nodes[node_index])
            norm = float(np.linalg.norm(displacement))
            if norm > maximum_node_shift > 0:
                displacement *= maximum_node_shift / norm
                candidate = np.asarray(base_nodes[node_index]) + displacement
            candidate[0] = np.clip(candidate[0], 0, width - 1)
            candidate[1] = np.clip(candidate[1], 0, height - 1)
            proposed_nodes[node_index] = float(candidate[0]), float(candidate[1])

        proposed_measurements = graph_measurements(proposed_nodes)
        current_mean = float(np.mean([item[0] for item in current_measurements]))
        proposed_mean = float(np.mean([item[0] for item in proposed_measurements]))
        if proposed_mean >= current_mean + config.minimum_local_gain:
            nodes = proposed_nodes
            continue

        halfway_nodes = [
            ((current[0] + proposed[0]) / 2, (current[1] + proposed[1]) / 2)
            for current, proposed in zip(nodes, proposed_nodes, strict=True)
        ]
        halfway_measurements = graph_measurements(halfway_nodes)
        halfway_mean = float(np.mean([item[0] for item in halfway_measurements]))
        if halfway_mean >= current_mean + config.minimum_local_gain:
            nodes = halfway_nodes
            continue
        break

    refined_segments: list[PixelLineProposal] = []
    evidence: list[SegmentRefinementEvidence] = []
    after_scores = []
    for index, original in enumerate(proposal.wall_segments):
        start_index, end_index = segment_nodes[index]
        start, end = nodes[start_index], nodes[end_index]
        score, residual = _segment_support(
            image, start, end, half_width=config.support_half_width_px
        )
        after_scores.append(score)
        review_required = original.review_required or score < config.minimum_segment_support
        evidence.append(
            SegmentRefinementEvidence(
                segment_id=original.id,
                before_support=before_scores[index],
                after_support=score,
                residual_px=residual,
                review_required=review_required,
            )
        )
        refined_segments.append(
            PixelLineProposal(
                id=original.id,
                start_px=start,
                end_px=end,
                confidence=min(original.confidence, score),
                uncertainty=max(original.uncertainty, 1.0 - score),
                source_ref_ids=original.source_ref_ids,
                model_version=f"{original.model_version}|{REFINER_VERSION}",
                review_required=review_required,
            )
        )
    refined = proposal.model_copy(
        update={
            "model_version": f"{proposal.model_version}|{REFINER_VERSION}",
            "wall_segments": refined_segments,
            "content_sha256": "",
        },
        deep=True,
    ).finalize()
    audit = WallRefinementAudit(
        tile_id=proposal.tile_id,
        phase_shift_px=phase_shift,
        phase_peak_ratio=phase_ratio,
        phase_shift_applied=phase_applied,
        before_mean_support=before_mean,
        after_mean_support=float(np.mean(after_scores)),
        candidate_evaluations=evaluations,
        elapsed_ns=time.perf_counter_ns() - started,
        segments=evidence,
    )
    return WallRefinementResult(proposal=refined, audit=audit)
