from __future__ import annotations

import math
from collections import deque
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..hashing import sha256_json
from .aec_taxonomy import AEC_SYMBOL_CLASSES
from .fourier_wall import FourierWallPrior, wall_segment_angle_deg


class PixelLineProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float | None = Field(default=None, gt=0)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1)
    model_version: str
    review_required: bool


class PixelSymbolProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    symbol_class: str
    center_px: tuple[float, float]
    bbox_px: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1)
    model_version: str
    review_required: bool


class PixelRoomProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    room_class: str
    polygon_px: list[tuple[float, float]] = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    source_ref_ids: list[str] = Field(min_length=1)
    model_version: str
    review_required: bool


class AecTileProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.aec-tile-proposal.v1"
    tile_id: str
    source_ref_ids: list[str] = Field(min_length=1)
    model_version: str
    wall_segments: list[PixelLineProposal]
    symbols: list[PixelSymbolProposal]
    room_regions: list[PixelRoomProposal] = Field(default_factory=list)
    rejected_candidates: int
    content_sha256: str = ""

    def finalize(self) -> AecTileProposal:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30, 30)
    return 1 / (1 + np.exp(-clipped))


def _softmax(value: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = value - value.max(axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -30, 30))
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    output: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            component.append((current_y, current_x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if not delta_x and not delta_y:
                        continue
                    next_y, next_x = current_y + delta_y, current_x + delta_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
        output.append(component)
    return output


def _weighted_centers(probability: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    centers = []
    for component in _components(probability >= threshold):
        y = np.asarray([point[0] for point in component], dtype=np.float64)
        x = np.asarray([point[1] for point in component], dtype=np.float64)
        weights = probability[y.astype(int), x.astype(int)]
        denominator = max(float(weights.sum()), 1e-9)
        centers.append(
            (float(np.dot(x, weights) / denominator), float(np.dot(y, weights) / denominator))
        )
    return centers


def _sample_line(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    distance = math.dist(start, end)
    samples = max(2, math.ceil(distance * 1.5))
    x = np.linspace(start[0], end[0], samples)
    y = np.linspace(start[1], end[1], samples)
    x0 = np.clip(np.floor(x).astype(int), 0, image.shape[1] - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, image.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, image.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, image.shape[0] - 1)
    weight_x, weight_y = x - x0, y - y0
    return (
        image[y0, x0] * (1 - weight_x) * (1 - weight_y)
        + image[y0, x1] * weight_x * (1 - weight_y)
        + image[y1, x0] * (1 - weight_x) * weight_y
        + image[y1, x1] * weight_x * weight_y
    )


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    squared = delta_x * delta_x + delta_y * delta_y
    if squared <= 1e-12:
        return math.dist(point, start), 0.0
    projection = ((point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y) / squared
    nearest = start[0] + projection * delta_x, start[1] + projection * delta_y
    return math.dist(point, nearest), projection


def decode_aec_tile(
    *,
    tile_id: str,
    source_ref_ids: list[str],
    model_version: str,
    structure_logits: np.ndarray,
    symbol_logits: np.ndarray,
    metric_offsets: np.ndarray,
    uncertainty: np.ndarray,
    junction_threshold: float = 0.65,
    wall_threshold: float = 0.62,
    symbol_threshold: float = 0.7,
    maximum_uncertainty: float = 0.35,
    fourier_prior: FourierWallPrior | None = None,
    strict_fourier_candidate_pruning: bool = False,
    fourier_tolerance_deg: float = 8.0,
) -> AecTileProposal:
    del metric_offsets  # Reserved for subpixel metric correction after scale is solved.
    structure = np.asarray(structure_logits, dtype=np.float32)
    symbols = np.asarray(symbol_logits, dtype=np.float32)
    uncertainty_map = np.asarray(uncertainty, dtype=np.float32)
    if structure.ndim != 3 or structure.shape[0] != 4:
        raise ValueError("structure_logits must be [4, height, width]")
    if symbols.ndim != 3 or symbols.shape[0] != len(AEC_SYMBOL_CLASSES):
        raise ValueError("symbol_logits do not match the AEC symbol taxonomy")
    if uncertainty_map.shape == (1, *structure.shape[1:]):
        uncertainty_map = uncertainty_map[0]
    if uncertainty_map.shape != structure.shape[1:]:
        raise ValueError("uncertainty must be [height, width] or [1, height, width]")
    if not source_ref_ids:
        raise ValueError("source_ref_ids cannot be empty")

    structure_probability = _sigmoid(structure)
    wall_probability = structure_probability[0]
    junctions = _weighted_centers(structure_probability[1], junction_threshold)
    wall_segments: list[PixelLineProposal] = []
    rejected = 0
    minimum_length = 6.0
    for left_index, right_index in combinations(range(len(junctions)), 2):
        start, end = junctions[left_index], junctions[right_index]
        if math.dist(start, end) < minimum_length:
            rejected += 1
            continue
        if (
            strict_fourier_candidate_pruning
            and fourier_prior is not None
            and fourier_prior.eligible_for_candidate_pruning
            and not fourier_prior.supports(
                wall_segment_angle_deg(start, end), tolerance_deg=fourier_tolerance_deg
            )
        ):
            rejected += 1
            continue
        blocked = False
        for candidate_index, candidate in enumerate(junctions):
            if candidate_index in {left_index, right_index}:
                continue
            distance, projection = _point_segment_distance(candidate, start, end)
            if distance <= 4.0 and 0.05 < projection < 0.95:
                blocked = True
                break
        if blocked:
            rejected += 1
            continue
        line_scores = _sample_line(wall_probability, start, end)
        confidence = float(np.quantile(line_scores, 0.2))
        if confidence < wall_threshold:
            rejected += 1
            continue
        line_uncertainty = float(_sample_line(uncertainty_map, start, end).mean())
        segment_id = f"{tile_id}:wall:{len(wall_segments)}"
        wall_segments.append(
            PixelLineProposal(
                id=segment_id,
                start_px=start,
                end_px=end,
                confidence=confidence,
                uncertainty=line_uncertainty,
                source_ref_ids=source_ref_ids,
                model_version=model_version,
                review_required=line_uncertainty > maximum_uncertainty,
            )
        )

    symbol_probability = _softmax(symbols, axis=0)
    symbol_class = symbol_probability.argmax(axis=0)
    symbol_output: list[PixelSymbolProposal] = []
    for class_index, class_name in enumerate(AEC_SYMBOL_CLASSES[1:], start=1):
        mask = (symbol_class == class_index) & (symbol_probability[class_index] >= symbol_threshold)
        for component in _components(mask):
            if len(component) < 2:
                continue
            y = np.asarray([point[0] for point in component], dtype=np.float64)
            x = np.asarray([point[1] for point in component], dtype=np.float64)
            confidence = float(symbol_probability[class_index, y.astype(int), x.astype(int)].mean())
            component_uncertainty = float(uncertainty_map[y.astype(int), x.astype(int)].mean())
            symbol_output.append(
                PixelSymbolProposal(
                    id=f"{tile_id}:symbol:{len(symbol_output)}",
                    symbol_class=class_name,
                    center_px=(float(x.mean()), float(y.mean())),
                    bbox_px=(float(x.min()), float(y.min()), float(x.max()), float(y.max())),
                    confidence=confidence,
                    uncertainty=component_uncertainty,
                    source_ref_ids=source_ref_ids,
                    model_version=model_version,
                    review_required=component_uncertainty > maximum_uncertainty,
                )
            )
    return AecTileProposal(
        tile_id=tile_id,
        source_ref_ids=source_ref_ids,
        model_version=model_version,
        wall_segments=wall_segments,
        symbols=symbol_output,
        rejected_candidates=rejected,
    ).finalize()
