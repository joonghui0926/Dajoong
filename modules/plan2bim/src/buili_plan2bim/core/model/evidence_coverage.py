from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from ..hashing import sha256_json
from .aec_decode import AecTileProposal, PixelLineProposal
from .wall_refinement import render_wall_mask

COVERAGE_VERSION = "dajoong-evidence-coverage-0.1"
ResidualKind = Literal[
    "possible_missing_linear_entity",
    "unexplained_compact_entity",
]


class CoverageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    foreground_threshold: float = Field(default=0.35, ge=0, le=1)
    wall_render_width_px: int = Field(default=3, ge=1, le=15)
    alignment_tolerance_px: int = Field(default=2, ge=0, le=8)
    minimum_component_pixels: int = Field(default=3, ge=1)
    linear_component_minimum_length_px: float = Field(default=10, ge=2)
    linear_component_minimum_elongation: float = Field(default=4, ge=1)
    minimum_coverage_ratio: float = Field(default=0.995, ge=0, le=1)


class ResidualEvidenceComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ResidualKind
    bbox_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    pixel_count: int = Field(ge=1)
    ink_mass: float = Field(ge=0)
    major_axis_start_px: tuple[float, float]
    major_axis_end_px: tuple[float, float]
    major_axis_length_px: float = Field(ge=0)
    orientation_deg: float = Field(ge=0, lt=180)
    elongation: float = Field(ge=1)
    source_ref_ids: list[str] = Field(min_length=1)
    review_required: bool = True


class EvidenceCoverageCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.evidence-coverage-certificate.v1"
    coverage_version: str = COVERAGE_VERSION
    tile_id: str
    foreground_pixels: int = Field(ge=0)
    explained_foreground_pixels: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    residual_components: list[ResidualEvidenceComponent]
    recovery_wall_proposals: list[PixelLineProposal]
    release_allowed: bool
    content_sha256: str = ""

    def finalize(self) -> EvidenceCoverageCertificate:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=np.bool_).copy()
    if radius <= 0:
        return result
    source = result.copy()
    height, width = source.shape
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            source_y_start = max(0, -delta_y)
            source_y_end = min(height, height - delta_y)
            source_x_start = max(0, -delta_x)
            source_x_end = min(width, width - delta_x)
            target_y_start = source_y_start + delta_y
            target_y_end = source_y_end + delta_y
            target_x_start = source_x_start + delta_x
            target_x_end = source_x_end + delta_x
            result[target_y_start:target_y_end, target_x_start:target_x_end] |= source[
                source_y_start:source_y_end, source_x_start:source_x_end
            ]
    return result


def _components_fallback(mask: np.ndarray) -> Iterator[np.ndarray]:
    visited = np.zeros_like(mask, dtype=np.bool_)
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        component = []
        while queue:
            current_y, current_x = queue.popleft()
            component.append((current_y, current_x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
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
        yield np.asarray(component, dtype=np.int32)


def _components(mask: np.ndarray) -> Iterator[np.ndarray]:
    """Yield 8-connected coordinates, using compiled labeling when available."""

    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - core-only installation fallback
        yield from _components_fallback(mask)
        return
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        local = np.argwhere(labels[slices] == label_id).astype(np.int32)
        if local.size == 0:
            continue
        local[:, 0] += int(slices[0].start or 0)
        local[:, 1] += int(slices[1].start or 0)
        yield local


def _symbol_mask(shape: tuple[int, int], proposal: AecTileProposal) -> np.ndarray:
    height, width = shape
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for symbol in proposal.symbols:
        draw.rectangle(symbol.bbox_px, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def _component_evidence(
    component: np.ndarray,
    image: np.ndarray,
    *,
    component_id: str,
    source_ref_ids: list[str],
    config: CoverageConfig,
) -> ResidualEvidenceComponent:
    points_yx = np.asarray(component, dtype=np.float64)
    points = points_yx[:, ::-1]
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(component) >= 2:
        covariance = centered.T @ centered / len(component)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
        major_value = float(max(eigenvalues))
        minor_value = float(min(eigenvalues))
        projection = centered @ major_vector
        start = centroid + major_vector * float(projection.min())
        end = centroid + major_vector * float(projection.max())
        length = float(projection.max() - projection.min())
        elongation = math.sqrt((major_value + 0.25) / (minor_value + 0.25))
    else:
        start = end = centroid
        length = 0.0
        elongation = 1.0
    orientation = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180
    kind: ResidualKind = (
        "possible_missing_linear_entity"
        if length >= config.linear_component_minimum_length_px
        and elongation >= config.linear_component_minimum_elongation
        else "unexplained_compact_entity"
    )
    y = points_yx[:, 0].astype(int)
    x = points_yx[:, 1].astype(int)
    return ResidualEvidenceComponent(
        id=component_id,
        kind=kind,
        bbox_px=(int(x.min()), int(y.min()), int(x.max()), int(y.max())),
        centroid_px=(float(centroid[0]), float(centroid[1])),
        pixel_count=len(component),
        ink_mass=float(image[y, x].sum()),
        major_axis_start_px=(float(start[0]), float(start[1])),
        major_axis_end_px=(float(end[0]), float(end[1])),
        major_axis_length_px=length,
        orientation_deg=orientation,
        elongation=max(1.0, float(elongation)),
        source_ref_ids=source_ref_ids,
    )


def certify_evidence_coverage(
    reference_ink: np.ndarray,
    proposal: AecTileProposal,
    *,
    known_text_mask: np.ndarray | None = None,
    known_dimension_mask: np.ndarray | None = None,
    known_hatch_mask: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    config: CoverageConfig | None = None,
) -> EvidenceCoverageCertificate:
    config = config or CoverageConfig()
    image = np.asarray(reference_ink, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("reference_ink must be a finite 2-D array")
    dynamic_range = float(np.ptp(image))
    if dynamic_range > 1e-8:
        image = (image - float(image.min())) / dynamic_range
    else:
        image = np.zeros_like(image)
    active = (
        np.ones_like(image, dtype=np.bool_) if active_mask is None else active_mask.astype(bool)
    )
    if active.shape != image.shape:
        raise ValueError("active_mask must match reference_ink")
    foreground = (image >= config.foreground_threshold) & active
    explained = (
        render_wall_mask(
            image.shape,
            proposal.wall_segments,
            width_px=config.wall_render_width_px,
        )
        > 0
    )
    explained |= _symbol_mask(image.shape, proposal)
    for name, mask in (
        ("known_text_mask", known_text_mask),
        ("known_dimension_mask", known_dimension_mask),
        ("known_hatch_mask", known_hatch_mask),
    ):
        if mask is None:
            continue
        mask_array = np.asarray(mask, dtype=np.bool_)
        if mask_array.shape != image.shape:
            raise ValueError(f"{name} must match reference_ink")
        explained |= mask_array
    explained = _dilate(explained, config.alignment_tolerance_px)
    explained_foreground = foreground & explained
    foreground_pixels = int(foreground.sum())
    explained_pixels = int(explained_foreground.sum())
    coverage = 1.0 if foreground_pixels == 0 else explained_pixels / foreground_pixels
    residual_mask = foreground & ~explained
    residual_components = []
    recovery = []
    for component in _components(residual_mask):
        if len(component) < config.minimum_component_pixels:
            continue
        item = _component_evidence(
            component,
            image,
            component_id=f"{proposal.tile_id}:residual:{len(residual_components)}",
            source_ref_ids=proposal.source_ref_ids,
            config=config,
        )
        residual_components.append(item)
        if item.kind == "possible_missing_linear_entity":
            confidence = min(0.49, 0.2 + 0.03 * math.log1p(item.pixel_count))
            recovery.append(
                PixelLineProposal(
                    id=f"{item.id}:wall-candidate",
                    start_px=item.major_axis_start_px,
                    end_px=item.major_axis_end_px,
                    confidence=confidence,
                    uncertainty=max(0.51, 1.0 - confidence),
                    source_ref_ids=item.source_ref_ids,
                    model_version=COVERAGE_VERSION,
                    review_required=True,
                )
            )
    release_allowed = coverage >= config.minimum_coverage_ratio and not residual_components
    return EvidenceCoverageCertificate(
        tile_id=proposal.tile_id,
        foreground_pixels=foreground_pixels,
        explained_foreground_pixels=explained_pixels,
        coverage_ratio=coverage,
        residual_components=residual_components,
        recovery_wall_proposals=recovery,
        release_allowed=release_allowed,
    ).finalize()
