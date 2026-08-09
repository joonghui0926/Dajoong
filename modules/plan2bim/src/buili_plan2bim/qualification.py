from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .core.hashing import sha256_file, sha256_json
from .core.model.cad_evidence import raster_ink

DifficultyClass = Literal["simple", "moderate", "difficult", "extreme"]


class DrawingComplexityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.drawing-complexity.v1"
    profiler_version: str = "dajoong-complexity-profiler-0.1"
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    megapixels: float = Field(ge=0)
    analysis_width_px: int = Field(ge=1)
    analysis_height_px: int = Field(ge=1)
    ink_fraction: float = Field(ge=0, le=1)
    significant_components: int = Field(ge=0)
    components_per_megapixel: float = Field(ge=0)
    tiny_component_ink_fraction: float = Field(ge=0, le=1)
    occupied_tile_fraction: float = Field(ge=0, le=1)
    orientation_entropy: float = Field(ge=0, le=1)
    orthogonal_intersection_fraction: float = Field(ge=0, le=1)
    enclosed_region_count: int = Field(ge=0)
    complexity_score: float = Field(ge=0, le=1)
    difficulty_class: DifficultyClass
    reasons: list[str]
    profiling_ms: float = Field(ge=0)
    content_sha256: str = ""

    def finalize(self) -> DrawingComplexityProfile:
        payload = self.model_dump(
            mode="json",
            exclude={"profiling_ms", "content_sha256"},
        )
        self.content_sha256 = sha256_json(payload)
        return self


class ClaimQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    status: Literal["measured", "unmeasured", "model_mismatch", "insufficient_sample"]
    metric: str = ""
    estimate: float | None = Field(default=None, ge=0, le=1)
    conservative_floor: float | None = Field(default=None, ge=0, le=1)
    sample_count: int = Field(default=0, ge=0)
    drawing_classes: list[str] = Field(default_factory=list)
    note: str


class ModelQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.model-qualification.v1"
    primary_model_version: str
    primary_model_sha256: str
    semantic_model_version: str = ""
    semantic_model_sha256: str = ""
    manifest_sha256: str
    exact_model_match: bool
    difficulty_class: DifficultyClass
    benchmark_cohort: str = ""
    benchmark_sample_count: int = Field(default=0, ge=0)
    claims: list[ClaimQualification]
    raw_model_confidence_is_calibrated: bool = False
    production_release_eligible: bool = False
    review_required: bool = True
    review_reasons: list[str]
    content_sha256: str = ""

    def finalize(self) -> ModelQualification:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def profile_drawing(
    image: Image.Image,
    *,
    maximum_analysis_side: int = 768,
) -> DrawingComplexityProfile:
    """Compute bounded, deterministic sheet complexity without learned inference."""

    if maximum_analysis_side < 128:
        raise ValueError("maximum_analysis_side must be at least 128")
    started = time.perf_counter()
    grayscale = image.convert("L")
    width, height = grayscale.size
    scale = min(1.0, maximum_analysis_side / max(width, height))
    analysis_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    analysis_image = (
        grayscale.resize(analysis_size, Image.Resampling.BILINEAR)
        if analysis_size != grayscale.size
        else grayscale
    )
    ink = raster_ink(analysis_image)
    foreground = ink >= 0.35
    foreground_pixels = int(foreground.sum())
    ink_fraction = foreground_pixels / max(1, foreground.size)

    labels, _ = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labels.ravel())[1:]
    significant = component_sizes[component_sizes >= 3]
    tiny_ink = int(component_sizes[component_sizes <= 12].sum())
    tiny_fraction = tiny_ink / max(1, foreground_pixels)
    analysis_megapixels = foreground.size / 1_000_000
    component_density = len(significant) / max(analysis_megapixels, 1e-6)

    tile_rows = np.array_split(foreground, 8, axis=0)
    tile_fractions = [
        float(tile.mean())
        for row in tile_rows
        for tile in np.array_split(row, 8, axis=1)
        if tile.size
    ]
    occupied_tile_fraction = sum(value >= 0.0025 for value in tile_fractions) / max(
        1, len(tile_fractions)
    )

    gradient_x = ndimage.sobel(ink, axis=1, mode="nearest")
    gradient_y = ndimage.sobel(ink, axis=0, mode="nearest")
    magnitude = np.hypot(gradient_x, gradient_y)
    active_gradient = magnitude > max(0.04, float(np.quantile(magnitude, 0.72)))
    if active_gradient.any():
        orientation = np.mod(
            np.arctan2(gradient_y[active_gradient], gradient_x[active_gradient]),
            math.pi,
        )
        histogram, _ = np.histogram(
            orientation,
            bins=12,
            range=(0.0, math.pi),
            weights=magnitude[active_gradient],
        )
        probabilities = histogram / max(float(histogram.sum()), 1e-12)
        nonzero = probabilities[probabilities > 0]
        orientation_entropy = float(
            -(nonzero * np.log(nonzero)).sum() / math.log(len(histogram))
        )
    else:
        orientation_entropy = 0.0

    skeleton = _morphological_skeleton(foreground)
    neighbors = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    intersections = skeleton & (neighbors >= 3)
    intersection_fraction = float(intersections.sum() / max(1, int(skeleton.sum())))

    barrier = ndimage.binary_dilation(foreground, iterations=1)
    enclosed = ndimage.binary_fill_holes(barrier) & ~barrier
    enclosed_labels, _ = ndimage.label(enclosed)
    enclosed_sizes = np.bincount(enclosed_labels.ravel())[1:]
    minimum_enclosure = max(20, round(foreground.size * 0.00008))
    enclosed_count = int((enclosed_sizes >= minimum_enclosure).sum())

    scores = {
        "ink": _clip01((ink_fraction - 0.015) / 0.20),
        "components": _clip01(math.log1p(component_density) / math.log1p(7000)),
        "noise": _clip01(tiny_fraction / 0.55),
        "coverage": occupied_tile_fraction,
        "orientation": orientation_entropy,
        "intersections": _clip01(intersection_fraction / 0.12),
        "enclosures": _clip01(math.log1p(enclosed_count) / math.log(65)),
    }
    score = _clip01(
        0.12 * scores["ink"]
        + 0.18 * scores["components"]
        + 0.08 * scores["noise"]
        + 0.17 * scores["coverage"]
        + 0.17 * scores["orientation"]
        + 0.13 * scores["intersections"]
        + 0.15 * scores["enclosures"]
    )
    if score < 0.36:
        difficulty: DifficultyClass = "simple"
    elif score < 0.54:
        difficulty = "moderate"
    elif score < 0.82:
        difficulty = "difficult"
    else:
        difficulty = "extreme"
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    labels_by_feature = {
        "ink": "dense drawing ink",
        "components": "many disconnected marks",
        "noise": "many small marks",
        "coverage": "content spans most of the sheet",
        "orientation": "many drafting orientations",
        "intersections": "dense orthogonal intersections",
        "enclosures": "many enclosed regions",
    }
    reasons = [labels_by_feature[name] for name, value in ranked[:3] if value >= 0.35]
    if not reasons:
        reasons = ["limited visible drafting structure"]
    return DrawingComplexityProfile(
        width_px=width,
        height_px=height,
        megapixels=round(width * height / 1_000_000, 6),
        analysis_width_px=analysis_size[0],
        analysis_height_px=analysis_size[1],
        ink_fraction=round(ink_fraction, 8),
        significant_components=int(len(significant)),
        components_per_megapixel=round(component_density, 3),
        tiny_component_ink_fraction=round(tiny_fraction, 8),
        occupied_tile_fraction=round(occupied_tile_fraction, 8),
        orientation_entropy=round(orientation_entropy, 8),
        orthogonal_intersection_fraction=round(intersection_fraction, 8),
        enclosed_region_count=enclosed_count,
        complexity_score=round(score, 8),
        difficulty_class=difficulty,
        reasons=reasons,
        profiling_ms=round((time.perf_counter() - started) * 1000, 3),
    ).finalize()


class ModelQualifier:
    """Match an exact model pair and drawing class to sealed benchmark evidence."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != "dajoong.model-qualification-manifest.v1":
            raise ValueError("unsupported model qualification manifest")
        self.manifest_sha256 = sha256_file(self.manifest_path)

    def qualify(
        self,
        complexity: DrawingComplexityProfile,
        *,
        primary_model_version: str,
        primary_model_sha256: str,
        primary_release_authorized: bool,
        semantic_model_version: str = "",
        semantic_model_sha256: str = "",
        semantic_release_authorized: bool = False,
    ) -> ModelQualification:
        expected_primary = str(self.manifest.get("primary_model_sha256") or "")
        expected_semantic = str(self.manifest.get("semantic_model_sha256") or "")
        exact_model_match = primary_model_sha256 == expected_primary and (
            not expected_semantic or semantic_model_sha256 == expected_semantic
        )
        cohorts = list(self.manifest.get("cohorts") or [])
        cohort = next(
            (
                item
                for item in cohorts
                if complexity.difficulty_class in item.get("drawing_classes", [])
            ),
            None,
        )
        sample_count = int((cohort or {}).get("sample_count") or 0)
        claims: list[ClaimQualification] = []
        measured_names: set[str] = set()
        for claim in list((cohort or {}).get("claims") or []):
            name = str(claim.get("claim") or "")
            measured_names.add(name)
            estimate = float(claim["estimate"]) if claim.get("estimate") is not None else None
            floor = (
                float(claim["conservative_floor"])
                if claim.get("conservative_floor") is not None
                else None
            )
            if not exact_model_match:
                status = "model_mismatch"
            elif sample_count < int(self.manifest.get("minimum_production_samples", 100)):
                status = "insufficient_sample"
            else:
                status = "measured"
            claims.append(
                ClaimQualification(
                    claim=name,
                    status=status,
                    metric=str(claim.get("metric") or ""),
                    estimate=estimate,
                    conservative_floor=floor,
                    sample_count=sample_count,
                    drawing_classes=list((cohort or {}).get("drawing_classes") or []),
                    note=str(claim.get("note") or ""),
                )
            )
        for required in list(self.manifest.get("required_release_claims") or []):
            if required in measured_names:
                continue
            claims.append(
                ClaimQualification(
                    claim=str(required),
                    status="unmeasured",
                    note="No sealed benchmark metric is available for this required BIM claim.",
                )
            )
        reasons: list[str] = []
        if not exact_model_match:
            reasons.append("qualification_model_pair_mismatch")
        if cohort is None:
            reasons.append("drawing_complexity_class_not_benchmarked")
        if sample_count < int(self.manifest.get("minimum_production_samples", 100)):
            reasons.append("qualification_sample_count_below_gate")
        if any(claim.status == "unmeasured" for claim in claims):
            reasons.append("required_bim_claims_unmeasured")
        if not primary_release_authorized:
            reasons.append("primary_model_not_release_authorized")
        if expected_semantic and not semantic_release_authorized:
            reasons.append("semantic_model_not_release_authorized")
        if not bool(self.manifest.get("production_authorized", False)):
            reasons.append("qualification_manifest_not_production_authorized")
        release_eligible = not reasons and all(claim.status == "measured" for claim in claims)
        return ModelQualification(
            primary_model_version=primary_model_version,
            primary_model_sha256=primary_model_sha256,
            semantic_model_version=semantic_model_version,
            semantic_model_sha256=semantic_model_sha256,
            manifest_sha256=self.manifest_sha256,
            exact_model_match=exact_model_match,
            difficulty_class=complexity.difficulty_class,
            benchmark_cohort=str((cohort or {}).get("id") or ""),
            benchmark_sample_count=sample_count,
            claims=claims,
            production_release_eligible=release_eligible,
            review_required=not release_eligible,
            review_reasons=sorted(set(reasons)),
        ).finalize()


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    remaining = np.asarray(mask, dtype=np.bool_).copy()
    skeleton = np.zeros_like(remaining)
    structure = ndimage.generate_binary_structure(2, 1)
    while remaining.any():
        eroded = ndimage.binary_erosion(remaining, structure=structure)
        opened = ndimage.binary_dilation(eroded, structure=structure)
        skeleton |= remaining & ~opened
        remaining = eroded
    return skeleton
