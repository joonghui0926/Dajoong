from __future__ import annotations

import math
import time
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class FourierOrientationPeak(BaseModel):
    model_config = ConfigDict(extra="forbid")

    angle_deg: float = Field(ge=0, lt=180)
    power_share: float = Field(ge=0, le=1)


class FourierWallPrior(BaseModel):
    """Global wall-orientation evidence, never authoritative wall geometry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.fourier-wall-prior.v1"
    image_shape: tuple[int, int]
    orientations: list[FourierOrientationPeak]
    spectral_concentration: float = Field(ge=0, le=1)
    foreground_fraction: float = Field(ge=0, le=1)
    eligible_for_candidate_pruning: bool

    def alignment(self, angle_deg: float, tolerance_deg: float = 8.0) -> float:
        if not self.orientations:
            return 0.0
        distance = min(
            _orientation_distance_deg(angle_deg, peak.angle_deg) for peak in self.orientations
        )
        return max(0.0, 1.0 - distance / max(tolerance_deg, 1e-6))

    def supports(self, angle_deg: float, tolerance_deg: float = 8.0) -> bool:
        return self.alignment(angle_deg, tolerance_deg) > 0.0


class FourierBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_shape: tuple[int, int]
    warmup_runs: int
    measured_runs: int
    median_ns: int
    p95_ns: int
    minimum_ns: int


def _orientation_distance_deg(left: float, right: float) -> float:
    difference = abs((left - right) % 180.0)
    return min(difference, 180.0 - difference)


def _circular_smooth(histogram: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return histogram.copy()
    output = np.zeros_like(histogram)
    kernel = np.asarray(
        [radius + 1 - abs(offset) for offset in range(-radius, radius + 1)],
        dtype=np.float64,
    )
    kernel /= kernel.sum()
    for offset, weight in zip(range(-radius, radius + 1), kernel, strict=True):
        output += np.roll(histogram, offset) * weight
    return output


@lru_cache(maxsize=16)
def _fourier_layout(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache all shape-dependent work; runtime only computes the tile FFT."""

    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    frequency_y = np.fft.fftfreq(height)
    frequency_x = np.fft.rfftfreq(width)
    grid_x, grid_y = np.meshgrid(frequency_x, frequency_y)
    cycles = np.hypot(grid_x * width, grid_y * height)
    maximum_cycles = max(4.0, min(height, width) * 0.35)
    frequency_indices = np.flatnonzero((cycles >= 2.0) & (cycles <= maximum_cycles))
    frequency_angle = np.degrees(np.arctan2(grid_y, grid_x))
    wall_angle = np.mod(frequency_angle - 90.0, 180.0)
    bin_index = (np.floor(wall_angle).astype(np.int16) % 180).ravel()[frequency_indices]
    return window, frequency_indices, bin_index


def detect_fourier_wall_prior(
    ink: np.ndarray,
    *,
    maximum_orientations: int = 4,
    minimum_peak_separation_deg: float = 12.0,
    orientation_band_deg: float = 6.0,
    minimum_spectral_concentration: float = 0.34,
) -> FourierWallPrior:
    """Estimate dominant line orientations from a raster tile's 2-D spectrum.

    The result is intentionally a prior. Fourier magnitude cannot recover wall
    endpoints, distinguish walls from text/hatches, or resolve door openings.
    """

    image = np.asarray(ink, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 16:
        raise ValueError("ink must be a 2-D image with both dimensions >= 16")
    if not np.isfinite(image).all():
        raise ValueError("ink contains non-finite values")
    dynamic_range = float(np.ptp(image))
    if dynamic_range <= 1e-8:
        return FourierWallPrior(
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            orientations=[],
            spectral_concentration=0.0,
            foreground_fraction=0.0,
            eligible_for_candidate_pruning=False,
        )

    normalized = (image - float(image.min())) / dynamic_range
    foreground_fraction = float(np.mean(normalized >= 0.5))
    centered = normalized - float(normalized.mean())
    window, frequency_indices, bin_index = _fourier_layout(*image.shape)
    spectrum = np.abs(np.fft.rfft2(centered * window)) ** 2
    weights = spectrum.ravel()[frequency_indices]
    total_power = float(weights.sum())
    if total_power <= 1e-12:
        return FourierWallPrior(
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            orientations=[],
            spectral_concentration=0.0,
            foreground_fraction=foreground_fraction,
            eligible_for_candidate_pruning=False,
        )

    histogram = np.bincount(bin_index, weights=weights, minlength=180)
    histogram = _circular_smooth(histogram.astype(np.float64), radius=2)

    candidates = np.argsort(histogram)[::-1]
    selected: list[int] = []
    for candidate in candidates:
        if all(
            _orientation_distance_deg(float(candidate), float(existing))
            >= minimum_peak_separation_deg
            for existing in selected
        ):
            selected.append(int(candidate))
        if len(selected) >= maximum_orientations:
            break

    half_band = max(1, int(round(orientation_band_deg)))
    peak_mass: list[tuple[int, float]] = []
    for peak in selected:
        indices = [(peak + offset) % 180 for offset in range(-half_band, half_band + 1)]
        mass = float(histogram[indices].sum())
        peak_mass.append((peak, mass))
    unique_mass = np.zeros(180, dtype=np.bool_)
    for peak, _ in peak_mass:
        for offset in range(-half_band, half_band + 1):
            unique_mass[(peak + offset) % 180] = True
    concentration = min(1.0, float(histogram[unique_mass].sum() / histogram.sum()))
    orientations = [
        FourierOrientationPeak(angle_deg=float(peak), power_share=min(1.0, mass / histogram.sum()))
        for peak, mass in peak_mass
        if mass / histogram.sum() >= 0.03
    ]
    eligible = (
        bool(orientations)
        and concentration >= minimum_spectral_concentration
        and 0.0005 <= foreground_fraction <= 0.5
    )
    return FourierWallPrior(
        image_shape=(int(image.shape[0]), int(image.shape[1])),
        orientations=orientations,
        spectral_concentration=concentration,
        foreground_fraction=foreground_fraction,
        eligible_for_candidate_pruning=eligible,
    )


def wall_segment_angle_deg(start: Sequence[float], end: Sequence[float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0


def benchmark_fourier_wall_prior(
    ink: np.ndarray,
    *,
    warmup_runs: int = 10,
    measured_runs: int = 100,
) -> FourierBenchmark:
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be >= 0 and measured_runs must be >= 1")
    for _ in range(warmup_runs):
        detect_fourier_wall_prior(ink)
    samples = np.empty(measured_runs, dtype=np.int64)
    for index in range(measured_runs):
        started = time.perf_counter_ns()
        detect_fourier_wall_prior(ink)
        samples[index] = time.perf_counter_ns() - started
    return FourierBenchmark(
        image_shape=(int(ink.shape[0]), int(ink.shape[1])),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        median_ns=int(np.median(samples)),
        p95_ns=int(np.quantile(samples, 0.95)),
        minimum_ns=int(samples.min()),
    )
