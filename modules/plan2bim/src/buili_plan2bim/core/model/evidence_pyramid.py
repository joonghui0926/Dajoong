"""Information-preserving full-sheet routing for high-resolution AEC drawings.

The router is deliberately not a saliency sampler.  It assigns every foreground
pixel to exactly one high-resolution *core* and gives that core a context-expanded
inference tile.  A tile budget can therefore block work, but it can never silently
drop a small symbol.  Full-sheet oriented/enclosure evidence is computed once and
cropped into every tile so local inference retains sheet-scale topology.

This module contains no hosted model call and has no training-time dependency.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..hashing import sha256_json
from .aec_decode import AecTileProposal, PixelLineProposal, PixelSymbolProposal
from .cad_evidence import (
    GLOBAL_ORIENTED_EVIDENCE_CONTRACT,
    build_cad_evidence,
    raster_ink,
)
from .evidence_coverage import (
    CoverageConfig,
    EvidenceCoverageCertificate,
    certify_evidence_coverage,
)

PYRAMID_VERSION = "dajoong-proof-carrying-evidence-pyramid-0.1"
LedgerIssue = Literal[
    "empty_sheet",
    "tile_budget_exceeded",
    "uncovered_foreground",
]


class EvidencePyramidConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_size: int = Field(default=256, ge=64, le=2048)
    context_margin: int = Field(default=32, ge=0, le=512)
    foreground_threshold: float = Field(default=0.20, ge=0, le=1)
    maximum_tiles: int = Field(default=4096, ge=1)
    minimum_foreground_pixels: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_core(self) -> EvidencePyramidConfig:
        if self.context_margin * 2 >= self.tile_size:
            raise ValueError("context_margin must leave a positive tile core")
        return self

    @property
    def core_size(self) -> int:
        return self.tile_size - 2 * self.context_margin


class EvidenceTile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_id: str
    tile_size_px: int = Field(ge=64)
    bbox_px: tuple[int, int, int, int]
    core_bbox_px: tuple[int, int, int, int]
    owned_foreground_pixels: int = Field(ge=1)
    context_foreground_pixels: int = Field(ge=1)
    source_ref_ids: list[str] = Field(min_length=1)


class EvidenceTileLedger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.evidence-tile-ledger.v1"
    router_version: str = PYRAMID_VERSION
    sheet_id: str
    source_ref_ids: list[str] = Field(min_length=1)
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)
    config: EvidencePyramidConfig
    foreground_pixels: int = Field(ge=0)
    owned_foreground_pixels: int = Field(ge=0)
    foreground_coverage_ratio: float = Field(ge=0, le=1)
    dense_grid_tiles: int = Field(ge=1)
    selected_tiles: int = Field(ge=0)
    selected_tile_fraction: float = Field(ge=0, le=1)
    required_tile_pixels: int = Field(ge=0)
    tiles: list[EvidenceTile]
    issues: list[LedgerIssue]
    release_allowed: bool
    content_sha256: str = ""

    def finalize(self) -> EvidenceTileLedger:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


class FullSheetAecResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.full-sheet-aec-result.v1"
    sheet_id: str
    model_version: str
    model_artifact_sha256: str = ""
    model_release_authorized: bool = False
    ledger: EvidenceTileLedger
    proposal: AecTileProposal | None
    coverage: EvidenceCoverageCertificate | None
    release_allowed: bool
    review_reasons: list[str]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    content_sha256: str = ""

    def finalize(self) -> FullSheetAecResult:
        payload = self.model_dump(mode="json", exclude={"content_sha256", "timings_ms"})
        self.content_sha256 = sha256_json(payload)
        return self


def _foreground_mask(
    image: Image.Image,
    *,
    threshold: float,
    active_mask: np.ndarray | None,
) -> np.ndarray:
    foreground = raster_ink(image) >= threshold
    if active_mask is not None:
        active = np.asarray(active_mask, dtype=np.bool_)
        if active.shape != foreground.shape:
            raise ValueError("active_mask must match the source image")
        foreground &= active
    return foreground


def _context_origin(core_origin: int, *, extent: int, config: EvidencePyramidConfig) -> int:
    return min(
        max(0, core_origin - config.context_margin),
        max(0, extent - config.tile_size),
    )


def build_evidence_tile_ledger(
    image: Image.Image,
    *,
    sheet_id: str,
    source_ref_ids: list[str],
    config: EvidencePyramidConfig | None = None,
    active_mask: np.ndarray | None = None,
) -> EvidenceTileLedger:
    """Build an exhaustive high-resolution tile ledger without saliency loss."""

    config = config or EvidencePyramidConfig()
    if not sheet_id:
        raise ValueError("sheet_id cannot be empty")
    if not source_ref_ids:
        raise ValueError("source_ref_ids cannot be empty")
    width, height = image.size
    foreground = _foreground_mask(
        image,
        threshold=config.foreground_threshold,
        active_mask=active_mask,
    )
    core = config.core_size
    columns = max(1, math.ceil(width / core))
    rows = max(1, math.ceil(height / core))
    ownership = np.zeros_like(foreground, dtype=np.bool_)
    tiles: list[EvidenceTile] = []
    for row in range(rows):
        core_top = row * core
        core_bottom = min(height, core_top + core)
        for column in range(columns):
            core_left = column * core
            core_right = min(width, core_left + core)
            owned = foreground[core_top:core_bottom, core_left:core_right]
            owned_pixels = int(owned.sum())
            if owned_pixels < config.minimum_foreground_pixels:
                continue
            left = _context_origin(core_left, extent=width, config=config)
            top = _context_origin(core_top, extent=height, config=config)
            right = min(width, left + config.tile_size)
            bottom = min(height, top + config.tile_size)
            ownership[core_top:core_bottom, core_left:core_right] |= owned
            context_pixels = int(foreground[top:bottom, left:right].sum())
            tiles.append(
                EvidenceTile(
                    tile_id=f"{sheet_id}:L0:R{row:04d}:C{column:04d}",
                    tile_size_px=config.tile_size,
                    bbox_px=(left, top, right, bottom),
                    core_bbox_px=(core_left, core_top, core_right, core_bottom),
                    owned_foreground_pixels=owned_pixels,
                    context_foreground_pixels=context_pixels,
                    source_ref_ids=source_ref_ids,
                )
            )
    total_foreground = int(foreground.sum())
    owned_foreground = int((foreground & ownership).sum())
    coverage = 1.0 if total_foreground == 0 else owned_foreground / total_foreground
    issues: list[LedgerIssue] = []
    if total_foreground == 0:
        issues.append("empty_sheet")
    if len(tiles) > config.maximum_tiles:
        issues.append("tile_budget_exceeded")
    if owned_foreground != total_foreground:
        issues.append("uncovered_foreground")
    # Required tiles are intentionally retained even when the budget is exceeded.
    # A caller can raise the budget or abstain; it cannot receive a silently pruned list.
    return EvidenceTileLedger(
        sheet_id=sheet_id,
        source_ref_ids=source_ref_ids,
        width_px=width,
        height_px=height,
        config=config,
        foreground_pixels=total_foreground,
        owned_foreground_pixels=owned_foreground,
        foreground_coverage_ratio=coverage,
        dense_grid_tiles=rows * columns,
        selected_tiles=len(tiles),
        selected_tile_fraction=len(tiles) / (rows * columns),
        required_tile_pixels=len(tiles) * config.tile_size * config.tile_size,
        tiles=tiles,
        issues=issues,
        release_allowed=not issues,
    ).finalize()


def crop_evidence_tile(full_evidence: np.ndarray, tile: EvidenceTile) -> np.ndarray:
    """Return a fixed-size padded evidence tensor for one ledger tile."""

    evidence = np.asarray(full_evidence, dtype=np.float32)
    if evidence.ndim != 3:
        raise ValueError("full_evidence must have shape [channels, height, width]")
    left, top, right, bottom = tile.bbox_px
    tile_size = tile.tile_size_px
    output = np.zeros((evidence.shape[0], tile_size, tile_size), dtype=np.float32)
    crop = evidence[:, top:bottom, left:right]
    output[:, : crop.shape[1], : crop.shape[2]] = crop
    return output


def _translate_line(line: PixelLineProposal, dx: int, dy: int) -> PixelLineProposal:
    return line.model_copy(
        update={
            "start_px": (line.start_px[0] + dx, line.start_px[1] + dy),
            "end_px": (line.end_px[0] + dx, line.end_px[1] + dy),
        }
    )


def _translate_symbol(symbol: PixelSymbolProposal, dx: int, dy: int) -> PixelSymbolProposal:
    left, top, right, bottom = symbol.bbox_px
    return symbol.model_copy(
        update={
            "center_px": (symbol.center_px[0] + dx, symbol.center_px[1] + dy),
            "bbox_px": (left + dx, top + dy, right + dx, bottom + dy),
        }
    )


def _line_distance(left: PixelLineProposal, right: PixelLineProposal) -> float:
    direct = max(
        math.dist(left.start_px, right.start_px),
        math.dist(left.end_px, right.end_px),
    )
    reversed_distance = max(
        math.dist(left.start_px, right.end_px),
        math.dist(left.end_px, right.start_px),
    )
    return min(direct, reversed_distance)


def _canonical_start(line: PixelLineProposal) -> tuple[float, float]:
    return min(line.start_px, line.end_px)


def _bbox_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _bbox_cells(
    box: tuple[float, ...],
    *,
    cell_size: float = 64.0,
) -> list[tuple[int, int]]:
    right = max(box[0], box[2] - 1e-9)
    bottom = max(box[1], box[3] - 1e-9)
    return [
        (cell_x, cell_y)
        for cell_y in range(math.floor(box[1] / cell_size), math.floor(bottom / cell_size) + 1)
        for cell_x in range(math.floor(box[0] / cell_size), math.floor(right / cell_size) + 1)
    ]


def _core_owns_point(
    point: tuple[float, float],
    core: tuple[int, int, int, int],
    *,
    sheet_width: int,
    sheet_height: int,
) -> bool:
    """Return whether a half-open tile core owns a global prediction center.

    Context margins exist only to improve inference.  Accepting predictions from
    those margins duplicates and fragments entities across adjacent tiles.
    """

    left, top, right, bottom = core
    x, y = point
    owns_x = left <= x < right or (right == sheet_width and x == right)
    owns_y = top <= y < bottom or (bottom == sheet_height and y == bottom)
    return owns_x and owns_y


def merge_tile_proposals(
    ledger: EvidenceTileLedger,
    proposals: list[AecTileProposal],
    *,
    line_tolerance_px: float = 3.0,
    symbol_iou_threshold: float = 0.5,
) -> AecTileProposal:
    """Translate overlapping local proposals and deterministically de-duplicate them."""

    proposal_by_tile = {proposal.tile_id: proposal for proposal in proposals}
    expected_ids = {tile.tile_id for tile in ledger.tiles}
    unknown = set(proposal_by_tile) - expected_ids
    if unknown:
        raise ValueError(f"proposals reference unknown ledger tiles: {sorted(unknown)}")
    missing = expected_ids - set(proposal_by_tile)
    if missing:
        raise ValueError(f"missing inference proposals for ledger tiles: {sorted(missing)}")
    model_versions = {proposal.model_version for proposal in proposals}
    if len(model_versions) != 1:
        raise ValueError("all tile proposals must use one immutable model version")

    walls: list[PixelLineProposal] = []
    wall_buckets: dict[tuple[int, int], list[int]] = {}
    symbols: list[PixelSymbolProposal] = []
    symbol_buckets: dict[tuple[str, int, int], list[int]] = {}
    rejected = 0
    for tile in ledger.tiles:
        proposal = proposal_by_tile[tile.tile_id]
        left, top, _, _ = tile.bbox_px
        width = tile.bbox_px[2] - left
        height = tile.bbox_px[3] - top
        if not set(ledger.source_ref_ids).issubset(proposal.source_ref_ids):
            raise ValueError(f"tile {tile.tile_id} lost required source references")
        rejected += proposal.rejected_candidates
        for local_line in proposal.wall_segments:
            if local_line.model_version != proposal.model_version:
                raise ValueError(f"tile {tile.tile_id} contains mixed model versions")
            if any(
                point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height
                for point in (local_line.start_px, local_line.end_px)
            ):
                raise ValueError(f"tile {tile.tile_id} contains an out-of-bounds wall")
            candidate = _translate_line(local_line, left, top)
            midpoint = (
                (candidate.start_px[0] + candidate.end_px[0]) / 2,
                (candidate.start_px[1] + candidate.end_px[1]) / 2,
            )
            if not _core_owns_point(
                midpoint,
                tile.core_bbox_px,
                sheet_width=ledger.width_px,
                sheet_height=ledger.height_px,
            ):
                continue
            bucket_scale = max(line_tolerance_px, 1e-6)
            canonical = _canonical_start(candidate)
            bucket = (
                math.floor(canonical[0] / bucket_scale),
                math.floor(canonical[1] / bucket_scale),
            )
            nearby_indices = {
                index
                for delta_x in (-1, 0, 1)
                for delta_y in (-1, 0, 1)
                for index in wall_buckets.get((bucket[0] + delta_x, bucket[1] + delta_y), [])
            }
            duplicate_index = next(
                (
                    index
                    for index in sorted(nearby_indices)
                    if _line_distance(candidate, walls[index]) <= line_tolerance_px
                ),
                None,
            )
            if duplicate_index is None:
                walls.append(candidate)
                wall_buckets.setdefault(bucket, []).append(len(walls) - 1)
            elif candidate.confidence > walls[duplicate_index].confidence:
                walls[duplicate_index] = candidate
                wall_buckets.setdefault(bucket, []).append(duplicate_index)
        for local_symbol in proposal.symbols:
            if local_symbol.model_version != proposal.model_version:
                raise ValueError(f"tile {tile.tile_id} contains mixed model versions")
            box = local_symbol.bbox_px
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                raise ValueError(f"tile {tile.tile_id} contains an out-of-bounds symbol")
            candidate = _translate_symbol(local_symbol, left, top)
            if not _core_owns_point(
                candidate.center_px,
                tile.core_bbox_px,
                sheet_width=ledger.width_px,
                sheet_height=ledger.height_px,
            ):
                continue
            cells = _bbox_cells(candidate.bbox_px)
            nearby_indices = {
                index
                for cell_x, cell_y in cells
                for index in symbol_buckets.get((candidate.symbol_class, cell_x, cell_y), [])
            }
            duplicate_index = next(
                (
                    index
                    for index in sorted(nearby_indices)
                    if _bbox_iou(candidate.bbox_px, symbols[index].bbox_px) >= symbol_iou_threshold
                ),
                None,
            )
            if duplicate_index is None:
                symbols.append(candidate)
                for cell_x, cell_y in cells:
                    symbol_buckets.setdefault((candidate.symbol_class, cell_x, cell_y), []).append(
                        len(symbols) - 1
                    )
            elif candidate.confidence > symbols[duplicate_index].confidence:
                symbols[duplicate_index] = candidate
                for cell_x, cell_y in cells:
                    symbol_buckets.setdefault((candidate.symbol_class, cell_x, cell_y), []).append(
                        duplicate_index
                    )
    # Stable global identifiers make reruns content-addressable even when overlap chooses
    # a proposal that originated in a different local tile.
    walls = [
        item.model_copy(update={"id": f"{ledger.sheet_id}:wall:{i}"})
        for i, item in enumerate(walls)
    ]
    symbols = [
        item.model_copy(update={"id": f"{ledger.sheet_id}:symbol:{i}"})
        for i, item in enumerate(symbols)
    ]
    return AecTileProposal(
        tile_id=ledger.sheet_id,
        source_ref_ids=ledger.source_ref_ids,
        model_version=model_versions.pop() if model_versions else "",
        wall_segments=walls,
        symbols=symbols,
        rejected_candidates=rejected,
    ).finalize()


TileInference = Callable[[np.ndarray, EvidenceTile], AecTileProposal]
BatchTileInference = Callable[[np.ndarray, list[EvidenceTile]], list[AecTileProposal]]


def run_information_preserving_aec(
    image: Image.Image,
    *,
    sheet_id: str,
    source_ref_ids: list[str],
    infer_tile: TileInference | None = None,
    infer_batch: BatchTileInference | None = None,
    inference_batch_size: int = 8,
    pyramid_config: EvidencePyramidConfig | None = None,
    coverage_config: CoverageConfig | None = None,
    active_mask: np.ndarray | None = None,
    known_text_mask: np.ndarray | None = None,
    known_dimension_mask: np.ndarray | None = None,
    known_hatch_mask: np.ndarray | None = None,
    model_artifact_sha256: str = "",
    model_release_authorized: bool = False,
) -> FullSheetAecResult:
    """Run the complete proof-carrying tile path and fail closed on any omission."""

    total_started = time.perf_counter()
    if (infer_tile is None) == (infer_batch is None):
        raise ValueError("provide exactly one of infer_tile or infer_batch")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")

    stage_started = time.perf_counter()
    ledger = build_evidence_tile_ledger(
        image,
        sheet_id=sheet_id,
        source_ref_ids=source_ref_ids,
        config=pyramid_config,
        active_mask=active_mask,
    )
    timings = {"tile_ledger": (time.perf_counter() - stage_started) * 1000}
    if not ledger.release_allowed:
        timings["total"] = (time.perf_counter() - total_started) * 1000
        return FullSheetAecResult(
            sheet_id=sheet_id,
            model_version="not-run",
            model_artifact_sha256=model_artifact_sha256,
            model_release_authorized=model_release_authorized,
            ledger=ledger,
            proposal=None,
            coverage=None,
            release_allowed=False,
            review_reasons=list(ledger.issues),
            timings_ms=timings,
        ).finalize()
    stage_started = time.perf_counter()
    full_evidence = build_cad_evidence(
        image,
        contract=GLOBAL_ORIENTED_EVIDENCE_CONTRACT,
    )
    timings["full_sheet_evidence"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    if infer_batch is None:
        assert infer_tile is not None
        local = [infer_tile(crop_evidence_tile(full_evidence, tile), tile) for tile in ledger.tiles]
    else:
        local = []
        for offset in range(0, len(ledger.tiles), inference_batch_size):
            tiles = ledger.tiles[offset : offset + inference_batch_size]
            evidence_batch = np.stack([crop_evidence_tile(full_evidence, tile) for tile in tiles])
            batch_output = infer_batch(evidence_batch, tiles)
            if len(batch_output) != len(tiles):
                raise ValueError("batch inference did not return one proposal per ledger tile")
            local.extend(batch_output)
    timings["specialist_inference_and_decode"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    proposal = merge_tile_proposals(ledger, local)
    timings["tile_merge"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    coverage = certify_evidence_coverage(
        full_evidence[0],
        proposal,
        known_text_mask=known_text_mask,
        known_dimension_mask=known_dimension_mask,
        known_hatch_mask=known_hatch_mask,
        active_mask=active_mask,
        config=coverage_config,
    )
    timings["reprojection_coverage"] = (time.perf_counter() - stage_started) * 1000
    reasons = [] if coverage.release_allowed else ["unexplained_reprojection_residual"]
    if not model_release_authorized:
        reasons.append("unpromoted_model")
    timings["total"] = (time.perf_counter() - total_started) * 1000
    return FullSheetAecResult(
        sheet_id=sheet_id,
        model_version=proposal.model_version,
        model_artifact_sha256=model_artifact_sha256,
        model_release_authorized=model_release_authorized,
        ledger=ledger,
        proposal=proposal,
        coverage=coverage,
        release_allowed=(
            ledger.release_allowed and coverage.release_allowed and model_release_authorized
        ),
        review_reasons=reasons,
        timings_ms=timings,
    ).finalize()
