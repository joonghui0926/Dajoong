"""Recover wall openings from the final whole-building wall graph.

Doors and windows are relationships, not free-standing crop classes.  This
module measures loss of ink along an accepted wall assembly and then inspects
the adjacent source pixels for a connected door leaf and swing.  The accepted
wall graph therefore owns position, span and orientation; a local semantic
model cannot place an opening away from its host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.model.aec_decode import PixelLineProposal, PixelSymbolProposal
from .core.model.cad_evidence import _ndimage, raster_ink
from .local_element_candidates import _otsu_threshold


class NativeOpeningDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.native-opening-candidates.v1-wall-gap-swing"
    wall_count: int = Field(ge=0)
    wall_assembly_count: int = Field(ge=0)
    orthogonal_wall_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    door_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
    unsupported_wall_count: int = Field(ge=0)


@dataclass(frozen=True)
class _WallAssembly:
    orientation: str
    axis_px: float
    start_px: float
    end_px: float
    thickness_px: float
    confidence: float
    source_ref_ids: tuple[str, ...]
    members: tuple[Any, ...]


def _wall_axis(wall: Any) -> tuple[str, float, float, float, float] | None:
    start_x, start_y = (float(value) for value in wall.start_px)
    end_x, end_y = (float(value) for value in wall.end_px)
    delta_x = abs(end_x - start_x)
    delta_y = abs(end_y - start_y)
    length = max(delta_x, delta_y)
    if length <= 1:
        return None
    # Raster projection is exact for orthogonal drafting.  Sloped walls are
    # retained in the wall graph but deliberately not guessed here.
    if min(delta_x, delta_y) > max(2.0, length * 0.035):
        return None
    thickness = max(2.0, float(wall.thickness_px or 4.0))
    if delta_x >= delta_y:
        return "horizontal", (start_y + end_y) / 2, min(start_x, end_x), max(
            start_x, end_x
        ), thickness
    return "vertical", (start_x + end_x) / 2, min(start_y, end_y), max(
        start_y, end_y
    ), thickness


def _assemble_collinear_walls(walls: list[Any], image_size: tuple[int, int]) -> tuple[
    list[_WallAssembly], int
]:
    width, height = image_size
    grouped: list[dict[str, Any]] = []
    unsupported = 0
    for wall in walls:
        axis = _wall_axis(wall)
        if axis is None:
            unsupported += 1
            continue
        orientation, center, start, end, thickness = axis
        match = next(
            (
                group
                for group in grouped
                if group["orientation"] == orientation
                and abs(group["axis_px"] - center)
                <= max(group["thickness_px"], thickness) * 0.75 + 2.0
            ),
            None,
        )
        if match is None:
            grouped.append(
                {
                    "orientation": orientation,
                    "axis_px": center,
                    "thickness_px": thickness,
                    "members": [(start, end, wall)],
                }
            )
        else:
            match["members"].append((start, end, wall))
            weights = [
                member_end - member_start
                for member_start, member_end, _ in match["members"]
            ]
            centers = [
                _wall_axis(member)[1]
                for _, _, member in match["members"]
                if _wall_axis(member) is not None
            ]
            match["axis_px"] = float(np.average(centers, weights=weights))
            match["thickness_px"] = float(
                np.median(
                    [
                        _wall_axis(member)[4]
                        for _, _, member in match["members"]
                        if _wall_axis(member) is not None
                    ]
                )
            )

    output: list[_WallAssembly] = []
    page_minor = min(width, height)
    for group in grouped:
        members = sorted(group["members"], key=lambda item: item[0])
        clusters: list[list[tuple[float, float, Any]]] = []
        for member in members:
            if not clusters:
                clusters.append([member])
                continue
            cluster_end = max(item[1] for item in clusters[-1])
            gap = member[0] - cluster_end
            maximum_opening_span = min(
                page_minor * 0.30,
                max(24.0, group["thickness_px"] * 9.0),
            )
            if gap <= maximum_opening_span:
                clusters[-1].append(member)
            else:
                clusters.append([member])
        for cluster in clusters:
            source_refs = sorted(
                {
                    source_ref
                    for _, _, wall in cluster
                    for source_ref in wall.source_ref_ids
                }
            )
            output.append(
                _WallAssembly(
                    orientation=group["orientation"],
                    axis_px=float(group["axis_px"]),
                    start_px=min(item[0] for item in cluster),
                    end_px=max(item[1] for item in cluster),
                    thickness_px=float(group["thickness_px"]),
                    confidence=float(min(item[2].confidence for item in cluster)),
                    source_ref_ids=tuple(source_refs),
                    members=tuple(item[2] for item in cluster),
                )
            )
    return output, unsupported


def consolidate_walls_across_openings(
    walls: list[PixelLineProposal],
    openings: list[PixelSymbolProposal],
    *,
    image_size: tuple[int, int],
) -> list[PixelLineProposal]:
    """Join collinear wall fragments only when hosted gaps explain the join."""

    assemblies, _ = _assemble_collinear_walls(walls, image_size)
    consumed: set[str] = set()
    joined: list[PixelLineProposal] = []
    for index, assembly in enumerate(assemblies):
        if len(assembly.members) < 2:
            continue
        member_intervals = []
        for member in assembly.members:
            axis = _wall_axis(member)
            if axis is not None:
                member_intervals.append((axis[2], axis[3]))
        member_intervals.sort()
        # A wide facade assembly can contain window + pier + door + pier +
        # window, while a learned wall fragment may end halfway through the
        # first/last opening. Requiring one opening to cover the entire fragment
        # gap therefore leaves the host split. It is sufficient that every gap
        # contains at least one independently recovered hosted opening; solid
        # ink between those openings is part of the same wall assembly.
        unexplained_gap = False
        for left, right in zip(member_intervals, member_intervals[1:], strict=False):
            gap_start, gap_end = left[1], right[0]
            if gap_end <= gap_start:
                continue
            explained = any(
                (
                    assembly.orientation == "horizontal"
                    and abs(opening.center_px[1] - assembly.axis_px)
                    <= assembly.thickness_px
                    and gap_start - 4 <= opening.center_px[0] <= gap_end + 4
                )
                or (
                    assembly.orientation == "vertical"
                    and abs(opening.center_px[0] - assembly.axis_px)
                    <= assembly.thickness_px
                    and gap_start - 4 <= opening.center_px[1] <= gap_end + 4
                )
                for opening in openings
            )
            if not explained:
                unexplained_gap = True
                break
        if unexplained_gap:
            continue
        if assembly.orientation == "horizontal":
            start_px = (assembly.start_px, assembly.axis_px)
            end_px = (assembly.end_px, assembly.axis_px)
        else:
            start_px = (assembly.axis_px, assembly.start_px)
            end_px = (assembly.axis_px, assembly.end_px)
        joined.append(
            PixelLineProposal(
                id=f"opening-host-wall:{index:04d}",
                start_px=start_px,
                end_px=end_px,
                thickness_px=assembly.thickness_px,
                confidence=assembly.confidence,
                uncertainty=1.0 - assembly.confidence,
                source_ref_ids=list(assembly.source_ref_ids),
                model_version=(
                    f"{assembly.members[0].model_version}"
                    "+opening-host-consolidation-v1"
                ),
                review_required=any(member.review_required for member in assembly.members),
            )
        )
        consumed.update(member.id for member in assembly.members)
    output = [wall for wall in walls if wall.id not in consumed]
    output.extend(joined)
    return sorted(
        output,
        key=lambda wall: (
            min(wall.start_px[1], wall.end_px[1]),
            min(wall.start_px[0], wall.end_px[0]),
            wall.id,
        ),
    )


def _intervals(mask: np.ndarray, *, offset: int) -> list[tuple[int, int]]:
    ndimage: Any = _ndimage()
    labels, _ = ndimage.label(mask)
    output: list[tuple[int, int]] = []
    for item in ndimage.find_objects(labels):
        if item is None:
            continue
        output.append((item[0].start + offset, item[0].stop + offset))
    return output


def _door_swing_score(
    foreground: np.ndarray,
    *,
    orientation: str,
    start: int,
    end: int,
    axis: float,
    thickness: float,
) -> float:
    """Measure a leaf/arc component connected to either end of a wall gap."""

    ndimage: Any = _ndimage()
    span = end - start
    if span < 3:
        return 0.0
    best = 0.0
    for side in (-1, 1):
        if orientation == "horizontal":
            near = axis - thickness / 2 if side < 0 else axis + thickness / 2
            far = near + side * span
            top, bottom = sorted((int(round(near)), int(round(far))))
            left, right = start - 3, end + 3
        else:
            near = axis - thickness / 2 if side < 0 else axis + thickness / 2
            far = near + side * span
            left, right = sorted((int(round(near)), int(round(far))))
            top, bottom = start - 3, end + 3
        left = max(0, left)
        top = max(0, top)
        right = min(foreground.shape[1], right)
        bottom = min(foreground.shape[0], bottom)
        if right - left < 3 or bottom - top < 3:
            continue
        crop = foreground[top:bottom, left:right]
        labels, _ = ndimage.label(crop, structure=np.ones((3, 3), dtype=np.uint8))
        if orientation == "horizontal":
            boundary = labels[-4:] if side < 0 else labels[:4]
            endpoint = min(12, span)
            touching = np.unique(
                boundary[:, np.r_[0:endpoint, max(0, crop.shape[1] - endpoint) : crop.shape[1]]]
            )
        else:
            boundary = labels[:, -4:] if side < 0 else labels[:, :4]
            endpoint = min(12, span)
            touching = np.unique(
                boundary[np.r_[0:endpoint, max(0, crop.shape[0] - endpoint) : crop.shape[0]], :]
            )
        sizes = np.bincount(labels.ravel())
        connected = max((int(sizes[index]) for index in touching if index), default=0)
        best = max(best, connected / max(1.0, float(span * span)))
    return best


def infer_openings_from_wall_graph(
    image: Image.Image,
    walls: list[Any],
    *,
    model_version: str,
) -> tuple[list[PixelSymbolProposal], NativeOpeningDiagnostics]:
    """Infer hosted openings from wall-band gaps and adjacent swing evidence."""

    ink = raster_ink(image)
    foreground = ink >= min(0.72, max(0.20, _otsu_threshold(ink)))
    assemblies, unsupported = _assemble_collinear_walls(walls, image.size)
    openings: list[PixelSymbolProposal] = []
    for assembly_index, assembly in enumerate(assemblies):
        thickness = assembly.thickness_px
        axis = assembly.axis_px
        start = max(0, int(round(assembly.start_px)))
        if assembly.orientation == "horizontal":
            end = min(image.width, int(round(assembly.end_px)) + 1)
            low = max(0, int(round(axis - thickness / 2)))
            high = min(image.height, int(round(axis + thickness / 2)) + 1)
            if high <= low or end <= start:
                continue
            profile = foreground[low:high, start:end].mean(axis=0)
        else:
            end = min(image.height, int(round(assembly.end_px)) + 1)
            low = max(0, int(round(axis - thickness / 2)))
            high = min(image.width, int(round(axis + thickness / 2)) + 1)
            if high <= low or end <= start:
                continue
            profile = foreground[start:end, low:high].mean(axis=1)
        if profile.size < 3:
            continue
        gap_threshold = min(0.55, max(0.18, _otsu_threshold(profile) + 0.08))
        gap_mask = profile < gap_threshold
        gap_mask = _ndimage().binary_opening(
            gap_mask,
            structure=np.ones(3, dtype=np.bool_),
        )
        minimum_span = max(7.0, thickness * 1.35)
        maximum_span = min(min(image.size) * 0.30, max(28.0, thickness * 12.0))
        support_span = max(3, int(round(thickness * 0.55)))
        for gap_start, gap_end in _intervals(gap_mask, offset=start):
            span = gap_end - gap_start
            if not minimum_span <= span <= maximum_span:
                continue
            local_start = gap_start - start
            local_end = gap_end - start
            before = profile[max(0, local_start - support_span) : local_start]
            after = profile[local_end : min(profile.size, local_end + support_span)]
            if before.size < 2 or after.size < 2:
                continue
            # An opening must interrupt otherwise substantive wall material on
            # both sides. This prevents empty page margins from becoming windows.
            if float(before.mean()) < gap_threshold or float(after.mean()) < gap_threshold:
                continue
            swing_score = _door_swing_score(
                foreground,
                orientation=assembly.orientation,
                start=gap_start,
                end=gap_end,
                axis=axis,
                thickness=thickness,
            )
            symbol_class = "door" if swing_score >= 0.035 else "window"
            if assembly.orientation == "horizontal":
                bbox = (
                    float(gap_start),
                    float(max(0.0, axis - thickness / 2)),
                    float(gap_end),
                    float(min(image.height, axis + thickness / 2)),
                )
            else:
                bbox = (
                    float(max(0.0, axis - thickness / 2)),
                    float(gap_start),
                    float(min(image.width, axis + thickness / 2)),
                    float(gap_end),
                )
            gap_support = float(1.0 - np.mean(profile[local_start:local_end]))
            confidence = float(
                np.clip(
                    0.55 * gap_support
                    + 0.25 * assembly.confidence
                    + (0.20 if symbol_class == "door" else 0.12),
                    0.0,
                    0.995,
                )
            )
            openings.append(
                PixelSymbolProposal(
                    id=f"native-opening:{assembly_index:04d}:{gap_start}:{gap_end}",
                    symbol_class=symbol_class,
                    center_px=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    bbox_px=bbox,
                    confidence=confidence,
                    uncertainty=1.0 - confidence,
                    source_ref_ids=list(assembly.source_ref_ids),
                    model_version=f"{model_version}+native-wall-gap-swing-v1",
                    review_required=True,
                )
            )
    # Assemblies cannot overlap along the same host axis after grouping, but a
    # thick-wall duplicate can survive the wall fusion step. Keep one exact gap.
    deduplicated: list[PixelSymbolProposal] = []
    for opening in sorted(openings, key=lambda item: item.confidence, reverse=True):
        if any(
            abs(opening.center_px[0] - kept.center_px[0]) <= 3
            and abs(opening.center_px[1] - kept.center_px[1]) <= 3
            and abs(
                (opening.bbox_px[2] - opening.bbox_px[0])
                - (kept.bbox_px[2] - kept.bbox_px[0])
            )
            <= 5
            and abs(
                (opening.bbox_px[3] - opening.bbox_px[1])
                - (kept.bbox_px[3] - kept.bbox_px[1])
            )
            <= 5
            for kept in deduplicated
        ):
            continue
        deduplicated.append(opening)
    deduplicated.sort(key=lambda item: (item.center_px[1], item.center_px[0]))
    return deduplicated, NativeOpeningDiagnostics(
        wall_count=len(walls),
        wall_assembly_count=len(assemblies),
        orthogonal_wall_count=len(walls) - unsupported,
        gap_count=len(deduplicated),
        door_count=sum(item.symbol_class == "door" for item in deduplicated),
        window_count=sum(item.symbol_class == "window" for item in deduplicated),
        unsupported_wall_count=unsupported,
    )
