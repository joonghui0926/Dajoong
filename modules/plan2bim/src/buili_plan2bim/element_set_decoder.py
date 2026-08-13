"""Building-aware set decoding for independently classified element proposals.

The local classifier intentionally over-covers native ink.  It is therefore not
allowed to emit the final BIM set by itself: several crops may describe one
physical item, fragments may sit inside a larger candidate, and wall-hosted
families require a structural host.  This module resolves those relationships
jointly and records every suppressed hypothesis for review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.model.aec_decode import PixelSymbolProposal


@dataclass(frozen=True)
class ElementSetDecision:
    candidate_id: str
    decision: str
    related_candidate_id: str | None = None


_WALL_HOSTED = {"door", "window"}
_CLASS_FAMILY = {
    "base_cabinet": "casework",
    "wall_cabinet": "casework",
    "closet": "casework",
    "coat_closet": "casework",
    "housing": "casework",
    "electrical_appliance": "appliance",
    "refrigerator": "appliance",
    "stove": "appliance",
    "dishwasher": "appliance",
    "washing_machine": "appliance",
    "tumble_dryer": "appliance",
    "toilet": "plumbing",
    "sink": "plumbing",
    "shower": "plumbing",
    "shower_screen": "plumbing",
    "bathtub": "plumbing",
    "water_tap": "plumbing",
    "jacuzzi": "plumbing",
}
_NESTED_COEXISTENCE = {
    frozenset(("sink", "water_tap")),
    frozenset(("shower", "water_tap")),
    frozenset(("bathtub", "water_tap")),
    frozenset(("shower", "shower_screen")),
}
_EQUIPMENT_RUN_FAMILIES = {"casework", "appliance", "plumbing"}
_SET_DEFERRED_MARKER = "+set-deferred-v1"


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = _intersection(left, right)
    return intersection / max(1e-6, _area(left) + _area(right) - intersection)


def _nested_relation_is_plausible(
    left: PixelSymbolProposal,
    right: PixelSymbolProposal,
    *,
    intersection: float,
    left_area: float,
    right_area: float,
) -> bool:
    class_pair = frozenset((left.symbol_class, right.symbol_class))
    if class_pair not in _NESTED_COEXISTENCE:
        return False
    smaller = min(left_area, right_area)
    larger = max(left_area, right_area)
    containment = intersection / max(1e-6, smaller)
    if containment < 0.70:
        return False
    # A tap is an accessory, not another full-size sink/shower hypothesis.
    if "water_tap" in class_pair:
        return smaller / max(1e-6, larger) <= 0.35
    # A screen can cross a shower footprint while remaining a long, thin item.
    screen = left if left.symbol_class == "shower_screen" else right
    width = max(1e-6, screen.bbox_px[2] - screen.bbox_px[0])
    height = max(1e-6, screen.bbox_px[3] - screen.bbox_px[1])
    return max(width, height) / min(width, height) >= 3.0


def _linear_duplicate(
    left: PixelSymbolProposal,
    right: PixelSymbolProposal,
    *,
    intersection: float,
) -> bool:
    left_width = max(1e-6, left.bbox_px[2] - left.bbox_px[0])
    left_height = max(1e-6, left.bbox_px[3] - left.bbox_px[1])
    right_width = max(1e-6, right.bbox_px[2] - right.bbox_px[0])
    right_height = max(1e-6, right.bbox_px[3] - right.bbox_px[1])
    left_horizontal = left_width >= left_height
    right_horizontal = right_width >= right_height
    if left_horizontal != right_horizontal:
        return False
    left_ratio = max(left_width, left_height) / min(left_width, left_height)
    right_ratio = max(right_width, right_height) / min(right_width, right_height)
    if min(left_ratio, right_ratio) < 3.0:
        return False
    left_major = left_width if left_horizontal else left_height
    right_major = right_width if right_horizontal else right_height
    left_minor = left_height if left_horizontal else left_width
    right_minor = right_height if right_horizontal else right_width
    overlap = intersection / max(
        1e-6,
        min(left_major, right_major) * min(left_minor, right_minor),
    )
    # Proposal fragments from the two ink edges of one thin symbol often get
    # different fine labels. Identity is geometric: a taxonomy disagreement
    # must not manufacture a second BIM object from the same linework.
    return overlap >= 0.65


def _run_neighbors(left: PixelSymbolProposal, right: PixelSymbolProposal) -> bool:
    left_family = _CLASS_FAMILY.get(left.symbol_class)
    right_family = _CLASS_FAMILY.get(right.symbol_class)
    if (
        left_family not in _EQUIPMENT_RUN_FAMILIES
        or right_family not in _EQUIPMENT_RUN_FAMILIES
    ):
        return False
    lx0, ly0, lx1, ly1 = left.bbox_px
    rx0, ry0, rx1, ry1 = right.bbox_px
    lw, lh = max(1e-6, lx1 - lx0), max(1e-6, ly1 - ly0)
    rw, rh = max(1e-6, rx1 - rx0), max(1e-6, ry1 - ry0)
    x_overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0)) / min(lw, rw)
    y_overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0)) / min(lh, rh)
    x_gap = max(0.0, max(lx0, rx0) - min(lx1, rx1))
    y_gap = max(0.0, max(ly0, ry0) - min(ly1, ry1))
    vertical_run = x_overlap >= 0.70 and y_gap <= max(18.0, 0.45 * max(lh, rh))
    horizontal_run = y_overlap >= 0.70 and x_gap <= max(18.0, 0.45 * max(lw, rw))
    return vertical_run or horizontal_run


def _normalize_equipment_run_semantics(
    proposals: list[PixelSymbolProposal],
) -> list[PixelSymbolProposal]:
    """Resolve generic casework labels from connected building assemblies."""

    adjacency = {index: set() for index in range(len(proposals))}
    for left_index, left in enumerate(proposals):
        for right_index in range(left_index + 1, len(proposals)):
            if _run_neighbors(left, proposals[right_index]):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    normalized = list(proposals)
    visited: set[int] = set()
    for start in range(len(proposals)):
        if start in visited:
            continue
        component: set[int] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(adjacency[current] - component)
        visited.update(component)
        if len(component) < 3 or not any(
            proposals[index].symbol_class == "base_cabinet" for index in component
        ):
            continue
        for index in component:
            proposal = proposals[index]
            if proposal.symbol_class != "housing":
                continue
            normalized[index] = proposal.model_copy(
                update={
                    "symbol_class": "base_cabinet",
                    "model_version": f"{proposal.model_version}+equipment-run-set-v1",
                    "review_required": True,
                }
            )
    return normalized


def _deferred_candidate_has_set_support(
    proposal: PixelSymbolProposal,
    anchors: list[PixelSymbolProposal],
) -> bool:
    if _SET_DEFERRED_MARKER not in proposal.model_version:
        return True
    neighbors = [anchor for anchor in anchors if _run_neighbors(proposal, anchor)]
    # Require evidence on both sides of a missing module when possible.  Two
    # independently accepted neighbors are a building-level relation; one is
    # too easily produced by text or a dimension fragment near an object.
    return len(neighbors) >= 2


def _instance_components(
    proposals: list[PixelSymbolProposal],
) -> list[list[PixelSymbolProposal]]:
    adjacency = {index: set() for index in range(len(proposals))}
    for left_index, left in enumerate(proposals):
        for right_index in range(left_index + 1, len(proposals)):
            if _same_physical_instance(left, proposals[right_index]):
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    output: list[list[PixelSymbolProposal]] = []
    visited: set[int] = set()
    for start in range(len(proposals)):
        if start in visited:
            continue
        component_indices: set[int] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop()
            if current in component_indices:
                continue
            component_indices.add(current)
            frontier.extend(adjacency[current] - component_indices)
        visited.update(component_indices)
        output.append([proposals[index] for index in sorted(component_indices)])
    return output


def _component_representative(
    component: list[PixelSymbolProposal],
) -> PixelSymbolProposal:
    if len(component) == 1:
        return component[0]
    areas = np.asarray([max(1e-6, _area(item.bbox_px)) for item in component])
    median_area = float(np.median(areas))
    linear_component = all(
        max(
            item.bbox_px[2] - item.bbox_px[0],
            item.bbox_px[3] - item.bbox_px[1],
        )
        / max(
            1e-6,
            min(
                item.bbox_px[2] - item.bbox_px[0],
                item.bbox_px[3] - item.bbox_px[1],
            ),
        )
        >= 3.0
        for item in component
    )

    def score(item: PixelSymbolProposal) -> tuple[float, float, str]:
        agreement = sum(
            _iou(item.bbox_px, other.bbox_px)
            for other in component
            if other.id != item.id
        ) / max(1, len(component) - 1)
        area = max(1e-6, _area(item.bbox_px))
        extent_penalty = (
            0.08 * area / max(1e-6, median_area)
            if linear_component
            else 0.20 * abs(math.log(area / max(1e-6, median_area)))
        )
        value = (
            item.confidence
            - item.uncertainty * 0.25
            + agreement * 0.65
            - extent_penalty
        )
        return value, -area, item.id

    return max(component, key=score)


def _same_physical_instance(
    left: PixelSymbolProposal,
    right: PixelSymbolProposal,
) -> bool:
    intersection = _intersection(left.bbox_px, right.bbox_px)
    if intersection <= 0:
        return False
    left_area = max(1e-6, _area(left.bbox_px))
    right_area = max(1e-6, _area(right.bbox_px))
    union = left_area + right_area - intersection
    iou = intersection / max(1e-6, union)
    containment = intersection / min(left_area, right_area)
    left_width = max(1e-6, left.bbox_px[2] - left.bbox_px[0])
    left_height = max(1e-6, left.bbox_px[3] - left.bbox_px[1])
    right_width = max(1e-6, right.bbox_px[2] - right.bbox_px[0])
    right_height = max(1e-6, right.bbox_px[3] - right.bbox_px[1])
    size_ratio = max(
        left_width / right_width,
        right_width / left_width,
        left_height / right_height,
        right_height / left_height,
    )
    center_distance = float(
        np.hypot(
            (left.center_px[0] - right.center_px[0])
            / max(left_width, right_width),
            (left.center_px[1] - right.center_px[1])
            / max(left_height, right_height),
        )
    )
    same_class = left.symbol_class == right.symbol_class
    same_family = (
        _CLASS_FAMILY.get(left.symbol_class) is not None
        and _CLASS_FAMILY.get(left.symbol_class) == _CLASS_FAMILY.get(right.symbol_class)
    )
    nested_coexistence = _nested_relation_is_plausible(
        left,
        right,
        intersection=intersection,
        left_area=left_area,
        right_area=right_area,
    )
    # Near-identical geometry cannot describe two physical instances even when
    # the conditional taxonomy disagrees.  Nested plumbing accessories are a
    # separate relation and intentionally survive containment suppression.
    if _linear_duplicate(left, right, intersection=intersection):
        return True
    if iou >= 0.58 and not nested_coexistence:
        return True
    if containment >= 0.90 and size_ratio <= 2.0 and not nested_coexistence:
        return True
    if (
        same_family
        and not nested_coexistence
        and containment >= 0.85
        and size_ratio <= 4.0
    ):
        return True
    if not same_class:
        return False
    return (
        iou >= 0.42
        or (containment >= 0.76 and size_ratio <= 3.0)
        or (center_distance <= 0.20 and size_ratio <= 2.1)
    )


def _has_wall_host(proposal: PixelSymbolProposal, walls: list[Any]) -> bool:
    if proposal.symbol_class not in _WALL_HOSTED:
        return True
    if not walls:
        return False
    left, top, right, bottom = proposal.bbox_px
    horizontal = (right - left) >= (bottom - top)
    center = np.asarray(proposal.center_px, dtype=np.float64)
    for wall in walls:
        start = np.asarray(wall.start_px, dtype=np.float64)
        end = np.asarray(wall.end_px, dtype=np.float64)
        vector = end - start
        squared_length = float(np.dot(vector, vector))
        if squared_length <= 1e-9:
            continue
        if (abs(vector[0]) >= abs(vector[1])) != horizontal:
            continue
        fraction = float(np.clip(np.dot(center - start, vector) / squared_length, 0, 1))
        distance = float(np.linalg.norm(center - (start + fraction * vector)))
        if distance <= max(5.0, float(wall.thickness_px or 4.0) * 1.5):
            return True
    return False


def decode_element_set(
    proposals: list[PixelSymbolProposal],
    *,
    host_walls: list[Any],
) -> tuple[list[PixelSymbolProposal], list[ElementSetDecision]]:
    """Resolve the final proposal set with relation constraints.

    This is deliberately conservative.  It removes only same-class identity
    conflicts and wall-host violations; adjacent cabinet cells of the same type
    remain separate because they do not overlap.  Ambiguity is retained through
    ``review_required`` instead of being silently forced into the BIM.
    """

    proposals = _normalize_equipment_run_semantics(proposals)
    anchors = [
        item for item in proposals if _SET_DEFERRED_MARKER not in item.model_version
    ]
    set_supported = [
        item
        for item in proposals
        if _deferred_candidate_has_set_support(item, anchors)
    ]
    set_unsupported = [item for item in proposals if item not in set_supported]
    host_valid = [
        item for item in set_supported if _has_wall_host(item, host_walls)
    ]
    host_invalid = [item for item in set_supported if item not in host_valid]
    selected: list[PixelSymbolProposal] = []
    decisions: list[ElementSetDecision] = []
    decisions.extend(
        ElementSetDecision(proposal.id, "insufficient_set_support")
        for proposal in set_unsupported
    )
    decisions.extend(
        ElementSetDecision(proposal.id, "wall_host_missing")
        for proposal in host_invalid
    )
    for component in _instance_components(host_valid):
        representative = _component_representative(component)
        selected.append(representative)
        decisions.append(ElementSetDecision(representative.id, "selected"))
        for proposal in component:
            if proposal.id == representative.id:
                continue
            decisions.append(
                ElementSetDecision(
                    proposal.id,
                    "same_instance_suppressed",
                    representative.id,
                )
            )
    return (
        sorted(selected, key=lambda item: (item.center_px[1], item.center_px[0], item.id)),
        decisions,
    )
