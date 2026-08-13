"""Dajoong-owned procedural supervision for topology pretraining only.

Generated compiler programs are useful for learning geometric priors, but they
are never real-drawing ground truth and must never enter benchmark reports.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import sha256_file, sha256_json
from .core.model.global_topology_student import ELEMENT_PROGRAM_CLASSES

Point = tuple[float, float]
Box = tuple[float, float, float, float]
Style = Literal["cad", "scan", "markup", "colored_cad", "layered_cad"]
CanvasProfile = Literal["square", "portrait", "landscape"]
LayoutProfile = Literal[
    "recursive_partition",
    "double_loaded_corridor",
    "l_shaped_building",
]
_SYNTHETIC_ROLE = "synthetic_pretrain_only"
_GENERATOR_VERSION = "dajoong-inverse-compiler-generator-v10-host-aware"
FIXTURE_HOSTING_CONTRACT = (
    "architectural_hosted_classes_snap_to_room_boundary_v1"
)


class SyntheticRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    polygon_px: list[Point] = Field(min_length=4)
    room_class: str


class SyntheticOpening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["door", "window"]
    center_px: Point
    width_px: float = Field(gt=0)
    orientation: Literal["horizontal", "vertical"]


class SyntheticFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    fixture_type: str
    bbox_px: Box
    yaw_deg: float = 0.0


class SyntheticPretrainingSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.synthetic-pretraining-sample.v1"
    sample_id: str
    role: Literal["synthetic_pretrain_only"] = _SYNTHETIC_ROLE
    real_drawing_ground_truth: Literal[False] = False
    evaluation_eligible: Literal[False] = False
    generator_version: str = _GENERATOR_VERSION
    fixture_hosting_contract: str = FIXTURE_HOSTING_CONTRACT
    seed: int
    style: Style
    canvas_profile: CanvasProfile
    image_size_px: tuple[int, int] | None = None
    layout_profile: LayoutProfile
    image_path: str
    image_sha256: str
    source_program_owner: str = "Dajoong"
    building_footprint_px: list[Point] = Field(min_length=4)
    rooms: list[SyntheticRoom]
    walls: list[tuple[Point, Point]]
    openings: list[SyntheticOpening]
    fixtures: list[SyntheticFixture]
    content_sha256: str = ""

    def finalize(self) -> SyntheticPretrainingSample:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def assert_synthetic_pretraining_only(payload: dict[str, object]) -> None:
    """Fail closed if generated supervision could be mistaken for real GT."""

    expected = {
        "role": _SYNTHETIC_ROLE,
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"synthetic supervision cannot enter ground-truth or evaluation paths: {mismatches}"
        )


@dataclass(frozen=True)
class _Partition:
    box: Box
    depth: int


_ROOM_CLASSES = (
    "living",
    "bedroom",
    "kitchen",
    "bathroom",
    "storage",
    "office",
    "mechanical",
    "garage",
    "utility",
    "outdoor",
    "other",
)
_FIXTURES_BY_ROOM = {
    "living": (
        "sofa",
        "armchair",
        "chair",
        "coffee_table",
        "dining_table",
        "base_cabinet",
        "fireplace",
        "fireplace_corner",
        "place_for_fireplace",
        "place_for_fireplace_corner",
        "wood_stove",
        "electrical_appliance",
        "light",
    ),
    "bedroom": (
        "bed",
        "chair",
        "desk",
        "closet",
        "coat_closet",
        "coat_rack",
        "electrical_appliance",
        "light",
    ),
    "kitchen": (
        "dining_table",
        "chair",
        "refrigerator",
        "stove",
        "dishwasher",
        "base_cabinet",
        "wall_cabinet",
        "electrical_appliance",
        "sink",
        "column",
    ),
    "bathroom": (
        "toilet",
        "sink",
        "shower",
        "shower_screen",
        "bathtub",
        "jacuzzi",
        "water_tap",
    ),
    "storage": ("closet", "coat_closet", "coat_rack", "wall_cabinet"),
    "office": ("desk", "chair", "armchair", "base_cabinet", "electrical_appliance", "light"),
    "mechanical": (
        "electrical_panel",
        "hvac_terminal",
        "riser",
        "sprinkler",
    ),
    "garage": ("electrical_appliance", "column", "light", "receptacle"),
    "utility": (
        "sink",
        "washing_machine",
        "tumble_dryer",
        "plumbing_fixture",
        "riser",
        "wall_cabinet",
    ),
    "outdoor": ("bench", "chair", "sauna_bench", "column", "light"),
    "other": ("housing", "misc"),
    "corridor": ("column", "light", "receptacle", "sprinkler"),
}
_GENERAL_FIXTURE_TYPES = (
    "base_cabinet",
    "wall_cabinet",
    "closet",
    "coat_closet",
    "electrical_appliance",
    "toilet",
    "sink",
    "shower",
    "shower_screen",
    "bathtub",
    "sauna_bench",
    "fireplace",
    "chimney",
    "column",
    "stair",
    "light",
    "electrical_panel",
    "receptacle",
    "hvac_terminal",
    "sprinkler",
    "riser",
    "plumbing_fixture",
    "housing",
    "coat_rack",
    "water_tap",
    "jacuzzi",
    "wood_stove",
    "fireplace_corner",
    "place_for_fireplace",
    "place_for_fireplace_corner",
    "misc",
    "bed",
    "sofa",
    "armchair",
    "chair",
    "dining_table",
    "coffee_table",
    "desk",
    "bench",
    "refrigerator",
    "stove",
    "dishwasher",
    "washing_machine",
    "tumble_dryer",
)

_WALL_HOSTED_FIXTURE_TYPES = {
    "base_cabinet",
    "wall_cabinet",
    "closet",
    "coat_closet",
    "refrigerator",
    "stove",
    "dishwasher",
    "washing_machine",
    "tumble_dryer",
    "sink",
    "shower",
    "shower_screen",
    "bathtub",
    "electrical_panel",
    "hvac_terminal",
    "riser",
    "receptacle",
    "fireplace",
    "fireplace_corner",
    "place_for_fireplace",
    "place_for_fireplace_corner",
    "coat_rack",
}

_CURRICULUM_ONLY_EXCLUSIONS = {"background", "door", "window", "unknown"}
if set(_GENERAL_FIXTURE_TYPES) != (
    set(ELEMENT_PROGRAM_CLASSES) - _CURRICULUM_ONLY_EXCLUSIONS
):
    raise RuntimeError("synthetic fixture curriculum does not cover the model taxonomy")


def _split_partitions(rng: random.Random, *, bounds: Box, room_count: int) -> list[Box]:
    partitions = [_Partition(bounds, 0)]
    while len(partitions) < room_count:
        candidate_index = max(
            range(len(partitions)),
            key=lambda index: (
                (partitions[index].box[2] - partitions[index].box[0])
                * (partitions[index].box[3] - partitions[index].box[1]),
                -partitions[index].depth,
            ),
        )
        candidate = partitions.pop(candidate_index)
        left, top, right, bottom = candidate.box
        width, height = right - left, bottom - top
        split_vertical = width > height * rng.uniform(0.75, 1.25)
        fraction = rng.uniform(0.38, 0.62)
        if split_vertical:
            cut = round(left + width * fraction)
            children = ((left, top, cut, bottom), (cut, top, right, bottom))
        else:
            cut = round(top + height * fraction)
            children = ((left, top, right, cut), (left, cut, right, bottom))
        partitions.extend(_Partition(box, candidate.depth + 1) for box in children)
    return [item.box for item in partitions]


def _wall_key(start: Point, end: Point) -> tuple[Point, Point]:
    return tuple(sorted((start, end)))  # type: ignore[return-value]


def _walls_from_rooms(rooms: list[Box]) -> list[tuple[Point, Point]]:
    horizontal: dict[float, list[tuple[float, float]]] = {}
    vertical: dict[float, list[tuple[float, float]]] = {}
    for left, top, right, bottom in rooms:
        horizontal.setdefault(top, []).append((left, right))
        horizontal.setdefault(bottom, []).append((left, right))
        vertical.setdefault(left, []).append((top, bottom))
        vertical.setdefault(right, []).append((top, bottom))

    walls: dict[tuple[Point, Point], tuple[Point, Point]] = {}
    for y, intervals in horizontal.items():
        coordinates = sorted({value for interval in intervals for value in interval})
        for start, end in pairwise(coordinates):
            if any(left <= start and end <= right for left, right in intervals):
                segment = (start, y), (end, y)
                walls[_wall_key(*segment)] = segment
    for x, intervals in vertical.items():
        coordinates = sorted({value for interval in intervals for value in interval})
        for start, end in pairwise(coordinates):
            if any(top <= start and end <= bottom for top, bottom in intervals):
                segment = (x, start), (x, end)
                walls[_wall_key(*segment)] = segment
    return list(walls.values())


def _partition_axis(
    rng: random.Random,
    start: float,
    end: float,
    count: int,
) -> list[float]:
    span = end - start
    nominal = span / count
    coordinates = [start]
    for index in range(1, count):
        jitter = rng.uniform(-0.12, 0.12) * nominal
        coordinates.append(round(start + nominal * index + jitter))
    coordinates.append(end)
    return coordinates


def _layout_program(
    rng: random.Random,
    *,
    bounds: Box,
    profile: LayoutProfile,
) -> tuple[list[Box], list[str | None], list[Point]]:
    left, top, right, bottom = bounds
    if profile == "recursive_partition":
        boxes = _split_partitions(rng, bounds=bounds, room_count=rng.randint(6, 14))
        return (
            boxes,
            [None] * len(boxes),
            [
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ],
        )
    if profile == "double_loaded_corridor":
        corridor_half_height = rng.randint(25, 34)
        middle = round((top + bottom) / 2)
        corridor_top = middle - corridor_half_height
        corridor_bottom = middle + corridor_half_height
        columns = rng.randint(3, 5)
        boundaries = _partition_axis(rng, left, right, columns)
        boxes = [(left, corridor_top, right, corridor_bottom)]
        classes: list[str | None] = ["corridor"]
        for x0, x1 in pairwise(boundaries):
            boxes.extend(
                (
                    (x0, top, x1, corridor_top),
                    (x0, corridor_bottom, x1, bottom),
                )
            )
            classes.extend((None, None))
        return (
            boxes,
            classes,
            [
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ],
        )

    elbow_x = round(left + (right - left) * rng.uniform(0.43, 0.57))
    elbow_y = round(top + (bottom - top) * rng.uniform(0.43, 0.57))
    vertical_bounds = _partition_axis(rng, top, bottom, rng.randint(3, 5))
    horizontal_bounds = _partition_axis(rng, elbow_x, right, rng.randint(2, 4))
    boxes = [(left, y0, elbow_x, y1) for y0, y1 in pairwise(vertical_bounds)]
    boxes.extend((x0, elbow_y, x1, bottom) for x0, x1 in pairwise(horizontal_bounds))
    classes = [None] * len(boxes)
    classes[max(0, len(vertical_bounds) - 3)] = "corridor"
    footprint = [
        (left, top),
        (elbow_x, top),
        (elbow_x, elbow_y),
        (right, elbow_y),
        (right, bottom),
        (left, bottom),
    ]
    return boxes, classes, footprint


def _shared_boundaries(rooms: list[Box]) -> list[tuple[Point, Point]]:
    output: list[tuple[Point, Point]] = []
    for left_index, left_room in enumerate(rooms):
        for right_room in rooms[left_index + 1 :]:
            lx0, ly0, lx1, ly1 = left_room
            rx0, ry0, rx1, ry1 = right_room
            if lx1 == rx0 or rx1 == lx0:
                x = lx1 if lx1 == rx0 else rx1
                top, bottom = max(ly0, ry0), min(ly1, ry1)
                if bottom - top >= 48:
                    output.append(((x, top), (x, bottom)))
            if ly1 == ry0 or ry1 == ly0:
                y = ly1 if ly1 == ry0 else ry1
                left, right = max(lx0, rx0), min(lx1, rx1)
                if right - left >= 48:
                    output.append(((left, y), (right, y)))
    return output


def _opening_on_segment(
    segment: tuple[Point, Point],
    *,
    index: int,
) -> SyntheticOpening:
    start, end = segment
    horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
    center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    return SyntheticOpening(
        id=f"opening-{index:03d}",
        kind="door",
        center_px=center,
        width_px=min(32.0, math.dist(start, end) * 0.35),
        orientation="horizontal" if horizontal else "vertical",
    )


def _window_on_segment(
    segment: tuple[Point, Point],
    *,
    index: int,
) -> SyntheticOpening:
    start, end = segment
    horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
    center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    return SyntheticOpening(
        id=f"opening-{index:03d}",
        kind="window",
        center_px=center,
        width_px=min(42.0, math.dist(start, end) * 0.42),
        orientation="horizontal" if horizontal else "vertical",
    )


def _polygon_segments(polygon: list[Point]) -> list[tuple[Point, Point]]:
    return list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))


def _fixture_for_room(
    rng: random.Random,
    room: Box,
    *,
    index: int,
    room_class: str,
    fixture_type: str | None = None,
) -> SyntheticFixture:
    left, top, right, bottom = room
    fixture_type = fixture_type or rng.choice(
        _FIXTURES_BY_ROOM.get(room_class) or _GENERAL_FIXTURE_TYPES
    )
    width = min(rng.uniform(18, 36), max(8, (right - left) * 0.32))
    depth = min(rng.uniform(14, 30), max(8, (bottom - top) * 0.32))
    if fixture_type in {"shower_screen", "receptacle"}:
        depth = min(depth, 5.0)
    elif fixture_type in {"bed", "sofa"}:
        width = min(max(width, 42.0), max(22, (right - left) * 0.54))
        depth = min(max(depth, 26.0), max(16, (bottom - top) * 0.44))
    elif fixture_type in {"dining_table", "desk", "bench"}:
        width = min(max(width, 34.0), max(18, (right - left) * 0.48))
        depth = min(depth, max(12, (bottom - top) * 0.28))
    elif fixture_type in {"armchair", "chair", "coffee_table"}:
        extent = min(max(width, depth), max(12, min(right - left, bottom - top) * 0.30))
        width = depth = extent
    elif fixture_type in {
        "refrigerator",
        "stove",
        "dishwasher",
        "washing_machine",
        "tumble_dryer",
    }:
        extent = min(max(width, depth), max(13, min(right - left, bottom - top) * 0.28))
        width = depth = extent
    elif fixture_type in {"bathtub", "jacuzzi"}:
        width = min(max(width, depth * 1.75), max(18, (right - left) * 0.48))
        depth = min(depth, max(12, (bottom - top) * 0.28))
    elif fixture_type == "shower":
        extent = min(max(width, depth), max(14, min(right - left, bottom - top) * 0.34))
        width = depth = extent
    elif fixture_type == "toilet":
        width = min(width, max(12, (right - left) * 0.24))
        depth = min(max(depth, width * 1.45), max(18, (bottom - top) * 0.38))
    elif fixture_type == "sink":
        depth = min(depth, max(10, width * 0.68))
    elif fixture_type == "plumbing_fixture":
        extent = min(width, depth)
        width = depth = extent
    elif fixture_type in {
        "riser",
        "sprinkler",
        "light",
        "column",
        "water_tap",
        "wood_stove",
    }:
        extent = min(width, depth)
        width = depth = extent
    elif fixture_type == "coat_rack":
        depth = min(depth, 7.0)
    elif fixture_type in {"fireplace_corner", "place_for_fireplace_corner"}:
        extent = min(width, depth)
        width = depth = extent
    elif fixture_type == "stair":
        width = min(max(width, 28.0), max(12, (right - left) * 0.42))
        depth = min(max(depth, 42.0), max(16, (bottom - top) * 0.5))
    x = rng.uniform(left + 12, max(left + 12, right - width - 12))
    y = rng.uniform(top + 12, max(top + 12, bottom - depth - 12))
    if fixture_type in _WALL_HOSTED_FIXTURE_TYPES:
        # Real architectural equipment and casework usually form a wall-hosted
        # run.  The retired generator floated nearly every fixture in the room,
        # teaching the learner that wall contact meant "not an object".
        side = rng.choice(("top", "right", "bottom", "left"))
        inset = rng.uniform(1.5, 5.0)
        if side == "top":
            y = top + inset
        elif side == "bottom":
            y = bottom - depth - inset
        elif side == "left":
            x = left + inset
        else:
            x = right - width - inset
    return SyntheticFixture(
        id=f"fixture-{index:03d}",
        fixture_type=fixture_type,
        bbox_px=(x, y, x + width, y + depth),
        # The canonical glyph is rendered at zero degrees.  C4 training
        # augmentation rotates both the pixels and yaw label together later;
        # assigning a random yaw here without rotating the glyph is contradictory.
        yaw_deg=0.0,
    )


def _bbox_intersection_ratio(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(1e-6, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-6, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / min(left_area, right_area)


def _place_fixture_for_room(
    rng: random.Random,
    room: Box,
    *,
    fixtures: list[SyntheticFixture],
    room_class: str,
    fixture_type: str | None = None,
) -> SyntheticFixture | None:
    """Place one visible symbol without teaching contradictory overlapping labels."""

    for _ in range(48):
        candidate = _fixture_for_room(
            rng,
            room,
            index=len(fixtures),
            room_class=room_class,
            fixture_type=fixture_type,
        )
        if all(
            _bbox_intersection_ratio(candidate.bbox_px, existing.bbox_px) < 0.08
            for existing in fixtures
        ):
            return candidate
    return None


def _render_fixture_symbol(
    draw: ImageDraw.ImageDraw,
    fixture: SyntheticFixture,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = fixture.bbox_px
    box = (left, top, right, bottom)
    kind = fixture.fixture_type
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    if kind == "toilet":
        draw.ellipse(box, outline=color, width=2)
        draw.rectangle(
            (left, top, right, top + (bottom - top) * 0.28),
            outline=color,
            width=1,
        )
        draw.arc(
            (left + width * 0.18, top + height * 0.32, right - width * 0.18, bottom),
            0,
            180,
            fill=color,
            width=1,
        )
        return
    if kind == "bed":
        draw.rectangle(box, outline=color, width=2)
        draw.line((left, top + height * 0.30, right, top + height * 0.30), fill=color, width=1)
        draw.rectangle(
            (left + width * 0.08, top + height * 0.06, left + width * 0.46, top + height * 0.26),
            outline=color,
            width=1,
        )
        draw.rectangle(
            (right - width * 0.46, top + height * 0.06, right - width * 0.08, top + height * 0.26),
            outline=color,
            width=1,
        )
        return
    if kind in {"sofa", "armchair", "chair"}:
        radius = max(2, round(min(width, height) * 0.12))
        draw.rounded_rectangle(box, radius=radius, outline=color, width=2)
        draw.line(
            (left + width * 0.12, top + height * 0.28, right - width * 0.12, top + height * 0.28),
            fill=color,
            width=1,
        )
        if kind == "sofa":
            draw.line(((left + right) / 2, top + height * 0.28, (left + right) / 2, bottom), fill=color, width=1)
        else:
            draw.line((left + width * 0.16, top, left + width * 0.16, bottom), fill=color, width=1)
            draw.line((right - width * 0.16, top, right - width * 0.16, bottom), fill=color, width=1)
        return
    if kind in {"dining_table", "coffee_table"}:
        inset = min(width, height) * (0.12 if kind == "dining_table" else 0.20)
        draw.ellipse(box, outline=color, width=2)
        draw.ellipse((left + inset, top + inset, right - inset, bottom - inset), outline=color, width=1)
        if kind == "dining_table":
            for x, y in (
                ((left + right) / 2, top + 2),
                ((left + right) / 2, bottom - 2),
                (left + 2, (top + bottom) / 2),
                (right - 2, (top + bottom) / 2),
            ):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
        return
    if kind in {"desk", "bench"}:
        draw.rectangle(box, outline=color, width=2)
        draw.line((left + width * 0.12, top, left + width * 0.12, bottom), fill=color, width=1)
        draw.line((right - width * 0.12, top, right - width * 0.12, bottom), fill=color, width=1)
        if kind == "desk":
            draw.rectangle(
                (left + width * 0.58, top + height * 0.18, right - width * 0.08, bottom - height * 0.18),
                outline=color,
                width=1,
            )
        return
    if kind in {
        "refrigerator",
        "stove",
        "dishwasher",
        "washing_machine",
        "tumble_dryer",
    }:
        draw.rectangle(box, outline=color, width=2)
        if kind == "refrigerator":
            draw.line((left, top + height * 0.34, right, top + height * 0.34), fill=color, width=1)
            draw.line((right - width * 0.18, top + height * 0.12, right - width * 0.18, bottom - height * 0.12), fill=color, width=1)
        elif kind == "stove":
            for cx, cy in (
                (left + width * 0.3, top + height * 0.3),
                (right - width * 0.3, top + height * 0.3),
                (left + width * 0.3, bottom - height * 0.3),
                (right - width * 0.3, bottom - height * 0.3),
            ):
                r = min(width, height) * 0.10
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=1)
        elif kind in {"washing_machine", "tumble_dryer"}:
            inset = min(width, height) * 0.18
            draw.ellipse((left + inset, top + inset, right - inset, bottom - inset), outline=color, width=2)
        else:
            for offset in (0.22, 0.42, 0.62, 0.82):
                draw.line((left + width * 0.12, top + height * offset, right - width * 0.12, top + height * offset), fill=color, width=1)
        return
    if kind == "sink":
        draw.rounded_rectangle(box, radius=3, outline=color, width=2)
        inset = min(width, height) * 0.18
        draw.ellipse(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=color,
            width=1,
        )
        center_x = (left + right) / 2
        draw.line((center_x, top, center_x, top + height * 0.3), fill=color, width=2)
        return
    if kind == "plumbing_fixture":
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        draw.polygon(
            (
                (center_x, top),
                (right, center_y),
                (center_x, bottom),
                (left, center_y),
            ),
            outline=color,
        )
        inset = min(width, height) * 0.24
        draw.ellipse(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=color,
            width=1,
        )
        return
    if kind == "shower":
        draw.rounded_rectangle(box, radius=3, outline=color, width=2)
        draw.line((left, top, right, bottom), fill=color, width=1)
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        radius = min(width, height) * 0.1
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=1,
        )
        return
    if kind == "bathtub":
        draw.rounded_rectangle(
            box,
            radius=max(3, round(min(width, height) * 0.3)),
            outline=color,
            width=2,
        )
        inset = min(width, height) * 0.18
        draw.rounded_rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            radius=max(2, round(inset)),
            outline=color,
            width=1,
        )
        draw.ellipse(
            (left + inset * 0.35, top + height * 0.38, left + inset, top + height * 0.62),
            outline=color,
            width=1,
        )
        return
    if kind == "jacuzzi":
        draw.rounded_rectangle(
            box,
            radius=max(3, round(min(width, height) * 0.25)),
            outline=color,
            width=2,
        )
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        radius = min(width, height) * 0.23
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=1,
        )
        for x_sign, y_sign in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            jet_x = center_x + x_sign * width * 0.28
            jet_y = center_y + y_sign * height * 0.24
            draw.ellipse((jet_x - 1, jet_y - 1, jet_x + 1, jet_y + 1), fill=color)
        return
    if kind == "shower_screen":
        draw.line((left, top, right, bottom), fill=color, width=3)
        return
    if kind in {"base_cabinet", "wall_cabinet", "closet", "coat_closet"}:
        draw.rectangle(box, outline=color, width=2)
        middle = (left + right) / 2
        if kind == "base_cabinet":
            draw.line(
                (left, bottom - height * 0.22, right, bottom - height * 0.22), fill=color, width=1
            )
            draw.line((middle, top, middle, bottom), fill=color, width=1)
        elif kind == "wall_cabinet":
            dash = max(2.0, width / 8)
            cursor = left
            while cursor < right:
                draw.line(
                    (cursor, top + height * 0.2, min(right, cursor + dash), top + height * 0.2),
                    fill=color,
                    width=1,
                )
                cursor += dash * 2
            draw.line((left, bottom - height * 0.2, right, top + height * 0.2), fill=color, width=1)
        if kind in {"closet", "coat_closet"}:
            draw.line((left, top, right, bottom), fill=color, width=1)
            draw.line((right, top, left, bottom), fill=color, width=1)
            if kind == "coat_closet":
                draw.line(
                    (
                        left + width * 0.15,
                        top + height * 0.28,
                        right - width * 0.15,
                        top + height * 0.28,
                    ),
                    fill=color,
                    width=1,
                )
        return
    if kind == "stair":
        draw.rectangle(box, outline=color, width=2)
        for step in range(1, 6):
            y = top + (bottom - top) * step / 6
            draw.line((left, y, right, y), fill=color, width=1)
        return
    if kind == "light":
        draw.ellipse(box, outline=color, width=2)
        draw.line((left, top, right, bottom), fill=color, width=1)
        draw.line((right, top, left, bottom), fill=color, width=1)
        return
    if kind == "sprinkler":
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        radius = min(width, height) * 0.18
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=2,
        )
        draw.line((center_x, top, center_x, bottom), fill=color, width=1)
        draw.line((left, center_y, right, center_y), fill=color, width=1)
        return
    if kind == "column":
        draw.rectangle(box, fill=color)
        return
    if kind == "electrical_appliance":
        draw.rectangle(box, outline=color, width=2)
        radius = min(width, height) * 0.18
        draw.ellipse(
            (
                left + width * 0.18 - radius,
                top + height * 0.22 - radius,
                left + width * 0.18 + radius,
                top + height * 0.22 + radius,
            ),
            outline=color,
            width=1,
        )
        draw.line(
            (left + width * 0.35, top + height * 0.22, right - width * 0.1, top + height * 0.22),
            fill=color,
            width=1,
        )
        return
    if kind == "electrical_panel":
        draw.rectangle(box, outline=color, width=2)
        for row in range(1, 4):
            y = top + height * row / 4
            draw.line((left + width * 0.15, y, right - width * 0.15, y), fill=color, width=1)
        draw.line(
            ((left + right) / 2, top + height * 0.12, (left + right) / 2, bottom - height * 0.12),
            fill=color,
            width=1,
        )
        return
    if kind == "hvac_terminal":
        draw.rectangle(box, outline=color, width=2)
        for offset in (0.2, 0.4, 0.6, 0.8):
            x = left + width * offset
            draw.line((x, top + height * 0.12, x, bottom - height * 0.12), fill=color, width=1)
        return
    if kind == "riser":
        draw.ellipse(box, outline=color, width=2)
        inset = min(width, height) * 0.22
        draw.ellipse(
            (left + inset, top + inset, right - inset, bottom - inset), outline=color, width=1
        )
        return
    if kind == "receptacle":
        draw.ellipse(box, outline=color, width=2)
        draw.line(
            (left + width * 0.36, top + height * 0.25, left + width * 0.36, bottom - height * 0.25),
            fill=color,
            width=1,
        )
        draw.line(
            (
                right - width * 0.36,
                top + height * 0.25,
                right - width * 0.36,
                bottom - height * 0.25,
            ),
            fill=color,
            width=1,
        )
        return
    if kind == "sauna_bench":
        draw.rectangle(box, outline=color, width=2)
        for offset in (0.25, 0.5, 0.75):
            y = top + height * offset
            draw.line((left, y, right, y), fill=color, width=1)
        return
    if kind == "fireplace":
        draw.rectangle(box, outline=color, width=2)
        draw.arc(
            (left + width * 0.18, top + height * 0.18, right - width * 0.18, bottom + height * 0.3),
            180,
            360,
            fill=color,
            width=2,
        )
        draw.line(
            (left, bottom - height * 0.15, right, bottom - height * 0.15), fill=color, width=2
        )
        return
    if kind == "chimney":
        draw.rectangle(box, outline=color, width=2)
        draw.line((left, top, right, bottom), fill=color, width=2)
        draw.line((right, top, left, bottom), fill=color, width=2)
        inset = min(width, height) * 0.25
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset), outline=color, width=1
        )
        return
    if kind == "coat_rack":
        draw.line((left, (top + bottom) / 2, right, (top + bottom) / 2), fill=color, width=2)
        for offset in (0.2, 0.4, 0.6, 0.8):
            x = left + width * offset
            draw.arc((x - 3, top, x + 3, bottom), 180, 360, fill=color, width=1)
        return
    if kind == "water_tap":
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        draw.ellipse(box, outline=color, width=2)
        draw.line((center_x, top, center_x, bottom), fill=color, width=1)
        draw.line((left, center_y, right, center_y), fill=color, width=1)
        draw.arc((center_x, top, right, center_y), 180, 360, fill=color, width=1)
        return
    if kind == "wood_stove":
        draw.ellipse(box, outline=color, width=2)
        inset = min(width, height) * 0.25
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=color,
            width=1,
        )
        draw.line(
            ((left + right) / 2, top, (left + right) / 2, top - height * 0.2), fill=color, width=2
        )
        return
    if kind in {"fireplace_corner", "place_for_fireplace_corner"}:
        points = ((left, top), (right, top), (left, bottom))
        if kind == "fireplace_corner":
            draw.polygon(points, outline=color)
            draw.arc((left, top, right, bottom), 180, 270, fill=color, width=2)
        else:
            draw.line((left, top, right, top), fill=color, width=1)
            draw.line((left, top, left, bottom), fill=color, width=1)
            for offset in (0.25, 0.55, 0.85):
                draw.line(
                    (left + width * offset, top, left, top + height * offset),
                    fill=color,
                    width=1,
                )
        return
    if kind == "place_for_fireplace":
        draw.rectangle(box, outline=color, width=1)
        dash = max(2.0, width / 7)
        cursor = left
        while cursor < right:
            draw.line((cursor, top, min(right, cursor + dash), top), fill=color, width=2)
            draw.line((cursor, bottom, min(right, cursor + dash), bottom), fill=color, width=2)
            cursor += dash * 2
        return
    if kind == "housing":
        draw.rectangle(box, outline=color, width=2)
        draw.line((left, (top + bottom) / 2, right, (top + bottom) / 2), fill=color, width=1)
        draw.line(((left + right) / 2, top, (left + right) / 2, bottom), fill=color, width=1)
        return
    if kind == "misc":
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        draw.polygon(
            (
                (center_x, top),
                (right, center_y),
                (center_x, bottom),
                (left, center_y),
            ),
            outline=color,
        )
        draw.line((left, center_y, right, center_y), fill=color, width=1)
        return
    draw.rectangle(box, outline=color, width=2)
    inset = max(2.0, min(right - left, bottom - top) * 0.2)
    if right - left > inset * 2 and bottom - top > inset * 2:
        draw.rectangle(
            (left + inset, top + inset, right - inset, bottom - inset),
            outline=color,
            width=1,
        )


def _room_box(room: SyntheticRoom) -> Box:
    return (
        min(point[0] for point in room.polygon_px),
        min(point[1] for point in room.polygon_px),
        max(point[0] for point in room.polygon_px),
        max(point[1] for point in room.polygon_px),
    )


def _render_room_surfaces(
    draw: ImageDraw.ImageDraw,
    rooms: list[SyntheticRoom],
    *,
    style: Style,
    rng: random.Random,
) -> None:
    """Render floor finishes and hatches that the structural targets must ignore."""

    if style not in {"colored_cad", "layered_cad", "scan"}:
        return
    fills = {
        "bathroom": (213, 231, 224),
        "kitchen": (233, 219, 213),
        "outdoor": (224, 226, 219),
        "mechanical": (226, 228, 217),
    }
    generic = ((238, 222, 219), (235, 229, 218), (226, 234, 229))
    for room in rooms:
        box = _room_box(room)
        if style == "colored_cad":
            draw.polygon(
                room.polygon_px,
                fill=fills.get(room.room_class, rng.choice(generic)),
            )
        if room.room_class in {"bathroom", "outdoor"} and rng.random() < 0.7:
            left, top, right, bottom = box
            spacing = rng.randint(7, 13)
            hatch = (190, 197, 189) if style != "scan" else (205, 201, 190)
            for x in range(round(left), round(right) + 1, spacing):
                draw.line((x, top, x, bottom), fill=hatch, width=1)
            for y in range(round(top), round(bottom) + 1, spacing):
                draw.line((left, y, right, y), fill=hatch, width=1)


def _render_dimension_clutter(
    draw: ImageDraw.ImageDraw,
    rooms: list[SyntheticRoom],
    *,
    canvas_size: tuple[int, int],
    style: Style,
    rng: random.Random,
) -> None:
    if not rooms or rng.random() > 0.78:
        return
    left = min(_room_box(room)[0] for room in rooms)
    top = min(_room_box(room)[1] for room in rooms)
    right = max(_room_box(room)[2] for room in rooms)
    bottom = max(_room_box(room)[3] for room in rooms)
    color = (213, 138, 90) if style in {"colored_cad", "layered_cad"} else (120, 121, 117)
    font = ImageFont.load_default()
    if top > 20:
        y = max(7, top - 14)
        draw.line((left, y, right, y), fill=color, width=1)
        tick_count = rng.randint(3, 7)
        for index in range(tick_count + 1):
            x = left + (right - left) * index / tick_count
            draw.line((x, y - 4, x, y + 4), fill=color, width=1)
        draw.text(
            (left + 4, max(0, y - 11)),
            rng.choice(("10x20", "15x14", "6x12")),
            fill=color,
            font=font,
        )
    _, canvas_height = canvas_size
    if bottom < canvas_height - 18:
        y = min(canvas_height - 7, bottom + 13)
        draw.line((left, y, right, y), fill=color, width=1)
        draw.text((right - 38, y + 1), rng.choice(("A-101", "L1", "1:100")), fill=color, font=font)


def _render_sample(
    image_path: Path,
    *,
    canvas_size: tuple[int, int],
    style: Style,
    walls: list[tuple[Point, Point]],
    rooms: list[SyntheticRoom],
    openings: list[SyntheticOpening],
    fixtures: list[SyntheticFixture],
    rng: random.Random,
) -> None:
    background = (252, 251, 248) if style != "scan" else (244, 241, 232)
    ink = (25, 28, 27) if style != "markup" else (20, 36, 32)
    if style == "colored_cad":
        ink = (64, 65, 62)
        accent = (222, 143, 91)
        fixture_color = (71, 162, 103)
    elif style == "layered_cad":
        ink = (48, 54, 56)
        accent = (80, 153, 139)
        fixture_color = (173, 82, 150)
    else:
        accent = (52, 105, 82) if style == "markup" else ink
        fixture_color = accent
    image = Image.new("RGB", canvas_size, background)
    draw = ImageDraw.Draw(image)
    _render_room_surfaces(draw, rooms, style=style, rng=rng)
    wall_width = rng.randint(7, 12) if style in {"colored_cad", "cad"} else rng.randint(5, 9)
    for start, end in walls:
        draw.line((*start, *end), fill=ink, width=wall_width)
    for opening in openings:
        x, y = opening.center_px
        half = opening.width_px / 2
        if opening.orientation == "horizontal":
            draw.line((x - half, y, x + half, y), fill=background, width=wall_width + 2)
            if opening.kind == "door":
                draw.arc(
                    (x - half, y - opening.width_px, x + half, y),
                    0,
                    90,
                    fill=accent,
                    width=2,
                )
            else:
                draw.line((x - half, y - 2, x + half, y - 2), fill=accent, width=1)
                draw.line((x - half, y + 2, x + half, y + 2), fill=accent, width=1)
        else:
            draw.line((x, y - half, x, y + half), fill=background, width=wall_width + 2)
            if opening.kind == "door":
                draw.arc(
                    (x, y - half, x + opening.width_px, y + half),
                    90,
                    180,
                    fill=accent,
                    width=2,
                )
            else:
                draw.line((x - 2, y - half, x - 2, y + half), fill=accent, width=1)
                draw.line((x + 2, y - half, x + 2, y + half), fill=accent, width=1)
    label_variants = {
        "living": ("LIVING", "LIVING ROOM", "LIV"),
        "bedroom": ("BEDROOM", "BED", "BR"),
        "kitchen": ("KITCHEN", "KIT", "K"),
        "bathroom": ("BATHROOM", "BATH", "WC"),
        "corridor": ("CORRIDOR", "HALL", "HALLWAY"),
        "storage": ("STORAGE", "STORE", "STOR"),
        "office": ("OFFICE", "WORK", "OFF"),
        "mechanical": ("MECHANICAL", "MECH", "M.E.P."),
        "garage": ("GARAGE", "PARKING", "GAR"),
        "utility": ("UTILITY", "LAUNDRY", "UTIL"),
        "outdoor": ("OUTDOOR", "BALCONY", "TERRACE"),
        "other": ("ROOM", "SPACE", "AREA"),
    }
    fonts: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for room in rooms:
        label = rng.choice(label_variants.get(room.room_class, (room.room_class.upper(),)))
        room_width = max(point[0] for point in room.polygon_px) - min(
            point[0] for point in room.polygon_px
        )
        room_height = max(point[1] for point in room.polygon_px) - min(
            point[1] for point in room.polygon_px
        )
        font_size = max(10, min(17, round(min(room_width, room_height) * 0.09)))
        if font_size not in fonts:
            try:
                fonts[font_size] = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:  # pragma: no cover - Pillow wheels normally bundle DejaVu.
                fonts[font_size] = ImageFont.load_default()
        font = fonts[font_size]
        center_x = sum(point[0] for point in room.polygon_px) / len(room.polygon_px)
        center_y = sum(point[1] for point in room.polygon_px) / len(room.polygon_px)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.text(
            (center_x - text_width / 2, center_y - text_height / 2),
            label,
            fill=ink,
            font=font,
        )
    for fixture in fixtures:
        _render_fixture_symbol(draw, fixture, fixture_color)
    _render_dimension_clutter(
        draw,
        rooms,
        canvas_size=canvas_size,
        style=style,
        rng=rng,
    )
    if style == "scan":
        generator = np.random.default_rng(rng.getrandbits(64))
        array = np.asarray(image, dtype=np.float32)
        array += generator.normal(
            0.0,
            rng.uniform(1.5, 5.0),
            size=array.shape,
        ).astype(np.float32)
        image = Image.fromarray(np.uint8(np.clip(array, 0, 255)), mode="RGB")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, optimize=True)


def generate_synthetic_pretraining_sample(
    output_root: str | Path,
    *,
    sample_index: int,
    seed: int,
    size: int = 512,
    canvas_profile: CanvasProfile | None = None,
) -> SyntheticPretrainingSample:
    rng = random.Random(seed)
    sample_id = f"synthetic-{sample_index:07d}"
    style: Style = rng.choice(
        ("cad", "scan", "markup", "colored_cad", "colored_cad", "layered_cad")
    )
    selected_canvas_profile: CanvasProfile = canvas_profile or rng.choice(
        ("square", "square", "portrait", "landscape")
    )
    if selected_canvas_profile == "portrait":
        canvas_width = round(size * rng.uniform(0.56, 0.78))
        canvas_height = size
    elif selected_canvas_profile == "landscape":
        canvas_width = size
        canvas_height = round(size * rng.uniform(0.56, 0.78))
    else:
        canvas_width = canvas_height = size
    margin_basis = min(canvas_width, canvas_height)
    horizontal_margin = rng.randint(
        round(margin_basis * 0.055),
        round(margin_basis * 0.10),
    )
    vertical_margin = rng.randint(
        round(margin_basis * 0.055),
        round(margin_basis * 0.10),
    )
    bounds = (
        float(horizontal_margin),
        float(vertical_margin),
        float(canvas_width - horizontal_margin),
        float(canvas_height - vertical_margin),
    )
    profile: LayoutProfile = rng.choice(
        ("recursive_partition", "double_loaded_corridor", "l_shaped_building")
    )
    boxes, forced_classes, footprint = _layout_program(
        rng,
        bounds=bounds,
        profile=profile,
    )
    rooms = [
        SyntheticRoom(
            id=f"room-{index:03d}",
            polygon_px=[
                (box[0], box[1]),
                (box[2], box[1]),
                (box[2], box[3]),
                (box[0], box[3]),
            ],
            room_class=forced_classes[index] or rng.choice(_ROOM_CLASSES),
        )
        for index, box in enumerate(boxes)
    ]
    walls = _walls_from_rooms(boxes)
    interior_openings = [
        _opening_on_segment(segment, index=index)
        for index, segment in enumerate(_shared_boundaries(boxes))
    ]
    exterior_segments = [
        segment for segment in _polygon_segments(footprint) if math.dist(*segment) >= 70
    ]
    window_offset = len(interior_openings)
    exterior_windows = [
        _window_on_segment(segment, index=window_offset + index)
        for index, segment in enumerate(exterior_segments)
    ]
    openings = [*interior_openings, *exterior_windows]
    fixtures: list[SyntheticFixture] = []
    for room_index, box in enumerate(boxes):
        if rng.random() >= 0.9:
            continue
        room_class = rooms[room_index].room_class
        count = rng.randint(1, 3 if room_class in {"kitchen", "bathroom", "mechanical"} else 2)
        for _ in range(count):
            placed = _place_fixture_for_room(
                rng,
                box,
                fixtures=fixtures,
                room_class=room_class,
            )
            if placed is not None:
                fixtures.append(placed)
    # Guarantee corpus-level coverage instead of hoping a room-specific sampling
    # table eventually emits every contract class.  This is a deterministic
    # curriculum assignment, not an evaluation label.
    coverage_room_indices = sorted(
        range(len(boxes)),
        key=lambda index: (boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1]),
        reverse=True,
    )
    for coverage_offset, preferred_room_index in enumerate(coverage_room_indices[:2]):
        coverage_fixture_type = _GENERAL_FIXTURE_TYPES[
            (sample_index + coverage_offset * 17) % len(_GENERAL_FIXTURE_TYPES)
        ]
        placed = None
        fallback_order = [
            preferred_room_index,
            *(
                room_index
                for room_index in coverage_room_indices
                if room_index != preferred_room_index
            ),
        ]
        for coverage_room_index in fallback_order:
            placed = _place_fixture_for_room(
                rng,
                boxes[coverage_room_index],
                fixtures=fixtures,
                room_class=rooms[coverage_room_index].room_class,
                fixture_type=coverage_fixture_type,
            )
            if placed is not None:
                break
        if placed is not None:
            fixtures.append(placed)
    root = Path(output_root).expanduser().resolve()
    relative_image_path = Path("images") / f"{sample_id}.png"
    image_path = root / relative_image_path
    annotation_path = root / "annotations" / f"{sample_id}.json"
    _render_sample(
        image_path,
        canvas_size=(canvas_width, canvas_height),
        style=style,
        walls=walls,
        rooms=rooms,
        openings=openings,
        fixtures=fixtures,
        rng=rng,
    )
    sample = SyntheticPretrainingSample(
        sample_id=sample_id,
        seed=seed,
        style=style,
        canvas_profile=selected_canvas_profile,
        image_size_px=(canvas_width, canvas_height),
        layout_profile=profile,
        image_path=relative_image_path.as_posix(),
        image_sha256=sha256_file(image_path),
        building_footprint_px=footprint,
        rooms=rooms,
        walls=walls,
        openings=openings,
        fixtures=fixtures,
    ).finalize()
    assert_synthetic_pretraining_only(sample.model_dump(mode="json"))
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        sample.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sample


def generate_synthetic_pretraining_corpus(
    output_root: str | Path,
    *,
    count: int,
    seed: int,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("count must be positive")
    canvas_curriculum: tuple[CanvasProfile, ...] = (
        "square",
        "portrait",
        "landscape",
    )
    samples = [
        generate_synthetic_pretraining_sample(
            output_root,
            sample_index=index,
            seed=seed + index,
            canvas_profile=canvas_curriculum[index % len(canvas_curriculum)],
        )
        for index in range(count)
    ]
    manifest = {
        "schema_version": "dajoong.synthetic-pretraining-corpus.v1",
        "role": _SYNTHETIC_ROLE,
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "sample_count": len(samples),
        "sample_sha256": [sample.content_sha256 for sample in samples],
    }
    assert_synthetic_pretraining_only(manifest)
    manifest["content_sha256"] = sha256_json(manifest)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def audit_synthetic_pretraining_corpus(
    corpus_root: str | Path,
    *,
    maximum_fixture_overlap_ratio: float = 0.08,
    require_complete_taxonomy: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Exhaustively validate a generated corpus before it can become supervision.

    This deliberately checks every annotation and source image.  A manifest count
    alone cannot detect distorted canvases, stale generator output, contradictory
    yaw labels, missing taxonomy classes, or fixture collisions.
    """

    root = Path(corpus_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        raise FileNotFoundError(f"synthetic corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_synthetic_pretraining_only(manifest)
    annotations = sorted((root / "annotations").glob("*.json"))
    expected_count = int(manifest.get("sample_count", -1))
    if len(annotations) != expected_count:
        errors.append(
            f"annotation_count:{len(annotations)}!=manifest_sample_count:{expected_count}"
        )

    fixture_counts: Counter[str] = Counter()
    canvas_profiles: Counter[str] = Counter()
    image_sizes: Counter[tuple[int, int]] = Counter()
    sample_ids: set[str] = set()
    image_paths: set[Path] = set()
    maximum_observed_overlap = 0.0
    overlapping_sample_count = 0
    invalid_bbox_count = 0

    for annotation_path in annotations:
        try:
            sample = SyntheticPretrainingSample.model_validate_json(
                annotation_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # pragma: no cover - defensive audit output
            errors.append(f"{annotation_path.name}:invalid_annotation:{exc}")
            continue
        assert_synthetic_pretraining_only(sample.model_dump(mode="json"))
        if sample.sample_id in sample_ids:
            errors.append(f"{sample.sample_id}:duplicate_sample_id")
        sample_ids.add(sample.sample_id)
        if sample.generator_version != _GENERATOR_VERSION:
            errors.append(
                f"{sample.sample_id}:generator_version:{sample.generator_version}"
            )
        expected_content_sha = sha256_json(
            sample.model_dump(mode="json", exclude={"content_sha256"})
        )
        if sample.content_sha256 != expected_content_sha:
            errors.append(f"{sample.sample_id}:content_sha256_mismatch")

        image_path = (root / sample.image_path).resolve()
        try:
            image_path.relative_to(root)
        except ValueError:
            errors.append(f"{sample.sample_id}:image_path_outside_corpus")
            continue
        if image_path in image_paths:
            errors.append(f"{sample.sample_id}:duplicate_image_path")
        image_paths.add(image_path)
        if not image_path.is_file():
            errors.append(f"{sample.sample_id}:missing_image")
            continue
        if sha256_file(image_path) != sample.image_sha256:
            errors.append(f"{sample.sample_id}:image_sha256_mismatch")
        try:
            with Image.open(image_path) as image:
                actual_size = image.size
                image.verify()
        except Exception as exc:  # pragma: no cover - defensive audit output
            errors.append(f"{sample.sample_id}:invalid_image:{exc}")
            continue
        if sample.image_size_px != actual_size:
            errors.append(
                f"{sample.sample_id}:image_size:{actual_size}!={sample.image_size_px}"
            )
        width, height = actual_size
        expected_profile = (
            "square" if width == height else "portrait" if width < height else "landscape"
        )
        if sample.canvas_profile != expected_profile:
            errors.append(
                f"{sample.sample_id}:canvas_profile:{sample.canvas_profile}!={expected_profile}"
            )
        canvas_profiles[sample.canvas_profile] += 1
        image_sizes[actual_size] += 1

        sample_overlaps = 0
        for fixture in sample.fixtures:
            fixture_counts[fixture.fixture_type] += 1
            if not math.isclose(fixture.yaw_deg, 0.0, abs_tol=1e-9):
                errors.append(f"{sample.sample_id}:{fixture.id}:contradictory_source_yaw")
            left, top, right, bottom = fixture.bbox_px
            bbox_valid = (
                all(math.isfinite(value) for value in fixture.bbox_px)
                and 0.0 <= left < right <= width
                and 0.0 <= top < bottom <= height
            )
            if not bbox_valid:
                invalid_bbox_count += 1
                errors.append(f"{sample.sample_id}:{fixture.id}:invalid_bbox")
        for left_fixture, right_fixture in combinations(sample.fixtures, 2):
            ratio = _bbox_intersection_ratio(
                left_fixture.bbox_px,
                right_fixture.bbox_px,
            )
            maximum_observed_overlap = max(maximum_observed_overlap, ratio)
            if ratio >= maximum_fixture_overlap_ratio:
                sample_overlaps += 1
                errors.append(
                    f"{sample.sample_id}:fixture_overlap:{left_fixture.id}:"
                    f"{right_fixture.id}:{ratio:.6f}"
                )
        if sample_overlaps:
            overlapping_sample_count += 1

    missing_fixture_classes = sorted(set(_GENERAL_FIXTURE_TYPES) - set(fixture_counts))
    unexpected_fixture_classes = sorted(set(fixture_counts) - set(_GENERAL_FIXTURE_TYPES))
    if require_complete_taxonomy and missing_fixture_classes:
        errors.append(f"missing_fixture_classes:{','.join(missing_fixture_classes)}")
    if unexpected_fixture_classes:
        errors.append(
            f"unexpected_fixture_classes:{','.join(unexpected_fixture_classes)}"
        )
    passed = not errors
    report: dict[str, Any] = {
        "schema_version": "dajoong.synthetic-pretraining-corpus-audit.v1",
        "corpus_root": root.as_posix(),
        "passed": passed,
        "sample_count": len(annotations),
        "manifest_sample_count": expected_count,
        "generator_version": _GENERATOR_VERSION,
        "maximum_fixture_overlap_ratio": maximum_fixture_overlap_ratio,
        "complete_taxonomy_required": require_complete_taxonomy,
        "maximum_observed_fixture_overlap_ratio": maximum_observed_overlap,
        "overlapping_sample_count": overlapping_sample_count,
        "invalid_bbox_count": invalid_bbox_count,
        "canvas_profile_counts": dict(sorted(canvas_profiles.items())),
        "unique_image_size_count": len(image_sizes),
        "fixture_class_counts": dict(sorted(fixture_counts.items())),
        "missing_fixture_classes": missing_fixture_classes,
        "unexpected_fixture_classes": unexpected_fixture_classes,
        "error_count": len(errors),
        "errors": errors,
    }
    report["content_sha256"] = sha256_json(report)
    if raise_on_error and not passed:
        preview = "; ".join(errors[:10])
        raise ValueError(f"synthetic corpus audit failed ({len(errors)} errors): {preview}")
    return report
