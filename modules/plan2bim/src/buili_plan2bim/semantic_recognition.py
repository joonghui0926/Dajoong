"""Optional full-sheet semantic recognition for raster floor plans.

The recognizer uses a content-addressed ONNX artifact and emits auditable pixel
geometry.  Model licensing and production authorization stay explicit in the
sidecar manifest; a research teacher can never silently become a release model.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from .core.hashing import sha256_file, sha256_json
from .core.model.aec_decode import PixelLineProposal, PixelRoomProposal, PixelSymbolProposal


class SemanticDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    class_name: str
    symbol_class: str
    bbox_px: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)
    pixel_area: int = Field(ge=1)
    review_required: bool
    promote_to_bim: bool


class SemanticRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    class_name: str
    polygon_px: list[tuple[float, float]] = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)
    pixel_area: int = Field(ge=1)
    review_required: bool


class SemanticRecognitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "buili.semantic-recognition.v1"
    input_path: str
    input_sha256: str
    input_mode: str = "raster_only"
    model_version: str
    model_sha256: str
    license_scope: str
    production_authorized: bool
    source_size: tuple[int, int]
    model_input_size: tuple[int, int]
    wall_pixels: int
    wall_centerlines_px: list[tuple[float, float, float, float]] = Field(default_factory=list)
    detections: list[SemanticDetection]
    rooms: list[SemanticRoom] = Field(default_factory=list)
    counts: dict[str, int]
    room_counts: dict[str, int] = Field(default_factory=dict)
    inference_ms: float
    total_ms: float
    overlay_path: str = ""
    content_sha256: str = ""

    def finalize(self) -> SemanticRecognitionResult:
        payload = self.model_dump(mode="json", exclude={"content_sha256", "total_ms"})
        self.content_sha256 = sha256_json(payload)
        return self


_SYMBOL_CLASS = {
    "Window": "window",
    "Door": "door",
    "Closet": "closet",
    "Electrical appliance": "electrical_appliance",
    "Toilet": "toilet",
    "Sink": "sink",
    "Sauna bench": "sauna_bench",
    "Fireplace": "fireplace",
    "Bathtub": "bathtub",
    "Chimney": "chimney",
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _softmax(value: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = value - value.max(axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -30, 30))
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _component_records(
    mask: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_area: int,
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    output: list[dict[str, Any]] = []
    objects = ndimage.find_objects(labels)
    for component, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        local = labels[slices] == component
        pixel_area = int(local.sum())
        if pixel_area < minimum_area:
            continue
        y_slice, x_slice = slices
        values = probability[slices][local]
        output.append(
            {
                "bbox_px": (
                    int(x_slice.start),
                    int(y_slice.start),
                    int(x_slice.stop),
                    int(y_slice.stop),
                ),
                "confidence": float(values.mean()),
                "pixel_area": pixel_area,
            }
        )
    return output


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1], strict=True)
        )
        / 2
    )


def _point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return float(np.hypot(point[0] - start[0], point[1] - start[1]))
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    nearest = start[0] + fraction * dx, start[1] + fraction * dy
    return float(np.hypot(point[0] - nearest[0], point[1] - nearest[1]))


def _rdp(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    distances = [_point_line_distance(point, points[0], points[-1]) for point in points[1:-1]]
    if not distances or max(distances) <= tolerance:
        return [points[0], points[-1]]
    pivot = distances.index(max(distances)) + 1
    return _rdp(points[: pivot + 1], tolerance)[:-1] + _rdp(points[pivot:], tolerance)


def _simplify_loop(
    points: list[tuple[float, float]],
    *,
    tolerance: float,
) -> list[tuple[float, float]]:
    if len(points) <= 4:
        return points
    anchor = max(
        range(1, len(points)),
        key=lambda index: math.dist(points[0], points[index]),
    )
    first = _rdp(points[: anchor + 1], tolerance)
    second = _rdp(points[anchor:] + [points[0]], tolerance)
    simplified = first[:-1] + second[:-1]
    output: list[tuple[float, float]] = []
    for point in simplified:
        if not output or math.dist(output[-1], point) > 1e-6:
            output.append(point)
    return output


def _mask_outer_loop(mask: np.ndarray) -> list[tuple[float, float]]:
    """Trace the largest pixel-boundary loop without an OpenCV dependency."""

    height, width = mask.shape
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    loops: list[list[tuple[float, float]]] = []
    edge_budget = sum(len(value) for value in edges.values()) + 1
    while edges:
        start = next(iter(edges))
        current = start
        loop: list[tuple[float, float]] = []
        for _ in range(edge_budget):
            loop.append((float(current[0]), float(current[1])))
            candidates = edges.get(current)
            if not candidates:
                break
            following = candidates.pop()
            if not candidates:
                del edges[current]
            current = following
            if current == start:
                if len(loop) >= 3:
                    loops.append(loop)
                break
    return max(loops, key=_polygon_area, default=[])


def _room_records(
    class_mask: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_area: int,
    source_size: tuple[int, int],
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(
        class_mask,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8),
    )
    scale_x = source_size[0] / class_mask.shape[1]
    scale_y = source_size[1] / class_mask.shape[0]
    output = []
    for component in range(1, count + 1):
        local = labels == component
        pixel_area = int(local.sum())
        if pixel_area < minimum_area:
            continue
        loop = _mask_outer_loop(local)
        if len(loop) < 3:
            continue
        simplified = _simplify_loop(loop, tolerance=2.0)
        polygon = [(x * scale_x, y * scale_y) for x, y in simplified]
        if len(polygon) < 3 or _polygon_area(polygon) < minimum_area * scale_x * scale_y:
            continue
        output.append(
            {
                "polygon_px": polygon,
                "confidence": float(probability[local].mean()),
                "pixel_area": pixel_area,
            }
        )
    return output


def _merge_axis_segments(
    segments: list[tuple[float, float, float, float]],
    *,
    horizontal: bool,
    coordinate_tolerance: float,
    gap_tolerance: float,
) -> list[tuple[float, float, float, float]]:
    if horizontal:
        normalized = sorted((y0, x0, x1) for x0, y0, x1, _ in segments)
    else:
        normalized = sorted((x0, y0, y1) for x0, y0, _, y1 in segments)
    groups: list[list[tuple[float, float, float]]] = []
    for coordinate, start, end in normalized:
        target = None
        for group in reversed(groups):
            group_coordinate = float(np.median([item[0] for item in group]))
            group_start = min(item[1] for item in group)
            group_end = max(item[2] for item in group)
            if coordinate - group_coordinate > coordinate_tolerance:
                break
            if abs(coordinate - group_coordinate) <= coordinate_tolerance and not (
                end < group_start - gap_tolerance or start > group_end + gap_tolerance
            ):
                target = group
                break
        if target is None:
            groups.append([(coordinate, start, end)])
        else:
            target.append((coordinate, start, end))
    output = []
    for group in groups:
        coordinate = float(np.median([item[0] for item in group]))
        start = float(min(item[1] for item in group))
        end = float(max(item[2] for item in group))
        output.append(
            (start, coordinate, end, coordinate)
            if horizontal
            else (coordinate, start, coordinate, end)
        )
    return output


def _axis_run_segments(
    support: np.ndarray,
    *,
    horizontal: bool,
    minimum_run: int,
) -> list[tuple[float, float, float, float]]:
    """Track long foreground runs across adjacent rows without joining T branches."""

    scan = support if horizontal else support.T
    active: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    for coordinate, row in enumerate(scan):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        runs = [
            (int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
            if end - start >= minimum_run
        ]
        candidates = [item for item in active if item["last"] == coordinate - 1]
        claimed: set[int] = set()
        next_active: list[dict[str, Any]] = []
        for start, end in runs:
            best_index = -1
            best_score = 0.0
            for index, group in enumerate(candidates):
                if index in claimed:
                    continue
                previous_start, previous_end = group["runs"][-1][1:]
                overlap = max(0, min(end, previous_end) - max(start, previous_start))
                score = overlap / max(1, min(end - start, previous_end - previous_start))
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0 and best_score >= 0.55:
                group = candidates[best_index]
                claimed.add(best_index)
                group["runs"].append((coordinate, start, end))
                group["last"] = coordinate
            else:
                group = {"last": coordinate, "runs": [(coordinate, start, end)]}
            next_active.append(group)
        for group in active:
            if group not in next_active:
                finished.append(group)
        active = next_active
    finished.extend(active)

    output: list[tuple[float, float, float, float]] = []
    for group in finished:
        records = group["runs"]
        if len(records) < 2:
            continue
        coordinate = float(np.median([record[0] for record in records]))
        start = float(np.median([record[1] for record in records]))
        end = float(np.median([record[2] for record in records]))
        thickness = records[-1][0] - records[0][0] + 1
        if end - start < max(minimum_run, thickness * 1.5):
            continue
        output.append(
            (start, coordinate, end, coordinate)
            if horizontal
            else (coordinate, start, coordinate, end)
        )
    return output


def _collapse_thick_parallel(
    segments: list[tuple[float, float, float, float]],
    wall_mask: np.ndarray,
    *,
    horizontal: bool,
) -> list[tuple[float, float, float, float]]:
    """Collapse multiple ridges inside one thick wall while preserving nearby walls."""

    pending = list(segments)
    changed = True
    maximum_band = max(24, round(max(wall_mask.shape) * 0.035))
    while changed:
        changed = False
        for left_index, left in enumerate(pending):
            left_coordinate = left[1] if horizontal else left[0]
            left_start, left_end = (left[0], left[2]) if horizontal else (left[1], left[3])
            for right_index in range(left_index + 1, len(pending)):
                right = pending[right_index]
                right_coordinate = right[1] if horizontal else right[0]
                distance = abs(right_coordinate - left_coordinate)
                if distance > maximum_band:
                    continue
                right_start, right_end = (
                    (right[0], right[2]) if horizontal else (right[1], right[3])
                )
                overlap_start = max(left_start, right_start)
                overlap_end = min(left_end, right_end)
                if overlap_end - overlap_start < 20:
                    continue
                c0 = max(0, round(min(left_coordinate, right_coordinate)))
                c1 = min(
                    wall_mask.shape[0 if horizontal else 1],
                    round(max(left_coordinate, right_coordinate)) + 1,
                )
                s0 = max(0, round(overlap_start))
                s1 = min(
                    wall_mask.shape[1 if horizontal else 0],
                    round(overlap_end),
                )
                band = wall_mask[c0:c1, s0:s1] if horizontal else wall_mask[s0:s1, c0:c1]
                if band.size == 0 or float(band.mean()) < 0.55:
                    continue
                coordinate = (left_coordinate + right_coordinate) / 2
                start = min(left_start, right_start)
                end = max(left_end, right_end)
                merged = (
                    (start, coordinate, end, coordinate)
                    if horizontal
                    else (coordinate, start, coordinate, end)
                )
                pending[left_index] = merged
                pending.pop(right_index)
                changed = True
                break
            if changed:
                break
    return pending


def _wall_centerlines(wall_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Vectorize an orthogonal semantic wall mask without consulting a reference graph."""

    mask = np.asarray(wall_mask, dtype=np.bool_)
    minimum_run = max(20, round(max(mask.shape) * 0.01))
    raw: list[tuple[float, float, float, float]] = []
    for horizontal in (True, False):
        # A separable run-support filter is equivalent to the long-axis evidence
        # needed here and avoids the quadratic cost of a large binary footprint.
        opened = (
            ndimage.uniform_filter1d(
                mask.astype(np.float32),
                size=minimum_run,
                axis=1 if horizontal else 0,
                mode="constant",
            )
            >= 0.95
        )
        raw.extend(
            _axis_run_segments(
                opened,
                horizontal=horizontal,
                minimum_run=minimum_run,
            )
        )

    horizontal = [line for line in raw if line[1] == line[3]]
    vertical = [line for line in raw if line[0] == line[2]]
    coordinate_tolerance = max(4.0, minimum_run * 0.45)
    gap_tolerance = max(24.0, max(mask.shape) * 0.075)
    horizontal = _merge_axis_segments(
        horizontal,
        horizontal=True,
        coordinate_tolerance=coordinate_tolerance,
        gap_tolerance=gap_tolerance,
    )
    vertical = _merge_axis_segments(
        vertical,
        horizontal=False,
        coordinate_tolerance=coordinate_tolerance,
        gap_tolerance=gap_tolerance,
    )
    horizontal = _collapse_thick_parallel(
        horizontal,
        mask,
        horizontal=True,
    )
    vertical = _collapse_thick_parallel(
        vertical,
        mask,
        horizontal=False,
    )

    split_horizontal: list[tuple[float, float, float, float]] = []
    split_vertical: list[tuple[float, float, float, float]] = []
    intersection_tolerance = max(8.0, minimum_run * 0.75)
    minimum_segment = max(10.0, minimum_run * 0.5)
    for x0, y, x1, _ in horizontal:
        cuts = [x0, x1]
        for x, y0, _, y1 in vertical:
            if x0 - intersection_tolerance <= x <= x1 + intersection_tolerance and (
                y0 - intersection_tolerance <= y <= y1 + intersection_tolerance
            ):
                cuts.append(float(np.clip(x, x0, x1)))
        cuts = sorted(set(round(value, 3) for value in cuts))
        split_horizontal.extend(
            (left, y, right, y)
            for left, right in zip(cuts, cuts[1:], strict=False)
            if right - left >= minimum_segment
        )
    for x, y0, _, y1 in vertical:
        cuts = [y0, y1]
        for x0, y, x1, _ in horizontal:
            if y0 - intersection_tolerance <= y <= y1 + intersection_tolerance and (
                x0 - intersection_tolerance <= x <= x1 + intersection_tolerance
            ):
                cuts.append(float(np.clip(y, y0, y1)))
        cuts = sorted(set(round(value, 3) for value in cuts))
        split_vertical.extend(
            (x, top, x, bottom)
            for top, bottom in zip(cuts, cuts[1:], strict=False)
            if bottom - top >= minimum_segment
        )
    return split_horizontal + split_vertical


class OnnxFloorPlanSemanticRecognizer:
    """Run one dense full-sheet semantic model without SVG or manual coordinates."""

    def __init__(self, model_path: str | Path, *, threads: int = 1) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.manifest_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.model_sha256 = sha256_file(self.model_path)
        if self.manifest.get("onnx_sha256") != self.model_sha256:
            raise ValueError("semantic ONNX artifact does not match its manifest")
        if self.manifest.get("output_contract") != "cubicasa44_dense_logits":
            raise ValueError("unsupported semantic output contract")
        self.model_version = str(self.manifest.get("model_version") or "")
        self.room_classes = tuple(self.manifest.get("room_classes") or ())
        self.icon_classes = tuple(self.manifest.get("icon_classes") or ())
        if len(self.room_classes) != 12 or len(self.icon_classes) != 11:
            raise ValueError("semantic manifest taxonomy is incomplete")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def recognize(
        self,
        image_path: str | Path,
        *,
        max_side: int = 1024,
    ) -> tuple[SemanticRecognitionResult, np.ndarray]:
        started = time.perf_counter()
        source_path = Path(image_path).expanduser().resolve()
        source = Image.open(source_path).convert("RGB")
        scale = min(1.0, max_side / max(source.size))
        width = max(64, round(source.width * scale / 32) * 32)
        height = max(64, round(source.height * scale / 32) * 32)
        resized = source.resize((width, height), Image.Resampling.LANCZOS)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor / 127.5 - 1.0)

        inference_started = time.perf_counter()
        prediction = self.session.run(None, {self.input_name: tensor})[0][0]
        inference_ms = (time.perf_counter() - inference_started) * 1000
        room_probability = _softmax(prediction[21:33], axis=0)
        icon_probability = _softmax(prediction[33:44], axis=0)
        room_class = room_probability.argmax(axis=0).astype(np.uint8)
        icon_class = icon_probability.argmax(axis=0).astype(np.uint8)
        room_original = np.asarray(
            Image.fromarray(room_class).resize(source.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        icon_original = np.asarray(
            Image.fromarray(icon_class).resize(source.size, Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        probability_original = np.stack(
            [
                np.asarray(
                    Image.fromarray(channel.astype(np.float32), mode="F").resize(
                        source.size, Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                for channel in icon_probability
            ]
        )

        minimum_area = max(12, round(source.width * source.height * 0.00001))
        detections: list[SemanticDetection] = []
        for class_index, class_name in enumerate(self.icon_classes[1:], start=1):
            records = _component_records(
                icon_original == class_index,
                probability_original[class_index],
                minimum_area=minimum_area,
            )
            for record in records:
                confidence = float(record["confidence"])
                pixel_area = int(record["pixel_area"])
                # Keep low-confidence evidence in the audit result, but flag it for
                # review instead of silently promoting a fragment to BIM geometry.
                review_required = confidence < 0.68 or pixel_area < minimum_area * 8
                detections.append(
                    SemanticDetection(
                        id=f"semantic:{class_index}:{len(detections)}",
                        class_name=class_name,
                        symbol_class=_SYMBOL_CLASS[class_name],
                        bbox_px=record["bbox_px"],
                        confidence=confidence,
                        pixel_area=pixel_area,
                        review_required=review_required,
                        promote_to_bim=not review_required,
                    )
                )
        # Door strokes can be fragmented while the same opening is classified as
        # one strong window region.  Fuse only when at least two independent door
        # components land inside the region; a single weak stroke cannot relabel it.
        consumed: set[str] = set()
        fused: list[SemanticDetection] = []
        for window in [item for item in detections if item.class_name == "Window"]:
            left, top, right, bottom = window.bbox_px
            supports = []
            for door in [item for item in detections if item.class_name == "Door"]:
                door_left, door_top, door_right, door_bottom = door.bbox_px
                center_x = (door_left + door_right) / 2
                center_y = (door_top + door_bottom) / 2
                if left <= center_x <= right and top <= center_y <= bottom:
                    supports.append(door)
            if len(supports) < 2:
                continue
            consumed.add(window.id)
            consumed.update(item.id for item in supports)
            fused.append(
                window.model_copy(
                    update={
                        "id": f"semantic:fused-door:{len(fused)}",
                        "class_name": "Door",
                        "symbol_class": "door",
                        "confidence": min(
                            window.confidence,
                            max(item.confidence for item in supports),
                        ),
                        "review_required": False,
                        "promote_to_bim": True,
                    }
                )
            )
        detections = [item for item in detections if item.id not in consumed] + fused
        counts = {
            name: sum(item.class_name == name for item in detections)
            for name in self.icon_classes[1:]
        }
        rooms: list[SemanticRoom] = []
        minimum_room_area = max(96, round(width * height * 0.0008))
        # Wall, background, outdoor, and railing pixels are not enclosed BIM spaces.
        room_class_indices = [3, 4, 5, 6, 7, 9, 10, 11]
        for class_index in room_class_indices:
            class_name = self.room_classes[class_index]
            for record in _room_records(
                room_class == class_index,
                room_probability[class_index],
                minimum_area=minimum_room_area,
                source_size=source.size,
            ):
                confidence = float(record["confidence"])
                pixel_area = int(record["pixel_area"])
                rooms.append(
                    SemanticRoom(
                        id=f"semantic:room:{class_index}:{len(rooms)}",
                        class_name=class_name,
                        polygon_px=record["polygon_px"],
                        confidence=confidence,
                        pixel_area=pixel_area,
                        review_required=(confidence < 0.52 or pixel_area < minimum_room_area * 1.5),
                    )
                )
        room_counts = {
            name: sum(item.class_name == name for item in rooms)
            for name in self.room_classes
            if name not in {"Background", "Outdoor", "Wall", "Railing"}
        }
        result = SemanticRecognitionResult(
            input_path=str(source_path),
            input_sha256=sha256_file(source_path),
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            license_scope=str(self.manifest.get("license_scope") or "unknown"),
            production_authorized=bool(self.manifest.get("production_authorized", False)),
            source_size=source.size,
            model_input_size=(width, height),
            wall_pixels=int((room_original == 2).sum()),
            wall_centerlines_px=_wall_centerlines(room_original == 2),
            detections=detections,
            rooms=rooms,
            counts=counts,
            room_counts=room_counts,
            inference_ms=round(inference_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
        ).finalize()
        return result, room_original == 2

    @staticmethod
    def wall_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelLineProposal]:
        lines = list(result.wall_centerlines_px)

        def supported(center_x: float, center_y: float) -> bool:
            for x0, y0, x1, y1 in lines:
                delta_x, delta_y = x1 - x0, y1 - y0
                length_squared = delta_x * delta_x + delta_y * delta_y
                if length_squared <= 1e-9:
                    continue
                fraction = ((center_x - x0) * delta_x + (center_y - y0) * delta_y) / length_squared
                if not 0 <= fraction <= 1:
                    continue
                nearest_x = x0 + fraction * delta_x
                nearest_y = y0 + fraction * delta_y
                if np.hypot(center_x - nearest_x, center_y - nearest_y) <= 35.0:
                    return True
            return False

        source_width, source_height = result.source_size
        for detection in result.detections:
            if not detection.promote_to_bim or detection.symbol_class not in {"door", "window"}:
                continue
            left, top, right, bottom = detection.bbox_px
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            if supported(center_x, center_y):
                continue
            margin = 24.0
            if right - left >= bottom - top:
                lines.append(
                    (
                        max(0.0, left - margin),
                        center_y,
                        min(float(source_width), right + margin),
                        center_y,
                    )
                )
            else:
                lines.append(
                    (
                        center_x,
                        max(0.0, top - margin),
                        center_x,
                        min(float(source_height), bottom + margin),
                    )
                )

        return [
            PixelLineProposal(
                id=f"semantic:wall:{index}",
                start_px=(line[0], line[1]),
                end_px=(line[2], line[3]),
                confidence=0.82,
                uncertainty=0.18,
                source_ref_ids=source_ref_ids,
                model_version=result.model_version,
                review_required=True,
            )
            for index, line in enumerate(lines)
        ]

    @staticmethod
    def symbol_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelSymbolProposal]:
        output = []
        for detection in result.detections:
            if not detection.promote_to_bim:
                continue
            left, top, right, bottom = detection.bbox_px
            output.append(
                PixelSymbolProposal(
                    id=detection.id,
                    symbol_class=detection.symbol_class,
                    center_px=((left + right) / 2, (top + bottom) / 2),
                    bbox_px=(left, top, right, bottom),
                    confidence=detection.confidence,
                    uncertainty=1.0 - detection.confidence,
                    source_ref_ids=source_ref_ids,
                    model_version=result.model_version,
                    review_required=detection.review_required,
                )
            )
        return output

    @staticmethod
    def room_proposals(
        result: SemanticRecognitionResult,
        *,
        source_ref_ids: list[str],
    ) -> list[PixelRoomProposal]:
        return [
            PixelRoomProposal(
                id=room.id,
                name=f"{room.class_name} {index + 1}",
                room_class=room.class_name,
                polygon_px=room.polygon_px,
                confidence=room.confidence,
                uncertainty=1.0 - room.confidence,
                source_ref_ids=source_ref_ids,
                model_version=result.model_version,
                review_required=room.review_required,
            )
            for index, room in enumerate(result.rooms)
        ]

    @staticmethod
    def render_overlay(
        image_path: str | Path,
        result: SemanticRecognitionResult,
        wall_mask: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        source = Image.open(image_path).convert("RGBA")
        paint = Image.new("RGBA", source.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(paint)
        wall = Image.fromarray(np.uint8(wall_mask) * 255, mode="L")
        wall_fill = Image.new("RGBA", source.size, (18, 139, 84, 88))
        paint.alpha_composite(Image.composite(wall_fill, Image.new("RGBA", source.size), wall))
        font = _font(max(12, round(source.width / 150)))
        for item in result.detections:
            left, top, right, bottom = item.bbox_px
            color = (
                (222, 122, 20, 255)
                if item.symbol_class in {"door", "window"}
                else (211, 55, 55, 255)
            )
            width = 2 if item.review_required else 4
            draw.rectangle((left, top, right, bottom), outline=color, width=width)
            if item.symbol_class not in {"door", "window"}:
                box = draw.textbbox((0, 0), item.class_name, font=font)
                label_width = box[2] - box[0] + 10
                label_top = max(0, top - (box[3] - box[1] + 7))
                draw.rounded_rectangle(
                    (left, label_top, left + label_width, top),
                    radius=3,
                    fill=(255, 252, 250, 238),
                    outline=color,
                    width=2,
                )
                draw.text(
                    (left + 5, label_top + 2),
                    item.class_name,
                    font=font,
                    fill=(132, 30, 30, 255),
                )
        for x0, y0, x1, y1 in result.wall_centerlines_px:
            draw.line((x0, y0, x1, y1), fill=(17, 92, 62, 210), width=2)
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(source, paint).convert("RGB").save(destination, optimize=True)
        return destination
