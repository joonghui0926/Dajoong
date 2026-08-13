"""Build whole-sheet Method v2 targets from isolated synthetic programs.

These targets are for topology pretraining only.  The function deliberately
rejects any payload that could be mistaken for visually reviewed real-drawing
ground truth or an evaluation record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .core.hashing import sha256_file, sha256_json
from .core.model.cad_evidence import (
    GLOBAL_PROGRAM_INPUT_CONTRACT,
    letterbox_content_bbox,
    pad_letterbox_content,
)
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
)
from .synthetic_pretraining import (
    SyntheticPretrainingSample,
    assert_synthetic_pretraining_only,
    audit_synthetic_pretraining_corpus,
)


def _scale_point(point: tuple[float, float], scale_x: float, scale_y: float) -> tuple[float, float]:
    return point[0] * scale_x, point[1] * scale_y


def _draw_line_mask(
    size: tuple[int, int],
    lines: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    scale_x: float,
    scale_y: float,
    width: int,
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for start, end in lines:
        draw.line(
            (*_scale_point(start, scale_x, scale_y), *_scale_point(end, scale_x, scale_y)),
            fill=255,
            width=width,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


def _polygon_boundary_lines(
    polygon: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))


def _junction_mask(
    sample: SyntheticPretrainingSample,
    size: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    endpoint_counts: dict[tuple[float, float], int] = {}
    for start, end in sample.walls:
        endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
        endpoint_counts[end] = endpoint_counts.get(end, 0) + 1
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for point, count in endpoint_counts.items():
        if count < 2:
            continue
        x, y = _scale_point(point, scale_x, scale_y)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=255)
    return np.asarray(image, dtype=np.float32) / 255.0


def _opening_mask(
    sample: SyntheticPretrainingSample,
    size: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for opening in sample.openings:
        x, y = _scale_point(opening.center_px, scale_x, scale_y)
        width = max(3.0, opening.width_px * (scale_x + scale_y) / 2)
        if opening.orientation == "horizontal":
            draw.line((x - width / 2, y, x + width / 2, y), fill=255, width=3)
        else:
            draw.line((x, y - width / 2, x, y + width / 2), fill=255, width=3)
    return np.asarray(image, dtype=np.float32) / 255.0


def _room_masks(
    sample: SyntheticPretrainingSample,
    size: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    interior = Image.new("L", size, 0)
    seeds = Image.new("L", size, 0)
    interior_draw = ImageDraw.Draw(interior)
    seed_draw = ImageDraw.Draw(seeds)
    for room in sample.rooms:
        polygon = [_scale_point(point, scale_x, scale_y) for point in room.polygon_px]
        interior_draw.polygon(polygon, fill=255)
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        seed_draw.ellipse(
            (center_x - 4, center_y - 4, center_x + 4, center_y + 4),
            fill=255,
        )
    return (
        np.asarray(seeds, dtype=np.float32) / 255.0,
        np.asarray(interior, dtype=np.float32) / 255.0,
    )


def _room_semantic_target(
    sample: SyntheticPretrainingSample,
    size: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    aliases = {
        "bath": "bathroom",
        "corridor": "hallway",
        "laundry": "utility",
    }
    image = Image.new("I", size, 0)
    draw = ImageDraw.Draw(image)
    for room in sample.rooms:
        label = aliases.get(room.room_class, room.room_class)
        class_index = (
            ROOM_PROGRAM_CLASSES.index(label)
            if label in ROOM_PROGRAM_CLASSES
            else ROOM_PROGRAM_CLASSES.index("other")
        )
        polygon = [_scale_point(point, scale_x, scale_y) for point in room.polygon_px]
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        radius = max(4.0, min(8.0, min(size) * 0.022))
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=class_index,
        )
    return np.asarray(image, dtype=np.int64)


def _element_program_targets(
    sample: SyntheticPretrainingSample,
    size: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantics = Image.new("I", size, 0)
    semantic_draw = ImageDraw.Draw(semantics)
    geometry = np.zeros((len(ELEMENT_GEOMETRY_CHANNELS), size[1], size[0]), dtype=np.float32)
    valid = np.zeros((size[1], size[0]), dtype=np.float32)

    def write_instance(
        class_name: str,
        bbox: tuple[float, float, float, float],
        yaw_deg: float,
    ) -> None:
        class_index = ELEMENT_PROGRAM_CLASSES.index(class_name)
        left = max(0, min(size[0] - 1, round(bbox[0] * scale_x)))
        top = max(0, min(size[1] - 1, round(bbox[1] * scale_y)))
        right = max(left + 1, min(size[0], round(bbox[2] * scale_x)))
        bottom = max(top + 1, min(size[1], round(bbox[3] * scale_y)))
        semantic_draw.rectangle((left, top, right - 1, bottom - 1), fill=class_index)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        yy, xx = np.mgrid[top:bottom, left:right]
        geometry[0, top:bottom, left:right] = (center_x - xx) / size[0]
        geometry[1, top:bottom, left:right] = (center_y - yy) / size[1]
        geometry[2, top:bottom, left:right] = np.log(max(1, right - left) / size[0])
        geometry[3, top:bottom, left:right] = np.log(max(1, bottom - top) / size[1])
        yaw = np.deg2rad(yaw_deg)
        geometry[4, top:bottom, left:right] = np.sin(yaw)
        geometry[5, top:bottom, left:right] = np.cos(yaw)
        valid[top:bottom, left:right] = 1.0

    for opening in sample.openings:
        half_width = opening.width_px / 2
        half_depth = 3.0
        if opening.orientation == "horizontal":
            bbox = (
                opening.center_px[0] - half_width,
                opening.center_px[1] - half_depth,
                opening.center_px[0] + half_width,
                opening.center_px[1] + half_depth,
            )
            yaw_deg = 0.0
        else:
            bbox = (
                opening.center_px[0] - half_depth,
                opening.center_px[1] - half_width,
                opening.center_px[0] + half_depth,
                opening.center_px[1] + half_width,
            )
            yaw_deg = 90.0
        write_instance(opening.kind, bbox, yaw_deg)
    for fixture in sample.fixtures:
        class_name = (
            fixture.fixture_type
            if fixture.fixture_type in ELEMENT_PROGRAM_CLASSES
            else "unknown"
        )
        write_instance(class_name, fixture.bbox_px, fixture.yaw_deg)
    return np.asarray(semantics, dtype=np.int64), geometry, valid


def build_synthetic_topology_target(
    annotation_path: str | Path,
    output_path: str | Path,
    *,
    target_size: int = 256,
) -> dict[str, Any]:
    if target_size < 64:
        raise ValueError("target_size must be at least 64")
    source_path = Path(annotation_path).expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    assert_synthetic_pretraining_only(payload)
    sample = SyntheticPretrainingSample.model_validate(payload)
    source_image = source_path.parent.parent / sample.image_path
    with Image.open(source_image) as image:
        source_width, source_height = image.size
    content_bbox = letterbox_content_bbox(
        (source_width, source_height),
        target_size,
    )
    left, top, right, bottom = content_bbox
    size = right - left, bottom - top
    scale_x = size[0] / source_width
    scale_y = size[1] / source_height
    wall = _draw_line_mask(
        size,
        sample.walls,
        scale_x=scale_x,
        scale_y=scale_y,
        width=3,
    )
    exterior = _draw_line_mask(
        size,
        _polygon_boundary_lines(sample.building_footprint_px),
        scale_x=scale_x,
        scale_y=scale_y,
        width=5,
    )
    junction = _junction_mask(
        sample,
        size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    opening = _opening_mask(
        sample,
        size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    room_seed, room_interior = _room_masks(
        sample,
        size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    target = np.stack((exterior, wall, junction, opening, room_seed, room_interior))
    room_semantics = _room_semantic_target(
        sample,
        size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    element_semantics, element_geometry, element_geometry_valid = _element_program_targets(
        sample,
        size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    target = pad_letterbox_content(target, target_size, content_bbox)
    room_semantics = pad_letterbox_content(
        room_semantics,
        target_size,
        content_bbox,
    )
    element_semantics = pad_letterbox_content(
        element_semantics,
        target_size,
        content_bbox,
    )
    element_geometry = pad_letterbox_content(
        element_geometry,
        target_size,
        content_bbox,
    )
    element_geometry_valid = pad_letterbox_content(
        element_geometry_valid,
        target_size,
        content_bbox,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        targets=target.astype(np.float32),
        channel_names=np.asarray(TOPOLOGY_TARGET_CHANNELS),
        room_semantics=room_semantics,
        room_classes=np.asarray(ROOM_PROGRAM_CLASSES),
        element_semantics=element_semantics,
        element_classes=np.asarray(ELEMENT_PROGRAM_CLASSES),
        element_geometry=element_geometry,
        element_geometry_valid=element_geometry_valid,
        element_geometry_channels=np.asarray(ELEMENT_GEOMETRY_CHANNELS),
    )
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.synthetic-topology-target.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "sample_id": sample.sample_id,
        "annotation_sha256": sha256_file(source_path),
        "target_path": destination.name,
        "target_sha256": sha256_file(destination),
        "shape": list(target.shape),
        "input_contract": GLOBAL_PROGRAM_INPUT_CONTRACT,
        "content_bbox": list(content_bbox),
        "channels": list(TOPOLOGY_TARGET_CHANNELS),
        "room_classes": list(ROOM_PROGRAM_CLASSES),
        "room_semantic_contract": "localized_label_seed_v1",
        "element_classes": list(ELEMENT_PROGRAM_CLASSES),
        "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "topology_positive_pixels": {
            channel: int((target[index] >= 0.5).sum())
            for index, channel in enumerate(TOPOLOGY_TARGET_CHANNELS)
        },
        "room_pixel_counts": {
            class_name: int((room_semantics == index).sum())
            for index, class_name in enumerate(ROOM_PROGRAM_CLASSES)
        },
        "element_pixel_counts": {
            class_name: int((element_semantics == index).sum())
            for index, class_name in enumerate(ELEMENT_PROGRAM_CLASSES)
        },
    }
    manifest["content_sha256"] = sha256_json(manifest)
    assert_synthetic_pretraining_only(manifest)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def build_synthetic_topology_target_corpus(
    corpus_root: str | Path,
    output_root: str | Path,
    *,
    target_size: int = 256,
) -> dict[str, Any]:
    source_root = Path(corpus_root).expanduser().resolve()
    corpus_manifest_path = source_root / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    assert_synthetic_pretraining_only(corpus_manifest)
    audit_synthetic_pretraining_corpus(
        source_root,
        require_complete_taxonomy=False,
        raise_on_error=True,
    )
    annotations = sorted((source_root / "annotations").glob("*.json"))
    if len(annotations) != int(corpus_manifest.get("sample_count", -1)):
        raise ValueError("synthetic corpus annotation count does not match its manifest")
    destination = Path(output_root).expanduser().resolve()
    records = []
    topology_positive_pixels = {
        channel: 0 for channel in TOPOLOGY_TARGET_CHANNELS
    }
    room_pixel_counts = {class_name: 0 for class_name in ROOM_PROGRAM_CLASSES}
    element_pixel_counts = {
        class_name: 0 for class_name in ELEMENT_PROGRAM_CLASSES
    }
    for annotation in annotations:
        target_path = destination / f"{annotation.stem}.npz"
        target_manifest = build_synthetic_topology_target(
            annotation,
            target_path,
            target_size=target_size,
        )
        records.append(
            {
                "sample_id": target_manifest["sample_id"],
                "target_path": target_path.name,
                "target_sha256": target_manifest["target_sha256"],
                "manifest_sha256": target_manifest["content_sha256"],
                "content_bbox": target_manifest["content_bbox"],
                "topology_positive_pixels": target_manifest[
                    "topology_positive_pixels"
                ],
                "room_pixel_counts": target_manifest["room_pixel_counts"],
                "element_pixel_counts": target_manifest["element_pixel_counts"],
            }
        )
        for channel, count in target_manifest["topology_positive_pixels"].items():
            topology_positive_pixels[channel] += int(count)
        for class_name, count in target_manifest["room_pixel_counts"].items():
            room_pixel_counts[class_name] += int(count)
        for class_name, count in target_manifest["element_pixel_counts"].items():
            element_pixel_counts[class_name] += int(count)
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.synthetic-topology-target-corpus.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "source_corpus_sha256": sha256_file(corpus_manifest_path),
        "target_size": target_size,
        "input_contract": GLOBAL_PROGRAM_INPUT_CONTRACT,
        "channels": list(TOPOLOGY_TARGET_CHANNELS),
        "room_classes": list(ROOM_PROGRAM_CLASSES),
        "room_semantic_contract": "localized_label_seed_v1",
        "element_classes": list(ELEMENT_PROGRAM_CLASSES),
        "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "topology_positive_pixels": topology_positive_pixels,
        "room_pixel_counts": room_pixel_counts,
        "element_pixel_counts": element_pixel_counts,
        "sample_count": len(records),
        "records": records,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    assert_synthetic_pretraining_only(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
