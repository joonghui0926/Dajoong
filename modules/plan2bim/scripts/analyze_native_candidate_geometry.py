"""Measure proposal geometry ceiling before semantic classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from buili_plan2bim.local_element_candidates import mine_native_element_candidates
from buili_plan2bim.ground_truth import validate_ground_truth_manifest


def _box(entity: dict[str, object]) -> tuple[float, float, float, float] | None:
    polygon = entity.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
    if len(points) < 3:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-9, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--sheet-id", action="append", default=[])
    parser.add_argument("--maximum-candidates", type=int, default=4096)
    args = parser.parse_args()
    requested = set(args.sheet_id)
    sheets = []
    for graph_path in sorted(args.ground_truth_root.glob("cubi-*/benchmark-graph.json")):
        sheet_id = graph_path.parent.name
        if requested and sheet_id not in requested:
            continue
        manifest = json.loads(
            (graph_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        validate_ground_truth_manifest(manifest)
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        target_entities = [
            fixture for fixture in graph.get("fixtures", []) if _box(fixture) is not None
        ]
        targets = [_box(entity) for entity in target_entities]
        with Image.open(manifest["source"]["image_path"]) as opened:
            candidates, diagnostics = mine_native_element_candidates(
                opened.convert("RGB"),
                source_ref_ids=[manifest["source"]["image_sha256"]],
                maximum_candidates=args.maximum_candidates,
            )
        candidate_boxes = [candidate.bbox_px for candidate in candidates]
        overlap = np.asarray(
            [[_iou(target, candidate) for candidate in candidate_boxes] for target in targets],
            dtype=np.float64,
        )
        rows, columns = linear_sum_assignment(1.0 - overlap)
        assigned = overlap[rows, columns]
        best = overlap.max(axis=1) if overlap.size else np.zeros(len(targets))
        per_target = []
        for index, entity in enumerate(target_entities):
            per_target.append(
                {
                    "target_index": index,
                    "fixture_type": str(entity.get("fixture_type") or "unknown"),
                    "best_candidate_iou": float(best[index]),
                }
            )
        sheet = {
            "sheet_id": sheet_id,
            "candidate_count": diagnostics.candidate_count,
            "target_count": len(targets),
            "best_candidate_recall": {
                str(threshold): float((best >= threshold).mean())
                for threshold in (0.25, 0.5, 0.75)
            },
            "one_to_one_recall": {
                str(threshold): float((assigned >= threshold).sum() / max(1, len(targets)))
                for threshold in (0.25, 0.5, 0.75)
            },
            "median_best_iou": float(np.median(best)),
            "per_target": per_target,
        }
        sheets.append(sheet)
    report = {
        "schema_version": "dajoong.native-candidate-geometry-diagnostic.v1",
        "ground_truth_policy": "direct_visual_source_annotation_only",
        "sheets": sheets,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sheets, indent=2))


if __name__ == "__main__":
    main()
