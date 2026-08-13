"""Deprecated adapter for the legacy 50-sheet SVG review-aid corpus.

This file intentionally fails closed.  The old implementation copied geometry
materialized from the CubiCasa SVG review aid and described it as direct visual
ground truth.  It also allowed incompatible raster/SVG coordinate frames to
enter the evaluator.  Ground truth must now be authored with
``freeze_direct_visual_ground_truth.py`` from independent, source-pixel
annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

KIND_GROUPS = {
    "rooms": "room",
    "walls": "wall",
    "openings": "opening",
    "fixtures": "fixture",
    "routes": "route",
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(entity: dict[str, Any]) -> list[float]:
    points = entity.get("polygon") or []
    if not points and entity.get("from") and entity.get("to"):
        points = [entity["from"], entity["to"]]
    if not points and entity.get("center_m"):
        x, y = entity["center_m"][:2]
        return [float(x) - 3, float(y) - 3, float(x) + 3, float(y) + 3]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _label(entity: dict[str, Any]) -> str:
    return str(
        entity.get("name")
        or entity.get("wall_class")
        or entity.get("opening_type")
        or entity.get("fixture_type")
        or entity.get("route_type")
        or "unknown"
    )


def _benchmark_graph(reviewed: dict[str, Any]) -> dict[str, Any]:
    level_id = "level-1"
    return {
        "schema_version": "dajoong.direct-visual-benchmark-graph.v2",
        "sheet_id": reviewed["sheet_id"],
        "levels": [{"id": level_id, "name": "Level 1"}],
        "rooms": [{**item, "level_id": level_id} for item in reviewed["rooms"]],
        "walls": [{**item, "level_id": level_id} for item in reviewed["walls"]],
        "openings": [{**item, "level_id": level_id} for item in reviewed["openings"]],
        "fixtures": [{**item, "level_id": level_id} for item in reviewed["fixtures"]],
        "routes": [{**item, "level_id": level_id} for item in reviewed["routes"]],
        "manual_reference_sha256": reviewed["manual_reference_sha256"],
    }


def main() -> None:
    raise RuntimeError(
        "legacy benchmark adapter disabled: SVG-derived review aids cannot be "
        "promoted to direct visual ground truth; use "
        "freeze_direct_visual_ground_truth.py with independent source-pixel "
        "annotations"
    )


def _legacy_main() -> None:
    """Retained only for forensic reproducibility; never call for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    compiler_root = args.compiler_root.resolve()
    output_root = args.output_root.resolve()
    reference_root = (
        compiler_root / "artifacts" / "real-drawing-benchmark-50" / "cubicasa5k"
    )
    sys.path.insert(0, str(compiler_root / "src"))
    from dajoong_spatial_compiler.training.manual_ground_truth import (  # noqa: PLC0415
        ManualGroundTruthPacket,
        materialize_manual_reference_graph,
    )

    summary = json.loads(
        (reference_root / "manual-reference-summary.json").read_text(encoding="utf-8")
    )
    if summary.get("sheet_count") != 50 or not all(
        item.get("ready_for_evaluation") for item in summary["samples"]
    ):
        raise ValueError("the audited 50-sheet manual reference is not ready")

    old_cwd = Path.cwd()
    output_rows = []
    try:
        os.chdir(compiler_root)
        for sample in summary["samples"]:
            sheet_id = sample["sample_id"]
            sheet_root = reference_root / sheet_id
            packet_path = sheet_root / "manual-review" / f"{sheet_id}-manual-review.json"
            packet = ManualGroundTruthPacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            if packet.content_sha256 != sample["manual_reference_sha256"]:
                raise ValueError(f"{sheet_id}: manual reference hash changed")
            reviewed = materialize_manual_reference_graph(packet)
            source_path = (sheet_root / "source.png").resolve()
            entities = []
            for group, entity_kind in KIND_GROUPS.items():
                for value in reviewed[group]:
                    entity = dict(value)
                    entity.pop("review_aid_source", None)
                    entities.append(
                        {
                            "entity_id": str(entity.get("id") or ""),
                            "entity_kind": entity_kind,
                            "class_label": _label(entity),
                            "directly_annotated": True,
                            "evidence_bbox_px": _bbox(entity),
                            "geometry": entity,
                        }
                    )
            output_dir = output_root / sheet_id
            output_dir.mkdir(parents=True, exist_ok=True)
            benchmark = _benchmark_graph(reviewed)
            benchmark_path = output_dir / "benchmark-graph.json"
            benchmark_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
            manifest = {
                "schema_version": "dajoong.manual-ground-truth.v2",
                "source": {
                    "sheet_id": sheet_id,
                    "image_path": source_path.as_posix(),
                    "image_sha256": _sha256(source_path),
                    "width_px": packet.source.width_px,
                    "height_px": packet.source.height_px,
                    "reviewed_plan_bbox_px": list(packet.source.plan_bbox_px),
                    "license_scope": "research_eval_only",
                    "dataset": "CubiCasa5K official test split",
                    "dataset_license": "CC-BY-NC-SA-4.0",
                },
                "visual_review": {
                    "annotation_method": "direct_visual_source_annotation",
                    "annotator": "OpenAI Codex / native whole-sheet visual review",
                    "reviewed_on": "2026-08-11",
                    "whole_sheet_reviewed": True,
                    "native_resolution_reviewed": True,
                    "review_passes": [
                        "full_sheet",
                        "rooms",
                        "walls",
                        "openings",
                        "fixtures",
                    ],
                    "candidate_output_role": "review_aid_only_not_ground_truth",
                },
                "target_contract": {
                    "content_profile": "structural_core",
                    "included": [
                        "wall_geometry",
                        "opening_entities",
                        "room_regions",
                        "installed_and_structural_objects",
                    ],
                    "excluded": ["dimensions", "text", "hatches", "movable_furniture"],
                },
                "entities": entities,
                "omission_scan": {
                    "completed": True,
                    "coverage": "twelve-cell entire-source scan",
                    "unresolved_findings": [],
                },
                "manual_reference_sha256": packet.content_sha256,
            }
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            output_rows.append(
                {
                    "sheet_id": sheet_id,
                    "manifest_sha256": _sha256(manifest_path),
                    "benchmark_graph_sha256": _sha256(benchmark_path),
                    "entity_count": len(entities),
                }
            )
    finally:
        os.chdir(old_cwd)

    freeze_summary = {
        "schema_version": "dajoong.benchmark50-ground-truth-freeze.v1",
        "sheet_count": len(output_rows),
        "entity_count": sum(row["entity_count"] for row in output_rows),
        "source_review_summary_sha256": _sha256(
            reference_root / "manual-reference-summary.json"
        ),
        "sheets": output_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "freeze-summary.json").write_text(
        json.dumps(freeze_summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sheet_count": len(output_rows),
                "entity_count": freeze_summary["entity_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
