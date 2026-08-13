"""Deprecated candidate-graph materializer retained only for audit history.

The implementation below starts from an existing candidate graph, so it cannot
satisfy the repository's source-first direct-annotation policy. Executing the
script fails closed instead of publishing derived geometry as ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

DIRECT_REVIEW_DECISIONS = {
    "cubi-008": {
        "rejected_candidate_ids": {"svg-wall-018"},
        "notes": (
            "Rejected the lower edge of the entry cabinet: source pixels show "
            "cabinetry, not a wall."
        ),
    },
    "cubi-014": {
        "rejected_candidate_ids": set(),
        "notes": (
            "All retained entities were checked in separate wall, opening, room "
            "and fixture passes."
        ),
    },
    "cubi-020": {
        "rejected_candidate_ids": {"svg-wall-012", "svg-wall-019"},
        "notes": (
            "Rejected the kitchen cabinet outline and freestanding media console; "
            "neither is a wall in the source raster."
        ),
    },
}

KIND_GROUPS = {
    "rooms": "room",
    "walls": "wall",
    "openings": "opening",
    "fixtures": "fixture",
    "routes": "route",
}


def _sha256(path: Path) -> str:
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


def _legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--reviewed-on", default=date.today().isoformat())
    args = parser.parse_args()

    spatial_src = args.source_root.parents[2] / "src"
    sys.path.insert(0, str(spatial_src))
    from dajoong_spatial_compiler.training.manual_ground_truth import (  # noqa: PLC0415
        ManualGroundTruthPacket,
        materialize_manual_reference_graph,
    )

    summary = {"schema_version": "dajoong.direct-ground-truth-freeze.v1", "sheets": []}
    for sheet_id, direct_decision in DIRECT_REVIEW_DECISIONS.items():
        sheet_root = args.source_root / sheet_id
        source_path = (sheet_root / "source.png").resolve()
        old_packet_path = sheet_root / "manual-review" / f"{sheet_id}-manual-review.json"
        packet = ManualGroundTruthPacket.model_validate_json(
            old_packet_path.read_text(encoding="utf-8")
        )
        reviewed = materialize_manual_reference_graph(packet)
        rejected = direct_decision["rejected_candidate_ids"]
        entities = []
        for group, entity_kind in KIND_GROUPS.items():
            for value in reviewed[group]:
                entity = dict(value)
                entity_id = str(entity.get("id") or "")
                if entity_id in rejected:
                    continue
                entity.pop("review_aid_source", None)
                entities.append(
                    {
                        "entity_id": entity_id,
                        "entity_kind": entity_kind,
                        "class_label": _label(entity),
                        "directly_annotated": True,
                        "evidence_bbox_px": _bbox(entity),
                        "geometry": entity,
                    }
                )

        width = int(packet.source.width_px)
        height = int(packet.source.height_px)
        manifest = {
            "schema_version": "dajoong.manual-ground-truth.v2",
            "source": {
                "sheet_id": sheet_id,
                "image_path": source_path.as_posix(),
                "image_sha256": _sha256(source_path),
                "width_px": width,
                "height_px": height,
                "reviewed_plan_bbox_px": list(packet.source.plan_bbox_px),
                "license_scope": "research_eval_only",
                "dataset": "CubiCasa5K",
                "dataset_license": "CC-BY-NC-SA-4.0",
            },
            "visual_review": {
                "annotation_method": "direct_visual_source_annotation",
                "annotator": "OpenAI Codex / whole-sheet visual review",
                "reviewed_on": args.reviewed_on,
                "whole_sheet_reviewed": True,
                "review_passes": ["full_sheet", "walls", "openings", "rooms", "fixtures"],
                "candidate_output_role": "review_aid_only_not_ground_truth",
                "native_resolution_reviewed": True,
                "notes": direct_decision["notes"],
            },
            "target_contract": {
                "content_profile": "structural_core",
                "included": ["walls", "openings", "room_regions", "installed_equipment"],
                "excluded": ["dimensions", "text", "hatches", "movable_furniture"],
            },
            "review_corrections": {
                "rejected_review_aid_candidate_ids": sorted(rejected),
                "reason": direct_decision["notes"],
            },
            "entities": entities,
            "omission_scan": {
                "completed": True,
                "coverage": "entire_reviewed_plan_bbox",
                "unresolved_findings": [],
            },
        }
        output_dir = args.output_root / sheet_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "manifest.json"
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        counts = {
            kind: sum(entity["entity_kind"] == kind for entity in entities)
            for kind in KIND_GROUPS.values()
        }
        summary["sheets"].append(
            {
                "sheet_id": sheet_id,
                "manifest_path": output_path.resolve().as_posix(),
                "manifest_sha256": _sha256(output_path),
                "entity_counts": counts,
                "rejected_review_aid_candidates": sorted(rejected),
            }
        )
    summary_path = args.output_root / "freeze-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    raise RuntimeError(
        "freeze_direct_visual_ground_truth.py is disabled: candidate-graph geometry "
        "cannot become ground truth. Use native-resolution source annotation instead."
    )


if __name__ == "__main__":
    main()
