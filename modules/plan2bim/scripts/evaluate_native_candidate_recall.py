"""Audit native proposal recall against direct whole-sheet visual ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from buili_plan2bim.local_element_candidates import (
    candidate_ledger_iou_recall,
    candidate_ledger_recall,
    mine_native_element_candidates,
)
from buili_plan2bim.ground_truth import (
    GroundTruthPolicyError,
    audit_benchmark_graph_geometry,
    validate_ground_truth_manifest,
)


def _entity_box(entity: dict[str, object]) -> tuple[float, float, float, float] | None:
    polygon = entity.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
    if len(points) < 3:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--maximum-candidates", type=int, default=4096)
    args = parser.parse_args()
    sheets = []
    invalid_sheets = []
    total_targets = 0
    total_matches = 0
    strict_total_matches = 0
    for target_path in sorted(args.ground_truth_root.glob("cubi-*/benchmark-graph.json")):
        manifest_path = target_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            validate_ground_truth_manifest(manifest)
        except GroundTruthPolicyError as error:
            invalid_sheets.append(
                {
                    "sheet_id": target_path.parent.name,
                    "reason": "ground_truth_provenance_invalid",
                    "issues": [{"code": "policy_rejected", "message": str(error)}],
                }
            )
            continue
        if not manifest["visual_review"]["whole_sheet_reviewed"]:
            raise ValueError(f"sheet is not whole-sheet reviewed: {target_path.parent.name}")
        graph = json.loads(target_path.read_text(encoding="utf-8"))
        image_size = (
            int(manifest["source"]["width_px"]),
            int(manifest["source"]["height_px"]),
        )
        geometry_issues = audit_benchmark_graph_geometry(
            graph,
            image_size=image_size,
            reviewed_plan_bbox_px=manifest["source"].get("reviewed_plan_bbox_px"),
        )
        if geometry_issues:
            invalid_sheets.append(
                {
                    "sheet_id": target_path.parent.name,
                    "reason": "ground_truth_geometry_invalid",
                    "issues": geometry_issues,
                }
            )
            continue
        targets = [
            box
            for fixture in graph.get("fixtures", [])
            if (box := _entity_box(fixture)) is not None
        ]
        source = Path(manifest["source"]["image_path"])
        with Image.open(source) as image:
            candidates, diagnostics = mine_native_element_candidates(
                image.convert("L"),
                source_ref_ids=[manifest["source"]["image_sha256"]],
                maximum_candidates=args.maximum_candidates,
            )
        score = candidate_ledger_recall(candidates, targets)
        strict_score = candidate_ledger_iou_recall(candidates, targets)
        missed_fixture_types = [
            str(graph["fixtures"][index].get("fixture_type") or "unknown")
            for index in score["missed_target_indices"]
        ]
        total_targets += score["target_count"]
        total_matches += score["matched_target_count"]
        strict_total_matches += strict_score["matched_target_count"]
        sheets.append(
            {
                "sheet_id": target_path.parent.name,
                "candidate_count": diagnostics.candidate_count,
                **score,
                "strict_iou_0_50": strict_score,
                "missed_fixture_types": missed_fixture_types,
            }
        )
    report = {
        "schema_version": "dajoong.native-candidate-recall-evaluation.v1",
        "ground_truth_policy": "direct_visual_source_annotation_only",
        "evaluation_status": "valid" if sheets else "invalid_no_eligible_ground_truth",
        "sheet_count": len(sheets),
        "discovered_sheet_count": len(sheets) + len(invalid_sheets),
        "invalid_sheet_count": len(invalid_sheets),
        "micro_recall": None if total_targets == 0 else total_matches / total_targets,
        "strict_iou_0_50_micro_recall": (
            None if total_targets == 0 else strict_total_matches / total_targets
        ),
        "target_count": total_targets,
        "matched_target_count": total_matches,
        "strict_iou_0_50_matched_target_count": strict_total_matches,
        "official_f1_allowed": False,
        "claim_scope": "proposal_geometry_diagnostic_only",
        "invalid_sheets": invalid_sheets,
        "sheets": sheets,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "sheets"}, indent=2))


if __name__ == "__main__":
    main()
