"""Evaluate the native local specialist without claiming final pipeline F1.

This fixture-only diagnostic uses direct whole-sheet manual ground truth. Doors
and windows are excluded because their mandatory wall host comes from the global
topology model; the exported pipeline evaluator remains the release authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from buili_plan2bim.local_element_inference import LocalElementOnnxRecognizer
from buili_plan2bim.pipeline_evaluation import _entity_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--sheet-id", action="append", default=[])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--discovery-threshold", type=float, default=0.78)
    args = parser.parse_args()

    recognizer = LocalElementOnnxRecognizer(
        args.model_path,
        threads=args.threads,
        require_production=False,
    )
    requested = set(args.sheet_id)
    sheets = []
    for graph_path in sorted(args.ground_truth_root.glob("cubi-*/benchmark-graph.json")):
        sheet_id = graph_path.parent.name
        if requested and sheet_id not in requested:
            continue
        manifest = json.loads(
            (graph_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        if not manifest["visual_review"].get("whole_sheet_reviewed"):
            raise ValueError(f"sheet is not whole-sheet reviewed: {sheet_id}")
        target = json.loads(graph_path.read_text(encoding="utf-8"))
        oracle_rooms = [
            SimpleNamespace(
                room_class=str(room.get("name") or room.get("room_class") or "other"),
                polygon_px=[tuple(point[:2]) for point in room.get("polygon", [])],
            )
            for room in target.get("rooms", [])
            if len(room.get("polygon", [])) >= 3
        ]
        oracle_walls = [
            SimpleNamespace(
                start_px=tuple(wall["from"][:2]),
                end_px=tuple(wall["to"][:2]),
                thickness_px=float(wall.get("thickness_px") or 4.0),
            )
            for wall in target.get("walls", [])
            if wall.get("from") and wall.get("to")
        ]
        with Image.open(manifest["source"]["image_path"]) as opened:
            image = opened.convert("RGB")
        prediction, diagnostics = recognizer.refine(
            image,
            [],
            discover_candidates=True,
            source_ref_ids=[manifest["source"]["image_sha256"]],
            discovery_threshold=args.discovery_threshold,
            host_walls=oracle_walls,
            room_regions=oracle_rooms,
        )
        fixtures = [
            item.model_dump(mode="json")
            for item in prediction
            if item.symbol_class not in {"door", "window"}
        ]
        semantic = _entity_scores(
            fixtures,
            target.get("fixtures", []),
            prediction_scale=1.0,
            kind="fixture",
            maximum_distance_px=32.0,
        )
        geometry = _entity_scores(
            fixtures,
            target.get("fixtures", []),
            prediction_scale=1.0,
            kind="fixture",
            maximum_distance_px=32.0,
            minimum_dimension_similarity=0.75,
        )
        sheets.append(
            {
                "sheet_id": sheet_id,
                "candidate_count": diagnostics.discovered_candidate_count,
                "accepted_fixture_count": len(fixtures),
                "semantic": semantic,
                "geometry_aware": geometry,
                "timings_ms": diagnostics.timings_ms,
            }
        )

    def aggregate(metric: str) -> dict[str, float | int]:
        matches = sum(int(sheet[metric]["matches"]) for sheet in sheets)
        prediction_count = sum(
            int(sheet[metric]["prediction_count"]) for sheet in sheets
        )
        target_count = sum(int(sheet[metric]["target_count"]) for sheet in sheets)
        precision = matches / max(1, prediction_count)
        recall = matches / max(1, target_count)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {
            "matches": matches,
            "prediction_count": prediction_count,
            "target_count": target_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    report = {
        "schema_version": "dajoong.direct-local-fixture-diagnostic.v1",
        "claim_scope": "local_specialist_diagnostic_not_final_pipeline_f1",
        "ground_truth_policy": "direct_visual_source_annotation_only",
        "model_version": recognizer.model_version,
        "model_sha256": recognizer.model_sha256,
        "aggregate": {
            "semantic": aggregate("semantic"),
            "geometry_aware": aggregate("geometry_aware"),
        },
        "sheets": sheets,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
