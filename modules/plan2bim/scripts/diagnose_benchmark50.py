"""Build a reproducible driver table for the reviewed 50-sheet regression set.

This is diagnosis only: it never changes ground truth or model output.  Every row
joins the frozen manual reference, the emitted plan graph, the semantic runtime
record, the drawing complexity profile, and the final exported-graph scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_type(entity: dict[str, Any], *, kind: str) -> str:
    if kind == "opening":
        value = entity.get("type") or entity.get("opening_type") or "unknown"
    else:
        value = (
            entity.get("fixture_type")
            or entity.get("type")
            or entity.get("family_id")
            or "unknown"
        )
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if kind == "opening":
        return normalized
    if "electricalappliance" in normalized or "electrical-appliance" in normalized:
        return "appliance"
    if "sink" in normalized:
        return "sink"
    if "toilet" in normalized or normalized == "wc":
        return "toilet"
    if "saunabench" in normalized or normalized.endswith("-bench"):
        return "bench"
    if "closet" in normalized or "cabinet" in normalized or "wardrobe" in normalized:
        return "storage"
    if "shower" in normalized:
        return "shower"
    if "column" in normalized:
        return "column"
    if "stair" in normalized:
        return "stairs"
    return normalized.removeprefix("residential-")


def _pearson(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left) is not None and row.get(right) is not None
    ]
    if len(pairs) < 3:
        return None
    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_values)
        * sum((value - right_mean) ** 2 for value in right_values)
    )
    return numerator / denominator if denominator else None


def _f1(matches: int, predictions: int, targets: int) -> float:
    precision = matches / max(1, predictions)
    recall = matches / max(1, targets)
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _class_breakdown(
    *,
    kind: str,
    score: dict[str, Any],
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    target_by_id = {str(entity["id"]): entity for entity in targets}
    prediction_by_id = {str(entity["id"]): entity for entity in predictions}
    output: dict[str, dict[str, int]] = defaultdict(
        lambda: {"targets": 0, "predictions": 0, "matches": 0}
    )
    for entity in targets:
        output[_entity_type(entity, kind=kind)]["targets"] += 1
    for entity in predictions:
        output[_entity_type(entity, kind=kind)]["predictions"] += 1
    for pair in score["matched_pairs"]:
        target = target_by_id[str(pair["target_id"])]
        prediction = prediction_by_id[str(pair["prediction_id"])]
        target_type = _entity_type(target, kind=kind)
        prediction_type = _entity_type(prediction, kind=kind)
        # Keep mismatched aliases observable instead of silently assigning credit.
        key = target_type if target_type == prediction_type else f"{target_type}<-{prediction_type}"
        output[key]["matches"] += 1
    return dict(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--review-cohorts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    evaluation = _load(args.evaluation_report)
    run = _load(args.run_summary)
    reviewed_cohorts = _load(args.review_cohorts)
    tags_by_sheet = {
        str(sheet_id): [str(tag) for tag in tags]
        for sheet_id, tags in reviewed_cohorts["sheets"].items()
    }
    evaluation_by_id = {sheet["sheet_id"]: sheet for sheet in evaluation["sheets"]}
    run_by_id = {sheet["sheet_id"]: sheet for sheet in run["sheets"]}
    rows: list[dict[str, Any]] = []
    aggregate_classes: dict[str, Counter[str]] = {
        "opening_targets": Counter(),
        "opening_predictions": Counter(),
        "opening_matches": Counter(),
        "fixture_targets": Counter(),
        "fixture_predictions": Counter(),
        "fixture_matches": Counter(),
    }

    for sheet_id in sorted(evaluation_by_id):
        gt_dir = args.ground_truth_root / sheet_id
        prediction_dir = args.prediction_root / sheet_id
        manifest = _load(gt_dir / "manifest.json")
        target = _load(gt_dir / "benchmark-graph.json")
        prediction = _load(prediction_dir / "03-plan-graph.json")
        semantic = _load(prediction_dir / "00-semantic-recognition.json")
        complexity = _load(prediction_dir / "00-drawing-complexity.json")
        score = evaluation_by_id[sheet_id]
        run_record = run_by_id[sheet_id]
        source_width, source_height = [int(value) for value in semantic["source_size"]]
        model_width, model_height = [int(value) for value in semantic["model_input_size"]]
        source_long_side = max(source_width, source_height)
        model_long_side = max(model_width, model_height)
        target_entity_count = sum(
            len(target.get(kind, [])) for kind in ("walls", "rooms", "openings", "fixtures")
        )
        review_tags = tags_by_sheet.get(sheet_id, [])
        row = {
            "sheet_id": sheet_id,
            "manual_reference_sha256": manifest["manual_reference_sha256"],
            "review_tags": ";".join(review_tags),
            "is_multiple_levels_same_sheet": int(
                "multiple_levels_same_sheet" in review_tags
            ),
            "is_rotated": int("rotated" in review_tags),
            "is_markup_noise": int("markup_noise" in review_tags),
            "is_partial_crop": int("partial_crop" in review_tags),
            "is_irregular_geometry": int("irregular_geometry" in review_tags),
            "is_large_or_institutional": int(
                "large_extent" in review_tags or "institutional" in review_tags
            ),
            "source_width_px": source_width,
            "source_height_px": source_height,
            "source_megapixels": source_width * source_height / 1_000_000,
            "semantic_width_px": model_width,
            "semantic_height_px": model_height,
            "semantic_linear_scale": model_long_side / max(1, source_long_side),
            "semantic_area_scale": (model_width * model_height)
            / max(1, source_width * source_height),
            "complexity_score": float(complexity["complexity_score"]),
            "difficulty_class": complexity["difficulty_class"],
            "ink_fraction": float(complexity["ink_fraction"]),
            "significant_components": int(complexity["significant_components"]),
            "enclosed_region_count": int(complexity["enclosed_region_count"]),
            "target_entity_count": target_entity_count,
            "target_wall_count": len(target.get("walls", [])),
            "target_room_count": len(target.get("rooms", [])),
            "target_opening_count": len(target.get("openings", [])),
            "target_fixture_count": len(target.get("fixtures", [])),
            "prediction_wall_count": len(prediction.get("walls", [])),
            "prediction_room_count": len(prediction.get("rooms", [])),
            "prediction_opening_count": len(prediction.get("openings", [])),
            "prediction_fixture_count": len(prediction.get("fixtures", [])),
            "wall_count_ratio": len(prediction.get("walls", []))
            / max(1, len(target.get("walls", []))),
            "room_count_ratio": len(prediction.get("rooms", []))
            / max(1, len(target.get("rooms", []))),
            "opening_count_ratio": len(prediction.get("openings", []))
            / max(1, len(target.get("openings", []))),
            "fixture_count_ratio": len(prediction.get("fixtures", []))
            / max(1, len(target.get("fixtures", []))),
            "wall_centerline_f1_4px": float(
                score["wall_centerline_by_tolerance_px"]["4"]["f1"]
            ),
            "wall_centerline_f1_8px": float(
                score["wall_centerline_by_tolerance_px"]["8"]["f1"]
            ),
            "wall_centerline_f1_16px": float(
                score["wall_centerline_by_tolerance_px"]["16"]["f1"]
            ),
            "wall_footprint_f1": float(score["wall_footprint"]["f1"]),
            "room_union_f1": float(score["room_union"]["f1"]),
            "opening_f1": float(score["openings"]["f1"]),
            "fixture_f1": float(score["fixtures"]["f1"]),
            "primary_inference_ms": float(run_record["timings_ms"]["inference"]),
            "semantic_inference_ms": float(
                run_record["timings_ms"]["semantic_recognition"]
            ),
            "total_ms": float(run_record["timings_ms"]["total"]),
        }
        rows.append(row)

        for kind in ("opening", "fixture"):
            collection = f"{kind}s"
            class_result = _class_breakdown(
                kind=kind,
                score=score[collection],
                targets=target.get(collection, []),
                predictions=prediction.get(collection, []),
            )
            for class_name, counts in class_result.items():
                aggregate_classes[f"{kind}_targets"][class_name] += counts["targets"]
                aggregate_classes[f"{kind}_predictions"][class_name] += counts["predictions"]
                aggregate_classes[f"{kind}_matches"][class_name] += counts["matches"]

    if len(rows) != 50:
        raise ValueError(f"expected 50 joined sheets, received {len(rows)}")

    class_rows: list[dict[str, Any]] = []
    for kind in ("opening", "fixture"):
        classes = sorted(
            set(aggregate_classes[f"{kind}_targets"])
            | set(aggregate_classes[f"{kind}_predictions"])
        )
        for class_name in classes:
            targets = aggregate_classes[f"{kind}_targets"][class_name]
            predictions = aggregate_classes[f"{kind}_predictions"][class_name]
            matches = aggregate_classes[f"{kind}_matches"][class_name]
            class_rows.append(
                {
                    "kind": kind,
                    "class_name": class_name,
                    "targets": targets,
                    "predictions": predictions,
                    "matches": matches,
                    "precision": matches / max(1, predictions),
                    "recall": matches / max(1, targets),
                    "f1": _f1(matches, predictions, targets),
                    "missing": targets - matches,
                    "extra": predictions - matches,
                }
            )

    predictors = (
        "source_megapixels",
        "semantic_linear_scale",
        "complexity_score",
        "ink_fraction",
        "significant_components",
        "enclosed_region_count",
        "target_entity_count",
        "target_wall_count",
        "target_room_count",
        "target_opening_count",
        "target_fixture_count",
        "wall_count_ratio",
        "room_count_ratio",
        "opening_count_ratio",
        "fixture_count_ratio",
        "is_multiple_levels_same_sheet",
        "is_rotated",
        "is_markup_noise",
        "is_partial_crop",
        "is_irregular_geometry",
        "is_large_or_institutional",
    )
    outcomes = (
        "wall_centerline_f1_4px",
        "wall_footprint_f1",
        "room_union_f1",
        "opening_f1",
        "fixture_f1",
    )
    correlations = [
        {
            "predictor": predictor,
            "outcome": outcome,
            "pearson_r": _pearson(rows, predictor, outcome),
        }
        for predictor in predictors
        for outcome in outcomes
    ]
    strongest = sorted(
        (item for item in correlations if item["pearson_r"] is not None),
        key=lambda item: abs(float(item["pearson_r"])),
        reverse=True,
    )[:20]
    all_tags = sorted({tag for tags in tags_by_sheet.values() for tag in tags})
    reviewed_cohort_metrics = []
    for tag in all_tags:
        tagged = [row for row in rows if tag in str(row["review_tags"]).split(";")]
        if not tagged:
            continue
        reviewed_cohort_metrics.append(
            {
                "tag": tag,
                "sheet_count": len(tagged),
                **{
                    outcome: statistics.fmean(float(row[outcome]) for row in tagged)
                    for outcome in outcomes
                },
            }
        )
    driver_summary = {
        "schema_version": "dajoong.benchmark50-driver-diagnosis.v1",
        "sheet_count": len(rows),
        "semantic_downscaled_sheet_count": sum(
            float(row["semantic_linear_scale"]) < 0.999 for row in rows
        ),
        "minimum_semantic_linear_scale": min(
            float(row["semantic_linear_scale"]) for row in rows
        ),
        "maximum_source_megapixels": max(float(row["source_megapixels"]) for row in rows),
        "strongest_correlations": strongest,
        "all_correlations": correlations,
        "direct_review_cohort_metrics": reviewed_cohort_metrics,
        "lowest_f1_sheets": {
            outcome: [
                {"sheet_id": row["sheet_id"], "f1": row[outcome]}
                for row in sorted(rows, key=lambda item: float(item[outcome]))[:5]
            ]
            for outcome in outcomes
        },
        "class_recall_gaps": sorted(
            class_rows,
            key=lambda item: (int(item["missing"]), int(item["targets"])),
            reverse=True,
        ),
        "diagnostic_only": True,
        "ground_truth_mutated": False,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "diagnostic-dataset.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (args.output_root / "diagnostic-dataset.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "entity-class-breakdown.json").write_text(
        json.dumps(class_rows, indent=2), encoding="utf-8"
    )
    (args.output_root / "driver-summary.json").write_text(
        json.dumps(driver_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(driver_summary, indent=2))


if __name__ == "__main__":
    main()
