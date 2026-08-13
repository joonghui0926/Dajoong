"""Create a compact, hash-linked summary for the 50-sheet regression run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _micro_entity(sheets: list[dict[str, Any]], kind: str) -> dict[str, float | int]:
    matches = sum(int(sheet[kind]["matches"]) for sheet in sheets)
    prediction_count = sum(int(sheet[kind]["prediction_count"]) for sheet in sheets)
    target_count = sum(int(sheet[kind]["target_count"]) for sheet in sheets)
    precision = matches / max(1, prediction_count)
    recall = matches / max(1, target_count)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "prediction_count": prediction_count,
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _worst(sheets: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    def value(sheet: dict[str, Any]) -> float:
        item: Any = sheet
        for key in path:
            item = item[key]
        return float(item)

    sheet = min(sheets, key=value)
    return {"sheet_id": sheet["sheet_id"], "f1": value(sheet)}


def _distribution(
    sheets: list[dict[str, Any]], path: tuple[str, ...]
) -> dict[str, float | int]:
    """Expose cross-sheet stability so a strong macro average cannot hide failures."""

    values: list[float] = []
    for sheet in sheets:
        item: Any = sheet
        for key in path:
            item = item[key]
        values.append(float(item))
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[int((len(ordered) - 1) * fraction)]

    return {
        "sheet_count": len(ordered),
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "sheets_below_0_90": sum(value < 0.90 for value in ordered),
        "sheets_below_0_95": sum(value < 0.95 for value in ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_freeze", type=Path)
    parser.add_argument("pipeline_run", type=Path)
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.ground_truth_freeze.read_text(encoding="utf-8"))
    run = json.loads(args.pipeline_run.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    if evaluation.get("content_profile") != "full_editable_bim":
        raise ValueError(
            "official product F1 requires a full_editable_bim evaluation profile"
        )
    integrity = evaluation.get("ground_truth_integrity")
    if not isinstance(integrity, dict) or integrity.get("passed") is not True:
        raise ValueError("official product F1 requires passed ground-truth integrity")
    if freeze.get("official_f1_allowed") is not True:
        raise ValueError("ground-truth freeze is not authorized for official F1")
    sheets = evaluation["sheets"]
    if freeze["sheet_count"] != 50 or len(run["sheets"]) != 50 or len(sheets) != 50:
        raise ValueError("all three benchmark stages must contain exactly 50 sheets")
    failures = [item for item in run["sheets"] if item["status"] == "failed"]
    timings = [float(item["timings_ms"]["total"]) for item in run["sheets"]]
    summary = {
        "schema_version": "dajoong.benchmark50-regression-summary.v2",
        "truth": {
            "sheet_count": freeze["sheet_count"],
            "entity_count": freeze["entity_count"],
            "review_method": "native-source direct visual review with typed overlays",
            "reference_standard": "single-reviewer research evaluation",
            "commercial_training_allowed": False,
        },
        "execution": {
            "converted_or_resumed": len(run["sheets"]) - len(failures),
            "failed": len(failures),
            "cpu_threads": 8,
            "total_ms": {
                "median": statistics.median(timings),
                "mean": statistics.fmean(timings),
                "maximum": max(timings),
            },
            "production_release_claim": False,
        },
        "metrics": {
            **evaluation["aggregate"],
            "opening_entity_micro": _micro_entity(sheets, "openings"),
            "fixture_entity_micro": _micro_entity(sheets, "fixtures"),
            "opening_geometry_aware_micro": _micro_entity(
                sheets,
                "openings_geometry_aware",
            ),
            "fixture_geometry_aware_micro": _micro_entity(
                sheets,
                "fixtures_geometry_aware",
            ),
        },
        "cross_sheet_distribution": {
            "wall_centerline_4px": _distribution(
                sheets, ("wall_centerline_by_tolerance_px", "4", "f1")
            ),
            "wall_footprint": _distribution(sheets, ("wall_footprint", "f1")),
            "room_union": _distribution(sheets, ("room_union", "f1")),
            "openings": _distribution(sheets, ("openings", "f1")),
            "fixtures": _distribution(sheets, ("fixtures", "f1")),
            "openings_geometry_aware": _distribution(
                sheets,
                ("openings_geometry_aware", "f1"),
            ),
            "fixtures_geometry_aware": _distribution(
                sheets,
                ("fixtures_geometry_aware", "f1"),
            ),
        },
        "worst_sheets": {
            "wall_centerline_4px": _worst(
                sheets, ("wall_centerline_by_tolerance_px", "4", "f1")
            ),
            "room_union": _worst(sheets, ("room_union", "f1")),
            "openings": _worst(sheets, ("openings", "f1")),
            "fixtures": _worst(sheets, ("fixtures", "f1")),
            "openings_geometry_aware": _worst(
                sheets,
                ("openings_geometry_aware", "f1"),
            ),
            "fixtures_geometry_aware": _worst(
                sheets,
                ("fixtures_geometry_aware", "f1"),
            ),
        },
        "integrity": {
            "ground_truth_freeze_sha256": _sha256(args.ground_truth_freeze),
            "pipeline_run_sha256": _sha256(args.pipeline_run),
            "evaluation_report_sha256": _sha256(args.evaluation_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
