"""Audit whether benchmark entities are measurable in their reviewed source raster."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from buili_plan2bim.ground_truth import audit_benchmark_graph_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    sheets = []
    issue_counts: Counter[str] = Counter()
    for graph_path in sorted(args.ground_truth_root.glob("*/benchmark-graph.json")):
        manifest_path = graph_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        issues = audit_benchmark_graph_geometry(
            graph,
            image_size=(int(source["width_px"]), int(source["height_px"])),
            reviewed_plan_bbox_px=source.get("reviewed_plan_bbox_px"),
        )
        issue_counts.update(issue["code"] for issue in issues)
        sheets.append(
            {
                "sheet_id": graph_path.parent.name,
                "valid_for_f1": not issues,
                "issue_count": len(issues),
                "issues": issues,
            }
        )

    report = {
        "schema_version": "dajoong.benchmark-geometry-audit.v1",
        "policy": "fail_closed_source_coordinate_validation",
        "sheet_count": len(sheets),
        "valid_sheet_count": sum(sheet["valid_for_f1"] for sheet in sheets),
        "invalid_sheet_count": sum(not sheet["valid_for_f1"] for sheet in sheets),
        "issue_counts": dict(sorted(issue_counts.items())),
        "official_f1_allowed": all(sheet["valid_for_f1"] for sheet in sheets),
        "sheets": sheets,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sheets"}, indent=2))


if __name__ == "__main__":
    main()
