"""Run the pinned end-to-end Plan2BIM pipeline on the frozen 50-sheet set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.pipeline import ConversionConfig, Plan2BimConverter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--semantic-model", type=Path)
    parser.add_argument("--global-program-model", type=Path)
    parser.add_argument("--local-element-model", type=Path)
    parser.add_argument("--allow-research-global-program", action="store_true")
    parser.add_argument("--discover-native-candidates", action="store_true")
    parser.add_argument("--sheet-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pixels-per-meter", type=float, default=100.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.semantic_model is None and args.global_program_model is None:
        parser.error("provide --semantic-model or --global-program-model")
    if args.local_element_model is not None and args.global_program_model is None:
        parser.error("--local-element-model requires --global-program-model")
    converter = Plan2BimConverter(
        threads=args.threads,
        batch_size=8,
        semantic_model_path=args.semantic_model,
        allow_legacy_semantic_teacher=args.semantic_model is not None,
        global_program_model_path=args.global_program_model,
        local_element_model_path=args.local_element_model,
        allow_research_global_program=args.allow_research_global_program,
        discover_native_candidates=args.discover_native_candidates,
    )
    rows = []
    manifests = sorted(args.ground_truth_root.glob("cubi-*/manifest.json"))
    if args.sheet_id:
        requested = set(args.sheet_id)
        manifests = [path for path in manifests if path.parent.name in requested]
        missing = requested - {path.parent.name for path in manifests}
        if missing:
            parser.error(f"unknown sheet ids: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        manifests = manifests[: args.limit]
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sheet_id = str(manifest["source"]["sheet_id"])
        destination = args.output_root / sheet_id
        conversion_manifest = destination / "conversion-manifest.json"
        if args.resume and conversion_manifest.is_file():
            result = json.loads(conversion_manifest.read_text(encoding="utf-8"))
            rows.append(
                {
                    "sheet_id": sheet_id,
                    "status": "resumed",
                    "timings_ms": result.get("timings_ms", {}),
                    "entity_counts": result.get("entity_counts", {}),
                }
            )
            continue
        try:
            result = converter.convert(
                manifest["source"]["image_path"],
                destination,
                ConversionConfig(
                    project_id="benchmark50",
                    sheet_id=sheet_id,
                    pixels_per_meter=args.pixels_per_meter,
                    threads=args.threads,
                    allow_draft_ifc=True,
                ),
            )
            rows.append(
                {
                    "sheet_id": sheet_id,
                    "status": "converted",
                    "timings_ms": result.timings_ms,
                    "entity_counts": result.entity_counts,
                    "review_required": result.review_required,
                }
            )
        except Exception as error:  # noqa: BLE001 - benchmark records all failures.
            rows.append(
                {
                    "sheet_id": sheet_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        report = {
            "schema_version": "dajoong.benchmark50-pipeline-run.v1",
            "sheets": rows,
        }
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "run-summary.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
