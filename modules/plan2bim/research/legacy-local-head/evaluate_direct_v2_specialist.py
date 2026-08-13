"""Evaluate a specialist on frozen direct-visual v2 references without leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dajoong_spatial_compiler.hashing import sha256_json
from dajoong_spatial_compiler.training.aec_evaluation import evaluate_aec_specialist

PLUMBING_TERMS = ("bathtub", "faucet", "shower", "sink", "toilet", "urinal")


def _extent(entity: dict[str, Any]) -> tuple[float, float]:
    polygon = entity.get("polygon") or []
    if not polygon:
        return 8.0, 8.0
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return max(xs) - min(xs), max(ys) - min(ys)


def _graph(manifest: dict[str, Any]) -> dict[str, Any]:
    level_id = "level-1"
    groups = {kind: [] for kind in ("room", "wall", "opening", "fixture", "route")}
    for reviewed in manifest["entities"]:
        entity = {**reviewed["geometry"], "level_id": level_id}
        groups[reviewed["entity_kind"]].append(entity)
    openings = []
    for entity in groups["opening"]:
        width, height = _extent(entity)
        openings.append(
            {
                **entity,
                "type": entity.get("opening_type", "unknown"),
                "width_m": max(width, height),
            }
        )
    fixtures = []
    for entity in groups["fixture"]:
        fixture_type = str(entity.get("fixture_type", "unknown"))
        lowered = fixture_type.lower()
        if lowered == "stairs":
            family_id = "stair"
        elif any(term in lowered for term in PLUMBING_TERMS):
            family_id = "plumbing-fixture"
        else:
            family_id = "unknown-symbol"
        width, height = _extent(entity)
        fixtures.append(
            {**entity, "family_id": family_id, "size_m": [max(width, 3), max(height, 3), 1]}
        )
    return {
        "schema_version": "dajoong.direct-visual-benchmark-graph.v2",
        "sheet_id": manifest["source"]["sheet_id"],
        "levels": [{"id": level_id, "name": "Level 1"}],
        "rooms": groups["room"],
        "walls": groups["wall"],
        "openings": openings,
        "fixtures": fixtures,
        "routes": groups["route"],
        "vertical_connections": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    reference_hashes = []
    for manifest_path in sorted(args.ground_truth_root.glob("cubi-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sheet_id = str(manifest["source"]["sheet_id"])
        graph_path = manifest_path.parent / "benchmark-graph.json"
        graph_path.write_text(json.dumps(_graph(manifest), indent=2), encoding="utf-8")
        rows.append(
            {
                "sample_id": sheet_id,
                "split": "test",
                "source_family_id": "cubicasa5k-direct-visual-v2",
                "image_path": manifest["source"]["image_path"],
                "plan_graph_path": graph_path.resolve().as_posix(),
                "level_id": "level-1",
                "transform": {
                    "metric_to_pixel": [1, 0, 0, 0, 1, 0],
                    "pixels_per_meter": 1,
                    "origin_px": [0, 0],
                },
            }
        )
        reference_hashes.append(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    benchmark_manifest = args.ground_truth_root / "benchmark.jsonl"
    benchmark_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = evaluate_aec_specialist(
        args.checkpoint,
        benchmark_manifest,
        split="test",
        tile_size=128,
        batch_size=8,
        device="cpu",
        full_sheet=True,
    )
    report.update(
        {
            "scope": "three-whole-sheet-direct-visual-v2-references",
            "dataset_license": "CC-BY-NC-SA-4.0",
            "allowed_use": "non-commercial research evaluation only",
            "public_real_drawings_used_for_training": False,
            "reference_count": len(rows),
            "reference_hashes": reference_hashes,
        }
    )
    report["content_sha256"] = sha256_json(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
