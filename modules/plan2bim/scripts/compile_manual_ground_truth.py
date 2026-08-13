from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.ground_truth import (
    assert_benchmark_graph_geometry,
    compile_benchmark_graph_from_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a metric graph from direct source-pixel annotation events. "
            "This command never reads SVG, model, candidate or pseudo-label geometry."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = compile_benchmark_graph_from_manifest(manifest)
    image_size = (
        int(manifest["source"]["width_px"]),
        int(manifest["source"]["height_px"]),
    )
    assert_benchmark_graph_geometry(
        graph,
        image_size=image_size,
        reviewed_plan_bbox_px=manifest["source"]["reviewed_plan_bbox_px"],
    )
    output = (args.output or manifest_path.with_name("benchmark-graph.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
