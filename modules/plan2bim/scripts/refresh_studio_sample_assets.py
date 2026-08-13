"""Refresh Studio's browser sample with current asset choices and lazy refs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim_studio.asset_delivery import externalize_graph_assets

from buili_plan2bim.core.asset_catalog import attach_family_assets
from buili_plan2bim.pipeline import _recertify_plan_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    graph.pop("asset_delivery", None)
    attach_family_assets(graph)
    _recertify_plan_graph(graph)
    externalize_graph_assets(graph)
    args.graph.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "fixtures": len(graph.get("fixtures") or []),
                "context_ranked": sum(
                    "asset_selection_context" in fixture
                    for fixture in graph.get("fixtures") or []
                ),
                "inline_family_assets": len(graph.get("family_assets") or {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
