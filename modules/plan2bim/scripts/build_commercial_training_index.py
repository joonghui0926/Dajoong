from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.training_corpus import build_commercial_training_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a validated, leakage-safe commercial training index."
    )
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--split-seed", default="dajoong-commercial-v1")
    arguments = parser.parse_args()
    manifests = sorted(arguments.manifest_root.rglob("manifest.json"))
    index = build_commercial_training_index(
        manifests,
        source_root=arguments.source_root,
        split_seed=arguments.split_seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index["split_counts"], indent=2))


if __name__ == "__main__":
    main()
