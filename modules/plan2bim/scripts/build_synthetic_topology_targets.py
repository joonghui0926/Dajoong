from __future__ import annotations

import argparse
import json

from buili_plan2bim.topology_supervision import (
    build_synthetic_topology_target_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build isolated whole-sheet topology targets from a synthetic corpus."
    )
    parser.add_argument("corpus_root")
    parser.add_argument("output_root")
    parser.add_argument("--target-size", type=int, default=256)
    args = parser.parse_args()
    manifest = build_synthetic_topology_target_corpus(
        args.corpus_root,
        args.output_root,
        target_size=args.target_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
