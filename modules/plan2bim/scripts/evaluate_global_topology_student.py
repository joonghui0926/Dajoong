from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.global_topology_training import (
    evaluate_synthetic_topology_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one synthetic-only global topology checkpoint."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("source_corpus_root")
    parser.add_argument("target_corpus_root")
    parser.add_argument("output_path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_synthetic_topology_checkpoint(
        args.checkpoint_path,
        args.source_corpus_root,
        args.target_corpus_root,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
