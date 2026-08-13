from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.local_element_training import evaluate_local_element_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a synthetic-only local element checkpoint."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("corpus_root")
    parser.add_argument("output_path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_local_element_checkpoint(
        args.checkpoint_path,
        args.corpus_root,
        batch_size=args.batch_size,
        device=args.device,
    )
    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
