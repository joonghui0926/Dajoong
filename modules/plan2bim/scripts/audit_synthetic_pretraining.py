from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.synthetic_pretraining import audit_synthetic_pretraining_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustively audit synthetic topology supervision before training."
    )
    parser.add_argument("corpus_root")
    parser.add_argument("--output")
    parser.add_argument("--maximum-fixture-overlap-ratio", type=float, default=0.08)
    args = parser.parse_args()
    report = audit_synthetic_pretraining_corpus(
        args.corpus_root,
        maximum_fixture_overlap_ratio=args.maximum_fixture_overlap_ratio,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
