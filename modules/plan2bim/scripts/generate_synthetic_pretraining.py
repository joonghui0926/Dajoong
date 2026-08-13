from __future__ import annotations

import argparse
import json

from buili_plan2bim.synthetic_pretraining import generate_synthetic_pretraining_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dajoong-owned topology pretraining supervision."
    )
    parser.add_argument("output_root")
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=26_081_100)
    args = parser.parse_args()
    manifest = generate_synthetic_pretraining_corpus(
        args.output_root,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
