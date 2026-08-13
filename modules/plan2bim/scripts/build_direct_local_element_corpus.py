from __future__ import annotations

import argparse
import json

from buili_plan2bim.direct_local_element_corpus import (
    build_direct_local_element_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build local-element evidence from direct native-resolution visual "
            "annotations. Candidate mining supplies crop locations, never labels."
        )
    )
    parser.add_argument("ground_truth_root")
    parser.add_argument("output_root")
    parser.add_argument(
        "--purpose",
        choices=("research_calibration", "production_training"),
        required=True,
    )
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--maximum-hard-negatives-per-sheet", type=int, default=512)
    parser.add_argument(
        "--proposal-aligned-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    manifest = build_direct_local_element_corpus(
        args.ground_truth_root,
        args.output_root,
        purpose=args.purpose,
        input_size=args.input_size,
        proposal_aligned_positive=args.proposal_aligned_positive,
        maximum_hard_negatives_per_sheet=args.maximum_hard_negatives_per_sheet,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
