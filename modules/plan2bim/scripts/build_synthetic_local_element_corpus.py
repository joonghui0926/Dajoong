from __future__ import annotations

import argparse
import json

from buili_plan2bim.local_element_training import build_synthetic_local_element_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native-detail crops from a synthetic drawing corpus."
    )
    parser.add_argument("source_corpus_root")
    parser.add_argument("output_root")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--negatives-per-sheet", type=int, default=4)
    parser.add_argument("--hard-negatives-per-sheet", type=int, default=48)
    parser.add_argument(
        "--proposal-aligned-positives-per-object",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--fragment-negatives-per-object",
        type=int,
        default=2,
        help="Train native stroke fragments as background siblings of whole objects.",
    )
    args = parser.parse_args()
    report = build_synthetic_local_element_corpus(
        args.source_corpus_root,
        args.output_root,
        input_size=args.input_size,
        negatives_per_sheet=args.negatives_per_sheet,
        hard_negatives_per_sheet=args.hard_negatives_per_sheet,
        proposal_aligned_positives_per_object=(
            args.proposal_aligned_positives_per_object
        ),
        fragment_negatives_per_object=args.fragment_negatives_per_object,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
