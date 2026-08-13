from __future__ import annotations

import argparse
import json

from buili_plan2bim.local_element_training import (
    LocalElementTrainOptions,
    calibrate_local_element_student_from_direct_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Research-calibrate the compact local student from direct visual "
            "labels while sealing every training source out of evaluation."
        )
    )
    parser.add_argument("corpus_root")
    parser.add_argument("output_root")
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = calibrate_local_element_student_from_direct_research(
        args.corpus_root,
        args.output_root,
        initial_checkpoint=args.initial_checkpoint,
        options=LocalElementTrainOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
