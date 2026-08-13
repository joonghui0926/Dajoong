from __future__ import annotations

import argparse
import json

from buili_plan2bim.local_element_training import (
    LocalElementTrainOptions,
    train_local_element_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain the native-detail local element student."
    )
    parser.add_argument("corpus_root")
    parser.add_argument("output_root")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-checkpoint")
    args = parser.parse_args()
    report = train_local_element_student(
        args.corpus_root,
        args.output_root,
        options=LocalElementTrainOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
