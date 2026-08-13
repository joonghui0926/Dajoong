from __future__ import annotations

import argparse
import json

from buili_plan2bim.global_topology_training import (
    TopologyTrainOptions,
    train_global_topology_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain the whole-sheet Method v2 topology student."
    )
    parser.add_argument("source_corpus_root")
    parser.add_argument("target_corpus_root")
    parser.add_argument("output_root")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initial-checkpoint")
    args = parser.parse_args()
    print(
        f"training global topology student: epochs={args.epochs}, "
        f"batch_size={args.batch_size}, device={args.device}",
        flush=True,
    )
    report = train_global_topology_student(
        args.source_corpus_root,
        args.target_corpus_root,
        args.output_root,
        options=TopologyTrainOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
