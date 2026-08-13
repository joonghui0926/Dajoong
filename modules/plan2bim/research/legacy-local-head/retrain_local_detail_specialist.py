"""Continue the tiny CPU specialist on Dajoong-owned complex drawings only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from dajoong_spatial_compiler.model.aec_specialist import AecSpecialistConfig
from dajoong_spatial_compiler.training.aec_dense import DenseTrainOptions, train_aec_specialist


def _config(value: dict[str, Any], model_version: str) -> AecSpecialistConfig:
    payload = dict(value)
    payload["model_version"] = model_version
    payload["symbol_classes"] = tuple(payload["symbol_classes"])
    payload["structure_channels"] = tuple(payload["structure_channels"])
    return AecSpecialistConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tiles-per-page", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    parent = torch.load(args.parent, map_location="cpu", weights_only=True)
    report = train_aec_specialist(
        args.manifest,
        args.output,
        model_config=_config(parent["config"], "dajoong-aec-local-detail-reviewed-v1"),
        options=DenseTrainOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            tile_size=128,
            tiles_per_page=args.tiles_per_page,
            workers=0,
            device="cpu",
            seed=20260810,
            drafting_augmentation=True,
        ),
        initial_checkpoint=args.parent,
    )
    report["training_rights"] = "Dajoong-owned synthetic complex drawings only"
    report["public_real_drawings_used_for_training"] = False
    (args.output / "run-card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
