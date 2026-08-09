from __future__ import annotations

import argparse
import json
from pathlib import Path

from .building_pipeline import BuildingConversionConfig, BuildingPlan2BimConverter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dajoong-building2bim",
        description="Convert explicit PDF pages or plan images into one multi-level IFC and GLB.",
    )
    parser.add_argument("config", type=Path, help="Building conversion JSON contract")
    parser.add_argument("output_dir", type=Path, help="Directory for building artifacts")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--semantic-model", type=Path, default=None)
    parser.add_argument("--semantic-max-side", type=int, default=1024)
    args = parser.parse_args()
    config = BuildingConversionConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    converter = BuildingPlan2BimConverter(
        model_path=args.model,
        threads=config.threads,
        batch_size=config.batch_size,
        semantic_model_path=args.semantic_model,
        semantic_max_side=args.semantic_max_side,
    )
    result = converter.convert(args.output_dir, config)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
