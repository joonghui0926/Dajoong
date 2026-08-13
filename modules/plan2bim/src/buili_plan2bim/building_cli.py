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
    parser.add_argument("--global-program-model", type=Path, default=None)
    parser.add_argument("--local-element-model", type=Path, default=None)
    parser.add_argument("--allow-research-global-program", action="store_true")
    parser.add_argument("--allow-legacy-semantic-teacher", action="store_true")
    parser.add_argument(
        "--semantic-max-side",
        type=int,
        default=None,
        help="Override the active runtime full-sheet inference resolution",
    )
    args = parser.parse_args()
    config = BuildingConversionConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    converter = BuildingPlan2BimConverter(
        model_path=args.model,
        threads=config.threads,
        batch_size=config.batch_size,
        semantic_model_path=args.semantic_model,
        semantic_max_side=args.semantic_max_side,
        global_program_model_path=args.global_program_model,
        local_element_model_path=args.local_element_model,
        discover_native_candidates=None,
        allow_research_global_program=args.allow_research_global_program,
        allow_legacy_semantic_teacher=args.allow_legacy_semantic_teacher,
    )
    result = converter.convert(args.output_dir, config)
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
