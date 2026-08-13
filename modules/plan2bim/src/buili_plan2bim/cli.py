from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ConversionConfig, Plan2BimConverter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buili-plan2bim",
        description="Convert an image or PDF floor plan into a review-gated IFC BIM model.",
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="PDF, PNG, JPEG, TIFF, or another Pillow-supported image",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Directory for IFC and audit artifacts",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        required=False,
        help="Drawing scale in source-image pixels per physical meter",
    )
    parser.add_argument("--project-id", default="dajoong-project")
    parser.add_argument("--sheet-id", default="")
    parser.add_argument(
        "--plan-instance-id",
        default="",
        help="Select one detected plan when a sheet contains multiple floor plans",
    )
    parser.add_argument("--level-id", default="L1")
    parser.add_argument("--level-name", default="Level 1")
    parser.add_argument("--elevation-m", type=float, default=0.0)
    parser.add_argument("--height-m", type=float, default=3.0)
    parser.add_argument("--wall-thickness-m", type=float, default=0.12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--semantic-model",
        type=Path,
        default=None,
        help="Optional content-addressed full-sheet semantic ONNX model",
    )
    parser.add_argument(
        "--semantic-max-side",
        type=int,
        default=None,
        help="Override the active runtime full-sheet inference resolution",
    )
    parser.add_argument(
        "--global-program-model",
        type=Path,
        default=None,
        help="Immutable whole-sheet architectural ONNX program model",
    )
    parser.add_argument(
        "--local-element-model",
        type=Path,
        default=None,
        help="Optional immutable native-resolution element refinement model",
    )
    parser.add_argument(
        "--allow-research-global-program",
        action="store_true",
        help="Allow an explicitly supplied non-production model for sealed evaluation",
    )
    parser.add_argument(
        "--allow-legacy-semantic-teacher",
        action="store_true",
        help="Allow the pinned non-commercial semantic teacher for sealed evaluation",
    )
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument(
        "--discover-native-candidates",
        dest="discover_native_candidates",
        action="store_true",
        default=None,
        help="Run exhaustive native-resolution candidate mining (default with a local model)",
    )
    discovery.add_argument(
        "--skip-native-candidate-discovery",
        dest="discover_native_candidates",
        action="store_false",
        help="Disable native candidate mining for a controlled ablation only",
    )
    parser.add_argument("--page", type=int, default=1, help="One-based PDF page number")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="PDF rasterization DPI")
    parser.add_argument(
        "--strict-ifc",
        action="store_true",
        help="Do not emit a draft IFC when verification requires review",
    )
    parser.add_argument(
        "--allow-primary-only-smoke",
        action="store_true",
        help="Explicitly allow the incomplete non-semantic smoke-test path",
    )
    parser.add_argument(
        "--specialist-mode",
        choices=("architectural", "building_systems", "combined"),
        default="architectural",
        help=(
            "Run the architectural program, building-system specialists, or both. "
            "Architectural mode avoids unrelated MEP inference."
        ),
    )
    parser.add_argument(
        "--model-card",
        action="store_true",
        help="Print the bundled model card and exit",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    converter = Plan2BimConverter(
        model_path=args.model,
        threads=args.threads,
        batch_size=args.batch_size,
        semantic_model_path=args.semantic_model,
        semantic_max_side=args.semantic_max_side,
        global_program_model_path=args.global_program_model,
        local_element_model_path=args.local_element_model,
        discover_native_candidates=args.discover_native_candidates,
        allow_research_global_program=args.allow_research_global_program,
        allow_legacy_semantic_teacher=args.allow_legacy_semantic_teacher,
    )
    if args.model_card:
        print(json.dumps(converter.model_card(), indent=2))
        return
    if args.image is None or args.output_dir is None or args.pixels_per_meter is None:
        parser = _parser()
        parser.error("image, output_dir, and --pixels-per-meter are required for conversion")
    config = ConversionConfig(
        project_id=args.project_id,
        sheet_id=args.sheet_id,
        plan_instance_id=args.plan_instance_id,
        level_id=args.level_id,
        level_name=args.level_name,
        pixels_per_meter=args.pixels_per_meter,
        elevation_m=args.elevation_m,
        nominal_height_m=args.height_m,
        wall_thickness_m=args.wall_thickness_m,
        threads=args.threads,
        batch_size=args.batch_size,
        page_number=args.page,
        pdf_dpi=args.pdf_dpi,
        allow_draft_ifc=not args.strict_ifc,
        allow_primary_only_smoke=args.allow_primary_only_smoke,
        specialist_mode=args.specialist_mode,
    )
    result = converter.convert(args.image, args.output_dir, config)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
