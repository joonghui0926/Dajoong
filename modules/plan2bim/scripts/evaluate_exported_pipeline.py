from __future__ import annotations

import argparse
import json

from buili_plan2bim.pipeline_evaluation import evaluate_exported_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate final exported plan graphs against direct visual ground truth."
    )
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pixels-per-meter", type=float, required=True)
    parser.add_argument(
        "--content-profile",
        choices=("structural_core", "full_editable_bim"),
        default="full_editable_bim",
    )
    arguments = parser.parse_args()
    report = evaluate_exported_pipeline(
        ground_truth_root=arguments.ground_truth_root,
        prediction_root=arguments.prediction_root,
        output_root=arguments.output_root,
        pixels_per_meter=arguments.pixels_per_meter,
        required_content_profile=arguments.content_profile,
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
