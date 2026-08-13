from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from buili_plan2bim.sheet_layout import discover_plan_regions


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a review overlay and manifest for whole-sheet plan proposals."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--sheet-id", default="review-sheet")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    image = Image.open(args.image).convert("RGB")
    analysis = discover_plan_regions(image, sheet_id=args.sheet_id)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_directory / "sheet-layout.json"
    manifest_path.write_text(
        json.dumps(analysis.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(18, min(image.size) // 90))
    stroke_width = max(4, min(image.size) // 500)
    for region in analysis.regions:
        left, top, right, bottom = region.bbox_px
        draw.rectangle(
            (left, top, right, bottom),
            outline=(0, 137, 188),
            width=stroke_width,
        )
        label = f"{region.id}  {region.confidence:.2f}  REVIEW"
        label_box = draw.textbbox((left, top), label, font=font)
        label_height = label_box[3] - label_box[1] + 12
        draw.rectangle(
            (left, max(0, top - label_height), right, top),
            fill=(0, 137, 188),
        )
        draw.text(
            (left + 8, max(2, top - label_height + 5)),
            label,
            font=font,
            fill="white",
        )
    overlay.save(args.output_directory / "sheet-layout-overlay.png", optimize=True)


if __name__ == "__main__":
    main()
