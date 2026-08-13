from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from buili_plan2bim.ground_truth import compile_benchmark_graph_from_manifest


COLORS = {
    "rooms": (42, 111, 151, 62),
    "walls": (211, 72, 55, 220),
    "openings": (15, 134, 115, 210),
    "fixtures": (174, 116, 27, 205),
    "routes": (100, 68, 159, 205),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = compile_benchmark_graph_from_manifest(manifest)
    source = Image.open(manifest["source"]["image_path"]).convert("RGBA")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for group in ("rooms", "walls", "openings", "fixtures", "routes"):
        color = COLORS[group]
        for entity in graph[group]:
            polygon = entity.get("polygon")
            if polygon:
                points = [(round(x), round(y)) for x, y in polygon]
                fill = color if group == "rooms" else (*color[:3], 34)
                draw.polygon(points, fill=fill, outline=color, width=2)
                center = (
                    round(sum(point[0] for point in points) / len(points)),
                    round(sum(point[1] for point in points) / len(points)),
                )
            else:
                start = tuple(round(value) for value in entity["from"])
                end = tuple(round(value) for value in entity["to"])
                draw.line((start, end), fill=color, width=3)
                center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            draw.text(center, entity["id"], fill=(*color[:3], 255), font=font, anchor="mm")
    output = (
        args.output
        or manifest_path.with_name("direct-source-pixel-review-overlay.png")
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(source, overlay).convert("RGB").save(output)
    print(output)


if __name__ == "__main__":
    main()
