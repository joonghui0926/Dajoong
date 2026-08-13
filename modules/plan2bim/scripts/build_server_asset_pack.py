"""Compile the server-only family library into lazy browser mesh payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from buili_plan2bim.core.asset_catalog import family_asset_library
from buili_plan2bim.core.hashing import sha256_json

MAGIC = b"DJMSH001"


def _binary_mesh(definition: dict[str, object]) -> bytes:
    vertices = np.asarray(definition["mesh_vertices"], dtype="<f4")
    faces = np.asarray(definition["mesh_faces"], dtype="<u4")
    colors = np.asarray(definition.get("mesh_face_colors") or [], dtype=np.uint8)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("asset vertices must have shape Nx3")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("asset faces must have shape Mx3")
    if colors.shape != (len(faces), 3):
        colors = np.tile(np.asarray([[128, 128, 128]], dtype=np.uint8), (len(faces), 1))
    return b"".join(
        (
            struct.pack("<8sII", MAGIC, len(vertices), len(faces)),
            vertices.tobytes(order="C"),
            faces.tobytes(order="C"),
            colors.tobytes(order="C"),
        )
    )


def build(destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    records = family_asset_library()
    entries: list[dict[str, object]] = []
    expected_files: set[str] = set()
    total_bytes = 0
    for record in records:
        definition = dict(record["definition"])
        mesh_hash = str(definition["asset_mesh_sha256"])
        filename = f"{mesh_hash}.mesh"
        payload = _binary_mesh(definition)
        file_sha256 = hashlib.sha256(payload).hexdigest()
        target = destination / filename
        target.write_bytes(payload)
        expected_files.add(filename)
        total_bytes += len(payload)
        entries.append(
            {
                key: value
                for key, value in record.items()
                if key != "definition"
            }
            | {
                "asset_mesh_sha256": mesh_hash,
                "mesh_file": filename,
                "mesh_file_sha256": file_sha256,
                "mesh_bytes": len(payload),
                "format": "dajoong.mesh.v1",
                "normalized_to_unit_envelope": True,
            }
        )
    for stale in destination.glob("*.mesh"):
        if stale.name not in expected_files:
            stale.unlink()
    unique_geometry = {str(item["geometry_ref"]) for item in entries}
    catalog: dict[str, object] = {
        "schema_version": "dajoong.server-family-catalog.v1",
        "asset_count": len(entries),
        "geometry_count": len(unique_geometry),
        "licensed_asset_count": sum(
            item["geometry_status"] == "licensed_api_asset" for item in entries
        ),
        "native_parametric_count": sum(
            item["geometry_status"] == "native_bim_parametric" for item in entries
        ),
        "mesh_bytes": total_bytes,
        "browser_bundle_mesh_bytes": 0,
        "delivery": "content-addressed lazy fetch",
        "model_parameters_included": False,
        "entries": entries,
    }
    catalog["content_sha256"] = sha256_json(catalog)
    (destination / "catalog.json").write_text(
        json.dumps(catalog, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parents[3]
        / "apps"
        / "plan2bim-studio"
        / "backend"
        / "src"
        / "buili_plan2bim_studio"
        / "assets"
        / "v1",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.destination.resolve()), indent=2)[:2000])


if __name__ == "__main__":
    main()

