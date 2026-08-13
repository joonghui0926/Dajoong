"""Build the production runtime catalog from an audited source catalog.

The conversion hot path never downloads assets.  This maintenance script copies a
license-allowlisted set of normalized meshes and their provenance records into the
server-side Plan2BIM package.  Browsers receive geometry only through the lazy,
content-addressed asset API.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CURATED_UIDS = {
    "toilet": "825f8e31ba0e4864ad4b902aab31af22",
    "sink": "e696375b6645453d9696e3728a959d3c",
    "wardrobe": "fc042d9a11b44df0af7ff4acff9a8ff7",
    "dishwasher": "f52b015abe1c473e8fddb704b7f031b8",
    "washing_machine": "056c4e7895834a4e8661c3e27b51583c",
    "refrigerator": "fe686076735042bebef264f3b9484b20",
    "stove": "8a983b17bc484fedbc1ead6eb61b55b5",
    "bed": "ca1990835c3c4be3b2f1c59eb3355bd9",
    "sofa": "bef41fecbdff4a3fac184c6a1645ab35",
    "armchair": "a9e826fe9bf4492fb338a8b5e2cac482",
    "chair": "30d1c501adde4e9ea5d60b4fbe64e1ef",
    "dining_table": "5f7332013e4a41e6805b33d5ec1d7cb3",
    "coffee_table": "b4cb920d8f2e4dc9b714062577be81e0",
    "shower_head": "c48c05b0877a4ebf89ca92615519640c",
}

ALLOWED_LICENSES = {"by", "cc0"}


def build(source: Path, destination: Path, *, pinned_only: bool = False) -> None:
    catalog_path = source / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    source_entries = list(catalog.get("entries", []))
    by_uid = {str(item["uid"]): item for item in source_entries}
    missing = sorted(set(CURATED_UIDS.values()) - set(by_uid))
    if missing:
        raise ValueError(f"source catalog is missing pinned assets: {missing}")

    mesh_dir = destination / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str | None, str]]
    if pinned_only:
        selected = [(family, uid) for family, uid in CURATED_UIDS.items()]
    else:
        selected = [(None, str(item["uid"])) for item in source_entries]

    entries = []
    copied_uids: set[str] = set()
    for pinned_family, uid in selected:
        entry = dict(by_uid[uid])
        license_slug = str(entry.get("license") or "").lower()
        if license_slug not in ALLOWED_LICENSES:
            raise ValueError(f"asset {uid} has non-allowlisted license {license_slug!r}")
        source_mesh = source / str(entry["mesh_path"])
        target_mesh = mesh_dir / f"{uid}.npz"
        shutil.copy2(source_mesh, target_mesh)
        if pinned_family is not None:
            entry["families"] = [pinned_family]
        if not entry.get("families"):
            raise ValueError(f"asset {uid} has no audited semantic family")
        entry["mesh_path"] = target_mesh.relative_to(destination).as_posix()
        # Source GLBs remain traceable by URI/hash but are not shipped.  The
        # normalized mesh is the only file required by CPU conversion.
        entry.pop("source_glb", None)
        entries.append(entry)
        copied_uids.add(uid)

    for stale_mesh in mesh_dir.glob("*.npz"):
        if stale_mesh.stem not in copied_uids:
            stale_mesh.unlink()

    output = {
        "schema_version": "dajoong.curated-bim-family-catalog.v1",
        "source_catalog_schema": catalog.get("schema_version"),
        "provider": catalog.get("provider"),
        "license_allowlist": ["by", "cc0"],
        "selection_policy": "semantic-family + exact evidence envelope + local CPU cache",
        "delivery_policy": "server-only content-addressed lazy fetch",
        "network_on_conversion_path": False,
        "entry_count": len(entries),
        "entries": entries,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "catalog.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    attributions = {
        item["uid"]: {
            "name": item.get("name", ""),
            "author": item.get("author", ""),
            "license": item.get("license", ""),
            "source_uri": item.get("source_uri", ""),
            "source_sha256": item.get("source_sha256", ""),
        }
        for item in entries
    }
    (destination / "ATTRIBUTIONS.json").write_text(
        json.dumps(attributions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).parents[1]
        / "src"
        / "buili_plan2bim"
        / "assets"
        / "curated-v1",
    )
    parser.add_argument(
        "--pinned-only",
        action="store_true",
        help="build the legacy 14-asset smoke-test catalog instead of all audited assets",
    )
    args = parser.parse_args()
    build(
        args.source.expanduser().resolve(strict=True),
        args.destination.resolve(),
        pinned_only=args.pinned_only,
    )


if __name__ == "__main__":
    main()
