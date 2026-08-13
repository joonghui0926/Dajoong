"""Content-addressed, immutable delivery of server-owned 3D family geometry."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "v1"
MESH_HASH = re.compile(r"^[a-f0-9]{64}$")


@lru_cache(maxsize=1)
def asset_catalog() -> dict[str, Any]:
    path = ASSET_ROOT / "catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dajoong.server-family-catalog.v1":
        raise RuntimeError("unsupported server family catalog")
    if int(payload.get("asset_count") or 0) < 100:
        raise RuntimeError("production family catalog must contain at least 100 assets")
    return payload


@lru_cache(maxsize=1)
def _known_geometry_refs() -> frozenset[str]:
    return frozenset(str(item["geometry_ref"]) for item in asset_catalog()["entries"])


def asset_mesh_path(mesh_sha256: str) -> Path:
    if not MESH_HASH.fullmatch(mesh_sha256):
        raise KeyError(mesh_sha256)
    path = ASSET_ROOT / f"{mesh_sha256}.mesh"
    if not path.is_file():
        raise KeyError(mesh_sha256)
    return path


def externalize_graph_assets(graph: dict[str, Any]) -> dict[str, Any]:
    """Remove only library-backed inline meshes from a browser response."""

    embedded = graph.get("family_assets") or {}
    known = _known_geometry_refs()
    graph["family_assets"] = {
        key: value for key, value in embedded.items() if str(key) not in known
    }
    graph["asset_delivery"] = {
        "schema_version": "dajoong.asset-delivery.v1",
        "catalog_url": "/api/assets/v1/catalog",
        "mesh_url_template": "/api/assets/v1/{mesh_sha256}.mesh",
        "format": "dajoong.mesh.v1",
        "content_addressed": True,
        "lazy_by_visible_level": True,
    }
    graph.setdefault("pipeline", {}).setdefault("family_resolution", {})[
        "browser_inline_geometry_count"
    ] = len(graph["family_assets"])
    return graph

