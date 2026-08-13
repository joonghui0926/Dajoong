from __future__ import annotations

import json
import struct

from fastapi.testclient import TestClient

from buili_plan2bim_studio.asset_delivery import externalize_graph_assets
from buili_plan2bim_studio.main import app


def test_catalog_has_at_least_one_hundred_server_only_assets() -> None:
    with TestClient(app) as client:
        response = client.get("/api/assets/v1/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_count"] >= 100
    assert payload["geometry_count"] >= 100
    assert payload["browser_bundle_mesh_bytes"] == 0
    assert payload["model_parameters_included"] is False
    assert "public" in response.headers["cache-control"]
    assert len(response.content) < 200_000


def test_mesh_is_binary_immutable_and_content_addressed() -> None:
    with TestClient(app) as client:
        catalog = client.get("/api/assets/v1/catalog").json()
        mesh_hash = catalog["entries"][0]["asset_mesh_sha256"]
        response = client.get(f"/api/assets/v1/{mesh_hash}.mesh")

    assert response.status_code == 200
    magic, vertex_count, face_count = struct.unpack_from("<8sII", response.content)
    assert magic == b"DJMSH001"
    assert vertex_count >= 4
    assert face_count >= 4
    assert response.headers["etag"] == f'"{mesh_hash}"'
    assert "immutable" in response.headers["cache-control"]


def test_browser_graph_externalizes_only_known_library_geometry() -> None:
    with TestClient(app) as client:
        catalog = client.get("/api/assets/v1/catalog").json()
    known_ref = catalog["entries"][0]["geometry_ref"]
    unknown_ref = "mesh:" + "f" * 64
    graph = {
        "family_assets": {
            known_ref: {"mesh_vertices": []},
            unknown_ref: {"mesh_vertices": [[0, 0, 0]]},
        },
        "pipeline": {"family_resolution": {}},
    }

    result = externalize_graph_assets(json.loads(json.dumps(graph)))

    assert known_ref not in result["family_assets"]
    assert unknown_ref in result["family_assets"]
    assert result["asset_delivery"]["lazy_by_visible_level"] is True
    assert result["pipeline"]["family_resolution"]["browser_inline_geometry_count"] == 1


def test_unknown_asset_fails_closed() -> None:
    with TestClient(app) as client:
        response = client.get(f"/api/assets/v1/{'0' * 64}.mesh")

    assert response.status_code == 404


def test_development_asset_delivery_accepts_loopback_frontend_ports() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/assets/v1/catalog",
            headers={"Origin": "http://127.0.0.1:4178"},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:4178"
