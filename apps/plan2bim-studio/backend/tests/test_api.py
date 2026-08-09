from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from buili_plan2bim_studio import main
from buili_plan2bim_studio.corrections import graph_content_hash
from buili_plan2bim_studio.store import JobStore


def _graph() -> dict[str, object]:
    return {
        "schema_version": "buili.plan-graph.v2",
        "project_id": "studio-test",
        "sheet_id": "A1.1",
        "scale": {"px_per_meter": 100.0, "source": "test", "confidence": 1.0},
        "levels": [
            {
                "id": "L1",
                "name": "Level 1",
                "elevation_m": 0.0,
                "nominal_height_m": 3.0,
                "confidence": 1.0,
                "uncertainty": 0.0,
                "source_ref_ids": [],
                "model_version": "test",
                "review_state": "accepted",
            }
        ],
        "walls": [
            {
                "id": "L1:wall:1",
                "level_id": "L1",
                "from": [0.0, 0.0],
                "to": [4.0, 0.0],
                "height_m": 3.0,
                "thickness_m": 0.12,
                "confidence": 0.7,
                "uncertainty": 0.3,
                "source_ref_ids": [],
                "model_version": "test",
                "review_state": "review_required",
            }
        ],
        "rooms": [],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "sources": [],
        "unsupported_features": [],
        "extraction": {"method": "test"},
        "provenance": {"source_hash": "a" * 64, "source_revision_state": "test"},
        "confidence": {"review_required": True},
        "warnings": [],
        "pipeline": {"content_sha256": "old"},
    }


def test_production_origin_secret_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("DAJOONG_ORIGIN_VERIFY_SECRET", "test-origin-secret-at-least-32-bytes")
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/jobs").status_code == 403
        verified = client.get(
            "/api/jobs",
            headers={"X-Dajoong-Origin-Verify": "test-origin-secret-at-least-32-bytes"},
        )
        assert verified.status_code != 403


def test_private_api_responses_are_not_cached_and_uploads_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("DAJOONG_MAX_UPLOAD_BYTES", str(1024 * 1024))
    with TestClient(main.app) as client:
        listed = client.get("/api/jobs")
        assert listed.headers["cache-control"] == "private, no-store"
        assert listed.headers["x-content-type-options"] == "nosniff"
        oversized = client.post(
            "/api/jobs",
            data={"pixels_per_meter": "100"},
            files={"drawing": ("oversized.png", b"0" * (1024 * 1024 + 1), "image/png")},
        )
        assert oversized.status_code == 413


def test_import_patch_and_download(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "store", JobStore(tmp_path))
    graph = _graph()
    with TestClient(main.app) as client:
        imported = client.post(
            "/api/jobs/import",
            json={"source_name": "A1.1-plan-graph.json", "graph": graph},
        )
        assert imported.status_code == 200
        job_id = imported.json()["id"]
        patch = client.post(
            f"/api/jobs/{job_id}/corrections",
            json={
                "schema_version": "buili.plan2bim-corrections.v1",
                "expected_graph_sha256": graph_content_hash(graph),
                "reviewer": "QA",
                "operations": [
                    {
                        "id": "accept-1",
                        "action": "accept",
                        "collection": "walls",
                        "entity_id": "L1:wall:1",
                        "changes": {},
                        "reason": "visual_review",
                    }
                ],
            },
        )
        assert patch.status_code == 200
        assert patch.json()["summary"]["accept"] == 1
        downloaded = client.get(f"/api/jobs/{job_id}/artifacts/corrected-graph")
        assert downloaded.status_code == 200
        assert downloaded.json()["walls"][0]["review_state"] == "accepted"
        revised_graph = downloaded.json()
        second_patch = client.post(
            f"/api/jobs/{job_id}/corrections",
            json={
                "schema_version": "buili.plan2bim-corrections.v1",
                "expected_graph_sha256": graph_content_hash(revised_graph),
                "reviewer": "QA",
                "operations": [
                    {
                        "id": "wall-thickness-2",
                        "action": "update",
                        "collection": "walls",
                        "entity_id": "L1:wall:1",
                        "changes": {"thickness_m": 0.15},
                        "reason": "field_measurement",
                    }
                ],
            },
        )
        assert second_patch.status_code == 200
        chained = client.get(f"/api/jobs/{job_id}/artifacts/corrected-graph").json()
        assert chained["walls"][0]["thickness_m"] == 0.15
        assert len(chained["correction_log"]) == 2
        state = client.get(f"/api/jobs/{job_id}").json()
        snapshot = json.loads(json.dumps(chained))
        snapshot["walls"][0]["thickness_m"] = 0.18
        revision_payload = {
            "expected_job_version": state["version"],
            "expected_graph_sha256": graph_content_hash(chained),
            "reviewer": "QA",
            "operations": [],
            "graph": snapshot,
        }
        revision = client.post(f"/api/jobs/{job_id}/revisions", json=revision_payload)
        assert revision.status_code == 200
        assert revision.json()["job_version"] == state["version"] + 1
        stale_revision = client.post(
            f"/api/jobs/{job_id}/revisions",
            json=revision_payload,
        )
        assert stale_revision.status_code == 409
        assert (
            client.get(f"/api/jobs/{job_id}/artifacts/corrected-graph")
            .json()["walls"][0]["thickness_m"]
            == 0.18
        )


def test_pdf_job_preserves_page_and_exposes_render(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "store", JobStore(tmp_path))
    fixture = (
        Path(__file__).resolve().parents[4]
        / "modules"
        / "plan2bim"
        / "tests"
        / "fixtures"
        / "plan-A1.1.png"
    )
    pdf_path = tmp_path / "drawing-set.pdf"
    with Image.open(fixture) as source:
        source.convert("RGB").save(pdf_path, format="PDF", resolution=96)

    with TestClient(main.app) as client, pdf_path.open("rb") as drawing:
        created = client.post(
            "/api/jobs",
            data={"pixels_per_meter": "100", "page_number": "1", "pdf_dpi": "96"},
            files={"drawing": ("drawing-set.pdf", drawing, "application/pdf")},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        # Fresh CI hosts may still be warming ONNX Runtime and PDFium. This is a
        # completion assertion, not the separate conversion-latency benchmark.
        deadline = time.monotonic() + 60
        job = created.json()
        while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
            job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "review_required"
        assert job["result"]["source_kind"] == "raster_pdf"
        assert job["result"]["page_number"] == 1
        assert client.get(f"/api/jobs/{job_id}/artifacts/render").status_code == 200


def test_building_job_assembles_multiple_pdf_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "store", JobStore(tmp_path))
    fixture = (
        Path(__file__).resolve().parents[4]
        / "modules"
        / "plan2bim"
        / "tests"
        / "fixtures"
        / "plan-A1.1.png"
    )
    pdf_path = tmp_path / "building-set.pdf"
    with Image.open(fixture) as source:
        pages = [source.convert("RGB"), source.convert("RGB")]
        pages[0].save(pdf_path, format="PDF", save_all=True, append_images=pages[1:])
    config = {
        "project_id": "api-building",
        "pdf_dpi": 96,
        "levels": [
            {
                "source_path": "browser-upload",
                "page_number": 1,
                "level_id": "L1",
                "name": "Ground floor",
                "elevation_m": 0.0,
                "pixels_per_meter": 100.0,
            },
            {
                "source_path": "browser-upload",
                "page_number": 2,
                "level_id": "L2",
                "name": "Second floor",
                "elevation_m": 3.2,
                "pixels_per_meter": 100.0,
            },
        ],
    }
    with TestClient(main.app) as client, pdf_path.open("rb") as drawing:
        created = client.post(
            "/api/building-jobs",
            data={"building_config": json.dumps(config)},
            files={"drawing": ("building-set.pdf", drawing, "application/pdf")},
        )
        assert created.status_code == 200
        job_id = created.json()["id"]
        deadline = time.monotonic() + 180
        job = created.json()
        while job["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
            job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "review_required"
        graph = client.get(f"/api/jobs/{job_id}/artifacts/graph").json()
        assert [level["id"] for level in graph["levels"]] == ["L1", "L2"]
        consistency = client.get(f"/api/jobs/{job_id}/artifacts/consistency")
        assert consistency.status_code == 200
        assert consistency.json()["schema_version"] == "dajoong.building-consistency-report.v1"
        assert consistency.json()["level_order"] == ["L1", "L2"]
        assert client.get(f"/api/jobs/{job_id}/artifacts/glb").status_code == 200
        assert client.get(f"/api/jobs/{job_id}/artifacts/render").status_code == 200
        level_two_render = client.get(f"/api/jobs/{job_id}/artifacts/render?level_id=L2")
        assert level_two_render.status_code == 200
        assert "00-source-page-2.png" in level_two_render.headers["content-disposition"]
        assert client.get(f"/api/jobs/{job_id}/artifacts/render?level_id=L9").status_code == 404


def test_production_auth_gate_keeps_health_public(monkeypatch) -> None:
    class Verifier:
        def verify(self, token: str) -> dict[str, str]:
            assert token == "valid-test-token"
            return {"sub": "reviewer-1"}

    monkeypatch.setenv("DAJOONG_REQUIRE_AUTH", "true")
    monkeypatch.setattr(main, "_token_verifier_instance", Verifier())
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/jobs/missing").status_code == 401
        authenticated = client.get(
            "/api/jobs/missing",
            headers={"Authorization": "Bearer valid-test-token"},
        )
        assert authenticated.status_code == 404


def test_authenticated_jobs_are_owner_scoped(tmp_path, monkeypatch) -> None:
    class Verifier:
        def verify(self, token: str) -> dict[str, str]:
            return {"sub": token}

    monkeypatch.setenv("DAJOONG_REQUIRE_AUTH", "true")
    monkeypatch.setattr(main, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "store", JobStore(tmp_path))
    monkeypatch.setattr(main, "_token_verifier_instance", Verifier())
    owner_headers = {"Authorization": "Bearer owner-a"}
    with TestClient(main.app) as client:
        imported = client.post(
            "/api/jobs/import",
            json={"source_name": "owner-plan.json", "graph": _graph()},
            headers=owner_headers,
        )
        assert imported.status_code == 200
        assert "owner_id" not in imported.json()
        assert "organization_id" not in imported.json()
        assert "output_dir" not in imported.json()
        job_id = imported.json()["id"]
        assert (
            client.get(
                f"/api/jobs/{job_id}", headers={"Authorization": "Bearer owner-b"}
            ).status_code
            == 404
        )
        assert client.get(f"/api/jobs/{job_id}", headers=owner_headers).status_code == 200
        own_page = client.get("/api/jobs?limit=10", headers=owner_headers)
        assert own_page.status_code == 200
        assert [item["id"] for item in own_page.json()["items"]] == [job_id]
        other_page = client.get(
            "/api/jobs?limit=10",
            headers={"Authorization": "Bearer owner-b"},
        )
        assert other_page.status_code == 200
        assert other_page.json()["items"] == []
