from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from buili_plan2bim import (
    BuildingAssemblyConfig,
    BuildingConversionConfig,
    BuildingLevelInput,
    BuildingLevelSpec,
    BuildingPlan2BimConverter,
    BuildingVerticalConnection,
    ConversionConfig,
    Plan2BimConverter,
    assemble_building_graph,
)
from buili_plan2bim.cli import main as cli_main
from buili_plan2bim.core.cad_families import (
    approved_family_asset_sha256,
    parametric_family_parts,
)
from buili_plan2bim.core.glb_export import export_editable_glb
from buili_plan2bim.core.ifc_export import export_ifc
from buili_plan2bim.core.plan_graph_verification import PlanGraphVerifier
from buili_plan2bim.input_document import prepare_drawing
from buili_plan2bim.pipeline import ConversionError
from buili_plan2bim.semantic_recognition import (
    OnnxFloorPlanSemanticRecognizer,
    SemanticDetection,
    SemanticRecognitionResult,
    _refine_wall_vectors_from_raster,
    _room_records,
    _wall_centerlines,
    _wall_vectors,
)

EXPECTED_MODEL_SHA256 = "36bcfe230be22ed869eb7bc3a940805c516dd0970c66649f944f0d5451ff1817"


def test_metric_scale_is_required_and_positive() -> None:
    with pytest.raises(ValidationError):
        ConversionConfig(pixels_per_meter=0)


def test_bundled_model_is_content_addressed() -> None:
    converter = Plan2BimConverter()
    card = converter.model_card()
    assert card["model_sha256"] == EXPECTED_MODEL_SHA256
    assert card["parameters"] == 86_533
    assert card["release_authorized"] is False
    assert Path(converter.model_path).stat().st_size < 500_000


def test_missing_semantic_model_cannot_silently_run_primary_only(tmp_path: Path) -> None:
    converter = Plan2BimConverter()
    converter.semantic_model_missing = True

    with pytest.raises(ConversionError, match="pinned semantic model is unavailable"):
        converter.convert(
            tmp_path / "not-read.png",
            tmp_path / "output",
            ConversionConfig(pixels_per_meter=100),
        )


def test_cli_model_card_does_not_require_drawing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["dajoong-plan2bim", "--model-card"])

    cli_main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["parameters"] == 86_533


def test_python_module_cli_executes_instead_of_silent_success() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "buili_plan2bim.cli", "--model-card"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["parameters"] == 86_533


def test_image_to_ifc_smoke(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "plan-A1.1.png"
    converter = Plan2BimConverter(threads=1, batch_size=8)
    result = converter.convert(
        fixture,
        tmp_path,
        ConversionConfig(
            project_id="smoke",
            level_id="L1",
            level_name="Level 1",
            pixels_per_meter=100,
            allow_primary_only_smoke=True,
        ),
    )

    assert result.entity_counts["walls"] > 0
    assert result.review_required is True
    assert result.release_allowed is False
    assert Path(result.ifc_path).is_file()
    assert "IFCWALL" in Path(result.ifc_path).read_text(encoding="utf-8")
    assert Path(result.plan_graph_path).is_file()
    assert Path(result.complexity_path).is_file()
    assert Path(result.qualification_path).is_file()
    assert result.production_release_eligible is False
    assert result.difficulty_class in {"simple", "moderate", "difficult", "extreme"}
    assert Path(result.ifc_certificate_path).is_file()
    assert Path(result.glb_path).is_file()
    payload = Path(result.glb_path).read_bytes()
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    document = json.loads(payload[20 : 20 + json_length].rstrip(b" \x00"))
    assert document["asset"]["extras"]["canonicalAuthoringFormat"] == "IFC4.3"
    assert all("buili" in node.get("extras", {}) for node in document["nodes"])


def test_pdf_page_is_rasterized_with_original_provenance(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "plan-A1.1.png"
    pdf_path = tmp_path / "sheet-set.pdf"
    with Image.open(fixture) as source:
        source.convert("RGB").save(pdf_path, format="PDF", resolution=150)

    prepared = prepare_drawing(pdf_path, tmp_path / "prepared", pdf_dpi=144)

    assert prepared.source_kind == "raster_pdf"
    assert prepared.page_number == 1
    assert prepared.page_count == 1
    assert Path(prepared.render_path).is_file()
    assert Path(prepared.source_path) == pdf_path.resolve()
    assert prepared.width_px > 0 and prepared.height_px > 0


def test_semantic_wall_mask_vectorizes_closed_room() -> None:
    mask = np.zeros((160, 220), dtype=np.bool_)
    mask[20:32, 20:200] = True
    mask[128:140, 20:200] = True
    mask[20:140, 20:32] = True
    mask[20:140, 188:200] = True
    mask[20:140, 106:118] = True

    lines = _wall_centerlines(mask)

    assert len(lines) >= 7
    assert any(abs(start_y - end_y) < 1e-6 for _, start_y, _, end_y in lines)
    assert any(abs(start_x - end_x) < 1e-6 for start_x, _, end_x, _ in lines)


def test_semantic_wall_vector_recovers_center_and_thickness_across_opening() -> None:
    mask = np.zeros((120, 240), dtype=np.bool_)
    mask[32:48, 20:220] = True
    mask[32:48, 104:132] = False
    # A perpendicular junction must not inflate the long wall's thickness.
    mask[24:96, 176:192] = True

    vectors = _wall_vectors(mask)
    horizontal = max(
        (item for item in vectors if abs(item.start_px[1] - item.end_px[1]) < 1e-6),
        key=lambda item: abs(item.end_px[0] - item.start_px[0]),
    )

    assert horizontal.start_px[1] == pytest.approx(39.5, abs=1.0)
    assert horizontal.end_px[1] == pytest.approx(39.5, abs=1.0)
    assert horizontal.thickness_px == pytest.approx(16.0, abs=1.0)


def test_semantic_wall_vector_keeps_long_walls_at_thick_t_junctions() -> None:
    mask = np.zeros((180, 240), dtype=np.bool_)
    mask[20:48, 20:220] = True
    mask[20:160, 68:82] = True
    mask[20:160, 158:172] = True

    vectors = _wall_vectors(mask)
    vertical_centers = [
        (item.start_px[0] + item.end_px[0]) / 2
        for item in vectors
        if abs(item.start_px[0] - item.end_px[0]) < 1e-6
        and abs(item.end_px[1] - item.start_px[1]) > 80
    ]

    assert any(abs(center - 74.5) <= 2 for center in vertical_centers)
    assert any(abs(center - 164.5) <= 2 for center in vertical_centers)


def test_semantic_wall_vector_recovers_diagonal_wall() -> None:
    canvas = Image.new("1", (240, 200), 0)
    draw = ImageDraw.Draw(canvas)
    draw.line((35, 170, 125, 80), fill=1, width=14)
    draw.line((35, 30, 35, 170), fill=1, width=14)
    draw.line((125, 80, 220, 80), fill=1, width=14)

    vectors = _wall_vectors(np.asarray(canvas, dtype=np.bool_))
    diagonals = [
        item
        for item in vectors
        if abs(item.start_px[0] - item.end_px[0]) > 40
        and abs(item.start_px[1] - item.end_px[1]) > 40
    ]

    assert diagonals
    assert diagonals[0].thickness_px == pytest.approx(14.0, abs=3.0)


def test_raster_refinement_recenters_semantic_wall_without_adding_walls() -> None:
    source = np.full((120, 240), 255, dtype=np.uint8)
    source[40:56, 20:220] = 0
    semantic = np.zeros((120, 240), dtype=np.bool_)
    semantic[46:62, 20:220] = True
    vectors = _wall_vectors(semantic)

    refined = _refine_wall_vectors_from_raster(source, vectors)
    horizontal = max(
        (item for item in refined if abs(item.start_px[1] - item.end_px[1]) < 1e-6),
        key=lambda item: abs(item.end_px[0] - item.start_px[0]),
    )

    assert len(refined) == len(vectors)
    assert horizontal.start_px[1] == pytest.approx(47.5, abs=1.0)
    assert horizontal.thickness_px == pytest.approx(16.0, abs=2.0)


def test_semantic_room_mask_preserves_concave_polygon() -> None:
    mask = np.zeros((80, 100), dtype=np.bool_)
    mask[10:70, 10:45] = True
    mask[45:70, 45:90] = True
    probability = np.where(mask, 0.9, 0.1).astype(np.float32)

    records = _room_records(
        mask,
        probability,
        minimum_area=100,
        source_size=(200, 160),
    )

    assert len(records) == 1
    assert len(records[0]["polygon_px"]) >= 6
    assert records[0]["confidence"] == pytest.approx(0.9)


def test_review_required_semantic_object_stays_in_editable_draft() -> None:
    detection = SemanticDetection(
        id="semantic:fixture:1",
        class_name="Sink",
        symbol_class="sink",
        bbox_px=(20, 30, 40, 50),
        confidence=0.52,
        pixel_area=80,
        review_required=True,
        promote_to_bim=True,
    )
    result = SemanticRecognitionResult(
        input_path="source.png",
        input_sha256="a" * 64,
        model_version="test",
        model_sha256="b" * 64,
        license_scope="internal_eval_only",
        production_authorized=False,
        source_size=(100, 100),
        model_input_size=(96, 96),
        wall_pixels=0,
        detections=[detection],
        counts={"Sink": 1},
        inference_ms=1.0,
        total_ms=2.0,
    )

    proposals = OnnxFloorPlanSemanticRecognizer.symbol_proposals(
        result,
        source_ref_ids=["source"],
    )

    assert len(proposals) == 1
    assert proposals[0].symbol_class == "sink"
    assert proposals[0].review_required is True


def test_review_required_opening_cannot_invent_a_wall() -> None:
    detection = SemanticDetection(
        id="semantic:door:1",
        class_name="Door",
        symbol_class="door",
        bbox_px=(20, 30, 40, 50),
        confidence=0.52,
        pixel_area=80,
        review_required=True,
        promote_to_bim=True,
    )
    result = SemanticRecognitionResult(
        input_path="source.png",
        input_sha256="a" * 64,
        model_version="test",
        model_sha256="b" * 64,
        license_scope="internal_eval_only",
        production_authorized=False,
        source_size=(100, 100),
        model_input_size=(96, 96),
        wall_pixels=0,
        detections=[detection],
        counts={"Door": 1},
        inference_ms=1.0,
        total_ms=2.0,
    )

    proposals = OnnxFloorPlanSemanticRecognizer.wall_proposals(
        result,
        source_ref_ids=["source"],
    )

    assert proposals == []


def test_confident_opening_cannot_invent_a_wall_without_global_structure() -> None:
    detection = SemanticDetection(
        id="semantic:door:1",
        class_name="Door",
        symbol_class="door",
        bbox_px=(20, 30, 40, 50),
        confidence=0.95,
        pixel_area=80,
        review_required=False,
        promote_to_bim=True,
    )
    result = SemanticRecognitionResult(
        input_path="source.png",
        input_sha256="a" * 64,
        model_version="test",
        model_sha256="b" * 64,
        license_scope="internal_eval_only",
        production_authorized=False,
        source_size=(100, 100),
        model_input_size=(96, 96),
        wall_pixels=0,
        detections=[detection],
        counts={"Door": 1},
        inference_ms=1.0,
        total_ms=2.0,
    )

    proposals = OnnxFloorPlanSemanticRecognizer.wall_proposals(
        result,
        source_ref_ids=["source"],
    )

    assert proposals == []


def _minimal_graph(level_id: str = "L1") -> dict[str, object]:
    source_id = f"source-{level_id}"
    digest = "a" * 64
    return {
        "schema_version": "buili.plan-graph.v2",
        "project_id": "test",
        "sheet_id": level_id,
        "scale": {"px_per_meter": 1.0, "source": "test", "confidence": 1.0},
        "levels": [
            {
                "id": level_id,
                "name": level_id,
                "elevation_m": 0.0,
                "nominal_height_m": 3.0,
                "confidence": 1.0,
                "uncertainty": 0.0,
                "source_ref_ids": [source_id],
                "model_version": "test",
                "review_state": "accepted",
            }
        ],
        "rooms": [],
        "walls": [
            {
                "id": f"{level_id}:wall:0",
                "level_id": level_id,
                "room_id": "",
                "from": [0.0, 0.0],
                "to": [4.0, 0.0],
                "thickness_m": 0.12,
                "height_m": 3.0,
                "wall_type": "interior",
                "material": "gypsum",
                "confidence": 1.0,
                "uncertainty": 0.0,
                "source_ref_ids": [source_id],
                "model_version": "test",
                "review_state": "accepted",
            }
        ],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "sources": [
            {
                "source_ref_id": source_id,
                "source_hash": digest,
                "uri": "https://example.invalid/plan.png",
                "page": 1,
                "bbox": [],
                "source_type": "raster_image",
                "source_strength": "strong",
                "extractor": "test",
                "model_version": "test",
            }
        ],
        "unsupported_features": [],
        "extraction": {"method": "test"},
        "provenance": {"source_hash": digest, "source_revision_state": "test"},
        "confidence": {
            "overall": 1.0,
            "geometry": 1.0,
            "semantics": 1.0,
            "scale": 1.0,
            "traceability": 1.0,
            "review_required": False,
            "method": "test",
        },
        "warnings": [],
        "pipeline": {
            "contract_version": "buili.plan-graph.v2",
            "pipeline_version": "test",
            "deterministic": True,
            "review_required": False,
            "content_sha256": digest,
        },
    }


def test_door_operation_semantics_survive_ifc_and_glb_export(tmp_path: Path) -> None:
    graph = _minimal_graph()
    graph["openings"] = [
        {
            "id": "L1:door:0",
            "source_entity_id": "L1:door:0",
            "level_id": "L1",
            "type": "door",
            "wall_id": "L1:wall:0",
            "x_m": 2.0,
            "center_m": [2.0, 0.0],
            "width_m": 0.9,
            "height_m": 2.1,
            "sill_height_m": 0.0,
            "family_id": "single-door",
            "operation_type": "single_swing",
            "handing": "start",
            "swing_side": "negative",
            "confidence": 1.0,
            "uncertainty": 0.0,
            "source_ref_ids": ["source-L1"],
            "model_version": "test",
            "review_state": "accepted",
        }
    ]

    ifc_path = tmp_path / "door.ifc"
    export_ifc(graph, ifc_path, allow_draft=True)
    assert ".SINGLE_SWING_LEFT." in ifc_path.read_text(encoding="utf-8")

    glb_path = tmp_path / "door.glb"
    export_editable_glb(graph, glb_path)
    payload = glb_path.read_bytes()
    json_length, _ = struct.unpack_from("<I4s", payload, 12)
    document = json.loads(payload[20 : 20 + json_length].rstrip(b" \x00"))
    door_node = next(node for node in document["nodes"] if node["name"] == "door::L1:door:0")
    properties = door_node["extras"]["buili"]["properties"]
    assert properties["operation_type"] == "single_swing"
    assert properties["handing"] == "start"
    assert properties["swing_side"] == "negative"

    unresolved = json.loads(json.dumps(graph))
    unresolved["openings"][0]["handing"] = "unknown"
    certificate = PlanGraphVerifier().verify(unresolved, permit_review_required=True)
    assert certificate.release_allowed is False
    assert "UNRESOLVED_DOOR_OPERATION" in {violation.code for violation in certificate.violations}


def test_coincident_wall_constraints_are_verified_and_break_fail_closed() -> None:
    graph = _minimal_graph()
    second_wall = json.loads(json.dumps(graph["walls"][0]))
    second_wall.update({"id": "L1:wall:1", "from": [4.0, 0.0], "to": [4.0, 4.0]})
    graph["walls"].append(second_wall)
    graph["constraints"] = [
        {
            "id": "L1:constraint:0",
            "level_id": "L1",
            "type": "coincident",
            "references": [
                {"collection": "walls", "entity_id": "L1:wall:0", "handle": "to"},
                {"collection": "walls", "entity_id": "L1:wall:1", "handle": "from"},
            ],
            "confidence": 1.0,
            "uncertainty": 0.0,
            "review_state": "accepted",
            "model_version": "human-correction",
        }
    ]
    valid = PlanGraphVerifier().verify(graph, permit_review_required=True)
    assert "BROKEN_COINCIDENT_CONSTRAINT" not in {violation.code for violation in valid.violations}

    graph["walls"][1]["from"] = [4.1, 0.0]
    broken = PlanGraphVerifier().verify(graph, permit_review_required=True)
    assert broken.release_allowed is False
    assert "BROKEN_COINCIDENT_CONSTRAINT" in {violation.code for violation in broken.violations}


def test_dimension_geometry_must_have_two_distinct_metric_points() -> None:
    graph = _minimal_graph()
    graph["dimensions"] = [
        {
            "id": "L1:dimension:0",
            "level_id": "L1",
            "type": "aligned",
            "from": [1.0, 1.0],
            "to": [1.0, 1.0],
            "confidence": 1.0,
            "uncertainty": 0.0,
            "review_state": "accepted",
            "model_version": "human-correction",
        }
    ]
    certificate = PlanGraphVerifier().verify(graph, permit_review_required=True)
    assert certificate.release_allowed is False
    assert "INVALID_DIMENSION_GEOMETRY" in {violation.code for violation in certificate.violations}


def test_integrated_appliance_cannot_share_a_casework_solid() -> None:
    graph = _minimal_graph()
    fixture_common = {
        "level_id": "L1",
        "room_id": "",
        "center_m": [2.0, 2.0],
        "base_elevation_m": 0.0,
        "yaw_deg": 0.0,
        "required_count": 1,
        "observed_count": 1,
        "confidence": 1.0,
        "uncertainty": 0.0,
        "source_ref_ids": ["source-L1"],
        "model_version": "test",
        "review_state": "accepted",
    }
    graph["fixtures"] = [
        {
            **fixture_common,
            "id": "cabinet",
            "type": "residential-base-cabinet",
            "size_m": [2.4, 0.6, 0.9],
        },
        {
            **fixture_common,
            "id": "dishwasher",
            "type": "residential-dishwasher",
            "size_m": [0.6, 0.6, 0.85],
        },
    ]

    certificate = PlanGraphVerifier().verify(graph, permit_review_required=True)

    assert certificate.release_allowed is False
    assert "INTEGRATED_APPLIANCE_CASEWORK_COLLISION" in {
        violation.code for violation in certificate.violations
    }


def test_multilevel_assembly_uses_explicit_levels_without_phantom_stair() -> None:
    first = _minimal_graph("source-L1")
    second = _minimal_graph("source-L2")
    assembled = assemble_building_graph(
        {"L1": first, "L2": second},
        BuildingAssemblyConfig(
            project_id="building",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(
                    level_id="L2",
                    name="Second",
                    elevation_m=3.2,
                    x_offset_m=0.2,
                ),
            ],
        ),
    )
    assert [item["id"] for item in assembled["levels"]] == ["L1", "L2"]
    assert assembled["vertical_connections"] == []
    assert assembled["walls"][1]["from"][0] == pytest.approx(0.2)


def test_multilevel_assembly_aggregates_level_qualification_fail_closed() -> None:
    first = _minimal_graph("source-L1")
    second = _minimal_graph("source-L2")
    first["qualification"] = {
        "schema_version": "dajoong.model-qualification.v1",
        "production_release_eligible": False,
        "review_required": True,
        "review_reasons": ["required_bim_claims_unmeasured"],
    }
    assembled = assemble_building_graph(
        {"L1": first, "L2": second},
        BuildingAssemblyConfig(
            project_id="building",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=3.2),
            ],
        ),
    )

    assert assembled["qualification"]["production_release_eligible"] is False
    assert (
        "one_or_more_levels_missing_qualification" in assembled["qualification"]["review_reasons"]
    )
    assert assembled["verification"]["release_allowed"] is False


def test_building_pipeline_converts_two_pdf_pages_into_one_model(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "plan-A1.1.png"
    pdf_path = tmp_path / "two-level-set.pdf"
    with Image.open(fixture) as source:
        pages = [source.convert("RGB"), source.convert("RGB")]
        pages[0].save(pdf_path, format="PDF", save_all=True, append_images=pages[1:])
    converter = BuildingPlan2BimConverter(threads=1, batch_size=8)
    result = converter.convert(
        tmp_path / "building",
        BuildingConversionConfig(
            project_id="two-page-building",
            pdf_dpi=96,
            allow_primary_only_smoke=True,
            levels=[
                BuildingLevelInput(
                    source_path=str(pdf_path),
                    page_number=1,
                    level_id="L1",
                    name="Ground floor",
                    elevation_m=0.0,
                    pixels_per_meter=100.0,
                ),
                BuildingLevelInput(
                    source_path=str(pdf_path),
                    page_number=2,
                    level_id="L2",
                    name="Second floor",
                    elevation_m=3.2,
                    pixels_per_meter=100.0,
                ),
            ],
        ),
    )

    graph = json.loads(Path(result.plan_graph_path).read_text(encoding="utf-8"))
    assert [level["id"] for level in graph["levels"]] == ["L1", "L2"]
    assert set(result.level_results) == {"L1", "L2"}
    assert result.level_results["L2"].page_number == 2
    assert Path(result.ifc_path).is_file()
    assert Path(result.glb_path).is_file()
    assert Path(result.ifc_path).read_text(encoding="utf-8").count("IFCBUILDINGSTOREY") == 2
    consistency = json.loads(Path(result.consistency_report_path).read_text(encoding="utf-8"))
    assert consistency["schema_version"] == "dajoong.building-consistency-report.v1"
    assert consistency["level_order"] == ["L1", "L2"]
    assert len(consistency["content_sha256"]) == 64


def _closed_graph(level_id: str) -> dict[str, object]:
    graph = _minimal_graph(level_id)
    source_id = f"source-{level_id}"
    graph["walls"] = [
        {
            "id": f"{level_id}:wall:{index}",
            "level_id": level_id,
            "room_id": "",
            "from": list(start),
            "to": list(end),
            "thickness_m": 0.12,
            "height_m": 3.0,
            "wall_type": "interior",
            "material": "gypsum",
            "confidence": 1.0,
            "uncertainty": 0.0,
            "source_ref_ids": [source_id],
            "model_version": "test",
            "review_state": "accepted",
        }
        for index, (start, end) in enumerate(
            (
                ((0.0, 0.0), (6.0, 0.0)),
                ((6.0, 0.0), (6.0, 6.0)),
                ((6.0, 6.0), (0.0, 6.0)),
                ((0.0, 6.0), (0.0, 0.0)),
            )
        )
    ]
    return graph


def test_vertical_connection_is_verified_and_exported_to_ifc_and_glb(
    tmp_path: Path,
) -> None:
    connection = BuildingVerticalConnection(
        id="stair-a",
        shaft_id="stair-core-a",
        type="stair",
        from_level_id="L1",
        to_level_id="L2",
        center_m=(3.0, 3.0),
        footprint_m=(1.2, 3.0),
    )
    graph = assemble_building_graph(
        {"L1": _closed_graph("L1"), "L2": _closed_graph("L2")},
        BuildingAssemblyConfig(
            project_id="vertical-test",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=3.2),
            ],
            vertical_connections=[connection],
        ),
    )
    assert graph["verification"]["release_allowed"] is True
    ifc_path = tmp_path / "building.ifc"
    export_ifc(graph, ifc_path, allow_draft=False)
    ifc_text = ifc_path.read_text(encoding="utf-8")
    assert "IFCSTAIR" in ifc_text
    assert "ShaftId" in ifc_text
    assert "stair-core-a" in ifc_text
    glb_path = tmp_path / "building.glb"
    export_editable_glb(graph, glb_path)
    payload = glb_path.read_bytes()
    json_length, _ = struct.unpack_from("<I4s", payload, 12)
    document = json.loads(payload[20 : 20 + json_length].rstrip(b" \x00"))
    stair_node = next(
        node for node in document["nodes"] if node["name"].startswith("vertical::stair-a")
    )
    assert stair_node["extras"]["buili"]["properties"]["shaft_id"] == "stair-core-a"


def test_stair_cannot_skip_an_intermediate_level() -> None:
    with pytest.raises(ValidationError, match="adjacent levels"):
        BuildingAssemblyConfig(
            project_id="invalid-stair",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=3.2),
                BuildingLevelSpec(level_id="L3", name="Third", elevation_m=6.4),
            ],
            vertical_connections=[
                BuildingVerticalConnection(
                    id="phantom-stair",
                    type="stair",
                    from_level_id="L1",
                    to_level_id="L3",
                    center_m=(2.0, 2.0),
                    footprint_m=(1.2, 3.0),
                )
            ],
        )


def test_building_verifier_blocks_level_and_wall_overlap() -> None:
    graph = assemble_building_graph(
        {"L1": _closed_graph("L1"), "L2": _closed_graph("L2")},
        BuildingAssemblyConfig(
            project_id="overlap-test",
            levels=[
                BuildingLevelSpec(
                    level_id="L1",
                    name="Ground",
                    elevation_m=0.0,
                    nominal_height_m=3.0,
                ),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=2.8),
            ],
        ),
    )

    codes = {item["code"] for item in graph["verification"]["violations"]}
    assert graph["verification"]["release_allowed"] is False
    assert "LEVEL_VOLUME_OVERLAP" in codes
    assert "WALL_CROSSES_NEXT_LEVEL" in codes


def test_building_verifier_blocks_reversed_and_duplicate_vertical_connections() -> None:
    connection = BuildingVerticalConnection(
        id="stair-a",
        type="stair",
        from_level_id="L1",
        to_level_id="L2",
        center_m=(3.0, 3.0),
        footprint_m=(1.2, 3.0),
    )
    graph = assemble_building_graph(
        {"L1": _closed_graph("L1"), "L2": _closed_graph("L2")},
        BuildingAssemblyConfig(
            project_id="connection-test",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=3.2),
            ],
            vertical_connections=[connection],
        ),
    )
    duplicate = dict(graph["vertical_connections"][0])
    duplicate["id"] = "stair-duplicate"
    graph["vertical_connections"].append(duplicate)
    graph["vertical_connections"][0]["from_level_id"] = "L2"
    graph["vertical_connections"][0]["to_level_id"] = "L1"

    certificate = PlanGraphVerifier().verify(graph, permit_review_required=True)
    codes = {violation.code for violation in certificate.violations}
    assert certificate.release_allowed is False
    assert "REVERSED_VERTICAL_CONNECTION" in codes

    graph["vertical_connections"][0]["from_level_id"] = "L1"
    graph["vertical_connections"][0]["to_level_id"] = "L2"
    duplicate_certificate = PlanGraphVerifier().verify(graph, permit_review_required=True)
    duplicate_codes = {violation.code for violation in duplicate_certificate.violations}
    assert "DUPLICATE_VERTICAL_CONNECTION" in duplicate_codes


def test_building_verifier_blocks_misaligned_named_shaft() -> None:
    graph = assemble_building_graph(
        {
            "L1": _closed_graph("L1"),
            "L2": _closed_graph("L2"),
            "L3": _closed_graph("L3"),
        },
        BuildingAssemblyConfig(
            project_id="shaft-test",
            levels=[
                BuildingLevelSpec(level_id="L1", name="Ground", elevation_m=0.0),
                BuildingLevelSpec(level_id="L2", name="Second", elevation_m=3.2),
                BuildingLevelSpec(level_id="L3", name="Third", elevation_m=6.4),
            ],
            vertical_connections=[
                BuildingVerticalConnection(
                    id="riser-1",
                    shaft_id="plumbing-riser-a",
                    type="riser",
                    from_level_id="L1",
                    to_level_id="L2",
                    center_m=(2.0, 2.0),
                    footprint_m=(0.6, 0.6),
                ),
                BuildingVerticalConnection(
                    id="riser-2",
                    shaft_id="plumbing-riser-a",
                    type="riser",
                    from_level_id="L2",
                    to_level_id="L3",
                    center_m=(2.2, 2.0),
                    footprint_m=(0.6, 0.6),
                ),
            ],
        ),
    )

    codes = {item["code"] for item in graph["verification"]["violations"]}
    assert graph["verification"]["release_allowed"] is False
    assert "MISALIGNED_VERTICAL_SHAFT" in codes


def test_residential_detection_maps_to_non_box_parametric_family() -> None:
    parts = parametric_family_parts("residential-toilet", (0.7, 0.4, 0.75))
    assert len(parts) >= 4
    assert approved_family_asset_sha256("residential-toilet")


@pytest.mark.parametrize(
    ("family_id", "size", "minimum_parts"),
    [
        ("residential-base-cabinet", (1.2, 0.6, 0.9), 5),
        ("residential-wall-cabinet", (1.0, 0.35, 0.72), 5),
        ("residential-shower-enclosure", (0.9, 0.9, 2.1), 7),
        ("structural-column", (0.4, 0.4, 3.0), 3),
        ("residential-stair", (1.0, 2.8, 1.5), 8),
    ],
)
def test_extended_program_classes_have_semantic_cad_families(
    family_id: str,
    size: tuple[float, float, float],
    minimum_parts: int,
) -> None:
    parts = parametric_family_parts(family_id, size)

    assert len(parts) >= minimum_parts
    assert approved_family_asset_sha256(family_id)
