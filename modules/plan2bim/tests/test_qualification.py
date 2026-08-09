from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from buili_plan2bim import ModelQualifier, profile_drawing
from buili_plan2bim.core.plan_graph_verification import PlanGraphVerifier

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "buili_plan2bim"
QUALIFICATION_MANIFEST = (
    PACKAGE_ROOT / "models" / "aec-global-enclosure-v1.qualification.json"
)
PRIMARY_SHA = "36bcfe230be22ed869eb7bc3a940805c516dd0970c66649f944f0d5451ff1817"
SEMANTIC_SHA = "4d38d309b0d908e61fd837554e8a8c7a344df4f0f9ab429e8cf146c6dc506d85"


def test_complexity_profile_is_deterministic_except_runtime() -> None:
    image = Image.new("L", (640, 480), 255)
    draw = ImageDraw.Draw(image)
    for x in range(20, 621, 40):
        draw.line((x, 20, x, 460), fill=0, width=3)
    for y in range(20, 461, 40):
        draw.line((20, y, 620, y), fill=0, width=3)

    first = profile_drawing(image)
    second = profile_drawing(image)

    assert first.content_sha256 == second.content_sha256
    assert first.complexity_score == second.complexity_score
    assert first.difficulty_class == second.difficulty_class


def test_blank_sheet_is_simpler_than_dense_plan() -> None:
    blank = Image.new("L", (640, 480), 255)
    dense = blank.copy()
    draw = ImageDraw.Draw(dense)
    for inset in range(10, 220, 12):
        draw.rectangle((inset, inset // 2, 640 - inset, 480 - inset // 2), outline=0)

    blank_profile = profile_drawing(blank)
    dense_profile = profile_drawing(dense)

    assert blank_profile.difficulty_class == "simple"
    assert dense_profile.complexity_score > blank_profile.complexity_score


def test_exact_difficult_model_pair_still_fails_closed_on_small_benchmark() -> None:
    profile = profile_drawing(Image.new("L", (256, 256), 255)).model_copy(
        update={"difficulty_class": "difficult", "complexity_score": 0.75}
    )
    result = ModelQualifier(QUALIFICATION_MANIFEST).qualify(
        profile,
        primary_model_version="dajoong-aec-global-enclosure-router-v1",
        primary_model_sha256=PRIMARY_SHA,
        primary_release_authorized=False,
        semantic_model_version="cubicasa5k-research-teacher-onnx-v1",
        semantic_model_sha256=SEMANTIC_SHA,
        semantic_release_authorized=False,
    )

    opening = next(claim for claim in result.claims if claim.claim == "opening_detection")
    assert result.exact_model_match is True
    assert opening.status == "insufficient_sample"
    assert opening.sample_count == 3
    assert any(
        claim.claim == "whole_bim" and claim.status == "unmeasured"
        for claim in result.claims
    )
    assert result.production_release_eligible is False


def test_qualification_model_mismatch_is_explicit() -> None:
    profile = profile_drawing(Image.new("L", (256, 256), 255)).model_copy(
        update={"difficulty_class": "difficult", "complexity_score": 0.75}
    )
    result = ModelQualifier(QUALIFICATION_MANIFEST).qualify(
        profile,
        primary_model_version="other",
        primary_model_sha256="0" * 64,
        primary_release_authorized=True,
        semantic_model_version="other",
        semantic_model_sha256="1" * 64,
        semantic_release_authorized=True,
    )

    assert result.exact_model_match is False
    assert "qualification_model_pair_mismatch" in result.review_reasons


def test_plan_graph_qualification_blocks_release() -> None:
    graph: dict[str, object] = {
        "schema_version": "buili.plan-graph.v2",
        "levels": [],
        "rooms": [],
        "walls": [],
        "openings": [],
        "fixtures": [],
        "routes": [],
        "vertical_connections": [],
        "sources": [],
        "unsupported_features": [],
        "qualification": {
            "production_release_eligible": False,
            "review_required": True,
        },
    }

    certificate = PlanGraphVerifier().verify(graph)

    assert certificate.release_allowed is False
    assert any(
        violation.code == "MODEL_NOT_QUALIFIED_FOR_DRAWING_CLASS"
        for violation in certificate.violations
    )
