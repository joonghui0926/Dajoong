from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.bim_program import BimProgramCompiler, ProgramEvidence
from .core.glb_export import export_editable_glb
from .core.hashing import sha256_file, sha256_json
from .core.ifc_export import export_ifc
from .core.model.aec_decode import AecTileProposal
from .core.model.aec_runtime import ProofCarryingAecRunner
from .core.plan_graph_verification import PlanGraphVerifier
from .core.proposal_program import MetricLevelContext, build_program_from_tile_proposal
from .input_document import prepare_drawing
from .qualification import ModelQualifier, profile_drawing
from .semantic_recognition import OnnxFloorPlanSemanticRecognizer

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "aec-global-enclosure-v1.onnx"
DEFAULT_QUALIFICATION_PATH = (
    PACKAGE_ROOT / "models" / "aec-global-enclosure-v1.qualification.json"
)


class ConversionError(RuntimeError):
    """Raised when an input cannot produce auditable 3D geometry."""


class ConversionConfig(BaseModel):
    """Metric and runtime inputs that cannot be inferred safely from pixels alone."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(default="dajoong-project", min_length=1, max_length=160)
    sheet_id: str = Field(default="", max_length=160)
    level_id: str = Field(default="L1", min_length=1, max_length=160)
    level_name: str = Field(default="Level 1", min_length=1, max_length=300)
    pixels_per_meter: float = Field(gt=0)
    elevation_m: float = 0.0
    nominal_height_m: float = Field(default=3.0, gt=0)
    wall_thickness_m: float = Field(default=0.12, gt=0)
    threads: int = Field(default=1, ge=1, le=64)
    batch_size: int = Field(default=8, ge=1, le=256)
    page_number: int = Field(default=1, ge=1)
    pdf_dpi: int = Field(default=300, ge=72, le=600)
    allow_draft_ifc: bool = True


class ConversionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "buili.plan2bim-result.v1"
    input_path: str
    input_sha256: str
    source_render_path: str
    source_kind: str
    page_number: int
    page_count: int
    output_dir: str
    evidence_path: str
    program_path: str
    plan_graph_path: str
    ifc_path: str
    ifc_certificate_path: str
    glb_path: str
    glb_manifest_path: str
    manifest_path: str
    recognition_path: str = ""
    recognition_overlay_path: str = ""
    complexity_path: str
    qualification_path: str
    difficulty_class: str
    complexity_score: float = Field(ge=0, le=1)
    qualification_manifest_sha256: str
    production_release_eligible: bool
    model_version: str
    model_sha256: str
    semantic_model_version: str = ""
    semantic_model_sha256: str = ""
    semantic_production_authorized: bool = False
    release_allowed: bool
    review_required: bool
    review_reasons: list[str]
    unresolved_symbols: int
    entity_counts: dict[str, int]
    timings_ms: dict[str, float]
    content_sha256: str = ""

    def finalize(self) -> ConversionResult:
        payload = self.model_dump(mode="json", exclude={"content_sha256"})
        self.content_sha256 = sha256_json(payload)
        return self


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    staging.write_text(text + "\n", encoding="utf-8", newline="\n")
    staging.replace(path)


class Plan2BimConverter:
    """One-process CPU conversion from a plan image to proof-carrying IFC.

    Neural inference proposes geometry. The deterministic compiler performs metric
    conversion and IFC generation. The verifier keeps unsupported or uncertain
    output in a review-required draft state instead of silently approving it.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        threads: int = 1,
        batch_size: int = 8,
        semantic_model_path: str | Path | None = None,
        semantic_max_side: int = 1024,
        qualification_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
        self.runner = ProofCarryingAecRunner(
            self.model_path,
            threads=threads,
            batch_size=batch_size,
        )
        self.semantic_recognizer = (
            OnnxFloorPlanSemanticRecognizer(semantic_model_path, threads=threads)
            if semantic_model_path is not None
            else None
        )
        if semantic_max_side < 64:
            raise ValueError("semantic_max_side must be at least 64")
        self.semantic_max_side = semantic_max_side
        self.qualifier = ModelQualifier(qualification_path or DEFAULT_QUALIFICATION_PATH)

    def model_card(self) -> dict[str, Any]:
        card = self.runner.model_card()
        card["qualification_manifest_path"] = str(self.qualifier.manifest_path)
        card["qualification_manifest_sha256"] = self.qualifier.manifest_sha256
        card["qualification_production_authorized"] = bool(
            self.qualifier.manifest.get("production_authorized", False)
        )
        card["content_sha256"] = sha256_json(
            {key: value for key, value in card.items() if key != "content_sha256"}
        )
        return card

    def convert(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        config: ConversionConfig,
    ) -> ConversionResult:
        started = time.perf_counter()
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        sheet_id = config.sheet_id or source.stem
        source_hash = sha256_file(source)
        prepared = prepare_drawing(
            source,
            destination,
            page_number=config.page_number,
            pdf_dpi=config.pdf_dpi,
        )
        render_source = Path(prepared.render_path)

        complexity_started = time.perf_counter()
        with Image.open(render_source) as image:
            drawing_profile = profile_drawing(image)
        complexity_ms = (time.perf_counter() - complexity_started) * 1000
        complexity_path = destination / "00-drawing-complexity.json"
        _write_json(complexity_path, drawing_profile)

        inference_started = time.perf_counter()
        with Image.open(render_source) as image:
            inference = self.runner.compile_image(
                image.convert("L"),
                sheet_id=sheet_id,
                source_ref_ids=[source_hash],
            )
        inference_ms = (time.perf_counter() - inference_started) * 1000
        recognition_path = ""
        recognition_overlay_path = ""
        semantic_model_version = ""
        semantic_model_sha256 = ""
        semantic_production_authorized = False
        semantic_ms = 0.0
        if self.semantic_recognizer is not None:
            semantic_started = time.perf_counter()
            recognition, wall_mask = self.semantic_recognizer.recognize(
                render_source,
                max_side=self.semantic_max_side,
            )
            recognition_path_obj = destination / "00-semantic-recognition.json"
            recognition_overlay_obj = destination / "00-semantic-overlay.png"
            recognition.overlay_path = str(recognition_overlay_obj)
            recognition.finalize()
            _write_json(recognition_path_obj, recognition)
            self.semantic_recognizer.render_overlay(
                render_source,
                recognition,
                wall_mask,
                recognition_overlay_obj,
            )
            recognition_path = str(recognition_path_obj)
            recognition_overlay_path = str(recognition_overlay_obj)
            semantic_model_version = recognition.model_version
            semantic_model_sha256 = recognition.model_sha256
            semantic_production_authorized = recognition.production_authorized
            semantic_ms = (time.perf_counter() - semantic_started) * 1000
            if inference.proposal is not None:
                symbols = self.semantic_recognizer.symbol_proposals(
                    recognition,
                    source_ref_ids=[source_hash],
                )
                semantic_walls = self.semantic_recognizer.wall_proposals(
                    recognition,
                    source_ref_ids=[source_hash],
                )
                semantic_rooms = self.semantic_recognizer.room_proposals(
                    recognition,
                    source_ref_ids=[source_hash],
                )
                inference.proposal = AecTileProposal(
                    tile_id=inference.proposal.tile_id,
                    source_ref_ids=inference.proposal.source_ref_ids,
                    model_version=(
                        f"{inference.proposal.model_version}+{recognition.model_version}"
                    ),
                    wall_segments=(
                        semantic_walls if semantic_walls else inference.proposal.wall_segments
                    ),
                    symbols=symbols,
                    room_regions=semantic_rooms,
                    rejected_candidates=inference.proposal.rejected_candidates,
                ).finalize()
                inference.review_reasons = sorted(
                    set(inference.review_reasons + ["semantic_model_not_production_authorized"])
                )
                inference.release_allowed = (
                    inference.release_allowed and recognition.production_authorized
                )
                inference.finalize()

        runner_card = self.runner.model_card()
        qualification = self.qualifier.qualify(
            drawing_profile,
            primary_model_version=inference.model_version,
            primary_model_sha256=inference.model_artifact_sha256,
            primary_release_authorized=bool(runner_card.get("release_authorized", False)),
            semantic_model_version=semantic_model_version,
            semantic_model_sha256=semantic_model_sha256,
            semantic_release_authorized=semantic_production_authorized,
        )
        qualification_path = destination / "00-model-qualification.json"
        _write_json(qualification_path, qualification)
        evidence_path = destination / "01-evidence.json"
        _write_json(evidence_path, inference)
        if inference.proposal is None:
            raise ConversionError(
                "No proposal was produced; inspect 01-evidence.json for the tile-ledger reason."
            )
        if not inference.proposal.wall_segments:
            raise ConversionError(
                "The model produced no wall geometry; an empty IFC was not emitted. "
                "Inspect 01-evidence.json and verify the drawing crop and scale."
            )

        evidence = ProgramEvidence(
            id=source_hash,
            uri=f"{source.as_uri()}#page={prepared.page_number}",
            sha256=source_hash,
            page_number=prepared.page_number,
            source_kind=prepared.source_kind,
            extractor="buili-plan2bim-global-enclosure-v1",
            model_version=inference.model_version,
        )
        context = MetricLevelContext(
            project_id=config.project_id,
            level_id=config.level_id,
            level_name=config.level_name,
            elevation_m=config.elevation_m,
            nominal_height_m=config.nominal_height_m,
            pixels_per_meter=config.pixels_per_meter,
            wall_thickness_m=config.wall_thickness_m,
            evidence=evidence,
            independent_evidence_groups=1,
        )

        compile_started = time.perf_counter()
        build = build_program_from_tile_proposal(
            inference.proposal,
            context,
            coverage_certificate=inference.coverage,
        )
        program_path = destination / "02-bim-program.json"
        _write_json(program_path, build.program)
        graph = BimProgramCompiler().compile(build.program)
        graph["drawing_profile"] = drawing_profile.model_dump(mode="json")
        graph["qualification"] = qualification.model_dump(mode="json")
        if qualification.review_required:
            graph.setdefault("confidence", {})["review_required"] = True
            graph.setdefault("pipeline", {})["review_required"] = True
        _recertify_plan_graph(graph)
        graph_path = destination / "03-plan-graph.json"
        _write_json(graph_path, graph)
        compile_ms = (time.perf_counter() - compile_started) * 1000

        ifc_path = destination / "04-model.ifc"
        ifc_result = export_ifc(
            graph,
            ifc_path,
            allow_draft=config.allow_draft_ifc,
        )
        certificate_path = Path(ifc_result["certificate"])
        glb_started = time.perf_counter()
        glb_path = destination / "04-model.glb"
        glb_manifest_path = destination / "04-model.glb.manifest.json"
        glb_manifest = export_editable_glb(graph, glb_path)
        _write_json(glb_manifest_path, glb_manifest)
        glb_ms = (time.perf_counter() - glb_started) * 1000
        timings = {
            "complexity_profiling": round(complexity_ms, 3),
            "inference": round(inference_ms, 3),
            "semantic_recognition": round(semantic_ms, 3),
            "compile": round(compile_ms, 3),
            "ifc_export": round(float(ifc_result["exportSeconds"]) * 1000, 3),
            "glb_export": round(glb_ms, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        }
        counts = {
            "walls": len(graph.get("walls", [])),
            "rooms": len(graph.get("rooms", [])),
            "openings": len(graph.get("openings", [])),
            "fixtures": len(graph.get("fixtures", [])),
            "routes": len(graph.get("routes", [])),
        }
        review_reasons = list(inference.review_reasons)
        review_reasons.extend(qualification.review_reasons)
        if build.unresolved_symbols:
            review_reasons.append("unresolved_symbols")
        if bool(graph.get("confidence", {}).get("review_required", True)):
            review_reasons.append("plan_graph_review_required")
        review_reasons = sorted(set(review_reasons))
        result = ConversionResult(
            input_path=str(source),
            input_sha256=source_hash,
            source_render_path=str(render_source),
            source_kind=prepared.source_kind,
            page_number=prepared.page_number,
            page_count=prepared.page_count,
            output_dir=str(destination),
            evidence_path=str(evidence_path),
            program_path=str(program_path),
            plan_graph_path=str(graph_path),
            ifc_path=str(ifc_path),
            ifc_certificate_path=str(certificate_path),
            glb_path=str(glb_path),
            glb_manifest_path=str(glb_manifest_path),
            manifest_path=str(destination / "conversion-manifest.json"),
            recognition_path=recognition_path,
            recognition_overlay_path=recognition_overlay_path,
            complexity_path=str(complexity_path),
            qualification_path=str(qualification_path),
            difficulty_class=drawing_profile.difficulty_class,
            complexity_score=drawing_profile.complexity_score,
            qualification_manifest_sha256=qualification.manifest_sha256,
            production_release_eligible=qualification.production_release_eligible,
            model_version=inference.model_version,
            model_sha256=inference.model_artifact_sha256,
            semantic_model_version=semantic_model_version,
            semantic_model_sha256=semantic_model_sha256,
            semantic_production_authorized=semantic_production_authorized,
            release_allowed=bool(ifc_result["releaseAllowed"]),
            review_required=bool(ifc_result["reviewRequired"]),
            review_reasons=review_reasons,
            unresolved_symbols=len(build.unresolved_symbols),
            entity_counts=counts,
            timings_ms=timings,
        ).finalize()
        _write_json(Path(result.manifest_path), result)
        return result


def convert_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    pixels_per_meter: float,
    project_id: str = "dajoong-project",
    level_id: str = "L1",
    level_name: str = "Level 1",
    elevation_m: float = 0.0,
    nominal_height_m: float = 3.0,
    wall_thickness_m: float = 0.12,
    model_path: str | Path | None = None,
    threads: int = 1,
    batch_size: int = 8,
    semantic_model_path: str | Path | None = None,
    semantic_max_side: int = 1024,
    page_number: int = 1,
    pdf_dpi: int = 300,
) -> ConversionResult:
    """Convenience API for one-shot conversion."""

    config = ConversionConfig(
        project_id=project_id,
        level_id=level_id,
        level_name=level_name,
        pixels_per_meter=pixels_per_meter,
        elevation_m=elevation_m,
        nominal_height_m=nominal_height_m,
        wall_thickness_m=wall_thickness_m,
        threads=threads,
        batch_size=batch_size,
        page_number=page_number,
        pdf_dpi=pdf_dpi,
    )
    converter = Plan2BimConverter(
        model_path=model_path,
        threads=config.threads,
        batch_size=config.batch_size,
        semantic_model_path=semantic_model_path,
        semantic_max_side=semantic_max_side,
    )
    return converter.convert(image_path, output_dir, config)


def _recertify_plan_graph(graph: dict[str, Any]) -> None:
    """Replace stale compiler certification after qualification metadata is attached."""

    graph.pop("verification", None)
    pipeline = graph.setdefault("pipeline", {})
    for key in ("release_allowed", "certificate_sha256", "content_sha256"):
        pipeline.pop(key, None)
    certificate = PlanGraphVerifier().verify(graph)
    graph["verification"] = certificate.model_dump(mode="json")
    pipeline["release_allowed"] = certificate.release_allowed
    pipeline["certificate_sha256"] = certificate.content_sha256
    pipeline["content_sha256"] = sha256_json(
        {key: value for key, value in graph.items() if key != "verification"}
    )
