from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .core.building import (
    BuildingAssemblyConfig,
    BuildingLevelSpec,
    BuildingVerticalConnection,
    assemble_building_graph,
)
from .core.glb_export import export_editable_glb
from .core.hashing import sha256_json
from .core.ifc_export import export_ifc
from .pipeline import ConversionConfig, ConversionResult, Plan2BimConverter


class BuildingLevelInput(BaseModel):
    """One drawing page and its explicit metric placement in a building."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1, max_length=4096)
    page_number: int = Field(default=1, ge=1)
    sheet_id: str = Field(default="", max_length=160)
    plan_instance_id: str = Field(default="", max_length=220)
    level_id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=300)
    elevation_m: float
    nominal_height_m: float = Field(default=3.0, gt=0)
    pixels_per_meter: float = Field(gt=0)
    scale_source: Literal["user_supplied", "drawing_dimension", "vector_units"] = (
        "user_supplied"
    )
    wall_thickness_m: float = Field(default=0.12, gt=0)
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    rotation_deg: float = Field(default=0.0, ge=-180, le=180)

    def assembly_spec(self) -> BuildingLevelSpec:
        return BuildingLevelSpec(
            level_id=self.level_id,
            name=self.name,
            elevation_m=self.elevation_m,
            nominal_height_m=self.nominal_height_m,
            x_offset_m=self.x_offset_m,
            y_offset_m=self.y_offset_m,
            rotation_deg=self.rotation_deg,
        )


class BuildingConversionConfig(BaseModel):
    """Complete, serializable contract for a multi-page building conversion."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(default="dajoong-building", min_length=1, max_length=160)
    levels: list[BuildingLevelInput] = Field(min_length=1, max_length=10_000)
    vertical_connections: list[BuildingVerticalConnection] = Field(default_factory=list)
    pdf_dpi: int = Field(default=300, ge=72, le=600)
    threads: int = Field(default=1, ge=1, le=64)
    batch_size: int = Field(default=8, ge=1, le=256)
    allow_draft_ifc: bool = True
    allow_primary_only_smoke: bool = False

    @model_validator(mode="after")
    def validate_building(self) -> BuildingConversionConfig:
        BuildingAssemblyConfig(
            project_id=self.project_id,
            levels=[level.assembly_spec() for level in self.levels],
            vertical_connections=self.vertical_connections,
        )
        return self


class BuildingConversionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.building-conversion-result.v1"
    project_id: str
    output_dir: str
    level_results: dict[str, ConversionResult]
    plan_graph_path: str
    ifc_path: str
    ifc_certificate_path: str
    glb_path: str
    glb_manifest_path: str
    consistency_report_path: str
    manifest_path: str
    release_allowed: bool
    review_required: bool
    review_reasons: list[str]
    entity_counts: dict[str, int]
    timings_ms: dict[str, float]
    content_sha256: str = ""

    def finalize(self) -> BuildingConversionResult:
        self.content_sha256 = sha256_json(self.model_dump(mode="json", exclude={"content_sha256"}))
        return self


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    staging.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.replace(path)


def _consistency_report(graph: dict[str, Any]) -> dict[str, Any]:
    certificate = graph.get("verification") or {}
    level_ids = {str(level.get("id") or "") for level in graph.get("levels") or []}
    findings = []
    for violation in certificate.get("violations") or []:
        entity_ids = [str(item) for item in violation.get("entity_ids") or []]
        findings.append(
            {
                **violation,
                "level_ids": [entity_id for entity_id in entity_ids if entity_id in level_ids],
            }
        )
    report = {
        "schema_version": "dajoong.building-consistency-report.v1",
        "project_id": str(graph.get("project_id") or ""),
        "status": "pass" if certificate.get("release_allowed") else "blocked",
        "release_allowed": bool(certificate.get("release_allowed")),
        "review_required": bool(certificate.get("review_required", True)),
        "checked_invariants": int(certificate.get("checked_invariants") or 0),
        "passed_invariants": int(certificate.get("passed_invariants") or 0),
        "level_order": [
            str(level.get("id") or "")
            for level in sorted(
                graph.get("levels") or [], key=lambda item: float(item.get("elevation_m") or 0.0)
            )
        ],
        "findings": findings,
        "source_certificate_sha256": str(certificate.get("content_sha256") or ""),
    }
    report["content_sha256"] = sha256_json(report)
    return report


class BuildingPlan2BimConverter:
    """Convert explicit drawing pages, then assemble one verified building model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        threads: int = 1,
        batch_size: int = 8,
        semantic_model_path: str | Path | None = None,
        semantic_max_side: int | None = None,
        global_program_model_path: str | Path | None = None,
        local_element_model_path: str | Path | None = None,
        discover_native_candidates: bool | None = None,
        allow_research_global_program: bool = False,
        allow_legacy_semantic_teacher: bool = False,
    ) -> None:
        self.level_converter = Plan2BimConverter(
            model_path=model_path,
            threads=threads,
            batch_size=batch_size,
            semantic_model_path=semantic_model_path,
            semantic_max_side=semantic_max_side,
            global_program_model_path=global_program_model_path,
            local_element_model_path=local_element_model_path,
            discover_native_candidates=discover_native_candidates,
            allow_research_global_program=allow_research_global_program,
            allow_legacy_semantic_teacher=allow_legacy_semantic_teacher,
        )

    def convert(
        self,
        output_dir: str | Path,
        config: BuildingConversionConfig,
    ) -> BuildingConversionResult:
        started = time.perf_counter()
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        level_results: dict[str, ConversionResult] = {}
        level_graphs: dict[str, dict[str, Any]] = {}

        for level in sorted(config.levels, key=lambda item: item.elevation_m):
            level_output = destination / "levels" / level.level_id
            level_result = self.level_converter.convert(
                level.source_path,
                level_output,
                ConversionConfig(
                    project_id=config.project_id,
                    sheet_id=level.sheet_id or f"{level.level_id}-page-{level.page_number}",
                    plan_instance_id=level.plan_instance_id,
                    level_id=level.level_id,
                    level_name=level.name,
                    pixels_per_meter=level.pixels_per_meter,
                    scale_source=level.scale_source,
                    elevation_m=level.elevation_m,
                    nominal_height_m=level.nominal_height_m,
                    wall_thickness_m=level.wall_thickness_m,
                    threads=config.threads,
                    batch_size=config.batch_size,
                    page_number=level.page_number,
                    pdf_dpi=config.pdf_dpi,
                    allow_draft_ifc=config.allow_draft_ifc,
                    allow_primary_only_smoke=config.allow_primary_only_smoke,
                ),
            )
            level_results[level.level_id] = level_result
            level_graphs[level.level_id] = json.loads(
                Path(level_result.plan_graph_path).read_text(encoding="utf-8")
            )

        assembly_started = time.perf_counter()
        graph = assemble_building_graph(
            level_graphs,
            BuildingAssemblyConfig(
                project_id=config.project_id,
                levels=[level.assembly_spec() for level in config.levels],
                vertical_connections=config.vertical_connections,
            ),
        )
        graph_path = destination / "05-building-plan-graph.json"
        _write_json(graph_path, graph)
        consistency_report_path = destination / "05-building-consistency.json"
        _write_json(consistency_report_path, _consistency_report(graph))
        assembly_ms = (time.perf_counter() - assembly_started) * 1000

        ifc_path = destination / "05-building.ifc"
        ifc_result = export_ifc(graph, ifc_path, allow_draft=config.allow_draft_ifc)
        glb_started = time.perf_counter()
        glb_path = destination / "05-building.glb"
        glb_manifest_path = destination / "05-building.glb.manifest.json"
        glb_manifest = export_editable_glb(graph, glb_path)
        _write_json(glb_manifest_path, glb_manifest)
        glb_ms = (time.perf_counter() - glb_started) * 1000

        certificate = graph.get("verification") or {}
        review_reasons = {
            reason for result in level_results.values() for reason in result.review_reasons
        }
        review_reasons.update(
            str(item.get("code"))
            for item in certificate.get("violations") or []
            if item.get("code")
        )
        if any(result.review_required for result in level_results.values()):
            review_reasons.add("level_review_required")
        counts = {
            collection: len(graph.get(collection) or [])
            for collection in (
                "levels",
                "walls",
                "rooms",
                "openings",
                "fixtures",
                "routes",
                "vertical_connections",
                "constraints",
                "dimensions",
            )
        }
        result = BuildingConversionResult(
            project_id=config.project_id,
            output_dir=str(destination),
            level_results=level_results,
            plan_graph_path=str(graph_path),
            ifc_path=str(ifc_path),
            ifc_certificate_path=str(ifc_result["certificate"]),
            glb_path=str(glb_path),
            glb_manifest_path=str(glb_manifest_path),
            consistency_report_path=str(consistency_report_path),
            manifest_path=str(destination / "building-conversion-manifest.json"),
            release_allowed=bool(ifc_result["releaseAllowed"]),
            review_required=bool(ifc_result["reviewRequired"]),
            review_reasons=sorted(review_reasons),
            entity_counts=counts,
            timings_ms={
                "level_conversion_total": round(
                    sum(item.timings_ms["total"] for item in level_results.values()), 3
                ),
                "assembly": round(assembly_ms, 3),
                "ifc_export": round(float(ifc_result["exportSeconds"]) * 1000, 3),
                "glb_export": round(glb_ms, 3),
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        ).finalize()
        _write_json(Path(result.manifest_path), result)
        return result
