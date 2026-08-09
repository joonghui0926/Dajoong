from __future__ import annotations

import hmac
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

from buili_plan2bim import (
    BuildingConversionConfig,
    BuildingPlan2BimConverter,
    ConversionConfig,
    Plan2BimConverter,
)
from buili_plan2bim.core.plan_graph_verification import PlanGraphVerifier
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .aws_gateway import JobVersionConflict
from .corrections import (
    GraphCorrection,
    GraphCorrectionSet,
    apply_graph_corrections,
    correction_summary,
    graph_content_hash,
)
from .store import JobStore, StudioJob, StudioJobPage, StudioJobPublic

DATA_ROOT = Path(
    os.environ.get("DAJOONG_STUDIO_DATA") or os.environ.get("BUILI_STUDIO_DATA", "./.data/jobs")
)
store = JobStore(DATA_ROOT)
executor = ThreadPoolExecutor(
    max_workers=max(
        1,
        int(
            os.environ.get("DAJOONG_STUDIO_WORKERS") or os.environ.get("BUILI_STUDIO_WORKERS", "1")
        ),
    )
)
_aws_gateway_instance: Any | None = None
_token_verifier_instance: Any | None = None

app = FastAPI(title="Dajoong Plan2BIM Studio API", version="0.2.0")
allowed_origins = [
    item
    for item in (
        os.environ.get("DAJOONG_STUDIO_ORIGINS")
        or os.environ.get(
            "BUILI_STUDIO_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
    ).split(",")
    if item
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)


@app.middleware("http")
async def require_authentication(request: Request, call_next: Any) -> Response:
    global _token_verifier_instance
    origin_secret = os.environ.get("DAJOONG_ORIGIN_VERIFY_SECRET", "")
    if origin_secret and request.url.path != "/api/health":
        supplied_secret = request.headers.get("X-Dajoong-Origin-Verify", "")
        if not hmac.compare_digest(supplied_secret, origin_secret):
            return JSONResponse({"detail": "origin verification failed"}, status_code=403)
    require_auth = os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() == "true"
    if not require_auth or request.method == "OPTIONS" or request.url.path == "/api/health":
        return await call_next(request)
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    try:
        if _token_verifier_instance is None:
            from .auth import CognitoTokenVerifier

            _token_verifier_instance = CognitoTokenVerifier()
        request.state.identity = _token_verifier_instance.verify(authorization[7:])
    except Exception:
        return JSONResponse({"detail": "invalid or expired token"}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def protect_private_responses(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/api/") and request.url.path != "/api/health":
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


class PatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_sha256: str
    artifact_name: str
    summary: dict[str, int]
    review_required: bool
    release_allowed: bool
    job_version: int


class ImportedGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(default="imported-plan-graph.json", max_length=260)
    graph: dict[str, Any]


class AccountDeletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str


class GraphRevisionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_job_version: int = Field(ge=1)
    expected_graph_sha256: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(default="studio-user", min_length=1, max_length=200)
    operations: list[GraphCorrection] = Field(default_factory=list, max_length=20_000)
    graph: dict[str, Any]

    @model_validator(mode="after")
    def validate_snapshot_size(self) -> GraphRevisionSnapshot:
        if not all(isinstance(self.graph.get(key), list) for key in ("levels", "walls", "rooms")):
            raise ValueError("snapshot must contain PlanGraph levels, walls, and rooms")
        if len(json.dumps(self.graph, separators=(",", ":"))) > 10_000_000:
            raise ValueError("snapshot exceeds the 10 MB project-state limit")
        return self


def _using_aws() -> bool:
    return os.environ.get("DAJOONG_RUNTIME", "local").lower() == "aws"


def _aws_gateway() -> Any:
    global _aws_gateway_instance
    if _aws_gateway_instance is None:
        from .aws_gateway import AwsJobGateway

        _aws_gateway_instance = AwsJobGateway()
    return _aws_gateway_instance


def _identity(request: Request) -> tuple[str, str]:
    claims = getattr(request.state, "identity", {})
    return str(claims.get("sub", "")), str(claims.get("custom:organization_id", ""))


def _authorize(request: Request, job: StudioJob) -> None:
    if os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() != "true":
        return
    owner_id, organization_id = _identity(request)
    if job.owner_id == owner_id:
        return
    if organization_id and job.organization_id == organization_id:
        return
    raise HTTPException(status_code=404, detail="job not found")


def _safe_source_name(name: str | None) -> str:
    raw = Path(name or "drawing.png").name
    return "".join(character for character in raw if character.isalnum() or character in "._-")


def _validate_upload_size(drawing: UploadFile) -> None:
    max_bytes = int(os.environ.get("DAJOONG_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
    if not 1_048_576 <= max_bytes <= 1_073_741_824:
        raise RuntimeError("DAJOONG_MAX_UPLOAD_BYTES must be between 1 MB and 1 GB")
    position = drawing.file.tell()
    drawing.file.seek(0, 2)
    size = drawing.file.tell()
    drawing.file.seek(position)
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"drawing exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
        )


def _idempotency_key(request: Request) -> str:
    value = request.headers.get("Idempotency-Key", "").strip()
    if value and (
        not 8 <= len(value) <= 200
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in value
        )
    ):
        raise HTTPException(status_code=422, detail="invalid Idempotency-Key")
    return value


def _run_conversion(
    job_id: str,
    source_path: Path,
    config: ConversionConfig,
    semantic_model: str,
) -> None:
    job = store.get(job_id)
    job.status = "running"
    store.save(job)
    try:
        converter = Plan2BimConverter(
            threads=config.threads,
            batch_size=config.batch_size,
            semantic_model_path=semantic_model or None,
        )
        result = converter.convert(source_path, job.output_dir, config)
        job.result = result.model_dump(mode="json")
        job.graph_sha256 = graph_content_hash(
            json.loads(Path(result.plan_graph_path).read_text(encoding="utf-8"))
        )
        job.status = "review_required" if result.review_required else "complete"
    except Exception as exc:  # The persisted message is the job boundary.
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    store.save(job)


def _run_building_conversion(
    job_id: str,
    config: BuildingConversionConfig,
    semantic_model: str,
) -> None:
    job = store.get(job_id)
    job.status = "running"
    store.save(job)
    try:
        converter = BuildingPlan2BimConverter(
            threads=config.threads,
            batch_size=config.batch_size,
            semantic_model_path=semantic_model or None,
        )
        result = converter.convert(job.output_dir, config)
        job.result = result.model_dump(mode="json")
        job.graph_sha256 = graph_content_hash(
            json.loads(Path(result.plan_graph_path).read_text(encoding="utf-8"))
        )
        job.status = "review_required" if result.review_required else "complete"
    except Exception as exc:
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    store.save(job)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.delete("/api/account")
def delete_account(request: Request, payload: AccountDeletionRequest) -> dict[str, Any]:
    if payload.confirmation != "DELETE":
        raise HTTPException(status_code=422, detail="confirmation must be DELETE")
    if not _using_aws() or os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() != "true":
        raise HTTPException(
            status_code=409,
            detail="account deletion requires production authentication",
        )
    claims = getattr(request.state, "identity", {})
    owner_id = str(claims.get("sub", ""))
    username = str(claims.get("cognito:username") or claims.get("username") or "")
    if not owner_id or not username:
        raise HTTPException(status_code=401, detail="authenticated identity is incomplete")
    result = _aws_gateway().delete_account(owner_id=owner_id, username=username)
    return {"status": "deleted", **result}


@app.post("/api/jobs", response_model=StudioJobPublic)
async def create_job(
    request: Request,
    drawing: Annotated[UploadFile, File()],
    pixels_per_meter: Annotated[float, Form(gt=0)],
    project_id: Annotated[str, Form()] = "dajoong-project",
    level_id: Annotated[str, Form()] = "L1",
    level_name: Annotated[str, Form()] = "Level 1",
    elevation_m: Annotated[float, Form()] = 0.0,
    nominal_height_m: Annotated[float, Form(gt=0)] = 3.0,
    wall_thickness_m: Annotated[float, Form(gt=0)] = 0.12,
    threads: Annotated[int, Form(ge=1, le=64)] = 1,
    page_number: Annotated[int, Form(ge=1)] = 1,
    pdf_dpi: Annotated[int, Form(ge=72, le=600)] = 300,
    semantic_model: Annotated[str, Form()] = "",
) -> StudioJobPublic:
    source_name = _safe_source_name(drawing.filename)
    _validate_upload_size(drawing)
    if Path(source_name).suffix.lower() not in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        raise HTTPException(status_code=415, detail="unsupported drawing format")
    config = ConversionConfig(
        project_id=project_id,
        sheet_id=Path(source_name).stem,
        level_id=level_id,
        level_name=level_name,
        pixels_per_meter=pixels_per_meter,
        elevation_m=elevation_m,
        nominal_height_m=nominal_height_m,
        wall_thickness_m=wall_thickness_m,
        threads=threads,
        page_number=page_number,
        pdf_dpi=pdf_dpi,
    )
    owner_id, organization_id = _identity(request)
    idempotency_key = _idempotency_key(request)
    if _using_aws():
        job = _aws_gateway().create_job(
            source_name,
            drawing.file,
            config,
            semantic_model=semantic_model,
            owner_id=owner_id,
            organization_id=organization_id,
            idempotency_key=idempotency_key,
        )
    else:
        job = store.create(
            source_name,
            project_id=config.project_id,
            owner_id=owner_id,
            organization_id=organization_id,
        )
        source_path = DATA_ROOT.resolve() / job.id / source_name
        with source_path.open("wb") as output:
            shutil.copyfileobj(drawing.file, output)
        executor.submit(_run_conversion, job.id, source_path, config, semantic_model)
    return StudioJobPublic.from_job(job)


@app.post("/api/building-jobs", response_model=StudioJobPublic)
async def create_building_job(
    request: Request,
    drawing: Annotated[UploadFile, File()],
    building_config: Annotated[str, Form()],
    semantic_model: Annotated[str, Form()] = "",
) -> StudioJobPublic:
    source_name = _safe_source_name(drawing.filename)
    _validate_upload_size(drawing)
    if Path(source_name).suffix.lower() not in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        raise HTTPException(status_code=415, detail="unsupported drawing format")
    try:
        payload = json.loads(building_config)
        for level in payload.get("levels", []):
            level["source_path"] = source_name
        config = BuildingConversionConfig.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid building config: {exc}") from exc
    owner_id, organization_id = _identity(request)
    idempotency_key = _idempotency_key(request)
    if _using_aws():
        job = _aws_gateway().create_building_job(
            source_name,
            drawing.file,
            config,
            semantic_model=semantic_model,
            owner_id=owner_id,
            organization_id=organization_id,
            idempotency_key=idempotency_key,
        )
    else:
        job = store.create(
            source_name,
            project_id=config.project_id,
            owner_id=owner_id,
            organization_id=organization_id,
        )
        source_path = DATA_ROOT.resolve() / job.id / source_name
        with source_path.open("wb") as output:
            shutil.copyfileobj(drawing.file, output)
        resolved_config = config.model_copy(
            update={
                "levels": [
                    level.model_copy(update={"source_path": str(source_path)})
                    for level in config.levels
                ]
            }
        )
        executor.submit(
            _run_building_conversion,
            job.id,
            resolved_config,
            semantic_model,
        )
    return StudioJobPublic.from_job(job)


@app.post("/api/jobs/import", response_model=StudioJobPublic)
def import_graph(request: Request, payload: ImportedGraph) -> StudioJobPublic:
    owner_id, organization_id = _identity(request)
    project_id = str(payload.graph.get("project_id") or "dajoong-project")[:160]
    if _using_aws():
        job = _aws_gateway().create_imported_graph(
            _safe_source_name(payload.source_name),
            payload.graph,
            project_id=project_id,
            owner_id=owner_id,
            organization_id=organization_id,
        )
    else:
        job = store.create(
            _safe_source_name(payload.source_name),
            project_id=project_id,
            owner_id=owner_id,
            organization_id=organization_id,
        )
        output = Path(job.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        graph_path = output / "03-plan-graph.json"
        graph_path.write_text(json.dumps(payload.graph, indent=2) + "\n", encoding="utf-8")
        job.status = "review_required"
        job.result = {"plan_graph_path": str(graph_path)}
        job.graph_sha256 = graph_content_hash(payload.graph)
        store.save(job)
    return StudioJobPublic.from_job(job)


@app.get("/api/jobs", response_model=StudioJobPage)
def list_jobs(
    request: Request,
    scope: str = "personal",
    limit: int = 25,
    cursor: str = "",
) -> StudioJobPage:
    if scope not in {"personal", "organization"}:
        raise HTTPException(status_code=422, detail="scope must be personal or organization")
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    owner_id, organization_id = _identity(request)
    if scope == "organization" and not organization_id:
        raise HTTPException(status_code=403, detail="organization membership is required")
    gateway = _aws_gateway() if _using_aws() else store
    try:
        return gateway.list_for_identity(
            owner_id=owner_id,
            organization_id=organization_id,
            scope=scope,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid pagination cursor") from exc


@app.get("/api/jobs/{job_id}", response_model=StudioJobPublic)
def get_job(request: Request, job_id: str) -> StudioJobPublic:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job)
    return StudioJobPublic.from_job(job)


def _artifact_path(job: StudioJob, artifact_name: str, level_id: str = "") -> Path:
    is_building = str(job.result.get("schema_version", "")).startswith(
        "dajoong.building-conversion"
    )
    if artifact_name == "render" and job.result:
        render_path = job.result.get("source_render_path")
        if is_building:
            level_results = job.result.get("level_results") or {}
            if level_id and level_id not in level_results:
                raise HTTPException(status_code=404, detail="building level not found")
            level_result = (
                level_results[level_id] if level_id else next(iter(level_results.values()), {})
            )
            render_path = level_result.get("source_render_path")
        if isinstance(render_path, str):
            target = Path(render_path).resolve()
            job_root = (DATA_ROOT.resolve() / job.id).resolve()
            if job_root in target.parents and target.is_file():
                return target
    allowed = {
        "source": job.source_name,
        "graph": (
            "output/05-building-plan-graph.json" if is_building else "output/03-plan-graph.json"
        ),
        "glb": "output/05-building.glb" if is_building else "output/04-model.glb",
        "ifc": "output/05-building.ifc" if is_building else "output/04-model.ifc",
        "consistency": (
            "output/05-building-consistency.json"
            if is_building
            else "output/03-plan-graph.verification.json"
        ),
        "manifest": (
            "output/building-conversion-manifest.json"
            if is_building
            else "output/conversion-manifest.json"
        ),
        "overlay": "output/00-semantic-overlay.png",
        "corrected-graph": "corrected-plan-graph.json",
        "corrections": "corrections.json",
    }
    relative = allowed.get(artifact_name)
    if relative is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    target = (DATA_ROOT.resolve() / job.id / relative).resolve()
    if DATA_ROOT.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not ready")
    return target


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(
    request: Request,
    job_id: str,
    artifact_name: str,
    level_id: str = "",
) -> Response:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job)
    if _using_aws():
        try:
            body, content_type, content_length, filename = _aws_gateway().open_artifact(
                job, artifact_name, level_id=level_id
            )

            def chunks():
                try:
                    while data := body.read(1024 * 1024):
                        yield data
                finally:
                    body.close()

            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            }
            if content_length > 0:
                headers["Content-Length"] = str(content_length)
            return StreamingResponse(chunks(), media_type=content_type, headers=headers)
        except Exception as exc:  # AWS maps missing or unavailable objects to a client error.
            raise HTTPException(status_code=404, detail="artifact not ready") from exc
    path = _artifact_path(job, artifact_name, level_id=level_id)
    return FileResponse(path, filename=path.name)


@app.post("/api/jobs/{job_id}/corrections", response_model=PatchResult)
def patch_graph(request: Request, job_id: str, payload: GraphCorrectionSet) -> PatchResult:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job)
    if _using_aws():
        graph = _aws_gateway().read_editable_graph(job)
    else:
        try:
            graph_path = _artifact_path(job, "corrected-graph")
        except HTTPException:
            graph_path = _artifact_path(job, "graph")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    try:
        corrected = apply_graph_corrections(graph, payload)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    graph_sha256 = graph_content_hash(corrected)
    verification = corrected.get("verification", {})
    release_allowed = bool(verification.get("release_allowed", False))
    if _using_aws():
        try:
            _aws_gateway().commit_correction_revision(
                job,
                corrected_graph=corrected,
                corrections=payload.model_dump(mode="json"),
                graph_sha256=graph_sha256,
                release_allowed=release_allowed,
            )
        except JobVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="project changed in another session; reload before saving",
            ) from exc
        corrected_name = "corrected-plan-graph.json"
    else:
        corrected_path = store.write_json(job_id, "corrected-plan-graph.json", corrected)
        store.write_json(job_id, "corrections.json", payload.model_dump(mode="json"))
        job.active_revision = graph_sha256
        job.graph_sha256 = graph_sha256
        job.status = "complete" if release_allowed else "review_required"
        store.save(job)
        corrected_name = corrected_path.name
    return PatchResult(
        graph_sha256=graph_sha256,
        artifact_name=corrected_name,
        summary=correction_summary(payload.operations),
        review_required=bool(verification.get("review_required", True)),
        release_allowed=release_allowed,
        job_version=job.version,
    )


@app.post("/api/jobs/{job_id}/revisions", response_model=PatchResult)
def save_graph_revision(
    request: Request,
    job_id: str,
    payload: GraphRevisionSnapshot,
) -> PatchResult:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job)
    if job.version != payload.expected_job_version:
        raise HTTPException(
            status_code=409,
            detail="project changed in another session; reload before saving",
        )
    if _using_aws():
        current_graph = _aws_gateway().read_editable_graph(job)
    else:
        try:
            current_path = _artifact_path(job, "corrected-graph")
        except HTTPException:
            current_path = _artifact_path(job, "graph")
        current_graph = json.loads(current_path.read_text(encoding="utf-8"))
    if graph_content_hash(current_graph) != payload.expected_graph_sha256:
        raise HTTPException(
            status_code=409,
            detail="project changed in another session; reload before saving",
        )
    certificate = PlanGraphVerifier().verify(payload.graph)
    graph_sha256 = graph_content_hash(payload.graph)
    corrections = {
        "schema_version": "dajoong.studio-revision.v1",
        "reviewer": payload.reviewer,
        "base_graph_sha256": payload.expected_graph_sha256,
        "graph_sha256": graph_sha256,
        "operations": [item.model_dump(mode="json") for item in payload.operations],
        "verification": certificate.model_dump(mode="json"),
    }
    if _using_aws():
        try:
            _aws_gateway().commit_correction_revision(
                job,
                corrected_graph=payload.graph,
                corrections=corrections,
                graph_sha256=graph_sha256,
                release_allowed=certificate.release_allowed,
            )
        except JobVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="project changed in another session; reload before saving",
            ) from exc
    else:
        store.write_json(job_id, "corrected-plan-graph.json", payload.graph)
        store.write_json(job_id, "corrections.json", corrections)
        job.active_revision = graph_sha256
        job.graph_sha256 = graph_sha256
        job.status = "complete" if certificate.release_allowed else "review_required"
        store.save(job)
    return PatchResult(
        graph_sha256=graph_sha256,
        artifact_name="corrected-plan-graph.json",
        summary=correction_summary(payload.operations),
        review_required=certificate.review_required,
        release_allowed=certificate.release_allowed,
        job_version=job.version,
    )
