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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .corrections import (
    GraphCorrectionSet,
    apply_graph_corrections,
    correction_summary,
    graph_content_hash,
)
from .store import JobStore, StudioJob

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
    allow_headers=["Authorization", "Content-Type"],
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


class PatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_sha256: str
    artifact_name: str
    summary: dict[str, int]
    review_required: bool
    release_allowed: bool


class ImportedGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(default="imported-plan-graph.json", max_length=260)
    graph: dict[str, Any]


class AccountDeletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str


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


@app.post("/api/jobs", response_model=StudioJob)
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
) -> StudioJob:
    source_name = _safe_source_name(drawing.filename)
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
    if _using_aws():
        return _aws_gateway().create_job(
            source_name,
            drawing.file,
            config,
            semantic_model=semantic_model,
            owner_id=owner_id,
            organization_id=organization_id,
        )
    job = store.create(
        source_name,
        owner_id=owner_id,
        organization_id=organization_id,
    )
    source_path = DATA_ROOT.resolve() / job.id / source_name
    with source_path.open("wb") as output:
        shutil.copyfileobj(drawing.file, output)
    executor.submit(_run_conversion, job.id, source_path, config, semantic_model)
    return job


@app.post("/api/building-jobs", response_model=StudioJob)
async def create_building_job(
    request: Request,
    drawing: Annotated[UploadFile, File()],
    building_config: Annotated[str, Form()],
    semantic_model: Annotated[str, Form()] = "",
) -> StudioJob:
    source_name = _safe_source_name(drawing.filename)
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
    if _using_aws():
        return _aws_gateway().create_building_job(
            source_name,
            drawing.file,
            config,
            semantic_model=semantic_model,
            owner_id=owner_id,
            organization_id=organization_id,
        )
    job = store.create(
        source_name,
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
    return job


@app.post("/api/jobs/import", response_model=StudioJob)
def import_graph(request: Request, payload: ImportedGraph) -> StudioJob:
    owner_id, organization_id = _identity(request)
    if _using_aws():
        return _aws_gateway().create_imported_graph(
            _safe_source_name(payload.source_name),
            payload.graph,
            owner_id=owner_id,
            organization_id=organization_id,
        )
    job = store.create(
        _safe_source_name(payload.source_name),
        owner_id=owner_id,
        organization_id=organization_id,
    )
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "03-plan-graph.json"
    graph_path.write_text(json.dumps(payload.graph, indent=2) + "\n", encoding="utf-8")
    job.status = "review_required"
    job.result = {"plan_graph_path": str(graph_path)}
    store.save(job)
    return job


@app.get("/api/jobs/{job_id}", response_model=StudioJob)
def get_job(request: Request, job_id: str) -> StudioJob:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job)
    return job


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
        graph = _aws_gateway().read_json(job, "graph")
    else:
        graph_path = _artifact_path(job, "graph")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    try:
        corrected = apply_graph_corrections(graph, payload)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if _using_aws():
        _aws_gateway().write_json(job, "corrected-graph", corrected)
        _aws_gateway().write_json(job, "corrections", payload.model_dump(mode="json"))
        corrected_name = "corrected-plan-graph.json"
    else:
        corrected_path = store.write_json(job_id, "corrected-plan-graph.json", corrected)
        store.write_json(job_id, "corrections.json", payload.model_dump(mode="json"))
        corrected_name = corrected_path.name
    verification = corrected.get("verification", {})
    return PatchResult(
        graph_sha256=graph_content_hash(corrected),
        artifact_name=corrected_name,
        summary=correction_summary(payload.operations),
        review_required=bool(verification.get("review_required", True)),
        release_allowed=bool(verification.get("release_allowed", False)),
    )
