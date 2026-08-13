from __future__ import annotations

import hmac
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from buili_plan2bim import (
    BuildingConversionConfig,
    BuildingPlan2BimConverter,
    ConversionConfig,
    Plan2BimConverter,
    require_product_architectural_runtime,
)
from buili_plan2bim.core.plan_graph_verification import PlanGraphVerifier
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .aws_gateway import DurableJobEnqueueError, JobVersionConflict
from .asset_delivery import asset_catalog, asset_mesh_path, externalize_graph_assets
from .billing import BillingService, InsufficientCredit
from .collaboration import (
    CollaborationService,
    ModelVersionRecord,
    WorkspaceAccessError,
    WorkspaceConflict,
    WorkspaceRole,
)
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
RUNTIME = os.environ.get("DAJOONG_RUNTIME", "local").lower()
ENVIRONMENT = os.environ.get("DAJOONG_ENVIRONMENT", "development").lower()
if ENVIRONMENT == "production" and RUNTIME != "aws":
    raise RuntimeError("production must use the durable AWS runtime")
if (
    ENVIRONMENT == "production"
    and os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() != "true"
):
    raise RuntimeError("production must require authenticated identities")
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
_billing_service_instance: BillingService | None = None
_collaboration_service_instance: CollaborationService | None = None
_collaboration_service_root: Path | None = None

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
    allow_origin_regex=(
        r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$"
        if ENVIRONMENT != "production"
        else None
    ),
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
        "X-Dajoong-Organization",
        "X-Request-ID",
    ],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: Any) -> Response:
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if supplied.isalnum() and len(supplied) <= 64 else uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def require_authentication(request: Request, call_next: Any) -> Response:
    global _token_verifier_instance
    origin_secret = os.environ.get("DAJOONG_ORIGIN_VERIFY_SECRET", "")
    if origin_secret and request.url.path != "/api/health":
        supplied_secret = request.headers.get("X-Dajoong-Origin-Verify", "")
        if not hmac.compare_digest(supplied_secret, origin_secret):
            return JSONResponse({"detail": "origin verification failed"}, status_code=403)
    require_auth = os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() == "true"
    public_paths = {"/api/health", "/api/billing/stripe/webhook"}
    public_asset = request.method == "GET" and request.url.path.startswith("/api/assets/v1/")
    if (
        not require_auth
        or request.method == "OPTIONS"
        or request.url.path in public_paths
        or public_asset
    ):
        return await call_next(request)
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    try:
        if _token_verifier_instance is None:
            from .auth import CognitoTokenVerifier

            _token_verifier_instance = CognitoTokenVerifier()
        request.state.identity = await run_in_threadpool(
            _token_verifier_instance.verify,
            authorization[7:],
        )
    except Exception:
        return JSONResponse({"detail": "invalid or expired token"}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def protect_private_responses(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    if request.url.path.startswith("/api/assets/v1/"):
        response.headers["X-Content-Type-Options"] = "nosniff"
    elif request.url.path.startswith("/api/") and request.url.path != "/api/health":
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

    @model_validator(mode="after")
    def validate_graph_size(self) -> ImportedGraph:
        if len(json.dumps(self.graph, separators=(",", ":"))) > 10_000_000:
            raise ValueError("imported graph exceeds the 10 MB project-state limit")
        return self


class AccountDeletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(pattern="^(stripe|toss)$")
    country: str = Field(default="", max_length=2)
    units: int = Field(default=1, ge=1, le=20)
    plan: str = Field(default="per_drawing", pattern="^(per_drawing|unlimited_monthly)$")
    easy_pay: str = Field(default="", pattern="^(|KAKAOPAY|TOSSPAY)$")


class CheckoutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    order_id: str
    redirect_url: str = ""
    toss: dict[str, Any] = Field(default_factory=dict)


class TossConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=6, max_length=64)
    payment_key: str = Field(min_length=6, max_length=300)
    amount: int = Field(gt=0)


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


class OrganizationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=100)
    approved_domains: list[str] = Field(default_factory=list, max_length=20)
    domain_join_enabled: bool = False


class WorkspaceInvitationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    role: Literal["admin", "editor", "commenter", "viewer"] = "editor"


class InvitationAcceptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=20, max_length=500)


class MemberRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["admin", "editor", "commenter", "viewer"]


class ProjectCommentCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=1, max_length=5000)
    entity_ref: str = Field(default="", max_length=300)
    assigned_to: str = Field(default="", max_length=200)


class ProjectCommentStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["open", "resolved"]


class PresenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=8, max_length=100)
    active_entity: str = Field(default="", max_length=300)


class VersionRestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_job_version: int = Field(ge=1)


def _using_aws() -> bool:
    return RUNTIME == "aws"


def _aws_gateway() -> Any:
    global _aws_gateway_instance
    if _aws_gateway_instance is None:
        from .aws_gateway import AwsJobGateway

        _aws_gateway_instance = AwsJobGateway()
    return _aws_gateway_instance


def _billing_service() -> BillingService:
    global _billing_service_instance
    if _billing_service_instance is None:
        _billing_service_instance = BillingService(DATA_ROOT.parent)
    return _billing_service_instance


def _collaboration_service() -> CollaborationService:
    global _collaboration_service_instance, _collaboration_service_root
    root = DATA_ROOT.parent.resolve()
    if _collaboration_service_instance is None or _collaboration_service_root != root:
        _collaboration_service_instance = CollaborationService(root)
        _collaboration_service_root = root
    return _collaboration_service_instance


def _actor(request: Request) -> tuple[str, str, str]:
    claims = getattr(request.state, "identity", {})
    user_id = str(claims.get("sub", ""))
    email = str(claims.get("email", ""))
    display_name = str(
        claims.get("name")
        or claims.get("preferred_username")
        or (email.split("@", 1)[0] if email else "")
    )
    if user_id:
        return user_id, email, display_name or "Team member"
    if os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() == "true":
        raise HTTPException(status_code=401, detail="authentication required")
    return "local-development-user", "demo@dajoong.local", "Demo user"


def _require_verified_email(request: Request) -> None:
    if os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() != "true":
        return
    claims = getattr(request.state, "identity", {})
    if claims.get("email_verified") is not True:
        raise HTTPException(status_code=403, detail="verify your email address first")


def _active_membership(
    request: Request,
    minimum_role: WorkspaceRole = "viewer",
    *,
    required: bool = False,
) -> Any | None:
    user_id, _, _ = _actor(request)
    service = _collaboration_service()
    organization_id = request.headers.get("X-Dajoong-Organization", "").strip()
    if not organization_id:
        if required:
            raise HTTPException(status_code=409, detail="select a company workspace")
        return None
    try:
        return service.require_member(organization_id, user_id, minimum_role)
    except WorkspaceAccessError as exc:
        raise HTTPException(status_code=403, detail="workspace access is not available") from exc


def _identity(request: Request) -> tuple[str, str]:
    owner_id, _, _ = _actor(request)
    member = _active_membership(request)
    return owner_id, member.organization_id if member else ""


def _billing_account_id(request: Request) -> str:
    owner_id, organization_id = _identity(request)
    if organization_id:
        return organization_id
    if owner_id:
        return owner_id
    if os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() == "true":
        raise HTTPException(status_code=401, detail="authentication required")
    return "local-development-account"


def _billing_country(request: Request, override: str = "") -> str:
    return override or request.headers.get("X-Dajoong-Country", "US")


def _reserve_conversion(
    request: Request,
    units: int,
    idempotency_key: str,
) -> tuple[str, str] | None:
    service = _billing_service()
    if not service.billing_enforced():
        return None
    if not idempotency_key:
        raise HTTPException(status_code=422, detail="Idempotency-Key is required for conversion")
    account_id = _billing_account_id(request)
    try:
        service.reserve(account_id, units, idempotency_key)
    except InsufficientCredit as exc:
        context = service.context(account_id, _billing_country(request))
        raise HTTPException(
            status_code=402,
            detail={
                "code": "PAYMENT_REQUIRED",
                "required_units": exc.required,
                "available_units": exc.available,
                "checkout": context.model_dump(mode="json"),
            },
        ) from exc
    return account_id, idempotency_key


def _release_conversion(reservation: tuple[str, str] | None) -> None:
    if reservation:
        _billing_service().release(*reservation)


def _authorize(
    request: Request,
    job: StudioJob,
    minimum_role: WorkspaceRole = "viewer",
) -> None:
    if (
        os.environ.get("DAJOONG_REQUIRE_AUTH", "false").lower() != "true"
        and not job.organization_id
    ):
        return
    owner_id, organization_id = _identity(request)
    if job.organization_id:
        if organization_id and job.organization_id == organization_id:
            _active_membership(request, minimum_role, required=True)
            return
        raise HTTPException(status_code=404, detail="job not found")
    if job.owner_id == owner_id:
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


def _conversion_runtime() -> tuple[int, int]:
    threads = int(os.environ.get("DAJOONG_CONVERSION_THREADS", "1"))
    batch_size = int(os.environ.get("DAJOONG_CONVERSION_BATCH_SIZE", "1"))
    if not 1 <= threads <= 64 or not 1 <= batch_size <= 256:
        raise RuntimeError("conversion runtime limits are invalid")
    return threads, batch_size


def _require_conversion_runtime() -> None:
    """Fail before billing or upload persistence when full BIM is unavailable."""

    try:
        if _using_aws():
            global_path, local_path = _aws_gateway().architectural_model_paths()
            require_product_architectural_runtime(global_path or None, local_path or None)
        else:
            require_product_architectural_runtime()
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "ARCHITECTURAL_RUNTIME_UNAVAILABLE",
                "message": (
                    "Full drawing conversion is temporarily unavailable. "
                    "No credit was used and the drawing was not uploaded."
                ),
            },
        ) from exc


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
) -> None:
    job = store.get(job_id)
    job.status = "running"
    store.save(job)
    try:
        converter = Plan2BimConverter(
            threads=config.threads,
            batch_size=config.batch_size,
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
) -> None:
    job = store.get(job_id)
    job.status = "running"
    store.save(job)
    try:
        converter = BuildingPlan2BimConverter(
            threads=config.threads,
            batch_size=config.batch_size,
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


@app.get("/api/assets/v1/catalog")
def family_asset_catalog() -> Response:
    payload = asset_catalog()
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "public, max-age=300, stale-while-revalidate=86400",
            "ETag": f'"{payload["content_sha256"]}"',
        },
    )


@app.get("/api/assets/v1/{mesh_sha256}.mesh")
def family_asset_mesh(mesh_sha256: str) -> Response:
    try:
        path = asset_mesh_path(mesh_sha256)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="family asset not found") from exc
    return FileResponse(
        path,
        media_type="application/vnd.dajoong.mesh",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{mesh_sha256}"',
        },
    )


@app.get("/api/billing/context")
def billing_context(
    request: Request,
    country: str = "",
    platform: str = "",
) -> dict[str, Any]:
    if platform not in {"", "web", "ios", "android"}:
        raise HTTPException(status_code=422, detail="unsupported checkout platform")
    context = _billing_service().context(
        _billing_account_id(request),
        _billing_country(request, country),
        native_platform=platform if platform in {"ios", "android"} else "",
    )
    return context.model_dump(mode="json")


@app.post("/api/billing/checkout", response_model=CheckoutResponse)
def create_checkout(request: Request, payload: CheckoutRequest) -> CheckoutResponse:
    if _active_membership(request):
        _active_membership(request, "admin", required=True)
    account_id = _billing_account_id(request)
    idempotency_key = _idempotency_key(request)
    if not idempotency_key:
        raise HTTPException(status_code=422, detail="Idempotency-Key is required for checkout")
    try:
        order, _ = _billing_service().create_order(
            account_id,
            payload.provider,  # type: ignore[arg-type]
            _billing_country(request, payload.country),
            payload.units,
            payload.plan,  # type: ignore[arg-type]
            idempotency_key,
        )
        if payload.provider == "stripe":
            claims = getattr(request.state, "identity", {})
            redirect_url = _billing_service().stripe_checkout(
                order,
                str(claims.get("email", "")),
            )
            return CheckoutResponse(
                kind="redirect",
                order_id=order.id,
                redirect_url=redirect_url,
            )
        return CheckoutResponse(
            kind="toss_payment",
            order_id=order.id,
            toss=_billing_service().toss_prepare(order, payload.easy_pay),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(
            status_code=503,
            detail="payment provider is temporarily unavailable",
        ) from exc


@app.post("/api/billing/toss/confirm")
def confirm_toss_payment(request: Request, payload: TossConfirmation) -> dict[str, Any]:
    if _active_membership(request):
        _active_membership(request, "admin", required=True)
    try:
        account = _billing_service().confirm_toss(
            _billing_account_id(request),
            payload.order_id,
            payload.payment_key,
            payload.amount,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Toss payment confirmation failed") from exc
    return {
        "status": "paid",
        "free_units_remaining": account.free_units_remaining,
        "paid_units": account.paid_units,
    }


@app.post("/api/billing/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, bool]:
    payload = await request.body()
    try:
        event = _billing_service().verify_stripe_signature(
            payload,
            request.headers.get("Stripe-Signature", ""),
        )
        _billing_service().complete_stripe_event(event)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid Stripe webhook") from exc
    return {"received": True}


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
    memberships = _collaboration_service().store.organizations_for_user(owner_id)
    owned = [organization.name for organization, member in memberships if member.role == "owner"]
    if owned:
        raise HTTPException(
            status_code=409,
            detail="transfer company workspace ownership before deleting this account",
        )
    for organization, _ in memberships:
        _collaboration_service().store.delete_member(organization.id, owner_id)
    result = _aws_gateway().delete_account(owner_id=owner_id, username=username)
    return {"status": "deleted", **result}


def _workspace_payload(
    request: Request,
    member_limit: int,
    member_cursor: str,
) -> dict[str, Any]:
    user_id, _, _ = _actor(request)
    service = _collaboration_service()
    organizations = service.store.organizations_for_user(user_id)
    requested_id = request.headers.get("X-Dajoong-Organization", "").strip()
    selected = next((pair for pair in organizations if pair[0].id == requested_id), None)
    if not selected and len(organizations) == 1:
        selected = organizations[0]
    members: list[dict[str, Any]] = []
    next_member_cursor = ""
    invitations: list[dict[str, Any]] = []
    if selected:
        organization, membership = selected
        try:
            member_page, next_member_cursor = service.store.list_members_page(
                organization.id,
                member_limit,
                member_cursor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid member cursor") from exc
        members = [item.model_dump(mode="json") for item in member_page]
        if membership.role in {"owner", "admin"}:
            invitations = [
                _public_invitation(item)
                for item in service.store.list_invitations(organization.id)
                if item.status == "pending" and item.expires_at > int(time.time())
            ]
    return {
        "organizations": [
            {
                "organization": _public_organization(organization),
                "membership": membership.model_dump(mode="json"),
            }
            for organization, membership in organizations
        ],
        "active_organization_id": selected[0].id if selected else "",
        "members": members,
        "member_cursor": next_member_cursor,
        "invitations": invitations,
    }


def _public_organization(organization: Any) -> dict[str, Any]:
    return organization.model_dump(mode="json", exclude={"created_by"})


def _public_invitation(invitation: Any) -> dict[str, Any]:
    return invitation.model_dump(
        mode="json",
        exclude={"token_sha256", "invited_by", "accepted_by"},
    )


@app.get("/api/workspace")
def get_workspace(
    request: Request,
    member_limit: int = 100,
    member_cursor: str = "",
) -> dict[str, Any]:
    if not 1 <= member_limit <= 250:
        raise HTTPException(status_code=422, detail="member_limit must be between 1 and 250")
    return _workspace_payload(request, member_limit, member_cursor)


@app.post("/api/workspace/organizations")
def create_organization(
    request: Request,
    payload: OrganizationCreateRequest,
) -> dict[str, Any]:
    user_id, email, display_name = _actor(request)
    if payload.approved_domains:
        _require_verified_email(request)
        email_domain = email.lower().partition("@")[2]
        if email_domain not in {
            item.strip().lower().lstrip("@").rstrip(".")
            for item in payload.approved_domains
        }:
            raise HTTPException(
                status_code=403,
                detail="the approved domain must match your verified work email",
            )
    try:
        organization = _collaboration_service().create_organization(
            user_id=user_id,
            email=email,
            display_name=display_name,
            name=payload.name,
            approved_domains=payload.approved_domains,
            domain_join_enabled=payload.domain_join_enabled,
        )
    except (ValueError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"organization": _public_organization(organization)}


@app.post("/api/workspace/join-domain")
def join_workspace_by_domain(request: Request) -> dict[str, Any]:
    _require_verified_email(request)
    user_id, email, display_name = _actor(request)
    try:
        member = _collaboration_service().join_by_domain(
            user_id=user_id,
            email=email,
            display_name=display_name,
        )
    except (ValueError, WorkspaceAccessError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"membership": member.model_dump(mode="json")}


def _send_workspace_invitation(
    *,
    email: str,
    organization_name: str,
    inviter_name: str,
    accept_url: str,
) -> bool:
    sender = os.environ.get("DAJOONG_INVITE_FROM_EMAIL", "").strip()
    if not sender or not _using_aws():
        return False
    try:
        import boto3

        boto3.client("sesv2").send_email(
            FromEmailAddress=sender,
            Destination={"ToAddresses": [email]},
            Content={
                "Simple": {
                    "Subject": {"Data": f"Join {organization_name} in Dajoong"},
                    "Body": {
                        "Text": {
                            "Data": (
                                f"{inviter_name} invited you to {organization_name} in Dajoong.\n\n"
                                f"Accept the invitation: {accept_url}\n\n"
                                "This invitation expires in 7 days."
                            )
                        }
                    },
                }
            },
        )
    except Exception:
        return False
    return True


@app.post("/api/workspace/invitations")
def invite_workspace_member(
    request: Request,
    payload: WorkspaceInvitationRequest,
) -> dict[str, Any]:
    actor_id, _, actor_name = _actor(request)
    member = _active_membership(request, "admin", required=True)
    try:
        receipt = _collaboration_service().create_invitation(
            organization_id=member.organization_id,
            inviter_id=actor_id,
            email=payload.email,
            role=payload.role,
        )
    except WorkspaceAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    organization = _collaboration_service().store.get_organization(member.organization_id)
    app_url = os.environ.get("DAJOONG_APP_URL", "https://dajoongbim.com").rstrip("/")
    accept_url = f"{app_url}/studio?invite={receipt.token}"
    delivered = _send_workspace_invitation(
        email=receipt.invitation.email,
        organization_name=organization.name,
        inviter_name=actor_name,
        accept_url=accept_url,
    )
    return {
        "invitation": _public_invitation(receipt.invitation),
        "accept_url": accept_url,
        "email_delivered": delivered,
    }


@app.post("/api/workspace/invitations/accept")
def accept_workspace_invitation(
    request: Request,
    payload: InvitationAcceptRequest,
) -> dict[str, Any]:
    _require_verified_email(request)
    user_id, email, display_name = _actor(request)
    try:
        member = _collaboration_service().accept_invitation(
            token=payload.token,
            user_id=user_id,
            email=email,
            display_name=display_name,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="invitation not found") from exc
    except (ValueError, WorkspaceConflict, WorkspaceAccessError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"membership": member.model_dump(mode="json")}


@app.delete("/api/workspace/invitations/{invitation_id}")
def revoke_workspace_invitation(request: Request, invitation_id: str) -> dict[str, Any]:
    actor_id, _, _ = _actor(request)
    active = _active_membership(request, "admin", required=True)
    try:
        invitation = _collaboration_service().revoke_invitation(
            organization_id=active.organization_id,
            actor_id=actor_id,
            invitation_id=invitation_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="invitation not found") from exc
    except (WorkspaceAccessError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"invitation": _public_invitation(invitation)}


@app.post("/api/workspace/members/{user_id}/role")
def change_workspace_member_role(
    request: Request,
    user_id: str,
    payload: MemberRoleRequest,
) -> dict[str, Any]:
    actor_id, _, _ = _actor(request)
    active = _active_membership(request, "admin", required=True)
    try:
        member = _collaboration_service().change_role(
            organization_id=active.organization_id,
            actor_id=actor_id,
            user_id=user_id,
            role=payload.role,
        )
    except (WorkspaceAccessError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"membership": member.model_dump(mode="json")}


@app.delete("/api/workspace/members/{user_id}")
def remove_workspace_member(request: Request, user_id: str) -> dict[str, str]:
    actor_id, _, _ = _actor(request)
    active = _active_membership(request, "admin", required=True)
    try:
        _collaboration_service().remove_member(
            organization_id=active.organization_id,
            actor_id=actor_id,
            user_id=user_id,
        )
    except (WorkspaceAccessError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"status": "removed"}


@app.post("/api/workspace/ownership/{user_id}")
def transfer_workspace_ownership(request: Request, user_id: str) -> dict[str, Any]:
    actor_id, _, actor_name = _actor(request)
    active = _active_membership(request, "owner", required=True)
    try:
        previous_owner, next_owner = _collaboration_service().transfer_ownership(
            organization_id=active.organization_id,
            actor_id=actor_id,
            user_id=user_id,
        )
    except (WorkspaceAccessError, WorkspaceConflict) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    _collaboration_service().add_activity(
        organization_id=active.organization_id,
        job_id="",
        actor_id=actor_id,
        actor_name=actor_name,
        kind="workspace.ownership_transferred",
        summary=f"Transferred workspace ownership to {next_owner.display_name}",
    )
    return {
        "previous_owner": previous_owner.model_dump(mode="json"),
        "next_owner": next_owner.model_dump(mode="json"),
    }


@app.post("/api/jobs", response_model=StudioJobPublic)
def create_job(
    request: Request,
    drawing: Annotated[UploadFile, File()],
    pixels_per_meter: Annotated[float, Form(gt=0)],
    scale_source: Annotated[
        Literal["user_supplied", "drawing_dimension", "vector_units"], Form()
    ] = "user_supplied",
    project_id: Annotated[str, Form(min_length=1, max_length=160)] = "dajoong-project",
    plan_instance_id: Annotated[str, Form(max_length=220)] = "",
    level_id: Annotated[str, Form(min_length=1, max_length=80)] = "L1",
    level_name: Annotated[str, Form(min_length=1, max_length=160)] = "Level 1",
    elevation_m: Annotated[float, Form()] = 0.0,
    nominal_height_m: Annotated[float, Form(gt=0)] = 3.0,
    wall_thickness_m: Annotated[float, Form(gt=0)] = 0.12,
    page_number: Annotated[int, Form(ge=1)] = 1,
    pdf_dpi: Annotated[int, Form(ge=72, le=600)] = 300,
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
    _require_conversion_runtime()
    runtime_threads, runtime_batch_size = _conversion_runtime()
    config = ConversionConfig(
        project_id=project_id,
        sheet_id=Path(source_name).stem,
        plan_instance_id=plan_instance_id,
        level_id=level_id,
        level_name=level_name,
        pixels_per_meter=pixels_per_meter,
        scale_source=scale_source,
        elevation_m=elevation_m,
        nominal_height_m=nominal_height_m,
        wall_thickness_m=wall_thickness_m,
        threads=runtime_threads,
        batch_size=runtime_batch_size,
        page_number=page_number,
        pdf_dpi=pdf_dpi,
    )
    owner_id, organization_id = _identity(request)
    if organization_id:
        _active_membership(request, "editor", required=True)
    idempotency_key = _idempotency_key(request)
    reservation = _reserve_conversion(request, 1, idempotency_key)
    try:
        if _using_aws():
            job = _aws_gateway().create_job(
                source_name,
                drawing.file,
                config,
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
            executor.submit(_run_conversion, job.id, source_path, config)
    except DurableJobEnqueueError as exc:
        raise HTTPException(
            status_code=503,
            detail="conversion is saved and queue delivery is being retried",
        ) from exc
    except Exception:
        _release_conversion(reservation)
        raise
    return StudioJobPublic.from_job(job)


@app.post("/api/building-jobs", response_model=StudioJobPublic)
def create_building_job(
    request: Request,
    drawing: Annotated[UploadFile, File()],
    building_config: Annotated[str, Form()],
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
    _require_conversion_runtime()
    try:
        if len(building_config.encode()) > 1_000_000:
            raise ValueError("building config exceeds 1 MB")
        payload = json.loads(building_config)
        levels = payload.get("levels", [])
        if not isinstance(levels, list) or not 1 <= len(levels) <= 200:
            raise ValueError("building config must contain between 1 and 200 levels")
        for level in payload.get("levels", []):
            level["source_path"] = source_name
        config = BuildingConversionConfig.model_validate(payload)
        runtime_threads, runtime_batch_size = _conversion_runtime()
        config = config.model_copy(
            update={"threads": runtime_threads, "batch_size": runtime_batch_size}
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid building config: {exc}") from exc
    owner_id, organization_id = _identity(request)
    if organization_id:
        _active_membership(request, "editor", required=True)
    idempotency_key = _idempotency_key(request)
    reservation = _reserve_conversion(request, len(config.levels), idempotency_key)
    try:
        if _using_aws():
            job = _aws_gateway().create_building_job(
                source_name,
                drawing.file,
                config,
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
            )
    except DurableJobEnqueueError as exc:
        raise HTTPException(
            status_code=503,
            detail="building conversion is saved and queue delivery is being retried",
        ) from exc
    except Exception:
        _release_conversion(reservation)
        raise
    return StudioJobPublic.from_job(job)


@app.post("/api/jobs/import", response_model=StudioJobPublic)
def import_graph(request: Request, payload: ImportedGraph) -> StudioJobPublic:
    owner_id, organization_id = _identity(request)
    if organization_id:
        _active_membership(request, "editor", required=True)
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


def _load_job(job_id: str) -> StudioJob:
    try:
        return _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


def _require_team_job(request: Request, job: StudioJob, role: WorkspaceRole) -> Any:
    _authorize(request, job, role)
    if not job.organization_id:
        raise HTTPException(
            status_code=409,
            detail="move this project into a company workspace to collaborate",
        )
    member = _active_membership(request, role, required=True)
    if member.organization_id != job.organization_id:
        raise HTTPException(status_code=404, detail="job not found")
    return member


@app.get("/api/jobs/{job_id}/comments")
def list_project_comments(request: Request, job_id: str, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    job = _load_job(job_id)
    _require_team_job(request, job, "viewer")
    return {
        "items": [
            item.model_dump(mode="json")
            for item in _collaboration_service().store.list_comments(job_id, limit)
        ]
    }


@app.post("/api/jobs/{job_id}/comments")
def create_project_comment(
    request: Request,
    job_id: str,
    payload: ProjectCommentCreateRequest,
) -> dict[str, Any]:
    job = _load_job(job_id)
    member = _require_team_job(request, job, "commenter")
    actor_id, _, actor_name = _actor(request)
    request_id = _idempotency_key(request)
    if not request_id:
        raise HTTPException(status_code=422, detail="Idempotency-Key is required for comments")
    try:
        comment = _collaboration_service().add_comment(
            organization_id=member.organization_id,
            job_id=job_id,
            actor_id=actor_id,
            actor_name=actor_name,
            body=payload.body,
            request_id=request_id,
            entity_ref=payload.entity_ref,
            assigned_to=payload.assigned_to,
        )
    except (ValueError, WorkspaceAccessError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _collaboration_service().add_activity(
        organization_id=member.organization_id,
        job_id=job_id,
        actor_id=actor_id,
        actor_name=actor_name,
        kind="comment.created",
        summary="Added a project comment",
        entity_ref=payload.entity_ref,
        request_id=f"comment:{request_id}",
        created_at=comment.created_at,
    )
    return {"comment": comment.model_dump(mode="json")}


@app.post("/api/jobs/{job_id}/comments/{comment_id}/status")
def update_project_comment_status(
    request: Request,
    job_id: str,
    comment_id: str,
    payload: ProjectCommentStatusRequest,
) -> dict[str, Any]:
    job = _load_job(job_id)
    member = _require_team_job(request, job, "commenter")
    actor_id, _, actor_name = _actor(request)
    try:
        comment = _collaboration_service().set_comment_status(
            organization_id=member.organization_id,
            job_id=job_id,
            comment_id=comment_id,
            actor_id=actor_id,
            status=payload.status,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="comment not found") from exc
    except WorkspaceAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    _collaboration_service().add_activity(
        organization_id=member.organization_id,
        job_id=job_id,
        actor_id=actor_id,
        actor_name=actor_name,
        kind=f"comment.{payload.status}",
        summary=f"Marked a comment {payload.status}",
        entity_ref=comment.entity_ref,
    )
    return {"comment": comment.model_dump(mode="json")}


@app.get("/api/jobs/{job_id}/activity")
def list_project_activity(request: Request, job_id: str, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    job = _load_job(job_id)
    _require_team_job(request, job, "viewer")
    return {
        "items": [
            item.model_dump(mode="json")
            for item in _collaboration_service().store.list_activity(job_id, limit)
        ]
    }


@app.get("/api/jobs/{job_id}/versions")
def list_model_versions(request: Request, job_id: str, limit: int = 100) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    job = _load_job(job_id)
    member = _require_team_job(request, job, "viewer")
    versions = _collaboration_service().store.list_versions(job_id, limit)
    if not versions and job.graph_sha256:
        baseline = ModelVersionRecord(
            organization_id=member.organization_id,
            job_id=job_id,
            version=job.version,
            graph_sha256=job.graph_sha256,
            created_by=job.owner_id,
            created_by_name="Dajoong conversion",
            label="Converted model",
            release_allowed=job.status == "complete",
            created_at=job.created_at,
        )
        _collaboration_service().store.record_version(baseline)
        versions = [baseline]
    return {"items": [item.model_dump(mode="json") for item in versions]}


@app.post("/api/workspace/presence")
def heartbeat_workspace_presence(
    request: Request,
    payload: PresenceRequest,
) -> dict[str, Any]:
    job = _load_job(payload.job_id)
    member = _require_team_job(request, job, "viewer")
    actor_id, _, actor_name = _actor(request)
    presence = _collaboration_service().heartbeat(
        organization_id=member.organization_id,
        job_id=job.id,
        user_id=actor_id,
        display_name=actor_name,
        active_entity=payload.active_entity,
    )
    return {
        "self": presence.model_dump(mode="json"),
        "active": [
            item.model_dump(mode="json")
            for item in _collaboration_service().store.list_presence(job.id, int(time.time()))
        ],
    }


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


def _commit_local_revision(
    job: StudioJob,
    *,
    graph: dict[str, Any],
    corrections: dict[str, Any],
    graph_sha256: str,
    release_allowed: bool,
) -> None:
    revision_prefix = f"revisions/{graph_sha256}"
    store.write_json(job.id, f"{revision_prefix}/corrected-plan-graph.json", graph)
    store.write_json(job.id, f"{revision_prefix}/corrections.json", corrections)
    store.write_json(job.id, "corrected-plan-graph.json", graph)
    store.write_json(job.id, "corrections.json", corrections)
    job.active_revision = graph_sha256
    job.graph_sha256 = graph_sha256
    job.status = "complete" if release_allowed else "review_required"
    store.save(job)


def _read_revision_graph(job: StudioJob, graph_sha256: str) -> dict[str, Any]:
    if len(graph_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in graph_sha256
    ):
        raise HTTPException(status_code=404, detail="model version not found")
    if _using_aws():
        try:
            return _aws_gateway().read_revision_graph(job, graph_sha256)
        except Exception:
            try:
                baseline = _aws_gateway().read_json(job, "graph")
            except Exception as exc:
                raise HTTPException(status_code=404, detail="model version not found") from exc
            if graph_content_hash(baseline) == graph_sha256:
                return baseline
            raise HTTPException(status_code=404, detail="model version not found") from None
    path = (
        DATA_ROOT.resolve()
        / job.id
        / "revisions"
        / graph_sha256
        / "corrected-plan-graph.json"
    ).resolve()
    if DATA_ROOT.resolve() in path.parents and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        baseline_path = _artifact_path(job, "graph")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (HTTPException, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="model version not found") from exc
    if graph_content_hash(baseline) == graph_sha256:
        return baseline
    raise HTTPException(status_code=404, detail="model version not found")


@app.get("/api/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(
    request: Request,
    job_id: str,
    artifact_name: str,
    level_id: str = "",
    delivery: Literal["download", "lazy"] = "download",
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

            if delivery == "lazy" and artifact_name in {"graph", "corrected-graph"}:
                try:
                    graph = json.loads(body.read().decode("utf-8"))
                finally:
                    body.close()
                return JSONResponse(externalize_graph_assets(graph))

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
    if delivery == "lazy" and artifact_name in {"graph", "corrected-graph"}:
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="invalid plan graph artifact") from exc
        return JSONResponse(externalize_graph_assets(graph))
    return FileResponse(path, filename=path.name)


@app.post("/api/jobs/{job_id}/corrections", response_model=PatchResult)
def patch_graph(request: Request, job_id: str, payload: GraphCorrectionSet) -> PatchResult:
    try:
        job = _aws_gateway().get(job_id) if _using_aws() else store.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    _authorize(request, job, "editor")
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
        _commit_local_revision(
            job,
            graph=corrected,
            corrections=payload.model_dump(mode="json"),
            graph_sha256=graph_sha256,
            release_allowed=release_allowed,
        )
        corrected_name = "corrected-plan-graph.json"
    if job.organization_id:
        actor_id, _, actor_name = _actor(request)
        summary = correction_summary(payload.operations)
        _collaboration_service().store.record_version(
            ModelVersionRecord(
                organization_id=job.organization_id,
                job_id=job.id,
                version=job.version,
                graph_sha256=graph_sha256,
                created_by=actor_id,
                created_by_name=actor_name,
                label="Reviewed model",
                summary=summary,
                release_allowed=release_allowed,
                created_at=int(time.time()),
            )
        )
        _collaboration_service().add_activity(
            organization_id=job.organization_id,
            job_id=job.id,
            actor_id=actor_id,
            actor_name=actor_name,
            kind="model.reviewed",
            summary="Applied reviewed model corrections",
        )
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
    _authorize(request, job, "editor")
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
        _commit_local_revision(
            job,
            graph=payload.graph,
            corrections=corrections,
            graph_sha256=graph_sha256,
            release_allowed=certificate.release_allowed,
        )
    if job.organization_id:
        actor_id, _, actor_name = _actor(request)
        summary = correction_summary(payload.operations)
        _collaboration_service().store.record_version(
            ModelVersionRecord(
                organization_id=job.organization_id,
                job_id=job.id,
                version=job.version,
                graph_sha256=graph_sha256,
                created_by=actor_id,
                created_by_name=actor_name,
                label="Autosaved model",
                summary=summary,
                release_allowed=certificate.release_allowed,
                created_at=int(time.time()),
            )
        )
        _collaboration_service().add_activity(
            organization_id=job.organization_id,
            job_id=job.id,
            actor_id=actor_id,
            actor_name=actor_name,
            kind="model.saved",
            summary="Saved a new model version",
        )
    return PatchResult(
        graph_sha256=graph_sha256,
        artifact_name="corrected-plan-graph.json",
        summary=correction_summary(payload.operations),
        review_required=certificate.review_required,
        release_allowed=certificate.release_allowed,
        job_version=job.version,
    )


@app.post("/api/jobs/{job_id}/versions/{graph_sha256}/restore", response_model=PatchResult)
def restore_model_version(
    request: Request,
    job_id: str,
    graph_sha256: str,
    payload: VersionRestoreRequest,
) -> PatchResult:
    job = _load_job(job_id)
    member = _require_team_job(request, job, "editor")
    if job.version != payload.expected_job_version:
        raise HTTPException(
            status_code=409,
            detail="project changed in another session; reload before restoring",
        )
    known = {
        item.graph_sha256
        for item in _collaboration_service().store.list_versions(job_id, 200)
    }
    if graph_sha256 not in known:
        raise HTTPException(status_code=404, detail="model version not found")
    restored = _read_revision_graph(job, graph_sha256)
    certificate = PlanGraphVerifier().verify(restored)
    actor_id, _, actor_name = _actor(request)
    corrections = {
        "schema_version": "dajoong.studio-version-restore.v1",
        "reviewer": actor_name,
        "restored_graph_sha256": graph_sha256,
        "operations": [],
        "verification": certificate.model_dump(mode="json"),
    }
    if _using_aws():
        try:
            _aws_gateway().commit_correction_revision(
                job,
                corrected_graph=restored,
                corrections=corrections,
                graph_sha256=graph_sha256,
                release_allowed=certificate.release_allowed,
            )
        except JobVersionConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="project changed in another session; reload before restoring",
            ) from exc
    else:
        _commit_local_revision(
            job,
            graph=restored,
            corrections=corrections,
            graph_sha256=graph_sha256,
            release_allowed=certificate.release_allowed,
        )
    _collaboration_service().store.record_version(
        ModelVersionRecord(
            organization_id=member.organization_id,
            job_id=job.id,
            version=job.version,
            graph_sha256=graph_sha256,
            created_by=actor_id,
            created_by_name=actor_name,
            label="Restored model",
            release_allowed=certificate.release_allowed,
            created_at=int(time.time()),
        )
    )
    _collaboration_service().add_activity(
        organization_id=member.organization_id,
        job_id=job.id,
        actor_id=actor_id,
        actor_name=actor_name,
        kind="model.restored",
        summary="Restored an earlier model version",
    )
    return PatchResult(
        graph_sha256=graph_sha256,
        artifact_name="corrected-plan-graph.json",
        summary={},
        review_required=certificate.review_required,
        release_allowed=certificate.release_allowed,
        job_version=job.version,
    )
