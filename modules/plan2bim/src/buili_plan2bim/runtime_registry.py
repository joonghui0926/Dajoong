"""Single source of truth for the active Plan2BIM runtime artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core.hashing import sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = PACKAGE_ROOT / "models" / "ACTIVE_RUNTIME.json"
ACTIVE_RECONSTRUCTION_METHOD = "dajoong-forest-reconstruction-v2"
ACTIVE_PRIMARY_REPRESENTATION = "global_spatial_evidence_graph"


@dataclass(frozen=True)
class ActiveRuntime:
    registry_path: Path
    primary_model_path: Path
    primary_manifest_path: Path
    qualification_path: Path
    expected_primary_sha256: str
    production_authorized: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class ArchitecturalRuntimePaths:
    """Immutable server-side paths for the production architectural pair.

    The whole-sheet program and the native-resolution element refiner are one
    product contract.  Accepting only one of them would silently re-introduce the
    recall ceiling that the local refiner is meant to remove.
    """

    global_program_model_path: Path | None
    local_element_model_path: Path | None
    legacy_semantic_model_path: Path | None
    mode: str


def load_active_runtime(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> ActiveRuntime:
    """Load and validate the only runtime selection accepted by default."""

    path = Path(registry_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"active runtime registry is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dajoong.active-runtime.v1":
        raise ValueError("unsupported active runtime registry schema")
    reconstruction = payload.get("reconstruction_method")
    if not isinstance(reconstruction, dict):
        raise ValueError("active runtime registry has no reconstruction method")
    if reconstruction.get("version") != ACTIVE_RECONSTRUCTION_METHOD:
        raise ValueError("legacy or unknown reconstruction method is not executable")
    if reconstruction.get("primary_representation") != ACTIVE_PRIMARY_REPRESENTATION:
        raise ValueError("active runtime must use the global spatial evidence graph")
    if payload.get("legacy_runtime_policy") != "deny":
        raise ValueError("active runtime must deny legacy execution paths")
    primary = payload.get("active_primary_model")
    if not isinstance(primary, dict):
        raise ValueError("active runtime registry has no primary model")

    required = ("path", "manifest_path", "qualification_path", "sha256")
    missing = [key for key in required if not str(primary.get(key) or "").strip()]
    if missing:
        raise ValueError(f"active runtime registry is missing: {', '.join(missing)}")
    expected_hash = str(primary["sha256"]).lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("active primary model sha256 is invalid")

    base = path.parent
    return ActiveRuntime(
        registry_path=path,
        primary_model_path=(base / str(primary["path"])).resolve(),
        primary_manifest_path=(base / str(primary["manifest_path"])).resolve(),
        qualification_path=(base / str(primary["qualification_path"])).resolve(),
        expected_primary_sha256=expected_hash,
        production_authorized=bool(primary.get("production_authorized", False)),
        payload=payload,
    )


def verify_active_runtime(runtime: ActiveRuntime) -> None:
    """Fail closed when active artifacts are absent or differ from the registry."""

    for artifact in (
        runtime.primary_model_path,
        runtime.primary_manifest_path,
        runtime.qualification_path,
    ):
        if not artifact.is_file():
            raise FileNotFoundError(f"active runtime artifact is missing: {artifact}")
    actual_hash = sha256_file(runtime.primary_model_path)
    if actual_hash != runtime.expected_primary_sha256:
        raise ValueError("active primary model does not match ACTIVE_RUNTIME.json")


def resolve_active_semantic_model(runtime: ActiveRuntime) -> Path | None:
    """Resolve the pinned semantic artifact without silently accepting a substitute.

    The private semantic model is not checked into the repository. Production workers
    inject it through the registry-declared environment variable; a packaged install may
    place it at the registry-declared installed path. When neither exists, the lightweight
    primary-only path remains available for smoke tests, but it is never mistaken for the
    full semantic pipeline because conversion manifests record an empty semantic model.
    """

    semantic = runtime.payload.get("semantic_runtime")
    if not isinstance(semantic, dict):
        return None
    expected_hash = str(semantic.get("sha256") or "").strip().lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("active semantic model sha256 is invalid")

    environment_name = str(semantic.get("path_env") or "").strip()
    configured_path = os.environ.get(environment_name, "").strip() if environment_name else ""
    if configured_path:
        model_path = Path(configured_path).expanduser().resolve()
    else:
        installed_name = str(semantic.get("installed_path") or "").strip()
        model_path = (
            (runtime.registry_path.parent / installed_name).resolve()
            if installed_name
            else None
        )
        if model_path is None or not model_path.is_file():
            return None

    if not model_path.is_file():
        raise FileNotFoundError(f"active semantic runtime artifact is missing: {model_path}")
    if sha256_file(model_path) != expected_hash:
        raise ValueError("active semantic model does not match ACTIVE_RUNTIME.json")
    return model_path


def _resolve_configured_model(environment_name: str) -> Path | None:
    configured = os.environ.get(environment_name, "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"configured runtime artifact is missing ({environment_name}): {path}"
        )
    if not path.with_suffix(path.suffix + ".json").is_file():
        raise FileNotFoundError(
            f"configured runtime manifest is missing ({environment_name}): "
            f"{path.with_suffix(path.suffix + '.json')}"
        )
    return path


def _validate_architectural_runtime_pair(
    global_model: Path,
    local_model: Path,
    *,
    require_production: bool = False,
) -> None:
    """Reject mixed-generation model pairs before any conversion begins."""

    # Imports stay local so the registry remains cheap for non-architectural CLI use.
    from .global_program_inference import validate_global_program_manifest
    from .local_element_inference import validate_local_element_manifest

    global_manifest_path = global_model.with_suffix(global_model.suffix + ".json")
    local_manifest_path = local_model.with_suffix(local_model.suffix + ".json")
    global_manifest = json.loads(global_manifest_path.read_text(encoding="utf-8"))
    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    validate_global_program_manifest(
        global_manifest,
        global_model,
        require_production=require_production,
    )
    validate_local_element_manifest(
        local_manifest,
        local_model,
        require_production=require_production,
    )
    global_classes = tuple(str(value) for value in global_manifest.get("element_classes") or ())
    local_classes = tuple(str(value) for value in local_manifest.get("classes") or ())
    if global_classes != local_classes:
        raise ValueError("global and local architectural class contracts do not match")


def require_product_architectural_runtime(
    global_program_model_path: str | Path | None = None,
    local_element_model_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return one complete, production-authorized architectural runtime pair.

    Product entry points call this before reserving credit or persisting an
    upload.  The building-system primary model cannot stand in for walls,
    openings, rooms, or editable objects, so a missing architectural pair is a
    service-readiness failure rather than a partially successful conversion.
    """

    if (global_program_model_path is None) != (local_element_model_path is None):
        raise RuntimeError(
            "global-program and local-element model artifacts must be configured together"
        )
    if global_program_model_path is None:
        runtime = load_active_runtime()
        resolved = resolve_architectural_runtime(runtime)
        global_model = resolved.global_program_model_path
        local_model = resolved.local_element_model_path
    else:
        global_model = Path(global_program_model_path).expanduser().resolve()
        local_model = Path(local_element_model_path).expanduser().resolve()  # type: ignore[arg-type]
    if global_model is None or local_model is None:
        raise RuntimeError(
            "the production architectural runtime is unavailable; both the whole-sheet "
            "program and native-resolution element refiner are required"
        )
    _validate_architectural_runtime_pair(
        global_model,
        local_model,
        require_production=True,
    )
    return global_model, local_model


def resolve_architectural_runtime(
    runtime: ActiveRuntime,
    *,
    allow_legacy_semantic_teacher: bool = False,
) -> ArchitecturalRuntimePaths:
    """Resolve one unambiguous architectural execution route.

    Product workers must provide both lightweight architectural artifacts.  The
    non-commercial semantic teacher remains available only when the caller
    explicitly opts into a sealed research/evaluation run.
    """

    global_model = _resolve_configured_model("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH")
    local_model = _resolve_configured_model("DAJOONG_LOCAL_ELEMENT_MODEL_PATH")
    if (global_model is None) != (local_model is None):
        raise ValueError(
            "DAJOONG_GLOBAL_PROGRAM_MODEL_PATH and "
            "DAJOONG_LOCAL_ELEMENT_MODEL_PATH must be configured together"
        )
    if global_model is not None and local_model is not None:
        _validate_architectural_runtime_pair(global_model, local_model)
        return ArchitecturalRuntimePaths(
            global_program_model_path=global_model,
            local_element_model_path=local_model,
            legacy_semantic_model_path=None,
            mode="global_program_with_native_refiner",
        )

    legacy_allowed = allow_legacy_semantic_teacher or os.environ.get(
        "DAJOONG_ALLOW_LEGACY_SEMANTIC_TEACHER", ""
    ).strip().lower() in {"1", "true", "yes"}
    if legacy_allowed:
        legacy = resolve_active_semantic_model(runtime)
        if legacy is None:
            raise FileNotFoundError(
                "legacy semantic evaluation was requested but no pinned artifact is installed"
            )
        return ArchitecturalRuntimePaths(
            global_program_model_path=None,
            local_element_model_path=None,
            legacy_semantic_model_path=legacy,
            mode="sealed_legacy_semantic_evaluation",
        )

    return ArchitecturalRuntimePaths(
        global_program_model_path=None,
        local_element_model_path=None,
        legacy_semantic_model_path=None,
        mode="architectural_runtime_missing",
    )


def active_semantic_max_side(runtime: ActiveRuntime) -> int:
    """Return the registry-pinned full-sheet inference resolution."""

    semantic = runtime.payload.get("semantic_runtime")
    if not isinstance(semantic, dict):
        raise ValueError("active runtime registry has no semantic runtime")
    value = int(semantic.get("max_side") or 0)
    if value < 64:
        raise ValueError("active semantic max_side must be at least 64")
    return value
