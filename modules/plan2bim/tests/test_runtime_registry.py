from __future__ import annotations

import json
from pathlib import Path

import pytest

from buili_plan2bim.core.hashing import sha256_file
from buili_plan2bim.core.model.cad_evidence import GLOBAL_PROGRAM_INPUT_CONTRACT
from buili_plan2bim.core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
)
from buili_plan2bim.core.model.local_element_student import (
    ELEMENT_CLASS_FAMILY_INDICES,
    ELEMENT_FAMILY_CLASSES,
    ELEMENT_FAMILY_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_FEATURES,
)
from buili_plan2bim.local_element_crops import (
    CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LOCAL_ELEMENT_INPUT_CHANNELS,
)
from buili_plan2bim.runtime_registry import (
    active_semantic_max_side,
    load_active_runtime,
    require_product_architectural_runtime,
    resolve_active_semantic_model,
    resolve_architectural_runtime,
    verify_active_runtime,
)


def _method_contract() -> dict[str, object]:
    return {
        "legacy_runtime_policy": "deny",
        "reconstruction_method": {
            "version": "dajoong-forest-reconstruction-v2",
            "primary_representation": "global_spatial_evidence_graph",
        },
    }


def _write_architectural_pair(
    root: Path,
    *,
    local_classes: list[str] | None = None,
    production_authorized: bool = False,
) -> tuple[Path, Path]:
    global_model = root / "global.onnx"
    local_model = root / "local.onnx"
    global_model.write_bytes(b"global")
    local_model.write_bytes(b"local")
    global_model.with_suffix(".onnx.json").write_text(
        json.dumps(
            {
                "schema_version": "dajoong.global-program-onnx.v1",
                "artifact_sha256": sha256_file(global_model),
                "input_contract": GLOBAL_PROGRAM_INPUT_CONTRACT,
                "topology_channels": list(TOPOLOGY_TARGET_CHANNELS),
                "room_classes": list(ROOM_PROGRAM_CLASSES),
                "element_classes": list(ELEMENT_PROGRAM_CLASSES),
                "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
                "production_authorized": production_authorized,
            }
        ),
        encoding="utf-8",
    )
    local_model.with_suffix(".onnx.json").write_text(
        json.dumps(
            {
                "schema_version": "dajoong.local-element-onnx.v1",
                "artifact_sha256": sha256_file(local_model),
                "model_version": (
                    "dajoong-local-element-student-v14-equipment-run-"
                    "relational-hierarchy"
                ),
                "oriented_evidence_rotation_contract": (
                    "c4_spatial_rotate_swap_axis_channels_on_odd_quadrants_v1"
                ),
                "candidate_alignment_contract": (
                    "mutual_coverage_072_iou_055_or_truth_v1"
                ),
                "candidate_hypothesis_context_contract": (
                    CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
                ),
                "input_contract": LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                "local_view_contract": "native_detail_assembly_room_v1",
                "input_names": [
                    "element_crop_evidence",
                    "whole_sheet_evidence",
                    "candidate_context",
                ],
                "input_channels": LOCAL_ELEMENT_INPUT_CHANNELS,
                "whole_sheet_input_channels": 4,
                "whole_sheet_contract": "explicit_complete_plan_evidence_v1",
                "candidate_context_features": LOCAL_ELEMENT_CONTEXT_FEATURES,
                "candidate_context_contract": LOCAL_ELEMENT_CONTEXT_CONTRACT,
                "classes": local_classes or list(ELEMENT_PROGRAM_CLASSES),
                "geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
                "output_names": [
                    "class_logits",
                    "family_logits",
                    "objectness",
                    "geometry",
                    "uncertainty",
                ],
                "objectness_contract": "binary_object_existence_before_conditional_taxonomy_v1",
                "class_semantics": "foreground_taxonomy_conditional_on_objectness_v1",
                "family_classes": list(ELEMENT_FAMILY_CLASSES),
                "class_family_indices": list(ELEMENT_CLASS_FAMILY_INDICES),
                "family_contract": ELEMENT_FAMILY_CONTRACT,
                "production_authorized": production_authorized,
            }
        ),
        encoding="utf-8",
    )
    return global_model, local_model


def test_checked_in_registry_selects_the_default_runtime() -> None:
    runtime = load_active_runtime()

    assert runtime.primary_model_path.name == "aec-global-enclosure-v1.onnx"
    assert runtime.primary_manifest_path.name == "aec-global-enclosure-v1.onnx.json"
    assert runtime.qualification_path.name == "aec-global-enclosure-v1.qualification.json"
    assert runtime.production_authorized is False
    assert runtime.payload["legacy_runtime_policy"] == "deny"
    assert active_semantic_max_side(runtime) == 2048
    verify_active_runtime(runtime)


def test_registry_rejects_an_unpinned_or_substituted_model(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not-the-pinned-model")
    (tmp_path / "model.onnx.json").write_text("{}", encoding="utf-8")
    (tmp_path / "qualification.json").write_text("{}", encoding="utf-8")
    registry = tmp_path / "ACTIVE_RUNTIME.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "dajoong.active-runtime.v1",
                **_method_contract(),
                "active_primary_model": {
                    "path": "model.onnx",
                    "manifest_path": "model.onnx.json",
                    "qualification_path": "qualification.json",
                    "sha256": "0" * 64,
                    "production_authorized": False,
                },
            }
        ),
        encoding="utf-8",
    )

    runtime = load_active_runtime(registry)
    with pytest.raises(ValueError, match="does not match ACTIVE_RUNTIME.json"):
        verify_active_runtime(runtime)


def test_registry_resolves_and_verifies_environment_semantic_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "model.onnx"
    primary.write_bytes(b"primary")
    (tmp_path / "model.onnx.json").write_text("{}", encoding="utf-8")
    (tmp_path / "qualification.json").write_text("{}", encoding="utf-8")
    semantic = tmp_path / "semantic.onnx"
    semantic.write_bytes(b"semantic")

    import hashlib

    registry = tmp_path / "ACTIVE_RUNTIME.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "dajoong.active-runtime.v1",
                **_method_contract(),
                "active_primary_model": {
                    "path": primary.name,
                    "manifest_path": "model.onnx.json",
                    "qualification_path": "qualification.json",
                    "sha256": hashlib.sha256(b"primary").hexdigest(),
                },
                "semantic_runtime": {
                    "path_env": "TEST_DAJOONG_SEMANTIC_MODEL_PATH",
                    "sha256": hashlib.sha256(b"semantic").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DAJOONG_SEMANTIC_MODEL_PATH", str(semantic))

    runtime = load_active_runtime(registry)
    assert resolve_active_semantic_model(runtime) == semantic.resolve()


def test_registry_rejects_environment_semantic_model_with_wrong_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "model.onnx"
    primary.write_bytes(b"primary")
    (tmp_path / "model.onnx.json").write_text("{}", encoding="utf-8")
    (tmp_path / "qualification.json").write_text("{}", encoding="utf-8")
    semantic = tmp_path / "semantic.onnx"
    semantic.write_bytes(b"wrong")

    import hashlib

    registry = tmp_path / "ACTIVE_RUNTIME.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "dajoong.active-runtime.v1",
                **_method_contract(),
                "active_primary_model": {
                    "path": primary.name,
                    "manifest_path": "model.onnx.json",
                    "qualification_path": "qualification.json",
                    "sha256": hashlib.sha256(b"primary").hexdigest(),
                },
                "semantic_runtime": {
                    "path_env": "TEST_DAJOONG_SEMANTIC_MODEL_PATH",
                    "sha256": hashlib.sha256(b"expected").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DAJOONG_SEMANTIC_MODEL_PATH", str(semantic))

    runtime = load_active_runtime(registry)
    with pytest.raises(ValueError, match="active semantic model does not match"):
        resolve_active_semantic_model(runtime)


def test_registry_rejects_legacy_reconstruction_method(tmp_path: Path) -> None:
    registry = tmp_path / "ACTIVE_RUNTIME.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "dajoong.active-runtime.v1",
                "legacy_runtime_policy": "deny",
                "reconstruction_method": {
                    "version": "local-pixel-promotion-v1",
                    "primary_representation": "isolated_pixel_classes",
                },
                "active_primary_model": {
                    "path": "model.onnx",
                    "manifest_path": "model.onnx.json",
                    "qualification_path": "qualification.json",
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy or unknown reconstruction method"):
        load_active_runtime(registry)


def test_architectural_runtime_requires_global_and_local_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_model = tmp_path / "global.onnx"
    global_model.write_bytes(b"global")
    global_model.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH", str(global_model))
    monkeypatch.delenv("DAJOONG_LOCAL_ELEMENT_MODEL_PATH", raising=False)

    with pytest.raises(ValueError, match="must be configured together"):
        resolve_architectural_runtime(load_active_runtime())


def test_architectural_runtime_prefers_complete_global_local_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_model, local_model = _write_architectural_pair(tmp_path)
    monkeypatch.setenv("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH", str(global_model))
    monkeypatch.setenv("DAJOONG_LOCAL_ELEMENT_MODEL_PATH", str(local_model))

    resolved = resolve_architectural_runtime(load_active_runtime())

    assert resolved.mode == "global_program_with_native_refiner"
    assert resolved.global_program_model_path == global_model.resolve()
    assert resolved.local_element_model_path == local_model.resolve()
    assert resolved.legacy_semantic_model_path is None


def test_architectural_runtime_rejects_stale_local_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_model, local_model = _write_architectural_pair(
        tmp_path,
        local_classes=["background", "door", "unknown"],
    )
    monkeypatch.setenv("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH", str(global_model))
    monkeypatch.setenv("DAJOONG_LOCAL_ELEMENT_MODEL_PATH", str(local_model))

    with pytest.raises(ValueError, match="local element class contract mismatch"):
        resolve_architectural_runtime(load_active_runtime())


def test_architectural_runtime_never_enables_legacy_teacher_implicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH", raising=False)
    monkeypatch.delenv("DAJOONG_LOCAL_ELEMENT_MODEL_PATH", raising=False)
    monkeypatch.delenv("DAJOONG_ALLOW_LEGACY_SEMANTIC_TEACHER", raising=False)

    resolved = resolve_architectural_runtime(load_active_runtime())

    assert resolved.mode == "architectural_runtime_missing"
    assert resolved.legacy_semantic_model_path is None


def test_product_architectural_runtime_rejects_missing_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DAJOONG_GLOBAL_PROGRAM_MODEL_PATH", raising=False)
    monkeypatch.delenv("DAJOONG_LOCAL_ELEMENT_MODEL_PATH", raising=False)

    with pytest.raises(RuntimeError, match="production architectural runtime is unavailable"):
        require_product_architectural_runtime()


def test_product_architectural_runtime_requires_authorized_pair(tmp_path: Path) -> None:
    global_model, local_model = _write_architectural_pair(tmp_path)

    with pytest.raises(PermissionError, match="not authorized for production"):
        require_product_architectural_runtime(global_model, local_model)


def test_product_architectural_runtime_accepts_authorized_pair(tmp_path: Path) -> None:
    global_model, local_model = _write_architectural_pair(
        tmp_path,
        production_authorized=True,
    )

    resolved = require_product_architectural_runtime(global_model, local_model)

    assert resolved == (global_model.resolve(), local_model.resolve())
