from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = REPOSITORY_ROOT / "scripts" / "install_private_models.py"
SPEC = importlib.util.spec_from_file_location("install_private_models", INSTALLER_PATH)
assert SPEC is not None and SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


def test_private_bundle_cannot_replace_active_runtime_registry() -> None:
    with pytest.raises(ValueError, match="may not replace runtime registry"):
        INSTALLER.normalized_member("models/ACTIVE_RUNTIME.json")


def test_private_bundle_accepts_model_and_content_manifest() -> None:
    assert INSTALLER.normalized_member("models/model.onnx").name == "model.onnx"
    assert INSTALLER.normalized_member("models/model.onnx.json").name == "model.onnx.json"
