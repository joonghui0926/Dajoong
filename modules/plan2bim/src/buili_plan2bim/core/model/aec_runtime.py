"""Content-addressed CPU runtime for the proof-carrying AEC specialist path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..hashing import sha256_file, sha256_json
from .aec_decode import AecTileProposal, decode_aec_tile
from .aec_onnx import OnnxAecSpecialist
from .cad_evidence import GLOBAL_ORIENTED_EVIDENCE_CONTRACT
from .evidence_coverage import CoverageConfig
from .evidence_pyramid import (
    EvidencePyramidConfig,
    EvidenceTile,
    FullSheetAecResult,
    run_information_preserving_aec,
)
from .fourier_wall import detect_fourier_wall_prior


class ProofCarryingAecRunner:
    """Run one immutable local ONNX model behind exhaustive evidence routing.

    The ONNX manifest is part of the release decision.  A development artifact with
    ``authoritative_decisions=false`` can be evaluated, but its result is always
    ``release_allowed=false`` even if geometry reprojects perfectly.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        manifest_path: Path | None = None,
        expected_sha256: str = "",
        threads: int = 1,
        batch_size: int = 8,
        pyramid_config: EvidencePyramidConfig | None = None,
        coverage_config: CoverageConfig | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_path = model_path.resolve()
        self.manifest_path = (
            manifest_path.resolve()
            if manifest_path is not None
            else self.model_path.with_suffix(self.model_path.suffix + ".json")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.model_sha256 = sha256_file(self.model_path)
        manifest_sha256 = str(self.manifest.get("onnx_sha256") or "")
        if manifest_sha256 != self.model_sha256:
            raise ValueError("ONNX artifact does not match its content-addressed manifest")
        if expected_sha256 and expected_sha256 != self.model_sha256:
            raise ValueError("ONNX artifact does not match the configured production hash")
        if self.manifest.get("input_contract") != [
            "raster_ink",
            "horizontal_line_support",
            "vertical_line_support",
            "global_enclosure",
        ]:
            raise ValueError("AEC model does not use the required full-sheet evidence contract")
        config = dict(self.manifest.get("config") or {})
        if config.get("evidence_contract") != GLOBAL_ORIENTED_EVIDENCE_CONTRACT:
            raise ValueError("AEC model evidence contract is incompatible with the runtime")
        self.model_version = str(self.manifest.get("model_version") or "")
        if not self.model_version:
            raise ValueError("AEC model manifest has no model_version")
        self.release_authorized = bool(self.manifest.get("authoritative_decisions", False))
        self.symbol_classes = tuple(config.get("symbol_classes") or ())
        self.pyramid_config = pyramid_config or EvidencePyramidConfig()
        self.coverage_config = coverage_config or CoverageConfig()
        self.batch_size = batch_size
        self.runtime = OnnxAecSpecialist(self.model_path, threads=threads)

    def model_card(self) -> dict[str, Any]:
        card = {
            "schema_version": "dajoong.proof-carrying-aec-runtime.v1",
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "manifest_sha256": sha256_file(self.manifest_path),
            "parameters": int(self.manifest.get("parameters") or 0),
            "evidence_contract": GLOBAL_ORIENTED_EVIDENCE_CONTRACT,
            "release_authorized": self.release_authorized,
            "pyramid_config": self.pyramid_config.model_dump(mode="json"),
            "coverage_config": self.coverage_config.model_dump(mode="json"),
            "batch_size": self.batch_size,
        }
        card["content_sha256"] = sha256_json(card)
        return card

    def _infer_tile(self, evidence: np.ndarray, tile: EvidenceTile) -> AecTileProposal:
        return self._infer_batch(evidence[None, ...], [tile])[0]

    def _infer_batch(
        self,
        evidence: np.ndarray,
        tiles: list[EvidenceTile],
    ) -> list[AecTileProposal]:
        if evidence.shape[0] != len(tiles):
            raise ValueError("evidence batch and tile metadata differ")
        output = self.runtime.infer(evidence)
        proposals = []
        for index, tile in enumerate(tiles):
            prior = detect_fourier_wall_prior(evidence[index, 0])
            proposals.append(
                decode_aec_tile(
                    tile_id=tile.tile_id,
                    source_ref_ids=tile.source_ref_ids,
                    model_version=self.model_version,
                    structure_logits=output["structure_logits"][index],
                    symbol_logits=output["symbol_logits"][index],
                    metric_offsets=output["metric_offsets"][index],
                    uncertainty=output["uncertainty"][index],
                    fourier_prior=prior,
                    # Strict pruning remains disabled until held-out wall-family recall is
                    # high enough; the prior currently helps auditing, not removal.
                    strict_fourier_candidate_pruning=False,
                )
            )
        return proposals

    def compile_image(
        self,
        image: Image.Image,
        *,
        sheet_id: str,
        source_ref_ids: list[str],
        active_mask: np.ndarray | None = None,
        known_text_mask: np.ndarray | None = None,
        known_dimension_mask: np.ndarray | None = None,
        known_hatch_mask: np.ndarray | None = None,
    ) -> FullSheetAecResult:
        return run_information_preserving_aec(
            image,
            sheet_id=sheet_id,
            source_ref_ids=source_ref_ids,
            infer_batch=self._infer_batch,
            inference_batch_size=self.batch_size,
            pyramid_config=self.pyramid_config,
            coverage_config=self.coverage_config,
            active_mask=active_mask,
            known_text_mask=known_text_mask,
            known_dimension_mask=known_dimension_mask,
            known_hatch_mask=known_hatch_mask,
            model_artifact_sha256=self.model_sha256,
            model_release_authorized=self.release_authorized,
        )

    def compile_path(self, image_path: Path, *, sheet_id: str) -> FullSheetAecResult:
        source_hash = sha256_file(image_path)
        with Image.open(image_path) as image:
            return self.compile_image(
                image.convert("L"),
                sheet_id=sheet_id,
                source_ref_ids=[source_hash],
            )
