"""Batched ONNX refinement of global element proposals at native detail."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .core.hashing import sha256_file
from .core.model.aec_decode import PixelSymbolProposal
from .core.model.cad_evidence import build_cad_evidence, letterbox_cad_evidence
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
)
from .core.model.local_element_student import (
    ELEMENT_CLASS_FAMILY_INDICES,
    ELEMENT_FAMILY_CLASSES,
    ELEMENT_FAMILY_CONTRACT,
    LEGACY_ELEMENT_FAMILY_CONTRACT,
    LEGACY_LETTERBOX_LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LEGACY_LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LEGACY_LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_CONTRACT,
    LOCAL_ELEMENT_CONTEXT_FEATURES,
    LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT,
)
from .element_set_decoder import decode_element_set
from .local_element_candidates import (
    NativeElementCandidateDiagnostics,
    mine_native_element_candidates,
)
from .local_element_crops import (
    CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
    LEGACY_CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT,
    LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LEGACY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS,
    LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    LOCAL_ELEMENT_INPUT_CHANNELS,
    RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    WHOLE_SHEET_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    candidate_hypothesis_contexts,
    decode_element_geometry,
    extract_local_element_hierarchy_batch_from_map,
    extract_local_element_pyramid_batch_from_map,
    legacy_candidate_hypothesis_contexts,
    normalized_candidate_context,
    semantic_element_context,
)


class UnresolvedNativeElementCandidate(BaseModel):
    """Native ink found by the candidate miner but not safe to compile."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    bbox_px: tuple[float, float, float, float]
    proposed_class: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class LocalElementRefinementDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "dajoong.local-element-refinement.v2"
    proposal_count: int = Field(ge=0)
    reclassified_count: int = Field(ge=0)
    geometry_refined_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    invalid_proposal_count: int = Field(default=0, ge=0)
    discovered_candidate_count: int = Field(default=0, ge=0)
    accepted_discovered_count: int = Field(default=0, ge=0)
    semantic_window_candidate_count: int = Field(default=0, ge=0)
    accepted_semantic_window_count: int = Field(default=0, ge=0)
    unresolved_discovered: list[UnresolvedNativeElementCandidate] = Field(
        default_factory=list
    )
    native_candidate_diagnostics: NativeElementCandidateDiagnostics | None = None
    model_version: str
    model_sha256: str
    timings_ms: dict[str, float] = Field(default_factory=dict)


def validate_local_element_manifest(
    payload: dict[str, Any],
    artifact_path: str | Path,
    *,
    require_production: bool,
) -> None:
    artifact = Path(artifact_path).expanduser().resolve()
    if payload.get("schema_version") != "dajoong.local-element-onnx.v1":
        raise ValueError("unsupported local element ONNX manifest")
    if payload.get("artifact_sha256") != sha256_file(artifact):
        raise ValueError("local element ONNX artifact hash mismatch")
    model_version = str(payload.get("model_version") or "")
    if model_version.startswith((
        "dajoong-local-element-student-v10-",
        "dajoong-local-element-student-v11-",
        "dajoong-local-element-student-v12-",
        "dajoong-local-element-student-v13-",
        "dajoong-local-element-student-v14-",
        "dajoong-local-element-student-v15-",
        "dajoong-local-element-student-v16-",
    )):
        from .core.model.cad_evidence import ORIENTED_EVIDENCE_ROTATION_CONTRACT
        from .local_element_training import CANDIDATE_ALIGNMENT_CONTRACT

        if payload.get("oriented_evidence_rotation_contract") != (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ):
            raise ValueError(
                "local element oriented-evidence rotation contract mismatch"
            )
        if payload.get("candidate_alignment_contract") != (
            CANDIDATE_ALIGNMENT_CONTRACT
        ):
            raise ValueError("local element candidate alignment contract mismatch")
    if model_version.startswith((
        "dajoong-local-element-student-v11-",
        "dajoong-local-element-student-v12-",
        "dajoong-local-element-student-v13-",
        "dajoong-local-element-student-v14-",
        "dajoong-local-element-student-v15-",
        "dajoong-local-element-student-v16-",
    )):
        expected_graph_contract = (
            CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
            if model_version.startswith((
                "dajoong-local-element-student-v14-",
                "dajoong-local-element-student-v15-",
                "dajoong-local-element-student-v16-",
            ))
            else LEGACY_CANDIDATE_HYPOTHESIS_CONTEXT_CONTRACT
        )
        if payload.get("candidate_hypothesis_context_contract") != expected_graph_contract:
            raise ValueError("local element proposal graph context mismatch")
    input_contract = payload.get("input_contract")
    if input_contract not in {
        LEGACY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        WHOLE_SHEET_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    }:
        raise ValueError("local element evidence contract mismatch")
    expected_input_channels = (
        LOCAL_ELEMENT_INPUT_CHANNELS
        if input_contract in {
            LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        }
        else LEGACY_LOCAL_ELEMENT_INPUT_CHANNELS
    )
    if payload.get("input_channels") != expected_input_channels:
        raise ValueError("local element channel contract mismatch")
    if model_version.startswith("dajoong-local-element-student-v12-") and (
        input_contract != LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT
    ):
        raise ValueError("focused local element model requires focused detail evidence")
    if (
        input_contract == LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT
        and not model_version.startswith("dajoong-local-element-student-v12-")
    ):
        raise ValueError("focused detail evidence requires a v12 local element model")
    if model_version.startswith((
        "dajoong-local-element-student-v13-",
        "dajoong-local-element-student-v14-",
        "dajoong-local-element-student-v15-",
        "dajoong-local-element-student-v16-",
    )) and (
        input_contract != LOCAL_ELEMENT_EVIDENCE_CONTRACT
    ):
        raise ValueError("letterbox-aligned local model requires v7 evidence")
    if input_contract == LOCAL_ELEMENT_EVIDENCE_CONTRACT and not model_version.startswith(
        (
            "dajoong-local-element-student-v13-",
            "dajoong-local-element-student-v14-",
            "dajoong-local-element-student-v15-",
            "dajoong-local-element-student-v16-",
        )
    ):
        raise ValueError("v7 local evidence requires a v13 local element model")
    if model_version.startswith("dajoong-local-element-student-v15-") and (
        payload.get("perception_authority_contract")
        != LEGACY_LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
    ):
        raise ValueError("legacy dual-authority local element contract mismatch")
    if model_version.startswith("dajoong-local-element-student-v16-") and (
        payload.get("perception_authority_contract")
        != LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT
    ):
        raise ValueError("dual-authority local element contract mismatch")
    if tuple(payload.get("classes") or ()) != ELEMENT_PROGRAM_CLASSES:
        raise ValueError("local element class contract mismatch")
    if tuple(payload.get("geometry_channels") or ()) != ELEMENT_GEOMETRY_CHANNELS:
        raise ValueError("local element geometry contract mismatch")
    output_names = tuple(
        payload.get("output_names")
        or ("class_logits", "geometry", "uncertainty")
    )
    if output_names not in {
        ("class_logits", "geometry", "uncertainty"),
        ("class_logits", "objectness", "geometry", "uncertainty"),
        (
            "class_logits",
            "family_logits",
            "objectness",
            "geometry",
            "uncertainty",
        ),
    }:
        raise ValueError("unsupported local element output contract")
    if "objectness" in output_names:
        if payload.get("objectness_contract") != (
            "binary_object_existence_before_conditional_taxonomy_v1"
        ):
            raise ValueError("local element objectness contract mismatch")
        if payload.get("class_semantics") != (
            "foreground_taxonomy_conditional_on_objectness_v1"
        ):
            raise ValueError("local element class semantics mismatch")
    if input_contract in {
        LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    }:
        if output_names not in {
            ("class_logits", "objectness", "geometry", "uncertainty"),
            (
                "class_logits",
                "family_logits",
                "objectness",
                "geometry",
                "uncertainty",
            ),
        }:
            raise ValueError("hierarchical local element model requires objectness")
        if payload.get("local_view_contract") != (
            "native_detail_assembly_room_v1"
        ):
            raise ValueError("hierarchical local element view contract mismatch")
    if input_contract in {
        WHOLE_SHEET_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        LOCAL_ELEMENT_EVIDENCE_CONTRACT,
    }:
        if tuple(payload.get("input_names") or ()) != (
            "element_crop_evidence",
            "whole_sheet_evidence",
            "candidate_context",
        ):
            raise ValueError("context-aware local element input names mismatch")
        if payload.get("whole_sheet_input_channels") != 4:
            raise ValueError("whole-sheet local element channel contract mismatch")
        if payload.get("whole_sheet_contract") != "explicit_complete_plan_evidence_v1":
            raise ValueError("whole-sheet local element context contract mismatch")
        expected_features = (
            (
                LOCAL_ELEMENT_CONTEXT_FEATURES
                if model_version.startswith((
                    "dajoong-local-element-student-v11-",
                    "dajoong-local-element-student-v12-",
                    "dajoong-local-element-student-v13-",
                    "dajoong-local-element-student-v14-",
                    "dajoong-local-element-student-v15-",
                    "dajoong-local-element-student-v16-",
                ))
                else 20
            )
            if input_contract in {
                RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            }
            else 4
        )
        if payload.get("candidate_context_features") != expected_features:
            raise ValueError("candidate local element context width mismatch")
        expected_context_contract = (
            LOCAL_ELEMENT_CONTEXT_CONTRACT
            if model_version.startswith((
                "dajoong-local-element-student-v14-",
                "dajoong-local-element-student-v15-",
                "dajoong-local-element-student-v16-",
            ))
            else LEGACY_LETTERBOX_LOCAL_ELEMENT_CONTEXT_CONTRACT
            if model_version.startswith("dajoong-local-element-student-v13-")
            else LEGACY_LOCAL_ELEMENT_CONTEXT_CONTRACT
            if model_version.startswith((
                "dajoong-local-element-student-v11-",
                "dajoong-local-element-student-v12-",
            ))
            else "normalized_bbox_room_and_wall_relations_v2"
            if input_contract in {
                RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
                LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            }
            else "normalized_origin_extent_v1"
        )
        if payload.get("candidate_context_contract") != expected_context_contract:
            raise ValueError("candidate local element context contract mismatch")
    if "family_logits" in output_names:
        if tuple(payload.get("family_classes") or ()) != ELEMENT_FAMILY_CLASSES:
            raise ValueError("local element family class contract mismatch")
        if tuple(payload.get("class_family_indices") or ()) != (
            ELEMENT_CLASS_FAMILY_INDICES
        ):
            raise ValueError("local element class-to-family contract mismatch")
        expected_family_contract = (
            ELEMENT_FAMILY_CONTRACT
            if model_version.startswith((
                "dajoong-local-element-student-v13-",
                "dajoong-local-element-student-v14-",
                "dajoong-local-element-student-v15-",
                "dajoong-local-element-student-v16-",
            ))
            else LEGACY_ELEMENT_FAMILY_CONTRACT
        )
        if payload.get("family_contract") != expected_family_contract:
            raise ValueError("local element family output contract mismatch")
    if model_version.startswith((
        "dajoong-local-element-student-v11-",
        "dajoong-local-element-student-v12-",
        "dajoong-local-element-student-v13-",
        "dajoong-local-element-student-v14-",
        "dajoong-local-element-student-v15-",
        "dajoong-local-element-student-v16-",
    )) and (
        "family_logits" not in output_names
    ):
        raise ValueError("relational hierarchy model requires family logits")
    if require_production and not payload.get("production_authorized", False):
        raise PermissionError("local element model is not authorized for production")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -30.0, 30.0))
    return exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-9)


def _preserve_structural_candidate(
    *,
    requires_confirmation: bool,
    accepted_class: bool,
    set_deferred: bool,
    objectness: float,
    threshold: float,
) -> bool:
    """Keep source geometry when existence is strong but taxonomy is not.

    This makes the model hierarchy executable rather than documentary: the
    independent structural authority decides whether ink is an object, while
    the semantic authority may leave its exact family unresolved for review.
    """

    return (
        requires_confirmation
        and not accepted_class
        and not set_deferred
        and objectness >= threshold
    )


def _valid_refined_bbox(
    proposal: tuple[float, float, float, float],
    refined: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    left, top, right, bottom = refined
    if not all(math.isfinite(value) for value in refined) or right <= left or bottom <= top:
        return None
    proposal_width = proposal[2] - proposal[0]
    proposal_height = proposal[3] - proposal[1]
    width, height = right - left, bottom - top
    if not (
        proposal_width * 0.3 <= width <= proposal_width * 3.0
        and proposal_height * 0.3 <= height <= proposal_height * 3.0
    ):
        return None
    proposal_center = ((proposal[0] + proposal[2]) / 2, (proposal[1] + proposal[3]) / 2)
    center = ((left + right) / 2, (top + bottom) / 2)
    if (
        abs(center[0] - proposal_center[0]) > max(8.0, proposal_width)
        or abs(center[1] - proposal_center[1]) > max(8.0, proposal_height)
    ):
        return None
    return (
        max(0.0, left),
        max(0.0, top),
        min(float(image_size[0]), right),
        min(float(image_size[1]), bottom),
    )


def _native_geometry_choice(
    proposal: tuple[float, float, float, float],
    refined: tuple[float, float, float, float] | None,
    *,
    model_risk: float,
    native_candidate: bool,
) -> tuple[tuple[float, float, float, float], bool]:
    """Keep source-derived envelopes when regression is not corroborated.

    Native proposals are measured from the source raster itself.  A synthetic
    geometry head must therefore not replace a strong source envelope with a
    substantially different box merely because the local class score is high.
    Small, mutually supported corrections remain possible, while coarse global
    proposals keep the previous unrestricted refinement path.
    """

    if refined is None:
        return proposal, False
    changed = any(
        abs(left - right) > 0.5
        for left, right in zip(refined, proposal, strict=True)
    )
    if not native_candidate or not changed:
        return refined, changed
    agreement = _bbox_iou(proposal, refined)
    proposal_area = max(
        1e-6,
        (proposal[2] - proposal[0]) * (proposal[3] - proposal[1]),
    )
    refined_area = max(
        1e-6,
        (refined[2] - refined[0]) * (refined[3] - refined[1]),
    )
    intersection = max(0.0, min(proposal[2], refined[2]) - max(proposal[0], refined[0])) * max(
        0.0, min(proposal[3], refined[3]) - max(proposal[1], refined[1])
    )
    proposal_containment = intersection / proposal_area
    expansion = refined_area / proposal_area
    # A fragment candidate is allowed to grow into a complete object only when
    # the regression preserves virtually all source ink, stays within a bounded
    # semantic envelope, and reports very low epistemic risk.  Previously every
    # such correction was rejected solely for having low IoU with the fragment.
    source_preserving_completion = (
        model_risk <= 0.04
        and proposal_containment >= 0.92
        and 1.0 <= expansion <= 6.0
    )
    if source_preserving_completion:
        return refined, True
    # The proposal is measured from native source ink while regression is a
    # learned correction.  Require majority geometric corroboration even for a
    # semantically confident prediction; model risk only raises that bar.
    minimum_agreement = 0.74 if model_risk >= 0.20 else 0.60
    if agreement < minimum_agreement:
        return proposal, False
    return refined, True


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-9, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / (left_area + right_area - intersection)


_WALL_HOSTED_ELEMENT_CLASSES = {"door", "window"}


def _element_has_required_host(
    proposal: PixelSymbolProposal,
    walls: list[Any],
) -> bool:
    """Validate only relationships that are mandatory in a BIM graph.

    Text and dimension fragments were frequently classified as windows because
    their local glyphs look similar. A real door or window must intersect a wall
    band and align with its axis. Other objects are intentionally not subjected
    to heuristic room rules here; uncertainty stays visible for review.
    """

    if proposal.symbol_class not in _WALL_HOSTED_ELEMENT_CLASSES:
        return True
    if not walls:
        return False
    left, top, right, bottom = proposal.bbox_px
    width = max(1e-6, right - left)
    height = max(1e-6, bottom - top)
    center = np.asarray(proposal.center_px, dtype=np.float64)
    element_horizontal = width >= height
    for wall in walls:
        start = np.asarray(wall.start_px, dtype=np.float64)
        end = np.asarray(wall.end_px, dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        wall_horizontal = abs(vector[0]) >= abs(vector[1])
        if wall_horizontal != element_horizontal:
            continue
        fraction = float(np.clip(np.dot(center - start, vector) / (length * length), 0, 1))
        distance = float(np.linalg.norm(center - (start + fraction * vector)))
        tolerance = max(
            5.0,
            float(wall.thickness_px or 4.0) * 1.5,
            min(width, height) * 1.25,
        )
        if distance <= tolerance:
            return True
    return False


class LocalElementOnnxRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        *,
        threads: int = 1,
        require_production: bool = True,
        inference_batch_size: int = 128,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.manifest_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_local_element_manifest(
            self.manifest,
            self.model_path,
            require_production=require_production,
        )
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install onnxruntime to run the local element model") from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_size = int(self.manifest["input_size"])
        self.model_version = str(self.manifest["model_version"])
        self.model_sha256 = str(self.manifest["artifact_sha256"])
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.input_contract = str(self.manifest.get("input_contract"))
        self.context_aware = self.input_contract in {
            WHOLE_SHEET_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        }
        self.relation_aware = self.input_contract in {
            RELATION_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        }
        self.hierarchical_views = self.input_contract in {
            LEGACY_HIERARCHY_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        }
        self.focused_detail = self.input_contract in {
            LEGACY_FOCUSED_LOCAL_ELEMENT_EVIDENCE_CONTRACT,
            LOCAL_ELEMENT_EVIDENCE_CONTRACT,
        }
        self.proposal_graph_aware = self.model_version.startswith(
            (
                "dajoong-local-element-student-v11-",
                "dajoong-local-element-student-v12-",
                "dajoong-local-element-student-v13-",
                "dajoong-local-element-student-v14-",
                "dajoong-local-element-student-v15-",
                "dajoong-local-element-student-v16-",
            )
        )
        expected_inputs = (
            (
                "element_crop_evidence",
                "whole_sheet_evidence",
                "candidate_context",
            )
            if self.context_aware
            else ("element_crop_evidence",)
        )
        if self.input_names != expected_inputs:
            raise ValueError(
                "local element runtime inputs do not match the sealed manifest: "
                f"{self.input_names!r}"
            )
        if inference_batch_size < 1:
            raise ValueError("inference_batch_size must be positive")
        self.inference_batch_size = int(inference_batch_size)

    def refine(
        self,
        image: Image.Image,
        proposals: list[PixelSymbolProposal],
        *,
        class_threshold: float = 0.55,
        discover_candidates: bool = False,
        source_ref_ids: list[str] | None = None,
        discovery_threshold: float = 0.78,
        discovery_region_px: tuple[int, int, int, int] | None = None,
        full_evidence: np.ndarray | None = None,
        host_walls: list[Any] | None = None,
        room_regions: list[Any] | None = None,
    ) -> tuple[list[PixelSymbolProposal], LocalElementRefinementDiagnostics]:
        started = time.perf_counter()
        native_diagnostics = None
        discovered_ids: set[str] = set()
        semantic_window_ids = {
            proposal.id
            for proposal in proposals
            if "native-detail-window" in proposal.model_version
        }
        if discover_candidates:
            if not source_ref_ids:
                raise ValueError("source_ref_ids are required for native discovery")
            discovery_image = image
            discovery_offset = (0.0, 0.0)
            if discovery_region_px is not None:
                left, top, right, bottom = discovery_region_px
                if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                    raise ValueError("discovery_region_px falls outside the source image")
                discovery_image = image.crop(discovery_region_px)
                discovery_offset = (float(left), float(top))
            discovered, native_diagnostics = mine_native_element_candidates(
                discovery_image,
                source_ref_ids=source_ref_ids,
            )
            if discovery_offset != (0.0, 0.0):
                offset_x, offset_y = discovery_offset
                discovered = [
                    item.model_copy(
                        update={
                            "center_px": (
                                item.center_px[0] + offset_x,
                                item.center_px[1] + offset_y,
                            ),
                            "bbox_px": (
                                item.bbox_px[0] + offset_x,
                                item.bbox_px[1] + offset_y,
                                item.bbox_px[2] + offset_x,
                                item.bbox_px[3] + offset_y,
                            ),
                        }
                    )
                    for item in discovered
                ]
            # Do not delete native-resolution evidence merely because a coarse
            # dense head overlaps it.  A 0.25 IoU pre-filter used to discard the
            # exact object whenever the global box was oversized or shifted.
            # All hypotheses now reach the classifier; identity and duplicate
            # resolution happens once, over the complete accepted set below.
            discovered_ids = {item.id for item in discovered}
            proposals = [*proposals, *discovered]
        proposal_count = len(proposals)
        valid_proposals = []
        invalid_proposals = 0
        for proposal in proposals:
            left, top, right, bottom = proposal.bbox_px
            if (
                not all(math.isfinite(value) for value in proposal.bbox_px)
                or right <= left
                or bottom <= top
            ):
                invalid_proposals += 1
                discovered_ids.discard(proposal.id)
                continue
            valid_proposals.append(proposal)
        proposals = valid_proposals
        if not proposals:
            return [], LocalElementRefinementDiagnostics(
                proposal_count=proposal_count,
                reclassified_count=0,
                geometry_refined_count=0,
                rejected_count=invalid_proposals,
                invalid_proposal_count=invalid_proposals,
                native_candidate_diagnostics=native_diagnostics,
                model_version=self.model_version,
                model_sha256=self.model_sha256,
                timings_ms={"total": 0.0},
            )
        transforms = []
        probability_batches = []
        objectness_batches = []
        geometry_batches = []
        risk_batches = []
        if full_evidence is None:
            full_evidence = build_cad_evidence(image)
        elif full_evidence.ndim != 3 or full_evidence.shape[1:] != (
            image.height,
            image.width,
        ):
            raise ValueError("full_evidence does not match the source image")
        whole_plan_inputs: dict[str, np.ndarray] = {}
        if self.context_aware:
            full_plan_input, _ = letterbox_cad_evidence(full_evidence, self.input_size)
            whole_plan_inputs["full"] = full_plan_input.astype(np.float32)
            if discovery_region_px is not None:
                left, top, right, bottom = discovery_region_px
                region_evidence = full_evidence[:, top:bottom, left:right]
                region_input, _ = letterbox_cad_evidence(
                    region_evidence,
                    self.input_size,
                )
                whole_plan_inputs["region"] = region_input.astype(np.float32)
        preparation_ms = 0.0
        inference_ms = 0.0
        candidate_boxes = [proposal.bbox_px for proposal in proposals]
        proposal_graph_context = (
            (
                candidate_hypothesis_contexts(candidate_boxes)
                if self.model_version.startswith((
                    "dajoong-local-element-student-v14-",
                    "dajoong-local-element-student-v15-",
                    "dajoong-local-element-student-v16-",
                ))
                else legacy_candidate_hypothesis_contexts(candidate_boxes)
            )
            if self.proposal_graph_aware
            else None
        )
        for offset in range(0, len(proposals), self.inference_batch_size):
            batch = proposals[offset : offset + self.inference_batch_size]
            preparation_started = time.perf_counter()
            crop_boxes = [proposal.bbox_px for proposal in batch]
            if self.hierarchical_views:
                crops, batch_transforms = extract_local_element_hierarchy_batch_from_map(
                    full_evidence,
                    image.size,
                    crop_boxes,
                    input_size=self.input_size,
                    detail_scale=2.1,
                    assembly_scale=6.5,
                    room_scale=18.0,
                    focus_detail=self.focused_detail,
                )
            else:
                crops, batch_transforms = extract_local_element_pyramid_batch_from_map(
                    full_evidence,
                    image.size,
                    crop_boxes,
                    input_size=self.input_size,
                    detail_scale=2.1,
                    context_scale=5.5,
                )
            feed: dict[str, np.ndarray] = {
                "element_crop_evidence": crops.astype(np.float32)
            }
            if self.context_aware:
                whole_batch = []
                context_batch = []
                for batch_index, proposal in enumerate(batch):
                    use_region = False
                    if discovery_region_px is not None:
                        frame_left, frame_top, frame_right, frame_bottom = (
                            discovery_region_px
                        )
                        left, top, right, bottom = proposal.bbox_px
                        use_region = (
                            frame_left <= left < right <= frame_right
                            and frame_top <= top < bottom <= frame_bottom
                        )
                    frame = discovery_region_px if use_region else None
                    whole_batch.append(
                        whole_plan_inputs["region" if use_region else "full"][0]
                    )
                    bbox_context = normalized_candidate_context(
                        proposal.bbox_px,
                        image_size=image.size,
                        frame_bbox=frame,
                        letterbox_size=(
                            self.input_size
                            if self.model_version.startswith(
                                (
                                    "dajoong-local-element-student-v13-",
                                    "dajoong-local-element-student-v14-",
                                    "dajoong-local-element-student-v15-",
                                    "dajoong-local-element-student-v16-",
                                )
                            )
                            else None
                        ),
                    )
                    if self.relation_aware:
                        room_context = [
                            (room.room_class, list(room.polygon_px))
                            for room in (room_regions or [])
                        ]
                        wall_context = [
                            (
                                wall.start_px,
                                wall.end_px,
                                float(wall.thickness_px or 4.0),
                            )
                            for wall in (host_walls or [])
                        ]
                        bbox_context = np.concatenate(
                            (
                                bbox_context,
                                semantic_element_context(
                                    proposal.bbox_px,
                                    image_size=image.size,
                                    rooms=room_context,
                                    walls=wall_context,
                                ),
                            )
                        )
                        if self.proposal_graph_aware:
                            assert proposal_graph_context is not None
                            bbox_context = np.concatenate(
                                (
                                    bbox_context,
                                    proposal_graph_context[offset + batch_index],
                                )
                            )
                    context_batch.append(bbox_context)
                feed["whole_sheet_evidence"] = np.stack(whole_batch).astype(
                    np.float32
                )
                feed["candidate_context"] = np.stack(context_batch).astype(np.float32)
            preparation_ms += (time.perf_counter() - preparation_started) * 1000
            inference_started = time.perf_counter()
            output_names = list(self.manifest["output_names"])
            output_values = self.session.run(output_names, feed)
            model_output = dict(zip(output_names, output_values, strict=True))
            class_logits = model_output["class_logits"]
            geometry = model_output["geometry"]
            inference_ms += (time.perf_counter() - inference_started) * 1000
            if "objectness" in model_output:
                conditional = _softmax(
                    np.asarray(class_logits, dtype=np.float32)[:, 1:]
                )
                probabilities = np.zeros(
                    (len(batch), len(ELEMENT_PROGRAM_CLASSES)), dtype=np.float32
                )
                probabilities[:, 1:] = conditional
                # v13+ treats family classification as an auxiliary consistency
                # task.  Multiplying two independently normalized heads made
                # final confidence depend on family cardinality and suppressed
                # correct fine classes on real sheets.  Keep the legacy decoder
                # only for sealed v11/v12 artifacts.
                if "family_logits" in model_output and not self.model_version.startswith(
                    (
                        "dajoong-local-element-student-v13-",
                        "dajoong-local-element-student-v14-",
                        "dajoong-local-element-student-v15-",
                        "dajoong-local-element-student-v16-",
                    )
                ):
                    family_probability = _softmax(
                        np.asarray(model_output["family_logits"], dtype=np.float32)
                    )
                    family_lookup = np.asarray(
                        ELEMENT_CLASS_FAMILY_INDICES,
                        dtype=np.int64,
                    )
                    probabilities[:, 1:] = (
                        conditional * family_probability[:, family_lookup]
                    )
                objectness = np.asarray(
                    model_output["objectness"], dtype=np.float32
                ).reshape(-1)
            else:
                probabilities = _softmax(
                    np.asarray(class_logits, dtype=np.float32)
                )
                objectness = 1.0 - probabilities[:, 0]
            probability_batches.append(probabilities)
            objectness_batches.append(objectness)
            geometry_batches.append(np.asarray(geometry, dtype=np.float32))
            risk_batches.append(
                np.asarray(model_output["uncertainty"], dtype=np.float32).reshape(-1)
            )
            transforms.extend(batch_transforms)
        probability = np.concatenate(probability_batches, axis=0)
        objectness_probability = np.concatenate(objectness_batches, axis=0)
        geometry = np.concatenate(geometry_batches, axis=0)
        model_risk = np.concatenate(risk_batches, axis=0)
        output = []
        reclassified = 0
        geometry_refined = 0
        rejected = invalid_proposals
        unresolved_discovered: list[UnresolvedNativeElementCandidate] = []
        for index, proposal in enumerate(proposals):
            class_index = int(probability[index, 1:].argmax()) + 1
            confidence = float(probability[index, class_index])
            objectness = float(objectness_probability[index])
            background_probability = 1.0 - objectness
            class_name = ELEMENT_PROGRAM_CLASSES[class_index]
            is_discovered = proposal.id in discovered_ids
            is_semantic_window = proposal.id in semantic_window_ids
            requires_confirmation = is_discovered or is_semantic_window
            threshold = discovery_threshold if requires_confirmation else class_threshold
            foreground_margin = confidence - background_probability
            accepted_class = (
                objectness >= threshold
                and confidence >= threshold
                and foreground_margin >= (0.30 if requires_confirmation else 0.12)
                and class_name not in {"background", "unknown"}
            )
            family_name = (
                ELEMENT_FAMILY_CLASSES[ELEMENT_CLASS_FAMILY_INDICES[class_index - 1]]
                if class_index > 0
                else "misc"
            )
            set_deferred = (
                requires_confirmation
                and not accepted_class
                and class_name not in {"background", "unknown"}
                and family_name in {"casework", "appliance", "plumbing"}
                and proposal_graph_context is not None
                and float(np.max(proposal_graph_context[index, 2:4])) >= 0.55
                and confidence >= max(0.45, threshold - 0.25)
                # Set relations may resolve an uncertain fine taxonomy, but
                # they must never manufacture object existence.  The former
                # relaxed 0.40 gate let confident synthetic class logits revive
                # background ink as plumbing/fireplace objects on real sheets.
                and objectness >= threshold
            )
            # Existence and taxonomy are separate authorities. A native mark
            # with strong structural objectness must not disappear merely
            # because its exact 47-way class is uncertain. Preserve its source
            # footprint as an editable unknown object and require semantic
            # review. Only low structural objectness may reject the candidate.
            structure_preserved = _preserve_structural_candidate(
                requires_confirmation=requires_confirmation,
                accepted_class=accepted_class,
                set_deferred=set_deferred,
                objectness=objectness,
                threshold=threshold,
            )
            if requires_confirmation and not accepted_class and not set_deferred:
                unresolved_discovered.append(
                    UnresolvedNativeElementCandidate(
                        candidate_id=proposal.id,
                        bbox_px=proposal.bbox_px,
                        proposed_class=class_name,
                        confidence=(objectness if structure_preserved else confidence),
                        reason=(
                            "taxonomy_unresolved_structure_preserved"
                            if structure_preserved
                            else (
                                "background_or_unknown"
                                if class_name in {"background", "unknown"}
                                else (
                                    "insufficient_foreground_margin"
                                    if foreground_margin < 0.30
                                    else "below_discovery_threshold"
                                )
                            )
                        ),
                    )
                )
                if not structure_preserved:
                    rejected += 1
                    continue
            label = (
                "unknown"
                if structure_preserved
                else class_name
                if accepted_class or set_deferred
                else proposal.symbol_class
            )
            if accepted_class and label != proposal.symbol_class:
                reclassified += 1
            candidate_bbox, _ = decode_element_geometry(geometry[index], transforms[index])
            valid_candidate_bbox = _valid_refined_bbox(
                proposal.bbox_px,
                candidate_bbox,
                image.size,
            )
            if valid_candidate_bbox is None:
                rejected += 1
            refined_bbox, geometry_changed = _native_geometry_choice(
                proposal.bbox_px,
                valid_candidate_bbox,
                model_risk=float(model_risk[index]),
                native_candidate=is_discovered,
            )
            if geometry_changed:
                geometry_refined += 1
            joint_confidence = (
                objectness if structure_preserved else min(confidence, objectness)
            )
            calibrated_risk = max(
                1.0 - joint_confidence,
                float(model_risk[index]),
                1.0 - confidence if structure_preserved else 0.0,
            )
            output.append(
                proposal.model_copy(
                    update={
                        "symbol_class": label,
                        "center_px": (
                            (refined_bbox[0] + refined_bbox[2]) / 2,
                            (refined_bbox[1] + refined_bbox[3]) / 2,
                        ),
                        "bbox_px": refined_bbox,
                        "confidence": max(proposal.confidence, joint_confidence),
                        "uncertainty": min(proposal.uncertainty, calibrated_risk),
                        "model_version": f"{proposal.model_version}+{self.model_version}",
                        "review_required": (
                            proposal.review_required or not accepted_class
                        ),
                    }
                )
            )
            if set_deferred:
                output[-1] = output[-1].model_copy(
                    update={
                        "model_version": f"{output[-1].model_version}+set-deferred-v1"
                    }
                )
        host_validated = []
        for proposal in output:
            if _element_has_required_host(proposal, host_walls or []):
                host_validated.append(proposal)
                continue
            rejected += 1
            if proposal.id in discovered_ids or proposal.id in semantic_window_ids:
                unresolved_discovered.append(
                    UnresolvedNativeElementCandidate(
                        candidate_id=proposal.id,
                        bbox_px=proposal.bbox_px,
                        proposed_class=proposal.symbol_class,
                        confidence=proposal.confidence,
                        reason="required_wall_host_missing",
                    )
                )
        # Final identity is a set-level decision.  The earlier implementation
        # accepted each crop independently and only ran a shallow IoU pass,
        # allowing one drawing symbol to become many BIM objects.  Resolve
        # host and identity relations jointly over the complete accepted set.
        output, set_decisions = decode_element_set(
            host_validated,
            host_walls=host_walls or [],
        )
        set_rejections = sum(
            decision.decision != "selected" for decision in set_decisions
        )
        rejected += set_rejections
        rejected_by_set = {
            decision.candidate_id: decision.decision
            for decision in set_decisions
            if decision.decision != "selected"
        }
        for proposal in host_validated:
            reason = rejected_by_set.get(proposal.id)
            if reason != "insufficient_set_support":
                continue
            unresolved_discovered.append(
                UnresolvedNativeElementCandidate(
                    candidate_id=proposal.id,
                    bbox_px=proposal.bbox_px,
                    proposed_class=proposal.symbol_class,
                    confidence=proposal.confidence,
                    reason=reason,
                )
            )
        selected_ids = {proposal.id for proposal in output}
        accepted_discovered = len(selected_ids & discovered_ids)
        accepted_semantic_window = len(selected_ids & semantic_window_ids)
        return output, LocalElementRefinementDiagnostics(
            proposal_count=proposal_count,
            reclassified_count=reclassified,
            geometry_refined_count=geometry_refined,
            rejected_count=rejected,
            invalid_proposal_count=invalid_proposals,
            discovered_candidate_count=len(discovered_ids),
            accepted_discovered_count=accepted_discovered,
            semantic_window_candidate_count=len(semantic_window_ids),
            accepted_semantic_window_count=accepted_semantic_window,
            unresolved_discovered=unresolved_discovered,
            native_candidate_diagnostics=native_diagnostics,
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            timings_ms={
                "preparation": round(preparation_ms, 3),
                "onnx": round(inference_ms, 3),
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        )
