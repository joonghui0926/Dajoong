"""Tiny context-aware classifier for native architectural element proposals.

Native detail remains authoritative for class and box geometry.  A second branch
reads the complete plan and samples the proposal location, so identical local ink
is interpreted in its actual room and building context instead of in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
)

LOCAL_ELEMENT_CONTEXT_FEATURES = 24
LOCAL_ELEMENT_STRUCTURE_CONTEXT_FEATURES = 11
LEGACY_LOCAL_ELEMENT_CONTEXT_CONTRACT = (
    "normalized_bbox_room_wall_and_proposal_graph_relations_v3"
)
LEGACY_LETTERBOX_LOCAL_ELEMENT_CONTEXT_CONTRACT = (
    "letterbox_aligned_bbox_room_wall_and_proposal_graph_relations_v4"
)
LOCAL_ELEMENT_CONTEXT_CONTRACT = (
    "letterbox_aligned_bbox_room_wall_and_equipment_run_relations_v5"
)
LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT = (
    "independent_local_global_structure_and_semantic_authorities_v2"
)
LEGACY_LOCAL_ELEMENT_DUAL_AUTHORITY_CONTRACT = (
    "independent_structure_objectness_geometry_and_semantic_trunks_v1"
)
ELEMENT_FAMILY_CLASSES = (
    "opening",
    "casework",
    "appliance",
    "plumbing",
    "hearth",
    "structure",
    "building_system",
    "furniture",
    "misc",
)
_ELEMENT_CLASS_FAMILY = {
    "door": "opening",
    "window": "opening",
    "base_cabinet": "casework",
    "wall_cabinet": "casework",
    "closet": "casework",
    "coat_closet": "casework",
    "sauna_bench": "casework",
    "housing": "casework",
    "coat_rack": "casework",
    "electrical_appliance": "appliance",
    "refrigerator": "appliance",
    "stove": "appliance",
    "dishwasher": "appliance",
    "washing_machine": "appliance",
    "tumble_dryer": "appliance",
    "toilet": "plumbing",
    "sink": "plumbing",
    "shower": "plumbing",
    "shower_screen": "plumbing",
    "bathtub": "plumbing",
    "plumbing_fixture": "plumbing",
    "water_tap": "plumbing",
    "jacuzzi": "plumbing",
    "fireplace": "hearth",
    "chimney": "hearth",
    "wood_stove": "hearth",
    "fireplace_corner": "hearth",
    "place_for_fireplace": "hearth",
    "place_for_fireplace_corner": "hearth",
    "column": "structure",
    "stair": "structure",
    "light": "building_system",
    "electrical_panel": "building_system",
    "receptacle": "building_system",
    "hvac_terminal": "building_system",
    "sprinkler": "building_system",
    "riser": "building_system",
    "bed": "furniture",
    "sofa": "furniture",
    "armchair": "furniture",
    "chair": "furniture",
    "dining_table": "furniture",
    "coffee_table": "furniture",
    "desk": "furniture",
    "bench": "furniture",
    "misc": "misc",
    "unknown": "misc",
}
ELEMENT_CLASS_FAMILY_INDICES = tuple(
    ELEMENT_FAMILY_CLASSES.index(_ELEMENT_CLASS_FAMILY[class_name])
    for class_name in ELEMENT_PROGRAM_CLASSES[1:]
)
LEGACY_ELEMENT_FAMILY_CONTRACT = "foreground_hierarchical_family_joint_probability_v1"
ELEMENT_FAMILY_CONTRACT = "foreground_family_auxiliary_consistency_v2"

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError:  # pragma: no cover - inference installation does not require torch.
    torch = None
    nn = None
    functional = None


@dataclass(frozen=True)
class LocalElementStudentConfig:
    model_version: str = (
        "dajoong-local-element-student-v16-global-dual-authority-relational-hierarchy"
    )
    input_channels: int = 12
    whole_sheet_input_channels: int = 4
    candidate_context_features: int = LOCAL_ELEMENT_CONTEXT_FEATURES
    input_size: int = 64
    stem_width: int = 16
    classes: tuple[str, ...] = ELEMENT_PROGRAM_CLASSES
    geometry_channels: tuple[str, ...] = ELEMENT_GEOMETRY_CHANNELS

    def validate(self) -> None:
        if self.input_channels != 12:
            raise ValueError(
                "local element evidence requires detail, assembly, and room views"
            )
        if self.whole_sheet_input_channels != 4:
            raise ValueError("whole-sheet element context requires four channels")
        if self.candidate_context_features != LOCAL_ELEMENT_CONTEXT_FEATURES:
            raise ValueError(
                "candidate context requires bbox, room, wall, and proposal-graph relationships"
            )
        if self.input_size < 32 or self.input_size % 16:
            raise ValueError("local element input size must be a multiple of 16")
        if self.stem_width < 8 or self.stem_width % 8:
            raise ValueError("local element stem width must be a multiple of eight")
        if self.classes != ELEMENT_PROGRAM_CLASSES:
            raise ValueError("local element classes cannot be reordered")
        if self.geometry_channels != ELEMENT_GEOMETRY_CHANNELS:
            raise ValueError("local element geometry channels cannot be reordered")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classes"] = list(self.classes)
        payload["geometry_channels"] = list(self.geometry_channels)
        return payload


if nn is not None:

    class _ConvNormAct(nn.Sequential):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            *,
            stride: int = 1,
            groups: int = 1,
        ) -> None:
            super().__init__(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(inplace=True),
            )


    class _DepthwiseResidual(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.depthwise = _ConvNormAct(channels, channels, groups=channels)
            self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
            self.norm = nn.BatchNorm2d(channels)

        def forward(self, value: Any) -> Any:
            update = self.norm(self.pointwise(self.depthwise(value)))
            return functional.silu(value + update)


    class DajoongLocalElementStudent(nn.Module):
        def __init__(self, config: LocalElementStudentConfig | None = None) -> None:
            super().__init__()
            self.config = config or LocalElementStudentConfig()
            self.config.validate()
            c1 = self.config.stem_width
            c2, c3 = c1 * 2, c1 * 3
            self.encoder = nn.Sequential(
                _ConvNormAct(self.config.input_channels, c1, stride=2),
                _DepthwiseResidual(c1),
                _ConvNormAct(c1, c2, stride=2),
                _DepthwiseResidual(c2),
                _ConvNormAct(c2, c3, stride=2),
                _DepthwiseResidual(c3),
                # Preserve a compact 4x4 spatial program. Global average pooling
                # erased the very layout cues that separate tubs, showers, stairs,
                # doors, and windows.
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(c3 * 16, c3 * 2),
                nn.SiLU(inplace=True),
            )
            # Existence/geometry and semantic identity deliberately do not
            # share a deep trunk.  On real drawings an unfamiliar symbol can
            # still be a clearly bounded object.  The former shared trunk let
            # synthetic-taxonomy gradients erase that structural evidence.
            # Keeping the compact encoders independent makes "unknown class"
            # a review state instead of "no object".
            self.semantic_encoder = nn.Sequential(
                _ConvNormAct(self.config.input_channels, c1, stride=2),
                _DepthwiseResidual(c1),
                _ConvNormAct(c1, c2, stride=2),
                _DepthwiseResidual(c2),
                _ConvNormAct(c2, c3, stride=2),
                _DepthwiseResidual(c3),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(c3 * 16, c3 * 2),
                nn.SiLU(inplace=True),
            )
            head_width = c3 * 2
            self.whole_sheet_encoder = nn.Sequential(
                _ConvNormAct(self.config.whole_sheet_input_channels, c1, stride=2),
                _ConvNormAct(c1, c2, stride=2),
                _ConvNormAct(c2, c3, stride=2),
            )
            self.whole_sheet_global_projection = nn.Linear(c3, head_width)
            self.whole_sheet_location_projection = nn.Linear(c3, head_width)
            self.structure_whole_sheet_encoder = nn.Sequential(
                _ConvNormAct(self.config.whole_sheet_input_channels, c1, stride=2),
                _ConvNormAct(c1, c2, stride=2),
                _ConvNormAct(c2, c3, stride=2),
            )
            self.structure_whole_global_projection = nn.Linear(c3, head_width)
            self.structure_whole_location_projection = nn.Linear(c3, head_width)
            self.candidate_context_projection = nn.Sequential(
                nn.Linear(self.config.candidate_context_features, head_width),
                nn.SiLU(inplace=True),
                nn.Linear(head_width, head_width),
            )
            self.structure_context_projection = nn.Sequential(
                nn.Linear(LOCAL_ELEMENT_STRUCTURE_CONTEXT_FEATURES, head_width),
                nn.SiLU(inplace=True),
                nn.Linear(head_width, head_width),
            )
            self.class_head = nn.Linear(head_width, len(self.config.classes))
            self.family_head = nn.Linear(head_width, len(ELEMENT_FAMILY_CLASSES))
            # Object existence and semantic family are different questions.
            # A single 48-way softmax forced every background stroke to compete
            # with rare BIM classes and caused confident false objects under the
            # real candidate prior.  This head gates conditional classification.
            self.objectness_head = nn.Linear(head_width, 1)
            self.geometry_head = nn.Linear(head_width, len(self.config.geometry_channels))
            self.uncertainty_head = nn.Linear(head_width * 2, 1)
            self._initialize()

        def _initialize(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    nn.init.constant_(module.bias, 0.0)
            # New context residuals are neutral under warm start. They gain
            # authority only from evidence during contextual training.
            for projection in (
                self.whole_sheet_global_projection,
                self.whole_sheet_location_projection,
                self.structure_whole_global_projection,
                self.structure_whole_location_projection,
                self.candidate_context_projection[-1],
                self.structure_context_projection[-1],
            ):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        @staticmethod
        def _structure_context(candidate_context: Any) -> Any:
            """Remove room identity while retaining physical relationships.

            Candidate context is bbox(4), room one-hot(13), physical relations(3),
            and proposal-graph relations(4). Object existence and geometry may
            use the physical fields, but never the room semantic shortcut.
            """

            return torch.cat(
                (
                    candidate_context[:, :4],
                    candidate_context[:, 17:20],
                    candidate_context[:, 20:24],
                ),
                dim=1,
            )

        def _forward_element(
            self,
            evidence: Any,
            whole_sheet_evidence: Any,
            candidate_context: Any,
        ) -> dict[str, Any]:
            structure_features = self.encoder(evidence)
            semantic_features = self.semantic_encoder(evidence)
            whole_features = self.whole_sheet_encoder(whole_sheet_evidence)
            structure_whole_features = self.structure_whole_sheet_encoder(
                whole_sheet_evidence
            )
            whole_global = self.whole_sheet_global_projection(
                functional.adaptive_avg_pool2d(whole_features, (1, 1)).flatten(1)
            )
            center = torch.stack(
                (
                    candidate_context[:, 0] + candidate_context[:, 2] * 0.5,
                    candidate_context[:, 1] + candidate_context[:, 3] * 0.5,
                ),
                dim=1,
            )
            grid = (center * 2.0 - 1.0)[:, None, None, :]
            whole_location = self.whole_sheet_location_projection(
                functional.grid_sample(
                    whole_features,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).flatten(1)
            )
            structure_whole_global = self.structure_whole_global_projection(
                functional.adaptive_avg_pool2d(
                    structure_whole_features, (1, 1)
                ).flatten(1)
            )
            structure_whole_location = self.structure_whole_location_projection(
                functional.grid_sample(
                    structure_whole_features,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                ).flatten(1)
            )
            semantic_features = (
                semantic_features
                + whole_global
                + whole_location
                + self.candidate_context_projection(candidate_context)
            )
            structure_features = (
                structure_features
                + structure_whole_global
                + structure_whole_location
                + self.structure_context_projection(
                    self._structure_context(candidate_context)
                )
            )
            return {
                "class_logits": self.class_head(semantic_features),
                "family_logits": self.family_head(semantic_features),
                "objectness": torch.sigmoid(self.objectness_head(structure_features)),
                "geometry": self.geometry_head(structure_features),
                "uncertainty": torch.sigmoid(
                    self.uncertainty_head(
                        torch.cat((structure_features, semantic_features), dim=1)
                    )
                ),
            }

        def forward(
            self,
            evidence: Any,
            whole_sheet_evidence: Any | None = None,
            candidate_context: Any | None = None,
        ) -> dict[str, Any]:
            if evidence.ndim != 4 or evidence.shape[1] != self.config.input_channels:
                raise ValueError("evidence must have shape [batch, 12, height, width]")
            if whole_sheet_evidence is None:
                whole_sheet_evidence = evidence[:, :4]
            if candidate_context is None:
                candidate_context = evidence.new_zeros(
                    evidence.shape[0], self.config.candidate_context_features
                )
                candidate_context[:, 2:4] = 1.0
            if whole_sheet_evidence.shape != (
                evidence.shape[0],
                self.config.whole_sheet_input_channels,
                evidence.shape[2],
                evidence.shape[3],
            ):
                raise ValueError("whole_sheet_evidence must align with the crop batch")
            if candidate_context.shape != (
                evidence.shape[0],
                self.config.candidate_context_features,
            ):
                raise ValueError(
                    "candidate_context must have shape "
                    f"[batch, {self.config.candidate_context_features}]"
                )
            return self._forward_element(
                evidence,
                whole_sheet_evidence,
                candidate_context,
            )

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())


    class LocalElementStudentOnnxAdapter(nn.Module):
        def __init__(self, model: DajoongLocalElementStudent) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            evidence: Any,
            whole_sheet_evidence: Any,
            candidate_context: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            output = self.model._forward_element(
                evidence,
                whole_sheet_evidence,
                candidate_context,
            )
            return (
                output["class_logits"],
                output["family_logits"],
                output["objectness"],
                output["geometry"],
                output["uncertainty"],
            )

else:  # pragma: no cover

    class DajoongLocalElementStudent:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")


    class LocalElementStudentOnnxAdapter:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")
