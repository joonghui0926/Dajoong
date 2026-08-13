"""Small full-sheet building-program student for Method v2.

Unlike the legacy tile classifier, this network receives the complete normalized
sheet.  A low-resolution global context branch is fused back into every spatial
location before decoding structural targets.  Local specialists remain separate;
this model is responsible for the complete building program, including the
semantic vocabulary needed to name rooms and editable AEC elements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TOPOLOGY_TARGET_CHANNELS = (
    "exterior_boundary",
    "wall_centerline",
    "junction",
    "opening",
    "room_seed",
    "room_interior",
)

ROOM_PROGRAM_CLASSES = (
    "background",
    "living",
    "bedroom",
    "kitchen",
    "bathroom",
    "hallway",
    "storage",
    "office",
    "mechanical",
    "garage",
    "utility",
    "outdoor",
    "other",
)

ELEMENT_PROGRAM_CLASSES = (
    "background",
    "door",
    "window",
    "base_cabinet",
    "wall_cabinet",
    "closet",
    "coat_closet",
    "electrical_appliance",
    "toilet",
    "sink",
    "shower",
    "shower_screen",
    "bathtub",
    "sauna_bench",
    "fireplace",
    "chimney",
    "column",
    "stair",
    "light",
    "electrical_panel",
    "receptacle",
    "hvac_terminal",
    "sprinkler",
    "riser",
    "plumbing_fixture",
    "housing",
    "coat_rack",
    "water_tap",
    "jacuzzi",
    "wood_stove",
    "fireplace_corner",
    "place_for_fireplace",
    "place_for_fireplace_corner",
    "misc",
    # Drawn furniture and typed appliances are first-class BIM content.  The
    # earlier contract treated these visible symbols as background distractors,
    # which made perfect recall impossible by construction.
    "bed",
    "sofa",
    "armchair",
    "chair",
    "dining_table",
    "coffee_table",
    "desk",
    "bench",
    "refrigerator",
    "stove",
    "dishwasher",
    "washing_machine",
    "tumble_dryer",
    "unknown",
)

ELEMENT_GEOMETRY_CHANNELS = (
    "center_dx",
    "center_dy",
    "log_width",
    "log_height",
    "sin_yaw",
    "cos_yaw",
)

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError:  # pragma: no cover - inference package does not require torch.
    torch = None
    nn = None
    functional = None


@dataclass(frozen=True)
class GlobalTopologyStudentConfig:
    model_version: str = "dajoong-global-program-student-v8-axis-consistent-context"
    input_channels: int = 4
    stem_width: int = 16
    output_channels: tuple[str, ...] = TOPOLOGY_TARGET_CHANNELS
    room_classes: tuple[str, ...] = ROOM_PROGRAM_CLASSES
    element_classes: tuple[str, ...] = ELEMENT_PROGRAM_CLASSES
    element_geometry_channels: tuple[str, ...] = ELEMENT_GEOMETRY_CHANNELS
    output_stride: int = 1
    context_grids: tuple[int, ...] = (1, 2, 4)
    crop_context_features: int = 8

    def validate(self) -> None:
        if self.input_channels != 4:
            raise ValueError("global topology evidence requires four input channels")
        if self.stem_width < 8 or self.stem_width % 8:
            raise ValueError("stem_width must be a multiple of eight and at least eight")
        if self.output_channels != TOPOLOGY_TARGET_CHANNELS:
            raise ValueError("the Method v2 topology target contract cannot be reordered")
        if self.room_classes != ROOM_PROGRAM_CLASSES:
            raise ValueError("the room-program class contract cannot be reordered")
        if self.element_classes != ELEMENT_PROGRAM_CLASSES:
            raise ValueError("the element-program class contract cannot be reordered")
        if self.element_geometry_channels != ELEMENT_GEOMETRY_CHANNELS:
            raise ValueError("the element geometry contract cannot be reordered")
        if self.output_stride != 1:
            raise ValueError("topology outputs must align with the normalized full sheet")
        if not self.context_grids or self.context_grids[0] != 1:
            raise ValueError("context_grids must include a whole-sheet 1x1 token")
        if tuple(sorted(set(self.context_grids))) != self.context_grids:
            raise ValueError("context_grids must be unique and ascending")
        if self.crop_context_features != 8:
            raise ValueError("crop context must encode origin, extent, and four edges")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_channels"] = list(self.output_channels)
        payload["room_classes"] = list(self.room_classes)
        payload["element_classes"] = list(self.element_classes)
        payload["element_geometry_channels"] = list(self.element_geometry_channels)
        payload["context_grids"] = list(self.context_grids)
        return payload


if nn is not None:

    class _ConvNormAct(nn.Sequential):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            *,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
        ) -> None:
            super().__init__(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
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
            return functional.silu(value + self.norm(self.pointwise(self.depthwise(value))))


    class _EncoderBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                _ConvNormAct(input_channels, output_channels, stride=2),
                _DepthwiseResidual(output_channels),
            )

        def forward(self, value: Any) -> Any:
            return self.block(value)


    class _DecoderBlock(nn.Module):
        def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
            super().__init__()
            self.fuse = nn.Sequential(
                _ConvNormAct(input_channels + skip_channels, output_channels, kernel_size=1),
                _DepthwiseResidual(output_channels),
            )

        def forward(self, value: Any, skip: Any) -> Any:
            value = functional.interpolate(
                value,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return self.fuse(torch.cat((value, skip), dim=1))


    class _WholeSheetContext(nn.Module):
        """Broadcast pooled tokens so every output sees the complete building."""

        def __init__(self, channels: int, grids: tuple[int, ...]) -> None:
            super().__init__()
            self.grids = grids
            self.fuse = _ConvNormAct(channels * (len(grids) + 1), channels, kernel_size=1)

        def forward(self, value: Any) -> Any:
            height, width = value.shape[-2:]
            context = [value]
            for grid in self.grids:
                pooled = functional.adaptive_avg_pool2d(value, (grid, grid))
                context.append(
                    functional.interpolate(
                        pooled,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            return self.fuse(torch.cat(context, dim=1))


    class DajoongGlobalTopologyStudent(nn.Module):
        """Full-sheet topology model; deterministic graph decoding remains authoritative."""

        def __init__(self, config: GlobalTopologyStudentConfig | None = None) -> None:
            super().__init__()
            self.config = config or GlobalTopologyStudentConfig()
            self.config.validate()
            c1 = self.config.stem_width
            c2, c3, c4 = c1 * 2, c1 * 3, c1 * 4
            self.stem = nn.Sequential(
                _ConvNormAct(self.config.input_channels, c1),
                _DepthwiseResidual(c1),
            )
            self.down2 = _EncoderBlock(c1, c2)
            self.down3 = _EncoderBlock(c2, c3)
            self.down4 = _EncoderBlock(c3, c4)
            self.context = _WholeSheetContext(c4, self.config.context_grids)
            # A detail window cannot infer whether a dark line is a wall, table,
            # hatch, or title text from local pixels alone.  Encode the actual
            # complete sheet independently and inject that token into every
            # detail decode.  This is intentionally separate from ``context``:
            # that module pools the current view, which may only be a crop.
            self.whole_sheet_encoder = nn.Sequential(
                _ConvNormAct(self.config.input_channels, c1, stride=2),
                _EncoderBlock(c1, c2),
                _EncoderBlock(c2, c4),
            )
            self.whole_sheet_global_projection = nn.Conv2d(c4, c4, 1)
            self.whole_sheet_location_projection = nn.Conv2d(c4, c4, 1)
            self.crop_context = nn.Sequential(
                nn.Linear(self.config.crop_context_features, c4),
                nn.SiLU(inplace=True),
                nn.Linear(c4, c4),
            )
            self.up3 = _DecoderBlock(c4, c3, c3)
            self.up2 = _DecoderBlock(c3, c2, c2)
            self.up1 = _DecoderBlock(c2, c1, c1)
            self.topology_head = nn.Conv2d(c1, len(self.config.output_channels), 1)
            self.room_semantic_head = nn.Conv2d(c1, len(self.config.room_classes), 1)
            self.element_semantic_head = nn.Conv2d(c1, len(self.config.element_classes), 1)
            self.element_geometry_head = nn.Conv2d(
                c1,
                len(self.config.element_geometry_channels),
                1,
            )
            self.uncertainty_head = nn.Sequential(
                _DepthwiseResidual(c1),
                nn.Conv2d(c1, 3, 1),
            )
            self._initialize()

        def _initialize(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(self.topology_head.bias, -3.0)
            nn.init.constant_(self.room_semantic_head.bias, 0.0)
            nn.init.constant_(self.element_semantic_head.bias, 0.0)
            nn.init.constant_(self.element_geometry_head.bias, 0.0)
            nn.init.constant_(self.uncertainty_head[-1].bias, 1.0)
            # Warm-start additions must be identity-neutral.  Random context
            # residuals would destroy a qualified shared encoder before the new
            # branch has learned anything.
            nn.init.zeros_(self.whole_sheet_global_projection.weight)
            nn.init.zeros_(self.whole_sheet_global_projection.bias)
            nn.init.zeros_(self.whole_sheet_location_projection.weight)
            nn.init.zeros_(self.whole_sheet_location_projection.bias)
            nn.init.zeros_(self.crop_context[-1].weight)
            nn.init.zeros_(self.crop_context[-1].bias)

        def _forward_program(
            self,
            evidence: Any,
            whole_sheet_evidence: Any,
            crop_context: Any | None = None,
        ) -> dict[str, Any]:
            level1 = self.stem(evidence)
            level2 = self.down2(level1)
            level3 = self.down3(level2)
            level4 = self.context(self.down4(level3))
            if crop_context is None:
                crop_context = evidence.new_tensor(
                    (0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ).repeat(evidence.shape[0], 1)
            whole_sheet_features = self.whole_sheet_encoder(whole_sheet_evidence)
            whole_sheet_global = self.whole_sheet_global_projection(
                functional.adaptive_avg_pool2d(whole_sheet_features, (1, 1))
            )
            # Query the complete-sheet map at the current crop center. This is
            # the missing relation in the previous design: a local line now
            # knows which part of the building it belongs to, not merely that a
            # kitchen/table/wall exists somewhere on the page.
            crop_center = torch.stack(
                (
                    crop_context[:, 0] + crop_context[:, 2] * 0.5,
                    crop_context[:, 1] + crop_context[:, 3] * 0.5,
                ),
                dim=1,
            )
            sampling_grid = (crop_center * 2.0 - 1.0)[:, None, None, :]
            whole_sheet_location = self.whole_sheet_location_projection(
                functional.grid_sample(
                    whole_sheet_features,
                    sampling_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
            )
            whole_sheet_token = whole_sheet_global + whole_sheet_location
            level4 = (
                level4
                + whole_sheet_token
                + self.crop_context(crop_context)[:, :, None, None]
            )
            decoded = self.up1(self.up2(self.up3(level4, level3), level2), level1)
            return {
                "topology_logits": self.topology_head(decoded),
                "room_semantic_logits": self.room_semantic_head(decoded),
                "element_semantic_logits": self.element_semantic_head(decoded),
                "element_geometry": self.element_geometry_head(decoded),
                "uncertainty": torch.sigmoid(self.uncertainty_head(decoded)),
            }

        def forward(
            self,
            evidence: Any,
            whole_sheet_evidence: Any | None = None,
            crop_context: Any | None = None,
        ) -> dict[str, Any]:
            if evidence.ndim != 4 or evidence.shape[1] != self.config.input_channels:
                raise ValueError("evidence must have shape [batch, 4, height, width]")
            if whole_sheet_evidence is None:
                whole_sheet_evidence = evidence
            if (
                whole_sheet_evidence.ndim != 4
                or whole_sheet_evidence.shape != evidence.shape
            ):
                raise ValueError(
                    "whole_sheet_evidence must have the same [batch, 4, height, width] shape"
                )
            if crop_context is not None and (
                crop_context.ndim != 2
                or crop_context.shape
                != (evidence.shape[0], self.config.crop_context_features)
            ):
                raise ValueError("crop_context must have shape [batch, 8]")
            return self._forward_program(evidence, whole_sheet_evidence, crop_context)

        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())


    class GlobalTopologyStudentOnnxAdapter(nn.Module):
        def __init__(self, model: DajoongGlobalTopologyStudent) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            evidence: Any,
            whole_sheet_evidence: Any,
            crop_context: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            # ONNX tracing should capture the tensor program only.  Runtime input
            # validation remains in the public model forward method.
            output = self.model._forward_program(
                evidence,
                whole_sheet_evidence,
                crop_context,
            )
            return (
                output["topology_logits"],
                output["room_semantic_logits"],
                output["element_semantic_logits"],
                output["element_geometry"],
                output["uncertainty"],
            )

else:  # pragma: no cover

    class DajoongGlobalTopologyStudent:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")


    class GlobalTopologyStudentOnnxAdapter:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")
