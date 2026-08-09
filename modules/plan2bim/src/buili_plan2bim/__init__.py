"""Public API for the standalone Dajoong drawing-to-BIM pipeline."""

from .building_pipeline import (
    BuildingConversionConfig,
    BuildingConversionResult,
    BuildingLevelInput,
    BuildingPlan2BimConverter,
)
from .core.building import (
    BuildingAssemblyConfig,
    BuildingLevelSpec,
    BuildingVerticalConnection,
    assemble_building_graph,
)
from .input_document import PreparedDrawing, prepare_drawing
from .pipeline import (
    ConversionConfig,
    ConversionError,
    ConversionResult,
    Plan2BimConverter,
    convert_image,
)
from .qualification import (
    ClaimQualification,
    DrawingComplexityProfile,
    ModelQualification,
    ModelQualifier,
    profile_drawing,
)
from .semantic_recognition import (
    OnnxFloorPlanSemanticRecognizer,
    SemanticDetection,
    SemanticRecognitionResult,
)

__all__ = [
    "ConversionConfig",
    "ConversionError",
    "ConversionResult",
    "Plan2BimConverter",
    "convert_image",
    "PreparedDrawing",
    "prepare_drawing",
    "OnnxFloorPlanSemanticRecognizer",
    "SemanticDetection",
    "SemanticRecognitionResult",
    "BuildingAssemblyConfig",
    "BuildingLevelSpec",
    "BuildingVerticalConnection",
    "assemble_building_graph",
    "BuildingConversionConfig",
    "BuildingConversionResult",
    "BuildingLevelInput",
    "BuildingPlan2BimConverter",
    "ClaimQualification",
    "DrawingComplexityProfile",
    "ModelQualification",
    "ModelQualifier",
    "profile_drawing",
]

__version__ = "0.1.0"
