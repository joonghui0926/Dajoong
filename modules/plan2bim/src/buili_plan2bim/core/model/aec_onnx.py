from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..hashing import sha256_file, sha256_json
from .cad_evidence import GLOBAL_ORIENTED_EVIDENCE_CONTRACT

if TYPE_CHECKING:
    from .aec_specialist import DajoongAecSpecialist

OUTPUT_NAMES = ("structure_logits", "symbol_logits", "metric_offsets", "uncertainty")


def export_aec_specialist_onnx(
    model: DajoongAecSpecialist,
    output_path: Path,
    *,
    tile_size: int = 256,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    import torch

    from .aec_specialist import AecSpecialistOnnxAdapter

    if tile_size < 64 or tile_size % 8:
        raise ValueError("tile_size must be divisible by eight and at least 64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    adapter = AecSpecialistOnnxAdapter(model).eval()
    example = torch.zeros(1, model.config.input_channels, tile_size, tile_size)
    torch.onnx.export(
        adapter,
        (example,),
        output_path,
        input_names=["evidence"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            "evidence": {0: "batch", 2: "height", 3: "width"},
            **{name: {0: "batch", 2: "height", 3: "width"} for name in OUTPUT_NAMES},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    input_contract = (
        ["raster_ink", "horizontal_line_support", "vertical_line_support", "global_enclosure"]
        if model.config.evidence_contract == GLOBAL_ORIENTED_EVIDENCE_CONTRACT
        else ["raster_ink", "native_vector_ink", "ocr_text_mask", "active_mask"]
    )
    manifest = {
        "schema_version": "dajoong.aec-specialist-onnx.v1",
        "model_version": model.config.model_version,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": sha256_file(output_path),
        "input_contract": input_contract,
        "output_names": list(OUTPUT_NAMES),
        "authoritative_decisions": False,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def quantize_aec_specialist_int8(
    fp32_path: Path,
    int8_path: Path,
    calibration_batches: list[np.ndarray],
) -> dict[str, Any]:
    if not calibration_batches:
        raise ValueError("at least one calibration batch is required")
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, batches: list[np.ndarray]) -> None:
            self._batches = iter(batches)

        def get_next(self) -> dict[str, np.ndarray] | None:
            try:
                batch = next(self._batches)
            except StopIteration:
                return None
            array = np.asarray(batch, dtype=np.float32)
            if array.ndim != 4 or array.shape[1] != 4:
                raise ValueError("calibration batch must be [batch, 4, height, width]")
            return {"evidence": array}

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32_path),
        str(int8_path),
        _Reader(calibration_batches),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    manifest = {
        "schema_version": "dajoong.aec-specialist-quantization.v1",
        "source_onnx_sha256": sha256_file(fp32_path),
        "int8_onnx_sha256": sha256_file(int8_path),
        "format": "QDQ-S8S8-per-channel",
        "calibration_batches": len(calibration_batches),
        "fp32_bytes": fp32_size,
        "int8_bytes": int8_size,
        "compression_ratio": fp32_size / max(int8_size, 1),
        "accuracy_revalidation_required": True,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    int8_path.with_suffix(int8_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class OnnxAecSpecialist:
    def __init__(self, model_path: Path, *, threads: int = 1) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def infer(self, evidence: np.ndarray) -> dict[str, np.ndarray]:
        array = np.asarray(evidence, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 4:
            raise ValueError("evidence must be float32 [batch, 4, height, width]")
        outputs = self.session.run(list(OUTPUT_NAMES), {"evidence": array})
        return dict(zip(OUTPUT_NAMES, outputs, strict=True))

    def benchmark(
        self,
        evidence: np.ndarray,
        *,
        warmup: int = 5,
        runs: int = 25,
    ) -> dict[str, float | int]:
        if warmup < 0 or runs < 1:
            raise ValueError("warmup must be nonnegative and runs must be positive")
        for _ in range(warmup):
            self.infer(evidence)
        durations = []
        for _ in range(runs):
            started = time.perf_counter()
            self.infer(evidence)
            durations.append((time.perf_counter() - started) * 1000)
        ordered = sorted(durations)
        p95_index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * 0.95)) - 1))
        return {
            "runs": runs,
            "median_ms": statistics.median(durations),
            "p95_ms": ordered[p95_index],
            "minimum_ms": min(durations),
            "maximum_ms": max(durations),
        }
