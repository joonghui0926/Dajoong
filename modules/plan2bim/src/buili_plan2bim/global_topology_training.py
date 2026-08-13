"""Training path for the Method v2 whole-sheet topology student.

Synthetic programs are accepted only for pretraining and every emitted artifact
is explicitly non-production.  A separately validated commercial ground-truth
fine-tuning stage is required before promotion.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core.hashing import sha256_file, sha256_json
from .core.model.cad_evidence import (
    GLOBAL_PROGRAM_INPUT_CONTRACT,
    ORIENTED_EVIDENCE_ROTATION_CONTRACT,
    build_cad_evidence,
    letterbox_cad_evidence,
)
from .core.model.global_topology_student import (
    ELEMENT_GEOMETRY_CHANNELS,
    ELEMENT_PROGRAM_CLASSES,
    ROOM_PROGRAM_CLASSES,
    TOPOLOGY_TARGET_CHANNELS,
    DajoongGlobalTopologyStudent,
    GlobalTopologyStudentConfig,
    GlobalTopologyStudentOnnxAdapter,
)
from .synthetic_pretraining import (
    SyntheticPretrainingSample,
    assert_synthetic_pretraining_only,
    audit_synthetic_pretraining_corpus,
)
from .training_augmentation import (
    crop_dense_training_example,
    detail_crop_context,
    deterministic_detail_crop,
    deterministic_quadrant,
    rotate_dense_training_example,
    rotate_spatial_bbox,
)

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - inference installation intentionally stays small.
    torch = None
    nn = None
    functional = None
    DataLoader = None
    Dataset = object


@dataclass(frozen=True)
class TopologyTrainOptions:
    epochs: int = 12
    batch_size: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    workers: int = 0
    seed: int = 26_081_100
    device: str = "auto"
    validation_fraction: float = 0.1
    quadrant_augmentation: bool = True
    detail_window_augmentation: bool = True

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.workers < 0:
            raise ValueError("workers cannot be negative")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and one half")


def _split_records(
    records: list[dict[str, Any]],
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 2:
        raise ValueError("topology pretraining requires at least two samples")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(str(record["sample_id"]).encode()).hexdigest(),
    )
    validation_count = min(
        len(ordered) - 1,
        max(1, round(len(ordered) * validation_fraction)),
    )
    validation_ids = {str(record["sample_id"]) for record in ordered[:validation_count]}
    training = [
        record for record in records if str(record["sample_id"]) not in validation_ids
    ]
    validation = [
        record for record in records if str(record["sample_id"]) in validation_ids
    ]
    return training, validation


def _balanced_class_weights(
    pixel_counts: list[int] | tuple[int, ...] | np.ndarray,
    class_names: tuple[str, ...],
    *,
    background_weight: float,
) -> list[float]:
    """Build stable corpus-level weights without batch-to-batch class drift."""

    counts = np.asarray(pixel_counts, dtype=np.float64)
    if counts.shape != (len(class_names),):
        raise ValueError("pixel counts must match the class contract")
    if background_weight <= 0:
        raise ValueError("background_weight must be positive")
    weights = np.zeros_like(counts)
    foreground = counts[1:]
    supported = foreground > 0
    if not supported.any():
        raise ValueError("at least one foreground class must be represented")
    reference = float(np.median(foreground[supported]))
    raw = np.sqrt(reference / np.maximum(foreground, 1.0))
    raw = np.clip(raw, 0.35, 4.0)
    raw[~supported] = 0.0
    raw_mean = float(raw[supported].mean())
    weights[1:] = raw / max(raw_mean, 1e-12)
    weights[0] = background_weight
    return [float(value) for value in weights]


if torch is not None:

    class SyntheticTopologyDataset(Dataset):
        def __init__(
            self,
            source_corpus_root: str | Path,
            target_corpus_root: str | Path,
            *,
            split: str,
            validation_fraction: float,
            quadrant_augmentation: bool = False,
            detail_window_augmentation: bool = False,
            augmentation_seed: int = 26_081_100,
        ) -> None:
            if split not in {"train", "validation"}:
                raise ValueError("split must be train or validation")
            self.source_root = Path(source_corpus_root).expanduser().resolve()
            self.target_root = Path(target_corpus_root).expanduser().resolve()
            source_manifest = json.loads(
                (self.source_root / "manifest.json").read_text(encoding="utf-8")
            )
            target_manifest = json.loads(
                (self.target_root / "manifest.json").read_text(encoding="utf-8")
            )
            assert_synthetic_pretraining_only(source_manifest)
            assert_synthetic_pretraining_only(target_manifest)
            if source_manifest["sample_count"] != target_manifest["sample_count"]:
                raise ValueError("source and target synthetic corpora differ in size")
            self.target_size = int(target_manifest["target_size"])
            if target_manifest.get("input_contract") != GLOBAL_PROGRAM_INPUT_CONTRACT:
                raise ValueError("global program input/target transform contract mismatch")
            self.room_semantic_contract = str(
                target_manifest.get("room_semantic_contract") or "dense_room_v0"
            )
            training, validation = _split_records(
                list(target_manifest["records"]),
                validation_fraction,
            )
            self.records = validation if split == "validation" else training
            # Older generated manifests predate the explicit content frame.
            # Recover it once from each sealed target sidecar so detail crops
            # are sampled from the drawing, never from letterbox padding.
            for record in self.records:
                if record.get("content_bbox") is not None:
                    continue
                target_path = self.target_root / str(record["target_path"])
                sidecar = target_path.with_suffix(target_path.suffix + ".json")
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                record["content_bbox"] = payload["content_bbox"]
            self.quadrant_augmentation = (
                split == "train" and quadrant_augmentation
            )
            self.detail_window_augmentation = (
                split == "train" and detail_window_augmentation
            )
            self.augmentation_seed = augmentation_seed

        def class_pixel_counts(self, target_name: str) -> list[int]:
            if target_name == "room":
                record_key = "room_pixel_counts"
                classes = ROOM_PROGRAM_CLASSES
                array_key = "room_semantics"
            elif target_name == "element":
                record_key = "element_pixel_counts"
                classes = ELEMENT_PROGRAM_CLASSES
                array_key = "element_semantics"
            else:
                raise ValueError("target_name must be room or element")
            counts = np.zeros(len(classes), dtype=np.int64)
            for record in self.records:
                recorded = record.get(record_key)
                if isinstance(recorded, dict):
                    counts += np.asarray(
                        [int(recorded.get(name, 0)) for name in classes],
                        dtype=np.int64,
                    )
                    continue
                target_path = self.target_root / str(record["target_path"])
                with np.load(target_path) as target_file:
                    values = np.asarray(target_file[array_key], dtype=np.int64)
                counts += np.bincount(values.ravel(), minlength=len(classes))[
                    : len(classes)
                ]
            return [int(value) for value in counts]

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            sample_id = str(record["sample_id"])
            annotation_path = self.source_root / "annotations" / f"{sample_id}.json"
            annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            assert_synthetic_pretraining_only(annotation_payload)
            sample = SyntheticPretrainingSample.model_validate(annotation_payload)
            with Image.open(self.source_root / sample.image_path) as source_image:
                evidence = build_cad_evidence(source_image.convert("RGB"))
            evidence, _ = letterbox_cad_evidence(evidence, self.target_size)
            evidence = evidence[0]
            target_path = self.target_root / str(record["target_path"])
            with np.load(target_path) as target_file:
                targets = np.asarray(target_file["targets"], dtype=np.float32)
                channel_names = tuple(str(value) for value in target_file["channel_names"])
                room_semantics = np.asarray(target_file["room_semantics"], dtype=np.int64)
                room_classes = tuple(str(value) for value in target_file["room_classes"])
                element_semantics = np.asarray(
                    target_file["element_semantics"], dtype=np.int64
                )
                element_classes = tuple(
                    str(value) for value in target_file["element_classes"]
                )
                element_geometry = np.asarray(
                    target_file["element_geometry"], dtype=np.float32
                )
                element_geometry_valid = np.asarray(
                    target_file["element_geometry_valid"], dtype=np.float32
                )
                geometry_channels = tuple(
                    str(value) for value in target_file["element_geometry_channels"]
                )
            if channel_names != TOPOLOGY_TARGET_CHANNELS:
                raise ValueError("topology target channels do not match the model contract")
            if room_classes != ROOM_PROGRAM_CLASSES:
                raise ValueError("room target classes do not match the model contract")
            if element_classes != ELEMENT_PROGRAM_CLASSES:
                raise ValueError("element target classes do not match the model contract")
            if geometry_channels != ELEMENT_GEOMETRY_CHANNELS:
                raise ValueError("element geometry channels do not match the model contract")
            content_bbox = tuple(int(value) for value in record["content_bbox"])
            quadrants = 0
            if self.quadrant_augmentation:
                quadrants = deterministic_quadrant(
                    sample_id,
                    seed=self.augmentation_seed,
                )
                rotated = rotate_dense_training_example(
                    evidence=evidence,
                    topology=targets,
                    room_semantics=room_semantics,
                    element_semantics=element_semantics,
                    element_geometry=element_geometry,
                    element_geometry_valid=element_geometry_valid,
                    quadrants=quadrants,
                )
                evidence = rotated["evidence"]
                targets = rotated["topology"]
                room_semantics = rotated["room_semantics"]
                element_semantics = rotated["element_semantics"]
                element_geometry = rotated["element_geometry"]
                element_geometry_valid = rotated["element_geometry_valid"]
                content_bbox = rotate_spatial_bbox(
                    content_bbox,
                    size=(evidence.shape[-1], evidence.shape[-2]),
                    quadrants=quadrants,
                )
            whole_sheet_evidence = evidence.copy()
            detail_bbox = None
            if self.detail_window_augmentation:
                content_left, content_top, content_right, content_bottom = content_bbox
                local_detail_bbox = deterministic_detail_crop(
                    sample_id,
                    seed=self.augmentation_seed,
                    size=(content_right - content_left, content_bottom - content_top),
                )
                detail_bbox = (
                    None
                    if local_detail_bbox is None
                    else (
                        content_left + local_detail_bbox[0],
                        content_top + local_detail_bbox[1],
                        content_left + local_detail_bbox[2],
                        content_top + local_detail_bbox[3],
                    )
                )
                if detail_bbox is not None:
                    cropped = crop_dense_training_example(
                        evidence=evidence,
                        topology=targets,
                        room_semantics=room_semantics,
                        element_semantics=element_semantics,
                        element_geometry=element_geometry,
                        element_geometry_valid=element_geometry_valid,
                        bbox=detail_bbox,
                    )
                    evidence = cropped["evidence"]
                    targets = cropped["topology"]
                    room_semantics = cropped["room_semantics"]
                    element_semantics = cropped["element_semantics"]
                    element_geometry = cropped["element_geometry"]
                    element_geometry_valid = cropped["element_geometry_valid"]
            crop_context = detail_crop_context(
                detail_bbox,
                size=(whole_sheet_evidence.shape[-1], whole_sheet_evidence.shape[-2]),
                frame_bbox=content_bbox,
            )
            return {
                "sample_id": sample_id,
                "evidence": torch.from_numpy(evidence.copy()),
                "whole_sheet_evidence": torch.from_numpy(
                    whole_sheet_evidence.copy()
                ),
                "crop_context": torch.from_numpy(crop_context.copy()),
                "targets": torch.from_numpy(targets.copy()),
                "room_semantics": torch.from_numpy(room_semantics.copy()),
                "element_semantics": torch.from_numpy(element_semantics.copy()),
                "element_geometry": torch.from_numpy(element_geometry.copy()),
                "element_geometry_valid": torch.from_numpy(
                    element_geometry_valid.copy()
                ),
            }


    class GlobalTopologyCriterion(nn.Module):
        """Pixel evidence plus differentiable building-program consistency."""

        def __init__(
            self,
            *,
            room_weights: list[float] | tuple[float, ...] | None = None,
            element_weights: list[float] | tuple[float, ...] | None = None,
        ) -> None:
            super().__init__()
            room_weights = room_weights or [1.0] * len(ROOM_PROGRAM_CLASSES)
            if element_weights is None:
                element_weights = [0.08, *([1.0] * (len(ELEMENT_PROGRAM_CLASSES) - 1))]
            if len(room_weights) != len(ROOM_PROGRAM_CLASSES):
                raise ValueError("room weights do not match the class contract")
            if len(element_weights) != len(ELEMENT_PROGRAM_CLASSES):
                raise ValueError("element weights do not match the class contract")
            self.register_buffer(
                "room_weights",
                torch.as_tensor(room_weights, dtype=torch.float32),
            )
            self.register_buffer(
                "element_weights",
                torch.as_tensor(element_weights, dtype=torch.float32),
            )

        def forward(self, output: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
            logits = output["topology_logits"]
            topology_targets = targets["topology"]
            if logits.shape != topology_targets.shape:
                raise ValueError("model output and topology targets must have equal shape")
            probabilities = torch.sigmoid(logits)
            positive = topology_targets.sum(dim=(0, 2, 3))
            negative = topology_targets.numel() / topology_targets.shape[1] - positive
            positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 30.0)
            binary = functional.binary_cross_entropy_with_logits(
                logits,
                topology_targets,
                pos_weight=positive_weight.view(1, -1, 1, 1),
            )
            intersection = (probabilities * topology_targets).sum(dim=(0, 2, 3))
            denominator = (probabilities + topology_targets).sum(dim=(0, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

            exterior = probabilities[:, 0:1]
            wall = probabilities[:, 1:2]
            opening = probabilities[:, 3:4]
            room_seed = probabilities[:, 4:5]
            room_interior = probabilities[:, 5:6]
            wall_neighborhood = functional.max_pool2d(wall, kernel_size=9, stride=1, padding=4)
            exterior_subset = functional.relu(exterior - wall).mean()
            unhosted_opening = (opening * (1.0 - wall_neighborhood)).mean()
            seed_outside_room = (room_seed * (1.0 - room_interior)).mean()
            topology = exterior_subset + unhosted_opening + seed_outside_room
            room_semantics = functional.cross_entropy(
                output["room_semantic_logits"],
                targets["room_semantics"],
                weight=self.room_weights,
            )
            element_semantics = functional.cross_entropy(
                output["element_semantic_logits"],
                targets["element_semantics"],
                weight=self.element_weights,
            )
            geometry_valid = targets["element_geometry_valid"].unsqueeze(1)
            geometry_denominator = geometry_valid.sum() * len(ELEMENT_GEOMETRY_CHANNELS)
            geometry = (
                functional.smooth_l1_loss(
                    output["element_geometry"] * geometry_valid,
                    targets["element_geometry"] * geometry_valid,
                    reduction="sum",
                )
                / geometry_denominator.clamp_min(1.0)
            )
            total = (
                binary
                + dice
                + topology * 1.5
                + room_semantics * 0.5
                + element_semantics
                + geometry * 0.25
            )
            return {
                "total": total,
                "binary": binary,
                "dice": dice,
                "topology": topology,
                "room_semantics": room_semantics,
                "element_semantics": element_semantics,
                "element_geometry": geometry,
            }

else:  # pragma: no cover

    class SyntheticTopologyDataset:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")


    class GlobalTopologyCriterion:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install the plan2bim training dependencies")


def _device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: str,
    optimizer: Any | None,
    progress_label: str = "",
    progress_interval: int = 25,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    topology_tp = topology_fp = topology_fn = None
    room_confusion = element_confusion = None
    if not training:
        topology_tp = torch.zeros(len(TOPOLOGY_TARGET_CHANNELS), dtype=torch.int64)
        topology_fp = torch.zeros(len(TOPOLOGY_TARGET_CHANNELS), dtype=torch.int64)
        topology_fn = torch.zeros(len(TOPOLOGY_TARGET_CHANNELS), dtype=torch.int64)
        room_confusion = torch.zeros(
            (len(ROOM_PROGRAM_CLASSES), len(ROOM_PROGRAM_CLASSES)),
            dtype=torch.int64,
        )
        element_confusion = torch.zeros(
            (len(ELEMENT_PROGRAM_CLASSES), len(ELEMENT_PROGRAM_CLASSES)),
            dtype=torch.int64,
        )
    batches = 0
    for batch in loader:
        evidence = batch["evidence"].to(device)
        whole_sheet_evidence = batch["whole_sheet_evidence"].to(device)
        crop_context = batch["crop_context"].to(device)
        targets = {
            "topology": batch["targets"].to(device),
            "room_semantics": batch["room_semantics"].to(device),
            "element_semantics": batch["element_semantics"].to(device),
            "element_geometry": batch["element_geometry"].to(device),
            "element_geometry_valid": batch["element_geometry_valid"].to(device),
        }
        with torch.set_grad_enabled(training):
            output = model(evidence, whole_sheet_evidence, crop_context)
            losses = criterion(output, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if topology_tp is not None:
            prediction = (
                torch.sigmoid(output["topology_logits"]).detach().cpu() >= 0.5
            )
            truth = targets["topology"].detach().cpu() >= 0.5
            reduce_dims = (0, 2, 3)
            topology_tp += (prediction & truth).sum(dim=reduce_dims)
            topology_fp += (prediction & ~truth).sum(dim=reduce_dims)
            topology_fn += (~prediction & truth).sum(dim=reduce_dims)
            for output_name, target_name, confusion, classes in (
                (
                    "room_semantic_logits",
                    "room_semantics",
                    room_confusion,
                    ROOM_PROGRAM_CLASSES,
                ),
                (
                    "element_semantic_logits",
                    "element_semantics",
                    element_confusion,
                    ELEMENT_PROGRAM_CLASSES,
                ),
            ):
                class_count = len(classes)
                predicted_class = output[output_name].argmax(dim=1).detach().cpu()
                target_class = targets[target_name].detach().cpu()
                confusion += torch.bincount(
                    target_class.reshape(-1) * class_count
                    + predicted_class.reshape(-1),
                    minlength=class_count * class_count,
                ).reshape(class_count, class_count)
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        batches += 1
        if progress_label and batches % progress_interval == 0:
            print(
                f"{progress_label}: {batches}/{len(loader)} batches",
                flush=True,
            )
    metrics = {name: value / max(1, batches) for name, value in totals.items()}
    if topology_tp is not None:
        topology_f1 = _binary_macro_f1(topology_tp, topology_fp, topology_fn)
        room_f1 = _non_background_macro_f1(room_confusion)
        element_f1 = _non_background_macro_f1(element_confusion)
        scores = (topology_f1, room_f1, element_f1)
        metrics.update(
            {
                "topology_macro_f1": topology_f1,
                "room_macro_f1_non_background": room_f1,
                "element_macro_f1_non_background": element_f1,
                "quality_score": (min(scores) + sum(scores) / len(scores)) / 2,
            }
        )
    return metrics


def _binary_macro_f1(tp: Any, fp: Any, fn: Any) -> float:
    values = []
    for index in range(len(tp)):
        true_positive = int(tp[index])
        denominator = 2 * true_positive + int(fp[index]) + int(fn[index])
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / max(1, len(values))


def _non_background_macro_f1(confusion: Any) -> float:
    values = []
    for index in range(1, int(confusion.shape[0])):
        support = int(confusion[index, :].sum())
        if support == 0:
            continue
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / max(1, len(values))


def _warm_start_global_model(
    model: Any,
    payload: dict[str, Any],
    target_config: GlobalTopologyStudentConfig,
) -> dict[str, int]:
    """Transfer shared topology features and class rows across taxonomy growth."""

    source_config = dict(payload.get("config") or {})
    target_payload = target_config.to_dict()
    compatible_fields = (
        "input_channels",
        "stem_width",
        "output_channels",
        "room_classes",
        "element_geometry_channels",
        "output_stride",
        "context_grids",
    )
    if any(source_config.get(key) != target_payload.get(key) for key in compatible_fields):
        raise ValueError("initial checkpoint global architecture is incompatible")
    source_state = payload["state_dict"]
    target_state = model.state_dict()
    transferred_tensors = 0
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
            transferred_tensors += 1
    source_classes = tuple(source_config.get("element_classes") or ())
    shared_classes = set(source_classes) & set(target_config.element_classes)
    for suffix in ("weight", "bias"):
        name = f"element_semantic_head.{suffix}"
        if name not in source_state or name not in target_state:
            continue
        for class_name in shared_classes:
            target_index = target_config.element_classes.index(class_name)
            source_index = source_classes.index(class_name)
            target_state[name][target_index] = source_state[name][source_index]
    model.load_state_dict(target_state)
    return {
        "transferred_tensor_count": transferred_tensors,
        "transferred_element_class_count": len(shared_classes),
    }


def train_global_topology_student(
    source_corpus_root: str | Path,
    target_corpus_root: str | Path,
    output_root: str | Path,
    *,
    model_config: GlobalTopologyStudentConfig | None = None,
    options: TopologyTrainOptions | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None or DataLoader is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    options = options or TopologyTrainOptions()
    options.validate()
    model_config = model_config or GlobalTopologyStudentConfig()
    model_config.validate()
    audit_synthetic_pretraining_corpus(
        source_corpus_root,
        require_complete_taxonomy=True,
        raise_on_error=True,
    )
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)
    device = _device(options.device)
    train_data = SyntheticTopologyDataset(
        source_corpus_root,
        target_corpus_root,
        split="train",
        validation_fraction=options.validation_fraction,
        quadrant_augmentation=options.quadrant_augmentation,
        detail_window_augmentation=options.detail_window_augmentation,
        augmentation_seed=options.seed,
    )
    validation_data = SyntheticTopologyDataset(
        source_corpus_root,
        target_corpus_root,
        split="validation",
        validation_fraction=options.validation_fraction,
        quadrant_augmentation=False,
        detail_window_augmentation=False,
        augmentation_seed=options.seed,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.workers,
    )
    model = DajoongGlobalTopologyStudent(model_config).to(device)
    initialization: dict[str, Any] = {
        "initial_checkpoint_sha256": "",
        "initial_checkpoint_epoch": 0,
        "transferred_tensor_count": 0,
        "transferred_element_class_count": 0,
    }
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint).expanduser().resolve()
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        if initial_payload.get("role") != "synthetic_pretrain_only":
            raise ValueError("initial checkpoint must be synthetic pretraining only")
        if (
            initial_payload.get("oriented_evidence_rotation_contract")
            != ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ):
            raise ValueError(
                "initial checkpoint used a stale oriented-evidence rotation contract"
            )
        transfer = _warm_start_global_model(model, initial_payload, model_config)
        initialization = {
            "initial_checkpoint_sha256": sha256_file(initial_path),
            "initial_checkpoint_epoch": int(initial_payload.get("epoch") or 0),
            **transfer,
        }
    room_counts = train_data.class_pixel_counts("room")
    element_counts = train_data.class_pixel_counts("element")
    room_weights = _balanced_class_weights(
        room_counts,
        ROOM_PROGRAM_CLASSES,
        background_weight=0.03,
    )
    element_weights = _balanced_class_weights(
        element_counts,
        ELEMENT_PROGRAM_CLASSES,
        background_weight=0.03,
    )
    criterion = GlobalTopologyCriterion(
        room_weights=room_weights,
        element_weights=element_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best.pt"
    history = []
    best_validation = float("inf")
    best_validation_quality = -1.0
    started = time.time()
    for epoch in range(options.epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            progress_label=f"epoch {epoch + 1}/{options.epochs} train",
        )
        with torch.inference_mode():
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                criterion,
                device=device,
                optimizer=None,
                progress_label=f"epoch {epoch + 1}/{options.epochs} validation",
            )
        history.append(
            {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics}
        )
        validation_quality = validation_metrics["quality_score"]
        improves_quality = validation_quality > best_validation_quality + 1e-9
        ties_quality_with_lower_loss = (
            abs(validation_quality - best_validation_quality) <= 1e-9
            and validation_metrics["total"] < best_validation
        )
        if improves_quality or ties_quality_with_lower_loss:
            best_validation = validation_metrics["total"]
            best_validation_quality = validation_quality
            torch.save(
                {
                    "schema_version": "dajoong.global-program-checkpoint.v2",
                    "role": "synthetic_pretrain_only",
                    "real_drawing_ground_truth": False,
                    "evaluation_eligible": False,
                    "production_authorized": False,
                    "config": model_config.to_dict(),
                    "room_semantic_contract": train_data.room_semantic_contract,
                    "oriented_evidence_rotation_contract": (
                        ORIENTED_EVIDENCE_ROTATION_CONTRACT
                    ),
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": best_validation,
                    "validation_quality_score": best_validation_quality,
                },
                checkpoint_path,
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model = model.cpu().eval()
    onnx_path = destination / "global-topology-student.onnx"
    target_size = train_data.target_size
    torch.onnx.export(
        GlobalTopologyStudentOnnxAdapter(model),
        (
            torch.zeros(1, model_config.input_channels, target_size, target_size),
            torch.zeros(1, model_config.input_channels, target_size, target_size),
            torch.tensor(((0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),)),
        ),
        onnx_path,
        input_names=["view_evidence", "whole_sheet_evidence", "crop_context"],
        output_names=[
            "topology_logits",
            "room_semantic_logits",
            "element_semantic_logits",
            "element_geometry",
            "uncertainty",
        ],
        dynamic_axes={
            "view_evidence": {0: "batch"},
            "whole_sheet_evidence": {0: "batch"},
            "crop_context": {0: "batch"},
            "topology_logits": {0: "batch"},
            "room_semantic_logits": {0: "batch"},
            "element_semantic_logits": {0: "batch"},
            "element_geometry": {0: "batch"},
            "uncertainty": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_manifest: dict[str, Any] = {
        "schema_version": "dajoong.global-program-onnx.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "model_version": model_config.model_version,
        "artifact": onnx_path.name,
        "artifact_sha256": sha256_file(onnx_path),
        "input_name": "view_evidence",
        "input_names": ["view_evidence", "whole_sheet_evidence", "crop_context"],
        "whole_sheet_context_contract": "explicit_complete_sheet_evidence_v1",
        "crop_context_contract": "normalized_origin_extent_sheet_edges_v1",
        "input_contract": GLOBAL_PROGRAM_INPUT_CONTRACT,
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "input_size": [target_size, target_size],
        "onnx_opset": 17,
        "output_names": [
            "topology_logits",
            "room_semantic_logits",
            "element_semantic_logits",
            "element_geometry",
            "uncertainty",
        ],
        "topology_channels": list(TOPOLOGY_TARGET_CHANNELS),
        "room_classes": list(ROOM_PROGRAM_CLASSES),
        "element_classes": list(ELEMENT_PROGRAM_CLASSES),
        "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "decoder": "dajoong.global-program-decode.v1",
        "room_semantic_contract": train_data.room_semantic_contract,
        "promotion_requirements": [
            "commercial-rights direct whole-sheet visual labels",
            "fixed holdout class and topology qualification",
            "immutable artifact hash registration",
        ],
    }
    onnx_manifest["content_sha256"] = sha256_json(onnx_manifest)
    onnx_manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    onnx_manifest_path.write_text(
        json.dumps(onnx_manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report: dict[str, Any] = {
        "schema_version": "dajoong.global-program-pretraining.v2",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "fine_tuning_required": "commercial-rights direct whole-sheet visual labels",
        "model_config": model_config.to_dict(),
        "room_semantic_contract": train_data.room_semantic_contract,
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "train_options": asdict(options),
        **initialization,
        "parameter_count": model.parameter_count(),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "training_room_pixel_counts": dict(
            zip(ROOM_PROGRAM_CLASSES, room_counts, strict=True)
        ),
        "training_element_pixel_counts": dict(
            zip(ELEMENT_PROGRAM_CLASSES, element_counts, strict=True)
        ),
        "room_class_weights": dict(
            zip(ROOM_PROGRAM_CLASSES, room_weights, strict=True)
        ),
        "element_class_weights": dict(
            zip(ELEMENT_PROGRAM_CLASSES, element_weights, strict=True)
        ),
        "best_validation_loss": best_validation,
        "best_validation_quality_score": best_validation_quality,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_manifest_sha256": sha256_file(onnx_manifest_path),
        "duration_seconds": time.time() - started,
        "history": history,
    }
    report["content_sha256"] = sha256_json(report)
    assert_synthetic_pretraining_only(report)
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def export_synthetic_topology_checkpoint(
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    target_size: int = 256,
) -> dict[str, Any]:
    """Export a completed synthetic checkpoint without repeating model training."""

    if torch is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("role") != "synthetic_pretrain_only":
        raise ValueError("only synthetic pretraining checkpoints may use this exporter")
    config = GlobalTopologyStudentConfig()
    if payload.get("config") != config.to_dict():
        raise ValueError("checkpoint model contract does not match the current exporter")
    if (
        payload.get("oriented_evidence_rotation_contract")
        != ORIENTED_EVIDENCE_ROTATION_CONTRACT
    ):
        raise ValueError("checkpoint oriented-evidence rotation contract mismatch")
    model = DajoongGlobalTopologyStudent(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    onnx_path = destination / "global-topology-student.onnx"
    output_names = [
        "topology_logits",
        "room_semantic_logits",
        "element_semantic_logits",
        "element_geometry",
        "uncertainty",
    ]
    torch.onnx.export(
        GlobalTopologyStudentOnnxAdapter(model),
        (
            torch.zeros(1, config.input_channels, target_size, target_size),
            torch.zeros(1, config.input_channels, target_size, target_size),
            torch.tensor(((0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),)),
        ),
        onnx_path,
        input_names=["view_evidence", "whole_sheet_evidence", "crop_context"],
        output_names=output_names,
        dynamic_axes={
            "view_evidence": {0: "batch"},
            "whole_sheet_evidence": {0: "batch"},
            "crop_context": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    manifest: dict[str, Any] = {
        "schema_version": "dajoong.global-program-onnx.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "production_accuracy_claim": False,
        "model_version": config.model_version,
        "artifact": onnx_path.name,
        "artifact_sha256": sha256_file(onnx_path),
        "input_name": "view_evidence",
        "input_names": ["view_evidence", "whole_sheet_evidence", "crop_context"],
        "whole_sheet_context_contract": "explicit_complete_sheet_evidence_v1",
        "crop_context_contract": "normalized_origin_extent_sheet_edges_v1",
        "input_contract": GLOBAL_PROGRAM_INPUT_CONTRACT,
        "oriented_evidence_rotation_contract": (
            ORIENTED_EVIDENCE_ROTATION_CONTRACT
        ),
        "input_size": [target_size, target_size],
        "onnx_opset": 17,
        "output_names": output_names,
        "topology_channels": list(TOPOLOGY_TARGET_CHANNELS),
        "room_classes": list(ROOM_PROGRAM_CLASSES),
        "element_classes": list(ELEMENT_PROGRAM_CLASSES),
        "element_geometry_channels": list(ELEMENT_GEOMETRY_CHANNELS),
        "decoder": "dajoong.global-program-decode.v1",
        "room_semantic_contract": str(
            payload.get("room_semantic_contract") or "dense_room_v0"
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_validation_loss": float(payload["validation_loss"]),
        "promotion_requirements": [
            "commercial-rights direct whole-sheet visual labels",
            "fixed holdout class and topology qualification",
            "immutable artifact hash registration",
        ],
    }
    manifest["content_sha256"] = sha256_json(manifest)
    manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def evaluate_synthetic_topology_checkpoint(
    checkpoint_path: str | Path,
    source_corpus_root: str | Path,
    target_corpus_root: str | Path,
    *,
    batch_size: int = 8,
    validation_fraction: float = 0.1,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure a synthetic-only checkpoint without implying real-drawing quality."""

    if torch is None or DataLoader is None:
        raise RuntimeError("Install the plan2bim training dependencies")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("role") != "synthetic_pretrain_only":
        raise ValueError("this evaluator accepts synthetic pretraining checkpoints only")
    config = GlobalTopologyStudentConfig()
    if payload.get("config") != config.to_dict():
        raise ValueError("checkpoint model contract does not match the current evaluator")
    dataset = SyntheticTopologyDataset(
        source_corpus_root,
        target_corpus_root,
        split="validation",
        validation_fraction=validation_fraction,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = DajoongGlobalTopologyStudent(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    topology_tp = torch.zeros(len(TOPOLOGY_TARGET_CHANNELS), dtype=torch.float64)
    topology_fp = torch.zeros_like(topology_tp)
    topology_fn = torch.zeros_like(topology_tp)
    room_confusion = torch.zeros(
        (len(ROOM_PROGRAM_CLASSES), len(ROOM_PROGRAM_CLASSES)), dtype=torch.int64
    )
    element_confusion = torch.zeros(
        (len(ELEMENT_PROGRAM_CLASSES), len(ELEMENT_PROGRAM_CLASSES)), dtype=torch.int64
    )
    geometry_absolute_error = 0.0
    geometry_value_count = 0

    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["evidence"].to(device),
                batch["whole_sheet_evidence"].to(device),
                batch["crop_context"].to(device),
            )
            topology_prediction = (output["topology_logits"].sigmoid() >= 0.5).cpu()
            topology_target = batch["targets"] >= 0.5
            topology_tp += (topology_prediction & topology_target).sum(
                dim=(0, 2, 3)
            )
            topology_fp += (topology_prediction & ~topology_target).sum(
                dim=(0, 2, 3)
            )
            topology_fn += (~topology_prediction & topology_target).sum(
                dim=(0, 2, 3)
            )

            for logits_name, target_name, confusion in (
                ("room_semantic_logits", "room_semantics", room_confusion),
                ("element_semantic_logits", "element_semantics", element_confusion),
            ):
                prediction = output[logits_name].argmax(dim=1).cpu().reshape(-1)
                target = batch[target_name].reshape(-1)
                class_count = confusion.shape[0]
                counts = torch.bincount(
                    target * class_count + prediction,
                    minlength=class_count * class_count,
                ).reshape(class_count, class_count)
                confusion += counts

            valid = batch["element_geometry_valid"].unsqueeze(1).to(device)
            absolute_error = (
                (output["element_geometry"] - batch["element_geometry"].to(device)).abs()
                * valid
            )
            geometry_absolute_error += float(absolute_error.sum().cpu())
            geometry_value_count += int(valid.sum().cpu()) * len(ELEMENT_GEOMETRY_CHANNELS)

    def class_metrics(
        confusion: Any,
        class_names: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        rows = []
        for index, class_name in enumerate(class_names):
            true_positive = int(confusion[index, index])
            false_positive = int(confusion[:, index].sum()) - true_positive
            false_negative = int(confusion[index, :].sum()) - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            rows.append(
                {
                    "class_name": class_name,
                    "support_pixels": int(confusion[index, :].sum()),
                    "true_positive_pixels": true_positive,
                    "false_positive_pixels": false_positive,
                    "false_negative_pixels": false_negative,
                    "f1": 0.0 if denominator == 0 else 2 * true_positive / denominator,
                }
            )
        return rows

    topology_rows = []
    for index, name in enumerate(TOPOLOGY_TARGET_CHANNELS):
        denominator = 2 * topology_tp[index] + topology_fp[index] + topology_fn[index]
        topology_rows.append(
            {
                "channel": name,
                "f1": 0.0 if denominator == 0 else float(2 * topology_tp[index] / denominator),
                "true_positive_pixels": int(topology_tp[index]),
                "false_positive_pixels": int(topology_fp[index]),
                "false_negative_pixels": int(topology_fn[index]),
            }
        )
    room_rows = class_metrics(room_confusion, ROOM_PROGRAM_CLASSES)
    element_rows = class_metrics(element_confusion, ELEMENT_PROGRAM_CLASSES)

    def supported_macro(rows: list[dict[str, Any]]) -> float:
        supported = [row["f1"] for row in rows[1:] if row["support_pixels"] > 0]
        return sum(supported) / max(1, len(supported))

    return {
        "schema_version": "dajoong.synthetic-topology-evaluation.v1",
        "role": "synthetic_pretrain_only",
        "real_drawing_ground_truth": False,
        "evaluation_eligible": False,
        "production_authorized": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "validation_sample_count": len(dataset),
        "room_semantic_contract": dataset.room_semantic_contract,
        "topology": topology_rows,
        "topology_macro_f1": sum(row["f1"] for row in topology_rows)
        / len(topology_rows),
        "room_classes": room_rows,
        "room_macro_f1_non_background": supported_macro(room_rows),
        "element_classes": element_rows,
        "element_macro_f1_non_background": supported_macro(element_rows),
        "element_geometry_mae": geometry_absolute_error
        / max(1, geometry_value_count),
    }
