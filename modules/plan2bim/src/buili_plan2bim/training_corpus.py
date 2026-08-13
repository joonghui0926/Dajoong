"""Build a fail-closed, leakage-safe index for commercial model training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .core.hashing import sha256_json
from .ground_truth import (
    GroundTruthPolicyError,
    assert_commercial_training_eligible,
    validate_ground_truth_manifest,
)


def _stable_split(group_id: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


def build_commercial_training_index(
    manifest_paths: list[str | Path],
    *,
    source_root: str | Path | None = None,
    split_seed: str = "dajoong-commercial-v1",
) -> dict[str, Any]:
    """Validate annotations and assign whole collection groups to stable splits."""

    if not manifest_paths:
        raise GroundTruthPolicyError("commercial training index requires at least one manifest")
    records: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    group_splits: dict[str, str] = {}

    for manifest_value in sorted(manifest_paths, key=lambda value: str(value)):
        manifest_path = Path(manifest_value).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_ground_truth_manifest(manifest, source_root=source_root)
        assert_commercial_training_eligible(manifest)
        source = manifest["source"]
        source_hash = str(source["image_sha256"])
        if source_hash in source_hashes:
            raise GroundTruthPolicyError(
                f"duplicate source image in training corpus: {source_hash}"
            )
        source_hashes.add(source_hash)

        group_id = str(source["collection_group_id"])
        split = _stable_split(group_id, split_seed)
        existing_split = group_splits.setdefault(group_id, split)
        if existing_split != split:
            raise GroundTruthPolicyError(f"collection group crosses dataset splits: {group_id}")
        records.append(
            {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_json(manifest),
                "source_image_sha256": source_hash,
                "collection_group_id": group_id,
                "split": split,
                "entity_count": len(manifest["entities"]),
                "annotator": manifest["visual_review"]["annotator"],
                "reviewed_on": manifest["visual_review"]["reviewed_on"],
            }
        )

    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    payload: dict[str, Any] = {
        "schema_version": "dajoong.commercial-training-index.v1",
        "ground_truth_policy": "direct_whole_sheet_visual_annotation_only",
        "split_unit": "collection_group_id",
        "split_seed": split_seed,
        "sample_count": len(records),
        "split_counts": counts,
        "records": records,
        "production_accuracy_claim": False,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload
