"""Export the matched whole-sheet and native-detail checkpoints to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buili_plan2bim.core.hashing import sha256_file, sha256_json
from buili_plan2bim.global_topology_training import export_synthetic_topology_checkpoint
from buili_plan2bim.local_element_training import export_synthetic_local_element_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("global_checkpoint", type=Path)
    parser.add_argument("local_checkpoint", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--global-input-size", type=int, default=384)
    args = parser.parse_args()

    global_root = args.output_root / "global"
    local_root = args.output_root / "local"
    global_manifest = export_synthetic_topology_checkpoint(
        args.global_checkpoint,
        global_root,
        target_size=args.global_input_size,
    )
    local_manifest = export_synthetic_local_element_checkpoint(
        args.local_checkpoint,
        local_root,
    )
    global_path = global_root / str(global_manifest["artifact"])
    local_path = local_root / str(local_manifest["artifact"])
    pair = {
        "schema_version": "dajoong.architectural-checkpoint-pair.v1",
        "role": "synthetic_pretrain_only",
        "production_authorized": False,
        "production_accuracy_claim": False,
        "global": {
            "path": str(global_path.resolve()),
            "sha256": sha256_file(global_path),
            "manifest_path": str(
                global_path.with_suffix(global_path.suffix + ".json").resolve()
            ),
            "checkpoint_sha256": global_manifest["checkpoint_sha256"],
        },
        "local": {
            "path": str(local_path.resolve()),
            "sha256": sha256_file(local_path),
            "manifest_path": str(
                local_path.with_suffix(local_path.suffix + ".json").resolve()
            ),
            "checkpoint_sha256": local_manifest["checkpoint_sha256"],
        },
        "execution_contract": "whole_sheet_global_then_native_resolution_local",
    }
    pair["content_sha256"] = sha256_json(pair)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pair-manifest.json").write_text(
        json.dumps(pair, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(pair, indent=2))


if __name__ == "__main__":
    main()
