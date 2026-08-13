from __future__ import annotations

import argparse
import json

from buili_plan2bim.global_topology_training import (
    export_synthetic_topology_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a completed synthetic topology checkpoint to ONNX."
    )
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_root")
    parser.add_argument("--target-size", type=int, default=256)
    args = parser.parse_args()
    manifest = export_synthetic_topology_checkpoint(
        args.checkpoint_path,
        args.output_root,
        target_size=args.target_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
