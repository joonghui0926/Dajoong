"""Delete only explicitly ephemeral Plan2BIM training directories.

The script is dry-run by default. It never removes direct ground truth, model
checkpoints, evaluation reports, asset packs, or arbitrary user paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

EPHEMERAL_PREFIXES = (
    "architectural-routing-smoke",
    "direct-real-local-element-corpus-",
    "global-program-v8-",
    "local-element-corpus-",
    "outline-",
    "plan2bim-current-predictions-",
    "relations-",
    "structural-v8-detail-candidates-",
    "structural-v8-directgt-cubi014",
    "structural-v8-framing-",
    "structural-v8-objectness-",
    "synthetic-global-program-target-smoke-",
    "topology-source-",
    "topology-source-smoke-",
    "topology-targets-",
    "topology-targets-smoke-",
    "training-onnx-deps",
    "training-python-deps",
)
PROTECTED_EXACT_NAMES = {
    # Canonical geometry audit referenced by ACTIVE_RUNTIME.json.
    "structural-v8-directgt-cubi014-wallopeningv5",
}
PROTECTED_MARKERS = (
    "ground-truth",
    "direct-source",
    "student",
    "evaluation",
    "asset",
)
TMP_DIAGNOSTIC_PREFIXES = (
    "c020-manual-furniture-review-aid",
    "c020-missed-candidate-overlay",
    "cubi-014-layout-audit",
    "cubi-020-layout-audit",
    "cubi020-",
    "real-24-candidate-load-audit",
    "real-24-layout-visibility-audit",
    "wall-fill-",
    "wallmask-",
)
CURRENT_RESEARCH_SCRATCH_NAMES = {
    "architectural-pair-v22-v25-research",
    "pipeline-v22-v25-cubi014",
    "pipeline-v22-v25-cubi014-evaluation",
    "v16-local-corpus-64",
    "v16-synthetic-source-64",
    "logo-vector-trace",
    "logo-vector-trace-exact",
    "resvg-runtime",
    "npm-cache",
}


def _ephemeral_children(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    output = []
    for child in root.iterdir() if root.exists() else ():
        name = child.name.lower()
        if (
            not child.is_dir()
            or name in PROTECTED_EXACT_NAMES
            or any(marker in name for marker in PROTECTED_MARKERS)
        ):
            continue
        if name.startswith(EPHEMERAL_PREFIXES):
            output.append(child.resolve())
    return sorted(output)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _tmp_diagnostic_files(root: Path) -> list[Path]:
    """Return only named, top-level disposable diagnostics under repo/tmp."""

    root = root.expanduser().resolve()
    if root.name.lower() != "tmp":
        raise RuntimeError("diagnostic purge is limited to a directory named tmp")
    return sorted(
        child.resolve()
        for child in root.iterdir()
        if child.is_file() and child.name.lower().startswith(TMP_DIAGNOSTIC_PREFIXES)
    )


def _current_research_scratch(root: Path) -> list[Path]:
    """Return only completed, summarized research runs under repo/tmp."""

    root = root.expanduser().resolve()
    if root.name.lower() != "tmp":
        raise RuntimeError("research scratch purge is limited to a directory named tmp")
    return sorted(
        child.resolve()
        for child in root.iterdir()
        if child.is_dir() and child.name in CURRENT_RESEARCH_SCRATCH_NAMES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--purge-exact-scratch-root",
        action="store_true",
        help=(
            "remove every child only when root is the drive-level "
            "DajoongTrainingTemp scratch directory"
        ),
    )
    parser.add_argument(
        "--purge-tmp-diagnostics",
        action="store_true",
        help="remove only named top-level CUBI-020 diagnostic files under repo/tmp",
    )
    parser.add_argument(
        "--purge-current-research-scratch",
        action="store_true",
        help="remove only completed research runs summarized in ACTIVE_RUNTIME.json",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    selected_modes = sum(
        int(value)
        for value in (
            args.purge_exact_scratch_root,
            args.purge_tmp_diagnostics,
            args.purge_current_research_scratch,
        )
    )
    if selected_modes > 1:
        raise RuntimeError("choose only one explicit purge mode")
    if args.purge_exact_scratch_root:
        if root.name != "DajoongTrainingTemp" or root.parent != Path(root.anchor):
            raise RuntimeError(
                "full scratch purge is limited to a drive-level "
                "DajoongTrainingTemp directory"
            )
        candidates = sorted(child.resolve() for child in root.iterdir())
    elif args.purge_tmp_diagnostics:
        candidates = _tmp_diagnostic_files(root)
    elif args.purge_current_research_scratch:
        candidates = _current_research_scratch(root)
    else:
        candidates = _ephemeral_children(root)
    rows = []
    for path in candidates:
        if path.parent != root:
            raise RuntimeError(f"refusing path outside the artifact root: {path}")
        size = _directory_bytes(path) if path.is_dir() else path.stat().st_size
        rows.append({"path": str(path), "bytes": size})
        if args.execute:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry_run",
                "root": str(root),
                "candidate_count": len(rows),
                "total_bytes": sum(row["bytes"] for row in rows),
                "candidates": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
