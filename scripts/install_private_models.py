from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ALLOWED_SUFFIXES = (".onnx", ".json")
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe model-bundle member: {name}")
    if path.parts[0] == "models":
        path = PurePosixPath(*path.parts[1:])
    if len(path.parts) != 1:
        raise ValueError(f"Model bundle must contain a flat models directory: {name}")
    if not path.name.endswith(ALLOWED_SUFFIXES):
        raise ValueError(f"Unsupported model-bundle member: {name}")
    return path


def install(bundle: Path, destination: Path, expected_sha256: str) -> list[str]:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("Expected bundle SHA-256 must contain exactly 64 hexadecimal characters")
    actual = sha256(bundle)
    if actual != expected:
        raise ValueError(f"Private model bundle SHA-256 mismatch: expected {expected}, received {actual}")

    with tempfile.TemporaryDirectory(prefix="dajoong-models-") as temporary:
        staging = Path(temporary)
        installed: list[str] = []
        total = 0
        with tarfile.open(bundle, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"Model bundle may contain regular files only: {member.name}")
                relative = normalized_member(member.name)
                if member.size > MAX_FILE_BYTES:
                    raise ValueError(f"Model file exceeds the {MAX_FILE_BYTES}-byte limit: {member.name}")
                total += member.size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("Private model bundle exceeds the total size limit")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not read model-bundle member: {member.name}")
                target = staging / relative.name
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                installed.append(relative.name)

        checkpoints = sorted(name for name in installed if name.endswith(".onnx"))
        if not checkpoints:
            raise ValueError("Private model bundle does not contain an ONNX checkpoint")
        for checkpoint in checkpoints:
            if f"{checkpoint}.json" not in installed:
                raise ValueError(f"Checkpoint is missing its content-addressed manifest: {checkpoint}")

        destination.mkdir(parents=True, exist_ok=True)
        for name in installed:
            source = staging / name
            temporary_target = destination / f".{name}.tmp"
            shutil.copyfile(source, temporary_target)
            os.replace(temporary_target, destination / name)
        return sorted(installed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and install the private Dajoong model bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    names = install(arguments.bundle.resolve(), arguments.destination.resolve(), arguments.sha256)
    print(f"Installed {len(names)} verified private model files")


if __name__ == "__main__":
    main()
