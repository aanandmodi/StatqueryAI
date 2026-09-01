from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    root: Path,
    *,
    model_id: str,
    model_revision: str,
    dataset_revisions: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "satquery-artifact-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_revisions": dataset_revisions,
        "metrics": metrics,
        "files": files,
    }


def write_artifact_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    destination = root / "manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def verify_artifact_manifest(root: Path, manifest_path: Path | None = None) -> None:
    manifest_path = manifest_path or root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["size_bytes"]:
            failures.append(f"size mismatch: {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"hash mismatch: {item['path']}")
    if failures:
        raise ValueError("artifact verification failed: " + "; ".join(failures))
