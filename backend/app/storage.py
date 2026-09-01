from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.errors import PayloadTooLargeError, ValidationFailure

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class StoredUpload:
    asset_id: str
    path: Path
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str


class LocalAssetStore:
    """Immutable local asset store used by the hackathon/demo profile.

    Production deployments should replace this behind the same interface with
    S3/MinIO and signed multipart uploads; originals must remain immutable.
    """

    def __init__(self, root: Path, max_upload_bytes: int) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> StoredUpload:
        asset_id = f"ast_{uuid4().hex}"
        original_name = Path(upload.filename or "upload.bin").name
        sanitized = SAFE_NAME.sub("_", original_name).strip("._") or "upload.bin"
        destination = (self.root / f"{asset_id}__{sanitized}").resolve()
        if self.root not in destination.parents:
            raise ValidationFailure("Invalid upload filename")

        sha256 = hashlib.sha256()
        total = 0
        try:
            with destination.open("xb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_upload_bytes:
                        raise PayloadTooLargeError(
                            f"Upload exceeds {self.max_upload_bytes} bytes",
                            details={"limit_bytes": self.max_upload_bytes},
                        )
                    sha256.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if total == 0:
            destination.unlink(missing_ok=True)
            raise ValidationFailure("Uploaded file is empty")

        return StoredUpload(
            asset_id=asset_id,
            path=destination,
            original_name=original_name,
            content_type=upload.content_type or "application/octet-stream",
            size_bytes=total,
            sha256=sha256.hexdigest(),
        )

    def resolve(self, asset_id: str) -> Path:
        candidates = list(self.root.glob(f"{asset_id}__*"))
        if len(candidates) != 1:
            raise FileNotFoundError(asset_id)
        path = candidates[0].resolve()
        if self.root not in path.parents:
            raise ValidationFailure("Asset path escaped storage root")
        return path


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def allocate(self, analysis_id: str, filename: str) -> Path:
        safe_filename = SAFE_NAME.sub("_", Path(filename).name)
        analysis_dir = (self.root / analysis_id).resolve()
        analysis_dir.mkdir(parents=True, exist_ok=True)
        destination = (analysis_dir / safe_filename).resolve()
        if self.root not in destination.parents:
            raise ValidationFailure("Artifact path escaped storage root")
        return destination
