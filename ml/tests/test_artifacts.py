from __future__ import annotations

from pathlib import Path

import pytest

from satquery_ml.artifacts import (
    build_artifact_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)


def test_manifest_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "adapter.safetensors"
    artifact.write_bytes(b"safe-model")
    manifest = build_artifact_manifest(
        tmp_path,
        model_id="model",
        model_revision="abc123",
        dataset_revisions={"dataset": "v1"},
        metrics={"accuracy": 0.8},
    )
    write_artifact_manifest(tmp_path, manifest)
    verify_artifact_manifest(tmp_path)

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verification failed"):
        verify_artifact_manifest(tmp_path)
