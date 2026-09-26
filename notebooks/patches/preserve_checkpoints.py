"""Run this cell in the existing 02 session: preserve weights, never train/promote them."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def preserve_segmentation_checkpoints(search_root, destination):
    search_root, destination = Path(search_root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Backup already exists; it will not be overwritten.")
    roots = []
    for config in search_root.rglob("config.json"):
        if config.is_symlink() or not (config.parent / "model.safetensors").is_file():
            continue
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if data.get("model_type") == "segformer":
            roots.append(config.parent)
    if not roots:
        raise RuntimeError("No SegFormer checkpoint files found. Use the original running 02 session or its FULL saved Notebook Output, not an empty session.")
    files = set()
    safe_metadata = {"config.json", "preprocessor_config.json", "trainer_state.json",
                     "training_manifest.json", "sha256_manifest.json", "resume_config.json"}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink() and (path.suffix == ".safetensors" or path.name in safe_metadata
                                                            or path.parent.name == "archived_epochs" and path.suffix == ".json"):
                if not path.resolve().is_relative_to(search_root):
                    raise ValueError("Checkpoint path escapes the input folder.")
                files.add(path)
    metadata = {"purpose": "Weight preservation only. Not a release approval or optimizer-resume archive.",
                "checkpoints": [root.relative_to(search_root).as_posix() for root in roots],
                "file_sha256": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(files):
            name = path.relative_to(search_root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            metadata["file_sha256"][name] = digest.hexdigest()
            archive.write(path, name)
        archive.writestr("PRESERVED_WEIGHTS_MANIFEST.json", json.dumps(metadata, indent=2))
    return metadata


# Existing interactive session: /kaggle/working. For a fresh recovery notebook with the
# FULL saved Notebook Output attached, change ONLY this path to /kaggle/input.
SEARCH_ROOT = Path("/kaggle/working")
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = Path("/kaggle/working") / f"02_preserved_weights_{stamp}.zip"
result = preserve_segmentation_checkpoints(SEARCH_ROOT, output)
print({"download": str(output), "checkpoint_folders": result["checkpoints"],
       "saved_files": len(result["file_sha256"])})
print("Download this ZIP and save the original notebook output before stopping the session.")
print("Already deleted/rotated checkpoints cannot be recovered from an inference ZIP. Nothing was trained or promoted.")
