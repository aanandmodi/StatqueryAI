"""Embedded in the numbered Kaggle notebooks; no repository checkout required."""

import base64
import csv
import hashlib
import io
import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_flood_data(destination):
    """Download only official hand-labelled triplets, with GCS generation/MD5 checks."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 5 * 1024**3:
        raise RuntimeError(
            "Keep at least 5 GiB free for the Sen1Floods11 data and checkpoints."
        )
    base = "https://storage.googleapis.com/sen1floods11/"
    origins = []

    def fetch(object_name, target):
        metadata_url = (
            "https://storage.googleapis.com/storage/v1/b/sen1floods11/o/"
            + urllib.parse.quote(object_name, safe="")
        )
        with urllib.request.urlopen(metadata_url, timeout=60) as response:
            metadata = json.load(response)
        expected = metadata["md5Hash"]

        def matches(path):
            if not path.is_file() or path.stat().st_size != int(metadata["size"]):
                return False
            return (
                base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
                == expected
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        if not matches(target):
            temporary = target.with_suffix(target.suffix + ".partial")
            for attempt in range(3):
                try:
                    url = base + object_name + "?generation=" + metadata["generation"]
                    with (
                        urllib.request.urlopen(url, timeout=120) as response,
                        temporary.open("wb") as out,
                    ):
                        shutil.copyfileobj(response, out)
                    if not matches(temporary):
                        raise ValueError(f"Source checksum mismatch: {object_name}")
                    temporary.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        origins.append(
            {
                "object": object_name,
                "generation": metadata["generation"],
                "md5": expected,
                "sha256": file_sha256(target),
            }
        )

    chip_ids = set()
    for split in ("train", "valid", "test"):
        source = f"v1.1/splits/flood_handlabeled/flood_{split}_data.csv"
        csv_path = root / "splits" / f"flood_{split}_data.csv"
        fetch(source, csv_path)
        identifiers = []
        for row in csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8"))):
            if not row:
                continue
            name = Path(row[0].strip()).name
            if not name.endswith("_S1Hand.tif"):
                raise ValueError(f"Unexpected official split entry: {row}")
            identifier = name.removesuffix("_S1Hand.tif")
            identifiers.append(identifier)
            chip_ids.add(identifier)
        (root / "splits" / f"flood_{split}_data.txt").write_text(
            "\n".join(identifiers) + "\n", encoding="utf-8"
        )
    for number, identifier in enumerate(sorted(chip_ids), 1):
        for remote, local, suffix in (
            ("S1Hand", "S1GRDHand", "S1Hand"),
            ("S2Hand", "S2L1CHand", "S2Hand"),
            ("LabelHand", "LabelHand", "LabelHand"),
        ):
            filename = f"{identifier}_{suffix}.tif"
            fetch(
                f"v1.1/data/flood_events/HandLabeled/{remote}/{filename}",
                root / "data" / local / filename,
            )
        if number % 20 == 0 or number == len(chip_ids):
            print(
                f"Verified Sen1Floods11 triplets: {number}/{len(chip_ids)}", flush=True
            )
    (root / "source_objects.json").write_text(
        json.dumps(origins, indent=2), encoding="utf-8"
    )
    return root


def find_trained_artifacts(input_root):
    """Identify attached exports by content and verify the two files used to load weights."""
    candidates = {"segmentation": [], "change": [], "fusion": []}
    for path in Path(input_root).rglob("config.json"):
        root = path.parent
        if not (root / "model.safetensors").is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        architecture = config.get("architecture")
        if architecture == "shared_resnet18_gru_answer_mask":
            role = "change"
        elif architecture == "terramind_s1_s2_pixel_flood_segmentation":
            role = "fusion"
        elif (
            config.get("model_type") == "segformer"
            and (root / "training_manifest.json").is_file()
        ):
            role = "segmentation"
        else:
            continue
        manifest_path = root / "sha256_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing hash manifest in {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in ("model.safetensors", "config.json"):
            if manifest.get(name) != file_sha256(root / name):
                raise ValueError(f"Checkpoint integrity failed: {root / name}")
        if role == "segmentation":
            report = json.loads(
                (root / "training_manifest.json").read_text(encoding="utf-8")
            )
            candidate = report.get("release_candidate", False)
            if not candidate:
                raise ValueError(
                    "SegFormer validation gate failed. Review its metrics before serving."
                )
        else:
            gate_path = root / "release_gate.json"
            gate = (
                json.loads(gate_path.read_text(encoding="utf-8"))
                if gate_path.is_file()
                else {}
            )
            if not gate.get("validation_gate_passed") or not gate.get(
                "test_gate_passed"
            ):
                raise ValueError(
                    f"{role} needs passing validation/test gates from this numbered pack."
                )
        candidates[role].append(root)
    for role, roots in candidates.items():
        if len(roots) != 1:
            raise ValueError(
                f"Attach exactly one passing {role} output from notebooks 02/03/04. Found {len(roots)}. "
                "Use Kaggle Add Input → Notebook Output, or attach the extracted inference zip."
            )
    return {role: roots[0] for role, roots in candidates.items()}


def export_inference_zip(source, filename):
    import zipfile

    source, target = Path(source), Path(filename)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != "training_state.pt":
                archive.write(path, Path(source.name) / path.relative_to(source))
    print(f"DOWNLOAD / PRESERVE: {target}", flush=True)
    return target
