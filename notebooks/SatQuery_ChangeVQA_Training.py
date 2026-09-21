# ruff: noqa: E402
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SatQuery Change-VQA specialist — CDVQA + SECOND
#
# This is a standalone free Kaggle/Colab training notebook for the mandatory bi-temporal expert.
# It uses shared visual weights for time A/B, explicit difference/product features, a question GRU,
# a closed answer head, and a real mask head supervised by differences between SECOND semantic maps.
# It never calls the single-image Qwen adapter for change detection and never provisions paid hosting.
#
# **Run All with GPU + Internet:** CDVQA annotations and matching SECOND images/labels download
# automatically. The processed SECOND mirror is published by the PerASCD authors (not the original
# SECOND publisher). Its revision and archive SHA256 are pinned; CDVQA defines our train/val/test
# membership, NOT the mirror's folder names. No HF token or manual dataset attachment is required.
# Source: https://github.com/SathShen/PerASCD (Dataset Preparation).
# For research/demo use; the mirror's license tag does not override original imagery rights.

# %% [markdown]
# ## 0. Install and configure

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch>=2.6,<2.11", "torchvision>=0.21,<0.26", "safetensors>=0.5,<1",
    "huggingface_hub>=0.36,<1", "Pillow>=11,<13", "tqdm>=4.66,<5"])

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import random
import re
import urllib.request

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    seed: int = 42
    image_size: int = 256
    max_question_tokens: int = 32
    batch_size: int = 4
    epochs: int = 16
    learning_rate: float = 5e-5
    decoder_version: str = "multiscale-r2"
    questions_per_pair: int = 4
    weight_decay: float = 0.01
    mask_weight: float = 2.0
    workers: int = 0
    resume: bool = True
    smoke_test: bool = False
    run_test_once: bool = False
    max_train_qa: int | None = None
    max_validation_qa: int | None = None
    max_test_qa: int | None = None
    # Declare these only after reviewing validation; the notebook invents no target.
    release_minimum_answer_accuracy: float | None = None
    release_minimum_mask_iou: float | None = None
    repo_id: str = "aanandmodi/satquery-change-vqa-cdvqa"
    push_to_hub: bool = False


CFG = Config()
ROOT = Path("/kaggle/working/satquery-change-r2" if Path("/kaggle/working").exists() else "/content/satquery-change-r2")
CDVQA_REVISION = "cc5893123dd32326de38745b65d2ffe45055937b"
ANNOTATIONS, ARTIFACTS = ROOT / "annotations" / CDVQA_REVISION, ROOT / "artifacts"
for path in (ANNOTATIONS, ARTIFACTS):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a free Kaggle/Colab GPU before continuing.")
print({"config": asdict(CFG), "gpu": torch.cuda.get_device_name(0)})

# %% [markdown]
# ## 1. Download CDVQA annotations and locate SECOND imagery

# %%
CDVQA_BASE = f"https://raw.githubusercontent.com/YZHJessica/CDVQA/{CDVQA_REVISION}"
for split in ("Train", "Val", "Test", "Test2"):
    for part in ("images", "questions", "answers"):
        filename = f"{split}_{part}.json"
        target = ANNOTATIONS / filename
        if not target.exists():
            urllib.request.urlretrieve(f"{CDVQA_BASE}/{filename}", target)
        json.loads(target.read_text(encoding="utf-8"))
print("PASS: CDVQA annotation JSON downloaded and parsed.")


def looks_like_second(root: Path) -> bool:
    layouts = [("im1", "im2", "label1", "label2"), ("A", "B", "label1", "label2")]
    return any(all((root / name).is_dir() for name in layout) for layout in layouts)


def discover_second_root() -> Path | None:
    explicit = os.environ.get("SECOND_ROOT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    for parent in (Path("/kaggle/input"), Path("/content/drive/MyDrive"), Path("/content")):
        if parent.exists():
            candidates.extend(path for path in parent.glob("**/*") if path.is_dir())
    for candidate in candidates:
        if looks_like_second(candidate) or any(looks_like_second(candidate / split) for split in ("train", "val", "test")):
            return candidate
    if explicit:
        raise FileNotFoundError("SECOND_ROOT is set but does not contain supported images/labels.")
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_second_files(annotation_dir, splits=("Train", "Val", "Test")):
    required = set()
    seen = set()
    for split in splits:
        rows = json.loads((Path(annotation_dir) / f"{split}_images.json").read_text())["images"]
        names = {str(row["file_name"]) for row in rows if row.get("active", True)}
        if names & seen:
            raise ValueError("CDVQA pair filenames overlap across train/validation/test.")
        if any(Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".png") for name in names):
            raise ValueError("Unsafe or unexpected annotation filename.")
        seen.update(names)
        required.update((role, name) for role in ("im1", "im2", "label1", "label2") for name in names)
    return required


def extract_second_pairs(archive_path, destination, required):
    """Select exact CDVQA filenames, without flattening unrequested augmentation or using extractall."""
    import shutil
    import zipfile
    from pathlib import PurePosixPath
    from PIL import Image

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        selected = {}
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or len(parts) < 2:
                continue
            key = (parts[-2], parts[-1])
            if key not in required:
                continue
            if ".." in parts or info.filename.startswith(("/", "\\")) or "\\" in info.filename:
                raise ValueError("Unsafe SECOND archive path.")
            if key in selected:
                raise ValueError(f"Ambiguous duplicate SECOND pair member: {key}")
            if info.file_size > 32 * 1024**2 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected SECOND archive member type/size.")
            selected[key] = info
        missing = required - selected.keys()
        if missing:
            raise ValueError(f"Archive missing {len(missing)} required image/label files: {sorted(missing)[:4]}")
        needed = sum(info.file_size for info in selected.values())
        if shutil.disk_usage(destination).free < needed + 2 * 1024**3:
            raise RuntimeError("Insufficient disk space for SECOND extraction plus checkpoint reserve.")
        hashes = {}
        for index, (key, info) in enumerate(sorted(selected.items())):
            target = destination / key[0] / key[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".png.partial")
            # ZipFile streams and checks member CRC; never hold the archive in RAM.
            with archive.open(info) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            with Image.open(partial) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Unexpected SECOND image dimensions: {info.filename}")
                if key[0].startswith("label"):
                    values = np.asarray(image)
                    if values.ndim != 2 or values.dtype != np.uint8 or values.max() > 6:
                        raise ValueError(f"Expected processed SECOND index labels 0..6: {info.filename}")
                else:
                    image.verify()
            partial.replace(target)
            hashes[f"{key[0]}/{key[1]}"] = sha256_file(target)
            if (index + 1) % 1000 == 0:
                print(f"Extracted and checked {index + 1}/{len(selected)} files", flush=True)
    return hashes


def download_second(annotation_dir, root):
    import shutil
    from huggingface_hub import hf_hub_download

    repo = "SathShen/PerASCD-datasets"
    revision = "c50fab55c275ffa55c113565af54cfa73e2bd709"
    archive_sha = "e5d9be06636034bfff526f39b7ee3eb5d2a4bd145171238ed4cb4d1ffb588672"
    destination = Path(root) / "second-cdvqa"
    required = required_second_files(annotation_dir)
    manifest_path = destination / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        hashes = manifest.get("files_sha256", {})
        expected_paths = {f"{role}/{name}" for role, name in required}
        if (manifest.get("archive_sha256") == archive_sha and set(hashes) == expected_paths
                and all((destination / name).is_file() and sha256_file(destination / name) == digest
                        for name, digest in hashes.items())):
            print("PASS: existing SECOND extraction hashes verified.")
            return destination
    # Archive is 3.78 GB; reserve for extracted images, HF partial download and training outputs.
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError("Keep at least 10 GiB free before automatic SECOND download/extraction.")
    print("Downloading SECOND (3.78 GB) from the pinned PerASCD research mirror; no token needed.", flush=True)
    archive = Path(hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision,
                                 filename="SECONDbi.zip", local_dir=Path(root) / "downloads", token=False))
    if sha256_file(archive) != archive_sha:
        raise ValueError("SECOND archive SHA256 differs from pinned Hub LFS object. Refusing training.")
    hashes = extract_second_pairs(archive, destination, required)
    manifest = {"source_repo": repo, "revision": revision, "archive_sha256": archive_sha,
                "source_kind": "PerASCD author-published processed SECOND mirror",
                "annotation_revision": CDVQA_REVISION, "split_authority": "CDVQA filenames, not mirror folders",
                "files_sha256": hashes, "pairs": len(required) // 4}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"PASS": "SECOND images and both real label maps verified", "pairs": manifest["pairs"]})
    return destination


SECOND_ROOT = discover_second_root() or download_second(ANNOTATIONS, ROOT)
if (SECOND_ROOT / "source_manifest.json").exists():
    import shutil
    shutil.copyfile(SECOND_ROOT / "source_manifest.json", ARTIFACTS / "dataset_source_manifest.json")
print({"SECOND_ROOT": str(SECOND_ROOT)})

# %% [markdown]
# ## 2. Build leakage-safe QA manifests

# %%
def resolve_file(root: Path, split: str, role: str, filename: str) -> Path:
    aliases = {
        "time_a": ("im1", "A", "T1"), "time_b": ("im2", "B", "T2"),
        "label_a": ("label1", "labelA", "GT1"), "label_b": ("label2", "labelB", "GT2"),
    }[role]
    candidates = [root / alias / filename for alias in aliases]
    candidates += [root / split.lower() / alias / filename for alias in aliases]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {role} for {split}/{filename}")


def load_records(split: str, limit: int | None) -> list[dict]:
    images = json.loads((ANNOTATIONS / f"{split}_images.json").read_text(encoding="utf-8"))["images"]
    questions = json.loads((ANNOTATIONS / f"{split}_questions.json").read_text(encoding="utf-8"))["questions"]
    answers = json.loads((ANNOTATIONS / f"{split}_answers.json").read_text(encoding="utf-8"))["answers"]
    image_by_id = {int(row["id"]): row for row in images if row.get("active", True)}
    answer_by_question = {int(row["question_id"]): str(row["answer"]).strip().lower() for row in answers if row.get("active", True)}
    rows = []
    for question in questions:
        question_id = int(question["id"])
        if not question.get("active", True) or question_id not in answer_by_question:
            continue
        image = image_by_id[int(question["img_id"])]
        filename = str(image["file_name"])
        rows.append({
            "question_id": question_id, "image_id": int(question["img_id"]), "filename": filename,
            "question": str(question["question"]).strip(), "question_type": str(question["type"]),
            "answer": answer_by_question[question_id],
            "time_a": resolve_file(SECOND_ROOT, split, "time_a", filename),
            "time_b": resolve_file(SECOND_ROOT, split, "time_b", filename),
            "label_a": resolve_file(SECOND_ROOT, split, "label_a", filename),
            "label_b": resolve_file(SECOND_ROOT, split, "label_b", filename),
        })
    rows.sort(key=lambda row: hashlib.sha256(f"{row['question_id']}:{CFG.seed}".encode()).hexdigest())
    return rows[:limit] if limit else rows


train_records = load_records("Train", CFG.max_train_qa)
validation_records = load_records("Val", CFG.max_validation_qa)
test_records = load_records("Test", CFG.max_test_qa) if CFG.run_test_once else []
# Official annotation IDs restart from zero within EACH split. They are not global image IDs.
# SECOND filenames identify the underlying pair; checking local IDs falsely reports leakage.
train_images = {row["filename"] for row in train_records}
validation_images = {row["filename"] for row in validation_records}
test_images = {row["filename"] for row in test_records}
assert train_records and validation_records
assert not train_images & validation_images, "Image-pair leakage between train and validation"
assert not train_images & test_images, "Image-pair leakage between train and test"
assert not validation_images & test_images, "Image-pair leakage between validation and test"
print({"train_qa": len(train_records), "validation_qa": len(validation_records),
       "train_pairs": len(train_images), "validation_pairs": len(validation_images),
       "test_pairs": len(test_images)})
(ARTIFACTS / "split_manifest.json").write_text(json.dumps({
    "identity": "SECOND pair filename, not split-local annotation ID",
    "train_pairs": sorted(train_images), "validation_pairs": sorted(validation_images),
    "test_pairs": sorted(test_images),
    "annotation_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ANNOTATIONS.glob("*.json"))
    },
}, indent=2), encoding="utf-8")

# %% [markdown]
# ## 3. Vocabulary, supervised dataset, and model

# %%
from collections import Counter
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def build_vocab(texts: list[str], minimum: int = 2) -> dict[str, int]:
    counts = Counter(token for text in texts for token in TOKEN_PATTERN.findall(text.lower()))
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= minimum:
            vocabulary[token] = len(vocabulary)
    return vocabulary


word_vocab = build_vocab([row["question"] for row in train_records])
answer_values = sorted({row["answer"] for row in train_records})
answer_vocab = {answer: index for index, answer in enumerate(answer_values)}
unknown_answers = sorted({row["answer"] for row in validation_records} - set(answer_vocab))
if unknown_answers:
    raise RuntimeError(f"Validation answers missing from training vocabulary: {unknown_answers}")


class ChangeDataset(Dataset):
    def __init__(self, records: list[dict], training=False):
        self.records = records
        self.training = training
        self.transform = transforms.Compose([
            transforms.Resize((CFG.image_size, CFG.image_size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        time_a = self.transform(Image.open(row["time_a"]).convert("RGB"))
        time_b = self.transform(Image.open(row["time_b"]).convert("RGB"))
        label_a, label_b = np.asarray(Image.open(row["label_a"])), np.asarray(Image.open(row["label_b"]))
        mask = np.any(label_a != label_b, axis=-1) if label_a.ndim == 3 else label_a != label_b
        mask = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                             size=(CFG.image_size, CFG.image_size), mode="nearest")[0]
        if self.training and not re.search(r"\b(left|right|north|south|east|west|top|bottom)\b", row["question"].lower()):
            # Both dates and labels MUST receive the same spatial transform.
            for dimension in (-1, -2):
                if random.random() < 0.5:
                    time_a, time_b, mask = time_a.flip(dimension), time_b.flip(dimension), mask.flip(dimension)
        tokens = [word_vocab.get(token, 1) for token in TOKEN_PATTERN.findall(row["question"].lower())]
        tokens = (tokens[:CFG.max_question_tokens] + [0] * CFG.max_question_tokens)[:CFG.max_question_tokens]
        return {"time_a": time_a, "time_b": time_b, "tokens": torch.tensor(tokens),
                "answer": torch.tensor(answer_vocab.get(row["answer"], -1)), "mask": mask,
                "question_id": row["question_id"], "filename": row["filename"],
                "question_text": row["question"], "reference_text": row["answer"]}


class ChangeExpert(torch.nn.Module):
    def __init__(self, vocabulary_size, answer_classes, pretrained=True, decoder_version="legacy"):
        super().__init__()
        if decoder_version not in {"legacy", "multiscale-r2"}:
            raise ValueError("Unsupported change decoder version")
        self.decoder_version = decoder_version
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = torch.nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4)
        self.fuse = torch.nn.Sequential(torch.nn.Conv2d(2048, 512, 1, bias=False),
            torch.nn.BatchNorm2d(512), torch.nn.GELU(), torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.GELU())
        self.embedding = torch.nn.Embedding(vocabulary_size, 256, padding_idx=0)
        self.question = torch.nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.answer_head = torch.nn.Sequential(torch.nn.Linear(768, 512), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(512, answer_classes))
        self.mask_head = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.GELU(), torch.nn.Conv2d(128, 1, 1))
        if decoder_version == "multiscale-r2":
            self.laterals = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Conv2d(channels * 4, 64, 1),
                    torch.nn.GroupNorm(8, 64), torch.nn.GELU())
                for channels in (64, 128, 256)])
            self.detail_head = torch.nn.Sequential(torch.nn.Conv2d(256 + 192, 128, 3, padding=1),
                torch.nn.GroupNorm(8, 128), torch.nn.GELU(), torch.nn.Conv2d(128, 1, 1))

    def encode_pair(self, time_a, time_b):
        feature_a, feature_b = time_a, time_b
        details = []
        for index, layer in enumerate(self.stem):
            feature_a, feature_b = layer(feature_a), layer(feature_b)
            if self.decoder_version == "multiscale-r2" and index in (4, 5, 6):
                joined = torch.cat([feature_a, feature_b, (feature_a-feature_b).abs(), feature_a*feature_b], 1)
                details.append(self.laterals[index - 4](joined))
        fused = self.fuse(torch.cat([feature_a, feature_b, (feature_a-feature_b).abs(), feature_a*feature_b], 1))
        visual = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        mask = self.mask_head(fused)
        if details:
            size = details[0].shape[-2:]
            features = [F.interpolate(x, size=size, mode="bilinear", align_corners=False) for x in [fused, *details]]
            mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False) + self.detail_head(torch.cat(features, 1))
        return visual, F.interpolate(mask, size=time_a.shape[-2:], mode="bilinear", align_corners=False)

    def answer_from_visual(self, visual, tokens):
        _, hidden = self.question(self.embedding(tokens))
        question = torch.cat([hidden[-2], hidden[-1]], 1)
        return self.answer_head(torch.cat([visual, question], 1))

    def forward(self, time_a, time_b, tokens):
        visual, mask = self.encode_pair(time_a, time_b)
        return self.answer_from_visual(visual, tokens), mask


def objective(answer_logits, mask_logits, answers, masks):
    answer_logits, mask_logits, masks = answer_logits.float(), mask_logits.float(), masks.float()
    if not torch.isfinite(answer_logits).all() or not torch.isfinite(mask_logits).all():
        raise FloatingPointError("Non-finite change logits; stop without publishing.")
    answer_loss = F.cross_entropy(answer_logits, answers)
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks)
    probability = torch.sigmoid(mask_logits)
    intersection = (probability * masks).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1)).mean()
    mask_loss = 0.5 * bce + 0.5 * dice
    return answer_loss + CFG.mask_weight * mask_loss

# %% [markdown]
# ## 4. Train, validate, checkpoint, and export auditable predictions

# %%
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

train_loader = DataLoader(ChangeDataset(train_records, training=True), batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.workers, pin_memory=True)
validation_loader = DataLoader(ChangeDataset(sorted(validation_records, key=lambda row: row["filename"])), batch_size=CFG.batch_size, shuffle=False,
                               num_workers=CFG.workers, pin_memory=True)
test_loader = (DataLoader(ChangeDataset(sorted(test_records, key=lambda row: row["filename"])), batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.workers, pin_memory=True) if test_records else None)
model = ChangeExpert(len(word_vocab), len(answer_vocab), decoder_version=CFG.decoder_version).cuda()
# Optional previous 03 input: retain learned answers, initialize only the NEW detail decoder.
# Never import an optimizer, gate or metrics as if the new candidate had passed.
warm_candidates = []
for path in Path("/kaggle/input").rglob("config.json"):
    try:
        config = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if config.get("architecture") == "shared_resnet18_gru_answer_mask":
        warm_candidates.append((path.parent, config))
if len(warm_candidates) > 1:
    raise RuntimeError("Attach at most one previous 03 output.")
warm_start = None
if warm_candidates:
    previous, previous_config = warm_candidates[0]
    if previous_config["word_vocab"] != word_vocab or previous_config["answer_vocab"] != answer_vocab:
        raise RuntimeError("Previous 03 vocabulary differs; refuse unsafe warm start.")
    weight_sha = hashlib.sha256((previous / "model.safetensors").read_bytes()).hexdigest()
    if json.loads((previous / "sha256_manifest.json").read_text()).get("model.safetensors") != weight_sha:
        raise RuntimeError("Previous 03 checksum mismatch.")
    weights = load_file(previous / "model.safetensors")
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise FloatingPointError("Previous change weights are non-finite.")
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys or any(not name.startswith(("laterals.", "detail_head.")) for name in result.missing_keys):
        raise RuntimeError(f"Incompatible warm start: {result}")
    warm_start = {"weights_sha256": weight_sha}
    del weights
print({"warm_start": warm_start, "decoder": CFG.decoder_version})
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
use_bf16 = torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
scaler = torch.amp.GradScaler("cuda", enabled=False)
best, start_epoch, history = -1.0, 0, []
training_state = ARTIFACTS / "training_state.pt"
if CFG.resume and training_state.is_file():
    state = torch.load(training_state, map_location="cpu", weights_only=True)
    if state.get("config") != asdict(CFG) or state.get("warm_start") != warm_start:
        raise RuntimeError("Resume configuration differs. Use a fresh r2 output; preserve old files.")
    if not all(torch.isfinite(value).all() for value in state["model"].values()):
        raise FloatingPointError("Refusing non-finite resume state.")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    start_epoch, best, history = int(state["epoch"]), float(state["best"]), list(state["history"])
    print({"resumed_after_epoch": start_epoch, "best_selection_score": best})


def true_runs(mask: np.ndarray) -> list[list[int]]:
    indexes = np.flatnonzero(mask.reshape(-1))
    if not len(indexes):
        return []
    runs, start, previous = [], int(indexes[0]), int(indexes[0])
    for value in indexes[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append([start, previous - start + 1])
            start = value
        previous = value
    runs.append([start, previous - start + 1])
    return runs


@torch.inference_mode()
def evaluate(loader, split: str, prediction_path: Path | None = None) -> dict[str, float]:
    model.eval()
    answer_correct = answer_total = known_total = 0
    intersection = union = predicted_pixels = target_pixels = 0
    seen_pairs = set()
    visual_cache = {}  # Only current pair(s), never a full validation prediction cache.
    rows = []
    for batch in tqdm(loader, desc=f"evaluate {split}"):
        time_a, time_b = batch["time_a"].cuda(), batch["time_b"].cuda()
        tokens = batch["tokens"].cuda()
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_bf16):
            missing = {}
            for index, name in enumerate(batch["filename"]):
                if name not in visual_cache:
                    missing.setdefault(name, index)
            if missing:
                positions = list(missing.values())
                visuals, masks = model.encode_pair(time_a[positions], time_b[positions])
                for offset, name in enumerate(missing):
                    visual_cache[name] = (visuals[offset:offset+1], masks[offset:offset+1])
            visual = torch.cat([visual_cache[name][0] for name in batch["filename"]])
            mask_logits = torch.cat([visual_cache[name][1] for name in batch["filename"]])
            answer_logits = model.answer_from_visual(visual, tokens)
            last_name = batch["filename"][-1]
            visual_cache = {last_name: visual_cache[last_name]}
        if not torch.isfinite(answer_logits).all() or not torch.isfinite(mask_logits).all():
            raise FloatingPointError("Evaluation returned non-finite logits.")
        answer_indexes = answer_logits.argmax(1).cpu()
        references = batch["answer"]
        known = references >= 0
        answer_correct += int(((answer_indexes == references) & known).sum())
        known_total += int(known.sum())
        answer_total += len(references)
        probabilities = torch.sigmoid(mask_logits.float()).cpu().numpy()[:, 0]
        predictions = probabilities >= 0.5
        targets = batch["mask"].numpy()[:, 0] >= 0.5
        for index, (prediction, probability, target) in enumerate(
            zip(predictions, probabilities, targets, strict=True)
        ):
            item_intersection = int((prediction & target).sum())
            item_union = int((prediction | target).sum())
            pair_name = batch["filename"][index]
            if pair_name not in seen_pairs:
                seen_pairs.add(pair_name)
                intersection += item_intersection
                union += item_union
                predicted_pixels += int(prediction.sum())
                target_pixels += int(target.sum())
            if prediction_path is not None:
                predicted_answer = answer_values[int(answer_indexes[index])]
                rows.append({
                    "question_id": int(batch["question_id"][index]),
                    "pair_filename": batch["filename"][index],
                    "split": split,
                    "question": batch["question_text"][index],
                    "reference_answer": batch["reference_text"][index],
                    "predicted_answer": predicted_answer,
                    "answer_correct": predicted_answer == batch["reference_text"][index],
                    "shape": list(prediction.shape),
                    "intersection_pixels": item_intersection,
                    "union_pixels": item_union,
                    "mean_change_score": float(probability.mean()),
                    "prediction_true_runs": true_runs(prediction),
                    "reference_true_runs": true_runs(target),
                })
    metrics = {
        "answer_accuracy": answer_correct / max(answer_total, 1),
        "known_answer_examples": known_total,
        "unknown_answer_examples": answer_total - known_total,
        "unique_mask_pairs": len(seen_pairs),
        "mask_iou": intersection / max(union, 1),
        "mask_dice": 2 * intersection / max(predicted_pixels + target_pixels, 1),
        "examples": len(loader.dataset),
    }
    if prediction_path is not None:
        prediction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return metrics


for epoch in range(start_epoch, 1 if CFG.smoke_test else CFG.epochs):
    # Equal pair exposure; rotate QA choices each epoch, instead of re-training
    # the same mask dozens of times for pairs with more questions.
    grouped = {}
    for record in train_records:
        grouped.setdefault(record["filename"], []).append(record)
    epoch_rng = random.Random(CFG.seed + epoch)
    selected = [row for rows in grouped.values()
                for row in epoch_rng.sample(rows, min(CFG.questions_per_pair, len(rows))) ]
    train_loader = DataLoader(ChangeDataset(selected, training=True), batch_size=CFG.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    model.train()
    running = 0.0
    for batch in tqdm(train_loader, desc=f"train {epoch + 1}/{CFG.epochs}"):
        tensors = {key: value.cuda(non_blocking=True) for key, value in batch.items()
                   if isinstance(value, torch.Tensor)}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_bf16):
            answer_logits, mask_logits = model(tensors["time_a"], tensors["time_b"], tensors["tokens"])
            loss = objective(answer_logits, mask_logits, tensors["answer"], tensors["mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}; stop.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())

    validation = evaluate(validation_loader, "validation")
    metrics = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1),
               **validation}
    # Optimize the weakest normalized validation gate, not the average that
    # previously preferred a strong answer head with a failing mask.
    selection_score = min(metrics["answer_accuracy"] / (CFG.release_minimum_answer_accuracy or 0.60),
                          metrics["mask_iou"] / (CFG.release_minimum_mask_iou or 0.40))
    metrics["selection_score"] = selection_score
    history.append(metrics)
    (ARTIFACTS / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False))
    print(metrics)
    if selection_score > best:
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise FloatingPointError("Cannot save non-finite change weights.")
        best = selection_score
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                  ARTIFACTS / "model.safetensors")
        (ARTIFACTS / "config.json").write_text(json.dumps({"artifact_version": "satquery-pair-v2",
            "architecture": "shared_resnet18_gru_answer_mask", "mask_supervision": "SECOND changed semantic labels",
            "config": asdict(CFG), "warm_start": warm_start, "word_vocab": word_vocab, "answer_vocab": answer_vocab,
            "best_metrics": metrics}, indent=2), encoding="utf-8")
    torch.save({"epoch": epoch + 1, "best": best, "history": history,
                "config": asdict(CFG), "warm_start": warm_start,
                "model": model.state_dict(), "optimizer": optimizer.state_dict()}, training_state)

(ARTIFACTS / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

# Every reported result below comes from the selected best weights.
model.load_state_dict(load_file(ARTIFACTS / "model.safetensors"), strict=True)
validation_predictions = ARTIFACTS / "validation_predictions.jsonl"
validation_metrics = evaluate(validation_loader, "validation", validation_predictions)
test_predictions = ARTIFACTS / "test_predictions.jsonl"
validation_ready = (validation_metrics["answer_accuracy"] >= (CFG.release_minimum_answer_accuracy or 0.60)
                    and validation_metrics["mask_iou"] >= (CFG.release_minimum_mask_iou or 0.40))
test_metrics = evaluate(test_loader, "test", test_predictions) if test_loader and validation_ready else None
(ARTIFACTS / "evaluation_summary.json").write_text(
    json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2),
    encoding="utf-8",
)
print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))

# Fresh architecture/weight reload, without downloading a second pretrained backbone.
reloaded = ChangeExpert(len(word_vocab), len(answer_vocab), pretrained=False, decoder_version=CFG.decoder_version).cuda().eval()
reloaded.load_state_dict(load_file(ARTIFACTS / "model.safetensors"), strict=True)
reload_batch = next(iter(validation_loader))
with torch.inference_mode():
    reload_answers, reload_masks = reloaded(reload_batch["time_a"].cuda(), reload_batch["time_b"].cuda(), reload_batch["tokens"].cuda())
assert torch.isfinite(reload_answers).all() and torch.isfinite(reload_masks).all()
assert reload_masks.shape[-2:] == (CFG.image_size, CFG.image_size)
del reloaded, reload_batch, reload_answers, reload_masks
torch.cuda.empty_cache()
print("PASS: saved ChangeVQA weights reloaded strictly and produced finite answers/masks. Not a test-set accuracy certificate.")

# %% [markdown]
# ## 5. Hash, optional free Hub upload, and safe stop

# %%
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {path.name: sha256(path) for path in sorted(ARTIFACTS.iterdir())
            if path.is_file() and path.name not in {"sha256_manifest.json", "training_state.pt"}}
(ARTIFACTS / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
thresholds_declared = (CFG.release_minimum_answer_accuracy is not None
                       and CFG.release_minimum_mask_iou is not None)
validation_passed = (thresholds_declared
                     and validation_metrics["answer_accuracy"] >= CFG.release_minimum_answer_accuracy
                     and validation_metrics["mask_iou"] >= CFG.release_minimum_mask_iou)
test_passed = (thresholds_declared and test_metrics is not None
               and test_metrics["answer_accuracy"] >= CFG.release_minimum_answer_accuracy
               and test_metrics["mask_iou"] >= CFG.release_minimum_mask_iou)
if CFG.push_to_hub:
    if not validation_passed or not test_passed:
        raise RuntimeError("Hub upload refused: declare and pass validation/test answer and mask gates first.")
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
    if not token:
        raise RuntimeError("Add HF_TOKEN to notebook Secrets; never paste it into source.")
    api = HfApi(token=token)
    api.create_repo(CFG.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=CFG.repo_id, repo_type="model", folder_path=ARTIFACTS,
                      commit_message="Release audited SatQuery Change-VQA specialist",
                      ignore_patterns=["training_state.pt"])

gate = {"weights": (ARTIFACTS / "model.safetensors").is_file(),
        "config": (ARTIFACTS / "config.json").is_file(),
        "hash_manifest": (ARTIFACTS / "sha256_manifest.json").is_file(),
        "raw_validation_predictions": validation_predictions.is_file(),
        "release_thresholds_declared": thresholds_declared,
        "validation_gate_passed": validation_passed,
        "test_run_completed": test_metrics is not None,
        "test_gate_passed": test_passed,
        "paid_endpoint_created": False}
print(json.dumps(gate, indent=2))
assert gate["weights"] and gate["config"] and gate["hash_manifest"] and gate["raw_validation_predictions"]
assert not gate["paid_endpoint_created"]
print("SAFE STOP — download artifacts, save the notebook version, then turn off the GPU.")
