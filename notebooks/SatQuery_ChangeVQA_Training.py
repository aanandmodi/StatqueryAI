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
# **Before Run All:** attach the SECOND dataset as a Kaggle Dataset (or mount it in Colab). CDVQA's
# repository contains QA annotations but not the underlying SECOND pixels. The notebook discovers
# common folder layouts automatically and fails closed if a pair or label map is missing.

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
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    mask_weight: float = 0.5
    workers: int = 2
    max_train_qa: int | None = None
    max_validation_qa: int | None = None
    repo_id: str = "aanandmodi/satquery-change-vqa-cdvqa"
    push_to_hub: bool = False


CFG = Config()
ROOT = Path("/kaggle/working/satquery-change" if Path("/kaggle/working").exists() else "/content/satquery-change")
ANNOTATIONS, ARTIFACTS = ROOT / "annotations", ROOT / "artifacts"
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
CDVQA_BASE = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
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


def discover_second_root() -> Path:
    explicit = os.environ.get("SECOND_ROOT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    for parent in (Path("/kaggle/input"), Path("/content/drive/MyDrive"), Path("/content")):
        if parent.exists():
            candidates.extend(path for path in parent.glob("**/*") if path.is_dir())
    for candidate in candidates:
        if looks_like_second(candidate) or any(looks_like_second(candidate / split) for split in ("train", "val", "test")):
            return candidate
    raise FileNotFoundError(
        "SECOND imagery was not found. Attach the SECOND dataset, then set SECOND_ROOT to the folder "
        "containing im1/im2/label1/label2 (or A/B/label1/label2)."
    )


SECOND_ROOT = discover_second_root()
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
# Official annotation IDs restart from zero within EACH split. They are not global image IDs.
# SECOND filenames identify the underlying pair; checking local IDs falsely reports leakage.
train_images = {row["filename"] for row in train_records}
validation_images = {row["filename"] for row in validation_records}
assert train_records and validation_records
assert not train_images & validation_images, "Image-pair leakage between train and validation"
print({"train_qa": len(train_records), "validation_qa": len(validation_records),
       "train_pairs": len(train_images), "validation_pairs": len(validation_images)})
(ARTIFACTS / "split_manifest.json").write_text(json.dumps({
    "identity": "SECOND pair filename, not split-local annotation ID",
    "train_pairs": sorted(train_images), "validation_pairs": sorted(validation_images),
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
    def __init__(self, records: list[dict]):
        self.records = records
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
        tokens = [word_vocab.get(token, 1) for token in TOKEN_PATTERN.findall(row["question"].lower())]
        tokens = (tokens[:CFG.max_question_tokens] + [0] * CFG.max_question_tokens)[:CFG.max_question_tokens]
        return {"time_a": time_a, "time_b": time_b, "tokens": torch.tensor(tokens),
                "answer": torch.tensor(answer_vocab[row["answer"]]), "mask": mask}


class ChangeExpert(torch.nn.Module):
    def __init__(self, vocabulary_size, answer_classes, pretrained=True):
        super().__init__()
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

    def forward(self, time_a, time_b, tokens):
        feature_a, feature_b = self.stem(time_a), self.stem(time_b)
        fused = self.fuse(torch.cat([feature_a, feature_b, (feature_a-feature_b).abs(), feature_a*feature_b], 1))
        visual = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        _, hidden = self.question(self.embedding(tokens))
        question = torch.cat([hidden[-2], hidden[-1]], 1)
        answer = self.answer_head(torch.cat([visual, question], 1))
        mask = F.interpolate(self.mask_head(fused), size=time_a.shape[-2:], mode="bilinear", align_corners=False)
        return answer, mask


def objective(answer_logits, mask_logits, answers, masks):
    answer_loss = F.cross_entropy(answer_logits, answers)
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks)
    probability = torch.sigmoid(mask_logits)
    intersection = (probability * masks).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (probability.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1)).mean()
    mask_loss = 0.5 * bce + 0.5 * dice
    return answer_loss + CFG.mask_weight * mask_loss

# %% [markdown]
# ## 4. Train, validate, checkpoint

# %%
from safetensors.torch import save_file
from tqdm.auto import tqdm

train_loader = DataLoader(ChangeDataset(train_records), batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.workers, pin_memory=True)
validation_loader = DataLoader(ChangeDataset(validation_records), batch_size=CFG.batch_size, shuffle=False,
                               num_workers=CFG.workers, pin_memory=True)
model = ChangeExpert(len(word_vocab), len(answer_vocab)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
use_bf16 = torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
best = -1.0
history = []

for epoch in range(CFG.epochs):
    model.train()
    running = 0.0
    for batch in tqdm(train_loader, desc=f"train {epoch + 1}/{CFG.epochs}"):
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            answer_logits, mask_logits = model(batch["time_a"], batch["time_b"], batch["tokens"])
            loss = objective(answer_logits, mask_logits, batch["answer"], batch["mask"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())

    model.eval()
    correct = total = 0
    intersection = union = predicted_pixels = target_pixels = 0
    with torch.inference_mode():
        for batch in validation_loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            answers, masks = model(batch["time_a"], batch["time_b"], batch["tokens"])
            correct += int((answers.argmax(1) == batch["answer"]).sum())
            total += len(batch["answer"])
            predicted, target = torch.sigmoid(masks) >= 0.5, batch["mask"] >= 0.5
            intersection += int((predicted & target).sum())
            union += int((predicted | target).sum())
            predicted_pixels += int(predicted.sum())
            target_pixels += int(target.sum())
    metrics = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1),
        "validation_accuracy": correct / max(total, 1), "mask_iou": intersection / max(union, 1),
        "mask_dice": 2 * intersection / max(predicted_pixels + target_pixels, 1)}
    history.append(metrics)
    print(metrics)
    if metrics["validation_accuracy"] > best:
        best = metrics["validation_accuracy"]
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                  ARTIFACTS / "model.safetensors")
        (ARTIFACTS / "config.json").write_text(json.dumps({"artifact_version": "satquery-pair-v2",
            "architecture": "shared_resnet18_gru_answer_mask", "mask_supervision": "SECOND changed semantic labels",
            "config": asdict(CFG), "word_vocab": word_vocab, "answer_vocab": answer_vocab,
            "best_metrics": metrics}, indent=2), encoding="utf-8")

(ARTIFACTS / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

# Fresh architecture/weight reload, without downloading a second pretrained backbone.
from safetensors.torch import load_file
reloaded = ChangeExpert(len(word_vocab), len(answer_vocab), pretrained=False).cuda().eval()
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


manifest = {path.name: sha256(path) for path in sorted(ARTIFACTS.iterdir()) if path.is_file() and path.name != "sha256_manifest.json"}
(ARTIFACTS / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
if CFG.push_to_hub:
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
                      commit_message="Release audited SatQuery Change-VQA specialist")

gate = {"weights": (ARTIFACTS / "model.safetensors").is_file(),
        "config": (ARTIFACTS / "config.json").is_file(),
        "hash_manifest": (ARTIFACTS / "sha256_manifest.json").is_file(),
        "paid_endpoint_created": False}
print(json.dumps(gate, indent=2))
assert gate["weights"] and gate["config"] and gate["hash_manifest"] and not gate["paid_endpoint_created"]
print("PASS — download artifacts or verify the Hub upload, save the notebook, then stop the GPU.")
