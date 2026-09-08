# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ruff: noqa: E402

# %% [markdown]
# # SatQuery semantic-mask specialist — free Kaggle training
#
# This notebook trains a **pixel-supervised** land-cover model for the single-image overlay.
# It does not pretend that additional Qwen VQA examples can teach exact pixel boundaries.
#
# The default free-T4 profile uses 70% of LoveDA's official **training split**, stratified by
# urban/rural domain. The official validation split remains completely untouched. Increase the
# fraction only after the default run completes; never mix validation/test images into training.
#
# LoveDA is academic/non-commercial and derived from Google Earth imagery. It is suitable for an
# SIH research prototype, not automatically for a commercial release. The first transfer model
# must still be evaluated on representative Indian/ISRO imagery before operational claims.

# %% [markdown]
# ## 0. Install a pinned cloud environment

# %%
import subprocess
import sys

PACKAGES = [
    "transformers==4.57.1",
    "accelerate==1.7.0",
    "huggingface_hub==0.36.2",
    "albumentations==2.0.8",
    "safetensors>=0.5,<1",
    "tensorboard>=2.18,<3",
    "requests>=2.32,<3",
    "tqdm>=4.66,<5",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *PACKAGES])
print("Dependencies installed. Restart once only if the next cell imports cached old modules.")

# %% [markdown]
# ## 1. Configuration — edit only this cell

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    base_model: str = "nvidia/mit-b0"
    base_revision: str = "80983a413c30d36a39c20203974ae7807835e2b4"
    output_repo: str = "aanandmodi/satquery-segformer-loveda"
    train_fraction: float = 0.70
    seed: int = 42
    crop_size: int = 512
    epochs: int = 12
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 0.01
    dice_weight: float = 0.5
    num_workers: int = 2
    push_to_hub: bool = False
    make_repo_private: bool = True


CFG = Config()
assert 0 < CFG.train_fraction <= 1
ROOT = Path(
    "/kaggle/working/satquery-segmentation"
    if Path("/kaggle/working").exists()
    else "/content/satquery-segmentation"
)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "best-model"
EVAL_DIR = ROOT / "evaluation"
for directory in (DATA_DIR, OUTPUT_DIR, EVAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
print(asdict(CFG))

# %% [markdown]
# ## 2. GPU, seeds and optional Hugging Face token

# %%
import getpass
import json
import random
import numpy as np
import torch

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle/Colab GPU before continuing.")
torch.backends.cuda.matmul.allow_tf32 = True


def optional_secret(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return (UserSecretsClient().get_secret(name) or "").strip() or None
    except Exception:
        return None


HF_TOKEN = optional_secret("HF_TOKEN")
if CFG.push_to_hub and not HF_TOKEN:
    HF_TOKEN = getpass.getpass("HF write token (hidden): ").strip()
    if not HF_TOKEN:
        raise RuntimeError("push_to_hub=True requires an HF write token.")
print({"gpu": torch.cuda.get_device_name(0), "push_to_hub": CFG.push_to_hub})

# %% [markdown]
# ## 3. Download the official LoveDA train and validation splits
#
# The archives and MD5 values are from the dataset publisher's release. The public test masks are
# unavailable, so this notebook reports development metrics on the official validation split and
# never calls them test metrics. Download is several GB; enable Internet in Kaggle.

# %%
import hashlib
import zipfile
import requests
from PIL import Image
from tqdm.auto import tqdm

ARCHIVES = {
    "train": {
        "url": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
        "filename": "Train.zip",
        "md5": "de2b196043ed9b4af1690b3f9a7d558f",
    },
    "val": {
        "url": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
        "filename": "Val.zip",
        "md5": "84cae2577468ff0b5386758bb386d31d",
    },
}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 -- publisher checksum, not a security primitive
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(split: str) -> None:
    metadata = ARCHIVES[split]
    destination = DATA_DIR / metadata["filename"]
    extracted = DATA_DIR / split.capitalize()
    if not destination.exists() or file_md5(destination) != metadata["md5"]:
        response = requests.get(metadata["url"], stream=True, timeout=(30, 300))
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with destination.open("wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=metadata["filename"]
        ) as progress:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    if file_md5(destination) != metadata["md5"]:
        raise RuntimeError(f"Publisher checksum failed for {destination.name}")
    if not extracted.exists():
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(DATA_DIR)


class LoveDASource:
    def __init__(self, split: str):
        root = DATA_DIR / split.capitalize()
        images = sorted(root.glob("*/images_png/*.png"))
        self.files = [
            {"image": str(path), "mask": str(path).replace("images_png", "masks_png")}
            for path in images
        ]
        if not self.files or any(not Path(item["mask"]).exists() for item in self.files):
            raise RuntimeError(f"Incomplete LoveDA {split} extraction at {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        with Image.open(item["image"]) as image:
            pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1)
        with Image.open(item["mask"]) as mask:
            labels = torch.from_numpy(np.array(mask, copy=True))
        return {"image": pixels, "mask": labels}


for split_name in ARCHIVES:
    download_and_extract(split_name)
train_source = LoveDASource("train")
validation_source = LoveDASource("val")
print({"official_train": len(train_source), "official_validation": len(validation_source)})

# %% [markdown]
# ## 4. Select 70% of train without touching validation
#
# Sampling is deterministic and stratified by LoveDA's urban/rural folders. The selected path
# hashes are written to the manifest, making the run reproducible and auditable.

# %%
def domain_for(item: dict[str, str]) -> str:
    normalized = item["image"].replace("\\", "/").lower()
    return "urban" if "/urban/" in normalized else "rural"


def stratified_indexes(source, fraction: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {"urban": [], "rural": []}
    for index, item in enumerate(source.files):
        groups[domain_for(item)].append(index)
    selected: list[int] = []
    for indexes in groups.values():
        rng.shuffle(indexes)
        selected.extend(indexes[: max(1, round(len(indexes) * fraction))])
    return sorted(selected)


train_indexes = stratified_indexes(train_source, CFG.train_fraction, CFG.seed)
train_names = [Path(train_source.files[index]["image"]).as_posix() for index in train_indexes]
selection_sha256 = hashlib.sha256("\n".join(train_names).encode()).hexdigest()
print({
    "selected_train": len(train_indexes),
    "train_fraction": len(train_indexes) / len(train_source),
    "selection_sha256": selection_sha256,
})

# %% [markdown]
# ## 5. Pixel-safe augmentation and datasets
#
# LoveDA labels are 1–7 with no-data 0. We remap them to model labels 0–6 and no-data 255. Image
# transforms use bilinear interpolation while masks use nearest-neighbour interpolation.

# %%
import albumentations as A
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

LABELS = ["background", "building", "road", "water", "barren", "forest", "agricultural"]
ID2LABEL = {index: label for index, label in enumerate(LABELS)}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}

train_transform = A.Compose([
    A.PadIfNeeded(min_height=CFG.crop_size, min_width=CFG.crop_size),
    A.RandomCrop(height=CFG.crop_size, width=CFG.crop_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.35),
])
validation_transform = A.Compose([
    A.Resize(height=CFG.crop_size, width=CFG.crop_size),
])
processor = SegformerImageProcessor(
    do_resize=False,
    do_reduce_labels=False,
)


class LoveDASegmentationDataset(Dataset):
    def __init__(self, source, indexes, transform):
        self.source = source
        self.indexes = list(indexes)
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, position):
        source_index = self.indexes[position]
        sample = self.source[source_index]
        image = sample["image"].permute(1, 2, 0).byte().numpy()
        mask = sample["mask"].numpy().astype(np.uint8)
        augmented = self.transform(image=image, mask=mask)
        remapped = np.where(augmented["mask"] == 0, 255, augmented["mask"] - 1).astype(np.uint8)
        encoded = processor(
            images=augmented["image"],
            segmentation_maps=remapped,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
            "source_index": torch.tensor(source_index),
        }


train_dataset = LoveDASegmentationDataset(
    train_source, train_indexes, train_transform
)
validation_dataset = LoveDASegmentationDataset(
    validation_source, range(len(validation_source)), validation_transform
)
print({"train": len(train_dataset), "validation": len(validation_dataset), "labels": LABELS})

# %% [markdown]
# ## 6. Estimate class weights from training masks only

# %%
class_counts = np.zeros(len(LABELS), dtype=np.int64)
for position, index in enumerate(train_indexes):
    mask = train_source[index]["mask"].numpy()
    for publisher_id in range(1, 8):
        class_counts[publisher_id - 1] += int((mask == publisher_id).sum())
    if (position + 1) % 500 == 0:
        print("scanned", position + 1)
frequencies = class_counts / class_counts.sum()
class_weights = 1 / np.log(1.02 + frequencies)
class_weights = class_weights / class_weights.mean()
print({label: round(float(class_weights[index]), 4) for index, label in ID2LABEL.items()})

# %% [markdown]
# ## 7. Load SegFormer and define weighted cross-entropy + Dice

# %%
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, Trainer

model = SegformerForSemanticSegmentation.from_pretrained(
    CFG.base_model,
    revision=CFG.base_revision,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    num_labels=len(LABELS),
    ignore_mismatched_sizes=True,
    trust_remote_code=False,
)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32)


class DiceCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs.pop("source_index", None)
        outputs = model(**inputs)
        logits = F.interpolate(
            outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        ce = F.cross_entropy(
            logits, labels, weight=weight_tensor.to(logits.device), ignore_index=255
        )
        valid = labels != 255
        safe_labels = labels.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_labels, num_classes=len(LABELS)).permute(0, 3, 1, 2)
        one_hot = one_hot.to(logits.dtype) * valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid.unsqueeze(1)
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        dice_loss = 1 - ((2 * intersection[present] + 1) / (denominator[present] + 1)).mean()
        loss = ce + CFG.dice_weight * dice_loss
        return (loss, outputs) if return_outputs else loss


print({"parameters": sum(parameter.numel() for parameter in model.parameters())})

# %% [markdown]
# ## 8. Real validation metrics

# %%
from transformers import EvalPrediction


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def segmentation_metrics(evaluation: EvalPrediction) -> dict[str, float]:
    predictions = np.asarray(evaluation.predictions)
    references = np.asarray(evaluation.label_ids)
    metrics: dict[str, float] = {}
    ious, dices = [], []
    for class_id, label in ID2LABEL.items():
        predicted = predictions == class_id
        expected = references == class_id
        intersection = np.logical_and(predicted, expected).sum()
        union = np.logical_or(predicted, expected).sum()
        denominator = predicted.sum() + expected.sum()
        iou = float(intersection / union) if union else float("nan")
        dice = float(2 * intersection / denominator) if denominator else float("nan")
        metrics[f"iou_{label}"] = iou
        metrics[f"dice_{label}"] = dice
        if np.isfinite(iou):
            ious.append(iou)
        if np.isfinite(dice):
            dices.append(dice)
    valid = references != 255
    metrics["mean_iou"] = float(np.mean(ious))
    metrics["mean_dice"] = float(np.mean(dices))
    metrics["pixel_accuracy"] = float((predictions[valid] == references[valid]).mean())
    return metrics

# %% [markdown]
# ## 9. Train with checkpoint/resume

# %%
from transformers import EarlyStoppingCallback, TrainingArguments

training_args = TrainingArguments(
    output_dir=str(ROOT / "checkpoints"),
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    dataloader_num_workers=CFG.num_workers,
    eval_accumulation_steps=1,
    remove_unused_columns=False,
    report_to="none",
    seed=CFG.seed,
    data_seed=CFG.seed,
)
trainer = DiceCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=segmentation_metrics,
    preprocess_logits_for_metrics=preprocess_logits,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
checkpoint_root = Path(training_args.output_dir)
checkpoints = sorted(checkpoint_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
final_metrics = trainer.evaluate()
print(json.dumps(final_metrics, indent=2, sort_keys=True))

# %% [markdown]
# ## 10. Per-scene records, calibration diagnostic and jury previews
#
# This second validation pass stores inspectable evidence instead of aggregate-only claims. The
# confidence calculation is a diagnostic ECE of raw maximum softmax, not a calibrated probability.

# %%
from PIL import ImageDraw

PALETTE = np.array([
    [122, 132, 146],  # background
    [247, 166, 77],   # building
    [245, 232, 112],  # road
    [44, 174, 235],   # water
    [191, 134, 78],   # barren
    [57, 166, 93],    # forest
    [157, 207, 89],   # agricultural
], dtype=np.uint8)
calibration_count = np.zeros(15, dtype=np.int64)
calibration_confidence = np.zeros(15, dtype=np.float64)
calibration_correct = np.zeros(15, dtype=np.int64)
scene_rows = []
preview_dir = EVAL_DIR / "jury_previews"
preview_dir.mkdir(exist_ok=True)
trainer.model.eval()


def scene_scores(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    scores = {}
    for class_id, label in ID2LABEL.items():
        predicted = prediction == class_id
        expected = reference == class_id
        intersection = int(np.logical_and(predicted, expected).sum())
        union = int(np.logical_or(predicted, expected).sum())
        denominator = int(predicted.sum() + expected.sum())
        scores[f"iou_{label}"] = intersection / union if union else None
        scores[f"dice_{label}"] = 2 * intersection / denominator if denominator else None
    return scores


for position in tqdm(range(len(validation_dataset)), desc="Validation evidence"):
    item = validation_dataset[position]
    pixel_values = item["pixel_values"].unsqueeze(0).to("cuda")
    reference = item["labels"].numpy()
    with torch.inference_mode():
        logits = trainer.model(pixel_values=pixel_values).logits
        logits = F.interpolate(
            logits, size=reference.shape, mode="bilinear", align_corners=False
        )[0]
        probabilities = logits.softmax(dim=0)
        confidence, prediction_tensor = probabilities.max(dim=0)
    prediction = prediction_tensor.cpu().numpy()
    confidence = confidence.float().cpu().numpy()
    valid = reference != 255
    bins = np.minimum((confidence[valid] * 15).astype(int), 14)
    correct = prediction[valid] == reference[valid]
    for bin_id in range(15):
        selected = bins == bin_id
        calibration_count[bin_id] += int(selected.sum())
        calibration_confidence[bin_id] += float(confidence[valid][selected].sum())
        calibration_correct[bin_id] += int(correct[selected].sum())
    source_index = int(item["source_index"])
    source_path = Path(validation_source.files[source_index]["image"])
    scene_rows.append({
        "validation_position": position,
        "source_id": f"{source_path.parent.parent.name}/{source_path.name}",
        **scene_scores(prediction, reference),
    })
    if position < 12:
        with Image.open(source_path) as source:
            rgb = source.convert("RGB").resize(
                (CFG.crop_size, CFG.crop_size), Image.Resampling.BILINEAR
            )
        ground_truth = np.zeros((*reference.shape, 3), dtype=np.uint8)
        ground_truth[valid] = PALETTE[reference[valid]]
        predicted_rgb = PALETTE[prediction]
        overlay = Image.blend(rgb, Image.fromarray(predicted_rgb), 0.48)
        canvas = Image.new("RGB", (CFG.crop_size * 3, CFG.crop_size + 32), "white")
        canvas.paste(rgb, (0, 32))
        canvas.paste(Image.fromarray(ground_truth), (CFG.crop_size, 32))
        canvas.paste(overlay, (CFG.crop_size * 2, 32))
        ImageDraw.Draw(canvas).text(
            (10, 9), "SOURCE                      LABELLED REFERENCE                      MODEL OVERLAY",
            fill="black",
        )
        canvas.save(preview_dir / f"validation_{position:03d}.png")

with (EVAL_DIR / "validation_scene_metrics.jsonl").open("w", encoding="utf-8") as handle:
    for row in scene_rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
total_pixels = max(1, int(calibration_count.sum()))
calibration_bins, ece = [], 0.0
for bin_id in range(15):
    count = int(calibration_count[bin_id])
    average_confidence = calibration_confidence[bin_id] / count if count else None
    accuracy = calibration_correct[bin_id] / count if count else None
    if count:
        ece += count / total_pixels * abs(average_confidence - accuracy)
    calibration_bins.append({
        "lower": bin_id / 15,
        "upper": (bin_id + 1) / 15,
        "count": count,
        "mean_raw_softmax": average_confidence,
        "pixel_accuracy": accuracy,
    })
calibration_diagnostic = {
    "status": "uncalibrated diagnostic only",
    "pixel_ece": ece,
    "bins": calibration_bins,
}
(EVAL_DIR / "calibration_diagnostic.json").write_text(
    json.dumps(calibration_diagnostic, indent=2), encoding="utf-8"
)
print({"scene_rows": len(scene_rows), "raw_softmax_pixel_ece": ece, "previews": 12})

# %% [markdown]
# ## 11. Release gate, raw manifest and immutable artifact hashes

# %%
import platform
import transformers

required_iou = {
    "eval_iou_water": 0.35,
    "eval_iou_forest": 0.35,
    "eval_iou_agricultural": 0.35,
}
release_checks = {
    "mean_iou_at_least_0_45": final_metrics.get("eval_mean_iou", 0) >= 0.45,
    **{
        f"{name}_at_least_{threshold:.2f}": final_metrics.get(name, 0) >= threshold
        for name, threshold in required_iou.items()
    },
}
release_candidate = all(release_checks.values())
trainer.save_model(OUTPUT_DIR, safe_serialization=True)
processor.save_pretrained(OUTPUT_DIR)

manifest = {
    "purpose": "SatQuery single-image semantic-mask transfer baseline",
    "base_model": CFG.base_model,
    "base_revision": CFG.base_revision,
    "dataset": "LoveDA official train/validation splits",
    "dataset_license_constraint": "academic/non-commercial; Google Earth terms also apply",
    "selected_train_examples": len(train_indexes),
    "official_train_examples": len(train_source),
    "train_fraction": len(train_indexes) / len(train_source),
    "train_selection_sha256": selection_sha256,
    "official_validation_examples": len(validation_source),
    "labels": ID2LABEL,
    "metrics": final_metrics,
    "release_checks": release_checks,
    "release_candidate": release_candidate,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "config": asdict(CFG),
}
(OUTPUT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

hashes = {}
for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUTPUT_DIR / "sha256_manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
(EVAL_DIR / "validation_metrics.json").write_text(
    json.dumps(final_metrics, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps({"release_candidate": release_candidate, "checks": release_checks}, indent=2))

# %% [markdown]
# ## 12. Fresh local reload smoke test

# %%
del trainer, model
torch.cuda.empty_cache()
reloaded_processor = SegformerImageProcessor.from_pretrained(OUTPUT_DIR)
reloaded = SegformerForSemanticSegmentation.from_pretrained(
    OUTPUT_DIR, use_safetensors=True, trust_remote_code=False
).to("cuda").eval()
smoke = validation_source[0]
smoke_image = smoke["image"].permute(1, 2, 0).byte().numpy()
smoke_batch = reloaded_processor(images=smoke_image, return_tensors="pt").to("cuda")
with torch.inference_mode():
    smoke_logits = reloaded(**smoke_batch).logits
assert smoke_logits.shape[1] == len(LABELS)
print({"PASS": "fresh safetensors reload", "logits": list(smoke_logits.shape)})

# %% [markdown]
# ## 13. Optional Hugging Face upload — model repository only, no paid endpoint
#
# Upload only if the release gate passes. A Hugging Face model repository stores artifacts; it
# does not provide always-on free GPU inference. Keep it private until the metrics and model card
# have been reviewed.

# %%
from huggingface_hub import HfApi

if CFG.push_to_hub:
    if not release_candidate:
        raise RuntimeError("Release gate failed. Do not upload or claim this checkpoint yet.")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(
        CFG.output_repo,
        repo_type="model",
        private=CFG.make_repo_private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=CFG.output_repo,
        repo_type="model",
        folder_path=OUTPUT_DIR,
        commit_message="Upload evaluated SatQuery LoveDA SegFormer checkpoint",
    )
    print({"repo": CFG.output_repo, "commit": commit.oid})
else:
    print("Hub upload disabled. Download best-model/ and evaluation/ before stopping Kaggle.")

# %% [markdown]
# ## 14. Safe stop
#
# Download `best-model/`, `evaluation/`, and the final notebook output. Then save a Kaggle version
# and stop the GPU session. Do not call the model calibrated or India-validated merely because the
# notebook completed.

# %%
print({
    "training_completed": True,
    "real_validation_metrics_saved": (EVAL_DIR / "validation_metrics.json").exists(),
    "safetensors_saved": (OUTPUT_DIR / "model.safetensors").exists(),
    "release_candidate": release_candidate,
})
print("SAFE STOP: download artifacts, save the notebook version, then turn off the GPU session.")
