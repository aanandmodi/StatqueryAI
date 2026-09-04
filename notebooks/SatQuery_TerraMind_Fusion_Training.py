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
# # SatQuery optical + SAR fusion — TerraMind
#
# Standalone free Kaggle/Colab training for the mandatory multi-sensor specialist. It consumes
# co-registered Sentinel-2 L2A (12 bands) and Sentinel-1 GRD (VV/VH), fine-tunes a TerraMind
# backbone, records fused/S2-only/S1-only ablations, and produces patch-level evidence maps.
# Qwen never receives raw multispectral/SAR tensors. No paid service is provisioned.

# %% [markdown]
# ## 0. Install and configure

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "terratorch>=1.2.5,<2", "setuptools<81", "scikit-learn>=1.6,<2",
    "huggingface_hub>=0.36,<1", "lmdb>=1.5,<2", "safetensors>=0.5,<1",
    "pandas>=2.2,<3", "pyarrow>=18,<23", "tqdm>=4.66,<5"])
print("Installed. Restart once only if a stale preinstalled package causes an import error.")

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Config:
    dataset_repo: str = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
    dataset_revision: str = "7a83ae701109ec232d40665b9677fc46310a1a8b"
    seed: int = 42
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 5
    max_train_patches: int = 8_000
    max_validation_patches: int = 1_500
    head_learning_rate: float = 2e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    modality_drop_rate: float = 0.10
    workers: int = 2
    repo_id: str = "aanandmodi/satquery-terramind-s1-s2-fusion"
    push_to_hub: bool = False


CFG = Config()
ROOT = Path("/kaggle/working/satquery-fusion" if Path("/kaggle/working").exists() else "/content/satquery-fusion")
DATA, ARTIFACTS = ROOT / "data", ROOT / "artifacts"
for path in (DATA, ARTIFACTS):
    path.mkdir(parents=True, exist_ok=True)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a free Kaggle/Colab GPU before continuing.")
vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print({"config": asdict(CFG), "gpu": torch.cuda.get_device_name(0), "vram_gib": round(vram_gib, 2)})

# %% [markdown]
# ## 1. Download the bounded co-registered S1/S2 subset
#
# The project guide requests BigEarthNet-MM-style co-registered optical/SAR training. This public
# LMDB subset contains all official 12 S2 bands and the paired S1 VV/VH channels while fitting on a
# free Kaggle disk. Revisions are immutable. The model card must retain the Europe-to-India domain
# gap warning.

# %%
from huggingface_hub import HfApi, snapshot_download

info = HfApi().dataset_info(CFG.dataset_repo, revision=CFG.dataset_revision)
assert info.sha == CFG.dataset_revision
dataset_root = Path(snapshot_download(
    CFG.dataset_repo, repo_type="dataset", revision=CFG.dataset_revision,
    local_dir=DATA / "bigearthnet-mm-subset",
    allow_patterns=["BENv2_lithuania_summer.lmdb/*", "metadata_lithuania_summer.parquet", "README.md"],
))
LMDB_DIR = dataset_root / "BENv2_lithuania_summer.lmdb"
METADATA = dataset_root / "metadata_lithuania_summer.parquet"
required = [LMDB_DIR / "data.mdb", LMDB_DIR / "lock.mdb", METADATA]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Incomplete S1/S2 dataset: {missing}")

# %% [markdown]
# ## 2. Reader, integrity checks, deterministic splits

# %%
import lmdb
from safetensors.numpy import load as load_safetensors_bytes

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


class PairStore:
    def __init__(self, path: Path, metadata: pd.DataFrame):
        self.path = path
        self.s1_for_s2 = dict(zip(metadata.patch_id.astype(str), metadata.s1_name.astype(str)))
        self.labels = dict(zip(metadata.patch_id.astype(str), metadata.labels))
        self._env = None

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=True, meminit=False)
        return self._env

    def read(self, key: str) -> dict[str, np.ndarray]:
        with self.env.begin(write=False, buffers=True) as transaction:
            payload = transaction.get(str(key).encode("utf-8"))
            payload = bytes(payload) if payload is not None else None
        if payload is None:
            raise KeyError(f"LMDB key missing: {key}")
        return load_safetensors_bytes(bytes(payload))

    def pair(self, patch_id: str):
        s2 = self.read(patch_id)
        s1_name = self.s1_for_s2.get(patch_id)
        if not s1_name:
            raise KeyError(f"S1 mapping missing for {patch_id}")
        s1 = self.read(s1_name)
        if not set(S2_BANDS).issubset(s2) or not set(S1_BANDS).issubset(s1):
            raise ValueError(f"Required bands missing for {patch_id}")
        return s2, s1


metadata = pd.read_parquet(METADATA)
if "contains_seasonal_snow" in metadata:
    metadata = metadata[~metadata.contains_seasonal_snow]
if "contains_cloud_or_shadow" in metadata:
    metadata = metadata[~metadata.contains_cloud_or_shadow]
classes = sorted({str(label) for labels in metadata.labels for label in labels})


def bounded_ids(split_names: tuple[str, ...], limit: int) -> list[str]:
    values = metadata[metadata.split.astype(str).str.lower().isin(split_names)].patch_id.astype(str).tolist()
    return sorted(values, key=lambda value: hashlib.sha256(f"{value}:{CFG.seed}".encode()).hexdigest())[:limit]


train_ids = bounded_ids(("train",), CFG.max_train_patches)
validation_ids = bounded_ids(("validation", "val"), CFG.max_validation_patches)
assert train_ids and validation_ids
assert set(train_ids).isdisjoint(validation_ids), "Patch leakage between train and validation"
store = PairStore(LMDB_DIR, metadata)
for patch_id in (train_ids[0], validation_ids[0]):
    store.pair(patch_id)
print({"train": len(train_ids), "validation": len(validation_ids), "classes": len(classes),
       "dataset_sha": CFG.dataset_revision})

# %% [markdown]
# ## 3. TerraMind-normalized dataset and fusion head
#
# Band order and normalization match the published TerraMind S2L2A and S1GRD inputs. Spatial
# patch logits are upsampled as evidence; they are not described as pixel-perfect segmentation.

# %%
from sklearn.metrics import average_precision_score, f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from terratorch.registry import BACKBONE_REGISTRY

TM_S2_MEAN = torch.tensor([1390.458,1503.317,1718.197,1853.910,2199.100,2779.975,
                           2987.011,3083.234,3132.220,3162.988,2424.884,1857.648])
TM_S2_STD = torch.tensor([2106.761,2141.107,2038.973,2134.138,2085.321,1889.926,
                          1820.257,1871.918,1753.829,1797.379,1434.261,1334.311])
TM_S1_MEAN = torch.tensor([-12.599, -20.293])
TM_S1_STD = torch.tensor([5.195, 5.890])


class FusionDataset(Dataset):
    def __init__(self, patch_ids: list[str], training: bool):
        self.patch_ids, self.training = patch_ids, training

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index: int):
        patch_id = self.patch_ids[index]
        s2_raw, s1_raw = store.pair(patch_id)
        # Native Sentinel bands are on different grids (10/20/60 m): resize each before stacking.
        def aligned_bands(raw, names):
            return torch.stack([F.interpolate(
                torch.as_tensor(np.asarray(raw[name]), dtype=torch.float32)[None, None],
                (CFG.image_size, CFG.image_size), mode="bilinear", align_corners=False)[0, 0]
                for name in names])
        s2, s1 = aligned_bands(s2_raw, S2_BANDS), aligned_bands(s1_raw, S1_BANDS)
        s2 = (s2 - TM_S2_MEAN[:, None, None]) / TM_S2_STD[:, None, None]
        s1 = (s1 - TM_S1_MEAN[:, None, None]) / TM_S1_STD[:, None, None]
        if self.training and random.random() < 0.5:
            s2, s1 = s2.flip(-1), s1.flip(-1)
        if self.training and random.random() < 0.5:
            s2, s1 = s2.flip(-2), s1.flip(-2)
        labels = set(store.labels[patch_id])
        target = torch.tensor([float(name in labels) for name in classes])
        return {"s2": s2, "s1": s1, "target": target}


class FusionExpert(torch.nn.Module):
    def __init__(self, backbone, embedding_dim: int, num_classes: int, image_size: int = 224):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        self.num_classes, self.image_size = num_classes, image_size

    def forward(self, *, s2=None, s1=None):
        inputs = {}
        if s2 is not None:
            inputs["S2L2A"] = s2
        if s1 is not None:
            inputs["S1GRD"] = s1
        if not inputs:
            raise ValueError("At least one modality is required")
        tokens = self.backbone(inputs)[-1]
        patch_logits = self.classifier(self.norm(tokens))
        class_logits = patch_logits.mean(1)
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("Backbone tokens cannot be reshaped to a square evidence grid")
        evidence = patch_logits.transpose(1, 2).reshape(tokens.shape[0], self.num_classes, side, side)
        evidence = F.interpolate(evidence, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return class_logits, evidence


train_loader = DataLoader(FusionDataset(train_ids, True), batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.workers, pin_memory=True)
validation_loader = DataLoader(FusionDataset(validation_ids, False), batch_size=CFG.batch_size,
                               shuffle=False, num_workers=CFG.workers, pin_memory=True)
backbone_name = "terramind_v1_tiny" if vram_gib < 22 else "terramind_v1_base"
backbone = BACKBONE_REGISTRY.build(backbone_name, pretrained=True,
                                   modalities=["S2L2A", "S1GRD"], merge_method="mean").cuda()
probe = next(iter(train_loader))
with torch.inference_mode():
    embedding_dim = backbone({"S2L2A": probe["s2"].cuda(), "S1GRD": probe["s1"].cuda()})[-1].shape[-1]
model = FusionExpert(backbone, embedding_dim, len(classes), CFG.image_size).cuda()
del probe

# %% [markdown]
# ## 4. Staged fine-tuning and mandatory modality ablation

# %%
from safetensors.torch import save_file
from tqdm.auto import tqdm

use_bf16 = torch.cuda.is_bf16_supported()
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
for parameter in model.backbone.parameters():
    parameter.requires_grad = False


@torch.inference_mode()
def evaluate(mode: str) -> dict[str, float]:
    model.eval()
    all_targets, all_probabilities = [], []
    for batch in validation_loader:
        s2 = batch["s2"].cuda() if mode in {"fused", "s2"} else None
        s1 = batch["s1"].cuda() if mode in {"fused", "s1"} else None
        with torch.autocast("cuda", dtype=autocast_dtype):
            logits, _ = model(s2=s2, s1=s1)
        all_targets.append(batch["target"].numpy())
        all_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    targets, probabilities = np.concatenate(all_targets), np.concatenate(all_probabilities)
    predictions = probabilities >= 0.5
    return {"macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
            "macro_average_precision": float(average_precision_score(targets, probabilities, average="macro"))}


head_parameters = [parameter for name, parameter in model.named_parameters() if "backbone" not in name]
optimizer = torch.optim.AdamW(head_parameters, lr=CFG.head_learning_rate, weight_decay=CFG.weight_decay)
best_f1 = -1.0
history = []
for epoch in range(CFG.epochs):
    if epoch == 1:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": CFG.head_learning_rate},
            {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": CFG.backbone_learning_rate},
        ], weight_decay=CFG.weight_decay)
    model.train()
    if epoch == 0:
        model.backbone.eval()
    running = 0.0
    for batch in tqdm(train_loader, desc=f"fusion {epoch + 1}/{CFG.epochs}"):
        optimizer.zero_grad(set_to_none=True)
        s2, s1, target = batch["s2"].cuda(), batch["s1"].cuda(), batch["target"].cuda()
        draw = random.random()
        s2_input = None if draw < CFG.modality_drop_rate else s2
        s1_input = None if CFG.modality_drop_rate <= draw < 2 * CFG.modality_drop_rate else s1
        with torch.autocast("cuda", dtype=autocast_dtype):
            logits, _ = model(s2=s2_input, s1=s1_input)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
    fused = evaluate("fused")
    row = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1), **fused}
    history.append(row)
    print(row)
    if fused["macro_f1"] > best_f1:
        best_f1 = fused["macro_f1"]
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                  ARTIFACTS / "model.safetensors")

# All release metrics must describe the saved best weights, not the last epoch.
from safetensors.torch import load_file
model.load_state_dict(load_file(ARTIFACTS / "model.safetensors"), strict=True)
ablation = {mode: evaluate(mode) for mode in ("fused", "s2", "s1")}
(ARTIFACTS / "config.json").write_text(json.dumps({
    "artifact_version": "satquery-pair-v2", "embedding_dim": embedding_dim,
    "mask_supervision": "none; scene labels only", "calibrated": False,
    "architecture": "TerraMind S2L2A+S1GRD mean fusion with multilabel patch-evidence head",
    "backbone": backbone_name, "dataset_revision": CFG.dataset_revision, "config": asdict(CFG),
    "classes": classes, "s2_bands": S2_BANDS, "s1_bands": S1_BANDS,
    "normalization": {"s2_mean": TM_S2_MEAN.tolist(), "s2_std": TM_S2_STD.tolist(),
                      "s1_mean": TM_S1_MEAN.tolist(), "s1_std": TM_S1_STD.tolist()},
    "ablation": ablation, "history": history,
    "score_semantics": "Per-class sigmoid scores require validation-set calibration before probability claims.",
    "domain_gap": "European Sentinel training does not establish performance on Indian Cartosat/RISAT imagery.",
}, indent=2), encoding="utf-8")
print(json.dumps(ablation, indent=2))

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
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
    if not token:
        raise RuntimeError("Add HF_TOKEN in notebook Secrets; never paste it into source.")
    api = HfApi(token=token)
    api.create_repo(CFG.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=CFG.repo_id, repo_type="model", folder_path=ARTIFACTS,
                      commit_message="Release audited SatQuery TerraMind S1/S2 fusion specialist")

gate = {"weights": (ARTIFACTS / "model.safetensors").is_file(),
        "config": (ARTIFACTS / "config.json").is_file(),
        "hash_manifest": (ARTIFACTS / "sha256_manifest.json").is_file(),
        "fused_ablation": "fused" in ablation, "s2_ablation": "s2" in ablation,
        "s1_ablation": "s1" in ablation, "paid_endpoint_created": False}
print(json.dumps(gate, indent=2))
assert all(value for key, value in gate.items() if key != "paid_endpoint_created")
assert not gate["paid_endpoint_created"]
print("PASS — download artifacts or verify Hub upload, save the notebook, then stop the GPU.")
