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
# # SatQuery learned optical/SAR flood segmentation — TerraMind + Sen1Floods11
#
# Free Kaggle/Colab training for a genuinely pixel-supervised fusion specialist. It consumes
# co-registered Sentinel-2 L1C + Sentinel-1 GRD, learns from Sen1Floods11 `LabelHand` pixels,
# evaluates the official disjoint validation split, records fused/S2-only/S1-only ablations, and
# exports per-chip raw prediction statistics. It does not establish Cartosat/RISAT, India-domain,
# flood-depth, cause, or universal water accuracy. Hub upload is guarded and off by default.

# %% [markdown]
# ## 0. Install and configure
#
# Run this setup cell FIRST. If it says **SETUP COMPLETE — RESTART KERNEL**, restart only the
# Python kernel (keep the session/files), then Run All again. Installing NumPy into a process
# that already loaded its binary extensions is unsafe. Do not retry section 3 in that process.
# No dataset or model checkpoint is deleted. No automatic process kill/restart loop is used.

# %%
import subprocess
import sys
import os
from importlib import metadata
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
FUSION_PINS = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "terratorch": "1.2.13",
    "torchgeo": "0.9.0",
    "albumentations": "2.0.8",
    "albucore": "0.0.24",
    "opencv-python-headless": "4.11.0.86",
    "huggingface-hub": "0.36.2",
    # Pin the whole HF family: torchmetrics imports Transformers even for image-only training.
    # A preinstalled Transformers 5.x must not remain beside Hub 0.36.x.
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "diffusers": "0.35.1",
    "peft": "0.17.1",
    "accelerate": "1.7.0",
}


def fusion_package_versions():
    return {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }


def setup_fusion_environment(environment_dir):
    """Repair on-disk NumPy/SciPy, preserve cloud CUDA wheels, never hot-reload binaries."""
    before = fusion_package_versions()
    cuda_packages = [name for name in ("torch", "torchvision", "torchaudio") if name in before]
    if "torch" not in cuda_packages or "torchvision" not in cuda_packages:
        raise RuntimeError("Use a Kaggle GPU image with its preinstalled torch/torchvision wheels.")
    environment_dir = Path(environment_dir)
    environment_dir.mkdir(parents=True, exist_ok=True)
    constraints = environment_dir / "constraints.txt"
    constraints.write_text("\n".join(
        [f"{name}=={version}" for name, version in FUSION_PINS.items()]
        + [f"{name}=={before[name]}" for name in cuda_packages]
    ) + "\n", encoding="utf-8")
    packages = [f"{name}=={version}" for name, version in FUSION_PINS.items()]
    packages += ["setuptools<81", "safetensors>=0.5,<1", "rasterio>=1.4.3,<2", "tqdm>=4.66,<5"]
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
        "--prefer-binary", "--constraint", str(constraints), *packages,
    ])
    # A NEW interpreter distinguishes an on-disk corrupt install from the notebook's stale modules.
    numeric_probe = (
        "import numpy as np; import numpy.testing; import scipy; from scipy import special; "
        "np.testing.assert_allclose(special.expit(np.array([0.0])), [0.5]); "
        "print('Numeric binary check:', np.__version__, scipy.__version__)"
    )
    probe = subprocess.run([sys.executable, "-c", numeric_probe], capture_output=True, text=True)
    repaired = probe.returncode != 0
    if repaired:
        print("Fresh-process numeric import failed; reinstalling only the pinned NumPy/SciPy wheels.")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
            "--no-deps", "--only-binary=:all:",
            f"numpy=={FUSION_PINS['numpy']}", f"scipy=={FUSION_PINS['scipy']}",
        ])
        probe = subprocess.run([sys.executable, "-c", numeric_probe], capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError("Numeric environment still fails in a fresh interpreter:\n" + probe.stderr[-6000:])
    print(probe.stdout.strip())
    after = fusion_package_versions()
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    stale = any(
        name in sys.modules and getattr(sys.modules[name], "__version__", None) != after.get(name)
        for name in ("numpy", "scipy")
    )
    if changed or repaired or stale:
        print("Changed packages:", changed)
        raise SystemExit(
            "SETUP COMPLETE — RESTART KERNEL, then Run All again. Do not end/delete the session. "
            "This intentional stop prevents stale NumPy binary imports; downloaded data is kept."
        )
    print("PASS: pinned environment unchanged; safe to continue to the import preflight.")


setup_fusion_environment(
    Path("/kaggle/working/satquery-fusion-environment")
    if Path("/kaggle/working").exists() else Path("/content/satquery-fusion-environment")
)

# %% [markdown]
# ### 0b. Verify the complete import chain BEFORE downloading data
#
# Both a clean subprocess and the notebook kernel must pass. This catches a stale NumPy/SciPy
# installation at setup instead of after downloading the entire dataset. No model is downloaded here.

# %%
FUSION_IMPORT_PROBE = """
from importlib.metadata import version
print({'huggingface_stack': {name: version(name) for name in
      ('transformers', 'huggingface-hub', 'tokenizers', 'diffusers', 'peft', 'accelerate')}}, flush=True)
import numpy as np
import numpy.testing
import scipy
from scipy import special
import albumentations
import rasterio
import torch
import torchvision
from transformers import AutoModel, AutoTokenizer
import diffusers
import peft
from terratorch.registry import BACKBONE_REGISTRY
np.testing.assert_allclose(special.expit(np.array([0.0])), [0.5])
assert torch.cuda.is_available(), 'Select GPU T4/P100, not CPU or TPU.'
assert callable(BACKBONE_REGISTRY.build)
print({'PASS': 'NumPy -> SciPy -> Albumentations -> TerraTorch; CUDA ready',
       'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
       'gpu': torch.cuda.get_device_name(0)})
"""
clean_import = subprocess.run(
    [sys.executable, "-c", FUSION_IMPORT_PROBE], capture_output=True, text=True
)
if clean_import.returncode:
    raise RuntimeError("Fresh-process TerraTorch preflight failed. Preserve this output:\n"
                       + clean_import.stdout[-2000:] + "\n" + clean_import.stderr[-8000:])
print(clean_import.stdout.strip())
try:
    exec(FUSION_IMPORT_PROBE)
except Exception as exc:
    raise RuntimeError(
        "Fresh-process imports passed but the notebook kernel is stale. Restart the kernel "
        "(keep session files), then Run All from setup. Do not use importlib.reload on NumPy."
    ) from exc

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import random

import numpy as np
import rasterio
import torch


@dataclass(frozen=True)
class Config:
    dataset_root: str = ""  # Attached official Sen1Floods11 v1.1; blank = detect.
    seed: int = 42
    image_size: int = 224
    batch_size: int = 2
    epochs: int = 20
    decoder_version: str = "conv-r2"
    training_revision: str = "r2-finite-fp32"
    smoke_test: bool = False
    smoke_train_chips: int = 32
    smoke_validation_chips: int = 16
    head_learning_rate: float = 2e-4
    backbone_learning_rate: float = 5e-6
    weight_decay: float = 0.01
    modality_drop_rate: float = 0.10
    workers: int = 0
    resume: bool = True
    run_test_once: bool = False
    # Set only after validation review; the notebook does not invent a release target.
    release_minimum_flood_iou: float | None = None
    repo_id: str = "aanandmodi/satquery-terramind-sen1floods11-segmentation"
    push_to_hub: bool = False


CFG = Config()
ROOT = Path(
    "/kaggle/working/satquery-fusion-segmentation-r2"
    if Path("/kaggle/working").exists()
    else "/content/satquery-fusion-segmentation-r2"
)
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a free Kaggle/Colab GPU before continuing.")
vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(
    {
        "config": asdict(CFG),
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(vram_gib, 2),
    }
)

# %% [markdown]
# ## 1. Locate the attached official Sen1Floods11 data
#
# Expected: `data/S2L1CHand`, `data/S1GRDHand`, `data/LabelHand`, and official
# `splits/flood_{train,valid,test}_data.txt`. Missing labels or splits fail closed.


# %%
def looks_like_dataset(path: Path) -> bool:
    return all(
        item.exists()
        for item in (
            path / "data/S2L1CHand",
            path / "data/S1GRDHand",
            path / "data/LabelHand",
            path / "splits/flood_train_data.txt",
            path / "splits/flood_valid_data.txt",
            path / "splits/flood_test_data.txt",
        )
    )


def locate_dataset() -> Path:
    if CFG.dataset_root:
        configured = Path(CFG.dataset_root)
        if looks_like_dataset(configured):
            return configured
        raise FileNotFoundError(
            f"Configured Sen1Floods11 root is incomplete: {configured}"
        )
    candidates = []
    for root in (Path("/kaggle/input"), Path("/content")):
        if root.exists():
            candidates += [
                item.parent.parent
                for item in root.glob("**/splits/flood_train_data.txt")
                if looks_like_dataset(item.parent.parent)
            ]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Attach exactly one official Sen1Floods11 v1.1 dataset or set CFG.dataset_root. "
            f"Found: {[str(item) for item in candidates]}"
        )
    return candidates[0]


DATASET = locate_dataset()
DATA = DATASET / "data"
print({"sen1floods11_root": str(DATASET)})

# %% [markdown]
# ## 2. Reconstruct official disjoint splits and verify every triplet

# %%
SUFFIXES = ("_S1Hand", "_S2Hand", "_LabelHand")


def chip_id(line: str) -> str:
    value = Path(line.strip()).stem
    for suffix in SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    if not value:
        raise ValueError(f"Invalid split row: {line!r}")
    return value


def read_split(name: str) -> list[str]:
    values = [
        chip_id(line)
        for line in (DATASET / "splits" / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Split {name} is empty or contains duplicate chip IDs")
    return values


def triplet(identifier: str) -> tuple[Path, Path, Path]:
    paths = (
        DATA / "S2L1CHand" / f"{identifier}_S2Hand.tif",
        DATA / "S1GRDHand" / f"{identifier}_S1Hand.tif",
        DATA / "LabelHand" / f"{identifier}_LabelHand.tif",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Sen1Floods11 triplet for {identifier}: {missing}"
        )
    return paths


train_ids = read_split("flood_train_data.txt")
validation_ids = read_split("flood_valid_data.txt")
test_ids = read_split("flood_test_data.txt")
assert set(train_ids).isdisjoint(validation_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(validation_ids).isdisjoint(test_ids)
if CFG.smoke_test:
    train_ids = train_ids[: CFG.smoke_train_chips]
    validation_ids = validation_ids[: CFG.smoke_validation_chips]
for identifier in train_ids + validation_ids + test_ids:
    triplet(identifier)
split_manifest = {
    "identity": "official Sen1Floods11 v1.1 chip ID",
    "source_root": str(DATASET),
    "dataset_reference": "https://github.com/cloudtostreet/Sen1Floods11",
    "official_split_sha256": {
        name: hashlib.sha256((DATASET / "splits" / name).read_bytes()).hexdigest()
        for name in (
            "flood_train_data.txt",
            "flood_valid_data.txt",
            "flood_test_data.txt",
        )
    },
    "smoke_test": CFG.smoke_test,
    "train": train_ids,
    "validation": validation_ids,
    "test": test_ids,
}
(ARTIFACTS / "split_manifest.json").write_text(
    json.dumps(split_manifest, indent=2), encoding="utf-8"
)
print(
    {
        name: len(values)
        for name, values in split_manifest.items()
        if isinstance(values, list)
    }
)

# %% [markdown]
# ## 3. Pixel-supervised dataset and TerraMind segmentation head
#
# TerraMind's published Sen1Floods11 modality statistics are used. Label `-1` stays ignored;
# labels are resized only with nearest-neighbour sampling.

# %%
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from terratorch.registry import BACKBONE_REGISTRY

S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
S1_BANDS = ["VV", "VH"]
TM_S2_MEAN = torch.tensor(
    [
        2357.089,
        2137.385,
        2018.788,
        2082.986,
        2295.651,
        2854.537,
        3122.849,
        3040.560,
        3306.481,
        1473.847,
        506.070,
        2472.825,
        1838.929,
    ]
)
TM_S2_STD = torch.tensor(
    [
        1624.683,
        1675.806,
        1557.708,
        1833.702,
        1823.738,
        1733.977,
        1732.131,
        1679.732,
        1727.260,
        1024.687,
        442.165,
        1331.411,
        1160.419,
    ]
)
TM_S1_MEAN = torch.tensor([-12.599, -20.293])
TM_S1_STD = torch.tensor([5.195, 5.890])


def same_grid(paths: tuple[Path, Path, Path]) -> None:
    with rasterio.open(paths[0]) as first:
        reference = (first.width, first.height, first.crs, first.transform)
    for path in paths[1:]:
        with rasterio.open(path) as source:
            if (source.width, source.height, source.crs) != reference[
                :3
            ] or not source.transform.almost_equals(reference[3]):
                raise ValueError(f"Sen1Floods11 triplet is not co-registered: {paths}")


class FloodDataset(Dataset):
    def __init__(self, identifiers: list[str], training: bool):
        self.identifiers, self.training = identifiers, training

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index: int):
        identifier = self.identifiers[index]
        paths = triplet(identifier)
        same_grid(paths)
        with rasterio.open(paths[0]) as source:
            if source.count != 13:
                raise ValueError(f"{identifier}: expected 13 Sentinel-2 L1C bands")
            s2 = source.read().astype(np.float32)
            s2_valid = (source.read_masks() > 0).all(0) & np.isfinite(s2).all(0)
        with rasterio.open(paths[1]) as source:
            if source.count != 2:
                raise ValueError(f"{identifier}: expected Sentinel-1 VV/VH")
            s1 = source.read().astype(np.float32)
            s1_valid = (source.read_masks() > 0).all(0) & np.isfinite(s1).all(0)
        valid = s2_valid & s1_valid
        with rasterio.open(paths[2]) as source:
            label = source.read(1).astype(np.int64)
            valid &= source.dataset_mask() > 0
        unexpected = set(np.unique(label).tolist()) - {-1, 0, 1}
        if unexpected:
            raise ValueError(f"{identifier}: unsupported labels {sorted(unexpected)}")
        label[~valid] = -1
        # Ignore missing input pixels AND replace their input values before any
        # interpolation/attention: NaN * zero is still NaN. Never repair logits or weights.
        s2 = np.where(s2_valid[None], s2, TM_S2_MEAN.numpy()[:, None, None])
        s1 = np.where(s1_valid[None], s1, TM_S1_MEAN.numpy()[:, None, None])
        s2 = F.interpolate(
            torch.from_numpy(s2)[None],
            (CFG.image_size, CFG.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        s1 = F.interpolate(
            torch.from_numpy(s1)[None],
            (CFG.image_size, CFG.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        target = F.interpolate(
            torch.from_numpy(label.astype(np.float32))[None, None],
            (CFG.image_size, CFG.image_size),
            mode="nearest",
        )[0, 0].long()
        support = F.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None],
                                (CFG.image_size, CFG.image_size), mode="bilinear", align_corners=False)[0, 0]
        target[support < 1 - 1e-6] = -1
        s2 = (s2 - TM_S2_MEAN[:, None, None]) / TM_S2_STD[:, None, None]
        s1 = (s1 - TM_S1_MEAN[:, None, None]) / TM_S1_STD[:, None, None]
        if not torch.isfinite(s2).all() or not torch.isfinite(s1).all():
            raise FloatingPointError(f"Non-finite normalized inputs: {identifier}")
        if self.training and random.random() < 0.5:
            s2, s1, target = s2.flip(-1), s1.flip(-1), target.flip(-1)
        if self.training and random.random() < 0.5:
            s2, s1, target = s2.flip(-2), s1.flip(-2), target.flip(-2)
        return {"id": identifier, "s2": s2, "s1": s1, "target": target}


class FusionExpert(torch.nn.Module):
    """TerraMind token head trained with real per-pixel flood labels."""

    def __init__(
        self, backbone, embedding_dim: int, num_classes: int, image_size: int = 224,
        decoder_version: str = "legacy"
    ):
        super().__init__()
        self.backbone = backbone
        if decoder_version not in {"legacy", "conv-r2"}:
            raise ValueError("Unsupported fusion decoder version")
        self.decoder_version = decoder_version
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.segmenter = torch.nn.Linear(embedding_dim, num_classes)
        self.num_classes, self.image_size = num_classes, image_size
        if decoder_version == "conv-r2":
            self.refine = torch.nn.Sequential(
                torch.nn.Conv2d(embedding_dim, 128, 3, padding=1), torch.nn.GroupNorm(8, 128), torch.nn.GELU(),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(128, 64, 3, padding=1), torch.nn.GroupNorm(8, 64), torch.nn.GELU(),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(64, num_classes, 3, padding=1))

    def forward(self, *, s2=None, s1=None):
        inputs = {}
        if s2 is not None:
            inputs["S2L1C"] = s2
        if s1 is not None:
            inputs["S1GRD"] = s1
        if not inputs:
            raise ValueError("At least one modality is required")
        tokens = self.backbone(inputs)[-1]
        side = math.isqrt(tokens.shape[1])
        if (
            side * side != tokens.shape[1]
            and math.isqrt(tokens.shape[1] - 1) ** 2 == tokens.shape[1] - 1
        ):
            tokens, side = tokens[:, 1:], math.isqrt(tokens.shape[1] - 1)
        if side * side != tokens.shape[1]:
            raise ValueError(
                "Backbone tokens cannot be reshaped to a square segmentation grid"
            )
        logits = (
            self.segmenter(self.norm(tokens))
            .transpose(1, 2)
            .reshape(tokens.shape[0], self.num_classes, side, side)
        )
        if self.decoder_version == "conv-r2":
            grid = self.norm(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
            detail = self.refine(grid)
            logits = F.interpolate(logits, size=detail.shape[-2:], mode="bilinear", align_corners=False) + detail
        return F.interpolate(
            logits,
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


train_loader = DataLoader(
    FloodDataset(train_ids, True),
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.workers,
    pin_memory=True,
)
validation_loader = DataLoader(
    FloodDataset(validation_ids, False),
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.workers,
    pin_memory=True,
)
test_loader = DataLoader(
    FloodDataset(test_ids, False),
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.workers,
    pin_memory=True,
)
backbone_name = "terramind_v1_tiny"  # Fixed architecture across free GPUs; do not silently change experiments.
backbone = BACKBONE_REGISTRY.build(
    backbone_name, pretrained=True, modalities=["S2L1C", "S1GRD"], merge_method="mean"
).cuda()
probe = next(iter(train_loader))
with torch.inference_mode():
    embedding_dim = backbone(
        {"S2L1C": probe["s2"].cuda(), "S1GRD": probe["s1"].cuda()}
    )[-1].shape[-1]
model = FusionExpert(backbone, embedding_dim, 2, CFG.image_size, decoder_version=CFG.decoder_version).cuda().float()
del probe

# %% [markdown]
# ## 4. Train, resume, and save the best validation-IoU checkpoint

# %%
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

# FP32 recovery profile: first establish a finite training run before benchmarking AMP.
autocast_dtype = torch.float32
scaler = torch.amp.GradScaler("cuda", enabled=False)
for parameter in model.backbone.parameters():
    parameter.requires_grad = False


def segmentation_loss(logits, target):
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Fusion logits are non-finite. Stop; do not export this candidate.")
    if not (target >= 0).any():
        return logits.sum() * 0.0
    cross_entropy = F.cross_entropy(logits, target, ignore_index=-1)
    valid, probability, truth = (
        target >= 0,
        logits.softmax(1)[:, 1],
        (target == 1).float(),
    )
    intersection = (probability * truth * valid).sum()
    denominator = (probability * valid).sum() + (truth * valid).sum()
    return cross_entropy + 1 - (2 * intersection + 1) / (denominator + 1)


# One real labelled training batch, all modality paths, BEFORE the first epoch.
# This checks gradients but does not update weights or consume validation/test for tuning.
preflight = next((batch for batch in train_loader if (batch["target"] >= 0).any()), None)
if preflight is None:
    raise RuntimeError("No labelled training pixels after input validity masking.")
model.eval()
for mode in ("fused", "s2", "s1"):
    model.zero_grad(set_to_none=True)
    logits = model(s2=preflight["s2"].cuda() if mode != "s1" else None,
                   s1=preflight["s1"].cuda() if mode != "s2" else None)
    loss = segmentation_loss(logits, preflight["target"].cuda())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Preflight loss is non-finite: {mode}")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    print({"numerical_preflight": mode, "loss": float(loss.detach()), "gradient_norm": float(norm)})
model.zero_grad(set_to_none=True)
del preflight, logits, loss


def true_runs(mask: np.ndarray) -> list[list[int]]:
    """Compact, exactly reversible row-major runs for a binary mask."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    padded = np.pad(flat, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        [int(start), int(end - start)]
        for start, end in zip(changes[::2], changes[1::2], strict=True)
    ]


@torch.inference_mode()
def evaluate(
    loader, mode: str, prediction_path: Path | None = None
) -> dict[str, float]:
    model.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    fraction_errors, rows = [], []
    for batch in tqdm(loader, desc=f"evaluate {mode}"):
        s2 = batch["s2"].cuda() if mode in {"fused", "s2"} else None
        s1 = batch["s1"].cuda() if mode in {"fused", "s1"} else None
        with torch.autocast("cuda", enabled=False):
            logits = model(s2=s2, s1=s1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite evaluation logits: {mode}, {batch['id']}")
        probabilities = logits.softmax(1)[:, 1].float().cpu().numpy()
        predictions, targets = probabilities >= 0.5, batch["target"].numpy()
        for identifier, prediction, probability, target in zip(
            batch["id"], predictions, probabilities, targets, strict=True
        ):
            valid, reference = target >= 0, target == 1
            confusion += np.bincount(
                target[valid] * 2 + prediction[valid].astype(np.int64), minlength=4
            ).reshape(2, 2)
            predicted_fraction = float(prediction[valid].mean()) if valid.any() else 0.0
            reference_fraction = float(reference[valid].mean()) if valid.any() else 0.0
            if valid.any():
                fraction_errors.append(abs(predicted_fraction - reference_fraction))
            if prediction_path is not None:
                rows.append(
                    {
                        "chip_id": identifier,
                        "split": "validation"
                        if loader is validation_loader
                        else "test",
                        "mode": mode,
                        "valid_pixels": int(valid.sum()),
                        "reference_flood_pixels": int((reference & valid).sum()),
                        "predicted_flood_pixels": int((prediction & valid).sum()),
                        "intersection_pixels": int(
                            (prediction & reference & valid).sum()
                        ),
                        "union_pixels": int(((prediction | reference) & valid).sum()),
                        "mean_flood_score": float(probability[valid].mean())
                        if valid.any()
                        else 0.0,
                        "shape": list(prediction.shape),
                        "prediction_true_runs": true_runs(prediction & valid),
                        "reference_true_runs": true_runs(reference & valid),
                        "invalid_true_runs": true_runs(~valid),
                    }
                )
    true_positive, false_positive = confusion[1, 1], confusion[0, 1]
    false_negative, true_negative = confusion[1, 0], confusion[0, 0]
    flood_union, other_union = (
        true_positive + false_positive + false_negative,
        (true_negative + false_positive + false_negative),
    )
    if not flood_union or not other_union or not fraction_errors:
        raise ValueError("Evaluation has insufficient valid class support; no release metric can be computed.")
    flood_iou = true_positive / flood_union
    other_iou = true_negative / other_union
    metrics = {
        "flood_iou": float(flood_iou),
        "flood_dice": float(
            2
            * true_positive
            / max(1, 2 * true_positive + false_positive + false_negative)
        ),
        "flood_precision": float(
            true_positive / max(1, true_positive + false_positive)
        ),
        "flood_recall": float(true_positive / max(1, true_positive + false_negative)),
        "mean_iou": float(np.nanmean([flood_iou, other_iou])),
        "mean_absolute_flood_fraction_error": float(np.mean(fraction_errors)),
        "valid_pixels": int(confusion.sum()),
        "chips": int(len(loader.dataset)),
    }
    if prediction_path is not None:
        prediction_path.write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
    return metrics


optimizer = torch.optim.AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": CFG.head_learning_rate},
        {"params": list(model.backbone.parameters()), "lr": CFG.backbone_learning_rate},
    ],
    weight_decay=CFG.weight_decay,
)
best_iou, start_epoch, history = -1.0, 0, []
training_state = ARTIFACTS / "training_state.pt"
if (
    CFG.resume
    and training_state.is_file()
    and (ARTIFACTS / "model.safetensors").is_file()
):
    state = torch.load(training_state, map_location="cpu", weights_only=True)
    if state.get("config") != asdict(CFG):
        raise RuntimeError("Refuse old/incompatible training state. Preserve it; use the new r2 output directory.")
    if not all(torch.isfinite(value).all() for value in state["model"].values()):
        raise FloatingPointError("Refuse non-finite resume weights. The old failed 04 is not a warm start.")
    if "model" not in state or "optimizer" not in state:
        raise RuntimeError("Legacy resume state is incomplete. Use a fresh output directory.")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    start_epoch, best_iou, history = (
        int(state["epoch"]),
        float(state["best_iou"]),
        list(state["history"]),
    )
    print({"resumed_after_epoch": start_epoch, "best_flood_iou": best_iou})

for epoch in range(start_epoch, 1 if CFG.smoke_test else CFG.epochs):
    if epoch >= 2:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    model.train()
    if epoch < 2:
        model.backbone.eval()
    running, valid_batches, skipped_batches = 0.0, 0, 0
    for batch in tqdm(train_loader, desc=f"fusion segmentation {epoch + 1}"):
        optimizer.zero_grad(set_to_none=True)
        s2, s1, target = batch["s2"].cuda(), batch["s1"].cuda(), batch["target"].cuda()
        if not (target >= 0).any():
            skipped_batches += 1
            continue
        draw = random.random()
        s2_input = None if draw < CFG.modality_drop_rate else s2
        s1_input = (
            None if CFG.modality_drop_rate <= draw < 2 * CFG.modality_drop_rate else s1
        )
        with torch.autocast("cuda", enabled=False):
            loss = segmentation_loss(model(s2=s2_input, s1=s1_input), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, chips {batch['id']}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        valid_batches += 1
    if not valid_batches:
        raise RuntimeError("No labelled training batches; refusing an empty epoch.")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Non-finite model state; refuse checkpoint export.")
    validation = evaluate(validation_loader, "fused")
    row = {
        "epoch": epoch + 1,
        "train_loss": running / valid_batches,
        "skipped_unlabelled_batches": skipped_batches,
        **validation,
    }
    history.append(row)
    (ARTIFACTS / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False))
    print(row)
    if validation["flood_iou"] > best_iou:
        best_iou = validation["flood_iou"]
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            ARTIFACTS / "model.safetensors",
        )
    torch.save(
        {"epoch": epoch + 1, "best_iou": best_iou, "history": history,
         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "scaler": scaler.state_dict(), "config": asdict(CFG)}, training_state
    )

# %% [markdown]
# ## 5. Reload best weights, run ablations, and export raw predictions

# %%
model.load_state_dict(load_file(ARTIFACTS / "model.safetensors"), strict=True)
validation_predictions = ARTIFACTS / "validation_predictions.jsonl"
validation_metrics = {
    "fused": evaluate(validation_loader, "fused", validation_predictions),
    "s2": evaluate(validation_loader, "s2"),
    "s1": evaluate(validation_loader, "s1"),
}
test_metrics = None
validation_ready = (CFG.release_minimum_flood_iou is not None
                    and validation_metrics["fused"]["flood_iou"] >= CFG.release_minimum_flood_iou
                    and validation_metrics["fused"]["flood_iou"] >= max(validation_metrics[m]["flood_iou"] for m in ("s2", "s1")))
if CFG.run_test_once and validation_ready:
    test_metrics = {
        "fused": evaluate(test_loader, "fused", ARTIFACTS / "test_predictions.jsonl"),
        "s2": evaluate(test_loader, "s2"),
        "s1": evaluate(test_loader, "s1"),
    }
config = {
    "artifact_version": "satquery-pair-v3",
    "architecture": "terramind_s1_s2_pixel_flood_segmentation",
    "mask_supervision": "Sen1Floods11 v1.1 LabelHand pixels; -1 ignored",
    "calibrated": False,
    "embedding_dim": embedding_dim,
    "backbone": backbone_name,
    "modalities": ["S2L1C", "S1GRD"],
    "classes": ["other", "flood_water"],
    "s2_bands": S2_BANDS,
    "s1_bands": S1_BANDS,
    "normalization": {
        "s2_mean": TM_S2_MEAN.tolist(),
        "s2_std": TM_S2_STD.tolist(),
        "s1_mean": TM_S1_MEAN.tolist(),
        "s1_std": TM_S1_STD.tolist(),
    },
    "config": asdict(CFG),
    "validation": validation_metrics,
    "test": test_metrics,
    "history": history,
    "score_semantics": "Pixel softmax is uncalibrated; IoU/Dice use hand labels.",
    "domain_gap": "Sentinel training does not establish Cartosat/RISAT or India-domain accuracy.",
}
(ARTIFACTS / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
(ARTIFACTS / "evaluation_summary.json").write_text(
    json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2),
    encoding="utf-8",
)
print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))

# Fresh architecture reload, including the revised decoder, before release/export.
reload_backbone = BACKBONE_REGISTRY.build(backbone_name, pretrained=False,
    modalities=["S2L1C", "S1GRD"], merge_method="mean")
reloaded = FusionExpert(reload_backbone, embedding_dim, 2, CFG.image_size,
                        decoder_version=CFG.decoder_version).cuda().eval()
reloaded.load_state_dict(load_file(ARTIFACTS / "model.safetensors"), strict=True)
reload_batch = next(iter(validation_loader))
with torch.inference_mode():
    reload_logits = reloaded(s2=reload_batch["s2"].cuda(), s1=reload_batch["s1"].cuda())
assert torch.isfinite(reload_logits).all()
assert reload_logits.shape == (len(reload_batch["id"]), 2, CFG.image_size, CFG.image_size)
del reloaded, reload_backbone, reload_batch, reload_logits
torch.cuda.empty_cache()
print("PASS: fresh fusion architecture reloaded; finite outputs. Accuracy gates remain separate.")

# %% [markdown]
# ## 6. Hash, guarded free Hub upload, and safe stop


# %%
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = {
    path.name: sha256(path)
    for path in sorted(ARTIFACTS.iterdir())
    if path.is_file() and path.name not in {"sha256_manifest.json", "training_state.pt"}
}
(ARTIFACTS / "sha256_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
threshold_declared = CFG.release_minimum_flood_iou is not None
ablation_passed = validation_metrics["fused"]["flood_iou"] >= max(
    validation_metrics["s2"]["flood_iou"],
    validation_metrics["s1"]["flood_iou"],
)
validation_passed = (
    threshold_declared
    and (validation_metrics["fused"]["flood_iou"] >= CFG.release_minimum_flood_iou)
    and ablation_passed
)
test_passed = (
    threshold_declared
    and test_metrics is not None
    and (test_metrics["fused"]["flood_iou"] >= CFG.release_minimum_flood_iou)
)
gate = {
    "pixel_supervision": True,
    "official_disjoint_splits": True,
    "raw_validation_predictions": validation_predictions.is_file(),
    "weights": (ARTIFACTS / "model.safetensors").is_file(),
    "hash_manifest": (ARTIFACTS / "sha256_manifest.json").is_file(),
    "release_threshold_declared": threshold_declared,
    "fused_not_worse_than_single_modality": ablation_passed,
    "validation_gate_passed": validation_passed,
    "test_run_completed": test_metrics is not None,
    "test_gate_passed": test_passed,
    "paid_endpoint_created": False,
}
if CFG.push_to_hub:
    if not validation_passed or not test_passed:
        raise RuntimeError(
            "Hub upload refused: declare and pass validation/test IoU gates first."
        )
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
    if not token:
        raise RuntimeError(
            "Add HF_TOKEN in notebook Secrets; never paste it into source."
        )
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(CFG.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=CFG.repo_id,
        repo_type="model",
        folder_path=ARTIFACTS,
        commit_message="Release held-out TerraMind Sen1Floods11 segmentation",
        ignore_patterns=["training_state.pt"],
    )
print(json.dumps(gate, indent=2))
print(
    "SAFE STOP — download artifacts, save the notebook version, then turn off the GPU."
)
