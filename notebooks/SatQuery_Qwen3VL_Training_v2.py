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

# %% [markdown]
# # SatQuery AI — Qwen3-VL cloud training and audited Hub release (v2)
#
# **Target:** Kaggle or Google Colab with an NVIDIA GPU. No local training is required.
#
# This notebook follows the supplied SIH26167 guide as a project specification and intentionally
# trains only the first specialist: **single-image VQA + captioning + grounding** using Qwen3-VL
# with 4-bit QLoRA on BigEarthNet.txt. Change-VQA and optical+SAR fusion belong in separate notebooks
# and separate GPU sessions; `Run All` here cannot accidentally start those workloads.
#
# **Important truths**
#
# - The supplied `BigEarthNet.txt` link contains the 467 MB text/QA/grounding table. It does **not**
#   contain the 110 GiB Sentinel image archive. This notebook joins that table to a 2.48 GB
#   cloud-friendly S1/S2 LMDB subset by patch ID. A full-data path is also documented.
# - Uploading weights to the Hugging Face Hub stores and versions them; it does not automatically
#   create a running API. Dedicated Inference Endpoints remain disabled by default because they are
#   billed infrastructure. Use a separate ZeroGPU Gradio Space for the no-cost demo deployment.
# - Generated-token likelihood is not automatically a calibrated confidence probability. This
#   notebook labels it as a proxy, measures calibration error, and keeps the distinction in the API.

# %% [markdown]
# ## Run map
#
# Use **Run All** for sections **0 → 8**:
#
# - `0–2`: install, configure, authenticate, and verify the GPU.
# - `3`: download the supplied text dataset and matching S1/S2 imagery.
# - `4–6`: audit data, QLoRA fine-tune Qwen3-VL, and evaluate it.
# - `7`: save immutable artifacts and upload them to a private Hub repository.
# - `8`: fresh reload, create the bounded handler, and safely skip paid hosting unless explicitly
#   enabled. Free ZeroGPU deployment is intentionally handled in a separate Space package.

# %% [markdown]
# ## 0. Install a reproducible cloud environment
#
# In Kaggle, enable **Internet** and choose a **GPU** accelerator. In Colab choose
# `Runtime → Change runtime type → T4 GPU` (or L4/A100 if available). Run this cell once, then restart
# the runtime only if the notebook asks you to.

# %%
import subprocess
import sys

CORE_PACKAGES = [
    "transformers==4.57.1",
    "accelerate==1.7.0",
    "peft==0.17.1",
    "bitsandbytes==0.47.0",
    "huggingface_hub>=0.34.0,<1.0",
    "qwen-vl-utils==0.0.14",
    "duckdb==1.3.2",
    "lmdb>=1.5,<2",
    "safetensors>=0.5,<1",
]
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade-strategy", "only-if-needed", *CORE_PACKAGES,
    ]
)
print(
    "Core environment installed. Kaggle may report conflicts in optional preinstalled packages "
    "that this notebook does not use. Restart the runtime once before continuing."
)

# %% [markdown]
# ## 1. Configuration — edit only this cell first
#
# The default profile is deliberately small enough for a free T4 while still exercising all four
# BigEarthNet.txt task types. Increase caps only after one end-to-end run succeeds. Do not set a
# global `70%` target: the publisher corpus contains millions of image-text triplets, while this
# notebook has only the matching Lithuania LMDB image subset. Coverage must be calculated over
# uniquely matched, scene-isolated images—not over text-row duplicates. Grow caps in staged runs,
# compare base/LoRA on the untouched split, and stop when the held-out metrics plateau.

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    stage: str = "vlm"  # fixed: this v2 notebook is intentionally VLM-only
    data_profile: str = "lithuania-summer"  # cloud-safe 2.48 GB S1/S2 subset
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    requested_model_revision: str | None = None  # resolved to an immutable SHA below
    max_train_per_type: int = 1500  # up to 6,000 rows across 4 types
    max_validation_per_type: int = 200
    max_test_per_type: int = 200
    epochs: float = 1.0
    learning_rate: float = 1.0e-5
    gradient_accumulation_steps: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 1536
    min_pixels: int = 16 * 28 * 28
    max_pixels: int = 256 * 28 * 28  # reduce to 128*28*28 if T4 memory is tight
    eval_samples: int = 120
    train: bool = True
    evaluate: bool = True
    push_to_hub: bool = True
    make_repo_private: bool = False  # keep the published SatQuery adapter publicly reviewable
    fresh_reload_smoke_test: bool = True
    provision_paid_endpoint: bool = False  # NEVER silently start paid infrastructure
    overwrite_artifacts: bool = False  # set True only when intentionally replacing this run's local export


CFG = RunConfig()

IS_KAGGLE = Path("/kaggle").exists()
IS_COLAB = (not IS_KAGGLE) and ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ)
ROOT = Path("/kaggle/working/satquery") if IS_KAGGLE else Path("/content/satquery") if IS_COLAB else Path.cwd() / "satquery-cloud"
DATA_DIR = ROOT / "data"
RUN_DIR = ROOT / "runs" / "qwen3vl_bigearthnet_txt"
ARTIFACT_DIR = ROOT / "artifacts" / "satquery-qwen3vl-adapter"
for directory in (DATA_DIR, RUN_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

assert CFG.stage == "vlm", "Use the separate Change-VQA or fusion notebook for another specialist."
assert CFG.base_model in {
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
}, "Use the 2B model on T4; use 4B only on L4/A100 or after a successful 2B run."

print(asdict(CFG))
print({"root": str(ROOT), "kaggle": IS_KAGGLE, "colab": IS_COLAB})

# %% [markdown]
# ## 2. GPU, seeds, versions and Hugging Face secret
#
# Create a Hugging Face token with **write access only to your SatQuery model repository** if
# possible. Store it as a Kaggle secret named `HF_TOKEN` or a Colab secret named `HF_TOKEN`.
# Never paste a token into a notebook cell, print it, commit it, or put it in a frontend bundle.

# %%
import json
import platform
import random
import secrets
import numpy as np
import torch
import transformers
import peft
import huggingface_hub


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected. Enable a GPU accelerator before training.")

gpu = torch.cuda.get_device_properties(0)
runtime = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "gpu": gpu.name,
    "vram_gib": round(gpu.total_memory / 1024**3, 2),
    "cuda_capability": torch.cuda.get_device_capability(0),
}
print(json.dumps(runtime, indent=2))
if runtime["vram_gib"] < 14:
    raise RuntimeError("At least ~14 GiB VRAM is required for the default QLoRA configuration.")
if "4B" in CFG.base_model and runtime["vram_gib"] < 22:
    raise RuntimeError("The 4B configuration needs an L4/A100-class runtime; select the 2B model on T4.")

# %%
from getpass import getpass
from huggingface_hub import HfApi, login


def read_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN").strip()
        except Exception:
            pass
    if not token and IS_COLAB:
        try:
            from google.colab import userdata

            token = (userdata.get("HF_TOKEN") or "").strip()
        except Exception:
            pass
    if not token:
        token = getpass("HF_TOKEN (input is hidden): ").strip()
    if not token.startswith("hf_"):
        raise ValueError("A valid Hugging Face user access token is required.")
    return token


HF_TOKEN = read_hf_token()
login(token=HF_TOKEN, add_to_git_credential=False)
HF_API = HfApi(token=HF_TOKEN)
HF_ACCOUNT = HF_API.whoami()
HF_NAMESPACE = HF_ACCOUNT.get("name") or HF_ACCOUNT.get("fullname")
if not HF_NAMESPACE:
    raise RuntimeError("Could not resolve the Hugging Face namespace for this token.")
ADAPTER_REPO_ID = f"{HF_NAMESPACE}/satquery-qwen3vl-bigearthnet-txt-lora"
print({"authenticated_as": HF_NAMESPACE, "target_repo": ADAPTER_REPO_ID})

# %% [markdown]
# ## 3. Import BigEarthNet.txt and matching Sentinel imagery
#
# Dataset roles:
#
# - `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`: 9.55M text records, including binary QA, MCQ,
#   captioning, and bounding-box grounding. The parquet file is about 467 MB.
# - `hackelle/BigEarthNetV2-Lithuania-Summer-LMDB`: an unofficial convenience mirror of a matching
#   BigEarthNet v2 subset. It is about 2.48 GB and contains all 12 S2 bands and both S1 bands.
# - The authoritative full S2 and S1 archives are approximately 59 GiB and 51 GiB respectively.
#   Free Kaggle disks cannot safely hold both plus checkpoints. For a full run, preprocess the
#   official archives to LMDB once in persistent cloud storage and set `LMDB_DIR`/`BEN_METADATA`.
#
# The source dataset is CDLA-Permissive-1.0. Keep citations and the model-card limitation that the
# imagery is European Sentinel data and does not remove the India/Cartosat/RISAT domain gap.

# %%
from huggingface_hub import hf_hub_download, snapshot_download


TEXT_REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
SUBSET_REPO = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
TEXT_REVISION = HF_API.dataset_info(TEXT_REPO).sha
SUBSET_REVISION = HF_API.dataset_info(SUBSET_REPO).sha
if not TEXT_REVISION or not SUBSET_REVISION:
    raise RuntimeError("Could not resolve dataset repositories to immutable revisions")

TEXT_PARQUET = Path(
    hf_hub_download(
        repo_id=TEXT_REPO,
        repo_type="dataset",
        filename="BigEarthNet.txt.parquet",
        revision=TEXT_REVISION,
        local_dir=DATA_DIR / "bigearthnet_txt",
    )
)

if CFG.data_profile == "lithuania-summer":
    subset_root = Path(
        snapshot_download(
            repo_id=SUBSET_REPO,
            repo_type="dataset",
            revision=SUBSET_REVISION,
            local_dir=DATA_DIR / "bigearthnet_lithuania_summer",
            allow_patterns=[
                "BENv2_lithuania_summer.lmdb/*",
                "metadata_lithuania_summer.parquet",
                "README.md",
            ],
        )
    )
    LMDB_DIR = subset_root / "BENv2_lithuania_summer.lmdb"
    BEN_METADATA = subset_root / "metadata_lithuania_summer.parquet"
else:
    LMDB_DIR = Path(os.environ["LMDB_DIR"])
    BEN_METADATA = Path(os.environ["BEN_METADATA"])

required_files = [TEXT_PARQUET, BEN_METADATA, LMDB_DIR / "data.mdb", LMDB_DIR / "lock.mdb"]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError(f"Dataset import is incomplete. Missing: {missing}")
print({
    "text_parquet": str(TEXT_PARQUET), "text_revision": TEXT_REVISION,
    "lmdb": str(LMDB_DIR), "image_revision": SUBSET_REVISION,
    "metadata": str(BEN_METADATA),
})

# %% [markdown]
# ### Build a bounded, stratified manifest without loading 9.55M rows into RAM
#
# DuckDB scans parquet on disk, joins by `patch_id`, preserves the publisher's split, and takes a
# deterministic cap per `(split, type)`. No row from test or bench is used for training.

# %%
import duckdb
import pandas as pd


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


PREPARED_PARQUET = DATA_DIR / "prepared_bigearthnet_txt.parquet"
query = f"""
COPY (
    WITH matched AS (
        SELECT
            t.ID, t.s1_name, t.patch_id, t.input, t.output, t.type, t.category,
            t.split, t.latitude, t.longitude, t.country, t.season, t.climate_zone
        FROM read_parquet('{sql_path(TEXT_PARQUET)}') AS t
        SEMI JOIN read_parquet('{sql_path(BEN_METADATA)}') AS m USING (patch_id)
        WHERE t.split IN ('train', 'validation', 'test', 'bench')
          AND t.type IN ('binary', 'mcq', 'captioning', 'bounding box')
          AND lower(t.category) NOT IN ('country', 'season', 'climate zone', 'climate_zone', 'cliamte zone')
    ), ranked AS (
        SELECT *, row_number() OVER (
            PARTITION BY split, type
            ORDER BY hash(CAST(ID AS VARCHAR) || '{CFG.seed}')
        ) AS deterministic_rank
        FROM matched
    )
    SELECT * EXCLUDE (deterministic_rank)
    FROM ranked
    WHERE deterministic_rank <= CASE
        WHEN split = 'train' THEN {CFG.max_train_per_type}
        WHEN split = 'validation' THEN {CFG.max_validation_per_type}
        ELSE {CFG.max_test_per_type}
    END
) TO '{sql_path(PREPARED_PARQUET)}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""
duckdb.sql(query)
manifest = pd.read_parquet(PREPARED_PARQUET)
print(manifest.groupby(["split", "type"]).size().unstack(fill_value=0))
print(f"Prepared rows: {len(manifest):,}; unique image patches: {manifest.patch_id.nunique():,}")

# %% [markdown]
# ### Data integrity and leakage gates
#
# These are hard failures. Do not comment them out to make a run continue.

# %%
EXPECTED_COLUMNS = {
    "ID", "s1_name", "patch_id", "input", "output", "type", "category", "split",
    "latitude", "longitude", "country", "season", "climate_zone",
}
assert EXPECTED_COLUMNS.issubset(manifest.columns)
assert not manifest[["patch_id", "input", "output", "split", "type"]].isna().any().any()
assert manifest.ID.is_unique, "Duplicate text record IDs found"

patches_by_split = {
    split: set(frame.patch_id.astype(str))
    for split, frame in manifest.groupby("split")
}
for left_name, left in patches_by_split.items():
    for right_name, right in patches_by_split.items():
        if left_name < right_name:
            overlap = left & right
            assert not overlap, f"Image leakage between {left_name} and {right_name}: {len(overlap)} patches"

for split in ("train", "validation"):
    if split not in patches_by_split:
        raise RuntimeError(f"Required split {split!r} is absent after joining text and imagery")

print("PASS: schema, null, duplicate-ID, and patch-level split leakage checks")

# %% [markdown]
# ## 4. Image reader, coordinate conversion and training examples
#
# Qwen grounding uses integer coordinates normalized to `0..1000`. BigEarthNet.txt boxes and point
# prompts are normalized to `0..1`; this conversion is explicit and tested below. No Python `eval`
# is used on dataset content.

# %%
import io
import re
import lmdb
from PIL import Image
from safetensors.numpy import load as load_safetensors_bytes
from torch.nn import functional as F


S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["VV", "VH"]


class BigEarthNetLMDB:
    def __init__(self, lmdb_dir: Path, metadata_path: Path):
        self.lmdb_dir = Path(lmdb_dir)
        metadata = pd.read_parquet(metadata_path, columns=["patch_id", "s1_name", "labels", "split"])
        self.s1_for_s2 = dict(zip(metadata.patch_id.astype(str), metadata.s1_name.astype(str)))
        self.labels_for_s2 = dict(zip(metadata.patch_id.astype(str), metadata.labels))
        self.split_for_s2 = dict(zip(metadata.patch_id.astype(str), metadata.split.astype(str)))
        self._env = None

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_dir), readonly=True, lock=False, readahead=True,
                meminit=False, max_readers=64, subdir=True,
            )
        return self._env

    def _read_key(self, key: str) -> dict[str, np.ndarray]:
        with self.env.begin(write=False, buffers=True) as txn:
            raw = txn.get(key.encode("utf-8"))
        if raw is None:
            raise KeyError(f"LMDB has no entry for {key}")
        return load_safetensors_bytes(bytes(raw))

    def read_s2(self, patch_id: str) -> dict[str, np.ndarray]:
        data = self._read_key(str(patch_id))
        missing = set(S2_BANDS) - set(data)
        if missing:
            raise ValueError(f"S2 patch {patch_id} is missing bands {sorted(missing)}")
        return data

    def read_pair(self, patch_id: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        patch_id = str(patch_id)
        s2 = self.read_s2(patch_id)
        s1_name = self.s1_for_s2.get(patch_id)
        if not s1_name:
            raise KeyError(f"No S1 mapping for {patch_id}")
        s1 = self._read_key(s1_name)
        missing = set(S1_BANDS) - set(s1)
        if missing:
            raise ValueError(f"S1 patch {s1_name} is missing bands {sorted(missing)}")
        return s2, s1

    def rgb(self, patch_id: str, size: int = 448) -> Image.Image:
        s2 = self.read_s2(patch_id)
        channels = []
        for band in ("B04", "B03", "B02"):
            array = np.asarray(s2[band], dtype=np.float32)
            low, high = np.nanpercentile(array, (2, 98))
            scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
            channels.append((scaled * 255).astype(np.uint8))
        rgb = np.stack(channels, axis=-1)
        return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


BEN = BigEarthNetLMDB(LMDB_DIR, BEN_METADATA)

NUMBER = r"[-+]?\d*\.?\d+"


def to_qwen_coordinate(value: float) -> int:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Expected a normalized coordinate in [0,1], got {value}")
    return int(round(value * 1000))


def convert_points_in_prompt(text: str) -> str:
    pattern = re.compile(rf"<point>\s*\(({NUMBER})\s*,\s*({NUMBER})\)\s*</point>")

    def replace(match: re.Match) -> str:
        x, y = map(float, match.groups())
        return f"<point>({to_qwen_coordinate(x)},{to_qwen_coordinate(y)})</point>"

    return pattern.sub(replace, text)


def convert_box_answer(value: object) -> str:
    numbers = [float(item) for item in re.findall(NUMBER, str(value))]
    if len(numbers) != 4:
        raise ValueError(f"Could not parse a four-coordinate box from {value!r}")
    x1, y1, x2, y2 = numbers
    if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
        raise ValueError(f"Invalid normalized box ordering: {numbers}")
    box = [to_qwen_coordinate(item) for item in numbers]
    return f"<box>({box[0]},{box[1]}),({box[2]},{box[3]})</box>"


assert convert_points_in_prompt("at <point>(0.82, 0.28)</point>") == "at <point>(820,280)</point>"
assert convert_box_answer("[0.64 0.0, 1.0 0.71]") == "<box>(640,0),(1000,710)</box>"


def row_to_conversation(row: pd.Series) -> dict:
    question = convert_points_in_prompt(str(row["input"]).strip())
    answer = convert_box_answer(row["output"]) if row["type"] == "bounding box" else str(row["output"]).strip()
    return {
        "image": BEN.rgb(str(row["patch_id"])),
        "question": question,
        "answer": answer,
        "type": str(row["type"]),
        "patch_id": str(row["patch_id"]),
        "record_id": int(row["ID"]),
    }


sample_record = row_to_conversation(manifest.iloc[0])
display(sample_record["image"])
print({key: value for key, value in sample_record.items() if key != "image"})

# %% [markdown]
# ## 5. Load Qwen3-VL and apply language-only QLoRA
#
# Why Qwen3-VL here:
#
# - Apache-2.0 base weights.
# - Native image-text generation and grounding coordinate output.
# - The 2B model is practical with 4-bit NF4 QLoRA on a 16 GiB T4.
#
# The LoRA targets exclude modules whose names contain `visual`, so the first run adapts the language
# reasoning layers while keeping the vision tower frozen. This is the stable free-tier starting point.
# A later ablation may unfreeze the vision merger on an L4/A100; do not mix that experiment into the
# baseline checkpoint.

# %%
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


model_info = HF_API.model_info(CFG.base_model, revision=CFG.requested_model_revision)
if not model_info.sha:
    raise RuntimeError("Could not resolve the base model to an immutable revision")
BASE_REVISION = model_info.sha
print({"base_model": CFG.base_model, "immutable_revision": BASE_REVISION})

capability = torch.cuda.get_device_capability(0)
USE_BF16 = capability[0] >= 8
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(
    CFG.base_model,
    revision=BASE_REVISION,
    trust_remote_code=False,
    min_pixels=CFG.min_pixels,
    max_pixels=CFG.max_pixels,
)
processor.tokenizer.padding_side = "right"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    CFG.base_model,
    revision=BASE_REVISION,
    trust_remote_code=False,
    quantization_config=quantization,
    dtype=COMPUTE_DTYPE,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

target_modules = [
    name
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
    and "visual" not in name.lower()
    and not name.endswith("lm_head")
]
if not target_modules:
    raise RuntimeError("No non-visual linear modules were found for LoRA")

lora_config = LoraConfig(
    r=CFG.lora_rank,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %%
class BENTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        return row_to_conversation(self.frame.iloc[index])


class QwenSatelliteCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids, attention_masks, labels = [], [], []
        pixel_values, image_grids = [], []
        for example in examples:
            user_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]},
                ],
            }]
            full_messages = user_messages + [{
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
            }]
            prompt_text = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            images, videos = process_vision_info(full_messages)
            full = self.processor(
                text=[full_text], images=images, videos=videos,
                return_tensors="pt", truncation=True, max_length=self.max_length,
            )
            prompt = self.processor(
                text=[prompt_text], images=images, videos=videos,
                return_tensors="pt", truncation=True, max_length=self.max_length,
            )
            ids = full.input_ids[0]
            item_labels = ids.clone()
            prompt_length = min(prompt.input_ids.shape[1], item_labels.shape[0])
            item_labels[:prompt_length] = -100
            input_ids.append(ids)
            attention_masks.append(full.attention_mask[0])
            labels.append(item_labels)
            pixel_values.append(full.pixel_values)
            image_grids.append(full.image_grid_thw)

        pad_id = self.processor.tokenizer.pad_token_id
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grids, dim=0),
        }


train_frame = manifest[manifest.split == "train"].copy()
validation_frame = manifest[manifest.split == "validation"].copy()
train_dataset = BENTextDataset(train_frame)
validation_dataset = BENTextDataset(validation_frame)
collator = QwenSatelliteCollator(processor, CFG.max_length)

one_batch = collator([train_dataset[0]])
assert one_batch["input_ids"].shape == one_batch["labels"].shape
assert (one_batch["labels"] == -100).any(), "User prompt tokens must be masked from the loss"
assert (one_batch["labels"] != -100).any(), "Assistant answer tokens must remain trainable"
print({key: tuple(value.shape) for key, value in one_batch.items()})

# Fail before a long run if gradient checkpointing or quantization has disconnected LoRA.
probe_batch = {key: value.to(model.device) for key, value in one_batch.items()}
model.train()
with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
    probe_loss = model(**probe_batch).loss
probe_loss.backward()
probe_gradient_norms = [
    float(parameter.grad.float().norm())
    for name, parameter in model.named_parameters()
    if "lora_" in name and parameter.grad is not None
]
assert probe_gradient_norms and max(probe_gradient_norms) > 0, (
    "LoRA gradient preflight failed; do not start training because no adapter gradient is flowing."
)
print({"lora_gradient_preflight": "PASS", "probe_loss": float(probe_loss.detach())})
model.zero_grad(set_to_none=True)
model.eval()
del probe_batch, probe_loss, one_batch
torch.cuda.empty_cache()

# %% [markdown]
# ### Base-model smoke test before fine-tuning
#
# Keep this output as evidence that any improvement came from domain adaptation rather than prompt
# engineering alone.

# %%
@torch.inference_mode()
def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    # inference_mode() disables autograd but does not change Module.training. Generation must run
    # in eval mode because gradient checkpointing is enabled for the later Trainer run.
    was_training = model.training
    model.eval()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=images, videos=videos, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    finally:
        if was_training:
            model.train()


base_example = validation_dataset[0]
print("Question:", base_example["question"])
print("Reference:", base_example["answer"])
print("Base prediction:", generate_answer(base_example["image"], base_example["question"]))

# %% [markdown]
# ### Train with resumable checkpoints
#
# Typical free-T4 starting settings are batch size 1, gradient accumulation 16, one epoch, and
# 6,000 balanced training rows. If CUDA OOM occurs, reduce `max_pixels` first, restart the runtime,
# and rerun from section 1. Do not continue after an OOM with a fragmented CUDA allocator.

# %%
training_args = TrainingArguments(
    output_dir=str(RUN_DIR),
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    tf32=USE_BF16,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
    optim="paged_adamw_8bit",
    seed=CFG.seed,
    data_seed=CFG.seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
    processing_class=processor,
)

if CFG.train:
    checkpoints = sorted(RUN_DIR.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
    resume = str(checkpoints[-1]) if checkpoints else None
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trained_lora_b_norms = [
        float(parameter.detach().float().norm())
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]
    assert trained_lora_b_norms and max(trained_lora_b_norms) > 0, (
        "LoRA B matrices are still zero after training; refuse to publish an untrained adapter."
    )
    print({
        "trained_adapter_check": "PASS",
        "max_lora_B_norm": max(trained_lora_b_norms),
    })
else:
    print("Training skipped by configuration.")

# %% [markdown]
# ## 6. Evaluation: task-specific metrics and explicit confidence semantics
#
# - Binary/MCQ VQA: normalized exact match.
# - Grounding: mean IoU and accuracy at IoU ≥ 0.5.
# - Captioning: token-overlap F1 as a smoke metric; use CIDEr/SPICE for the formal benchmark report.
# - `token_likelihood_proxy`: geometric mean probability of generated tokens. It is measured for
#   calibration but is **not** returned as a calibrated probability.

# %%
import math
import string
from collections import Counter
from tqdm.auto import tqdm


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    table = str.maketrans("", "", string.punctuation.replace("<", "").replace(">", ""))
    return " ".join(text.translate(table).split())


def parse_qwen_box(text: str) -> tuple[int, int, int, int] | None:
    numbers = [int(round(float(value))) for value in re.findall(NUMBER, str(text))]
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = numbers[:4]
    if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
        return None
    return x1, y1, x2, y2


def box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.inference_mode()
def predict_with_proxy(image: Image.Image, question: str, max_new_tokens: int = 128) -> tuple[str, float]:
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    generated = model.generate(
        **batch, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, output_scores=True,
    )
    input_length = batch["input_ids"].shape[1]
    new_tokens = generated.sequences[0, input_length:]
    probabilities = []
    for step, logits in enumerate(generated.scores):
        if step >= len(new_tokens):
            break
        token_probability = torch.softmax(logits[0].float(), dim=-1)[new_tokens[step]].item()
        probabilities.append(max(token_probability, 1e-12))
    proxy = math.exp(sum(math.log(value) for value in probabilities) / max(len(probabilities), 1))
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, float(proxy)


def expected_calibration_error(confidences: list[float], correctness: list[int], bins: int = 10) -> float:
    confidences_np = np.asarray(confidences, dtype=np.float64)
    correctness_np = np.asarray(correctness, dtype=np.float64)
    total = max(len(confidences_np), 1)
    ece = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        selected = (confidences_np > lower) & (confidences_np <= upper)
        if selected.any():
            ece += selected.sum() / total * abs(correctness_np[selected].mean() - confidences_np[selected].mean())
    return float(ece)


if CFG.evaluate:
    model.eval()
    evaluation_frame = validation_frame.groupby("type", group_keys=False).head(
        max(1, CFG.eval_samples // max(validation_frame.type.nunique(), 1))
    )
    records = []
    for _, row in tqdm(evaluation_frame.iterrows(), total=len(evaluation_frame)):
        example = row_to_conversation(row)
        prediction, proxy = predict_with_proxy(example["image"], example["question"])
        task_type = example["type"]
        exact = int(normalize_answer(prediction) == normalize_answer(example["answer"]))
        iou = None
        caption_f1 = None
        if task_type == "bounding box":
            pred_box, ref_box = parse_qwen_box(prediction), parse_qwen_box(example["answer"])
            iou = box_iou(pred_box, ref_box) if pred_box and ref_box else 0.0
            exact = int(iou >= 0.5)
        elif task_type == "captioning":
            caption_f1 = token_f1(prediction, example["answer"])
        records.append({
            "record_id": example["record_id"], "patch_id": example["patch_id"],
            "type": task_type, "question": example["question"], "reference": example["answer"],
            "prediction": prediction, "correct": exact, "iou": iou,
            "caption_token_f1": caption_f1, "token_likelihood_proxy": proxy,
            "score_kind": "uncalibrated_token_likelihood_proxy",
        })
    eval_df = pd.DataFrame(records)
    eval_df.to_json(RUN_DIR / "validation_predictions.jsonl", orient="records", lines=True)
    summary = {
        "rows": len(eval_df),
        "vqa_exact_match": float(eval_df[eval_df.type.isin(["binary", "mcq"])].correct.mean()),
        "grounding_mean_iou": float(eval_df[eval_df.type == "bounding box"].iou.mean()),
        "grounding_accuracy_iou_0_5": float(eval_df[eval_df.type == "bounding box"].correct.mean()),
        "caption_token_f1": float(eval_df[eval_df.type == "captioning"].caption_token_f1.mean()),
        "proxy_ece_on_non_caption_rows": expected_calibration_error(
            eval_df[eval_df.type != "captioning"].token_likelihood_proxy.tolist(),
            eval_df[eval_df.type != "captioning"].correct.tolist(),
        ),
    }
    (RUN_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
else:
    summary = {"evaluation_skipped": True}

# %% [markdown]
# ## 7. Save auditable artifacts and upload to the Hugging Face Hub
#
# The Hub repository contains adapter-only SafeTensors, processor files, immutable base revision,
# metrics, dependency versions, SHA-256 hashes, a model card, and a custom endpoint handler.

# %%
import hashlib
import shutil
from datetime import datetime, timezone


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


existing_artifacts = list(ARTIFACT_DIR.iterdir()) if ARTIFACT_DIR.exists() else []
if existing_artifacts and not CFG.overwrite_artifacts:
    raise FileExistsError(
        f"Artifact directory is not empty: {ARTIFACT_DIR}. "
        "Download it or change the run path; set overwrite_artifacts=True only for an intentional replacement."
    )
if existing_artifacts and CFG.overwrite_artifacts:
    for old_file in existing_artifacts:
        if old_file.is_file() or old_file.is_symlink():
            old_file.unlink()
        elif old_file.is_dir():
            shutil.rmtree(old_file)

model.save_pretrained(ARTIFACT_DIR, safe_serialization=True)
processor.save_pretrained(ARTIFACT_DIR)
if (RUN_DIR / "evaluation_summary.json").exists():
    shutil.copy2(RUN_DIR / "evaluation_summary.json", ARTIFACT_DIR / "evaluation_summary.json")

training_manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "configuration": asdict(CFG),
    "runtime": runtime,
    "base_model": CFG.base_model,
    "base_revision": BASE_REVISION,
    "text_dataset": TEXT_REPO,
    "text_dataset_revision": TEXT_REVISION,
    "image_dataset": SUBSET_REPO if CFG.data_profile == "lithuania-summer" else "official-user-provided",
    "image_dataset_revision": SUBSET_REVISION if CFG.data_profile == "lithuania-summer" else None,
    "train_rows": len(train_frame),
    "validation_rows": len(validation_frame),
    "task_counts_train": train_frame.type.value_counts().to_dict(),
    "score_semantics": "Generated-token likelihood is an uncalibrated proxy, not a probability of correctness.",
}
(ARTIFACT_DIR / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")
(ARTIFACT_DIR / "base_revision.txt").write_text(BASE_REVISION + "\n", encoding="utf-8")

model_card = f"""---
license: apache-2.0
base_model: {CFG.base_model}
library_name: peft
pipeline_tag: image-text-to-text
tags:
- remote-sensing
- visual-question-answering
- image-captioning
- visual-grounding
- qwen3-vl
- lora
datasets:
- {TEXT_REPO}
---

# SatQuery Qwen3-VL BigEarthNet.txt adapter

QLoRA adapter for SIH26167 SatQuery AI single-image remote-sensing VQA, captioning and grounding.

## Base and data

- Base: `{CFG.base_model}` at immutable revision `{BASE_REVISION}`.
- Text: `{TEXT_REPO}`.
- Imagery profile: `{CFG.data_profile}`; {len(train_frame):,} bounded training rows.
- Boxes are converted from BigEarthNet.txt normalized `0..1` coordinates to Qwen `0..1000`.

## Intended use

Use on RGB previews derived from optical imagery. Do not pass raw SAR or 12-band Sentinel tensors
to this model. SatQuery routes those inputs to its TerraMind fusion specialist.

## Limitations

The training subset is European Sentinel imagery. Performance on Indian Cartosat/RISAT imagery is
not established. Outputs require validation, evidence display, uncertainty labeling and abstention.
Generated-token likelihood is not a calibrated probability of correctness.

## Validation summary

```json
{json.dumps(summary, indent=2)}
```
"""
(ARTIFACT_DIR / "README.md").write_text(model_card, encoding="utf-8")

hash_manifest = {
    path.name: sha256_file(path)
    for path in sorted(ARTIFACT_DIR.iterdir())
    if path.is_file() and path.name != "sha256_manifest.json"
}
(ARTIFACT_DIR / "sha256_manifest.json").write_text(json.dumps(hash_manifest, indent=2), encoding="utf-8")
assert all(sha256_file(ARTIFACT_DIR / name) == expected for name, expected in hash_manifest.items())
print(json.dumps(hash_manifest, indent=2))

# %%
if CFG.push_to_hub:
    HF_API.create_repo(
        repo_id=ADAPTER_REPO_ID,
        repo_type="model",
        private=CFG.make_repo_private,
        exist_ok=True,
    )
    commit = HF_API.upload_folder(
        repo_id=ADAPTER_REPO_ID,
        repo_type="model",
        folder_path=ARTIFACT_DIR,
        commit_message="Upload audited SatQuery Qwen3-VL QLoRA adapter",
    )
    uploaded_info = HF_API.model_info(ADAPTER_REPO_ID)
    print({"repo": f"https://huggingface.co/{ADAPTER_REPO_ID}", "commit": uploaded_info.sha})
else:
    print("Hub upload skipped by configuration.")

# %% [markdown]
# ## 8. Fresh-session reload, endpoint handler, hosting and API call
#
# A release is not valid until the artifact reloads without the Trainer object. The next cell removes
# the training model, loads base + adapter from the Hub, and runs one inference. Skip only if you plan
# to restart the notebook and run this exact cell first in the fresh session.

# %%
import gc
from peft import PeftModel, PeftConfig


FRESH_RELOAD_PASSED = False
if CFG.fresh_reload_smoke_test and CFG.push_to_hub:
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    remote_peft_config = PeftConfig.from_pretrained(ADAPTER_REPO_ID, token=HF_TOKEN)
    reloaded_processor = AutoProcessor.from_pretrained(ADAPTER_REPO_ID, token=HF_TOKEN, trust_remote_code=False)
    reloaded_base = Qwen3VLForConditionalGeneration.from_pretrained(
        remote_peft_config.base_model_name_or_path,
        revision=BASE_REVISION,
        token=HF_TOKEN,
        trust_remote_code=False,
        quantization_config=quantization,
        dtype=COMPUTE_DTYPE,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(reloaded_base, ADAPTER_REPO_ID, token=HF_TOKEN)
    processor = reloaded_processor
    model.eval()
    smoke = validation_dataset[0]
    fresh_reload_prediction = generate_answer(smoke["image"], smoke["question"])
    print({
        "question": smoke["question"],
        "reference": smoke["answer"],
        "fresh_reload_prediction": fresh_reload_prediction,
    })
    FRESH_RELOAD_PASSED = bool(fresh_reload_prediction.strip())
else:
    print("Fresh reload skipped. Do not mark the model released until this test passes in a clean session.")

# %% [markdown]
# ### Add a protected custom Inference Endpoint handler
#
# The handler accepts a base64 RGB image and a question. It clamps payload and generation sizes,
# uses deterministic decoding, and returns the score semantics explicitly. Your backend must keep
# `HF_TOKEN` server-side and call the protected endpoint; the browser must never receive the token.

# %%
HANDLER_CODE = r'''
import base64
import io
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class EndpointHandler:
    def __init__(self, path: str = ""):
        config = PeftConfig.from_pretrained(path)
        revision = (Path(path) / "base_revision.txt").read_text(encoding="utf-8").strip()
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=False)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            revision=revision,
            trust_remote_code=False,
            dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base, path).eval()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs = data.get("inputs", data)
        image_base64 = inputs.get("image_base64", "")
        question = str(inputs.get("question", "")).strip()
        if not question or len(question) > 2000:
            raise ValueError("question must contain 1..2000 characters")
        if not image_base64 or len(image_base64) > 14_000_000:
            raise ValueError("image_base64 is missing or exceeds the 10 MB decoded limit")
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
            if len(image_bytes) > 10_000_000:
                raise ValueError("decoded image exceeds 10 MB")
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > 25_000_000:
                raise ValueError("decoded image dimensions exceed 25 megapixels")
            image = image.convert("RGB")
            image.load()
        except Exception as exc:
            raise ValueError("invalid base64 image") from exc
        max_new_tokens = max(1, min(int(inputs.get("max_new_tokens", 128)), 256))
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        batch = self.processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
        batch = {key: value.to(self.model.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        answer_ids = output[:, batch["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return {
            "answer": answer,
            "model": "satquery-qwen3vl-bigearthnet-txt-lora",
            "score": None,
            "score_kind": "not_calculated",
            "warning": "Validate remote-sensing outputs; no calibrated probability is returned by this handler.",
        }
'''

ENDPOINT_REQUIREMENTS = """transformers==4.57.1
peft==0.17.1
accelerate==1.7.0
qwen-vl-utils==0.0.14
pillow>=10,<12
safetensors>=0.5,<1
"""

(ARTIFACT_DIR / "handler.py").write_text(HANDLER_CODE, encoding="utf-8")
(ARTIFACT_DIR / "requirements.txt").write_text(ENDPOINT_REQUIREMENTS, encoding="utf-8")

# The first upload happened before these deployment files existed. Rebuild the manifest so it
# describes the complete current release, including handler.py and requirements.txt.
hash_manifest = {
    path.name: sha256_file(path)
    for path in sorted(ARTIFACT_DIR.iterdir())
    if path.is_file() and path.name != "sha256_manifest.json"
}
(ARTIFACT_DIR / "sha256_manifest.json").write_text(
    json.dumps(hash_manifest, indent=2), encoding="utf-8"
)
assert all(sha256_file(ARTIFACT_DIR / name) == expected for name, expected in hash_manifest.items())

if CFG.push_to_hub:
    HF_API.upload_file(
        path_or_fileobj=ARTIFACT_DIR / "handler.py",
        path_in_repo="handler.py",
        repo_id=ADAPTER_REPO_ID,
        commit_message="Add bounded custom VLM endpoint handler",
    )
    HF_API.upload_file(
        path_or_fileobj=ARTIFACT_DIR / "requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=ADAPTER_REPO_ID,
        commit_message="Pin endpoint dependencies",
    )
    HF_API.upload_file(
        path_or_fileobj=ARTIFACT_DIR / "sha256_manifest.json",
        path_in_repo="sha256_manifest.json",
        repo_id=ADAPTER_REPO_ID,
        commit_message="Refresh complete release hash manifest",
    )
    print("Endpoint handler, requirements, and complete hash manifest uploaded.")

# %% [markdown]
# ### Inspect current endpoint hardware and price before provisioning
#
# Hugging Face dedicated Inference Endpoints are billed infrastructure. Prices and quota vary by
# account, cloud, region and date. This cell lists the live choices visible to your account. The next
# cell still refuses to create anything unless `provision_paid_endpoint=True` was set in section 1.

# %%
hardware_rows = []
if CFG.provision_paid_endpoint:
    try:
        from huggingface_hub import list_inference_endpoints_hardware
    except ImportError as exc:
        raise RuntimeError(
            "This huggingface_hub version cannot list dedicated endpoint hardware. "
            "Do not change the training dependency stack just for paid hosting; inspect prices "
            "from the Hugging Face endpoint dashboard in a separate deployment environment."
        ) from exc
    for item in list_inference_endpoints_hardware(namespace=HF_NAMESPACE, token=HF_TOKEN):
        if item.accelerator == "gpu":
            hardware_rows.append({
                "vendor": item.vendor,
                "region": item.region,
                "instance_type": item.instance_type,
                "instance_size": item.instance_size,
                "status": item.status,
                "price_per_hour": item.price_per_hour,
            })
else:
    print("Skipped dedicated endpoint hardware lookup: paid provisioning is disabled.")
hardware_df = pd.DataFrame(hardware_rows)
if CFG.provision_paid_endpoint and hardware_df.empty:
    print("No GPU endpoint hardware is visible to this namespace. Use the Hugging Face endpoint dashboard or request quota.")
elif CFG.provision_paid_endpoint:
    hardware_df = hardware_df.sort_values(["price_per_hour", "instance_type"])
    display(hardware_df.head(30))

# %%
ENDPOINT_NAME = "satquery-qwen3vl-v1"
ENDPOINT_URL = os.environ.get("SATQUERY_HF_ENDPOINT_URL", "").strip()

if CFG.provision_paid_endpoint:
    from huggingface_hub import create_inference_endpoint

    if hardware_df.empty:
        raise RuntimeError("No endpoint hardware is visible to this namespace.")
    available = hardware_df[(hardware_df.status == "available") & (hardware_df.instance_type == "nvidia-l4")]
    if available.empty:
        raise RuntimeError("No nvidia-l4 endpoint hardware is currently available to this namespace.")
    choice = available.iloc[0]
    endpoint = create_inference_endpoint(
        ENDPOINT_NAME,
        namespace=HF_NAMESPACE,
        repository=ADAPTER_REPO_ID,
        framework="pytorch",
        task="custom",
        accelerator="gpu",
        vendor=choice.vendor,
        region=choice.region,
        instance_type=choice.instance_type,
        instance_size=choice.instance_size,
        type="authenticated",
        token=HF_TOKEN,
    )
    endpoint.wait(timeout=1800)
    ENDPOINT_URL = endpoint.url
    print({"endpoint_url": ENDPOINT_URL, "status": endpoint.status})
else:
    print("No paid endpoint created. Set provision_paid_endpoint=True only after reviewing the displayed live price.")

# %% [markdown]
# ### Call the protected endpoint exactly as the SatQuery backend will
#
# If the endpoint was created in a different session, set the secret/environment variable
# `SATQUERY_HF_ENDPOINT_URL` first. The API token belongs in the backend environment, never React.

# %%
import base64
import requests


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


if ENDPOINT_URL:
    api_example = validation_dataset[0]
    response = requests.post(
        ENDPOINT_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
        json={"inputs": {
            "image_base64": pil_to_base64(api_example["image"]),
            "question": api_example["question"],
            "max_new_tokens": 128,
        }},
        timeout=180,
    )
    response.raise_for_status()
    print(response.json())
else:
    print("Set SATQUERY_HF_ENDPOINT_URL or provision an endpoint before running the API call.")

# %% [markdown]
# ## 9. VLM release gate and safe stopping point
#
# This is the end of the Qwen3-VL notebook. The Change-VQA and optical+SAR specialists require
# separate notebooks and fresh GPU sessions. A successful `Run All` stops here without starting
# unrelated training or provisioning paid infrastructure.

# %%
release_checks = {
    "base_revision_is_immutable_sha": (
        len(BASE_REVISION) == 40 and all(character in "0123456789abcdef" for character in BASE_REVISION.lower())
    ),
    "adapter_safetensors_exists": (ARTIFACT_DIR / "adapter_model.safetensors").is_file(),
    "training_manifest_exists": (ARTIFACT_DIR / "training_manifest.json").is_file(),
    "evaluation_summary_exists": (not CFG.evaluate) or (ARTIFACT_DIR / "evaluation_summary.json").is_file(),
    "complete_hash_manifest": {"handler.py", "requirements.txt"}.issubset(hash_manifest),
    "hub_upload_completed": (not CFG.push_to_hub) or ("uploaded_info" in globals()),
    "fresh_hub_reload_passed": (
        (not CFG.fresh_reload_smoke_test) or (not CFG.push_to_hub) or FRESH_RELOAD_PASSED
    ),
    "paid_endpoint_disabled": not CFG.provision_paid_endpoint,
}
print(json.dumps(release_checks, indent=2))
failed_checks = [name for name, passed in release_checks.items() if not passed]
if failed_checks:
    raise RuntimeError(f"VLM release gate failed: {failed_checks}")
print("PASS: Qwen3-VL adapter trained, evaluated, hashed, uploaded, and reloaded.")
print("SAFE STOP: save the notebook version, then turn off the Kaggle/Colab GPU session.")
