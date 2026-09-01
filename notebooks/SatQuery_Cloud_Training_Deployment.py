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
# # SatQuery AI — cloud training, evaluation and Hugging Face deployment
#
# **Target:** Kaggle or Google Colab with an NVIDIA GPU. No local training is required.
#
# This notebook follows the supplied SIH26167 guide as a project specification. It implements the
# mandatory model work as three specialists rather than pretending that one LLM can consume every
# sensor correctly:
#
# 1. **Single-image VQA + captioning + grounding:** Qwen3-VL with 4-bit QLoRA on BigEarthNet.txt.
# 2. **Bi-temporal Change-VQA:** a shared-weight visual encoder, question encoder, answer head and
#    a real change-mask head trained on CDVQA + SECOND.
# 3. **Optical + SAR fusion:** TerraMind with Sentinel-2 L2A and Sentinel-1 GRD inputs, a multilabel
#    head and weakly supervised patch-evidence maps.
#
# The Qwen stage is the first runnable path and the default. The other stages are included later in
# this notebook and should be run in fresh GPU sessions because all three base models do not fit in
# free-tier memory at the same time.
#
# **Important truths**
#
# - The supplied `BigEarthNet.txt` link contains the 467 MB text/QA/grounding table. It does **not**
#   contain the 110 GiB Sentinel image archive. This notebook joins that table to a 2.48 GB
#   cloud-friendly S1/S2 LMDB subset by patch ID. A full-data path is also documented.
# - Uploading weights to the Hugging Face Hub stores and versions them; it does not automatically
#   create a running API. The deployment section creates a custom handler and can provision a
#   protected, dedicated GPU Inference Endpoint only after you deliberately enable the paid switch.
# - Generated-token likelihood is not automatically a calibrated confidence probability. This
#   notebook labels it as a proxy, measures calibration error, and keeps the distinction in the API.

# %% [markdown]
# ## Run map
#
# For the first working model, run sections **0 → 8** in order:
#
# - `0–2`: install, configure, authenticate, and verify the GPU.
# - `3`: download the supplied text dataset and matching S1/S2 imagery.
# - `4–6`: audit data, QLoRA fine-tune Qwen3-VL, and evaluate it.
# - `7`: save immutable artifacts and upload them to a private Hub repository.
# - `8`: fresh reload, create the hosted handler, optionally provision an endpoint, and call it.
#
# Sections **9** and **10** train the Change-VQA and optical+SAR specialists in fresh sessions.
# The final checklist in section **11** is the release gate.

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
# BigEarthNet.txt task types. Increase caps only after one end-to-end run succeeds.

# %%
from dataclasses import asdict, dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    stage: str = "vlm"  # vlm | change | fusion
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
    make_repo_private: bool = True
    fresh_reload_smoke_test: bool = True
    provision_paid_endpoint: bool = False  # NEVER silently start paid infrastructure
    overwrite_artifacts: bool = False  # set True only when intentionally replacing this run's local export


CFG = RunConfig()

IS_KAGGLE = Path("/kaggle").exists()
IS_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
ROOT = Path("/kaggle/working/satquery") if IS_KAGGLE else Path("/content/satquery") if IS_COLAB else Path.cwd() / "satquery-cloud"
DATA_DIR = ROOT / "data"
RUN_DIR = ROOT / "runs" / "qwen3vl_bigearthnet_txt"
ARTIFACT_DIR = ROOT / "artifacts" / "satquery-qwen3vl-adapter"
for directory in (DATA_DIR, RUN_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

assert CFG.stage in {"vlm", "change", "fusion"}
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
        return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.BILINEAR)


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
    torch_dtype=COMPUTE_DTYPE,
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
del one_batch
torch.cuda.empty_cache()

# %% [markdown]
# ### Base-model smoke test before fine-tuning
#
# Keep this output as evidence that any improvement came from domain adaptation rather than prompt
# engineering alone.

# %%
@torch.inference_mode()
def generate_answer(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
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
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


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
        torch_dtype=COMPUTE_DTYPE,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(reloaded_base, ADAPTER_REPO_ID, token=HF_TOKEN)
    processor = reloaded_processor
    model.eval()
    smoke = validation_dataset[0]
    print({
        "question": smoke["question"],
        "reference": smoke["answer"],
        "fresh_reload_prediction": generate_answer(smoke["image"], smoke["question"]),
    })
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
            torch_dtype=torch.float16,
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
    print("Endpoint handler uploaded.")

# %% [markdown]
# ### Inspect current endpoint hardware and price before provisioning
#
# Hugging Face dedicated Inference Endpoints are billed infrastructure. Prices and quota vary by
# account, cloud, region and date. This cell lists the live choices visible to your account. The next
# cell still refuses to create anything unless `provision_paid_endpoint=True` was set in section 1.

# %%
from huggingface_hub import list_inference_endpoints_hardware


hardware_rows = []
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
hardware_df = pd.DataFrame(hardware_rows)
if hardware_df.empty:
    print("No GPU endpoint hardware is visible to this namespace. Use the Hugging Face endpoint dashboard or request quota.")
else:
    hardware_df = hardware_df.sort_values(["price_per_hour", "instance_type"])
    display(hardware_df.head(30))

# %%
from huggingface_hub import create_inference_endpoint


ENDPOINT_NAME = "satquery-qwen3vl-v1"
ENDPOINT_URL = os.environ.get("SATQUERY_HF_ENDPOINT_URL", "").strip()

if CFG.provision_paid_endpoint:
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
# ## 9. Change-VQA specialist — run in a fresh GPU session
#
# CDVQA's GitHub repository provides the QA annotations, not the underlying SECOND image pairs.
# Download/attach SECOND in Kaggle or Drive with paired image folders and semantic label folders.
# Set `SECOND_ROOT` to a directory containing one of these layouts:
#
# ```text
# SECOND_ROOT/{im1,im2,label1,label2}/10589.png
# SECOND_ROOT/{A,B,label1,label2}/10589.png
# SECOND_ROOT/train/{im1,im2,label1,label2}/10589.png
# ```
#
# The model below has shared ResNet weights, fuses `A`, `B`, `|A-B|`, and `A*B`, encodes the
# question with a bidirectional GRU, predicts the closed CDVQA answer vocabulary, and trains a real
# binary mask head from differences between the two semantic label maps.

# %%
import urllib.request
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


CDVQA_DIR = DATA_DIR / "cdvqa_annotations"
CDVQA_DIR.mkdir(parents=True, exist_ok=True)
CDVQA_BASE_URL = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"
for split in ("Train", "Val", "Test", "Test2"):
    for part in ("images", "questions", "answers"):
        filename = f"{split}_{part}.json"
        destination = CDVQA_DIR / filename
        if not destination.exists():
            urllib.request.urlretrieve(f"{CDVQA_BASE_URL}/{filename}", destination)
print("CDVQA annotation files downloaded. SECOND imagery must be attached separately.")

# %%
SECOND_ROOT = Path(os.environ.get("SECOND_ROOT", "/kaggle/input/second-dataset/SECOND"))


def resolve_second_file(root: Path, split: str, role: str, filename: str) -> Path:
    aliases = {
        "time_a": ["im1", "A", "T1"],
        "time_b": ["im2", "B", "T2"],
        "label_a": ["label1", "labelA", "GT1"],
        "label_b": ["label2", "labelB", "GT2"],
    }[role]
    candidates = []
    for alias in aliases:
        candidates.extend([root / alias / filename, root / split.lower() / alias / filename])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {role} for {filename}; checked {[str(p) for p in candidates]}")


def load_cdvqa_records(split: str) -> list[dict]:
    images_json = json.loads((CDVQA_DIR / f"{split}_images.json").read_text(encoding="utf-8"))["images"]
    questions_json = json.loads((CDVQA_DIR / f"{split}_questions.json").read_text(encoding="utf-8"))["questions"]
    answers_json = json.loads((CDVQA_DIR / f"{split}_answers.json").read_text(encoding="utf-8"))["answers"]
    image_by_id = {int(item["id"]): item for item in images_json if item.get("active", True)}
    answer_by_question = {
        int(item["question_id"]): str(item["answer"]).strip().lower()
        for item in answers_json if item.get("active", True)
    }
    records = []
    for item in questions_json:
        if not item.get("active", True) or int(item["id"]) not in answer_by_question:
            continue
        image_meta = image_by_id[int(item["img_id"])]
        filename = str(image_meta["file_name"])
        records.append({
            "question_id": int(item["id"]), "image_id": int(item["img_id"]),
            "filename": filename, "question": str(item["question"]).strip(),
            "question_type": str(item["type"]), "answer": answer_by_question[int(item["id"])],
            "time_a": resolve_second_file(SECOND_ROOT, split, "time_a", filename),
            "time_b": resolve_second_file(SECOND_ROOT, split, "time_b", filename),
            "label_a": resolve_second_file(SECOND_ROOT, split, "label_a", filename),
            "label_b": resolve_second_file(SECOND_ROOT, split, "label_b", filename),
        })
    return records


if SECOND_ROOT.exists():
    cdvqa_train = load_cdvqa_records("Train")
    cdvqa_val = load_cdvqa_records("Val")
    assert not ({r["image_id"] for r in cdvqa_train} & {r["image_id"] for r in cdvqa_val})
    print({"train_qa": len(cdvqa_train), "val_qa": len(cdvqa_val)})
else:
    cdvqa_train, cdvqa_val = [], []
    print(f"Attach SECOND and set SECOND_ROOT. Current path does not exist: {SECOND_ROOT}")

# %%
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def build_vocab(texts: list[str], min_count: int = 2) -> dict[str, int]:
    counts = Counter(token for text in texts for token in TOKEN_PATTERN.findall(text.lower()))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_count:
            vocab[token] = len(vocab)
    return vocab


class CDVQADataset(Dataset):
    def __init__(self, records, word_vocab, answer_vocab, size=256, max_question_tokens=32):
        self.records = records
        self.word_vocab = word_vocab
        self.answer_vocab = answer_vocab
        self.size = size
        self.max_question_tokens = max_question_tokens
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        time_a = self.image_transform(Image.open(record["time_a"]).convert("RGB"))
        time_b = self.image_transform(Image.open(record["time_b"]).convert("RGB"))
        label_a = np.asarray(Image.open(record["label_a"]))
        label_b = np.asarray(Image.open(record["label_b"]))
        if label_a.ndim == 3:
            label_a = label_a.reshape(label_a.shape[0], label_a.shape[1], -1)
        if label_b.ndim == 3:
            label_b = label_b.reshape(label_b.shape[0], label_b.shape[1], -1)
        mask = np.any(label_a != label_b, axis=-1) if label_a.ndim == 3 else label_a != label_b
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask_tensor = F.interpolate(mask_tensor, size=(self.size, self.size), mode="nearest")[0]
        tokens = [self.word_vocab.get(token, 1) for token in TOKEN_PATTERN.findall(record["question"].lower())]
        tokens = (tokens[:self.max_question_tokens] + [0] * self.max_question_tokens)[:self.max_question_tokens]
        return {
            "time_a": time_a, "time_b": time_b, "question_tokens": torch.tensor(tokens),
            "answer": torch.tensor(self.answer_vocab[record["answer"]]), "mask": mask_tensor,
        }


class ChangeExpert(torch.nn.Module):
    def __init__(self, vocabulary_size: int, answer_classes: int, question_dim: int = 256):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4,
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 4, 512, 1, bias=False), torch.nn.BatchNorm2d(512), torch.nn.GELU(),
            torch.nn.Conv2d(512, 256, 3, padding=1, bias=False), torch.nn.BatchNorm2d(256), torch.nn.GELU(),
        )
        self.embedding = torch.nn.Embedding(vocabulary_size, question_dim, padding_idx=0)
        self.question_encoder = torch.nn.GRU(question_dim, question_dim, batch_first=True, bidirectional=True)
        self.answer_head = torch.nn.Sequential(
            torch.nn.Linear(256 + 2 * question_dim, 512), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(512, answer_classes),
        )
        self.mask_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(128, 1, 1),
        )

    def forward(self, time_a, time_b, question_tokens):
        feature_a, feature_b = self.stem(time_a), self.stem(time_b)
        fused = self.fuse(torch.cat([feature_a, feature_b, (feature_a-feature_b).abs(), feature_a*feature_b], 1))
        visual = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        _, hidden = self.question_encoder(self.embedding(question_tokens))
        question = torch.cat([hidden[-2], hidden[-1]], 1)
        answers = self.answer_head(torch.cat([visual, question], 1))
        masks = F.interpolate(self.mask_head(fused), size=time_a.shape[-2:], mode="bilinear", align_corners=False)
        return answers, masks


def change_objective(answer_logits, mask_logits, answers, masks, mask_weight=0.5):
    answer_loss = F.cross_entropy(answer_logits, answers)
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks)
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * masks).sum((1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1)).mean()
    mask_loss = 0.5 * bce + 0.5 * dice
    return answer_loss + mask_weight * mask_loss

# %%
from torch.utils.data import DataLoader
from safetensors.torch import save_file as save_safetensors


if cdvqa_train:
    word_vocab = build_vocab([record["question"] for record in cdvqa_train])
    answers = sorted({record["answer"] for record in cdvqa_train})
    answer_vocab = {answer: index for index, answer in enumerate(answers)}
    unknown_val_answers = sorted({r["answer"] for r in cdvqa_val} - set(answer_vocab))
    if unknown_val_answers:
        raise RuntimeError(f"Validation contains answers absent from training: {unknown_val_answers}")
    train_loader = DataLoader(CDVQADataset(cdvqa_train, word_vocab, answer_vocab), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(CDVQADataset(cdvqa_val, word_vocab, answer_vocab), batch_size=16, shuffle=False, num_workers=2)
    change_model = ChangeExpert(len(word_vocab), len(answer_vocab)).cuda()
    optimizer = torch.optim.AdamW(change_model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=not USE_BF16)
    best_accuracy = -1.0
    CHANGE_DIR = ROOT / "artifacts" / "satquery-change-vqa"
    CHANGE_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(10):
        change_model.train()
        for batch in tqdm(train_loader, desc=f"change epoch {epoch+1}"):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
                answer_logits, mask_logits = change_model(batch["time_a"], batch["time_b"], batch["question_tokens"])
                loss = change_objective(answer_logits, mask_logits, batch["answer"], batch["mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(change_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        change_model.eval()
        correct = total = 0
        intersection = union = predicted_pixels = target_pixels = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                logits, mask_logits = change_model(batch["time_a"], batch["time_b"], batch["question_tokens"])
                correct += (logits.argmax(1) == batch["answer"]).sum().item()
                total += len(batch["answer"])
                predicted_mask = torch.sigmoid(mask_logits) >= 0.5
                target_mask = batch["mask"] >= 0.5
                intersection += (predicted_mask & target_mask).sum().item()
                union += (predicted_mask | target_mask).sum().item()
                predicted_pixels += predicted_mask.sum().item()
                target_pixels += target_mask.sum().item()
        accuracy = correct / max(total, 1)
        mask_iou = intersection / max(union, 1.0)
        mask_dice = 2 * intersection / max(predicted_pixels + target_pixels, 1.0)
        print({"epoch": epoch + 1, "validation_accuracy": accuracy, "mask_iou": mask_iou, "mask_dice": mask_dice})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_safetensors(change_model.state_dict(), CHANGE_DIR / "model.safetensors")
            (CHANGE_DIR / "config.json").write_text(json.dumps({
                "architecture": "shared_resnet18_gru_answer_and_mask_heads",
                "word_vocab": word_vocab, "answer_vocab": answer_vocab,
                "image_size": 256, "validation_accuracy": accuracy,
                "validation_mask_iou": mask_iou, "validation_mask_dice": mask_dice,
            }, indent=2), encoding="utf-8")
else:
    print("Change training is gated until SECOND_ROOT contains paired images and both semantic label maps.")

# %% [markdown]
# ## 10. Optical + SAR TerraMind specialist — run in another fresh session
#
# Start a new T4/L4 session, rerun sections 0–4 to recreate `BEN`, then run this section. TerraMind
# receives all 12 official S2L2A bands and S1GRD `[VV,VH]`; Qwen never receives these raw tensors.
# The first epoch freezes the TerraMind backbone. Later epochs fine-tune it at a lower learning rate.
# Validation reports fused, S2-only and S1-only macro-F1 as the required modality ablation.

# %%
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
    "terratorch>=1.2.5,<2", "setuptools<81", "scikit-learn>=1.5,<1.9",
])
print("TerraTorch installed. Restart once if the import below sees a stale package cache.")

# %%
from sklearn.metrics import average_precision_score, f1_score
from terratorch.registry import BACKBONE_REGISTRY


TM_S2_MEAN = torch.tensor([1390.458,1503.317,1718.197,1853.910,2199.100,2779.975,2987.011,3083.234,3132.220,3162.988,2424.884,1857.648])
TM_S2_STD = torch.tensor([2106.761,2141.107,2038.973,2134.138,2085.321,1889.926,1820.257,1871.918,1753.829,1797.379,1434.261,1334.311])
TM_S1_MEAN = torch.tensor([-12.599, -20.293])
TM_S1_STD = torch.tensor([5.195, 5.890])


class FusionDataset(Dataset):
    def __init__(self, patch_ids: list[str], reader: BigEarthNetLMDB, classes: list[str], training=False, size=224):
        self.patch_ids, self.reader, self.classes, self.training, self.size = patch_ids, reader, classes, training, size

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]
        s2_dict, s1_dict = self.reader.read_pair(patch_id)
        s2 = torch.stack([torch.as_tensor(np.asarray(s2_dict[b]), dtype=torch.float32) for b in S2_BANDS])
        s1 = torch.stack([torch.as_tensor(np.asarray(s1_dict[b]), dtype=torch.float32) for b in S1_BANDS])
        s2 = F.interpolate(s2[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
        s1 = F.interpolate(s1[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
        s2 = (s2 - TM_S2_MEAN[:, None, None]) / TM_S2_STD[:, None, None]
        s1 = (s1 - TM_S1_MEAN[:, None, None]) / TM_S1_STD[:, None, None]
        if self.training and random.random() < 0.5:
            s2, s1 = s2.flip(-1), s1.flip(-1)
        if self.training and random.random() < 0.5:
            s2, s1 = s2.flip(-2), s1.flip(-2)
        labels = set(self.reader.labels_for_s2[patch_id])
        target = torch.tensor([float(name in labels) for name in self.classes])
        return {"s2": s2, "s1": s1, "target": target, "patch_id": patch_id}


class TerraMindFusionExpert(torch.nn.Module):
    def __init__(self, backbone, embedding_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, s2=None, s1=None):
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
        evidence = patch_logits.transpose(1, 2).reshape(tokens.shape[0], patch_logits.shape[-1], side, side)
        evidence = F.interpolate(evidence, size=(224, 224), mode="bilinear", align_corners=False)
        return class_logits, evidence

# %%
subset_meta = pd.read_parquet(BEN_METADATA)
subset_meta = subset_meta[~subset_meta.contains_seasonal_snow & ~subset_meta.contains_cloud_or_shadow]
fusion_classes = sorted({label for labels in subset_meta.labels for label in labels})
train_ids = subset_meta[subset_meta.split == "train"].patch_id.astype(str).tolist()
val_ids = subset_meta[subset_meta.split == "validation"].patch_id.astype(str).tolist()
if not val_ids:
    val_ids = subset_meta[subset_meta.split == "val"].patch_id.astype(str).tolist()
assert set(train_ids).isdisjoint(val_ids)

fusion_train_loader = DataLoader(FusionDataset(train_ids, BEN, fusion_classes, training=True), batch_size=8, shuffle=True, num_workers=2)
fusion_val_loader = DataLoader(FusionDataset(val_ids, BEN, fusion_classes), batch_size=8, shuffle=False, num_workers=2)

backbone_name = "terramind_v1_tiny" if runtime["vram_gib"] < 22 else "terramind_v1_base"
tm_backbone = BACKBONE_REGISTRY.build(
    backbone_name, pretrained=True, modalities=["S2L2A", "S1GRD"], merge_method="mean"
).cuda()
probe = next(iter(fusion_train_loader))
with torch.inference_mode(), torch.autocast("cuda", dtype=COMPUTE_DTYPE):
    probe_features = tm_backbone({"S2L2A": probe["s2"].cuda(), "S1GRD": probe["s1"].cuda()})[-1]
embedding_dim = probe_features.shape[-1]
del probe_features, probe

fusion_model = TerraMindFusionExpert(tm_backbone, embedding_dim, len(fusion_classes)).cuda()
for parameter in fusion_model.backbone.parameters():
    parameter.requires_grad = False


@torch.inference_mode()
def evaluate_fusion(loader, mode: str) -> dict:
    fusion_model.eval()
    all_targets, all_probabilities = [], []
    for batch in loader:
        s2 = batch["s2"].cuda() if mode in {"fused", "s2"} else None
        s1 = batch["s1"].cuda() if mode in {"fused", "s1"} else None
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = fusion_model(s2=s2, s1=s1)
        all_targets.append(batch["target"].numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities >= 0.5
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "macro_average_precision": float(average_precision_score(targets, probabilities, average="macro")),
    }


FUSION_DIR = ROOT / "artifacts" / "satquery-terramind-fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)
best_f1 = -1.0
head_parameters = [p for n, p in fusion_model.named_parameters() if "backbone" not in n and p.requires_grad]
optimizer = torch.optim.AdamW(head_parameters, lr=2e-4, weight_decay=0.01)
fusion_scaler = torch.amp.GradScaler("cuda", enabled=not USE_BF16)
for epoch in range(5):
    if epoch == 1:
        for parameter in fusion_model.backbone.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in fusion_model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 2e-4},
            {"params": [p for n, p in fusion_model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
        ], weight_decay=0.01)
    fusion_model.train()
    if epoch == 0:
        fusion_model.backbone.eval()
    for batch in tqdm(fusion_train_loader, desc=f"fusion epoch {epoch+1}"):
        optimizer.zero_grad(set_to_none=True)
        s2, s1, target = batch["s2"].cuda(), batch["s1"].cuda(), batch["target"].cuda()
        # Modality dropout improves robustness; never drop both.
        draw = random.random()
        s2_input = None if draw < 0.10 else s2
        s1_input = None if 0.10 <= draw < 0.20 else s1
        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = fusion_model(s2=s2_input, s1=s1_input)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        fusion_scaler.scale(loss).backward()
        fusion_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
        fusion_scaler.step(optimizer)
        fusion_scaler.update()
    fused_metrics = evaluate_fusion(fusion_val_loader, "fused")
    print({"epoch": epoch + 1, "fused": fused_metrics})
    if fused_metrics["macro_f1"] > best_f1:
        best_f1 = fused_metrics["macro_f1"]
        save_safetensors(fusion_model.state_dict(), FUSION_DIR / "model.safetensors")

ablation = {mode: evaluate_fusion(fusion_val_loader, mode) for mode in ("fused", "s2", "s1")}
(FUSION_DIR / "config.json").write_text(json.dumps({
    "architecture": "TerraMind S2L2A+S1GRD mean-fusion with multilabel and patch-evidence heads",
    "backbone": backbone_name, "classes": fusion_classes, "bands_s2": S2_BANDS,
    "bands_s1": S1_BANDS, "ablation": ablation,
}, indent=2), encoding="utf-8")
print(json.dumps(ablation, indent=2))

# %% [markdown]
# ## 11. Release gate — do not skip
#
# A model is ready for backend integration only when every applicable item passes:
#
# - [ ] Qwen base revision is an immutable SHA and adapter weights are SafeTensors.
# - [ ] BigEarthNet text rows are joined to real imagery by `patch_id`.
# - [ ] No patch ID overlaps train, validation, test or benchmark splits.
# - [ ] VQA, captioning and grounding are evaluated separately.
# - [ ] Grounding conversion is `0..1 → 0..1000`, and the conversion assertions pass.
# - [ ] Fresh-session Hub reload produces a sensible answer.
# - [ ] `sha256_manifest.json`, training manifest, metrics and model card are uploaded.
# - [ ] The protected endpoint returns `score_kind`; it does not invent calibrated confidence.
# - [ ] HF token exists only in the backend/secret manager, never in frontend code.
# - [ ] Change-VQA uses actual paired images and actual label-derived mask supervision.
# - [ ] Fusion uses all 12 S2 bands plus `[VV,VH]`, and fused/S2-only/S1-only ablations are recorded.
# - [ ] Domain-gap limitation (Europe Sentinel → India Cartosat/RISAT) is visible in the report.
# - [ ] Endpoint cost, scale-to-zero behavior, cold-start latency and a cached demo fallback are tested.
#
# Recommended final benchmark additions after this notebook succeeds: VRSBench/RSVQA for external
# single-image VQA and grounding evaluation, CDVQA Test/Test2 for change QA, and a geographically
# held-out BigEarthNet v2 test for fusion. Never tune on those test splits.
