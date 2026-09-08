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
# # SatQuery Qwen3-VL — evaluation only
#
# Use this notebook after training. It downloads the exact public adapter and immutable base
# revision, reconstructs a leakage-safe BigEarthNet.txt validation/test subset, evaluates every
# supported task, compares the pinned base against base+LoRA on the exact same inputs, and exports
# raw predictions plus real generation scores. It performs **no training** and provisions **no
# paid endpoint**. Run on a free Kaggle/Colab GPU; stop the GPU after the final PASS cell.

# %% [markdown]
# ## 0. Install the pinned evaluation environment

# %%
import subprocess
import sys

PACKAGES = [
    "transformers==4.57.1",
    "accelerate==1.7.0",
    "peft==0.17.1",
    "bitsandbytes==0.47.0",
    "huggingface_hub==0.36.2",
    "qwen-vl-utils==0.0.14",
    "duckdb==1.3.2",
    "pandas>=2.2,<3",
    "pyarrow>=18,<23",
    "lmdb>=1.5,<2",
    "safetensors>=0.5,<1",
    "tqdm>=4.66,<5",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
print("Installed. Restart the runtime once only if a later import reports a cached old package.")

# %% [markdown]
# ## 1. Configuration
#
# `validation` is the repeatable development benchmark. Run `test` once for the final report.
# The default 200 examples is a bounded evidence run that fits a free T4 session. Each example is
# evaluated twice without loading a second copy of the model: PEFT temporarily disables the adapter
# for the base pass.

# %%
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    adapter_repo: str = "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora"
    adapter_revision: str = "ed12e59e0def9468bdf4a226789fc1b77c7900e7"
    base_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    base_revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    text_repo: str = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
    text_revision: str = "72d865f2146f0a85b720f7f3ca1cdbaeafc3d316"
    image_repo: str = "hackelle/BigEarthNetV2-Lithuania-Summer-LMDB"
    image_revision: str = "7a83ae701109ec232d40665b9677fc46310a1a8b"
    split: str = "validation"  # validation or test
    rows_per_type: int = 50
    seed: int = 42
    image_size: int = 448
    max_new_tokens: int = 128
    upload_results: bool = False


CFG = Config()
assert CFG.split in {"validation", "test"}
ROOT = Path("/kaggle/working/satquery-eval" if Path("/kaggle/working").exists() else "/content/satquery-eval")
DATA_DIR, OUTPUT_DIR = ROOT / "data", ROOT / "outputs"
for path in (DATA_DIR, OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
print(CFG)

# %% [markdown]
# ## 2. GPU and artifact integrity gates

# %%
import json
import random
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle/Colab GPU before continuing.")

api = HfApi()
adapter_info = api.model_info(CFG.adapter_repo, revision=CFG.adapter_revision)
assert adapter_info.sha == CFG.adapter_revision
base_revision_file = Path(hf_hub_download(
    CFG.adapter_repo, "base_revision.txt", revision=CFG.adapter_revision
))
assert base_revision_file.read_text(encoding="utf-8").strip() == CFG.base_revision
print({"gpu": torch.cuda.get_device_name(0), "adapter_sha": adapter_info.sha, "base_sha": CFG.base_revision})

# %% [markdown]
# ## 3. Download and join text records to matching imagery
#
# The user-supplied dataset is a text/QA table, not the image pixels. The 2.48 GB LMDB mirror below
# supplies matching Sentinel-2 imagery by patch ID. The SQL query preserves the publisher split and
# applies a deterministic per-task cap without loading all 9.55 million text rows into RAM.

# %%
text_path = Path(hf_hub_download(
    CFG.text_repo,
    "BigEarthNet.txt.parquet",
    repo_type="dataset",
    revision=CFG.text_revision,
    local_dir=DATA_DIR / "text",
))
image_root = Path(snapshot_download(
    CFG.image_repo,
    repo_type="dataset",
    revision=CFG.image_revision,
    local_dir=DATA_DIR / "images",
    allow_patterns=["BENv2_lithuania_summer.lmdb/*", "metadata_lithuania_summer.parquet"],
))
lmdb_dir = image_root / "BENv2_lithuania_summer.lmdb"
metadata_path = image_root / "metadata_lithuania_summer.parquet"
required = [text_path, metadata_path, lmdb_dir / "data.mdb", lmdb_dir / "lock.mdb"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Incomplete dataset download: {missing}")

# %%
import duckdb
import pandas as pd


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


prepared_path = DATA_DIR / f"{CFG.split}-evaluation.parquet"
duckdb.sql(f"""
COPY (
  WITH matched AS (
    SELECT t.ID, t.patch_id, t.input, t.output, t.type, t.category, t.split
    FROM read_parquet('{sql_path(text_path)}') t
    SEMI JOIN read_parquet('{sql_path(metadata_path)}') m USING (patch_id)
    WHERE t.split = '{CFG.split}'
      AND t.type IN ('binary', 'mcq', 'captioning', 'bounding box')
      AND lower(t.category) NOT IN ('country','season','climate zone','climate_zone','cliamte zone')
  ), ranked AS (
    SELECT *, row_number() OVER (
      PARTITION BY type ORDER BY hash(CAST(ID AS VARCHAR) || '{CFG.seed}')
    ) AS rank
    FROM matched
  )
  SELECT * EXCLUDE (rank) FROM ranked WHERE rank <= {CFG.rows_per_type}
) TO '{sql_path(prepared_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
frame = pd.read_parquet(prepared_path)
assert frame.ID.is_unique and not frame[["patch_id", "input", "output", "type"]].isna().any().any()
assert set(frame.type) == {"binary", "mcq", "captioning", "bounding box"}
print(frame.groupby("type").size())

# %% [markdown]
# ## 4. Safe LMDB image reader and coordinate conversion

# %%
import lmdb
import re
from PIL import Image
from safetensors.numpy import load as load_safetensors_bytes

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
NUMBER = r"[-+]?\d*\.?\d+"


class ImageStore:
    def __init__(self, path: Path):
        self.path = path
        self._env = None

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=True, meminit=False)
        return self._env

    def rgb(self, patch_id: str) -> Image.Image:
        with self.env.begin(write=False, buffers=True) as transaction:
            raw = transaction.get(str(patch_id).encode("utf-8"))
        if raw is None:
            raise KeyError(f"LMDB image missing for {patch_id}")
        bands = load_safetensors_bytes(bytes(raw))
        if not set(S2_BANDS).issubset(bands):
            raise ValueError(f"S2 bands missing for {patch_id}")
        channels = []
        for name in ("B04", "B03", "B02"):
            band = np.asarray(bands[name], dtype=np.float32)
            low, high = np.nanpercentile(band, (2, 98))
            channels.append((np.clip((band - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8))
        image = Image.fromarray(np.stack(channels, axis=-1))
        return image.resize((CFG.image_size, CFG.image_size), Image.Resampling.BILINEAR)


def qwen_coordinate(value: float) -> int:
    if not 0 <= value <= 1:
        raise ValueError(f"Coordinate outside [0,1]: {value}")
    return int(round(1000 * value))


def convert_question(text: str) -> str:
    pattern = re.compile(rf"<point>\s*\(({NUMBER})\s*,\s*({NUMBER})\)\s*</point>")
    return pattern.sub(lambda m: f"<point>({qwen_coordinate(float(m[1]))},{qwen_coordinate(float(m[2]))})</point>", text)


def reference_answer(row: pd.Series) -> str:
    if row["type"] != "bounding box":
        return str(row["output"]).strip()
    values = [float(item) for item in re.findall(NUMBER, str(row["output"]))]
    if len(values) != 4 or not (0 <= values[0] <= values[2] <= 1 and 0 <= values[1] <= values[3] <= 1):
        raise ValueError(f"Invalid reference box: {row['output']}")
    box = [qwen_coordinate(item) for item in values]
    return f"<box>({box[0]},{box[1]}),({box[2]},{box[3]})</box>"


images = ImageStore(lmdb_dir)
display(images.rgb(str(frame.iloc[0].patch_id)))
print(convert_question(str(frame.iloc[0].input)), reference_answer(frame.iloc[0]))

# %% [markdown]
# ## 5. Load the published adapter in 4-bit and run a smoke test

# %%
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(CFG.adapter_repo, revision=CFG.adapter_revision, trust_remote_code=False)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    CFG.base_model,
    revision=CFG.base_revision,
    trust_remote_code=False,
    quantization_config=quantization,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, CFG.adapter_repo, revision=CFG.adapter_revision, is_trainable=False)
model.eval()


def prepared_inputs(image: Image.Image, question: str):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    return batch


@torch.inference_mode()
def generate_scored(batch) -> tuple[str, float, float]:
    output = model.generate(
        **batch,
        max_new_tokens=CFG.max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated = output.sequences[:, batch["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    transition = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )[0]
    # This is a real generation statistic, not a calibrated correctness probability.
    mean_log_probability = float(transition.mean().item()) if transition.numel() else -20.0
    sequence_confidence = float(np.clip(np.exp(mean_log_probability), 1e-6, 1 - 1e-6))
    sequence_logit = float(np.log(sequence_confidence / (1 - sequence_confidence)))
    return text, sequence_confidence, sequence_logit


def predict_base_and_lora(image: Image.Image, question: str) -> dict[str, float | str]:
    batch = prepared_inputs(image, question)
    lora_pred, lora_confidence, lora_logit = generate_scored(batch)
    with model.disable_adapter():
        base_pred, base_confidence, base_logit = generate_scored(batch)
    return {
        "base_pred": base_pred,
        "base_sequence_confidence": base_confidence,
        "base_sequence_logit": base_logit,
        "lora_pred": lora_pred,
        "lora_sequence_confidence": lora_confidence,
        "lora_sequence_logit": lora_logit,
    }


smoke = frame.iloc[0]
print({
    "question": str(smoke.input),
    "reference": reference_answer(smoke),
    **predict_base_and_lora(
        images.rgb(str(smoke.patch_id)), convert_question(str(smoke.input))
    ),
})

# %% [markdown]
# ## 6. Full bounded evaluation
#
# Exact match is used for binary/MCQ, IoU for grounding, and token F1 as a caption smoke metric.
# These are task metrics—not a fabricated confidence score.

# %%
import string
from collections import Counter
from tqdm.auto import tqdm


def normalize(text: str) -> str:
    table = str.maketrans("", "", string.punctuation.replace("<", "").replace(">", ""))
    return " ".join(str(text).lower().translate(table).split())


def parse_box(text: str):
    values = [int(round(float(item))) for item in re.findall(NUMBER, str(text))]
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = values[:4]
    return (x1, y1, x2, y2) if 0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000 else None


def iou(left, right) -> float:
    ix1, iy1, ix2, iy2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1)


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref:
        return float(pred == ref)
    return 0.0 if not common else 2 * (common / len(pred)) * (common / len(ref)) / ((common / len(pred)) + (common / len(ref)))


def score_prediction(prediction: str, reference: str, task_type: str):
    exact, box_score, caption_score = int(normalize(prediction) == normalize(reference)), None, None
    if task_type == "bounding box":
        predicted_box, reference_box = parse_box(prediction), parse_box(reference)
        box_score = iou(predicted_box, reference_box) if predicted_box and reference_box else 0.0
        exact = int(box_score >= 0.5)
    elif task_type == "captioning":
        caption_score = token_f1(prediction, reference)
    return exact, box_score, caption_score


records = []
for _, row in tqdm(frame.iterrows(), total=len(frame)):
    question = convert_question(str(row.input).strip())
    reference = reference_answer(row)
    compared = predict_base_and_lora(images.rgb(str(row.patch_id)), question)
    base_correct, base_iou, base_caption_f1 = score_prediction(
        str(compared["base_pred"]), reference, str(row.type)
    )
    lora_correct, lora_iou, lora_caption_f1 = score_prediction(
        str(compared["lora_pred"]), reference, str(row.type)
    )
    records.append({
        "id": int(row.ID), "patch_id": str(row.patch_id), "type": str(row.type),
        "question": question, "reference": reference,
        **compared,
        "base_correct": base_correct,
        "base_iou": base_iou,
        "base_caption_token_f1": base_caption_f1,
        "lora_correct": lora_correct,
        "lora_iou": lora_iou,
        "lora_caption_token_f1": lora_caption_f1,
    })

predictions = pd.DataFrame(records)
predictions.to_json(OUTPUT_DIR / f"{CFG.split}_predictions.jsonl", orient="records", lines=True)
vqa = predictions.type.isin(["binary", "mcq"])
grounding = predictions.type == "bounding box"
captioning = predictions.type == "captioning"
summary = {
    "adapter_revision": CFG.adapter_revision,
    "base_revision": CFG.base_revision,
    "dataset_revision": CFG.text_revision,
    "image_dataset_revision": CFG.image_revision,
    "split": CFG.split,
    "rows": len(predictions),
    "base": {
        "vqa_exact_match": float(predictions.loc[vqa, "base_correct"].mean()),
        "grounding_mean_iou": float(predictions.loc[grounding, "base_iou"].mean()),
        "grounding_accuracy_iou_0_5": float(predictions.loc[grounding, "base_correct"].mean()),
        "caption_token_f1": float(predictions.loc[captioning, "base_caption_token_f1"].mean()),
    },
    "lora": {
        "vqa_exact_match": float(predictions.loc[vqa, "lora_correct"].mean()),
        "grounding_mean_iou": float(predictions.loc[grounding, "lora_iou"].mean()),
        "grounding_accuracy_iou_0_5": float(predictions.loc[grounding, "lora_correct"].mean()),
        "caption_token_f1": float(predictions.loc[captioning, "lora_caption_token_f1"].mean()),
    },
    "confidence_semantics": (
        "sequence_confidence is an observed generation statistic, not a correctness probability. "
        "Use scripts/score-real-evaluation.py for held-out temperature scaling and bootstrap CIs."
    ),
}
(OUTPUT_DIR / f"{CFG.split}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 7. Optional result upload and safe stop
#
# Uploading small metric files to the existing Hub repository is free. It is disabled by default
# because the model itself is already released. If enabled, keep `HF_TOKEN` in Kaggle/Colab Secrets.

# %%
if CFG.upload_results:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set HF_TOKEN as a notebook secret; never paste it into this cell.")
    HfApi(token=token).upload_folder(
        repo_id=CFG.adapter_repo,
        repo_type="model",
        folder_path=OUTPUT_DIR,
        path_in_repo=f"evaluation/{CFG.split}",
        commit_message=f"Add pinned {CFG.split} evaluation",
    )

gate = {
    "adapter_revision_verified": adapter_info.sha == CFG.adapter_revision,
    "predictions_written": (OUTPUT_DIR / f"{CFG.split}_predictions.jsonl").is_file(),
    "summary_written": (OUTPUT_DIR / f"{CFG.split}_summary.json").is_file(),
    "paid_endpoint_created": False,
}
print(json.dumps(gate, indent=2))
assert all(value for key, value in gate.items() if key != "paid_endpoint_created")
assert gate["paid_endpoint_created"] is False
print("PASS — evaluation complete. Download outputs, save the notebook version, and stop the GPU.")
