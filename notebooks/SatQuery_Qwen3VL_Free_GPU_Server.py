# ---
# ruff: noqa: E402
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
# # SatQuery — Kaggle/Colab Qwen3-VL server through free ngrok
#
# This notebook loads the released SatQuery LoRA on a free GPU, proves local inference works,
# starts the exact FastAPI contract used by the SatQuery backend, and exposes it temporarily
# through ngrok. It creates no permanent model endpoint or website deployment.
#
# **Before Run all**
#
# 1. Kaggle: Settings → Accelerator → GPU and Internet → On.
# 2. Create a free ngrok account and copy its authtoken.
# 3. Kaggle: Add-ons → Secrets. Add `NGROK_AUTHTOKEN` and a long random
#    `SATQUERY_MODEL_SERVICE_TOKEN`; enable both for this notebook.
# 4. Colab users may enter both secrets into hidden prompts instead.
#
# Never place either token directly in a code cell or public notebook output.

# %%
import subprocess
import sys

PACKAGES = [
    "accelerate==1.7.0",
    "bitsandbytes==0.50.2",
    "fastapi>=0.115,<1",
    "huggingface-hub==0.34.4",
    "peft==0.17.1",
    "pillow>=10,<12",
    "pydantic>=2.10,<3",
    "pyngrok>=7.2,<8",
    "python-multipart>=0.0.20,<1",
    "qwen-vl-utils==0.0.14",
    "rasterio>=1.4,<2",
    "requests>=2.32,<3",
    "transformers==4.57.1",
    "uvicorn>=0.34,<1",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *PACKAGES])
print("Dependencies installed. Restart the runtime once only if the next cell imports old modules.")

# %% [markdown]
# ## 1. Read secrets without printing them

# %%
import getpass
import os


def read_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient

            value = UserSecretsClient().get_secret(name).strip()
        except Exception:
            value = getpass.getpass(f"Enter {name} (input hidden): ").strip()
    if not value:
        raise RuntimeError(f"Missing required secret: {name}")
    return value


NGROK_AUTHTOKEN = read_secret("NGROK_AUTHTOKEN")
SERVICE_TOKEN = read_secret("SATQUERY_MODEL_SERVICE_TOKEN")
assert len(SERVICE_TOKEN) >= 32, "Use a random service token containing at least 32 characters."
print("Secrets loaded safely.")

# %% [markdown]
# ## 2. Verify the free GPU
#
# Stop here if `cuda_available` is false.

# %%
import json
import torch

gpu = {
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    if torch.cuda.is_available()
    else None,
}
print(json.dumps(gpu, indent=2))
assert gpu["cuda_available"], "Enable a Kaggle/Colab GPU before continuing."

# %% [markdown]
# ## 3. Load the exact released base + adapter

# %%
import gc

from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

ADAPTER_REPO = "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora"
ADAPTER_REVISION = "ed12e59e0def9468bdf4a226789fc1b77c7900e7"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_VERSION = f"{ADAPTER_REPO}@{ADAPTER_REVISION[:12]}"

gc.collect()
torch.cuda.empty_cache()
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
processor = AutoProcessor.from_pretrained(
    ADAPTER_REPO,
    revision=ADAPTER_REVISION,
    trust_remote_code=False,
    min_pixels=256 * 28 * 28,
    max_pixels=448 * 448,
)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    trust_remote_code=False,
    quantization_config=quantization,
    dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(
    base,
    ADAPTER_REPO,
    revision=ADAPTER_REVISION,
    is_trainable=False,
    autocast_adapter_dtype=False,
    low_cpu_mem_usage=True,
).eval()
print({"model": MODEL_VERSION, "device": str(model.device)})

# %% [markdown]
# ## 4. Define bounded image, prompt, and evidence handling

# %%
import io
import re
from typing import Any

import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

SUPPORTED_TASKS = {"single_vqa", "caption", "grounding"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
BOX_PATTERN = re.compile(
    r"<box>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*,"
    r"\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*</box>",
    re.IGNORECASE,
)


def scale_band(band: np.ndarray) -> np.ndarray:
    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros(band.shape, dtype=np.uint8)
    low, high = np.percentile(band[finite], [2.0, 98.0])
    if high <= low:
        low, high = float(np.min(band[finite])), float(np.max(band[finite]))
    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)
    output = np.clip((band - low) / (high - low), 0, 1)
    output[~finite] = 0
    return (output * 255).round().astype(np.uint8)


def decode_uploaded_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("image is empty or exceeds 50 MB")
    try:
        with MemoryFile(payload) as memory:
            with memory.open() as source:
                scale = min(1.0, 448 / max(source.width, source.height))
                height = max(1, round(source.height * scale))
                width = max(1, round(source.width * scale))
                bands = [1, 2, 3] if source.count >= 3 else [1, 1, 1]
                raster = source.read(
                    bands,
                    out_shape=(3, height, width),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                rgb = np.stack([scale_band(raster[index]) for index in range(3)], axis=-1)
                return Image.fromarray(rgb, mode="RGB")
    except (rasterio.errors.RasterioError, ValueError):
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.width * source.height > 25_000_000:
                    raise ValueError("image exceeds 25 megapixels")
                image = source.convert("RGB")
                image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                return image
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("upload is not a readable TIFF, PNG, or JPEG") from exc


def context_text(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    safe = {
        "latitude": context.get("latitude"),
        "longitude": context.get("longitude"),
        "altitude_m": context.get("altitude_m"),
        "captured_at": context.get("captured_at"),
        "sensor": context.get("sensor"),
        "source": context.get("source"),
        "metadata": context.get("metadata", {}),
    }
    return (
        "\nUser-supplied context (treat as metadata, not as something inferred from pixels): "
        + json.dumps(safe, ensure_ascii=True, separators=(",", ":"))
    )


def task_prompt(task: str, question: str, context: dict[str, Any] | None) -> str:
    suffix = context_text(context)
    if task == "caption":
        return (
            f"{question}\nReturn a factual remote-sensing scene description. "
            f"Do not invent dates, sensors, coordinates, or confidence values.{suffix}"
        )
    if task == "grounding":
        return (
            f"{question}\nReturn a short answer and every visible target box on a 0..1000 scale "
            f"in the form <box>(x1,y1),(x2,y2)</box>.{suffix}"
        )
    return (
        f"{question}\nAnswer from visible image evidence only. "
        f"If evidence is insufficient, say so.{suffix}"
    )


def parse_boxes(text: str, asset_id: str) -> list[dict[str, Any]]:
    evidence = []
    for match in BOX_PATTERN.finditer(text):
        x1, y1, x2, y2 = [max(0.0, min(1000.0, float(item))) for item in match.groups()]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right <= left or bottom <= top:
            continue
        evidence.append(
            {
                "id": f"ev_{len(evidence) + 1}",
                "type": "box",
                "label": "model-grounded region",
                "score": 0.5,
                "coordinate_space": "normalized",
                "geometry": {
                    "x": left / 1000,
                    "y": top / 1000,
                    "width": (right - left) / 1000,
                    "height": (bottom - top) / 1000,
                },
                "asset_id": asset_id,
                "artifact_url": None,
            }
        )
    return evidence


@torch.inference_mode()
def generate(image: Image.Image, prompt: str, max_new_tokens: int = 128) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = processor(text=[rendered], images=images, videos=videos, return_tensors="pt")
    batch = {name: value.to(model.device) for name, value in batch.items()}
    output = model.generate(
        **batch,
        max_new_tokens=max(1, min(int(max_new_tokens), 128)),
        do_sample=False,
        use_cache=True,
    )
    generated = output[:, batch["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

# %% [markdown]
# ## 5. Local model smoke test — do not continue unless this passes

# %%
sample = Image.new("RGB", (448, 448), (70, 105, 82))
answer = generate(sample, "Is this image mostly green? Answer briefly.", 24)
print({"answer": answer, "model": MODEL_VERSION})
assert answer.strip(), "FAIL: model returned no text."
print("PASS: base + adapter generated text on the Kaggle GPU.")

# %% [markdown]
# ## 6. Start the protected FastAPI model service

# %%
import asyncio
import secrets
import threading
import time

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated, Literal


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Step(StrictModel):
    step_id: str
    task: Literal["single_vqa", "caption", "grounding", "change_vqa", "optical_sar_fusion"]
    asset_ids: list[str]
    permitted_params: dict[str, Any]
    policy_reason: str


class Asset(StrictModel):
    id: str
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str
    role: str
    modality: str
    source_dataset: str | None = None
    created_at: str
    metadata: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)


class GeospatialContext(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float | None = Field(default=None, ge=-500, le=100_000)
    captured_at: str | None = None
    sensor: str | None = Field(default=None, max_length=120)
    source: Literal["user", "gps", "exif", "raster"] = "user"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class InferencePayload(StrictModel):
    step: Step
    query: str = Field(min_length=2, max_length=2_000)
    assets: list[Asset] = Field(min_length=1, max_length=2)
    context: GeospatialContext | None = None


app = FastAPI(title="SatQuery temporary Kaggle model service", docs_url=None, redoc_url=None)
inference_lock = asyncio.Lock()


def authorize(value: str | None) -> None:
    expected = f"Bearer {SERVICE_TOKEN}"
    if value is None or not secrets.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="Invalid service credential")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready", "capability": "vlm", "model_version": MODEL_VERSION}


@app.post("/v1/infer/{task}")
async def infer(
    task: str,
    payload: Annotated[str, Form()],
    assets: Annotated[list[UploadFile], File()],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    authorize(authorization)
    try:
        contract = InferencePayload.model_validate_json(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid inference contract") from exc
    if task != contract.step.task or task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=409, detail="Unsupported or mismatched task")
    if len(assets) != 1 or len(contract.assets) != 1:
        raise HTTPException(status_code=422, detail="Qwen3-VL requires exactly one image")
    data = await assets[0].read(MAX_UPLOAD_BYTES + 1)
    await assets[0].close()
    try:
        image = decode_uploaded_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with inference_lock:
        text = await asyncio.to_thread(
            generate,
            image,
            task_prompt(
                task,
                contract.query,
                contract.context.model_dump(mode="json") if contract.context else None,
            ),
            128,
        )
    evidence = parse_boxes(text, contract.assets[0].id) if task == "grounding" else []
    warnings = [
        "The model score is not a calibrated probability.",
        "The adapter was trained on a bounded European Sentinel subset and requires domain validation.",
    ]
    if task == "grounding" and not evidence:
        warnings.append("No machine-readable box could be parsed; the overlay will contain no boxes.")
    facts: list[dict[str, Any]] = [
        {"name": "execution_mode", "value": "temporary_kaggle_ngrok"},
        {"name": "confidence_semantics", "value": "uncalibrated"},
    ]
    if contract.context:
        facts.append(
            {
                "name": "user_location",
                "value": {
                    "latitude": contract.context.latitude,
                    "longitude": contract.context.longitude,
                    "altitude_m": contract.context.altitude_m,
                },
            }
        )
    return {
        "task": task,
        "text": text,
        "facts": facts,
        "evidence": evidence,
        "raw_score": 0.5,
        "score_kind": "uncalibrated",
        "model_version": MODEL_VERSION,
        "warnings": warnings,
    }


server = uvicorn.Server(
    uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="info", access_log=False)
)
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()
for _ in range(60):
    if server.started:
        break
    time.sleep(0.25)
assert server.started, "FastAPI did not start on port 8080."
print("PASS: protected model service is listening on 127.0.0.1:8080.")

# %% [markdown]
# ## 7. Verify the local HTTP contract before exposing it

# %%
import hashlib
import requests

buffer = io.BytesIO()
sample.save(buffer, format="TIFF")
sample_bytes = buffer.getvalue()
asset = {
    "id": "ast_smoke",
    "original_name": "smoke.tif",
    "content_type": "image/tiff",
    "size_bytes": len(sample_bytes),
    "sha256": hashlib.sha256(sample_bytes).hexdigest(),
    "role": "primary",
    "modality": "optical",
    "source_dataset": None,
    "created_at": "2026-09-02T00:00:00Z",
    "metadata": None,
    "validation_errors": [],
}
contract = {
    "step": {
        "step_id": "step-1",
        "task": "single_vqa",
        "asset_ids": ["ast_smoke"],
        "permitted_params": {},
        "policy_reason": "notebook smoke test",
    },
    "query": "Is the sample mostly green?",
    "assets": [asset],
    "context": {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "altitude_m": 216,
        "source": "user",
        "metadata": {"purpose": "contract smoke test"},
    },
}
local_response = requests.post(
    "http://127.0.0.1:8080/v1/infer/single_vqa",
    headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
    data={"payload": json.dumps(contract)},
    files=[("assets", ("smoke.tif", sample_bytes, "image/tiff"))],
    timeout=180,
)
print({"status": local_response.status_code, "body": local_response.json()})
local_response.raise_for_status()
assert local_response.json()["text"].strip()
print("PASS: backend-compatible local HTTP request returned model text.")

# %% [markdown]
# ## 8. Open the temporary free ngrok tunnel

# %%
from pyngrok import ngrok

ngrok.set_auth_token(NGROK_AUTHTOKEN)
ngrok.kill()
tunnel = ngrok.connect(8080, proto="http", bind_tls=True)
PUBLIC_MODEL_URL = str(tunnel.public_url).replace("http://", "https://")
print("SATQUERY_MODEL_SERVICE_URL=", PUBLIC_MODEL_URL)
print("Do not print SATQUERY_MODEL_SERVICE_TOKEN; copy its existing secret value locally.")

# %% [markdown]
# ## 9. Verify inference through the public tunnel

# %%
public_response = requests.post(
    f"{PUBLIC_MODEL_URL}/v1/infer/single_vqa",
    headers={
        "Authorization": f"Bearer {SERVICE_TOKEN}",
        "ngrok-skip-browser-warning": "1",
    },
    data={"payload": json.dumps(contract)},
    files=[("assets", ("smoke.tif", sample_bytes, "image/tiff"))],
    timeout=180,
)
print({"status": public_response.status_code, "body": public_response.json()})
public_response.raise_for_status()
assert public_response.json()["text"].strip()
print("PASS: internet -> ngrok -> Kaggle -> Qwen3-VL -> structured response works.")

# %% [markdown]
# ## 10. SAFE RUN CELL — keep this cell running while SatQuery is in use
#
# On your computer, set these in `.env`:
#
# ```text
# SATQUERY_MODEL_BACKEND=http
# SATQUERY_MODEL_SERVICE_URL=<the printed HTTPS ngrok URL>
# SATQUERY_MODEL_SERVICE_TOKEN=<the same Kaggle secret>
# ```
#
# Then run the local backend and frontend. Stop **this exact cell** or stop the Kaggle session
# when finished; both the API and tunnel immediately go offline.

# %%
print("SatQuery GPU endpoint is live at:", PUBLIC_MODEL_URL)
print("Interrupt this cell to stop serving. Keep the notebook tab/session alive.")
try:
    while True:
        time.sleep(30)
except KeyboardInterrupt:
    ngrok.disconnect(PUBLIC_MODEL_URL)
    server.should_exit = True
    print("SAFE STOP: ngrok tunnel and FastAPI server stopped.")
