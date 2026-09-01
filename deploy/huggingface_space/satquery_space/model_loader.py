from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ADAPTER_REPO = "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora"
ADAPTER_REVISION = "ed12e59e0def9468bdf4a226789fc1b77c7900e7"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_VERSION = f"{ADAPTER_REPO}@{ADAPTER_REVISION[:12]}"


@dataclass(frozen=True)
class Runtime:
    processor: AutoProcessor
    model: PeftModel


def load_runtime() -> Runtime:
    processor = AutoProcessor.from_pretrained(
        ADAPTER_REPO,
        revision=ADAPTER_REVISION,
        trust_remote_code=False,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO,
        revision=ADAPTER_REVISION,
        is_trainable=False,
    )
    # ZeroGPU emulates CUDA during Space startup. Keeping the model global prevents
    # each queued request from paying the several-minute model loading cost.
    model = model.to("cuda")
    model.eval()
    return Runtime(processor=processor, model=model)


@torch.inference_mode()
def generate(runtime: Runtime, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages)
    batch = runtime.processor(
        text=[text_prompt],
        images=images,
        videos=videos,
        return_tensors="pt",
    )
    batch = {key: value.to(runtime.model.device) for key, value in batch.items()}
    output_ids = runtime.model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generated_ids = output_ids[:, batch["input_ids"].shape[1] :]
    return runtime.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

