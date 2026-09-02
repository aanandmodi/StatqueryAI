from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VLMResponse:
    text: str
    parsed: dict[str, Any] | None


class QwenVLRuntime:
    """Lazy Qwen3-VL runtime for RGB/false-colour previews only.

    Raw SAR and 12-band optical tensors are deliberately rejected by architecture;
    TerraMind is the sensor-fusion specialist.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        adapter_path: str | Path | None = None,
        adapter_revision: str | None = None,
        processor_id: str | Path | None = None,
        processor_revision: str | None = None,
        four_bit: bool = True,
        max_pixels: int = 448 * 448,
        device: str = "cuda",
    ) -> None:
        if not revision:
            raise ValueError("A pinned Hugging Face revision is required")
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        quantization = None
        if four_bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Start the free Kaggle/Colab inference notebook or "
                "install a CUDA-enabled PyTorch build and NVIDIA driver."
            )
        if four_bit and device != "cuda":
            raise RuntimeError("4-bit SatQuery inference currently requires CUDA")

        self.processor = AutoProcessor.from_pretrained(
            processor_id or adapter_path or model_id,
            revision=processor_revision or adapter_revision or revision,
            trust_remote_code=False,
            min_pixels=256 * 28 * 28,
            max_pixels=max_pixels,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            quantization_config=quantization,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": 0} if device == "cuda" else {"": "cpu"},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                revision=adapter_revision,
                is_trainable=False,
                autocast_adapter_dtype=False,
                low_cpu_mem_usage=True,
            )
        self.model.eval()

    def generate(
        self,
        image: Path,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        expect_json: bool = False,
    ) -> VLMResponse:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        parsed = None
        if expect_json:
            try:
                parsed = json.loads(answer)
            except json.JSONDecodeError:
                parsed = None
        return VLMResponse(text=answer, parsed=parsed)
