from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VLMTrainingConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str | None = None
    epochs: float = 2.0
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 16
    max_length: int = 4096
    max_pixels: int = 576 * 28 * 28
    min_pixels: int = 16 * 28 * 28
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42


def resolve_model_revision(model_id: str, requested_revision: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=requested_revision)
    if not info.sha:
        raise RuntimeError(f"Unable to resolve an immutable revision for {model_id}")
    return info.sha


class ConversationDataset:
    def __init__(self, annotation_path: Path) -> None:
        self.records = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "image" not in record or "conversations" not in record:
                    raise ValueError(f"invalid training record at line {line_number}")
                self.records.append(record)
        if not self.records:
            raise ValueError("training annotations are empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class QwenConversationCollator:
    def __init__(self, processor: Any, *, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info
        from torch.nn.utils.rnn import pad_sequence

        input_ids = []
        attention_masks = []
        labels = []
        pixel_values = []
        image_grids = []
        for record in examples:
            image = record["image"]
            conversations = record["conversations"]
            human = next(
                item["value"] for item in conversations if item["from"] == "human"
            )
            answer = next(
                item["value"] for item in conversations if item["from"] == "gpt"
            )
            human = human.replace("<image>", "").strip()
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": human},
                    ],
                }
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            prompt_text = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            images, videos = process_vision_info(full_messages)
            full = self.processor(
                text=[full_text],
                images=images,
                videos=videos,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt = self.processor(
                text=[prompt_text],
                images=images,
                videos=videos,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            ids = full.input_ids[0]
            item_labels = ids.clone()
            item_labels[: min(prompt.input_ids.shape[1], item_labels.shape[0])] = -100
            input_ids.append(ids)
            attention_masks.append(full.attention_mask[0])
            labels.append(item_labels)
            pixel_values.append(full.pixel_values)
            image_grids.append(full.image_grid_thw)

        pad_id = self.processor.tokenizer.pad_token_id
        return {
            "input_ids": pad_sequence(
                input_ids, batch_first=True, padding_value=pad_id
            ),
            "attention_mask": pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grids, dim=0),
        }


def train_qwen_qlora(
    annotation_path: Path,
    output_dir: Path,
    *,
    config: VLMTrainingConfig,
) -> str:
    """Fine-tune Qwen3-VL with 4-bit QLoRA and save adapter-only safetensors."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    revision = resolve_model_revision(config.model_id, config.revision)
    capability = torch.cuda.get_device_capability()
    bf16 = capability[0] >= 8
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=revision,
        trust_remote_code=False,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        revision=revision,
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
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
        raise RuntimeError("No language-model linear modules were found for LoRA")
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    dataset = ConversationDataset(annotation_path)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=QwenConversationCollator(processor, max_length=config.max_length),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    (output_dir / "base_revision.txt").write_text(f"{revision}\n", encoding="utf-8")
    return revision
