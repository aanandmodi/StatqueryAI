---
title: SatQuery Qwen3-VL
emoji: 🛰️
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# SatQuery Qwen3-VL ZeroGPU Space

Free, queue-backed inference for the published SatQuery remote-sensing LoRA adapter.

The Space intentionally exposes only the three tasks supported by the trained artifact:
single-image VQA, captioning, and visual grounding. Change detection and optical/SAR fusion
are separate specialist models and are never silently routed through this adapter.

## API

Every Gradio Space exposes an API automatically. The backend calls the named `analyze`
endpoint with four inputs:

1. a base64-encoded RGB image,
2. `single_vqa`, `caption`, or `grounding`,
3. the question/instruction,
4. a generation limit from 1 to 256 tokens.

The response contains structured text, evidence boxes when parseable, immutable model
revisions, and an explicit `uncalibrated` score label.

## Zero-cost deployment

Upload this folder to a public Gradio Space, then choose **ZeroGPU** in the Space hardware
settings. Do not select a paid instance. Free personal accounts in good standing currently
support up to two ZeroGPU Spaces, subject to Hugging Face quota and availability.

