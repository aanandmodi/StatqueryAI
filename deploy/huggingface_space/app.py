from __future__ import annotations

import os
import time

# ZeroGPU must patch CUDA before Gradio, Transformers, or any other library can
# initialize torch. Expandable segments also prevent avoidable transient OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import spaces

import gradio as gr
from PIL import Image

from satquery_space.contracts import build_task_prompt, make_response, validate_request
from satquery_space.inference import decode_base64_image, parse_grounding_boxes
from satquery_space.model_loader import MODEL_VERSION, generate, load_runtime


# ZeroGPU requires the CUDA-backed model to be created at module scope.
RUNTIME = load_runtime()


def _analyze(image: Image.Image, task: str, question: str, max_new_tokens: int) -> dict:
    started = time.perf_counter()
    clean_task, clean_question, token_limit = validate_request(task, question, max_new_tokens)
    prompt = build_task_prompt(clean_task, clean_question)
    answer = generate(RUNTIME, image.convert("RGB"), prompt, token_limit)
    evidence = parse_grounding_boxes(answer) if clean_task == "grounding" else []
    response = make_response(
        task=clean_task,
        answer=answer,
        evidence=evidence,
        model_version=MODEL_VERSION,
    )
    response["latency_seconds"] = round(time.perf_counter() - started, 3)
    return response


@spaces.GPU(duration=60)
def analyze_api(image_base64: str, task: str, question: str, max_new_tokens: int) -> dict:
    """Stable named endpoint consumed by the SatQuery FastAPI backend."""
    return _analyze(decode_base64_image(image_base64), task, question, max_new_tokens)


@spaces.GPU(duration=60)
def analyze_ui(image: Image.Image | None, task: str, question: str, max_new_tokens: int) -> dict:
    """Analyze one RGB satellite image with the published SatQuery adapter."""
    if image is None:
        raise gr.Error("Upload an RGB satellite image first.")
    return _analyze(image, task, question, max_new_tokens)


with gr.Blocks(title="SatQuery Qwen3-VL") as demo:
    gr.Markdown(
        """
        # SatQuery · remote-sensing Qwen3-VL
        Public, zero-cost demo of the pinned SatQuery LoRA adapter. Results are
        **uncalibrated** and must be checked against the visible imagery.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", image_mode="RGB", label="RGB satellite image")
            task_input = gr.Dropdown(
                choices=["single_vqa", "caption", "grounding"],
                value="single_vqa",
                label="Task",
            )
            question_input = gr.Textbox(
                value="What land-cover features are visible?",
                label="Question or instruction",
                lines=3,
            )
            tokens_input = gr.Slider(16, 256, value=128, step=1, label="Maximum new tokens")
            run_button = gr.Button("Analyze image", variant="primary")
        with gr.Column(scale=1):
            result_output = gr.JSON(label="Auditable model response")

    run_button.click(
        analyze_ui,
        inputs=[image_input, task_input, question_input, tokens_input],
        outputs=result_output,
        api_name="analyze-ui",
    )

    # Hidden base64 components make the machine API independent of Gradio's temporary
    # file serialization. FastAPI calls this endpoint via the standard Gradio queue API.
    api_image = gr.Textbox(visible=False)
    api_task = gr.Textbox(visible=False)
    api_question = gr.Textbox(visible=False)
    api_tokens = gr.Number(visible=False)
    api_result = gr.JSON(visible=False)
    api_trigger = gr.Button(visible=False)
    api_trigger.click(
        analyze_api,
        inputs=[api_image, api_task, api_question, api_tokens],
        outputs=api_result,
        api_name="analyze",
    )


if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=1).launch(mcp_server=True)
