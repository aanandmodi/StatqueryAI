from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path


BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER_MODEL = "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora"
ADAPTER_REVISION = "ed12e59e0def9468bdf4a226789fc1b77c7900e7"
ESA_SAMPLE = (
    "https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2015/11/"
    "desert_crop_fields/27347103-1-eng-GB/Desert_crop_fields.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the released SatQuery VLM once.")
    parser.add_argument("--image", type=Path, help="Optional RGB JPEG/PNG/TIFF path")
    parser.add_argument(
        "--question",
        default="Describe the dominant land-cover patterns visible in this satellite image.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    import torch
    from PIL import Image

    from satquery_ml.models.vlm import QwenVLRuntime

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is unavailable. Run scripts/setup-local.ps1 or use the free GPU notebook."
        )
    token_limit = max(1, min(args.max_new_tokens, 128))

    with tempfile.TemporaryDirectory(prefix="satquery-smoke-") as temporary:
        root = Path(temporary)
        source = args.image.resolve() if args.image else root / "esa-desert-crops.jpg"
        if args.image is None:
            print("Downloading the ESA Sentinel-2 desert-crops smoke image...", flush=True)
            urllib.request.urlretrieve(ESA_SAMPLE, source)  # noqa: S310
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            preview = image.convert("RGB")
            preview.thumbnail((448, 448), Image.Resampling.LANCZOS)
            preview_path = root / "preview.jpg"
            preview.save(preview_path, format="JPEG", quality=90)

        print("Loading the pinned 4-bit base and LoRA...", flush=True)
        runtime = QwenVLRuntime(
            BASE_MODEL,
            revision=BASE_REVISION,
            adapter_path=ADAPTER_MODEL,
            adapter_revision=ADAPTER_REVISION,
            processor_id=ADAPTER_MODEL,
            processor_revision=ADAPTER_REVISION,
            four_bit=True,
            max_pixels=448 * 448,
            device="cuda",
        )
        response = runtime.generate(
            preview_path,
            (
                f"{args.question.strip()}\nAnswer only from visible image evidence. "
                "Do not invent a location, acquisition date, sensor, or confidence value."
            ),
            max_new_tokens=token_limit,
        )
        result = {
            "passed": bool(response.text.strip()),
            "answer": response.text,
            "base": f"{BASE_MODEL}@{BASE_REVISION}",
            "adapter": f"{ADAPTER_MODEL}@{ADAPTER_REVISION}",
            "cuda_device": torch.cuda.get_device_name(0),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "sample_credit": "Contains modified Copernicus Sentinel data (2015), processed by ESA",
            "sample_license": "CC BY-SA 3.0 IGO or ESA Standard Licence",
        }
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
