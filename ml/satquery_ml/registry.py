from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    purpose: tuple[str, ...]
    source: str
    license: str
    expected_scale: str
    acquisition: str
    caveats: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "vrsbench": DatasetSpec(
        name="VRSBench",
        version="official 2024 release",
        purpose=("vqa", "captioning", "grounding"),
        source="https://huggingface.co/datasets/xiang709/VRSBench",
        license=(
            "Treat as CC-BY-NC-4.0 until upstream GitHub/Hugging Face license metadata "
            "is reconciled; source-image licenses also vary"
        ),
        expected_scale="29,614 images; 29,614 captions; 52,472 referring expressions; 123,221 QA pairs",
        acquisition='datasets.load_dataset("xiang709/VRSBench", streaming=True)',
        caveats=(
            "Split by image, never by QA row.",
            "Some DOTA-derived images are academic-use only.",
            "Verify grounding coordinates and convert from the dataset convention with tests.",
        ),
    ),
    "rsvqa": DatasetSpec(
        name="RSVQA LR/HR",
        version="official repository release",
        purpose=("vqa",),
        source="https://github.com/charlesmarais/RSVQA",
        license="verify source-image and annotation terms before redistribution",
        expected_scale="LR and HR image/question sets",
        acquisition="manual from the official repository/data links",
        caveats=(
            "Preserve official splits.",
            "Do not mix image identities across train/eval.",
        ),
    ),
    "cdvqa": DatasetSpec(
        name="CDVQA",
        version="official repository release",
        purpose=("change_vqa",),
        source="https://github.com/YZHJessica/CDVQA",
        license="research dataset; verify SECOND imagery terms",
        expected_scale="bi-temporal image pairs and generated change questions",
        acquisition="clone annotations; obtain SECOND imagery separately as instructed upstream",
        caveats=(
            "The annotation repository is not the complete imagery package.",
            "Preserve official pair splits.",
            "Evaluate question-only and shuffled-image baselines for shortcut learning.",
        ),
    ),
    "bigearthnet_v2": DatasetSpec(
        name="BigEarthNet v2.0",
        version="2.0",
        purpose=("optical_sar_fusion", "dense_evidence"),
        source="https://bigearth.net/",
        license="CDLA-Permissive-1.0",
        expected_scale="549,488 S1/S2 pairs; about 51 GiB S1 + 59 GiB S2, excluding reference maps",
        acquisition="download official archives/metadata or attach a prepared subset",
        caveats=(
            "Do not download the full archive in an ordinary free notebook session.",
            "Use the official geographic split assignment.",
            "Exclude or flag patches dominated by snow, cloud, or shadow for classification.",
        ),
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    capability: str
    model_id: str
    role: str
    license: str
    minimum_vram_gib: float
    fallback: str | None = None


MODELS: dict[str, ModelSpec] = {
    "vlm": ModelSpec(
        capability="vqa_caption_grounding",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        role="4-bit QLoRA domain adapter on VRSBench",
        license="Apache-2.0 model card (verify pinned revision)",
        minimum_vram_gib=14.0,
        fallback="Qwen/Qwen3-VL-2B-Instruct",
    ),
    "change": ModelSpec(
        capability="change_vqa_and_mask",
        model_id="satquery/change-segformer-b1",
        role="shared encoder, multi-scale difference fusion, answer and mask heads",
        license="project artifact; inherits training-data obligations",
        minimum_vram_gib=10.0,
    ),
    "fusion": ModelSpec(
        capability="optical_sar_fusion",
        model_id="ibm-esa-geospatial/TerraMind-1.0-base",
        role="TerraMind raw S1/S2 backbone with multi-label and segmentation heads",
        license="Apache-2.0 model and code; verify pinned model card",
        minimum_vram_gib=14.0,
        fallback="CROMA-base (MIT) with identical heads and evaluation split",
    ),
}
