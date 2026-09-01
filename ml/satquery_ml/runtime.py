from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RuntimeInfo:
    platform: str
    python: str
    environment: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_vram_gib: float | None
    disk_free_gib: float
    packages: dict[str, str]


def detect_environment() -> str:
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.getenv("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def inspect_runtime(path: Path | None = None) -> RuntimeInfo:
    path = path or Path.cwd()
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_vram_gib = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    except ImportError:
        pass

    package_names = [
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "rasterio",
    ]
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    free = shutil.disk_usage(path).free / 2**30
    return RuntimeInfo(
        platform=platform.platform(),
        python=sys.version.split()[0],
        environment=detect_environment(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_vram_gib=gpu_vram_gib,
        disk_free_gib=free,
        packages=packages,
    )


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_runtime_manifest(destination: Path, info: RuntimeInfo) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(info), indent=2, sort_keys=True), encoding="utf-8"
    )


def require_gpu(min_vram_gib: float) -> RuntimeInfo:
    info = inspect_runtime()
    if not info.cuda_available:
        raise RuntimeError("CUDA GPU is required for this stage")
    if info.gpu_vram_gib is None or info.gpu_vram_gib < min_vram_gib:
        raise RuntimeError(
            f"This stage needs at least {min_vram_gib:.1f} GiB VRAM; found {info.gpu_vram_gib or 0:.1f}"
        )
    return info
