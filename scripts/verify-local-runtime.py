from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path


def main() -> int:
    import bitsandbytes
    import torch
    import transformers
    from peft import PeftModel  # noqa: F401

    root = Path(__file__).resolve().parents[1]
    free = shutil.disk_usage(Path.home()).free
    checks = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_vram_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        ),
        "home_free_gib": round(free / 1024**3, 2),
        "project": str(root),
    }
    print(json.dumps(checks, indent=2))
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA-enabled PyTorch cannot see the NVIDIA GPU. "
            "Use notebooks/SatQuery_Qwen3VL_Free_GPU_Server.ipynb instead.",
            file=sys.stderr,
        )
        return 2
    if torch.cuda.get_device_properties(0).total_memory < 3.5 * 1024**3:
        print("ERROR: less than 3.5 GiB VRAM is available to the CUDA device.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
