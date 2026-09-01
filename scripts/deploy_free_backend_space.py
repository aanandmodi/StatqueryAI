from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name} in the current terminal. Never commit it to the repository.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy SatQuery API to the second free ZeroGPU slot; never request paid hardware."
    )
    parser.add_argument("--repo-id", default="aanandmodi/satquery-api")
    parser.add_argument(
        "--model-space-url",
        default="https://aanandmodi-satquery-qwen3vl-space.hf.space",
    )
    parser.add_argument("--allowed-origin", default="http://localhost:3000")
    args = parser.parse_args()

    hf_token = require_secret("HF_TOKEN")
    api_key = require_secret("SATQUERY_API_KEY")
    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware=SpaceHardware.ZERO_A10G,
        private=False,
        exist_ok=True,
    )
    api.request_space_hardware(args.repo_id, SpaceHardware.ZERO_A10G)

    with tempfile.TemporaryDirectory(prefix="satquery-api-space-") as temporary:
        stage = Path(temporary)
        shutil.copy2(PROJECT_ROOT / "deploy" / "huggingface_backend" / "README.md", stage)
        shutil.copy2(PROJECT_ROOT / "deploy" / "huggingface_backend" / "app.py", stage)
        shutil.copy2(PROJECT_ROOT / "backend" / "requirements.txt", stage)
        shutil.copytree(
            PROJECT_ROOT / "backend" / "app",
            stage / "app",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=stage,
            commit_message="Deploy SatQuery free ZeroGPU orchestration API",
        )

    variables = {
        "SATQUERY_ENVIRONMENT": "production",
        "SATQUERY_MODEL_BACKEND": "space",
        "SATQUERY_SPACE_URL": args.model_space_url,
        "SATQUERY_ALLOWED_ORIGINS": args.allowed_origin,
        "SATQUERY_ENABLE_DOCS": "false",
        "GRADIO_SSR_MODE": "false",
        "SATQUERY_MAX_UPLOAD_BYTES": str(50 * 1024 * 1024),
        "SATQUERY_JOB_TIMEOUT_SECONDS": "600",
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id=args.repo_id, key=key, value=value)
    api.add_space_secret(
        repo_id=args.repo_id,
        key="SATQUERY_API_KEY",
        value=api_key,
        description="Shared only with the server-side frontend proxy",
    )

    print(f"Uploaded commit: {commit.oid}")
    print(f"API Space: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Readiness: https://{args.repo_id.replace('/', '-')}.hf.space/v1/health/ready")
    print("Hardware: ZeroGPU (zero-a10g). Orchestration routes do not reserve GPU time.")


if __name__ == "__main__":
    main()
