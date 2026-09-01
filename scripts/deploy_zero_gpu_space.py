from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, SpaceHardware
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_REPO = "aanandmodi/satquery-qwen3vl-space"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the audited SatQuery Gradio app without provisioning paid hardware."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--space-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "deploy" / "huggingface_space",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set HF_TOKEN in the current terminal. Never paste it into source code.")
    if not args.space_dir.is_dir():
        raise SystemExit(f"Space directory does not exist: {args.space_dir}")

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,
            private=args.private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 402:
            raise SystemExit(
                "ZeroGPU is unavailable to this account. Run "
                "scripts/request_zero_gpu_grant.py; never fall back to paid hardware."
            ) from exc
        raise
    # create_repo(exist_ok=True) preserves old settings, so explicitly enforce
    # the only unpaid Gradio tier. Never substitute a billable accelerator.
    api.request_space_hardware(args.repo_id, SpaceHardware.ZERO_A10G)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=args.space_dir,
        ignore_patterns=["tests/**", "**/__pycache__/**", "*__pycache__*", "*.pyc", ".pytest_cache/**"],
        commit_message="Deploy pinned SatQuery ZeroGPU inference service",
    )
    print(f"Uploaded commit: {commit.oid}")
    print(f"Space: https://huggingface.co/spaces/{args.repo_id}")
    print("Hardware: ZeroGPU (zero-a10g). This script never requests paid hardware.")


if __name__ == "__main__":
    main()
