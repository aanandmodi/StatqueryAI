from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


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
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=args.space_dir,
        ignore_patterns=["tests/**", "**/__pycache__/**", ".pytest_cache/**"],
        commit_message="Deploy pinned SatQuery ZeroGPU inference service",
    )
    print(f"Uploaded commit: {commit.oid}")
    print(f"Space: https://huggingface.co/spaces/{args.repo_id}")
    print("Required account step: Settings -> Hardware -> ZeroGPU. Do not choose paid hardware.")


if __name__ == "__main__":
    main()

