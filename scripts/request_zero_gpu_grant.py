from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRANT_TITLE = "Apply for a GPU community grant: Academic project"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the SatQuery holding Space and request a free ZeroGPU grant."
    )
    parser.add_argument("--repo-id", default="aanandmodi/satquery-qwen3vl-space")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set HF_TOKEN in the current terminal. Never commit it to source code.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    holding_commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=PROJECT_ROOT / "deploy" / "huggingface_space_holding",
        commit_message="Publish SatQuery grant-pending demo page",
    )
    source_commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=PROJECT_ROOT / "deploy" / "huggingface_space",
        path_in_repo="zero_gpu_app",
        ignore_patterns=["tests/**", "**/__pycache__/**", "*__pycache__*", "*.pyc", ".pytest_cache/**"],
        commit_message="Publish complete audited ZeroGPU application source",
    )

    existing = list(api.get_repo_discussions(args.repo_id, repo_type="space"))
    discussion = next((item for item in existing if item.title == GRANT_TITLE), None)
    if discussion is None:
        discussion = api.create_discussion(
            repo_id=args.repo_id,
            repo_type="space",
            title=GRANT_TITLE,
            description=(
                "Description of the app: SatQuery is a public Smart India Hackathon educational "
                "demo for auditable satellite-image question answering, captioning, and visual "
                "grounding. It serves an open Qwen3-VL 2B LoRA adapter fine-tuned on the public "
                "BigEarthNet text/grounding corpus and exposes both an accessible UI and a bounded "
                "machine API. The application source, immutable model revisions, evaluation summary, "
                "and safety limits are public in this Space and linked model repository.\n\n"
                "Justification: This is an open-source academic/hackathon project and the owner cannot "
                "cover GPU hosting cost. The account is too new to allocate its otherwise-free "
                "ZeroGPU slot. Qwen3-VL inference requires a GPU, while ZeroGPU's queued, per-user "
                "quota is a good fit for this bounded public demonstration. No paid endpoint or "
                "persistent storage is requested."
            ),
        )

    print(f"Holding commit: {holding_commit.oid}")
    print(f"Source commit: {source_commit.oid}")
    print(f"Space: https://huggingface.co/spaces/{args.repo_id}")
    print(f"Grant discussion: https://huggingface.co/spaces/{args.repo_id}/discussions/{discussion.num}")


if __name__ == "__main__":
    main()
