from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from satquery_ml.artifacts import verify_artifact_manifest
from satquery_ml.registry import DATASETS, MODELS
from satquery_ml.runtime import inspect_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery ML pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Inspect GPU, disk and package versions")
    subparsers.add_parser("registry", help="Show chosen datasets and models")
    verify = subparsers.add_parser(
        "verify-artifact", help="Verify exported artifact hashes"
    )
    verify.add_argument("path", type=Path)
    train = subparsers.add_parser("train-vlm", help="Run Qwen3-VL QLoRA training")
    train.add_argument("annotations", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    train.add_argument("--revision")
    train.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    if args.command == "doctor":
        print(json.dumps(asdict(inspect_runtime()), indent=2, sort_keys=True))
    elif args.command == "registry":
        print(
            json.dumps(
                {
                    "datasets": {key: asdict(value) for key, value in DATASETS.items()},
                    "models": {key: asdict(value) for key, value in MODELS.items()},
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "verify-artifact":
        verify_artifact_manifest(args.path)
        print("Artifact manifest verified.")
    elif args.command == "train-vlm":
        from satquery_ml.training.vlm_qlora import VLMTrainingConfig, train_qwen_qlora

        revision = train_qwen_qlora(
            args.annotations,
            args.output,
            config=VLMTrainingConfig(
                model_id=args.model,
                revision=args.revision,
                epochs=args.epochs,
            ),
        )
        print(f"Saved adapter against immutable base revision {revision}")


if __name__ == "__main__":
    main()
