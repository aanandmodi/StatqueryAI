from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


TASKS = (
    ("single_vqa", "Which land-cover features are visible in this scene?"),
    ("caption", "Describe the land-cover and major objects visible in this image."),
    ("grounding", "Highlight the water body referred to in the query."),
    ("change_vqa", "What changed between these two dates, and where did it occur?"),
    (
        "optical_sar_fusion",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all five mandatory SatQuery SIH demonstration workflows."
    )
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--time-a", type=Path, required=True)
    parser.add_argument("--time-b", type=Path, required=True)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--sar", type=Path, required=True)
    parser.add_argument(
        "--single-modality",
        choices=("optical", "multispectral", "sar"),
        default="optical",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sih-acceptance"))
    return parser.parse_args()


def upload(
    client: httpx.Client, base_url: str, path: Path, modality: str, role: str
) -> dict[str, object]:
    with path.open("rb") as handle:
        response = client.post(
            f"{base_url}/assets",
            data={"modality": modality, "role": role},
            files={"file": (path.name, handle, "image/tiff")},
        )
    response.raise_for_status()
    asset = response.json()
    if asset.get("validation_errors"):
        raise RuntimeError(f"{path.name} failed validation: {asset['validation_errors']}")
    return asset


def analyze(
    client: httpx.Client,
    base_url: str,
    task: str,
    query: str,
    asset_ids: list[str],
    timeout_seconds: float,
) -> dict[str, object]:
    response = client.post(
        f"{base_url}/analyses",
        headers={"Idempotency-Key": f"sih-{task}-{time.time_ns()}"},
        json={
            "query": query,
            "asset_ids": asset_ids,
            "requested_tasks": [task],
            "parameters": {},
        },
    )
    response.raise_for_status()
    record = response.json()
    deadline = time.monotonic() + timeout_seconds
    while record["status"] not in {"succeeded", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{task} exceeded {timeout_seconds:g} seconds")
        time.sleep(1.0)
        response = client.get(f"{base_url}/analyses/{record['id']}")
        response.raise_for_status()
        record = response.json()
    if record["status"] != "succeeded":
        raise RuntimeError(
            f"{task} failed: {record.get('error_code')} {record.get('error_message')}"
        )
    result = record.get("result") or {}
    if not str(result.get("answer", "")).strip():
        raise RuntimeError(f"{task} returned an empty answer")
    model_versions = result.get("provenance", {}).get("model_versions", {})
    if task not in model_versions:
        raise RuntimeError(f"{task} result is missing specialist provenance")
    if not result.get("trace"):
        raise RuntimeError(f"{task} result is missing the observable execution trace")
    return record


def save_artifacts(
    client: httpx.Client,
    base_url: str,
    output: Path,
    task: str,
    record: dict[str, object],
) -> None:
    analysis_id = str(record["id"])
    (output / f"{task}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    for suffix, endpoint in (("overlay.jpg", "overlay"), ("report.pdf", "report")):
        response = client.get(f"{base_url}/analyses/{analysis_id}/{endpoint}")
        response.raise_for_status()
        (output / f"{task}.{suffix}").write_bytes(response.content)


def main() -> None:
    args = parse_args()
    paths = [args.single, args.time_a, args.time_b, args.optical, args.sar]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(headers=headers, timeout=60.0) as client:
        capabilities = client.get(f"{args.base_url.rstrip('/')}/capabilities")
        capabilities.raise_for_status()
        available = set(capabilities.json()["tasks"])
        required = {task for task, _ in TASKS}
        if missing_tasks := required - available:
            raise RuntimeError(f"Controller is missing required tasks: {sorted(missing_tasks)}")

        base_url = args.base_url.rstrip("/")
        assets = {
            "single": upload(client, base_url, args.single, args.single_modality, "primary"),
            "time_a": upload(client, base_url, args.time_a, "optical", "time_a"),
            "time_b": upload(client, base_url, args.time_b, "optical", "time_b"),
            "optical": upload(client, base_url, args.optical, "multispectral", "optical"),
            "sar": upload(client, base_url, args.sar, "sar", "sar"),
        }
        task_assets = {
            "single_vqa": [assets["single"]["id"]],
            "caption": [assets["single"]["id"]],
            "grounding": [assets["single"]["id"]],
            "change_vqa": [assets["time_a"]["id"], assets["time_b"]["id"]],
            "optical_sar_fusion": [assets["optical"]["id"], assets["sar"]["id"]],
        }
        summary: dict[str, object] = {"capabilities": sorted(available), "runs": {}}
        for task, query in TASKS:
            record = analyze(
                client,
                base_url,
                task,
                query,
                [str(item) for item in task_assets[task]],
                args.timeout_seconds,
            )
            save_artifacts(client, base_url, args.output, task, record)
            result = record["result"]
            assert isinstance(result, dict)
            summary["runs"][task] = {
                "analysis_id": record["id"],
                "model_versions": result["provenance"]["model_versions"],
                "confidence": result["confidence"],
                "evidence_count": len(result["evidence"]),
            }
            print(f"PASS {task}: {record['id']}")
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"PASS all mandatory SIH workflows; evidence saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()

