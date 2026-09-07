from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Any

import httpx


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SatQuery's three evidence-set workflows through the same-origin web proxy "
            "and export a jury-safe local evidence bundle."
        )
    )
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--time-a", type=Path, required=True)
    parser.add_argument("--time-b", type=Path, required=True)
    parser.add_argument("--optical", type=Path, required=True)
    parser.add_argument("--sar", type=Path, required=True)
    parser.add_argument(
        "--base-url", default="http://localhost:3000/api/satquery"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/jury-evidence/runs")
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def upload(
    client: httpx.Client,
    base_url: str,
    path: Path,
    *,
    modality: str,
    role: str,
    registration_basis: str = "geospatial",
    input_profile: str = "exploration",
    source_dataset: str,
) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post(
            f"{base_url}/assets",
            data={
                "modality": modality,
                "role": role,
                "registration_basis": registration_basis,
                "input_profile": input_profile,
                "source_dataset": source_dataset,
            },
            files={"file": (path.name, handle, media_type(path))},
        )
    if response.is_error:
        raise RuntimeError(
            f"Upload failed for {path.name}: HTTP {response.status_code} {response.text[:1200]}"
        )
    asset = response.json()
    if asset.get("validation_errors"):
        raise RuntimeError(f"Validation failed for {path.name}: {asset['validation_errors']}")
    return asset


def run_analysis(
    client: httpx.Client,
    base_url: str,
    *,
    mode: str,
    query: str,
    asset_ids: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/analyses",
        headers={"Idempotency-Key": f"jury-{mode}-{time.time_ns()}"},
        json={
            "query": query,
            "asset_ids": asset_ids,
            "requested_tasks": None,
            "parameters": {},
        },
    )
    response.raise_for_status()
    record = response.json()
    deadline = time.monotonic() + timeout_seconds
    while record["status"] not in TERMINAL_STATUSES:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{mode} exceeded {timeout_seconds:g} seconds")
        time.sleep(1.5)
        response = client.get(f"{base_url}/analyses/{record['id']}")
        response.raise_for_status()
        record = response.json()
    if record["status"] != "succeeded":
        raise RuntimeError(
            f"{mode} failed: {record.get('error_code')} {record.get('error_message')}"
        )
    result = record.get("result") or {}
    if not str(result.get("answer", "")).strip():
        raise RuntimeError(f"{mode} returned no answer")
    versions = result.get("provenance", {}).get("model_versions", {})
    if any(str(value).startswith("demo-simulator") for value in versions.values()):
        raise RuntimeError(f"{mode} used simulated inference instead of the configured backend")
    return record


def save_response(response: httpx.Response, path: Path) -> None:
    response.raise_for_status()
    path.write_bytes(response.content)


def export_run(
    client: httpx.Client,
    base_url: str,
    output: Path,
    record: dict[str, Any],
    assets: list[dict[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    analysis_id = record["id"]
    (output / "analysis.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    save_response(
        client.get(f"{base_url}/analyses/{analysis_id}/report"),
        output / "report.pdf",
    )
    for index, asset in enumerate(assets, start=1):
        asset_id = asset["id"]
        save_response(
            client.get(f"{base_url}/assets/{asset_id}/preview"),
            output / f"input-{index}-preview.jpg",
        )
        save_response(
            client.get(
                f"{base_url}/analyses/{analysis_id}/overlay",
                params={"asset_id": asset_id},
            ),
            output / f"input-{index}-overlay.jpg",
        )
    for evidence in record["result"].get("evidence", []):
        if evidence.get("type") != "mask":
            continue
        mask_id = evidence["id"]
        save_response(
            client.get(f"{base_url}/analyses/{analysis_id}/masks/{mask_id}"),
            output / f"{mask_id}-binary.png",
        )
        save_response(
            client.get(
                f"{base_url}/analyses/{analysis_id}/masks/{mask_id}",
                params={"colored": "true"},
            ),
            output / f"{mask_id}-colored.png",
        )


def copy_inputs(paths: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        shutil.copy2(path, output / f"source-{index}{path.suffix.lower()}")


def main() -> None:
    args = parse_args()
    all_paths = [args.single, args.time_a, args.time_b, args.optical, args.sar]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")

    base_url = args.base_url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        readiness = client.get(f"{base_url}/health/ready")
        readiness.raise_for_status()
        if readiness.json().get("status") != "ok":
            raise RuntimeError(f"Controller is not ready: {readiness.text[:1200]}")

        single_asset = upload(
            client,
            base_url,
            args.single,
            modality="optical",
            role="primary",
            source_dataset="SAR2Opt (MIT; optical display export)",
        )
        single = run_analysis(
            client,
            base_url,
            mode="single",
            query=(
                "Outline the visible vegetation and provide a detailed, evidence-bound scene "
                "assessment. Distinguish direct observations from hypotheses and state image "
                "limitations."
            ),
            asset_ids=[single_asset["id"]],
            timeout_seconds=args.timeout_seconds,
        )
        single_dir = args.output / "single"
        copy_inputs([args.single], single_dir)
        export_run(client, base_url, single_dir, single, [single_asset])
        print(f"PASS single: {single['id']}", flush=True)

        temporal_assets = [
            upload(
                client,
                base_url,
                args.time_a,
                modality="optical",
                role="time_a",
                registration_basis="pixel_grid",
                source_dataset="Google RSRCC (Apache-2.0; aligned display pair)",
            ),
            upload(
                client,
                base_url,
                args.time_b,
                modality="optical",
                role="time_b",
                registration_basis="pixel_grid",
                source_dataset="Google RSRCC (Apache-2.0; aligned display pair)",
            ),
        ]
        temporal = run_analysis(
            client,
            base_url,
            mode="temporal",
            query=(
                "Highlight visible building footprints in both dates, compare their candidate "
                "extent, and explain the observable changes in depth. Do not invent dates, "
                "causes, or geographic area."
            ),
            asset_ids=[item["id"] for item in temporal_assets],
            timeout_seconds=args.timeout_seconds,
        )
        temporal_dir = args.output / "temporal"
        copy_inputs([args.time_a, args.time_b], temporal_dir)
        export_run(client, base_url, temporal_dir, temporal, temporal_assets)
        print(f"PASS temporal: {temporal['id']}", flush=True)

        fusion_assets = [
            upload(
                client,
                base_url,
                args.optical,
                modality="optical",
                role="optical",
                registration_basis="pixel_grid",
                source_dataset="SAR2Opt (MIT; co-registered display pair)",
            ),
            upload(
                client,
                base_url,
                args.sar,
                modality="sar",
                role="sar",
                registration_basis="pixel_grid",
                source_dataset="SAR2Opt (MIT; co-registered display pair)",
            ),
        ]
        fusion = run_analysis(
            client,
            base_url,
            mode="fusion",
            query=(
                "Fuse the aligned optical and SAR display evidence. Identify consistent and "
                "disagreeing land-cover patterns, describe candidate built-up, vegetation, and "
                "low-return regions, and clearly state that the JPEG SAR is not calibrated."
            ),
            asset_ids=[item["id"] for item in fusion_assets],
            timeout_seconds=args.timeout_seconds,
        )
        fusion_dir = args.output / "fusion"
        copy_inputs([args.optical, args.sar], fusion_dir)
        export_run(client, base_url, fusion_dir, fusion, fusion_assets)
        print(f"PASS fusion: {fusion['id']}", flush=True)

        manifest = {
            "generated_at_unix": time.time(),
            "scope": (
                "Live integration evidence. It is not a benchmark accuracy, calibration, or "
                "event-causation claim."
            ),
            "base_url": base_url,
            "runs": {
                "single": single["id"],
                "temporal": temporal["id"],
                "fusion": fusion["id"],
            },
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"PASS jury bundle: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
