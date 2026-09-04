"""Run one real compound case through the local web proxy. No tokens are printed."""
import argparse
import json
import time

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("before", help="Existing validated time_a asset ID")
parser.add_argument("after", help="Existing validated time_b asset ID")
args = parser.parse_args()
with httpx.Client(base_url="http://localhost:3000/api/satquery", timeout=35) as client:
    response = client.post("/analyses", json={"asset_ids": [args.before, args.after],
        "query": "Highlight the water bodies, compare before and after, and calculate the lost area."})
    response.raise_for_status()
    analysis_id = response.json()["id"]
    print("Analysis:", analysis_id, flush=True)
    last = None
    deadline = time.monotonic() + 650
    while time.monotonic() < deadline:
        response = client.get(f"/analyses/{analysis_id}")
        response.raise_for_status()
        case = response.json()
        if case["status"] != last:
            print("Status:", case["status"], flush=True)
            last = case["status"]
        if case["status"] in {"succeeded", "failed", "cancelled"}:
            result = case.get("result") or {}
            summary = {"id": analysis_id, "status": case["status"],
                       "error": case.get("error_message"), "plan": case.get("plan"),
                       "models": result.get("provenance", {}).get("step_models"),
                       "measurements": [fact for fact in result.get("facts", [])
                                        if fact.get("name") == "target_extent_change"],
                       "masks": len([item for item in result.get("evidence", []) if item["type"] == "mask"])}
            print(json.dumps(summary, indent=2), flush=True)
            if case["status"] != "succeeded":
                raise SystemExit(1)
            break
        time.sleep(2)
    else:
        raise TimeoutError("Case has not completed; inspect it in the local Casebook")
