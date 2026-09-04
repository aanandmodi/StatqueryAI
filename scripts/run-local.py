from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000/v1/health/ready"


def readiness_ok(body: object, status: int) -> bool:
    if not isinstance(body, dict):
        return False
    checks = body.get("checks", {})
    return (
        status == 200
        and body.get("status") in {"ready", "ok"}
        and isinstance(checks, dict)
        and all(value is True for value in checks.values())
    )


def wait_for(
    url: str,
    process: subprocess.Popen[bytes] | None,
    timeout: int,
    label: str,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    """Require healthy JSON, not an ngrok HTML page or HTTP-200/degraded."""
    deadline = time.monotonic() + timeout
    last_update = 0.0
    reason = "no response"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} exited with code {process.returncode}")
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read(65_536))
                if readiness_ok(body, response.status):
                    return
                reason = "JSON readiness reports unhealthy dependencies"
        except (urllib.error.URLError, TimeoutError, ValueError, AttributeError):
            reason = "connection failed or response was not readiness JSON"
        if time.monotonic() - last_update >= 20:
            print(f"Waiting for {label}: {reason}...", flush=True)
            last_update = time.monotonic()
        time.sleep(2)
    raise TimeoutError(f"{label} not ready after {timeout}s: {reason}. Check its terminal/notebook.")


def terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            if os.name == "nt":
                # Only PIDs created by this runner; include npm's node children.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
    for process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start SatQuery locally; never deploys a website.")
    parser.add_argument(
        "--mode", choices=("remote", "demo", "local"), default="remote",
        help="remote: Kaggle URL from .env (default); demo: no VLM; local: opt-in local GPU",
    )
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Run scripts/setup-local.ps1 to install controller dependencies first.") from exc
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.local", override=False)
    environment = os.environ.copy()
    defaults = {
        "SATQUERY_ENVIRONMENT": "development",
        "SATQUERY_PAIR_BACKEND": "local",
        "SATQUERY_MODEL_TIMEOUT_SECONDS": "300",
        "SATQUERY_ALLOWED_ORIGINS": '["http://localhost:3000"]',
        "SATQUERY_BACKEND_URL": "http://127.0.0.1:8000",
    }
    for name, value in defaults.items():
        environment.setdefault(name, value)

    environment["SATQUERY_MODEL_BACKEND"] = "demo" if args.mode == "demo" else "http"
    model_base_url = environment.get("SATQUERY_MODEL_SERVICE_URL", "").strip().rstrip("/")
    model_headers = {"ngrok-skip-browser-warning": "1"}
    if args.mode == "remote":
        token = environment.get("SATQUERY_MODEL_SERVICE_TOKEN", "").strip()
        if not model_base_url.startswith("https://") or "YOUR-" in model_base_url.upper():
            raise RuntimeError("Set the real HTTPS ngrok URL from notebook section 8 in root .env.")
        if len(token) < 32 or "REPLACE_" in token or "THE_SAME" in token:
            raise RuntimeError("Set the matching >=32-character Kaggle service secret in root .env.")
        model_headers["Authorization"] = f"Bearer {token}"
    elif args.mode == "local":
        model_base_url = "http://127.0.0.1:8080"
        environment["SATQUERY_MODEL_SERVICE_URL"] = model_base_url
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        environment.setdefault("SATQUERY_MODEL_MAX_NEW_TOKENS", "96")

    python_path = str(Path(sys.executable).resolve())
    npm = "npm.cmd" if os.name == "nt" else "npm"
    processes: list[subprocess.Popen[bytes]] = []

    def start(command: list[str]) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        processes.append(process)
        return process

    def stop_handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        if args.mode == "local":
            print("Starting the explicitly requested local GPU model...", flush=True)
            model = start([
                python_path, "-m", "uvicorn", "satquery_model_service.main:app",
                "--app-dir", "model_service", "--host", "127.0.0.1", "--port", "8080",
            ])
            wait_for(f"{model_base_url}/ready", model, 900, "local model service")
        elif args.mode == "remote":
            print("Checking the temporary Kaggle model service (no local model load)...", flush=True)
            wait_for(f"{model_base_url}/ready", None, 60, "Kaggle model service", headers=model_headers)
        else:
            print("DEMO MODE: single-image outputs are simulated; pair tools run on CPU.", flush=True)

        print("Starting the local controller on 127.0.0.1:8000...", flush=True)
        backend = start([
            python_path, "-m", "uvicorn", "app.main:app", "--app-dir", "backend",
            "--host", "127.0.0.1", "--port", "8000",
        ])
        wait_for(BACKEND_URL, backend, 60, "controller")
        print("Starting the local web app at http://localhost:3000 ...", flush=True)
        start([npm, "run", "dev"])
        print("Nothing deployed. Ctrl+C stops local processes; stop Kaggle separately.", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        failed = next(process for process in processes if process.poll() is not None)
        raise RuntimeError(f"A local service exited with code {failed.returncode}")
    except KeyboardInterrupt:
        print("Stopping local SatQuery services...", flush=True)
        return 0
    finally:
        terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
