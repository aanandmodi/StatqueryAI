from __future__ import annotations

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


def wait_for(
    url: str,
    process: subprocess.Popen[bytes] | None,
    timeout: int,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout
    last_update = 0.0
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"{label} exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() - last_update >= 20:
            print(f"Waiting for {label}...", flush=True)
            last_update = time.monotonic()
        time.sleep(2)
    raise TimeoutError(f"{label} did not become ready within {timeout} seconds")


def terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(ROOT / ".env.local", override=False)
    except ImportError:
        pass
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    defaults = {
            "HF_HOME": str(cache),
            "TOKENIZERS_PARALLELISM": "false",
            "SATQUERY_MODEL_CAPABILITY": "vlm",
            "SATQUERY_MODEL_BASE_MODEL": "Qwen/Qwen3-VL-2B-Instruct",
            "SATQUERY_MODEL_BASE_REVISION": "89644892e4d85e24eaac8bacfd4f463576704203",
            "SATQUERY_MODEL_ADAPTER_MODEL": "aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora",
            "SATQUERY_MODEL_ADAPTER_REVISION": "ed12e59e0def9468bdf4a226789fc1b77c7900e7",
            "SATQUERY_MODEL_FOUR_BIT": "true",
            "SATQUERY_MODEL_MAX_IMAGE_EDGE": "448",
            "SATQUERY_MODEL_MAX_PIXELS": str(448 * 448),
            "SATQUERY_MODEL_MAX_NEW_TOKENS": "96",
            "SATQUERY_ENVIRONMENT": "development",
            "SATQUERY_MODEL_BACKEND": "http",
            "SATQUERY_MODEL_SERVICE_URL": "http://127.0.0.1:8080",
            "SATQUERY_MODEL_TIMEOUT_SECONDS": "300",
            "SATQUERY_ALLOWED_ORIGINS": "http://localhost:3000",
            "SATQUERY_BACKEND_URL": "http://127.0.0.1:8000",
        }
    for name, value in defaults.items():
        environment.setdefault(name, value)
    model_base_url = environment["SATQUERY_MODEL_SERVICE_URL"].rstrip("/")
    local_model = model_base_url in {"http://127.0.0.1:8080", "http://localhost:8080"}
    python_path = str(Path(sys.executable).resolve())
    npm = "npm.cmd" if os.name == "nt" else "npm"
    processes: list[subprocess.Popen[bytes]] = []

    def stop_handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    try:
        if local_model:
            print("Starting the pinned Qwen3-VL 2B + SatQuery LoRA service...", flush=True)
            model = subprocess.Popen(
                [python_path, "-m", "uvicorn", "satquery_model_service.main:app", "--host", "127.0.0.1", "--port", "8080"],
                cwd=ROOT / "model_service",
                env=environment,
            )
            processes.append(model)
            wait_for(f"{model_base_url}/ready", model, 900, "model service")
        else:
            print(f"Using temporary remote model service: {model_base_url}", flush=True)
            wait_for(f"{model_base_url}/ready", None, 60, "Kaggle model service")

        print("Starting the SatQuery orchestration API...", flush=True)
        backend = subprocess.Popen(
            [python_path, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT / "backend",
            env=environment,
        )
        processes.append(backend)
        wait_for(BACKEND_URL, backend, 60, "backend")

        print("Starting the local web app at http://localhost:3000 ...", flush=True)
        frontend = subprocess.Popen([npm, "run", "dev"], cwd=ROOT, env=environment)
        processes.append(frontend)
        print("SatQuery website and controller are local. Press Ctrl+C to stop them.", flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(1)
        failed = next(process for process in processes if process.poll() is not None)
        raise RuntimeError(f"A local service exited with code {failed.returncode}")
    except KeyboardInterrupt:
        print("Stopping SatQuery local services...", flush=True)
        return 0
    finally:
        terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
