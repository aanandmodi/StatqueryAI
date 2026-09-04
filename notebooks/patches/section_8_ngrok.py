import threading
import time
from collections.abc import Mapping
from os import environ
from pathlib import Path

from pyngrok import ngrok


def detect_notebook_runtime(environ: Mapping[str, str], kaggle_working_exists: bool) -> str:
    # Kaggle builds on a Colab image: inherited COLAB_* variables or an imported google.colab
    # package are NOT proof that the live notebook is hosted by Colab.
    if environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip() and kaggle_working_exists:
        return "kaggle"
    if any(environ.get(name, "").strip() for name in (
        "COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION", "COLAB_JUPYTER_IP", "COLAB_GPU",
    )):
        return "colab"
    return "unknown"


runtime = detect_notebook_runtime(environ, Path("/kaggle/working").is_dir())
if runtime == "colab":
    raise RuntimeError(
        "Remote proxy tunnels are not allowed on managed Colab runtimes. "
        "This notebook will not create one. See https://research.google.com/colaboratory/faq.html"
    )
if runtime != "kaggle":
    raise RuntimeError(
        "Could not verify a Kaggle runtime. Run this notebook on Kaggle with GPU and Internet "
        "enabled. Do not manually spoof runtime markers or disable this check."
    )
if any(name not in globals() for name in ("server", "server_thread", "NGROK_AUTHTOKEN", "SERVICE_TOKEN")):
    raise RuntimeError("Missing session state. Run sections 1–7 first; no retraining is needed.")
server = globals()["server"]
server_thread = globals()["server_thread"]
NGROK_AUTHTOKEN = globals()["NGROK_AUTHTOKEN"]
if not server_thread.is_alive() or not server.started or server.should_exit:
    raise RuntimeError("The API server is not running. Rerun sections 6 and 7 before section 8.")
print("Runtime verified: Kaggle. Model service is running.")

ACTIVE_DEMO_MINUTES = 60  # Shorten this for your demo; no automatic session extension.
assert 1 <= ACTIVE_DEMO_MINUTES <= 60, "The attended demo window must be 1–60 minutes."
previous_url = globals().get("PUBLIC_MODEL_URL")
if previous_url:
    ngrok.disconnect(str(previous_url))
previous_timer = globals().get("demo_shutdown_timer")
if previous_timer is not None:
    previous_timer.cancel()
# A failed reconnect must not leave an old URL available to section 9.
globals().pop("PUBLIC_MODEL_URL", None)
globals().pop("DEMO_EXPIRES_AT", None)
ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(addr="http://127.0.0.1:8080", proto="http", bind_tls=True)
if not str(tunnel.public_url).startswith("https://"):
    ngrok.disconnect(str(tunnel.public_url))
    raise RuntimeError("ngrok did not return an HTTPS tunnel. Do not send the service token over HTTP.")
PUBLIC_MODEL_URL = str(tunnel.public_url)
DEMO_EXPIRES_AT = time.monotonic() + ACTIVE_DEMO_MINUTES * 60


def stop_demo(run_server, run_thread, tunnel_url: str) -> None:
    try:
        ngrok.disconnect(tunnel_url)
    except Exception as exc:
        print(f"Tunnel cleanup: {type(exc).__name__}; stop the Kaggle session to release resources.")
    finally:
        run_server.should_exit = True
        run_thread.join(timeout=20)
        print("SAFE STOP: demo window ended or was interrupted; model API shutdown requested.")


# Bind the current objects so a stale timer cannot close a later server/tunnel.
demo_shutdown_timer = threading.Timer(
    ACTIVE_DEMO_MINUTES * 60, stop_demo, args=(server, server_thread, PUBLIC_MODEL_URL)
)
demo_shutdown_timer.daemon = True
demo_shutdown_timer.start()
print("SATQUERY_MODEL_SERVICE_URL=", PUBLIC_MODEL_URL)
print("Do not print SATQUERY_MODEL_SERVICE_TOKEN; copy its existing secret value locally.")
print(f"Attended demo limit: {ACTIVE_DEMO_MINUTES} minutes. Stop the Kaggle session when finished.")
