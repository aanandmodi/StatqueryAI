# OPTIONAL: only after training/exporting v2 ChangeVQA or TerraMind artifacts.
# Attach each exported artifact folder as a private Kaggle input or copy it into /kaggle/working.
# Never populate these paths with random/untrained weights. Empty values leave Qwen unchanged.
PAIR_ARTIFACT_DIRS = {"change": "", "fusion": ""}

from pathlib import Path
from types import SimpleNamespace
import tempfile

from satquery_model_service.paired_adapters import ChangeAdapter, FusionAdapter
from satquery_model_service.contracts import InferencePayload as PairInferencePayload

assert not inference_lock.locked(), "Wait for the active request to finish."
pair_adapters = {}
for capability, raw_path in PAIR_ARTIFACT_DIRS.items():
    if not raw_path:
        continue
    root = Path(raw_path).resolve()
    if not any(root.is_relative_to(Path(prefix)) for prefix in ("/kaggle/input", "/kaggle/working")):
        raise ValueError("Pair artifact must be inside an explicitly attached Kaggle input/working folder")
    cls = ChangeAdapter if capability == "change" else FusionAdapter
    pair_adapters[capability] = cls(SimpleNamespace(artifact_dir=root, device="cuda:0"))
    print(f"{capability}: strict artifact load passed. Run a real paired HTTP smoke test before enabling it in the backend.")


async def paired_infer(task: str, payload: Annotated[str, Form()],
                       assets: Annotated[list[UploadFile], File()],
                       authorization: Annotated[str | None, Header()] = None):
    if task not in {"change_vqa", "optical_sar_fusion"}:
        return await quality_infer(task, payload, assets, authorization)
    authorize(authorization)
    capability = "change" if task == "change_vqa" else "fusion"
    if capability not in pair_adapters:
        raise HTTPException(409, f"No trained {capability} artifact loaded. No analytical fallback is disguised as learned inference.")
    try:
        contract = PairInferencePayload.model_validate_json(payload)
        if task != contract.step.task or len(assets) != 2 or len(contract.assets) != 2:
            raise ValueError("A pair task requires exactly two matching assets")
        if contract.step.asset_ids != [item.id for item in contract.assets]:
            raise ValueError("Asset order/IDs must match the declared step")
        async with inference_lock:
            with tempfile.TemporaryDirectory(prefix="satquery-pair-") as directory:
                paths = []
                for index, upload in enumerate(assets):
                    data = await upload.read(MAX_UPLOAD_BYTES + 1)
                    await upload.close()
                    if not data or len(data) > MAX_UPLOAD_BYTES:
                        raise ValueError("Invalid pair upload size")
                    if hashlib.sha256(data).hexdigest() != contract.assets[index].sha256:
                        raise ValueError("Uploaded bytes do not match the declared asset hash")
                    path = Path(directory) / f"asset-{index}.tif"
                    path.write_bytes(data)
                    paths.append(path)
                result = await asyncio.to_thread(pair_adapters[capability].infer, contract, paths)
                return result.model_dump()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


async def combined_ready():
    return {"status": "ready", "capability": "vlm", "capabilities": ["vlm", *pair_adapters],
            "model_version": MODEL_VERSION, "quality_pipeline": QUALITY_VERSION,
            "planner": "qwen-intent-v1" if "learned_plan" in globals() else None,
            "pair_artifacts": {key: value.version for key, value in pair_adapters.items()}}


for route in app.routes:
    if getattr(route, "path", None) == "/v1/infer/{task}":
        route.endpoint = route.dependant.call = paired_infer
    elif getattr(route, "path", None) == "/ready":
        route.endpoint = route.dependant.call = combined_ready
app.openapi_schema = None
print("Optional paired routes installed. Trained capabilities:", sorted(pair_adapters))
print("Qwen remains available. No paid endpoint, new tunnel, training job or automatic fallback was created.")
