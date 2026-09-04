# SATQUERY QUALITY UPGRADE — paste this ENTIRE file into ONE Kaggle code cell.
# Run after section 6 has started, BEFORE section 8 opens the tunnel.
# Existing live session: stop sending requests, interrupt only the section 10 waiting cell,
# then run this cell. It does not restart training, create an endpoint, or extend the tunnel timer.
# Requires the existing notebook's model, processor, helpers and FastAPI schemas.
import base64
from contextlib import nullcontext

from transformers import Sam2Model, Sam2Processor

QUALITY_VERSION = "satquery-quality-v3"
SAM_REPO = "facebook/sam2.1-hiera-tiny"
SAM_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
QUALITY_IMAGE_EDGE = 1024
QUALITY_MAX_TOKENS = 768
QUALITY_MAX_TARGETS = 3
QUALITY_MAX_BOXES = 4

assert "model" in globals() and "app" in globals(), "Run notebook sections 0–6 first."
assert not inference_lock.locked(), "Wait for the current analysis to finish before applying this cell."

# Separate processor configuration for the report. The adapter smoke-test remains unchanged.
quality_processor = AutoProcessor.from_pretrained(
    ADAPTER_REPO, revision=ADAPTER_REVISION, trust_remote_code=False,
    min_pixels=256 * 28 * 28, max_pixels=768 * 768,
)
# Small, separate segmentation model on the SAME free Kaggle GPU. No endpoint is provisioned.
if globals().get("quality_sam_revision") != SAM_REVISION:
    quality_sam_processor = Sam2Processor.from_pretrained(
        SAM_REPO, revision=SAM_REVISION, trust_remote_code=False,
    )
    quality_sam = Sam2Model.from_pretrained(
        SAM_REPO, revision=SAM_REVISION, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32,
    ).to("cuda:0").eval()
    quality_sam_revision = SAM_REVISION


def quality_decode(data):
    """Preserve source aspect ratio and no-data at a bounded 1024px grid, not the old 448px."""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is empty or exceeds 50 MiB")
    with MemoryFile(data) as memory, memory.open() as source:
        scale = min(1.0, QUALITY_IMAGE_EDGE / max(source.width, source.height))
        height = max(1, round(source.height * scale))
        width = max(1, round(source.width * scale))
        indexes = rgb_band_indexes(source)
        raw = source.read(indexes, out_shape=(3, height, width), masked=True,
                          resampling=Resampling.bilinear).astype(np.float32)
        raw = np.ma.filled(raw, np.nan)
        valid = np.isfinite(raw).all(axis=0)
        valid &= source.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
        if all(source.dtypes[index - 1] == "uint8" for index in indexes):
            rgb = np.moveaxis(np.clip(np.nan_to_num(raw), 0, 255).astype(np.uint8), 0, -1)
        else:
            rgb = np.stack([scale_band(band) for band in raw], axis=-1)
        rgb[~valid] = 0
        declared_rgb = semantic_indexes(source, ["red", "green", "blue"]) is not None
        info = {"width": source.width, "height": source.height, "band_indexes": indexes,
                "declared_rgb": bool(declared_rgb), "mask_grid": [width, height]}
    return Image.fromarray(rgb), valid, info


@torch.inference_mode()
def quality_generate(image, prompt, max_new_tokens=768, *, use_adapter=False):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image, "max_pixels": 768 * 768},
        {"type": "text", "text": prompt},
    ]}]
    rendered = quality_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    batch = quality_processor(text=[rendered], images=images, videos=videos, return_tensors="pt")
    batch = {name: value.to(model.device) for name, value in batch.items()}
    # The fine-tuned adapter remains the observation specialist. Base instruct supplies the
    # narrative/box proposals because the current short-answer SFT does not train those outputs.
    # All calls are serialized by inference_lock, so adapter toggling cannot race another query.
    with nullcontext() if use_adapter else model.disable_adapter():
        output = model.generate(
            **batch, max_new_tokens=max(1, min(int(max_new_tokens), QUALITY_MAX_TOKENS)),
            do_sample=False, temperature=1.0, top_p=1.0, top_k=50, use_cache=True,
        )
    new_ids = output[:, batch["input_ids"].shape[1]:]
    return quality_processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def quality_png(mask):
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


@torch.inference_mode()
def quality_segment(image, boxes, valid):
    """SAM refines supplied boxes; it DOES NOT independently identify water/semantic classes."""
    union = np.zeros((image.height, image.width), dtype=bool)
    scores = []
    # One prompt at a time bounds peak GPU memory and preserves all holes (no box filling).
    for box in boxes[:QUALITY_MAX_BOXES]:
        geo = box["geometry"]
        coordinates = [geo["x"] * image.width, geo["y"] * image.height,
                       (geo["x"] + geo["width"]) * image.width,
                       (geo["y"] + geo["height"]) * image.height]
        inputs = quality_sam_processor(images=image, input_boxes=[[coordinates]], return_tensors="pt").to(quality_sam.device)
        outputs = quality_sam(**inputs, multimask_output=False)
        masks = quality_sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        )[0]
        candidate = masks[0, 0].numpy().astype(bool)
        if candidate.shape != union.shape:
            raise ValueError("SAM output grid does not match the source preview")
        union |= candidate & valid
        scores.append(float(outputs.iou_scores.flatten()[0].float().cpu().item()))
    return union, scores


def quality_analyze(data, task, contract):
    image, valid, info = quality_decode(data)
    asset_id = contract.assets[0].id
    context = contract.context.model_dump(mode="json") if contract.context else None
    observation = quality_generate(
        image, contract.query + "\nGive a brief observation from visible pixels only.", 128, use_adapter=True,
    )
    narrative = quality_generate(image,
        "You are writing an evidence-conscious remote-sensing report. Answer the user's actual question "
        "in 250–400 words when supported. Use six concise headings: Direct answer; Water and drainage; "
        "Vegetation and bare surfaces; Built features and access; Visibility and ambiguity; "
        "Verification priorities. Cover each category briefly; explicitly say unclear or not resolved "
        "when evidence is insufficient. Describe image-relative shapes, distribution, texture and "
        "relationships only where visible. Do not give hidden reasoning. Do not invent "
        "counts, percentages, areas, species, place names, dates, water depth, flow velocity, "
        "soil types, pollution or change "
        "from one image. Visible cloud/haze is not rainfall or historical weather. Do not infer "
        "event causes, casualties, damage or land ownership. Distinguish observations from "
        "hypotheses and identify evidence needed to test each hypothesis. Do not pad or repeat. "
        "The image may be a stretched band preview, not calibrated true color. "
        f"\nUser question: {contract.query}\n"
        + context_text(context), QUALITY_MAX_TOKENS,
    )
    warnings = [
        "Narrative and target proposals use the base Qwen3-VL instruction model with the adapter temporarily disabled; "
        "the released adapter supplies the short observation. Neither is a calibrated correctness estimate.",
        "SAM 2 only refines proposed regions. A water label is inherited from the Qwen proposal, not "
        "independently verified by SAM. Masks are candidates and may miss water or include non-water.",
        "No pixel accuracy or IoU on this scene is known. Thin features and boundaries may be lost on the bounded analysis grid.",
    ]
    if not info["declared_rgb"]:
        warnings.append("RGB band mapping was not declared; the first three bands form an assumed display preview. "
                        "Verify their order before trusting color-based interpretation or target proposals.")
    evidence = []
    mask_diagnostics = []
    if task == "grounding" and contract.assets[0].modality != "sar":
        targets = contract.step.permitted_params.get("targets", [])
        if not targets:
            warnings.append("No supported target class was identified. Ask to outline water, buildings, roads, forest or cropland.")
        for target in targets[:QUALITY_MAX_TARGETS]:
            proposal = quality_generate(image,
                f"Locate only visible {target} regions in this remote-sensing image. "
                f"Return up to {QUALITY_MAX_BOXES} tight boxes, separately for disconnected visible regions, "
                "using coordinates normalized to 0..1000. Exact format: <box>(x1,y1),(x2,y2)</box>. "
                "Return NONE if not confidently visible. Do not return a full-image box to mean unknown.", 256,
            )
            boxes = parse_boxes(proposal, asset_id)[:QUALITY_MAX_BOXES]
            # A whole-image guess is not an acceptable fallback spatial explanation.
            boxes = [box for box in boxes if box["geometry"]["width"] * box["geometry"]["height"] < .95]
            if not boxes:
                warnings.append(f"No bounded {target} proposal was obtained; no mask was invented.")
                continue
            try:
                mask, scores = quality_segment(image, boxes, valid)
                evidence.append({
                    "id": f"ev_sam_{len(evidence) + 1}", "type": "mask",
                    "label": f"{target} candidate (Qwen + SAM 2)", "score": 0.5,
                    "coordinate_space": "pixel", "asset_id": asset_id, "artifact_url": None,
                    "geometry": {"encoding": "png-base64", "data": quality_png(mask),
                                 "width": image.width, "height": image.height,
                                 "method": "Qwen base box proposals + SAM 2.1 tiny masks",
                                 "status": "candidate", "target": target},
                })
                mask_diagnostics.append({"target": target, "proposal_count": len(boxes),
                                         "proposal_boxes_normalized": [box["geometry"] for box in boxes],
                                         "sam_predicted_iou_uncalibrated": scores})
            except Exception as exc:
                warnings.append(f"Segmentation unavailable for {target}: {type(exc).__name__}. "
                                "The text is available, but no substitute rectangle or mask was fabricated.")
                torch.cuda.empty_cache()
    elif task == "grounding":
        warnings.append("RGB SAM segmentation is disabled for SAR. Use a validated SAR specialist.")
    return {
        "task": task, "text": narrative or observation or "No visual interpretation was returned.",
        "facts": [
            {"name": "short_adapter_observation", "value": observation, "model": MODEL_VERSION},
            {"name": "narrative_model", "value": f"{BASE_MODEL}@{BASE_REVISION}", "adapter_enabled": False},
            {"name": "segmentation_model", "value": f"{SAM_REPO}@{SAM_REVISION}"},
            {"name": "analysis_grid", "value": info},
            {"name": "mask_diagnostics", "value": mask_diagnostics},
            {"name": "quality_pipeline", "value": QUALITY_VERSION},
        ],
        "evidence": evidence, "raw_score": 0.5, "score_kind": "uncalibrated",
        "model_version": f"{QUALITY_VERSION};adapter={MODEL_VERSION};narrative={BASE_MODEL}@{BASE_REVISION[:12]};sam={SAM_REVISION[:12]}",
        "warnings": warnings,
    }


async def quality_infer(
    task: str,
    payload: Annotated[str, Form()],
    assets: Annotated[list[UploadFile], File()],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    authorize(authorization)
    try:
        contract = InferencePayload.model_validate_json(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid inference contract") from exc
    if task != contract.step.task or task not in SUPPORTED_TASKS:
        raise HTTPException(status_code=409, detail="Unsupported or mismatched task")
    if len(assets) != 1 or len(contract.assets) != 1 or contract.step.asset_ids != [contract.assets[0].id]:
        raise HTTPException(status_code=422, detail="Exactly one matching image asset is required")
    data = await assets[0].read(MAX_UPLOAD_BYTES + 1)
    await assets[0].close()
    async with inference_lock:
        try:
            return await asyncio.to_thread(quality_analyze, data, task, contract)
        except (ValueError, rasterio.errors.RasterioError) as exc:
            raise HTTPException(status_code=422, detail="Unreadable TIFF or invalid image grid") from exc


async def quality_ready():
    return {"status": "ready", "capability": "vlm", "model_version": MODEL_VERSION,
            "quality_pipeline": QUALITY_VERSION, "segmentation": f"{SAM_REPO}@{SAM_REVISION[:12]}"}


# Execute a real SAM forward pass before switching the HTTP handler. Synthetic data validates
# tensor dimensions and memory only; it does not establish water recognition or boundary accuracy.
quality_probe_mask, quality_probe_scores = quality_segment(
    sample,
    [{"geometry": {"x": 0.2, "y": 0.2, "width": 0.5, "height": 0.5}}],
    np.ones((sample.height, sample.width), dtype=bool),
)
assert quality_probe_mask.shape == (sample.height, sample.width)
assert quality_probe_scores and np.isfinite(quality_probe_scores).all()
print("PASS: SAM 2 executed a GPU forward pass and returned a source-aligned mask (not an accuracy test).")

# Replace only these existing routes; preserve their auth/header/form dependencies and listener.
# This supports BOTH an already-running session and the updated notebook's section 6b.
for quality_route in app.routes:
    if getattr(quality_route, "path", None) == "/v1/infer/{task}":
        quality_route.endpoint = quality_infer
        quality_route.dependant.call = quality_infer
    elif getattr(quality_route, "path", None) == "/ready":
        quality_route.endpoint = quality_ready
        quality_route.dependant.call = quality_ready

print("Quality v3 installed: evidence-category reports + SAM 2 candidate masks. No paid service created.")
print("Now run section 7, then sections 8 and 9 if the tunnel is not already live. Keep section 10 running for the attended demo.")
print("Test in the website: 'Outline the visible water bodies and give a detailed report of their spatial pattern and limitations.'")
print("Loading these models is not evidence of mask accuracy. Inspect real satellite cases and evaluate labelled masks.")
