# SATQUERY QUALITY UPGRADE — paste this ENTIRE file into ONE Kaggle code cell.
# Run after section 6 has started, BEFORE section 8 opens the tunnel.
# Existing live session: stop sending requests, interrupt only the section 10 waiting cell,
# then run this cell. It does not restart training, create an endpoint, or extend the tunnel timer.
# Requires the existing notebook's model, processor, helpers and FastAPI schemas.
import base64
import json
from pathlib import Path
from contextlib import nullcontext

from transformers import (
    AutoImageProcessor,
    Sam2Model,
    Sam2Processor,
    SegformerForSemanticSegmentation,
)

QUALITY_VERSION = "satquery-quality-v6-scale-matched"
SAM_REPO = "facebook/sam2.1-hiera-tiny"
SAM_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SEGMENTATION_REPO = globals().get("SATQUERY_SEGMENTATION_PATH") or "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"
SEGMENTATION_REVISION = globals().get("SATQUERY_SEGMENTATION_SHA") or "5c74556c08bebb5f45f50b6f78f61a62c5d220c7"
SEGMENTATION_LOAD_REVISION = None if globals().get("SATQUERY_SEGMENTATION_PATH") else SEGMENTATION_REVISION
SEGMENTATION_RELEASE_STATUS = globals().get("SATQUERY_SEGMENTATION_RELEASE_STATUS", "transfer_baseline_unvalidated")
SEGMENTATION_RELEASE_WARNING = globals().get("SATQUERY_SEGMENTATION_RELEASE_WARNING", "")
QUALITY_IMAGE_EDGE = 1024
QUALITY_MAX_TOKENS = 768
QUALITY_MAX_TARGETS = 3
QUALITY_MAX_BOXES = 4
QUALITY_VERBOSE_NARRATIVE = True  # Jury/demo profile: favor a complete report over lower latency.
QUALITY_TILED_SEGMENTATION = False  # Experimental ablation; validate before promoting.


def quality_preprocessing_contract(root):
    """Recover the actual validation resize, not the processor's unused size default."""
    if not root:
        return {"mode": "published_processor", "input_size": None}
    report = json.loads((Path(root) / "training_manifest.json").read_text())
    size = report.get("config", {}).get("crop_size")
    if type(size) is not int or not 128 <= size <= 1024:
        raise ValueError("SegFormer export has no supported, recorded validation crop_size.")
    return {"mode": "training_matched_full_scene", "input_size": size,
            "image_resize": "opencv_linear", "logit_resize": "bilinear_align_corners_false",
            "source": "training_manifest.json:config.crop_size"}


SEGMENTATION_PREPROCESSING = quality_preprocessing_contract(globals().get("SATQUERY_SEGMENTATION_PATH"))

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

# LoveDA supplies whole-scene semantic classes. This fixes the architectural failure where SAM
# precisely traced a semantically wrong Qwen box. The checkpoint is still a transfer baseline,
# not proof of accuracy on India, ISRO sensors, Sentinel-2 composites or arbitrary resolutions.
if globals().get("quality_segmentation_revision") != SEGMENTATION_REVISION:
    quality_segmentation_processor = AutoImageProcessor.from_pretrained(
        SEGMENTATION_REPO,
        revision=SEGMENTATION_LOAD_REVISION,
        trust_remote_code=False,
    )
    quality_segmentation = SegformerForSemanticSegmentation.from_pretrained(
        SEGMENTATION_REPO,
        revision=SEGMENTATION_LOAD_REVISION,
        trust_remote_code=False,
        # This pinned transfer checkpoint publishes pytorch_model.bin, not safetensors.
        # The exact immutable revision is mandatory; our own trained replacement exports
        # safetensors and should supersede this experimental baseline after evaluation.
        use_safetensors=bool(globals().get("SATQUERY_SEGMENTATION_PATH")),
        torch_dtype=torch.float32 if globals().get("SATQUERY_SEGMENTATION_PATH") else torch.float16,
    ).to("cuda:0").eval()
    quality_segmentation_revision = SEGMENTATION_REVISION
    if not all(torch.isfinite(value).all() for value in quality_segmentation.state_dict().values()):
        raise FloatingPointError("Segmentation checkpoint has non-finite weights; refusing to serve it.")


SEMANTIC_TARGETS = {
    "water": {"water"},
    "river": {"water"},
    "reservoir": {"water"},
    "lake": {"water"},
    "vegetation": {"forest", "agricultural"},
    "forest": {"forest"},
    "cropland": {"agricultural"},
    "agriculture": {"agricultural"},
    "agricultural": {"agricultural"},
    "building": {"building"},
    "buildings": {"building"},
    "built-up": {"building"},
    "urban": {"building"},
    "road": {"road"},
    "roads": {"road"},
    "barren": {"barren"},
    "bare land": {"barren"},
}


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


@torch.inference_mode()
def quality_semantic_prediction(image):
    """Reuse scene probabilities across targets; optional bounded overlap tiles."""
    tiled = globals().get("QUALITY_TILED_SEGMENTATION", False)
    contract = globals().get("SEGMENTATION_PREPROCESSING", {"input_size": None})
    if tiled and contract.get("input_size"):
        raise ValueError("Tiled inference is not validated for this full-scene-trained checkpoint.")
    edge, stride = 512, 448
    def starts(length):
        return sorted(set([*range(0, max(1, length - edge + 1), stride), max(0, length - edge)]))
    crops = [(0, 0, image.width, image.height)] if not tiled else [
        (x, y, min(x + edge, image.width), min(y + edge, image.height))
        for y in starts(image.height) for x in starts(image.width)
    ]
    total = np.zeros((int(quality_segmentation.config.num_labels), image.height, image.width), np.float32)
    counts = np.zeros((image.height, image.width), np.float32)
    for left, top, right, bottom in crops:
        crop = image.crop((left, top, right, bottom))
        if contract.get("input_size"):
            # Match Albumentations A.Resize's default cv2.INTER_LINEAR exactly.
            import cv2
            size = contract["input_size"]
            crop = cv2.resize(np.asarray(crop), (size, size), interpolation=cv2.INTER_LINEAR)
        inputs = quality_segmentation_processor(images=crop, return_tensors="pt")
        values = inputs["pixel_values"].to(quality_segmentation.device, dtype=quality_segmentation.dtype)
        logits = quality_segmentation(pixel_values=values).logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite semantic logits")
        probabilities = torch.nn.functional.interpolate(
            logits, size=(bottom - top, right - left), mode="bilinear", align_corners=False
        )[0].softmax(dim=0).cpu().numpy()
        total[:, top:bottom, left:right] += probabilities
        counts[top:bottom, left:right] += 1
    if not (counts > 0).all():
        raise ValueError("Uncovered segmentation grid")
    return total / counts[None]


@torch.inference_mode()
def quality_semantic_mask(image, target, valid, prediction_data=None):
    """Return a whole-scene LoveDA class mask; confidence is diagnostic, not calibrated."""
    requested_labels = SEMANTIC_TARGETS.get(str(target).strip().lower())
    if not requested_labels:
        return None
    probabilities = prediction_data if prediction_data is not None else quality_semantic_prediction(image)
    prediction = probabilities.argmax(axis=0)
    id2label = {
        int(class_id): str(label).strip().lower()
        for class_id, label in quality_segmentation.config.id2label.items()
    }
    selected_ids = [
        class_id for class_id, label in id2label.items() if label in requested_labels
    ]
    if not selected_ids:
        raise ValueError(f"The semantic checkpoint has no class mapping for {target}")
    mask_tensor = np.zeros_like(prediction, dtype=bool)
    for class_id in selected_ids:
        mask_tensor |= prediction == class_id
    mask = mask_tensor & valid
    selected_probability = probabilities[selected_ids].sum(axis=0)
    mean_probability = float(selected_probability[mask].mean()) if mask.any() else 0.0
    return mask, mean_probability, [id2label[class_id] for class_id in selected_ids]


def quality_guard_narrative(text):
    """Remove visually unsupported physical-condition claims if the VLM ignores its prompt."""
    risky = re.compile(
        r"\b(?:shallow|water depth|depth|slow[- ]moving|flow velocity|consistent flow|"
        r"water flow|current speed|erosion|sediment transport|vegetation health|healthy|"
        r"degradation|well[- ]maintained|paved|unpaved|rainfall|historical weather)\b",
        flags=re.IGNORECASE,
    )
    explicit_limit = re.compile(
        r"\b(?:cannot|can't|not (?:clear|resolved|enough|possible)|unclear|unknown|"
        r"requires? verification|would require)\b",
        flags=re.IGNORECASE,
    )
    kept = []
    removed = []
    for line in str(text or "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            pieces = [line]
        else:
            pieces = re.split(r"(?<=[.!?])\s+", line.strip())
        accepted = []
        for sentence in pieces:
            if risky.search(sentence) and not explicit_limit.search(sentence):
                removed.append(sentence.strip())
            else:
                accepted.append(sentence.strip())
        if accepted:
            kept.append(" ".join(accepted))
        elif not line.strip():
            kept.append("")
    guarded = "\n".join(kept).strip()
    if removed:
        guarded += (
            "\n\n## Guarded attributes\n\n"
            "Depth, flow/velocity, road-surface material, vegetation health and weather history "
            "cannot be established from this display image; unsupported assertions about those "
            "attributes were omitted from the main report."
        )
    return guarded, removed


def quality_mask_only_request(task, contract):
    """Skip language generation only for an explicitly simple, supported overlay request."""
    if globals().get("QUALITY_VERBOSE_NARRATIVE", True):
        return False
    targets = contract.step.permitted_params.get("targets", [])
    return (task == "grounding" and contract.assets[0].modality != "sar"
            and bool(targets) and len(targets) <= QUALITY_MAX_TARGETS
            and all(t in SEMANTIC_TARGETS for t in targets)
            and bool(re.fullmatch(
                r"(?:please\s+)?(?:show|mark|outline|highlight|segment|map)(?:\s+me)?\s+"
                r"(?:(?:all|the)\s+)*(?:water(?:\s+bodies)?|vegetation|forest|buildings?|roads?|cropland)"
                r"(?:\s+(?:in|on)\s+(?:this|the)\s+(?:image|scene|region|area))?[.!?]*",
                contract.query.strip(), flags=re.IGNORECASE)))


def quality_analyze(data, task, contract):
    import time
    started = time.perf_counter()
    image, valid, info = quality_decode(data)
    asset_id = contract.assets[0].id
    context = contract.context.model_dump(mode="json") if contract.context else None
    is_sar = contract.assets[0].modality == "sar"
    mask_only = globals().get("quality_mask_only_request", lambda *_: False)(task, contract)
    observation = ("Standalone SAR language interpretation is not validated by this optical VLM. "
                   "No water/land-cover claim is asserted; use the sensor-qualified S1/S2 pair specialist.") if is_sar else ("" if mask_only else quality_generate(
        image, contract.query + "\nGive a brief observation from visible pixels only.", 128, use_adapter=True,
    ))
    verbose = globals().get("QUALITY_VERBOSE_NARRATIVE", False) and not is_sar and not mask_only
    narrative = quality_generate(image,
        "You are writing an evidence-conscious remote-sensing report. Answer the user's actual question "
        "in 350–550 words when the pixels support that depth. Use six clear headings: Direct answer; Water and drainage; "
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
        + context_text(context), min(512, QUALITY_MAX_TOKENS),
    ) if verbose else observation
    narrative, removed_narrative_claims = quality_guard_narrative(narrative)
    warnings = [
        ("Mask-only fast path: semantic model ran; no Qwen text generation or SAM proposal ran." if mask_only else
         "Optional narrative uses the base instruction model; it is not verified evidence." if verbose else
         "Fast mode uses one adapted observation. Report measurements come from masks/metadata, not longer invented prose."),
        "SAM 2 only refines proposed regions. A water label is inherited from the Qwen proposal, not "
        "independently verified by SAM. Masks are candidates and may miss water or include non-water.",
        "No pixel accuracy or IoU on this scene is known. Thin features and boundaries may be lost on the bounded analysis grid.",
        "The LoveDA SegFormer is a whole-scene remote-sensing transfer baseline. Its class "
        "probabilities are uncalibrated and its geographic/resolution transfer to this image "
        "has not been established.",
    ]
    if SEGMENTATION_RELEASE_WARNING:
        warnings.insert(0, SEGMENTATION_RELEASE_WARNING)
    if not info["declared_rgb"]:
        warnings.append("RGB band mapping was not declared; the first three bands form an assumed display preview. "
                        "Verify their order before trusting color-based interpretation or target proposals.")
    evidence = []
    mask_diagnostics = []
    shared_semantic_prediction = None
    if task == "grounding" and contract.assets[0].modality != "sar":
        targets = contract.step.permitted_params.get("targets", [])
        if not targets:
            warnings.append("No supported target class was identified. Ask to outline water, buildings, roads, forest or cropland.")
        for target in targets[:QUALITY_MAX_TARGETS]:
            if target in globals().get("SEMANTIC_TARGETS", {}) and shared_semantic_prediction is None:
                shared_semantic_prediction = quality_semantic_prediction(image)
            semantic = quality_semantic_mask(image, target, valid, shared_semantic_prediction)
            if semantic is not None:
                mask, semantic_score, semantic_labels = semantic
                if not mask.any():
                    warnings.append(
                        f"The semantic baseline selected no {target} pixels; no mask was invented."
                    )
                    continue
                evidence.append({
                    "id": f"ev_semantic_{len(evidence) + 1}", "type": "mask",
                    "label": f"{target} candidate (LoveDA SegFormer" + ("; EXPERIMENTAL" if SEGMENTATION_RELEASE_WARNING else "") + ")",
                    "score": min(0.59, semantic_score),
                    "coordinate_space": "pixel", "asset_id": asset_id, "artifact_url": None,
                    "geometry": {"encoding": "png-base64", "data": quality_png(mask),
                                 "width": image.width, "height": image.height,
                                 "method": "tiled LoveDA SegFormer experimental ablation" if globals().get("QUALITY_TILED_SEGMENTATION", False) else "whole-scene LoveDA SegFormer semantic classes",
                                 "status": "candidate", "target": target,
                                 "release_status": SEGMENTATION_RELEASE_STATUS,
                                 "semantic_classes": semantic_labels},
                })
                mask_diagnostics.append({
                    "target": target,
                    "method": "LoveDA SegFormer whole-scene semantic mask",
                    "semantic_classes": semantic_labels,
                    "selected_pixel_fraction": float(mask.sum() / max(1, valid.sum())),
                    "mean_selected_probability_uncalibrated": semantic_score,
                })
                continue
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
    if mask_only:
        rows = [f"- {d['target']}: {d['selected_pixel_fraction']:.2%} of valid analysis pixels selected."
                for d in mask_diagnostics if "selected_pixel_fraction" in d]
        narrative = ("## Requested overlay\n\n" + ("\n".join(rows) if rows else
                     "The model selected no pixels for the requested target. This does not prove absence.")
                     + "\n\nThese are candidate semantic masks, not verified boundaries or physical-area measurements. "
                     "Thin features may be lost. Review the original image and held-out labelled examples.")
    return {
        "task": task, "text": narrative or observation or "No visual interpretation was returned.",
        "facts": [
            {"name": "scope_abstention" if is_sar else ("measured_mask_report" if mask_only else "short_adapter_observation"), "value": narrative if mask_only else observation, "model": "mask-measurements" if mask_only else ("scope-policy" if is_sar else MODEL_VERSION)},
            {"name": "narrative_model", "value": "mask-measurements" if mask_only else ("scope-policy" if is_sar else (f"{BASE_MODEL}@{BASE_REVISION}" if verbose else MODEL_VERSION)), "adapter_enabled": not verbose and not is_sar and not mask_only},
            {"name": "segmentation_model", "value": f"{SAM_REPO}@{SAM_REVISION}"},
            {"name": "semantic_segmentation_model",
             "value": f"{SEGMENTATION_REPO}@{SEGMENTATION_REVISION}"},
            {"name": "semantic_segmentation_release_status", "value": SEGMENTATION_RELEASE_STATUS},
            {"name": "segmentation_preprocessing", "value": globals().get("SEGMENTATION_PREPROCESSING", {})},
            {"name": "analysis_grid", "value": info},
            {"name": "mask_diagnostics", "value": mask_diagnostics},
            {"name": "guarded_narrative_claim_count", "value": len(removed_narrative_claims)},
            {"name": "quality_pipeline", "value": QUALITY_VERSION},
            {"name": "runtime_profile", "value": {"verbose_narrative": verbose,
                "mask_only_fastpath": mask_only,
                "tiled_segmentation": globals().get("QUALITY_TILED_SEGMENTATION", False),
                "total_seconds": round(time.perf_counter() - started, 3)}},
        ],
        "evidence": evidence, "raw_score": 0.0 if is_sar else 0.5, "score_kind": "uncalibrated",
        "model_version": f"{QUALITY_VERSION};adapter={MODEL_VERSION};narrative={BASE_MODEL}@{BASE_REVISION[:12]};semantic={SEGMENTATION_REVISION[:12]};sam={SAM_REVISION[:12]}" + (":abstained" if is_sar else ""),
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
            "quality_pipeline": QUALITY_VERSION,
            "semantic_segmentation": f"{SEGMENTATION_REPO}@{SEGMENTATION_REVISION[:12]}",
            "segmentation": f"{SAM_REPO}@{SAM_REVISION[:12]}"}


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
quality_probe_semantic = quality_semantic_mask(sample, "vegetation", np.ones(
    (sample.height, sample.width), dtype=bool
))
assert quality_probe_semantic is not None and quality_probe_semantic[0].shape == (
    sample.height, sample.width
)
print("PASS: LoveDA SegFormer returned a source-aligned whole-scene class mask (not an accuracy test).")

# Replace only these existing routes; preserve their auth/header/form dependencies and listener.
# This supports BOTH an already-running session and the updated notebook's section 6b.
for quality_route in app.routes:
    if getattr(quality_route, "path", None) == "/v1/infer/{task}":
        quality_route.endpoint = quality_infer
        quality_route.dependant.call = quality_infer
    elif getattr(quality_route, "path", None) == "/ready":
        quality_route.endpoint = quality_ready
        quality_route.dependant.call = quality_ready

print("Quality v5 installed: lean adapted observation, class-aware masks and optional ablations. No paid service created.")
print("Now run section 7, then sections 8 and 9 if the tunnel is not already live. Keep section 10 running for the attended demo.")
print("Test in the website: 'Outline the visible water bodies and give a detailed report of their spatial pattern and limitations.'")
print("Loading these models is not evidence of mask accuracy. Inspect real satellite cases and evaluate labelled masks.")
