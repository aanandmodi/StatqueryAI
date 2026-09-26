"""Embedded by build-kaggle-pack.py: real inference only, never trains or changes a gate."""
import cv2
import time
from types import SimpleNamespace


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value

assert warm_start, "Attach your extracted notebook 02 checkpoint before Run All."
audit_dir = ROOT / "runtime-audit"
audit_dir.mkdir(parents=True, exist_ok=True)
quality_segmentation = trainer.model.float().eval()
quality_segmentation_processor = processor
SEGMENTATION_PREPROCESSING = quality_preprocessing_contract(old_root)
QUALITY_TILED_SEGMENTATION = False

# Reproduce the FULL validation protocol at its original 384px label grid first.
# This is a regression/reproduction run, not a newly untouched test split.
reproduced = trainer.evaluate()
recorded = old_manifest["metrics"]
metric_names = ["eval_mean_iou", "eval_iou_water", "eval_iou_forest", "eval_iou_agricultural"]
reproduction_deltas = {key: float(reproduced[key] - recorded[key]) for key in metric_names}
reproduction_passed = all(abs(delta) <= 0.005 for delta in reproduction_deltas.values())
print({"full_validation_reproduced": reproduction_passed, "metric_deltas": reproduction_deltas})

# A fixed seed chooses scenes BEFORE inference. Both modes see exactly the same examples,
# labels, precision and output grid. The only changed variable is the image resize.
positions = np.sort(np.random.default_rng(20260924).choice(
    len(validation_dataset), size=min(200, len(validation_dataset)), replace=False
))
matrices = {mode: np.zeros((7, 7), np.int64) for mode in ("legacy_native", "training_matched")}
records = []
palette = np.array([[122,132,146], [247,166,77], [245,232,112], [44,174,235],
                    [191,134,78], [57,166,93], [157,207,89]], np.uint8)
for number, position in enumerate(tqdm(positions, desc="Runtime preprocessing comparison")):
    item = validation_source[int(position)]
    rgb = item["image"].permute(1, 2, 0).byte().numpy()
    publisher_mask = item["mask"].numpy().astype(np.uint8)
    reference = np.where(publisher_mask == 0, 255, publisher_mask - 1).astype(np.uint8)
    image = Image.fromarray(rgb)
    source_path = Path(validation_source.files[int(position)]["image"])
    scene_id = f"{source_path.parent.parent.name}/{source_path.name}"
    row = {"scene_id": scene_id, "validation_position": int(position),
           "image_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "modes": {}}
    previews = [image]
    for mode in matrices:
        SEGMENTATION_PREPROCESSING = (quality_preprocessing_contract(old_root)
                                     if mode == "training_matched" else {"input_size": None})
        torch.cuda.synchronize()
        started = time.perf_counter()
        probabilities = quality_semantic_prediction(image)
        torch.cuda.synchronize()
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        elapsed = time.perf_counter() - started
        confusion = np.zeros((7, 7), np.int64)
        update_confusion(confusion, prediction, reference)
        matrices[mode] += confusion
        row["modes"][mode] = {"seconds": elapsed, "confusion": confusion.tolist(),
                               "metrics": metrics_from_confusion(confusion)}
        path = audit_dir / "predictions" / mode / scene_id
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(path.with_suffix(".png"))
        if number < 12:
            color = palette[prediction].copy()
            color[reference == 255] = rgb[reference == 255]
            previews.append(Image.blend(image, Image.fromarray(color), 0.45))
        del probabilities, prediction
    if number < 12:
        canvas = Image.new("RGB", (image.width * 3, image.height + 32), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for index, (preview, title) in enumerate(zip(previews, ["Source", "Old native input", "Training-matched input"])):
            canvas.paste(preview, (image.width * index, 32))
            draw.text((image.width * index + 8, 8), title, fill="black")
        canvas.save(audit_dir / f"preview_{number:02d}.png")
    records.append(row)
    # Save incrementally: a session interruption must not lose the completed predictions.
    with (audit_dir / "scene_predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), allow_nan=False) + "\n")

summary = {"purpose": "preprocessing ablation; does not promote a checkpoint or calibrate confidence",
           "weights_sha256": warm_start["weights_sha256"],
           "preprocessing": quality_preprocessing_contract(old_root),
           "comparison_scenes": len(records), "precision": "float32",
           "full_validation_reproduced": reproduction_passed, "reproduction_deltas": reproduction_deltas,
           "full_validation_metrics": reproduced,
           "native_label_grid_comparison": {mode: metrics_from_confusion(cm) for mode, cm in matrices.items()},
           "bootstrap": {}, "release_candidate_unchanged": old_manifest.get("release_candidate")}
# Paired SCENE bootstrap, never treat a million correlated pixels as a million samples.
rng = np.random.default_rng(20260924)
scene_cm = {mode: np.array([r["modes"][mode]["confusion"] for r in records]) for mode in matrices}
for metric in ("mean_iou", "iou_water", "iou_forest", "iou_agricultural"):
    deltas = []
    for _ in range(1000):
        selected = rng.integers(0, len(records), size=len(records))
        old = metrics_from_confusion(scene_cm["legacy_native"][selected].sum(axis=0))[metric]
        new = metrics_from_confusion(scene_cm["training_matched"][selected].sum(axis=0))[metric]
        if np.isfinite(new - old):
            deltas.append(new - old)
    summary["bootstrap"][metric] = {"paired_scene_delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None}
(audit_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
export_inference_zip(audit_dir, "/kaggle/working/00_runtime_preprocessing_evidence.zip")
print(json.dumps(summary, indent=2))
print("Preserve the ZIP. No training occurred, no weights changed, and no release gate was relaxed.")
if not reproduction_passed:
    raise RuntimeError("Recorded validation was not reproduced within 0.005. Review the saved evidence before training/serving.")
