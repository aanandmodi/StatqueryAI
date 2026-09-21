# Recorded model evaluation

## Validation run received on 2026-09-08

The exact immutable revisions and aggregate values supplied from the completed Kaggle evaluation
notebook are preserved in [`validation-200-aggregate.json`](validation-200-aggregate.json). This is
a real base-versus-LoRA validation result, but it is not yet a complete release evidence bundle:
the raw per-example JSONL and split manifest have not been added to this repository.

| Metric | Base | LoRA | LoRA − base |
|---|---:|---:|---:|
| VQA exact match | 0.0200 | 0.5700 | +0.5500 |
| Grounding mean IoU | 0.0169 | 0.3807 | +0.3638 |
| Grounding accuracy at IoU ≥ 0.5 | 0.0200 | 0.4200 | +0.4000 |
| Caption token F1 | 0.2811 | 0.3652 | +0.0840 |

Allowed claim: “On this 200-row validation run, the pinned LoRA outperformed the pinned base on
all four reported aggregates.” Do not call this test-set performance, attach a confidence interval,
or describe sequence likelihood as correctness probability.

## Release gate

1. Add the notebook-exported `validation_predictions.jsonl` and split manifest.
2. Run `scripts/score-real-evaluation.py` locally to produce scene-cluster bootstrap intervals and
   the validation-only calibration diagnostic.
3. Review error slices and freeze the calibration artifact and abstention policy.
4. Run once on an untouched test split. Only then may the backend expose
   `score_kind="calibrated_probability"` with the frozen calibration version.

Until all four steps pass, the backend's uncalibrated score cap remains intentional.
