# Evaluation, release gates and limitations

## Evidence policy

All numbers below come from preserved JSON in `docs/evaluation/submission/` or `upgrade/`. No random or hand-authored benchmark generator is included. Different tasks use different data and denominators; they cannot be averaged into one system score.

## Qwen3-VL LoRA

Pinned base revision `89644892e4d85e24eaac8bacfd4f463576704203`; adapter revision `ed12e59e0def9468bdf4a226789fc1b77c7900e7`. Held-out test, 200 mixed rows:

| Metric | Base | LoRA | Delta |
|---|---:|---:|---:|
| VQA exact match | 0.0000 | 0.6100 | +0.6100 |
| Grounding mean IoU | 0.0217 | 0.4227 | +0.4010 |
| Grounding accuracy at IoU ≥ 0.5 | 0.0200 | 0.4200 | +0.4000 |
| Caption token F1 | 0.3026 | 0.4014 | +0.0989 |

This proves improvement on the sampled protocol, not arbitrary imagery. Sequence confidence is not a correctness probability. The frozen-temperature diagnostic records ECE 0.5516 → 0.1304; automatic probability release remains false.

Source: `docs/evaluation/submission/qwen_test_summary.json` and `upgrade/05_qwen_test_evidence/outputs/test_scorecard.json`.

## Single-image segmentation

LoveDA official validation, 1,669 scenes; selected incumbent after R4:

| Class / aggregate | IoU | Dice |
|---|---:|---:|
| Background | 0.3489 | 0.5173 |
| Building | 0.4737 | 0.6429 |
| Road | 0.5079 | 0.6736 |
| Water | 0.5852 | 0.7383 |
| Barren | 0.1544 | 0.2675 |
| Forest | 0.3395 | 0.5069 |
| Agricultural | 0.5350 | 0.6971 |
| Mean | **0.4206** | **0.5777** |

Strict gate: mean IoU ≥ 0.45 **failed**; water ≥ 0.35 passed; forest ≥ 0.35 **failed**; agriculture ≥ 0.35 passed. R4 reached mean IoU 0.4302 but regressed protected classes, so incumbent retention worked. The UI labels these masks experimental.

Source: `docs/evaluation/submission/segmentation_R4_training_manifest.json`.

## Learned temporal change

| Split | QA examples | Unique pairs | Answer accuracy | Mask IoU | Mask Dice |
|---|---:|---:|---:|---:|---:|
| Validation | 16,441 | 400 | 0.7311 | 0.4842 | 0.6525 |
| Test | 39,686 | 968 | **0.7275** | **0.4961** | **0.6632** |

Mask metrics use unique pairs so repeated questions do not duplicate evidence. The target is generic semantic change, not a query-specific water/forest-loss boundary. Cloud, shadow, season and co-registration remain confounders.

Source: `docs/evaluation/submission/change_evaluation_summary.json`.

## Learned optical/SAR flood fusion

Sen1Floods11 test, 90 chips and 3,908,105 valid pixels:

| Input | Flood IoU | Dice | Precision | Recall | Mean IoU |
|---|---:|---:|---:|---:|---:|
| Fused S2 + S1 | **0.7103** | **0.8306** | 0.8577 | 0.8051 | 0.8327 |
| S2 only | 0.7099 | 0.8304 | 0.8539 | 0.8081 | 0.8324 |
| S1 only | 0.6056 | 0.7543 | 0.8271 | 0.6934 | 0.7724 |

The fused head passes the project flood gate, but the S2 ablation is nearly equal here. This supports a working multimodal architecture, not a claim that SAR gives a large universal improvement. It cannot separate permanent water from new inundation without pre-event evidence.

Source: `docs/evaluation/submission/fusion_evaluation_summary.json`.

## Confidence semantics

| Value | Meaning | Must not be called |
|---|---|---|
| `calibrated_probability` | frozen calibration on compatible held-out data | universal correctness on new sensors |
| `uncalibrated` | raw model statistic | confidence percentage |
| `evidence_quality` | deterministic completeness/compatibility | model accuracy |
| region evidence score | model support under its documented head | real-world truth probability |

Until a task-specific calibration gate passes, SatQuery caps or hides the global score and states the limitation.

## Remaining validation gap

The organizer’s complete evaluation specification and labelled Cartosat-2S/RISAT pairs were unavailable. Sensor contracts are implemented conservatively, but LoveDA/SECOND/Sentinel results do not validate those hidden sensors. Closing this requires a disjoint organizer-domain set, not more prose or a larger LLM.
