# Verification and release gates

## Test pyramid

```mermaid
flowchart TB
    E2E[Real model end-to-end smoke] --> I[API and gateway integration tests]
    I --> U[Unit tests: validation, routing, parsing, metrics, artifacts]
    U --> S[Static checks: Ruff, TypeScript lint, production build]
```

Passing unit tests does not prove that the model loads. Local acceptance requires the real-model
smoke below.

## Automated suite

```powershell
$python = "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe"
Set-Location D:\Projects\Sih-2026

& $python -m pytest backend\tests -q
& $python -m pytest ml\tests -q
& $python -m ruff check backend ml model_service scripts
npm run lint
npm run build
```

| Suite | What it proves | What it does not prove |
|---|---|---|
| Backend API | Upload, persistence, jobs, reports, overlays, errors | Real CUDA generation |
| Pair specialists | Synthetic co-registered change and one-band SAR inputs | Semantic accuracy on CDVQA or ISRO/SAC |
| ML utilities | Splits, preprocessing, metrics, manifests | Model quality on target domain |
| Ruff/lint/build | Source consistency and compilability | Runtime hardware fit |

## Real model startup gate

1. Run the Kaggle notebook through cell 9.
2. Require GPU, model, local HTTP, and public ngrok smoke tests to print `PASS`.
3. Configure `.env` with the printed URL and matching bearer token.
4. Start `scripts/run-local.py`.
5. Require `GET :8000/v1/health/ready` to report database and model gateway true.

If startup fails with repeatable CUDA OOM on a clean GPU, local-laptop inference does not pass; use
the free-GPU bridge and record that as the accepted execution profile.

## Five mandatory end-to-end acceptance scenarios

Use licensed single, temporal-pair, and optical/SAR fixtures. Run
`scripts/sih-acceptance.py` as documented in [SIH_DEMO_RUNBOOK.md](SIH_DEMO_RUNBOOK.md).

| Step | Assertion |
|---|---|
| Single VQA | One valid raster; non-empty adapted-Qwen answer and pinned provenance |
| Additional single task | Caption and grounding both run; invalid box becomes a warning |
| Temporal change | Time A/B validation; quantified change facts and time-B candidate evidence |
| Optical/SAR | Declared modalities; pair compatibility; complementary proxy facts/evidence |
| Automatic orchestration | `requested_tasks=null` selects by query and evidence configuration |
| Upload | Every asset returns `201`, SHA-256, and raster metadata |
| Preview | `200 image/jpeg`, non-empty body |
| Analysis create | `202`, status `queued` |
| Location context | Valid lat/lon/altitude appears in request provenance and user-location fact |
| Completion | Terminal state becomes `succeeded` within five minutes |
| Text | `result.answer` is non-empty and not demo/simulator copy |
| Model provenance | Revision contains adapter SHA `ed12e59e0de...` |
| Confidence | Marked uncalibrated/evidence quality, not correctness probability |
| Report | PDF endpoint returns non-empty `application/pdf` |
| Grounding | Valid box produces an overlay; invalid/no box produces warning, not fake geometry |

The automated run executes these five registered tasks:

1. `single_vqa`: a land-cover question whose answer is visually checkable.
2. `caption`: the representative land-cover/major-objects description.
3. `grounding`: highlight one water region.
4. `change_vqa`: describe what changed and where.
5. `optical_sar_fusion`: identify candidate built-up and water-covered regions jointly.

Record the image license/source, query, reference answer, model answer, latency, revision, warnings,
and manual judgement. One successful query proves integration, not model accuracy.

## Model-quality gates

| Specialist | Required evidence before release |
|---|---|
| Qwen single-image | Pinned fresh reload; task-specific validation and held-out test report |
| Grounding | Box parse rate plus localization IoU on labeled samples |
| Change-VQA | Answer accuracy/F1 plus mask IoU/Dice on disjoint SECOND test identities |
| Optical/SAR | Macro-F1/AP plus S2-only/S1-only ablations and dense-evidence metric |

CPU pair baselines pass a **functional demonstration gate**, not the learned model-quality gate.
Final submission evidence must include scores on the exact organizer-prescribed public splits.

## Regression policy

- A changed model revision requires rerunning real-model and quality gates.
- A changed preprocessing rule requires a new runtime manifest and evaluation.
- A changed API schema requires backend integration tests and frontend build.
- A visual-only change still requires the build and one upload/result interaction.
- Failed or skipped gates must remain explicit in `PROGRESS.md`; never relabel them complete.
