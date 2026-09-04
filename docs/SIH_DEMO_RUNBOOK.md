# SIH five-workflow demonstration runbook

## Inputs

Prepare licensed GeoTIFFs with meaningful CRS and affine transforms:

| Argument | Required content |
|---|---|
| `--single` | One optical, multispectral, or SAR scene |
| `--time-a`, `--time-b` | Same-area, aligned optical temporal pair with compatible bands; this CLI declares these inputs optical |
| `--optical`, `--sar` | Same-area, aligned optical/SAR pair |

The controller rejects pairs below 98% overlap, above 2% relative resolution difference, or above
0.25-pixel fractional grid offset. Do not resave a normal photograph with a `.tif` extension.
For a non-optical single scene, pass `--single-modality multispectral` or `--single-modality sar`.
Use genuine same-area acquisitions and retain dataset/source, licensing and reference labels.
Synthetic fixtures are useful for pipeline tests, not for benchmark claims.

## Run

1. Start the temporary Kaggle Qwen server using
   [NGROK_KAGGLE_RUNBOOK.md](NGROK_KAGGLE_RUNBOOK.md). Require sections 5, 7 and 9 to pass; keep
   section 10 running during the attended demo. Managed Colab proxy serving is not supported.
2. Set the matching URL/token in repository-root `.env`, keep `.env.local` pointed at the local
   controller, and start `scripts/run-local.py --mode remote` using the `satquery-cloud` virtualenv.
3. In a second terminal, run all mandatory scenarios. The recommended command enables automatic
   query-driven routing and uses strict defaults that reject simulator responses:

```powershell
Set-Location D:\Projects\Sih-2026
$satqueryPython = "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe"
& $satqueryPython .\scripts\sih-acceptance.py `
  --single D:\fixtures\single.tif `
  --time-a D:\fixtures\before.tif `
  --time-b D:\fixtures\after.tif `
  --optical D:\fixtures\optical.tif `
  --sar D:\fixtures\sar.tif `
  --auto-route `
  --output .\outputs\sih-real-acceptance
```

Replace the fixture paths with your actual files. No dataset images are implied to exist at these
example locations. The five chosen queries cover VQA, caption, grounding, change and optical/SAR.
`--auto-route` sends `requested_tasks=null` and requires each expected task in result provenance.
Run again without `--auto-route`, using a different output directory, to test explicit task selection.

The CLI rejects `demo-simulator` model versions by default. For an intentionally simulated
UI/API-only check, start `run-local.py --mode demo` and explicitly add `--allow-simulated` to the
acceptance command, saving to a distinct directory such as `outputs/sih-plumbing`. Such a result
is **not** a real-model acceptance result or a benchmark score.

## Expected artifacts

```text
outputs/sih-real-acceptance/
├── summary.json
├── single_vqa.json/.overlay.jpg/.report.pdf
├── caption.json/.overlay.jpg/.report.pdf
├── grounding.json/.overlay.jpg/.report.pdf
├── change_vqa.json/.overlay.jpg/.report.pdf
└── optical_sar_fusion.json/.overlay.jpg/.report.pdf
```

The command must finish with `PASS real-model integration`, and every task must contain a trace,
nonempty answer and model/tool provenance. Inspect the exact Qwen revision, image overlay and
PDF manually. A parsed box is a model proposal, not proof of localization correctness; an absent
box must remain an explicit warning. Test user-location metadata separately through the UI.

Do not claim a live pass merely because the code compiles, the site builds or the notebook's
synthetic transport fixture succeeds. Keep recorded real-image runs and semantic evaluation results
distinct. `summary.json` records whether automatic routing or simulator allowance was enabled.

## Reproducible local plumbing check — no GPU claim

To test before Kaggle is available, start the local app with `run-local.py --mode demo` in one
terminal. In a second terminal, create explicitly marked synthetic fixtures and exercise the
actual frontend proxy, controller, simulator and CPU pair tools:

```powershell
Set-Location D:\Projects\Sih-2026
$satqueryPython = "$env:USERPROFILE\.venvs\satquery-cloud\Scripts\python.exe"
& $satqueryPython .\scripts\make-smoke-fixtures.py --output .\outputs\synthetic-smoke
& $satqueryPython .\scripts\sih-acceptance.py `
  --base-url http://localhost:3000/api/satquery `
  --single .\outputs\synthetic-smoke\single.tif `
  --time-a .\outputs\synthetic-smoke\time-a.tif `
  --time-b .\outputs\synthetic-smoke\time-b.tif `
  --optical .\outputs\synthetic-smoke\optical.tif `
  --sar .\outputs\synthetic-smoke\sar.tif `
  --auto-route `
  --allow-simulated `
  --output .\outputs\sih-plumbing
```

The generator refuses to overwrite existing fixtures; choose a new output directory if necessary
and update all five paths. This run must be labeled `SIMULATOR-ALLOWED plumbing`, not real-model
acceptance. The fixture coordinates and imagery are synthetic, even though they are valid GeoTIFFs.
Once Kaggle is connected, use genuine licensed inputs and remove `--allow-simulated` for the
real-model command above. `node --test tests/proxy.test.mjs` separately tests the HTTP proxy route.

## Judge-facing explanation

| Scenario | What to show | What not to claim |
|---|---|---|
| Single VQA | Pinned adapted Qwen revision, text answer, evidence-quality caveat | Do not call the score a probability of correctness |
| Caption | Remote-sensing description and trace | Do not infer invisible bands or sensor metadata |
| Grounding | Valid model box or explicit parse warning | Do not draw a decorative/fake box |
| Change | Two-input validation, changed fraction, candidate box on the time-A reference grid | Do not call spectral difference a confirmed semantic change class |
| Optical/SAR | Complementary optical/backscatter proxies and candidate regions | Do not call baseline output TerraMind or calibrated land-cover detection |

The demo establishes functional coverage and orchestration. Benchmark tables establish accuracy;
neither should be presented as a substitute for the other.

## Finish safely

Stop the local runner with `Ctrl+C`, interrupt notebook section 10, then stop the Kaggle session
to release GPU quota. The 60-minute maximum window is not an idle-timeout bypass or uptime promise.
If the model is still loaded after stopping, rerun sections **6–10** for another permitted attended
test; never automate renewals to evade platform limits. No website or permanent endpoint is deployed.
