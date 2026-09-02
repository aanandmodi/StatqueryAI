# SIH five-workflow demonstration runbook

## Inputs

Prepare licensed GeoTIFFs with meaningful CRS and affine transforms:

| Argument | Required content |
|---|---|
| `--single` | One optical, multispectral, or SAR scene |
| `--time-a`, `--time-b` | Same-area, aligned temporal pair |
| `--optical`, `--sar` | Same-area, aligned optical/SAR pair |

The controller rejects pairs below 98% overlap, above 2% relative resolution difference, or above
0.25-pixel fractional grid offset. Do not resave a normal photograph with a `.tif` extension.

## Run

1. Start the free Kaggle/Colab Qwen server and protected ngrok cell using
   [NGROK_KAGGLE_RUNBOOK.md](NGROK_KAGGLE_RUNBOOK.md).
2. Set the matching URL/token in `.env` and start `scripts/run-local.py`.
3. Run all mandatory scenarios:

```powershell
$python = "$env:USERPROFILE\.venvs\satquery\Scripts\python.exe"
& $python .\scripts\sih-acceptance.py `
  --single D:\fixtures\single.tif `
  --time-a D:\fixtures\before.tif `
  --time-b D:\fixtures\after.tif `
  --optical D:\fixtures\optical.tif `
  --sar D:\fixtures\sar.tif
```

## Expected artifacts

```text
artifacts/sih-acceptance/
├── summary.json
├── single_vqa.json/.overlay.jpg/.report.pdf
├── caption.json/.overlay.jpg/.report.pdf
├── grounding.json/.overlay.jpg/.report.pdf
├── change_vqa.json/.overlay.jpg/.report.pdf
└── optical_sar_fusion.json/.overlay.jpg/.report.pdf
```

## Judge-facing explanation

| Scenario | What to show | What not to claim |
|---|---|---|
| Single VQA | Pinned adapted Qwen revision, text answer, evidence-quality caveat | Do not call the score a probability of correctness |
| Caption | Remote-sensing description and trace | Do not infer invisible bands or sensor metadata |
| Grounding | Valid model box or explicit parse warning | Do not draw a decorative/fake box |
| Change | Two-input validation, changed fraction, time-B candidate box | Do not call spectral difference a confirmed semantic change class |
| Optical/SAR | Complementary optical/backscatter proxies and candidate regions | Do not call baseline output TerraMind or calibrated land-cover detection |

The demo establishes functional coverage and orchestration. Benchmark tables establish accuracy;
neither should be presented as a substitute for the other.

