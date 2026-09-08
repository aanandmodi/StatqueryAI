# Jury evidence pack

Local demonstration artifacts are generated under `outputs/jury-evidence/live-three-mode-v2/` and are
ignored by Git because they are run outputs, not benchmark truth. Run IDs make each case retrievable
from the local SQLite audit store while it is preserved.

| Mode | Analysis ID | What the run proves | What it does not prove |
|---|---|---|---|
| Single image | `anl_dacb7462e8ec4814bc06919da9169702` | Local UI/controller → temporary Kaggle Qwen/SAM transport, structured response, one mask and seven bounded polygons | Regional accuracy, calibrated confidence or perfect masks |
| Bi-temporal pair | `anl_c73d35849c254c0a8103ef596f50c1b2` | Two aligned real display images, three candidate masks, 134 bounded polygons and analytical comparison | A released learned CDVQA checkpoint, dates, causality or metric area |
| Optical/SAR pair | `anl_096e54f5aa65489a9cf5caa0af133441` | Co-registered real display pair, one proxy mask and 64 bounded polygons under explicit exploration limits | Calibrated SAR backscatter, learned fusion or pixel-label accuracy |

## Sources and licenses

- Single/fusion display inputs come from the MIT-licensed
  [SAR2Opt dataset](https://huggingface.co/datasets/umkc-mcc/SAR2Opt). The selected pair is stored in
  `outputs/jury-evidence/sources/sar2opt/`.
- Temporal display inputs come from the Apache-2.0
  [RSRCC dataset](https://huggingface.co/datasets/google/RSRCC). The selected pair is stored in
  `outputs/jury-evidence/sources/temporal/`.

The selected JPG/PNG files lack embedded CRS, ground sampling distance, calibrated SAR values and
verified acquisition dates. The demo therefore uses normalized pixel geometry and withholds metric
area and event-cause claims. Use georeferenced, metadata-complete GeoTIFF inputs for operational
measurements.

## Regenerate while Kaggle is live

```powershell
Set-Location D:\Projects\Sih-2026
.\.venv\Scripts\python.exe scripts\run-jury-evidence.py `
  --single outputs\jury-evidence\sources\sar2opt\11_1200_0_optical.jpg `
  --time-a outputs\jury-evidence\sources\temporal\rsrcc_7df86677_before.png `
  --time-b outputs\jury-evidence\sources\temporal\rsrcc_7df86677_after.png `
  --optical outputs\jury-evidence\sources\sar2opt\11_1200_0_optical.jpg `
  --sar outputs\jury-evidence\sources\sar2opt\11_1200_0_sar.jpg `
  --output outputs\jury-evidence\live-three-mode-v2
```

Keep the 60-minute ngrok window attended. If the tunnel expires, rerun Kaggle sections 6–10 and
update only the local URL; never expose the shared service token in a screenshot or jury document.
