# SatQuery jury demonstration TIFF pack

These are the exact TIFF assets assembled for the local SatQuery demonstrations. Their names state
the intended workflow and role. The hashes in [`manifest.json`](manifest.json) let you verify that
the presented inputs did not change.

## Fast demo map

| Workflow | Upload these files | UI settings | Suggested query |
|---|---|---|---|
| Single scene | `01_single_water_vegetation_scene.tif` | One scene; optical; strict | “Describe the scene and outline visible water candidates.” |
| Temporal change | `02_temporal_nepal_bhotekoshi_before.tif`, then `03_temporal_nepal_bhotekoshi_after.tif` | Bi-temporal pair; optical; exploration; pixel-grid registration | “Compare both dates, quantify visible change, and mark the changed regions.” |
| Optical/SAR fusion | `04_fusion_optical_reference_scene.tif`, then `05_fusion_sar_display_proxy.tif` | Optical/SAR pair; exploration; pixel-grid registration | “Fuse both views and explain where the optical and radar cues agree or disagree.” |

## Important evidence limits

- File 01 is georeferenced, but its bands have no semantic descriptions. It is suitable for upload,
  visual VQA and generic grounding; do not claim NDVI/NDWI from guessed RGB/NIR bands.
- Files 02 and 03 are display RGB TIFFs without CRS/geotransform. They are intended for the
  pixel-aligned exploration workflow. The system must not report square metres or geographic
  coordinates from them.
- File 05 is a grayscale SAR **display proxy**, not calibrated complex SAR or certified sigma0.
  Use it to demonstrate the fusion workflow and honest cue reporting, not quantitative
  backscatter physics.
- All masks remain candidate evidence unless independently checked against labelled ground truth.

## Automated local acceptance

With the backend running on port 8000 and the Kaggle model tunnel reachable:

```powershell
.\.venv\Scripts\python.exe scripts\sih-acceptance.py `
  --single demo-assets\01_single_water_vegetation_scene.tif `
  --time-a demo-assets\02_temporal_nepal_bhotekoshi_before.tif `
  --time-b demo-assets\03_temporal_nepal_bhotekoshi_after.tif `
  --optical demo-assets\04_fusion_optical_reference_scene.tif `
  --sar demo-assets\05_fusion_sar_display_proxy.tif `
  --pair-profile exploration `
  --auto-route `
  --output outputs\jury-evidence\tiff-demo-pack
```

This verifies transport, routing, reports, overlays and provenance. It is not a semantic accuracy
benchmark.

## Latest local preflight

On 2026-09-08 the renamed TIFFs passed the real local upload/validation and pair-analysis paths:

| Workflow | Analysis | Evidence | Specialist |
|---|---|---:|---|
| Single upload validation | asset `ast_2f5de44f924949aa8aeac73983b15392` | n/a | strict GeoTIFF validation passed |
| Temporal | `anl_68d261e5a2db4d27abf0f120bb317a57` | 2 masks | `satquery-spectral-change-tool-v2` |
| Optical/SAR | `anl_ff3ac04b121040198030cd72e288b3b2` | 2 masks | `satquery-optical-sar-proxy-tool-v2` |

The generated overlays and PDF reports are under
`outputs/jury-evidence/tiff-demo-pack/`. These are integration evidence for the explicitly labelled
analytical baselines, not learned-checkpoint accuracy evidence.
