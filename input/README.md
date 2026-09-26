# Demo image inputs — choose the folder matching the evidence mode

All TIFFs are byte-identical copies of previously retained inputs, renamed for use. Checksums,
band descriptions, CRS and provenance are in [manifest.json](manifest.json). No new model results
are claimed by copying these images. Leave latitude/longitude/altitude blank unless known; never
use the form's example coordinates as observations.

| Demo | Files | UI settings | Suggested query |
|---|---|---|---|
| Single-image description | `01_single/01_mixed_landcover_rgb.tif` | Single scene; Optical; strict or exploration | Describe the visible land cover and distinguish observations from uncertainty. |
| Single-image experimental vegetation mask | Same file | Single scene; Optical | Show me the vegetation in this image. |
| Verified-band water analysis | `01_single/02_india_water_multispectral.tif` | Single scene; Multispectral; strict | Show me all the water bodies in this region. |
| Temporal change | `02_temporal/01_nepal_BEFORE.tif` then `02_temporal/02_nepal_AFTER.tif` | Bi-temporal pair; Optical; **Exploration** | What has changed between the two images? Show the changed regions. |
| Optical/SAR flood fusion | `03_fusion/india/01_OPTICAL_Sentinel2_13bands.tif` then `02_SAR_Sentinel1_VV_VH.tif` in the same folder | Optical + SAR pair; Multispectral; **strict** | Use both images to identify water-covered regions and show the flood-water mask. |
| Alternate fusion scenes | Same two filenames under `03_fusion/ghana/` or `03_fusion/mekong/` | Same settings; never mix regions | Identify water-covered regions using the optical and SAR inputs together. |

Keep specialist route on automatic first. Check the resulting trace to establish which model/tool
actually ran. A spectral mask is an analytical result; do not label it learned segmentation.

## Important distinctions

- Single TIFF has three unnamed bands. Treat it as visual imagery; do not claim verified spectral indices.
- Nepal TIFFs have matching dimensions but **no CRS or verified acquisition dates**. Their pixel-grid
  comparison is an exploration demonstration, not independently verified registration, metric area,
  a documented flood timeline or a causal explanation.
- Fusion files are real **Sen1Floods11** held-out test chips with 13 Sentinel-2 bands and two Sentinel-1
  VV/VH channels. They retain the documented band/units metadata used during the earlier audit.
  The two rasters in each regional folder have matching dimensions, CRS and transform.
- `04_reference_labels_DO_NOT_UPLOAD/` contains labelled reference masks for review only.
  **Never upload them as a satellite image or pass them into model inference.** They are not predictions.
- Flood water includes water segmentation; it does not by itself distinguish permanent water from
  new inundation or establish a disaster cause.
- User-provided sample/Nepal acquisition provenance and redistribution rights are not fully verified.
  Keep their attribution note in the video; obtain permission before broader redistribution.
- Official flood source: [Sen1Floods11 repository](https://github.com/cloudtostreet/Sen1Floods11).

## Recording order

Description → single mask (show experimental warning) → temporal before/after → real optical/SAR
fusion → trace → downloaded PDF. If a run abstains or falls back, keep that visible and explain it.
Do not promise every example will succeed on an unseen runtime; rehearse once after starting 06.
