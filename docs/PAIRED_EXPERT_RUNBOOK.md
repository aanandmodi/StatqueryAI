# Free cloud model upgrades — exact execution order

Keep frontend/controller on localhost. Use the existing attended Kaggle GPU and existing ngrok
tunnel; this does not provision paid infrastructure. Free-provider quotas/session limits still apply.
Do not send tokens in chat or place them in model artifacts.

## 1. Improve the currently running Qwen service now

1. Open `notebooks/SatQuery_Live_Planning_Upgrade.ipynb` locally to copy its code, or open
   `notebooks/patches/live_planning_upgrade.py`.
2. In the **existing** Kaggle server notebook, interrupt only section 10's waiting cell if running.
   Do not restart the GPU or rerun model-loading/training cells.
3. Paste the entire patch into one new code cell and run it after quality section 6b.
4. Expect `Learned intent route installed`. The cell also installs sensor-qualified RGB decoding.
5. If the original attended tunnel window remains active, resume section 10. Otherwise follow
   sections 6–9 of the updated server notebook and apply 6b/6c in order; do not automate renewal.
6. The local backend defaults to `SATQUERY_PLANNER_BACKEND=auto`. No new API key is required;
   it uses the existing service bearer token. A valid proposal produces `learned-intent:...` in
   the case trace; unavailable/invalid proposals produce an explicit fallback label.

For a new GPU server session, use the updated `SatQuery_Qwen3VL_Free_GPU_Server.ipynb`.
Its section **6c** embeds the planner. A `.ipynb` opened as a separate fresh session does not inherit
the existing notebook's model/app globals: copy the live cell into the active server, not a new kernel.

## 2. Train the learned temporal expert (separate training session)

Notebook: `notebooks/SatQuery_ChangeVQA_Training.ipynb`.

- Obtain the actual licensed SECOND before/after pixels and semantic labels; the official
  [CDVQA repository](https://github.com/YZHJessica/CDVQA) contains question/answer annotations,
  not an automatically available drop-in trained model.
- Attach SECOND as a Kaggle dataset and follow the path-discovery/configuration cell. Missing
  pixels deliberately stop execution. Train/validation SECOND pair filenames must be disjoint;
  annotation IDs restart within each split and are not global identities.
- The model learns shared ResNet image features, temporal feature fusion, question encoding,
  closed-vocabulary answers and a supervised binary semantic-change mask. It is not a new
  fine-tune of the existing Qwen LoRA and it is not a flood-cause model.
- Export is `artifact_version=satquery-pair-v2`, `model.safetensors`, `config.json`, training
  history, raw validation/test JSONL and `sha256_manifest.json`. A fresh architecture strictly
  reloads the selected weights and runs a finite-output check. Keep Kaggle outputs/private input
  for serving.
- Optional Hub storage uses `HF_TOKEN` in Kaggle Secrets with `push_to_hub=True`; storing weights
  does not create inference hosting. Default is no upload.
- Training can still exceed free session time/VRAM. Optimizer/model state resumes from the local
  artifact directory, but a Kaggle session reset still requires preserving that directory as a
  private dataset/version. Tune sample limits openly; do not call a subset a full benchmark.
  Hub upload is refused until operator-declared answer-accuracy and mask-IoU thresholds pass on
  validation and the untouched test split.

## 3. Train learned optical/SAR pixel fusion

Notebook: `notebooks/SatQuery_TerraMind_Fusion_Training.ipynb`.

- Attach the official Sen1Floods11 v1.1 folders `S2L1CHand`, `S1GRDHand`, `LabelHand` and its
  `flood_{train,valid,test}_data.txt` files. The notebook refuses missing triplets, duplicate IDs,
  overlapping splits or non-co-registered grids and records SHA-256 for every split file.
- TerraMind consumes all 13 declared Sentinel-2 L1C channels plus Sentinel-1 VV/VH. The head is a
  two-class pixel decoder trained with cross-entropy plus Dice loss; invalid label `-1` is ignored.
- The saved best validation-IoU weights—not the last epoch—are evaluated on fused, S2-only and
  S1-only inputs. Validation/test JSONL retains reversible mask runs and per-chip intersections,
  unions and flood fractions so reported metrics can be audited.
- Export is the strict `satquery-pair-v3` architecture
  `terramind_s1_s2_pixel_flood_segmentation`. Hub upload is refused unless an operator-declared
  flood-IoU target passes on validation and untouched test and fused validation is no worse than
  either single-modality ablation.
- Serving accepts exactly declared Sentinel-2 L1C `toa_reflectance_10000` (or the equivalent
  top-of-atmosphere label), all 13 bands B01–B12 including B10, and Sentinel-1 `sigma0_db` VV/VH.
  It returns a learned flood/water candidate mask. It explicitly rejects Cartosat/RISAT transfer;
  this checkpoint cannot justify arbitrary vegetation, building or burn-scar boundaries.

## 4. Load real paired artifacts behind the same ngrok service

Notebook: `notebooks/SatQuery_Optional_Paired_Runtime.ipynb`.

1. Finish the training session and preserve its output. Attach the artifact folder(s) to your
   GPU **server** notebook as private Kaggle inputs.
2. Copy the runtime-bundle cell and optional-route cell into that existing server after 6b/6c.
   The bundle is repository-owned Python, not code fetched from an arbitrary model repository.
3. Set `PAIR_ARTIFACT_DIRS["change"]` / `["fusion"]` to the exact attached artifact directories.
   Leave missing models empty. Fusion needs `terratorch` from the training environment; install
   it in a clean server environment if unavailable, then rerun sections in order. Do not change
   torch/CUDA versions underneath an active model.
4. Strict config, hash and state-dict checks must pass. `/ready` lists loaded capabilities
   independently. Loading is not an accuracy check and it is not a paired HTTP smoke test.
5. Test real compatible pairs via `/v1/infer/change_vqa` and `/v1/infer/optical_sar_fusion` with
   the existing bearer token. Pair uploads require real byte SHA256 values and explicit roles.
   Masks return bounded PNG/base64 with valid source-grid geometry. Missing experts return 409.
6. After at least one learned expert has a pinned checkpoint and passes its held-out and HTTP
   tests, set root `.env` and give each available expert its explicit URL:

   ```dotenv
   SATQUERY_PAIR_BACKEND=http
   SATQUERY_CHANGE_SERVICE_URL=https://YOUR-CHANGE-SERVICE
   SATQUERY_FUSION_SERVICE_URL=https://YOUR-FUSION-SERVICE
   ```

7. Restart only the local backend. The controller tries the learned endpoint and records its
   learned method on success. A missing, incompatible or unreachable expert automatically returns
   to the existing local analytical method and records the fallback reason. This availability
   fallback is not permission to call the analytical result learned.

## 5. Test the compound controller workflow

Upload two aligned optical images with time-A/time-B roles. Ask:

> Highlight the water bodies, compare before and after, and calculate the lost area.

Expect four steps and explicit method provenance. Final loss/gain counts must match the masks.
If target masks are missing, thresholds differ, or grids are not identical, expect a withheld
measurement. This is a successful safety behavior, not permission to fabricate an area.

For existing validated asset IDs:

```powershell
.\.venv\Scripts\python.exe scripts/verify-compound-live.py YOUR_TIME_A_ASSET_ID YOUR_TIME_B_ASSET_ID
```

See [critical-gap audit](CRITICAL_GAPS.md) for measured vs unvalidated claims and sensor sources.
