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
  history and `sha256_manifest.json`. A fresh architecture strictly reloads the saved weights
  and runs a finite-output check. Keep Kaggle outputs/private input for serving.
- Optional Hub storage uses `HF_TOKEN` in Kaggle Secrets with `push_to_hub=True`; storing weights
  does not create inference hosting. Default is no upload.
- Training can still exceed free session time/VRAM. Tune batch size and sample limits openly;
  do not claim a full benchmark result from a small subset. Optimizer-state resume and independent
  test-set release automation remain future work; validation metrics alone are insufficient.

## 3. Train learned optical/SAR scene classification

Notebook: `notebooks/SatQuery_TerraMind_Fusion_Training.ipynb`.

- Uses an immutable BigEarthNet-derived paired LMDB dataset revision, actual S1/S2 channels,
  official split labels and TerraMind normalization. Native-resolution bands resize separately.
- TerraMind tiny/base is selected by available VRAM; the probed embedding width is recorded.
- The saved best weights—not the last epoch—are evaluated with fused, optical-only and SAR-only
  inputs. Class scores remain uncalibrated.
- **This notebook does not train precision segmentation.** Scene labels supervise the head;
  activation maps do not establish object boundaries. Runtime intentionally returns no fusion masks.
- Serving expects declared Sentinel-2 L2A `surface_reflectance_10000`, Sentinel-1 `sigma0_db`, all
  12 explicit S2 band descriptions, and explicitly identified VV/VH. It rejects Cartosat/RISAT
  substitutions rather than silently fabricating channels.

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
6. Only after **both** experts pass their tests, set root `.env`:

   ```dotenv
   SATQUERY_PAIR_BACKEND=http
   # Empty specialist URL overrides use the same SATQUERY_MODEL_SERVICE_URL.
   ```

7. Restart only the local backend. Never set `pair_backend=http` expecting the old single-image
   Qwen server to synthesize missing pair checkpoints. With only one expert trained, keep the
   production controller baseline configuration and test that expert directly until both are ready.

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
