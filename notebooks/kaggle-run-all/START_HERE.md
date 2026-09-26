# SatQuery Kaggle notebooks

These are the canonical upload-ready notebooks. Import the `.ipynb` file into Kaggle, enable Internet, choose a T4/P100 GPU where required, attach only the stated inputs and use **Run all**.

| Order | Notebook | Purpose | Input |
|---:|---|---|---|
| 00 | `00_Audit_Runtime_Preprocessing.ipynb` | Runtime/preprocessing contract | none |
| 01 | `01_Qwen_Validation_and_Scorecard.ipynb` | Base-vs-LoRA validation | none |
| 02 | `02_Train_Vegetation_Water_Masks.ipynb` | SegFormer training | none |
| 02 R4 | `02_R4_Refine_Strict_Segmentation.ipynb` | Protected full-data refinement | selected 02 artifact |
| 02B | `02B_Preserve_Training_Checkpoints.ipynb` | Preserve interrupted checkpoints | full saved 02 output |
| 03 | `03_Train_Temporal_Change.ipynb` | Learned change answer + mask | none |
| 04 | `04_Train_Optical_SAR_Flood_Masks.ipynb` | Learned S1/S2 flood fusion | none |
| 05 | `05_Qwen_Final_Test.ipynb` | Frozen Qwen test | saved 01 output |
| 06 strict | `06_All_Models_Ngrok_Server.ipynb` | Enforce all release gates, then serve | one 02/03/04 artifact each |
| 06 demo | `06_Demo_Current_Models_Ngrok.ipynb` | Serve selected models with failed 02 labelled experimental | one 02/03/04 artifact each |

For the current jury demonstration, training outputs are already preserved. Use **06 demo**. Strict 06 correctly refuses the single-image segmentation artifact because its mean/forest release checks did not pass.

Notebook 06 requires Kaggle Secrets named `HF_TOKEN`, `NGROK_AUTHTOKEN` and `SATQUERY_MODEL_SERVICE_TOKEN`. Keep values out of code and screenshots. It prints a temporary HTTPS URL and a local backend environment file; the tunnel works only while the attended notebook session is running.

Full setup, artifact handling and troubleshooting: [`docs/RUNBOOK.md`](../../docs/RUNBOOK.md). Metrics and gates: [`docs/EVALUATION.md`](../../docs/EVALUATION.md).
