# SatQuery AI
### SIH26167 — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
**Sponsoring Organization:** Indian Space Research Organisation (ISRO) | **Theme:** Space Technology | **Prize:** ₹1,00,000 | **Deadline:** 20 September 2026

> This document is aligned to the **actual official problem statement text**. Every mandatory requirement is traceable to the PS (§4). On top of that compliant core, §9 adds an optional **dynamic query-planning + wide-area layer** that brings back the "ask about anywhere, in any phrasing" breadth from the original product vision — without weakening the graded core.

---

## 1. What is this, in plain English?

Most "AI for satellite images" tools are one-trick ponies — one model that only does land classification, or only does change detection. SatQuery AI is a **dispatcher with a toolbox**, and — with the addition in §9 — a dispatcher that can also plan multi-step investigations and fetch its own evidence for any place on Earth, not just the files you hand it.

Think of it like a hospital. A patient (query) walks in. The receptionist (the agentic controller) figures out *what kind of case this is*, checks what test results are available (input images), sends the patient to the right specialist doctor (a fine-tuned remote-sensing model), and writes up a report combining what the specialist found. The upgraded version of this receptionist can also: order additional tests itself if the question needs more than one doctor's opinion (query planning), and pull a patient's medical history from records if none was handed over in person (wide-area satellite fetch).

The specialist doctors this system must have:
1. A doctor who reads **one image** and answers questions about it, describes it, or points at things in it.
2. A doctor who compares **two images of the same place, taken at different times**, and explains what changed.
3. A doctor who reads **one optical photo + one radar scan of the same place together** and combines what each uniquely reveals.
4. The receptionist itself — decides which doctor(s) to call, in what order, and keeps a written log of exactly what it did (this log is what gets graded).

---

## 2. Why this exact framing matters (context, from the PS itself)

- Remote sensing spans agriculture, disaster response, urban planning, forestry, water resources, and infrastructure — but existing AI tools are isolated single-task systems, forcing non-expert users to understand GIS workflows just to ask a simple question.
- A single image is often not enough. Optical/multispectral imagery gives color and texture; SAR gives structural information and works through clouds and at night. Combining them, or comparing the same place across time, answers questions neither can answer alone.
- A general-purpose LLM/VLM, unmodified, cannot reliably interpret sensor-specific remote-sensing imagery. **Domain adaptation is mandatory, not optional.**
- The novelty ISRO wants is the **agentic framework** — a system that decides which specialist tool(s) to use per query, rather than one monolithic model doing everything.

---

## 3. Core idea in one sentence

> **An agentic AI system that reads a natural-language query — potentially a multi-part or follow-up query — plans which specialist tasks it needs, checks or fetches the right image(s), routes each task to the correct fine-tuned remote-sensing model, and returns a synthesized answer with visual evidence, confidence scores, and a transparent log of exactly which models ran and why.**

---

## 4. Mandatory requirements → what we build (compliance map)

Build against this table directly. Every row is graded.

| PS Requirement | What it means | What you build |
|---|---|---|
| Remote-sensing adaptation | ≥1 vision/VLM component must be fine-tuned or domain-adapted | LoRA fine-tune a base VLM on BigEarthNet-MM (+ VRSBench) |
| Single-image VQA (mandatory) | Answer free-text questions about one image | Fine-tuned VQA head/VLM, trained/evaluated on RSVQA + VRSBench |
| + Captioning OR grounding (mandatory, pick ≥1) | Describe the scene, or point to a region matching a text phrase | Recommend **grounding** — matches the PS's own example query ("highlight the water body") |
| Multi-image change analysis (mandatory) | Describe/answer questions about what changed between two dates | Change-VQA model trained/evaluated on CDVQA |
| Cross-modal optical+SAR analysis (mandatory) | Extract info only visible when combining both sensors | Dual-branch fusion classifier trained on BigEarthNet-MM |
| Agentic orchestration (mandatory) | Auto-select/sequence/execute the right tool(s) per query | A controller with a fixed tool registry + structured JSON routing + execution log |
| Input compatibility checking | Validate format, modality, count, metadata before running | Format/CRS/co-registration checks via `rasterio` before dispatch |
| Confidence + visual evidence + execution summary | Every answer must show its work | Confidence scores, overlay images, audit trail with every response |
| GeoTIFF/TIFF as primary format | PNG/JPEG only allowed for the named public benchmarks | Enforced at the upload/fetch step — see §9, both entry paths produce real GeoTIFF |

*(§9 extends this system — it doesn't replace or weaken any row above.)*

---

## 5. High-level architecture — mandatory core

```
┌─────────────────────────────────────────────────────────────────────┐
│                  USER (Interactive GUI / Web App)                   │
│   Uploads: single image | optical+SAR pair | bi-temporal pair       │
│   Types: natural-language query                                     │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT VALIDATION LAYER                                              │
│   - Format check (GeoTIFF/TIFF mandatory; PNG/JPEG only for          │
│     the named public benchmark datasets)                             │
│   - Modality check (optical / SAR / both)                            │
│   - Count check (1 image / pair / bi-temporal pair)                  │
│   - Metadata check (CRS, geotransform, co-registration bounds match) │
└───────────────────────────────┬───────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENTIC CONTROLLER                                                  │
│   1. Interpret query → classify task type                            │
│   2. Cross-check task type against validated input configuration     │
│   3. Select model(s)/tool(s) from a fixed registry                   │
│   4. Configure only permitted parameters, execute                    │
│   5. Combine outputs, estimate confidence, collect visual evidence   │
│   6. Emit an auditable execution trace (task, tools used, params)    │
└───────┬─────────────┬─────────────┬─────────────┬───────────────────┘
        ▼             ▼             ▼             ▼
   ┌─────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐
   │ VQA /    │  │ Grounding  │  │ Change-VQA │  │ Optical+SAR    │
   │ single-  │  │ (region    │  │ (bi-       │  │ fusion         │
   │ image    │  │ pointing)  │  │ temporal)  │  │ (cross-modal)  │
   │ model    │  │ model      │  │ model      │  │ model          │
   └─────────┘  └───────────┘  └───────────┘  └───────────────┘
        └─────────────┴─────────────┴─────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT INTEGRATION LAYER                                            │
│   - Merge text answer + spatial evidence (bounding box / mask)      │
│   - Confidence score (softmax / branch-agreement)                   │
│   - Downloadable report (PDF: query, trace, evidence, confidence)   │
└─────────────────────────────────────────────────────────────────────┘
                                 ▼
                 ANSWER + VISUAL EVIDENCE + EXECUTION LOG shown to user
```

This is the version to build and validate first — it alone satisfies every mandatory row in §4.

---

## 6. Why LoRA fine-tuning GeoChat/LLaVA is the practical starting point

You do not need to train a remote-sensing VLM from scratch. Instead:

- **GeoChat** (CVPR 2024, open-source, MBZUAI) is already a LoRA fine-tune of LLaVA-1.5 (CLIP-ViT-L/14 + Vicuna-v1.5) adapted for remote sensing: scene classification, VQA, region captioning, and visual grounding with spatial coordinates. It's the closest existing thing to "half of this problem statement already solved."
- **Your job**: take GeoChat (or LLaVA-1.5 directly) and do a **second round of LoRA fine-tuning** on:
  - BigEarthNet-MM-derived pseudo-captions (convert multi-label land-cover tags into template sentences, aligning image-text representations to multisensor data — exactly as the PS specifies)
  - VRSBench's training split (29,614 images, human-verified captions, 52,472 grounding phrases, 123,221 QA pairs — covers VQA, captioning, AND grounding in one dataset)
- This satisfies "must include remote-sensing fine-tuning or domain adaptation" cleanly, on free Colab/Kaggle GPU time.

**For change-VQA:** run the same fine-tuned visual backbone on time-A and time-B (Siamese/shared-weights), take the feature difference, and train a small classification head on **CDVQA's training split** (65,967 QA pairs from 1,600 image pairs, 6 land-cover classes) to answer change questions directly. Let the LLM handle only final phrasing.

**For optical+SAR fusion:** BigEarthNet-MM is built for this — 590K+ co-registered Sentinel-1 (SAR) + Sentinel-2 (optical) patch pairs with shared labels. A dual-branch CNN/ViT (one branch per modality, features fused before classification) is well-trodden and buildable.

---

## 7. Datasets you'll actually use

| Dataset | Role | Notes |
|---|---|---|
| **BigEarthNet-MM** | Primary training set for multisensor adaptation + optical-SAR fusion | 590K+ co-registered Sentinel-1 SAR + Sentinel-2 optical patches, CORINE Land Cover labels, GeoTIFF format |
| **VRSBench** | Train + evaluate single-image captioning, grounding, VQA | 29,614 images, captions, grounding phrases, QA pairs — covers 3 mandatory tasks at once |
| **RSVQA (LR + HR)** | Additional VQA training/eval | LR: 772 images/77,232 QA; HR: 10,659 images/1,066,316 QA |
| **CDVQA** | Train + evaluate change-VQA | 2,968 bi-temporal pairs, 122,000 QA pairs; train split 65,967 QA/1,600 pairs |
| **ISRO/SAC evaluation set** | Final hidden judging set | Co-registered Cartosat-2S optical + RISAT SAR pairs — treat domain generalization (Europe-trained → India-tested) as a real risk (§13) |

---

## 8. How the mandatory core communicates (single-task flow)

| Step | Sender → Receiver | Payload |
|---|---|---|
| 1 | User → Input Validation | Uploaded file(s) + query text |
| 2 | Input Validation → Controller | `{valid: true, modality: "optical+sar", count: 2, crs: "EPSG:32644", coreg_ok: true}` |
| 3 | Controller (query parse) | `{task: "change_vqa", requested_layer: "built-up", confidence_required: true}` |
| 4 | Controller → Specialist Tool | Validated image tensor(s) + task parameters |
| 5 | Specialist Tool → Controller | `{answer: "increased", region_mask: [...], confidence: 0.87}` |
| 6 | Controller → Output Layer | Combined result + **execution trace**: `{task: "change_vqa", models_used: ["change-head-v1"], params: {...}}` |
| 7 | Output Layer → User | Answer text + overlay image + confidence + downloadable PDF report |

The execution trace in step 6 is not cosmetic — only this observable trace is evaluated, not internal chain-of-thought. §9 extends this same trace mechanism to multi-step flows.

---

## 9. Making it dynamic and wide: query planning, reasoning, and wide-area mode

This is an optional layer wrapped **around** the mandatory core from §5–8. It doesn't touch the graded specialist models — it changes how queries reach them and how answers get assembled, bringing back the "ask about anywhere, in plain English" breadth from the original product idea while staying fully format-compliant.

### 9.1 Query Planner (LLM) — turns one message into an ordered task list

Right now the controller handles one task per message. The planner sits just above it and can decompose a richer question into a **sequence** of tool calls:

> *"Compare water and vegetation change between these two dates, and tell me which changed more"*

```json
[
  {"task": "change_vqa", "layer": "water"},
  {"task": "change_vqa", "layer": "vegetation"}
]
```

The controller still validates and executes each list item exactly as in §8 — one at a time, each producing its own logged trace — so reliability is unchanged. The difference is a **Reasoning/Synthesis pass** afterward: the LLM takes both structured results and writes one composite answer ("water extent shrank 8%, vegetation grew 3% — water changed more").

This directly strengthens the PS's own wording for agentic orchestration — *"select, sequence, and execute"* — since the system now visibly plans multi-step investigations instead of one call per message.

**Also add:**
- **Session memory** — a lightweight store (in-memory dict, or Redis if you want it persistent) keeps the current image(s) and prior results in context, so a user can ask follow-ups ("now check the SAR view of the same area", "what was the confidence on that?") without re-uploading. Every follow-up still produces its own execution trace.
- **Flexible parameter extraction** — the planner can pull out thresholds, class names, or sub-regions from free text ("only show change above 20%") and pass them as the controller's *"configure only permitted task parameters"* step — this is literally called out in the PS as required controller behavior.

### 9.2 Wide-Area Mode — answer questions about a place, not just a file

The mandatory core only works on files the user manually supplies. This adds a second entry path:

- User types a place name or draws a region on a map — no file upload
- Free geocoding via **OpenStreetMap Nominatim** turns it into a bounding box
- The system fetches matching Sentinel-2 (optical) + Sentinel-1 (SAR) tiles for that bounding box via **Google Earth Engine** or the **Copernicus Data Space Ecosystem**, exported as genuine **GeoTIFF with real CRS/geotransform metadata**

This is not a format violation — it's a different *source* of the same GeoTIFF input the mandatory pipeline already expects, so it passes through the identical input-validation layer in §5.

**Extended architecture:**

```
┌───────────────────────────────────────────────────────────────────────┐
│                    USER (Interactive GUI / Web App)                   │
│  Path A: uploads image(s)         Path B: names a place / draws a bbox│
│  + a query, possibly multi-part, possibly a follow-up                 │
└───────────────┬───────────────────────────────┬───────────────────────┘
                 ▼ (Path A)                      ▼ (Path B)
      [Direct GeoTIFF upload]        [Nominatim geocode → fetch Sentinel-
                                       1/2 GeoTIFF via GEE/Copernicus for
                                       that bbox + date(s)]
                 └───────────────┬───────────────┘
                                 ▼
                  INPUT VALIDATION LAYER (same checks, either path)
                                 ▼
        QUERY PLANNER (LLM) → ordered list of {task, layer, params},
        using session memory for follow-up context
                                 ▼
              AGENTIC CONTROLLER (loops over the list; §5/§8 flow
              runs once per item, each with its own trace)
                                 ▼
        REASONING / SYNTHESIS LAYER (LLM) — merges all structured
        results into one narrative, evidence and confidence intact
                                 ▼
     OUTPUT: answer + overlays + trend chart (if multi-date) + trace + PDF
```

**Trend mode (brings back the "10-year comparison" idea, done properly):** instead of a random uploaded photo needing geolocation, fetch a real yearly time series of GeoTIFFs for the named place (e.g., 2016–2026, one cloud-filtered composite per year via GEE), then have the planner emit N-1 pairwise `change_vqa` calls across consecutive years, and the reasoning layer aggregate them into one trend narrative plus a line chart. This is the same mandatory change-VQA model, just sequenced more times — a genuine demonstration of "select, sequence, and execute," not a new graded capability.

**Why this is worth building:** in a live demo, a judge can name almost any place and get a real answer, not just the 2–3 files you pre-loaded — this is the "wide" and "dynamic" feel from the original product vision. Build it *after* §5–8 are solid; if it's not finished in time, the mandatory core still stands on its own.

---

## 10. Free data & API providers (for wide-area mode)

| Provider | What you get | Notes |
|---|---|---|
| **Google Earth Engine** | Sentinel-1/2, Landsat, MODIS + server-side compositing | Free for hackathon/research use; exports valid GeoTIFF with CRS |
| **Copernicus Data Space Ecosystem** | Sentinel-1/2/3/5P | Free with monthly processing-unit quota |
| **OpenStreetMap Nominatim** | Free geocoding (place name → coordinates) | No key required, rate-limited |
| **ISRO Bhuvan / VEDAS** | Indian satellite data (Resourcesat, Cartosat) | India-specific, aligns with ISRO as sponsor — also a route to closing the domain-gap risk in §13 |
| **USGS EarthExplorer / NASA Earthdata** | Landsat archive back to 1972 | Fallback for time spans older than Sentinel's 2015 start |

---

## 11. Tech stack

| Layer | Recommended Tool | Why |
|---|---|---|
| Base VLM | LLaVA-1.5 / GeoChat (open weights) | Closest existing RS-adapted starting point |
| Fine-tuning | HuggingFace `transformers` + `peft` (LoRA) | Feasible on free Colab/Kaggle T4/P100 GPU |
| Optical-SAR fusion model | PyTorch, dual-branch CNN or ViT | Standard architecture for BigEarthNet-MM-style fusion |
| Change-VQA head | PyTorch, Siamese feature-diff + classification head | Buildable in hackathon time, trainable on CDVQA |
| Geospatial I/O + validation | `rasterio`, `GDAL`, `pyproj` | Reads GeoTIFF, checks CRS/co-registration/metadata |
| Query planner + reasoning | Gemini/Groq API with JSON/function-calling | Multi-task decomposition + final answer synthesis |
| Agentic controller | Closed-set task classifier + fixed tool registry | Reliable routing, avoids open-ended hallucination |
| Wide-area geocoding/fetch | Nominatim + Google Earth Engine Python API / Copernicus API | Free, exports valid GeoTIFF |
| Session memory | In-memory dict (hackathon) or Redis (if persistence needed) | Supports follow-up queries |
| Backend | Python + FastAPI | Ties everything together |
| Frontend | React (or Streamlit for speed) + Leaflet for overlay display | Interactive GUI requirement |
| Report generation | `reportlab` or HTML-to-PDF | Downloadable execution summary requirement |
| Training compute | Google Colab / Kaggle (free T4/P100 GPU) | Zero-cost fine-tuning |

---

## 12. Step-by-step build plan (assume ~40–48 hours including pre-hackathon prep)

### Phase 0 — Before the hackathon starts
- Download BigEarthNet-MM, VRSBench, RSVQA, CDVQA — large files, don't wait until hour 1
- Set up Colab Pro or Kaggle GPU access; confirm GeoChat/LLaVA-1.5 runs
- (If attempting §9) request Google Earth Engine access early — approval can take time

### Phase 1 — Fine-tuning (highest priority — Hour 0–14)
- BigEarthNet pseudo-captions; LoRA fine-tune on pseudo-captions + VRSBench train split
- Train optical-SAR fusion classifier on BigEarthNet-MM
- Train change-VQA head on CDVQA train split
- **Protect this phase's time budget above all else**

### Phase 2 — Input validation layer (Hour 14–18)
- GeoTIFF reader, CRS check, co-registration check, modality/count detection

### Phase 3 — Agentic controller, mandatory single-task flow (Hour 18–24)
- Query parser → structured JSON, closed-set task classifier, tool registry wiring, execution trace logger

### Phase 4 — Output integration (Hour 24–28)
- Confidence aggregation, visual evidence rendering, PDF report generator

### Phase 5 — GUI, mandatory flow (Hour 28–33)
- Upload + query interface, results display with trace panel and download button

### Phase 6 — Evaluation against benchmarks (Hour 33–37)
- Run against official test splits, fix obvious failure modes
- **At this point the mandatory core is complete and demoable — everything below is stretch**

### Phase 7 (stretch) — Dynamic layer: query planner + session memory (Hour 37–41)
- Multi-task JSON decomposition, reasoning/synthesis pass, in-memory session store

### Phase 8 (stretch) — Wide-area mode (Hour 41–45)
- Nominatim geocoding, GEE/Copernicus fetch → GeoTIFF export, trend-mode pairwise chaining + chart

### Phase 9 — Polish + demo prep (Hour 45–48)
- Rehearse the PS's representative queries (§14) plus one wide-area example if built
- Prepare 2–3 backup examples in case live inference or live fetch is slow

---

## 13. Known risks — be upfront about these in your pitch

- **Domain gap**: BigEarthNet/VRSBench/CDVQA are built on European Sentinel/aerial imagery; the hidden ISRO/SAC test set uses Indian Cartosat-2S optical + RISAT SAR data. Say so proactively; propose ISRO Bhuvan/VEDAS data as a mitigation path.
- **Compute-constrained fine-tuning**: LoRA on free-tier GPUs means training on a subset — be transparent about sample sizes used.
- **SAR interpretation is genuinely hard**: budget real time for the fusion branch, don't treat it as an afterthought.
- **Routing reliability**: a wrong tool selection cascades into a wrong answer — deterministic/closed-set routing (§8) matters more than a flashy conversational controller.
- **Wide-area mode is a stretch, not a requirement**: if §9 isn't finished, don't let it delay or destabilize the mandatory core — it's additive, and the compliance table in §4 doesn't depend on it.
- **Live external API calls in a demo are a single point of failure**: if using GEE/Copernicus live during judging, pre-fetch and cache your planned demo region(s) as a fallback in case of rate limits or network issues.

---

## 14. Suggested team role split (5–6 members)

| Role | Responsibility |
|---|---|
| VLM fine-tuning lead | LoRA fine-tuning on BigEarthNet + VRSBench for VQA/captioning/grounding |
| Change-detection lead | Siamese diff architecture + CDVQA fine-tuning |
| Optical-SAR fusion lead | Dual-branch classifier on BigEarthNet-MM |
| Agentic backend lead | FastAPI, controller logic, routing, execution trace, input validation, query planner |
| Frontend/GUI lead | React/Streamlit interface, overlay rendering, trend chart, report download |
| Data/eval lead | Dataset prep, benchmark evaluation, domain-gap documentation, wide-area data fetch |

---

## 15. Demo script

### Mandatory-core queries (from the PS itself — rehearse these first)

1. *"Describe the land-cover and major objects visible in this image."* → single-image captioning
2. *"Highlight the water body referred to in the query."* → grounding
3. *"What changed between these two dates, and where did the change occur?"* → change-VQA
4. *"Use the optical and SAR images together to identify built-up and water-covered regions."* → optical-SAR fusion
5. *"Has the built-up area increased, decreased, or remained unchanged?"* → change-VQA

> "Watch — I upload this optical and SAR pair of the same region and ask which areas are built-up versus water. The controller checks that both images are valid and co-registered, routes the query to our fusion model — trained on BigEarthNet's paired Sentinel-1/2 data — and returns the answer with a confidence score and a highlighted overlay. Here's the execution log showing exactly which model made this call, and here's the downloadable report."

### Bonus wide-area query (if §9 is built)

> "Now — without uploading anything — I'll just type a place name and ask how its water extent has changed over the last decade. The planner breaks that into nine year-over-year comparisons, fetches each year's satellite image itself, runs the same change-detection model nine times, and gives me one trend chart and one summary — with all nine steps logged in the execution trace."

---

*This document is a working build guide for SIH26167 — SatQuery AI. §1–8 is the graded, must-ship core. §9–10 is the differentiator layer — build it only once §1–8 is solid and demoable.*
