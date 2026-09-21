# UI and visual design specification

## Orbital intelligence workspace — 2026-09-20

The active investigation surface is a full-viewport three-column workspace derived from the
approved visual concept: a compact midnight command rail, an evidence/input column, a large
source-locked scene canvas and a persistent editorial intelligence brief. The interface uses a
deep-ocean, warm-paper, lime, aqua, amber and coral palette with restrained contour lines,
high-contrast scientific typography and bounded reveal motion. It does not use decorative stock
imagery or simulated map content.

### Real-data contract

| Surface | Permitted source | Empty or unavailable behavior |
|---|---|---|
| Scene canvas | The user's selected uploaded asset | Displays an explicit awaiting-source state |
| Overlay and legend | Returned evidence geometry and class labels | No boundary, mask or class is invented |
| Dimensions and CRS | Validated asset metadata | Displays `not supplied` or `unavailable` |
| Area and coverage | Numeric evidence properties returned by the controller | Metric card is omitted |
| Confidence | Backend score plus its declared score semantics | Never relabelled as calibrated probability |
| Intelligence brief | Backend answer, sections, facts and warnings | Displays a truthful no-result state |
| Provenance | Result provenance and revision identifiers | Missing fields remain visibly unavailable |
| Trace | Recorded validation, routing, model and integration events | No simulated progress or timing rows |

The scene and report are visible together so a user can compare every textual claim with the
actual uploaded source and returned spatial evidence. For paired inputs, the canvas offers a real
before/after split using only the two selected assets. Zoom, overlay opacity, source selection and
fullscreen controls modify presentation only; they do not alter evidence.

The supporting Casebook, Archive and Methods routes share the same command-shell identity. Their
tables and cards continue to render stored or fetched records only; loading, unavailable and empty
states never substitute demo rows.

### Unified navigation and case dossier

- A single reusable left command rail now serves Investigation, Casebook, Historical Evidence and
  Methods. On narrow screens it becomes a bottom navigation bar without creating a second menu.
- Casebook adds client-side search and status filtering over the controller records already loaded
  for the current page. It does not generate or insert example cases.
- Case detail is an evidence dossier rather than a raw report page: source-locked imagery and mask
  controls, the returned intelligence brief, structured report sections, facts, warnings, trace and
  provenance are visually separated but remain in one audit flow.
- Recharts visualizations are conditional. Candidate-coverage charts use returned
  `coverage_percent`; factor charts use returned confidence factors; timing charts use trace
  `duration_ms`. When none exists, the page says that no quantitative evidence was returned.
- Body copy and metadata contrast were increased across all routes. Editorial headings remain
  visually distinct, while scientific limitations and source identifiers are darker and easier to
  scan.

### Analysis workspace interaction pass

- The Evidence desk now exposes a visible Configure → Attach → Interpret workflow state without
  simulating backend progress.
- Native-looking flat selectors were replaced with accessible Base UI menus. Each option includes
  a plain-language title and contract summary; disabled routes continue to come from the real
  capability response and input compatibility rules.
- The question composer includes task-specific presets and a descriptive empty placeholder. File
  surfaces, focus states, source selection and the run action have clearer hover/active feedback.
- Low-contrast contour lines, orbital rings and grid textures make the surrounding workspace feel
  cartographic. They remain outside the source canvas and cannot be interpreted as uploaded pixels,
  returned masks or geographic evidence.

## Current studio revision — 2026-09-04

Refinement: local Bahnschrift/Aptos/Segoe UI typography, calmer heading tracking, 12px panel and
8px control tokens, subtle surface depth, and balanced headings. The charcoal/mint identity and
untinted source imagery remain. Methods now distinguishes learned planning, deterministic fallback,
candidate extent measurement and sensor-specific limitations. No remote font service is needed
for the active studio font stack. Build/type checks cover this update; no new browser visual QA
or claim of pixel-perfect rendering is made in this pass.

The frosted direction below is retained as historical context and is superseded by the current
instrument-style studio. `app/studio.css` overrides the base theme with charcoal surfaces,
mint/cyan evidence, amber warnings, readable sans-serif and compact mono metadata. Source imagery
is never decoratively tinted. The Sites skill informed local layout/interaction work only;
hosting was deliberately excluded by the user's instruction.

| Page/surface | Current interaction |
|---|---|
| Investigation | Evidence profiles, role checks, editable Water/Forestry/Journalism/Change presets |
| Scene inspector | Uncropped originals, source selection, mask toggle and opacity |
| Temporal result | Two source-locked panes; repeated masks not double-counted |
| Report / trace | Real result sections and actual task timings, no simulated progress |
| Casebook / case detail | Persistent cases, JSON/PDF/marked-image downloads |
| Archive | Working sourced historical/weather searches, explicit preparation limitations |
| Methods | Capability and scientific-limit disclosure |

File selection uses a native chooser; the historical drag/drop requirement below is not
implemented. Layouts adapt to narrower screens; motion respects reduced-motion preferences.
Desktop browser checks covered investigation, pair panes, reports, casebook and Archive. Full
keyboard/screen-reader/device-matrix testing remains acceptance work, not a claimed certification.

## Historical design specification (superseded)

## Direction

SatQuery should feel like a calm field instrument: atmospheric, cartographic and precise. The
visual language combines restrained glassmorphism and liquid highlights with frost-blue surfaces,
ink text, indigo/cyan signal color, and small topographic line details. It must not resemble a
generic AI chat template or a collection of glowing dashboard cards.

## Experience hierarchy

1. Scene canvas and evidence overlay are the visual anchor.
2. Natural-language question and task status are the primary action.
3. Answer, facts, warnings and evidence appear together.
4. Execution trace and report are inspectable secondary material.
5. Model versions and dataset provenance are available without dominating the interface.

## Component behavior

- Upload surface supports drag/drop and keyboard selection; each file shows modality, role,
  dimensions, bands, CRS and validation state.
- Pair tasks make time A/time B or optical/SAR roles unmistakable.
- The source-asset selector keeps preview and evidence on the same asset; previews preserve
  their aspect ratio without cropping. Marked-image downloads use that same selected asset.
- Marked-image artifacts append a readable legend/answer footer; text never covers the scene.
- Evidence boxes/polygons use class-specific color, clear focus states and a linked legend.
- Location context presents bounded latitude, longitude, altitude and sensor fields with provenance copy.
- Loading shows actual phases: validating, planning, waiting on Kaggle GPU, running, integrating.
- Quota/cold-start errors give a retry path and never imply the analysis completed.
- Confidence language uses `calibrated probability`, `evidence quality`, or `uncalibrated estimate`.
- The trace lists tool, task, version, permitted parameters, duration and status.
- Readiness parses dependency JSON, distinguishes degraded/offline, and refreshes every 20 seconds.
- Structured specialist facts are expandable; model score semantics remain visible.

## Palette and materials

- Canvas: cold off-white / deep ink, not flat pure white.
- Primary: indigo-violet; active evidence: cyan; success: moss; warning: amber; error: coral.
- Glass panels use subtle blur, one-pixel light borders and directional highlights.
- Liquid effects are confined to transitions and active controls; content text remains still.
- Cartographic doodles are sparse contour lines, coordinate ticks and orbital arcs—not novelty icons.

## Motion

- 160–260 ms transitions with spring only for evidence reveal and panel expansion.
- Progress animation follows real backend events.
- No perpetual floating cards, typing simulation or gratuitous gradient rotation.
- `prefers-reduced-motion` removes spatial motion and retains opacity/state changes.

## Accessibility

- WCAG AA contrast for body text and controls.
- Visible keyboard focus and logical reading/tab order.
- All canvas evidence has a text equivalent in the evidence list.
- Never communicate task/status using color alone.
- Touch targets are at least 44 px; error text is actionable and associated with its input.

## Analyze workspace empty state and score charts — 2026-09-20

- The no-source Evidence Canvas is an explicit acquisition state rather than a blank map. It uses
  orbital registration marks, a restrained scan field, supported-mode labels, and one working
  `Choose evidence` action that opens the existing local file selector.
- Decorative scanner elements never imply that a scene, location, region, or model result exists.
  The canvas continues to state `No source loaded · no invented map` until validation succeeds.
- The adjacent Intelligence brief explains the real three-stage flow—validate, route, report—so
  the empty workspace remains useful without inserting sample output or fabricated confidence.
- Confidence-factor donut slices now use a stable categorical palette (teal, amber, blue, coral,
  violet, green) with a matching legend. Colors distinguish returned factors; the values still
  come only from the stored controller response.
- The evidence, answer, timeline, and trace surfaces are visually separated as instrument panels
  over a subtle cartographic background. Contrast, reduced-motion behavior, and narrow-screen
  fallbacks are preserved.
