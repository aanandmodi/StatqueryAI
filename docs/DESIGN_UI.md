# UI and visual design specification

## Current studio revision — 2026-09-04

<<<<<<< HEAD
Refinement: local Bahnschrift/Aptos/Segoe UI typography, calmer heading tracking, 12px panel and
8px control tokens, subtle surface depth, and balanced headings. The charcoal/mint identity and
untinted source imagery remain. Methods now distinguishes learned planning, deterministic fallback,
candidate extent measurement and sensor-specific limitations. No remote font service is needed
for the active studio font stack. Build/type checks cover this update; no new browser visual QA
or claim of pixel-perfect rendering is made in this pass.

=======
>>>>>>> 2f620623f8897788bd2df2ce4f5700cb183d84f8
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
