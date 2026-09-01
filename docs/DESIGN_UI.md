# UI and visual design specification

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
- Evidence boxes/polygons use class-specific color, clear focus states and a linked legend.
- Loading shows actual phases: validating, planning, queued for ZeroGPU, running, integrating.
- Quota/cold-start errors give a retry path and never imply the analysis completed.
- Confidence language uses `calibrated probability`, `evidence quality`, or `uncalibrated estimate`.
- The trace lists tool, task, version, permitted parameters, duration and status.

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

