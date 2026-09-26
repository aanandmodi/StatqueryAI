"""CPU-only conservative guardrails. These are not learned/calibrated cloud detectors."""
from __future__ import annotations

import re
import numpy as np


def bright_obstruction_risk(rgb, valid):
    """Flag bright neutral surfaces: cloud/snow/roofs are ambiguous in RGB alone.

    Thresholds are engineering review triggers, not fitted confidence or cloud labels.
    Only use on uint8 RGB. Do not use percentile-stretched SAR or spectral DN.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*valid.shape, 3):
        raise ValueError("Obstruction screen requires aligned uint8 RGB")
    low, high = rgb.min(-1), rgb.max(-1)
    suspect = (low >= 210) & ((high.astype(float) - low) <= 30) & valid
    return float(suspect.sum() / max(1, valid.sum()))


def checked_change_answer(query, label):
    """Abstain for an incompatible answer type; never coerce a count into yes/no."""
    question = query.strip().lower()
    binary = bool(re.match(r"^(has|have|is|are|did|does|do|was|were)\b", question))
    count = bool(re.match(r"^(how many|what number)\b", question))
    label = str(label).strip()
    if binary and label.lower() not in {"yes", "no"}:
        return None
    if count and not re.fullmatch(r"\d+(?:_to_\d+|\+)?", label):
        return None
    return label


def mask_description(mask, support):
    """Describe the predicted grid only; no invented semantics, area, or cause."""
    mask = np.asarray(mask, dtype=bool) & support
    fraction = float(mask.sum() / max(1, support.sum()))
    regions = []
    h, w = mask.shape
    for name, rows, cols in (
        ("upper-left", slice(0, h//2), slice(0, w//2)),
        ("upper-right", slice(0, h//2), slice(w//2, w)),
        ("lower-left", slice(h//2, h), slice(0, w//2)),
        ("lower-right", slice(h//2, h), slice(w//2, w)),
    ):
        regions.append((int(mask[rows, cols].sum()), name))
    largest, region = max(regions)
    location = f"Most selected pixels are in the {region} image quadrant." if largest else "No pixels were selected; this does not prove absence."
    return f"The candidate mask selects {fraction:.2%} of shared-valid analysis pixels. {location}"
