"""Compare a binary prediction with a manually labelled, same-grid reference; no model needed.

Usage: python scripts/evaluate-water-mask.py prediction.png reference.png
Inputs must already be aligned, same dimensions, and binary 0/1 or 0/255. No automatic resizing.
This evaluates one image; use geographically held-out scenes for a real validation study.
"""

import argparse
import json

import numpy as np
from PIL import Image


def mask_metrics(prediction, reference):
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError(
            "Prediction and reference must be aligned 2D masks with identical dimensions"
        )
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    tp = int((prediction & reference).sum())
    fp = int((prediction & ~reference).sum())
    fn = int((~prediction & reference).sum())
    tn = int((~prediction & ~reference).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "note": "Null means undefined (for example no positive reference/prediction); not perfect accuracy.",
    }


def read_binary(path):
    with Image.open(path) as image:
        if image.mode not in {"1", "L"}:
            raise ValueError("Use a binary grayscale mask, not a colored overlay")
        values = np.asarray(image.convert("L"))
    if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
        raise ValueError("Expected binary 0/1 or 0/255 mask")
    return values > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction")
    parser.add_argument("reference")
    args = parser.parse_args()
    print(
        json.dumps(
            mask_metrics(read_binary(args.prediction), read_binary(args.reference)),
            indent=2,
        )
    )
