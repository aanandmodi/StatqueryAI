from __future__ import annotations

import base64
import binascii
import io
import json
import re
from typing import Any

from PIL import Image, UnidentifiedImageError


MAX_ENCODED_CHARS = 14_000_000
MAX_DECODED_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 25_000_000

_BOX_TAG = re.compile(
    r"<box>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*,"
    r"\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*</box>",
    re.IGNORECASE,
)
_QWEN_BOX = re.compile(
    r"<\|box_start\|>\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*,"
    r"\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>",
    re.IGNORECASE,
)
_JSON_BOX = re.compile(r'"(?:bbox|box)"\s*:\s*\[\s*([^\]]+)\]', re.IGNORECASE)


def decode_base64_image(value: str) -> Image.Image:
    encoded = str(value).strip()
    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
        except ValueError as exc:
            raise ValueError("invalid image data URL") from exc
        if ";base64" not in header.lower():
            raise ValueError("image data URL must use base64 encoding")
    if not encoded:
        raise ValueError("image_base64 is required")
    if len(encoded) > MAX_ENCODED_CHARS:
        raise ValueError("encoded image payload exceeds the allowed limit")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if len(payload) > MAX_DECODED_BYTES:
        raise ValueError("decoded image exceeds the 10 MB limit")

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("image exceeds the 25-megapixel limit")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_base64 does not contain a valid image") from exc


def encode_pil_image(image: Image.Image, *, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalise_box(values: list[float]) -> dict[str, float] | None:
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    scale = 1000.0 if max(abs(item) for item in values) > 1.0 else 1.0
    x1, y1, x2, y2 = (max(0.0, min(1.0, item / scale)) for item in values)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left <= 0 or bottom - top <= 0:
        return None
    return {
        "x": round(left, 6),
        "y": round(top, 6),
        "width": round(right - left, 6),
        "height": round(bottom - top, 6),
    }


def parse_grounding_boxes(text: str) -> list[dict[str, Any]]:
    candidates: list[list[float]] = []
    for pattern in (_BOX_TAG, _QWEN_BOX):
        for match in pattern.finditer(text):
            candidates.append([float(item) for item in match.groups()])

    for match in _JSON_BOX.finditer(text):
        try:
            values = json.loads(f"[{match.group(1)}]")
            if isinstance(values, list):
                candidates.append([float(item) for item in values])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    evidence: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for values in candidates:
        geometry = _normalise_box(values)
        if geometry is None:
            continue
        key = tuple(round(geometry[name], 6) for name in ("x", "y", "width", "height"))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "type": "box",
                "label": "model-grounded region",
                "score": 0.5,
                "coordinate_space": "normalized",
                "geometry": geometry,
            }
        )
    return evidence
