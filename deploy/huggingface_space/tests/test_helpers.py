from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from satquery_space.contracts import build_task_prompt, validate_request
from satquery_space.inference import decode_base64_image, parse_grounding_boxes


def _image_payload() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (30, 70, 110)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_decode_base64_image() -> None:
    image = decode_base64_image(_image_payload())
    assert image.mode == "RGB"
    assert image.size == (4, 3)


def test_validate_request_is_bounded() -> None:
    task, question, tokens = validate_request("single_vqa", "Visible roads?", 999)
    assert (task, question, tokens) == ("single_vqa", "Visible roads?", 256)
    with pytest.raises(ValueError, match="Unsupported task"):
        validate_request("change_vqa", "What changed?", 80)


def test_grounding_parser_supports_qwen_and_tag_formats() -> None:
    evidence = parse_grounding_boxes(
        "one <box>(100,200),(500,700)</box> two "
        "<|box_start|>(0,0),(1000,1000)<|box_end|>"
    )
    assert len(evidence) == 2
    assert evidence[0]["geometry"] == {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.5}
    assert evidence[1]["geometry"] == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}


def test_prompt_forbids_invented_confidence() -> None:
    assert "confidence" in build_task_prompt("caption", "Describe this.").lower()

