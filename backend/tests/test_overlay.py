from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image, ImageFont

from app import api
from app.errors import ValidationFailure
from app.schemas import EvidenceItem


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_id", [None, "ast_a", "ast_b"])
async def test_overlay_uses_only_selected_assets_evidence(
    monkeypatch, make_asset, selected_id: str | None
):
    evidence = [
        EvidenceItem(
            id=f"ev_{asset_id}",
            type="box",
            label=f"Evidence on {asset_id}",
            score=0.5,
            coordinate_space="normalized",
            geometry={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            asset_id=asset_id,
        )
        for asset_id in ["ast_b", "ast_a"]
    ]
    record = SimpleNamespace(
        request=SimpleNamespace(asset_ids=["ast_a", "ast_b"], context=None),
        result=SimpleNamespace(evidence=evidence, answer="Paired analysis"),
    )
    state = SimpleNamespace(
        repository=SimpleNamespace(
            get_analysis=AsyncMock(return_value=record),
            get_assets=AsyncMock(return_value=[make_asset("ast_a"), make_asset("ast_b")]),
        ),
        asset_store=SimpleNamespace(resolve=lambda identifier: Path(identifier)),
        settings=SimpleNamespace(space_preview_max_edge=512, space_preview_jpeg_quality=90),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=state)))
    render = Mock(return_value=b"preview")
    draw = Mock(return_value=b"marked image")
    monkeypatch.setattr(api, "render_rgb_preview", render)
    monkeypatch.setattr(api, "_draw_overlay", draw)

    response = await api.download_overlay("anl_abc", request, selected_id)

    expected_id = selected_id or "ast_b"
    assert response.body == b"marked image"
    assert render.call_args.args[0] == Path(expected_id)
    drawn_evidence = draw.call_args.args[1]
    assert len(drawn_evidence) == 1
    assert drawn_evidence[0].asset_id == expected_id


@pytest.mark.asyncio
async def test_overlay_rejects_asset_outside_analysis():
    repository = SimpleNamespace(
        get_analysis=AsyncMock(
            return_value=SimpleNamespace(
                request=SimpleNamespace(asset_ids=["ast_a"]), result=SimpleNamespace()
            )
        ),
        get_assets=AsyncMock(),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(repository=repository)))
    )
    with pytest.raises(ValidationFailure, match="does not belong"):
        await api.download_overlay("anl_abc", request, "ast_b")
    repository.get_assets.assert_not_called()


def _preview(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, (60, 120, 180))
    image.putpixel((0, 0), (80, 140, 200))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _assert_color(actual, expected, tolerance=8):
    assert all(abs(int(a) - int(e)) <= tolerance for a, e in zip(actual, expected, strict=True))


def test_tiny_overlay_appends_footer_without_hiding_scene(tmp_path: Path):
    marked = api._draw_overlay(
        _preview((128, 96)),
        [],
        context=SimpleNamespace(latitude=28.6139, longitude=77.209, altitude_m=216),
        answer="The two images contain a spectral-change proxy. " * 12,
    )
    (tmp_path / "tiny-appended-footer.jpg").write_bytes(marked)
    with Image.open(io.BytesIO(marked)) as image:
        # 128:96 becomes 768:576, then a separate footer is appended below it.
        assert image.width == 768
        assert 576 < image.height < 1200
        _assert_color(image.getpixel((300, 30)), (60, 120, 180))
        _assert_color(image.getpixel((300, 540)), (60, 120, 180))
        _assert_color(image.getpixel((10, 585)), (9, 22, 30))


def test_portrait_overlay_preserves_scene_and_box_coordinates(tmp_path: Path):
    evidence = EvidenceItem(
        id="ev_portrait",
        asset_id="ast_a",
        type="box",
        label="A very long model evidence label " * 40,
        score=0.5,
        coordinate_space="normalized",
        geometry={"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
    )
    marked = api._draw_overlay(
        _preview((64, 256)), [evidence], answer="Portrait source remains fully visible."
    )
    (tmp_path / "portrait-appended-footer.jpg").write_bytes(marked)
    with Image.open(io.BytesIO(marked)) as image:
        # 64:256 -> 192:768, centered in 640px canvas: left offset 224px.
        assert image.width == 640
        assert 768 < image.height < 1200
        _assert_color(image.getpixel((10, 400)), (9, 22, 30))
        _assert_color(image.getpixel((320, 30)), (60, 120, 180))
        _assert_color(image.getpixel((320, 735)), (60, 120, 180))
        _assert_color(image.getpixel((320, 400)), (60, 120, 180))
        # The top edge stays at normalized y=.25 of the SCENE, not artifact.
        _assert_color(image.getpixel((320, 192)), (109, 255, 197), tolerance=55)
        _assert_color(image.getpixel((320, 193)), (109, 255, 197), tolerance=55)


def test_overlay_text_is_measured_wrapped_and_explicitly_shortened():
    font = ImageFont.load_default(size=18)
    text = "An unusuallylongunbrokentoken" * 30
    lines, shortened = api._wrap_overlay_text(text, font, max_width=140, max_lines=3)
    assert shortened
    assert len(lines) == 3
    assert lines[-1].endswith("...")
    assert all(font.getlength(line) <= 140 for line in lines)
    label = api._fit_overlay_line("A label that cannot fit " * 30, font, max_width=140)
    assert label.endswith("...")
    assert font.getlength(label) <= 140
