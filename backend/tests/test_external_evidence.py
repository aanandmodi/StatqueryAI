from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.core import external_evidence as external
from app.errors import ModelUnavailableError


def history(**changes):
    return external.HistorySearch(
        **{
            "bbox": [85.2, 27.5, 85.5, 27.8],
            "acquisition_date": "2024-09-28",
            "start_date": "2024-08-01",
            "end_date": "2024-09-27",
            "footprint_confirmed": True,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "change",
    [
        {"footprint_confirmed": False},
        {"bbox": [0, 0, 30, 40]},
        {"bbox": [0, 0, float("nan"), 1]},
        {"end_date": "2024-09-29"},
        {"start_date": "2020-01-01"},
        {"limit": 21},
    ],
)
def test_history_scope_rejected(change):
    with pytest.raises(ValidationError):
        history(**change)


@pytest.mark.parametrize(
    "url",
    [
        "file:///secret",
        "http://127.0.0.1",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com.evil.test/x",
        "https://sentinel-cogs.s3.us-west-2.amazonaws.com:wrong/x",
        "https://[broken",
        None,
    ],
)
def test_asset_urls_fail_closed(url):
    assert external._public_asset_url(url) is None


@pytest.mark.asyncio
async def test_history_candidates_are_not_comparison_ready(monkeypatch):
    fetch = AsyncMock(
        return_value={
            "features": [
                {
                    "id": "scene",
                    "properties": {"datetime": "2024-09-20", "eo:cloud_cover": 2.5},
                    "assets": {
                        "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/x.tif"},
                        "visual": {"href": "https://evil.test/x"},
                    },
                }
            ]
        }
    )
    monkeypatch.setattr(external, "_fetch_json", fetch)
    output = await external.search_history(history())
    assert output["items"][0]["comparison_ready"] is False
    assert list(output["items"][0]["assets"]) == ["red"]
    assert fetch.call_args.args[2] == "https://earth-search.aws.element84.com/v1/search"
    assert fetch.call_args.kwargs["json"]["collections"] == ["sentinel-2-l2a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"features": None},
        {"features": [{"properties": None}]},
        {"features": [{"properties": {}, "assets": {"x": None}}]},
    ],
)
async def test_malformed_history_returns_actionable_error(monkeypatch, response):
    monkeypatch.setattr(external, "_fetch_json", AsyncMock(return_value=response))
    with pytest.raises(ModelUnavailableError, match="malformed"):
        await external.search_history(history())


@pytest.mark.asyncio
async def test_weather_missing_values_preserved(monkeypatch):
    monkeypatch.setattr(
        external,
        "_fetch_json",
        AsyncMock(
            return_value={
                "properties": {"parameter": {"T2M": {"20240920": -999}}},
                "parameters": {"T2M": {"units": "C"}},
                "header": {"fill_value": -999},
            }
        ),
    )
    output = await external.search_weather(
        external.WeatherSearch(
            latitude=27.7, longitude=85.3, start_date="2024-09-20", end_date="2024-09-20"
        )
    )
    assert output["days"][0]["T2M"] is None
    assert "reanalysis" in output["source"]


@pytest.mark.asyncio
async def test_weather_malformed_contract(monkeypatch):
    monkeypatch.setattr(
        external,
        "_fetch_json",
        AsyncMock(
            return_value={
                "properties": {"parameter": {"T2M": None}},
                "parameters": {},
                "header": {},
            }
        ),
    )
    with pytest.raises(ModelUnavailableError):
        await external.search_weather(
            external.WeatherSearch(
                latitude=27.7, longitude=85.3, start_date="2024-09-20", end_date="2024-09-20"
            )
        )
