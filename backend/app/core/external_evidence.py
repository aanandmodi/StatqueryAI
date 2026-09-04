"""Opt-in, bounded, no-token context discovery. Never auto-download or bill imagery."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import date
from urllib.parse import quote, urlsplit

import httpx
from pydantic import Field, model_validator

from app.errors import ModelUnavailableError
from app.schemas import StrictModel


class HistorySearch(StrictModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    acquisition_date: date
    start_date: date
    end_date: date
    max_cloud_cover: float = Field(default=30, ge=0, le=100, allow_inf_nan=False)
    limit: int = Field(default=8, ge=1, le=20)
    footprint_confirmed: bool

    @model_validator(mode="after")
    def validate_scope(self):
        w, s, e, n = self.bbox
        if not all(math.isfinite(x) for x in self.bbox) or not (
            -180 <= w < e <= 180 and -90 <= s < n <= 90
        ):
            raise ValueError(
                "Use a valid west,south,east,north WGS84 bounding box (no antimeridian crossing)"
            )
        if e - w > 2 or n - s > 2:
            raise ValueError("Keep the search region within 2 degrees on each axis")
        if not self.footprint_confirmed:
            raise ValueError(
                "Confirm the mapped footprint; camera GPS alone is not an image footprint"
            )
        if not (
            date(2015, 1, 1)
            <= self.start_date
            <= self.end_date
            < self.acquisition_date
            <= date.today()
        ):
            raise ValueError(
                "Historical dates must precede the confirmed acquisition date "
                "and not be in the future"
            )
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("Search at most one year per request")
        return self


class WeatherSearch(StrictModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_scope(self):
        if not (date(1981, 1, 1) <= self.start_date <= self.end_date <= date.today()):
            raise ValueError("Choose historical dates from 1981 through today")
        if (self.end_date - self.start_date).days > 30:
            raise ValueError("Request at most 31 days of weather context")
        return self


_cache: dict[str, tuple[float, dict]] = {}
_slots = asyncio.Semaphore(2)


async def _fetch_json(key: str, method: str, url: str, **kwargs) -> dict:
    cached = _cache.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    try:
        async with (
            asyncio.timeout(30),
            _slots,
            httpx.AsyncClient(timeout=25, follow_redirects=False) as client,
            client.stream(method, url, **kwargs) as response,
        ):
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 2_000_000:
                    raise ValueError("Context provider response exceeds the safety limit")
        result = json.loads(
            data,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite provider value")),
        )
        if not isinstance(result, dict):
            raise ValueError("Unexpected context response")
    except (httpx.HTTPError, ValueError, TimeoutError) as exc:
        raise ModelUnavailableError(
            "Public evidence source is unavailable; no context was invented. Retry later."
        ) from exc
    if len(_cache) >= 64:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (time.monotonic() + 3600, result)
    return result


def _public_asset_url(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme == "https"
        and parsed.hostname
        in {
            "sentinel-cogs.s3.us-west-2.amazonaws.com",
            "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
        }
        and not parsed.username
        and not parsed.password
        and port in {None, 443}
    ):
        return value
    return None


def _invalid_provider(key: str):
    _cache.pop(key, None)
    raise ModelUnavailableError(
        "Public evidence source returned malformed data; no context was invented."
    )


async def search_history(payload: HistorySearch) -> dict:
    key = "history:" + payload.model_dump_json()
    response = await _fetch_json(
        key,
        "POST",
        "https://earth-search.aws.element84.com/v1/search",
        json={
            "collections": ["sentinel-2-l2a"],
            "bbox": payload.bbox,
            "datetime": f"{payload.start_date}T00:00:00Z/{payload.end_date}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lte": payload.max_cloud_cover}},
            "sortby": [{"field": "properties.datetime", "direction": "desc"}],
            "limit": payload.limit,
        },
    )
    items = []
    if not isinstance(response.get("features"), list):
        _invalid_provider(key)
    for feature in response.get("features", [])[: payload.limit]:
        if (
            not isinstance(feature, dict)
            or not isinstance(feature.get("properties"), dict)
            or not isinstance(feature.get("assets"), dict)
        ):
            _invalid_provider(key)
        props, assets = feature.get("properties", {}), feature.get("assets", {})
        cloud = props.get("eo:cloud_cover")
        if cloud is not None and (
            not isinstance(cloud, int | float) or not math.isfinite(cloud) or not 0 <= cloud <= 100
        ):
            _invalid_provider(key)
        if not all(isinstance(value, dict) for value in assets.values()):
            _invalid_provider(key)
        scene_id = str(feature.get("id", ""))[:160]
        items.append(
            {
                "id": scene_id,
                "acquired_at": props.get("datetime"),
                "bbox": feature.get("bbox"),
                "cloud_cover_percent": props.get("eo:cloud_cover"),
                "platform": props.get("platform"),
                "comparison_ready": False,
                "source_url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"
                + quote(scene_id, safe=""),
                "assets": {
                    name: url
                    for name in ("visual", "red", "green", "blue", "nir", "swir16", "swir22", "scl")
                    if (url := _public_asset_url(assets.get(name, {}).get("href")))
                },
            }
        )
    return {
        "query": payload.model_dump(mode="json"),
        "items": items,
        "source": "Earth Search / Copernicus Sentinel-2 L2A",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warnings": [
            "Discovery only: candidates have not been downloaded, cropped, "
            "cloud-masked or registered.",
            "Scene-wide cloud cover is not cloud cover inside your region. "
            "Dates may contain acquisition gaps.",
            "A satellite scene and an oblique phone photo are not a pixel-aligned temporal pair.",
        ],
    }


async def search_weather(payload: WeatherSearch) -> dict:
    key = "weather:" + payload.model_dump_json()
    response = await _fetch_json(
        key,
        "GET",
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params={
            "parameters": "T2M,PRECTOTCORR,RH2M,WS10M",
            "community": "AG",
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "start": payload.start_date.strftime("%Y%m%d"),
            "end": payload.end_date.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        },
    )
    if not all(
        isinstance(response.get(name), dict) for name in ("properties", "parameters", "header")
    ):
        _invalid_provider(key)
    parameters = response["properties"].get("parameter")
    if not isinstance(parameters, dict) or not all(
        isinstance(values, dict) for values in parameters.values()
    ):
        _invalid_provider(key)
    if not all(
        isinstance(value, int | float) and math.isfinite(value)
        for values in parameters.values()
        for value in values.values()
    ):
        _invalid_provider(key)
    dates = sorted({day for values in parameters.values() for day in values})
    fill = response.get("header", {}).get("fill_value", -999)
    return {
        "query": payload.model_dump(mode="json"),
        "source": "NASA POWER / MERRA-2 regional reanalysis",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_url": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
        "parameters": response.get("parameters", {}),
        "days": [
            {
                "date": day,
                **{
                    name: (values.get(day) if values.get(day) != fill else None)
                    for name, values in parameters.items()
                },
            }
            for day in dates
        ],
        "warnings": [
            "Coarse regional reanalysis, not weather measured at the exact image pixel. "
            "Missing values remain missing.",
            "Rainfall context does not establish flood causation. Mountain valleys may "
            "differ substantially from the regional grid.",
        ],
    }
