"""Product metadata recognition. No filename, resolution or band-count sensor guesses.

This dependency-free module is mirrored into the ML package and Kaggle patch by
scripts/sync-expert-notebooks.py. Embedded metadata is a declaration, not certification.
"""

from __future__ import annotations

import re


def compact(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def sensor_profile(source):
    tags = dict(source.tags())
    for namespace in source.tag_namespaces()[:8]:
        if namespace not in {"IMAGE_STRUCTURE", "DERIVED_SUBDATASETS"}:
            tags.update(dict(list(source.tags(ns=namespace).items())[:60]))
    normalized = {compact(key): str(value).strip() for key, value in tags.items()}
    names = [
        normalized[key]
        for key in ("satid", "satellite", "platform", "satellitename")
        if key in normalized
    ]
    platforms = set()
    for name in names:
        value = compact(name)
        if value in {"eos04", "risat1a"}:
            platforms.add("eos-04")
        elif value == "risat1":
            platforms.add("risat-1")
        elif value in {"cartosat2s", "cartosat2e", "cartosat2f", "c2s", "c2e", "c2f"}:
            platforms.add("cartosat-2-series")
        elif value in {"sentinel2", "sentinel2a", "sentinel2b", "sentinel2c", "s2a", "s2b", "s2c"}:
            platforms.add("sentinel-2")
        elif value in {"sentinel1", "sentinel1a", "sentinel1b", "sentinel1c", "s1a", "s1b", "s1c"}:
            platforms.add("sentinel-1")
    if len(platforms) > 1:
        raise ValueError("Conflicting embedded platform declarations")
    platform = next(iter(platforms), "unknown")
    numeric = (
        {"b1": "blue", "b2": "green", "b3": "red", "b4": "nir"}
        if platform == "cartosat-2-series"
        else {"b2": "blue", "b3": "green", "b4": "red", "b8": "nir"}
        if platform == "sentinel-2"
        else {}
    )
    semantic = {
        "red": "red",
        "green": "green",
        "blue": "blue",
        "nir": "nir",
        "nearinfrared": "nir",
        "pan": "pan",
        "panchromatic": "pan",
    }
    bands = []
    for index, description in enumerate(source.descriptions, 1):
        band_tags = {
            compact(key): str(value) for key, value in list(source.tags(index).items())[:40]
        }
        declarations = [
            description or "",
            band_tags.get("bandname", ""),
            band_tags.get("description", ""),
        ]
        meanings, pols = set(), set()
        for declaration in declarations:
            value = compact(declaration)
            numeric_name = re.sub(r"^b0+", "b", value)
            meaning = semantic.get(value) or numeric.get(numeric_name)
            if meaning:
                meanings.add(meaning)
            if value.upper() in {"HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"}:
                pols.add(value.upper())
        color = source.colorinterp[index - 1].name
        if color in {"red", "green", "blue"}:
            meanings.add(color)
        pol = normalized.get(f"txrxpol{index}", band_tags.get("polarization", "")).upper()
        if pol in {"HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"}:
            pols.add(pol)
        if len(meanings) > 1 or len(pols) > 1:
            raise ValueError(f"Conflicting band declarations at index {index}")
        bands.append(
            {
                "index": index,
                "description": str(description or "")[:160],
                "meaning": next(iter(meanings), "unknown"),
                "polarization": next(iter(pols), None),
                "color": color,
                "unit": source.units[index - 1],
                "scale": source.scales[index - 1],
                "offset": source.offsets[index - 1],
            }
        )
    known = [band["meaning"] for band in bands if band["meaning"] != "unknown"]
    if len(known) != len(set(known)):
        raise ValueError("Duplicate spectral band meanings")
    return {
        "platform": platform,
        "source": "embedded_product_metadata",
        "sensor": normalized.get("sensor", "unknown"),
        "product_type": normalized.get("producttype", "unknown"),
        "imaging_mode": normalized.get("imagingmode", "unknown"),
        "representation": normalized.get("representation", "unknown"),
        "rtc_applied": {"0": False, "1": True}.get(normalized.get("rtcapplyflag")),
        "bands": bands,
        "warning": "Declared metadata only; raster spacing does not establish native resolution.",
    }


def semantic_indexes(source, meanings):
    bands = sensor_profile(source)["bands"]
    result = []
    for meaning in meanings:
        matches = [band["index"] for band in bands if band["meaning"] == meaning]
        if len(matches) != 1:
            return None
        result.append(matches[0])
    return result


def visual_indexes(source):
    return semantic_indexes(source, ["red", "green", "blue"]) or (
        [1, 2, 3] if source.count >= 3 else [1, 1, 1]
    )


def sentinel_fusion_indexes(optical, sar):
    """TerraMind's training channels are not interchangeable with RISAT/Cartosat."""
    s2, s1 = sensor_profile(optical), sensor_profile(sar)
    if s2["platform"] != "sentinel-2" or s1["platform"] != "sentinel-1":
        raise ValueError(
            "Fusion requires Sentinel-2 and Sentinel-1; ISRO transfer is unvalidated"
        )
    order = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    descriptions = [str(item or "").upper().strip() for item in optical.descriptions]
    if any(descriptions.count(name) != 1 for name in order):
        raise ValueError("Explicit ordered Sentinel-2 band names required")
    pols = [band["polarization"] for band in s1["bands"]]
    if any(pols.count(pol) != 1 for pol in ["VV", "VH"]):
        raise ValueError("This expert requires VV/VH; RH/RV or HH/HV cannot substitute")
    if s1["representation"].lower() not in {"sigma0_db", "sigma0db"}:
        raise ValueError("Calibrated sigma0 in dB must be declared; raw amplitude is unsupported")
    if s2["representation"].lower() != "surface_reflectance_10000":
        raise ValueError("S2 L2A reflectance scaled by 10000 must be declared")
    return [descriptions.index(name) + 1 for name in order], [
        pols.index(pol) + 1 for pol in ["VV", "VH"]
    ]
