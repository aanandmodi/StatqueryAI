"""Create synthetic GeoTIFFs for plumbing checks, never benchmark/model-quality evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic-smoke"))
    args = parser.parse_args()
    names = ("single", "time-a", "time-b", "optical", "sar")
    paths = {name: args.output / f"{name}.tif" for name in names}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Choose a new --output directory; existing fixtures are preserved.")
    args.output.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:96, :128]
    optical = np.stack([60 + xx, 70 + yy, 110 + (127 - xx) // 2]).astype(np.uint16)
    changed = optical.copy()
    changed[:, 30:68, 44:88] += 160
    sar = (15 + xx // 2 + yy)[None].astype(np.uint16)
    sar[:, 10:40, 5:32] = 8
    sar[:, 50:85, 85:115] = 240
    for name in names:
        data = sar if name == "sar" else changed if name == "time-b" else optical
        with rasterio.open(
            paths[name], "w", driver="GTiff", width=128, height=96, count=data.shape[0],
            dtype="uint16", crs="EPSG:32644", transform=from_origin(500000, 3200000, 10, 10),
            nodata=0,
        ) as dataset:
            dataset.write(data)
            if name != "sar":
                for index, description in enumerate(("red", "green", "blue"), start=1):
                    dataset.set_band_description(index, description)
            dataset.update_tags(
                synthetic="true", purpose="transport and routing test; not real satellite imagery"
            )
    print(f"SYNTHETIC fixtures written to {args.output.resolve()}; not accuracy evidence.")


if __name__ == "__main__":
    main()
