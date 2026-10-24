#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

FACTOR = 5
NODATA = -32768
INDICES = ["NBR", "NDVI", "NDMI"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Block-average monthly index composites to a coarser grid"
    )
    ap.add_argument("--data", type=Path, default=Path("s2_data"))
    args = ap.parse_args()

    files = sorted((args.data / "monthly").glob("S2_????-??.tif"))
    if not files:
        raise SystemExit(f"no monthly files in {args.data / 'monthly'}")
    out = args.data / "reduced"
    out.mkdir(exist_ok=True)
    months = [f.stem[3:] for f in files]

    with rasterio.open(files[0]) as src:
        h, w = src.height // FACTOR, src.width // FACTOR
        win = Window(0, 0, w * FACTOR, h * FACTOR)
        transform = src.transform * rasterio.Affine.scale(FACTOR)
        crs = src.crs

    prof = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": len(files),
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "DEFLATE",
    }
    dst = {
        k: rasterio.open(
            out / f"{k}_100m.tif.part",
            "w",
            dtype="int16",
            nodata=NODATA,
            predictor=2,
            **prof,
        )
        for k in INDICES
    }
    dst["n_clear"] = rasterio.open(
        out / "nclear_100m.tif.part", "w", dtype="uint8", **prof
    )

    for b, (f, m) in enumerate(zip(files, months), start=1):
        with rasterio.open(f) as src:
            arr = src.read(
                window=win,
                out_shape=(src.count, h, w),
                resampling=Resampling.average,
                masked=True,
            )
        for i, k in enumerate(INDICES):
            dst[k].write(np.ma.filled(arr[i].round(), NODATA).astype("int16"), b)
            dst[k].set_band_description(b, m)
        nc = np.ma.filled(arr[3], 0).round().clip(0, 254).astype("uint8")
        dst["n_clear"].write(nc, b)
        dst["n_clear"].set_band_description(b, m)
        print(f"{m} done ({b}/{len(files)})", flush=True)

    for ds in dst.values():
        name = ds.name
        ds.close()
        Path(name).replace(name.removesuffix(".part"))

    with open(out / "months.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["band", "month"])
        wr.writerows(enumerate(months, start=1))
    print(f"written: {out} ({h} x {w} pixels of 100 m, {len(files)} months)")


if __name__ == "__main__":
    main()
