#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

import rsconfig

CFG = rsconfig.load()

DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
    "Copernicus_DSM_COG_10_{t}_DEM.tif"
)
TILES = rsconfig.glo30_tiles(CFG["aoi_bbox"])
LAT, LON = CFG["climate_point"]
C0, C1 = CFG["climate_period"]
METEO = (
    f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
    f"&start_date={C0}&end_date={C1}"
    "&daily=precipitation_sum,et0_fao_evapotranspiration&models=era5&timezone=GMT"
)


def main() -> None:
    rec_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(rec_path) as rec:
        k, stage, sev = rec.read(1), rec.read(4), rec.read(8)
        crs, tr100 = rec.crs, rec.transform
        H, W = rec.shape

    f = 4
    tr25 = tr100 * rasterio.Affine.scale(1 / f)
    dem = np.full((H * f, W * f), np.nan, "float32")
    for t in TILES:
        with rasterio.open(DEM_URL.format(t=t)) as src:
            tmp = np.full_like(dem, np.nan)
            reproject(
                rasterio.band(src, 1),
                tmp,
                dst_transform=tr25,
                dst_crs=crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            dem = np.where(np.isfinite(tmp), tmp, dem)
    gy, gx = np.gradient(dem, 25.0)
    dzdx, dzdn = gx, -gy
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    ex, en = -dzdx, -dzdn
    norm = np.hypot(ex, en) + 1e-9

    def block(a: np.ndarray) -> np.ndarray:
        return np.nanmean(a.reshape(H, f, W, f), axis=(1, 3))

    elev = block(dem)
    slp = block(slope)
    asp = (np.degrees(np.arctan2(block(ex / norm), block(en / norm))) + 360) % 360

    m = np.isin(sev, [2, 3, 4, 5]) & (stage > 0)
    pd.DataFrame(
        {
            "k": np.round(k[m], 4),
            "stage": stage[m].astype(int),
            "severity": sev[m].astype(int),
            "elev": np.round(elev[m], 1),
            "slope": np.round(slp[m], 2),
            "aspect": np.round(asp[m], 1),
        }
    ).to_csv(out / "controls_pixels.csv", index=False)

    d = json.load(urllib.request.urlopen(METEO, timeout=120))["daily"]
    c = pd.DataFrame(
        {"P": d["precipitation_sum"], "ET0": d["et0_fao_evapotranspiration"]},
        index=pd.to_datetime(d["time"]),
    )
    mon = c.resample("MS").sum(min_count=25)
    mon["D"] = mon["P"] - mon["ET0"]
    mon.round(1).to_csv(out / "climate_monthly.csv", index_label="month")
    print(m.sum(), "pixels;", mon.dropna().index[[0, -1]].tolist())


if __name__ == "__main__":
    main()
