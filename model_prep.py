#!/usr/bin/env python3

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from scipy.spatial import cKDTree

import rsconfig

CFG = rsconfig.load()

DEM_URL = (
    "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{t}_DEM/Copernicus_DSM_COG_10_{t}_DEM.tif"
)
TILES = rsconfig.glo30_tiles(CFG["aoi_bbox"])
GFC_URL = (
    "/vsicurl/https://storage.googleapis.com/earthenginepartners-hansen/"
    f"GFC-2024-v1.12/Hansen_GFC-2024-v1.12_treecover2000_{rsconfig.gfc_tile(CFG['aoi_bbox'])}.tif"
)
MONTHS = pd.period_range(*CFG["input_months"], freq="M").strftime("%Y-%m").tolist()
BLOCK = 50
NFOLD = 5
STATIC_NAMES = [
    "elevation",
    "slope",
    "northness",
    "eastness",
    "twi",
    "treecover2000",
    "dnbr",
    "burn_doy",
    "edge_distance",
]


def warp(
    src_path: str,
    shape: tuple[int, int],
    tr: rasterio.Affine,
    crs: str,
    how: Resampling,
    factor: int = 1,
) -> np.ndarray:
    h, w = shape
    out = np.full((h * factor, w * factor), np.nan, "float32")
    with rasterio.open(src_path) as src:
        reproject(
            rasterio.band(src, 1),
            out,
            dst_transform=tr * rasterio.Affine.scale(1 / factor),
            dst_crs=crs,
            resampling=how,
            dst_nodata=np.nan,
        )
    return out


def d8_accumulation(z: np.ndarray) -> np.ndarray:
    h, w = z.shape
    zz = np.where(np.isfinite(z), z, -1e9)
    acc = np.ones(z.size)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dist = np.array([np.hypot(a, b) for a, b in offs])
    pad = np.pad(zz, 1, constant_values=-1e9)
    drops = np.stack(
        [
            (zz - pad[1 + a : 1 + a + h, 1 + b : 1 + b + w]) / d
            for (a, b), d in zip(offs, dist)
        ]
    )
    best = drops.argmax(0)
    has = drops.max(0) > 0
    rr, cc = np.indices((h, w))
    tr_ = rr + np.array([o[0] for o in offs])[best]
    tc_ = cc + np.array([o[1] for o in offs])[best]
    ok = has & (tr_ >= 0) & (tr_ < h) & (tc_ >= 0) & (tc_ < w)
    target = np.where(ok, tr_ * w + tc_, -1).ravel()
    for i in np.argsort(-zz.ravel(), kind="stable"):
        j = target[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(h, w)


def main() -> None:
    red, rec_path, fire_csv, out = (
        Path(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    months = pd.read_csv(red / "months.csv")
    bands = [int(months.band[months.month == m].iloc[0]) for m in MONTHS]
    with rasterio.open(rec_path) as rec:
        rri5, stage, sev = rec.read(3), rec.read(4), rec.read(8)
        crs, tr = rec.crs.to_string(), rec.transform
        shape = rec.shape
    h, w = shape

    x = np.zeros((len(MONTHS), 4, h, w), "int16")
    for c, name in enumerate(["NBR", "NDVI", "NDMI"]):
        with rasterio.open(red / f"{name}_100m.tif") as src:
            a = src.read(bands)
            ok = a != src.nodata
            x[:, c] = np.where(ok, a, 0)
            if c == 0:
                x[:, 3] = ok.astype("int16") * 10000
                full = src.read().astype("float32")
                full[full == src.nodata] = np.nan
                full /= 10000

    per = pd.PeriodIndex(months.month, freq="M")
    pre = per.year <= int(CFG["baseline_last_year"])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim = np.stack(
            [np.nanmean(full[pre & (per.month == m)], axis=0) for m in range(1, 13)]
        )
        ybase = np.nanmean(clim, axis=0)
    mon = np.array([int(m[5:]) for m in MONTHS])
    y_in = full[np.array(bands) - 1] - (clim[mon - 1] - ybase)
    jun = int(months.band[months.month == CFG["dnbr_pre_month"]].iloc[0]) - 1
    dnbr = full[jun] - full[bands[0] - 1]

    f = 4
    dem = np.full((h * f, w * f), np.nan, "float32")
    for t in TILES:
        tmp = warp(DEM_URL.format(t=t), shape, tr, crs, Resampling.bilinear, f)
        dem = np.where(np.isfinite(tmp), tmp, dem)
    gy, gx = np.gradient(dem, 25.0)
    dzdx, dzdn = gx, -gy
    slope25 = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    ex, en = -dzdx, -dzdn
    norm = np.hypot(ex, en) + 1e-9

    def block(a: np.ndarray) -> np.ndarray:
        return np.nanmean(a.reshape(h, f, w, f), axis=(1, 3))

    elev, slope = block(dem), block(slope25)
    east, north = block(ex / norm), block(en / norm)
    mag = np.hypot(east, north) + 1e-9
    northness, eastness = north / mag, east / mag
    acc = d8_accumulation(np.where(elev > 0, elev, np.nan))
    tanb = np.tan(np.radians(np.maximum(slope, 0.5)))
    twi = np.log(acc * 100.0 / tanb)
    tree = warp(GFC_URL, shape, tr, crs, Resampling.average)

    fp = pd.read_csv(fire_csv, parse_dates=["date"])
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    fx, fy = tf.transform(fp.lon.values, fp.lat.values)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    px, py = tr * (cols, rows)
    _, idx = cKDTree(np.c_[fx, fy]).query(np.c_[np.ravel(px), np.ravel(py)])
    doy = fp.date.dt.dayofyear.values[idx].reshape(h, w).astype("float32")

    burned = np.isin(sev, [2, 3, 4, 5])
    edge = ndimage.distance_transform_edt(burned) * 0.1
    doy = np.where(burned, doy, np.nan)

    statics = np.stack([elev, slope, northness, eastness, twi, tree, dnbr, doy, edge])
    stats = []
    for i, s in enumerate(statics):
        v = s[burned & np.isfinite(s)]
        mu, sd = float(v.mean()), float(v.std() + 1e-6)
        stats.append((STATIC_NAMES[i], mu, sd))
        statics[i] = np.nan_to_num((s - mu) / sd, nan=0.0)

    valid = burned & np.isin(stage, [1, 2, 3]) & np.isfinite(rri5)
    bid = (np.arange(h)[:, None] // BLOCK) * (w // BLOCK + 1) + np.arange(w)[
        None, :
    ] // BLOCK
    ids, counts = np.unique(bid[valid], return_counts=True)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(ids))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(NFOLD)
    fold_of = {}
    for o in order:
        k = int(load.argmin())
        fold_of[int(ids[o])] = k
        load[k] += counts[o]
    fold = np.full((h, w), -1, "int8")
    for b, k in fold_of.items():
        fold[bid == b] = k

    np.savez_compressed(
        out,
        x=x,
        s=statics.astype("float16"),
        y=np.nan_to_num(rri5).astype("float32"),
        y_in=y_in.astype("float16"),
        y_base=ybase.astype("float32"),
        valid=valid,
        severity=sev.astype("uint8"),
        fold=fold,
        block=bid.astype("int32"),
        transform=np.array(tr)[:6],
        crs=np.array(crs),
        months=np.array(MONTHS),
        static_names=np.array(STATIC_NAMES),
        static_stats=np.array([(m, s) for _, m, s in stats]),
    )
    print(
        "valid pixels",
        int(valid.sum()),
        "blocks",
        len(ids),
        "per fold",
        load.astype(int),
    )
    for n, m, s in stats:
        print(f"{n:14s} mean {m:9.3f} sd {s:8.3f}")


if __name__ == "__main__":
    main()
