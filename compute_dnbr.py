from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

import rsconfig

CFG = rsconfig.load()

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"
PERIMETER_URL = CFG["perimeter"]["url"]
PERIMETER_FIELD = CFG["perimeter"]["id_field"]
PERIMETER_ID = str(CFG["perimeter"]["id"])
PRE = "/".join(CFG["dnbr_pre"])
POST = "/".join(CFG["dnbr_post"])
EPSG = int(CFG["crs"].split(":")[1])
MAX_CLOUD = 5
KEEP_SCL = (2, 4, 5)
RES = 20.0
X0, X1, Y0, Y1 = (float(v) for v in CFG["grid_bounds"])
BUFFER_M = 500.0
BREAKS = [0.10, 0.27, 0.44, 0.66]
NAMES = ["Unburned", "Low", "Moderate-low", "Moderate-high", "High"]


def stac_items(dt: str) -> list[dict]:
    body = {
        "collections": [COLLECTION],
        "bbox": CFG["perimeter"]["stac_bbox"],
        "datetime": dt,
        "limit": 200,
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
    }
    req = urllib.request.Request(
        STAC,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.load(r)["features"]
    best: dict[tuple[str, str], dict] = {}
    for f in feats:
        key = (f["properties"]["datetime"][:10], f["id"].split("_")[1])
        cc = f["properties"]["eo:cloud_cover"]
        if key not in best or cc < best[key]["properties"]["eo:cloud_cover"]:
            best[key] = f
    return list(best.values())


def read_band(href: str, nx: int, ny: int) -> np.ndarray:
    out = np.zeros((ny, nx), dtype=np.uint16)
    with rasterio.open(href) as src:
        b = src.bounds
        ix0, ix1 = max(X0, b.left), min(X1, b.right)
        iy0, iy1 = max(Y0, b.bottom), min(Y1, b.top)
        if ix0 >= ix1 or iy0 >= iy1:
            return out
        w = (
            from_bounds(ix0, iy0, ix1, iy1, src.transform)
            .round_offsets()
            .round_lengths()
        )
        cols = round((ix1 - ix0) / RES)
        rows = round((iy1 - iy0) / RES)
        a = src.read(1, window=w, out_shape=(rows, cols))
        c0 = round((ix0 - X0) / RES)
        r0 = round((Y1 - iy1) / RES)
        out[r0 : r0 + rows, c0 : c0 + cols] = a
    return out


def nbr_stack(items: list[dict], nx: int, ny: int) -> np.ndarray:
    jobs = []
    for it in items:
        a = it["assets"]
        jobs.append(
            (
                it["properties"]["datetime"][:10],
                a["nir08"]["href"],
                a["swir22"]["href"],
                a["scl"]["href"],
            )
        )

    def load(
        j: tuple[str, str, str, str],
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        d, n, s, c = j
        return d, read_band(n, nx, ny), read_band(s, nx, ny), read_band(c, nx, ny)

    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(load, jobs))
    days: dict[str, np.ndarray] = {}
    for d, n, s, c in res:
        ok = np.isin(c, KEEP_SCL) & (n > 0) & (s > 0)
        rn = n.astype(np.float32) * 1e-4 - 0.1
        rs = s.astype(np.float32) * 1e-4 - 0.1
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (rn - rs) / (rn + rs)
        v[~ok | ~np.isfinite(v)] = np.nan
        if d in days:
            m = np.isnan(days[d])
            days[d][m] = v[m]
        else:
            days[d] = v
    print(f"  {len(items)} items, {len(days)} days: {sorted(days)}")
    return np.stack(list(days.values()))


def main(out: Path) -> None:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    nx, ny = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
    tr = from_origin(X0, Y1, RES, RES)

    with urllib.request.urlopen(PERIMETER_URL, timeout=90) as r:
        g = json.load(r)
    feat = next(
        f
        for f in g["features"]
        if str(f["properties"][PERIMETER_FIELD]) == PERIMETER_ID
    )
    (out / "perimeter.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [feat]})
    )
    to_utm = Transformer.from_crs(4326, EPSG, always_xy=True).transform
    poly = shp_transform(to_utm, shape(feat["geometry"]))
    inside = (
        rasterize(
            [(mapping(poly.buffer(BUFFER_M)), 1)], out_shape=(ny, nx), transform=tr
        )
        == 1
    )

    pre_items, post_items = stac_items(PRE), stac_items(POST)
    print("pre-fire:")
    pre = np.nanmedian(nbr_stack(pre_items, nx, ny), axis=0)
    print("post-fire:")
    post = np.nanmedian(nbr_stack(post_items, nx, ny), axis=0)
    dnbr = pre - post
    valid = np.isfinite(dnbr)

    d16 = np.where(valid, np.clip(np.round(dnbr * 1000), -2000, 2000), -32768).astype(
        np.int16
    )
    prof = {
        "driver": "GTiff",
        "width": nx,
        "height": ny,
        "count": 1,
        "crs": CFG["crs"],
        "transform": tr,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }
    with rasterio.open(
        out / "dnbr.tif", "w", dtype="int16", nodata=-32768, **prof
    ) as dst:
        dst.write(d16, 1)

    cls = (np.digitize(np.nan_to_num(dnbr, nan=-9), BREAKS) + 1).astype(np.uint8)
    cls[~(valid & inside)] = 0
    with rasterio.open(
        out / "severity_classes.tif", "w", dtype="uint8", nodata=0, **prof
    ) as dst:
        dst.write(cls, 1)

    px_ha = RES * RES / 1e4
    tot = int((cls > 0).sum())
    nod = int((inside & ~valid).sum())
    with open(out / "severity_area.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "dnbr_range", "area_ha", "share_pct"])
        lims = ["< 0.10", "0.10-0.27", "0.27-0.44", "0.44-0.66", ">= 0.66"]
        for k, (nm, lim) in enumerate(zip(NAMES, lims, strict=True), start=1):
            n = int((cls == k).sum())
            w.writerow([nm, lim, round(n * px_ha), round(100 * n / tot, 1)])
            print(f"{nm:14s} {lim:10s} {n * px_ha:9.0f} ha {100 * n / tot:5.1f} %")
    burned = int((cls >= 2).sum()) * px_ha
    print(
        f"mapped area {tot * px_ha:.0f} ha; burned (dNBR >= 0.10) {burned:.0f} ha; "
        f"perimeter+buffer pixels without valid data {nod * px_ha:.0f} ha; "
        f"perimeter {poly.area / 1e4:.0f} ha; dNBR median in perimeter "
        f"{np.nanmedian(dnbr[inside]):.3f}; {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
