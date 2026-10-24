from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import rsconfig

CFG = rsconfig.load()

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/"
BBOX = tuple(CFG["fire_bbox"])
D0, D1 = (dt.date.fromisoformat(d) for d in CFG["fire_window"])
DOY0, DOY1 = D0.timetuple().tm_yday, D1.timetuple().tm_yday
RES = 0.0025
YEAR = D0.year
MONTHS_SPAN = f"{D0:%Y-%m}-01/{(D1.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)}"


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return dict(json.load(r))


_TOKENS: dict[str, str] = {}


def sign(href: str) -> str:
    account = href.split("//")[1].split(".")[0]
    container = href.split("/")[3]
    key = f"{account}/{container}"
    if key not in _TOKENS:
        with urllib.request.urlopen(TOKEN + key, timeout=60) as r:
            _TOKENS[key] = str(json.load(r)["token"])
    return f"{href}?{_TOKENS[key]}"


def items(collection: str, when: str) -> list[dict]:
    body = {
        "collections": [collection],
        "bbox": list(BBOX),
        "datetime": when,
        "limit": 50,
    }
    return list(post(STAC, body)["features"])


def doy_to_date(doy: int) -> dt.date:
    return dt.date(YEAR, 1, 1) + dt.timedelta(days=doy - 1)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    w0, s0, e0, n0 = BBOX
    nx, ny = round((e0 - w0) / RES), round((n0 - s0) / RES)
    geo_tr = from_origin(w0, n0, RES, RES)

    geo = np.zeros((ny, nx), dtype=np.int16)
    area_by_doy: dict[int, float] = {}
    nat: np.ndarray | None = None
    for it in sorted(items("modis-64A1-061", MONTHS_SPAN), key=lambda f: f["id"]):
        href = sign(it["assets"]["Burn_Date"]["href"])
        with rasterio.open(href) as src:
            to_sin = Transformer.from_crs(4326, src.crs, always_xy=True)
            xs, ys = to_sin.transform([w0, e0, w0, e0], [s0, s0, n0, n0])
            win = rasterio.windows.from_bounds(
                min(xs), min(ys), max(xs), max(ys), src.transform
            )
            win = win.round_offsets().round_lengths()
            a = src.read(1, window=win)
            a = np.where((a >= DOY0) & (a <= DOY1), a, 0).astype(np.int16)
            nat = a if nat is None else np.where(a > 0, a, nat)
            wtr, crs = src.window_transform(win), src.crs
            px_ha = abs(src.res[0] * src.res[1]) / 1e4
        print("MCD64A1", it["id"])
    assert nat is not None, "no MCD64A1 granule found"
    rr, cc = np.nonzero(nat)
    x, y = rasterio.transform.xy(wtr, rr, cc)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    lon, lat = np.asarray(lon), np.asarray(lat)
    keep = (lon >= w0) & (lon <= e0) & (lat >= s0) & (lat <= n0)
    for d in nat[rr[keep], cc[keep]]:
        area_by_doy[int(d)] = area_by_doy.get(int(d), 0.0) + px_ha
    reproject(
        nat,
        geo,
        src_transform=wtr,
        src_crs=crs,
        dst_transform=geo_tr,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    with rasterio.open(
        out / "burndate_geo.tif",
        "w",
        driver="GTiff",
        width=nx,
        height=ny,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=geo_tr,
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(geo, 1)

    pix: list[tuple[float, float, str, str, str]] = []
    for it in items("modis-14A1-061", f"{D0}/{D1 - dt.timedelta(days=1)}"):
        sat = "Terra" if it["id"].startswith("MOD") else "Aqua"
        start = dt.date.fromisoformat(it["properties"]["start_datetime"][:10])
        href = sign(it["assets"]["FireMask"]["href"])
        with rasterio.open(href) as src:
            to_sin = Transformer.from_crs(4326, src.crs, always_xy=True)
            to_geo = Transformer.from_crs(src.crs, 4326, always_xy=True)
            xs, ys = to_sin.transform([w0, e0, w0, e0], [s0, s0, n0, n0])
            win = rasterio.windows.from_bounds(
                min(xs), min(ys), max(xs), max(ys), src.transform
            )
            win = win.round_offsets().round_lengths()
            wtr = src.window_transform(win)
            stack = src.read(window=win)
            for b in range(stack.shape[0]):
                day = start + dt.timedelta(days=b)
                rr, cc = np.nonzero(stack[b] >= 7)
                if not rr.size:
                    continue
                x, y = rasterio.transform.xy(wtr, rr, cc)
                lon, lat = to_geo.transform(x, y)
                for lo, la, v in zip(
                    np.atleast_1d(lon),
                    np.atleast_1d(lat),
                    stack[b][rr, cc],
                    strict=True,
                ):
                    if w0 <= lo <= e0 and s0 <= la <= n0:
                        conf = {7: "low", 8: "nominal", 9: "high"}[int(v)]
                        pix.append((float(lo), float(la), day.isoformat(), sat, conf))
        print("14A1", it["id"])
    pix.sort(key=lambda p: (p[2], p[3]))
    with open(out / "fire_pixels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lon", "lat", "date", "satellite", "confidence"])
        w.writerows((f"{a:.5f}", f"{b:.5f}", c, d, e) for a, b, c, d, e in pix)

    days = sorted(
        set(area_by_doy)
        | {(dt.date.fromisoformat(p[2]) - dt.date(YEAR, 1, 1)).days + 1 for p in pix}
    )
    cum = 0.0
    with open(out / "daily_progression.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["date", "doy", "burned_ha", "cumulative_ha", "terra_pixels", "aqua_pixels"]
        )
        for d in days:
            ds = doy_to_date(d).isoformat()
            cum += area_by_doy.get(d, 0.0)
            nt = sum(1 for p in pix if p[2] == ds and p[3] == "Terra")
            na = sum(1 for p in pix if p[2] == ds and p[3] == "Aqua")
            w.writerow([ds, d, round(area_by_doy.get(d, 0.0)), round(cum), nt, na])
            print(ds, d, round(area_by_doy.get(d, 0.0)), round(cum), nt, na)
    print(f"fire pixels: {len(pix)}; MCD64A1 burned area in window: {cum:.0f} ha")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
