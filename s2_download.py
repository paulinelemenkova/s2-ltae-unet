#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import time
import urllib.request
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from odc.stac import configure_rio, load
from pystac import Item
from pystac_client import Client

import rsconfig

CFG = rsconfig.load()

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
LEGACY = "sentinel-2-l2a"
LEGACY_OFFSET_IN_DN_BEFORE = dt.date(2022, 3, 1)
OFFSET_WARN = 0.06
CHECK_MIN_CLEAR = 0.05

BBOX = tuple(CFG["aoi_bbox"])
TEST_BBOX = tuple(CFG["test_bbox"])

CRS = CFG["crs"]
RES = 20

START, END = CFG["period"]
MAX_TILE_CLOUD = 80

OUT_DIR = Path("s2_data")

RETRY_PASSES = 3
NET_WAIT_MAX = 6 * 3600

BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12",
}
SUMMER_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

SCL_CLEAR = [2, 4, 5]
HOT_LIMIT = 0.08

INDEX_SCALE = 10000
INDEX_NODATA = -32768
REFL_NODATA = 0

log = logging.getLogger("s2")


def setup_gdal() -> None:
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_REGION", "us-west-2")
    configure_rio(
        cloud_defaults=True,
        aws={"aws_unsigned": True},
        GDAL_HTTP_MAX_RETRY="10",
        GDAL_HTTP_RETRY_DELAY="5",
        GDAL_HTTP_TIMEOUT="60",
    )


def wait_for_network() -> None:
    t0, said = time.time(), False
    while True:
        try:
            with urllib.request.urlopen(STAC_URL, timeout=20):
                if said:
                    log.info("network back after %.0f min", (time.time() - t0) / 60)
                return
        except OSError:
            if time.time() - t0 > NET_WAIT_MAX:
                raise RuntimeError("no network for 6 h") from None
            if not said:
                log.warning("no network (sleep?) - waiting")
                said = True
            time.sleep(30)


def _search_one(collection: str, bbox, t0: str, t1: str, cloud: bool = True) -> list:
    client = Client.open(STAC_URL)
    for attempt in range(5):
        try:
            res = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=f"{t0}/{t1}",
                query={"eo:cloud_cover": {"lt": MAX_TILE_CLOUD}} if cloud else None,
                max_items=None,
            )
            return list(res.items())
        except Exception as exc:
            log.warning("search %s failed (%s), retry %d", collection, exc, attempt + 1)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"STAC search failed for {collection} {t0}/{t1}")


def item_day(item) -> str:
    return str(item.properties["datetime"])[:10]


def item_tile(item) -> str:
    p = item.properties
    return (
        str(p.get("s2:mgrs_tile") or p.get("grid:code") or item.id.split("_")[1])
        .replace("MGRS-", "")
        .lstrip("T")
    )


def search(bbox: tuple[float, float, float, float], t0: str, t1: str) -> list:
    c1_all = _search_one(COLLECTION, bbox, t0, t1, cloud=False)
    c1_days = {item_day(it) for it in c1_all}
    c1 = [
        it for it in c1_all if it.properties.get("eo:cloud_cover", 0) < MAX_TILE_CLOUD
    ]
    legacy = [
        it for it in _search_one(LEGACY, bbox, t0, t1) if item_day(it) not in c1_days
    ]
    best: dict[tuple[str, str], Item] = {}
    for it in c1 + legacy:
        k = (item_day(it), item_tile(it))
        cc = it.properties.get("eo:cloud_cover", 100)
        if k not in best or cc < best[k].properties.get("eo:cloud_cover", 100):
            best[k] = it
    items = sorted(best.values(), key=lambda it: (item_day(it), item_tile(it)))
    n_leg = sum(1 for it in items if it.collection_id == LEGACY)
    if n_leg:
        log.info(
            "%s/%s: %d items, %d from %s (days missing in %s)",
            t0,
            t1,
            len(items),
            n_leg,
            LEGACY,
            COLLECTION,
        )
    return items


def boa_offset(item) -> float:
    if item.collection_id == COLLECTION:
        return -0.1
    p = item.properties
    if str(p.get("s2:processing_baseline", "00.00")) < "04.00":
        return 0.0
    if p.get("earthsearch:boa_offset_applied"):
        return 0.0
    day = dt.date.fromisoformat(item_day(item))
    return -0.1 if day < LEGACY_OFFSET_IN_DN_BEFORE else 0.0


def to_reflectance(dn: xr.DataArray, offset: float) -> xr.DataArray:
    rho = dn.astype("float32") * 0.0001 + offset
    return rho.where(dn != 0)


def nd(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    den = a + b
    return ((a - b) / den).where(den != 0)


def load_masked(items: list, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    groups: dict[float, list] = {}
    for it in items:
        groups.setdefault(boa_offset(it), []).append(it)
    parts, like = [], None
    for offset, its in sorted(groups.items()):
        ds = load(
            its,
            bands=[*BANDS, "scl"],
            **(
                {"like": like}
                if like is not None
                else {"bbox": bbox, "crs": CRS, "resolution": RES}
            ),
            groupby="solar_day",
            resampling={"*": "average", "scl": "nearest"},
            chunks={"x": 1024, "y": 1024},
            fail_on_error=False,
        )
        like = ds if like is None else like
        ds = ds.rename(BANDS)
        refl = {b: to_reflectance(ds[b], offset) for b in BANDS.values()}
        hot = refl["B02"] - 0.5 * refl["B04"]
        clear = ds["scl"].isin(SCL_CLEAR) & (hot <= HOT_LIMIT)
        part = xr.Dataset({b: v.where(clear) for b, v in refl.items()})
        part["clear"] = clear
        parts.append(part)
    out = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    out = out.assign_coords(time=out["time"].dt.floor("D")).sortby("time")
    if out.indexes["time"].has_duplicates:
        refl = out[list(BANDS.values())].groupby("time").mean(skipna=True)
        refl["clear"] = out["clear"].groupby("time").max()
        out = refl
    return out


def check_offsets(day_mean: xr.DataArray, day_clear: xr.DataArray, label: str) -> None:
    v = np.where(day_clear.values >= CHECK_MIN_CLEAR, day_mean.values, np.nan)
    if np.isfinite(v).sum() < 2:
        return
    ref = np.nanmedian(v)
    for t, x in zip(day_mean["time"].values, v, strict=True):
        if np.isfinite(x) and abs(x - ref) > OFFSET_WARN:
            log.warning(
                "%s: %s mean clear red %.3f vs median %.3f - residual cloud/snow or BOA offset",
                label,
                str(t)[:10],
                x,
                ref,
            )


def write_tif(da: xr.DataArray, path: Path, nodata: float) -> None:
    da = da.rio.write_crs(CRS).rio.write_nodata(nodata)
    tmp = path.with_suffix(".part.tif")
    da.rio.to_raster(
        tmp,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    tmp.replace(path)


def scale_int16(da: xr.DataArray) -> xr.DataArray:
    v = (da * INDEX_SCALE).round().clip(-INDEX_SCALE, INDEX_SCALE)
    return v.fillna(INDEX_NODATA).astype("int16")


def log_scenes(items: list, writer) -> None:
    for it in items:
        p = it.properties
        writer.writerow(
            [
                it.id,
                item_day(it),
                item_tile(it),
                p.get("eo:cloud_cover"),
                p.get("s2:processing_baseline", ""),
                it.collection_id,
                boa_offset(it),
            ]
        )


def monthly_indices(month: pd.Period, bbox, out_dir: Path, writer) -> None:
    path = out_dir / "monthly" / f"S2_{month}.tif"
    if path.exists():
        log.info("%s exists, skipped", path.name)
        return
    t0 = month.start_time.strftime("%Y-%m-%d")
    t1 = month.end_time.strftime("%Y-%m-%d")
    items = search(bbox, t0, t1)
    if not items:
        log.warning("%s: no scenes", month)
        return
    log_scenes(items, writer)
    ds = load_masked(items, bbox)

    nbr = nd(ds["B08"], ds["B12"])
    ndvi = nd(ds["B08"], ds["B04"])
    ndmi = nd(ds["B8A"], ds["B11"])

    def med(x: xr.DataArray) -> xr.DataArray:
        return x.chunk({"time": -1}).median("time", skipna=True)

    count = ds["clear"].sum("time").clip(max=254).astype("uint8")
    stack = xr.concat(
        [
            scale_int16(med(nbr)),
            scale_int16(med(ndvi)),
            scale_int16(med(ndmi)),
            count.astype("int16"),
        ],
        dim=pd.Index(["NBR", "NDVI", "NDMI", "n_clear"], name="band"),
    )
    stack, day_red, day_clear = dask.compute(
        stack, ds["B04"].mean(["y", "x"], skipna=True), ds["clear"].mean(["y", "x"])
    )
    check_offsets(day_red, day_clear, str(month))
    stack.attrs["long_name"] = ("NBR", "NDVI", "NDMI", "n_clear")
    stack.attrs["scale_factor_indices"] = 1 / INDEX_SCALE
    write_tif(stack, path, INDEX_NODATA)
    n_days = ds.sizes["time"]
    frac = float((count > 0).mean())
    log.info(
        "%s: %d dates, %.0f %% of pixels with >= 1 clear obs", month, n_days, 100 * frac
    )


def summer_composite(year: int, bbox, out_dir: Path, writer) -> None:
    path = out_dir / "summer" / f"S2_summer_{year}.tif"
    if path.exists():
        log.info("%s exists, skipped", path.name)
        return
    items = search(bbox, f"{year}-06-01", f"{year}-08-31")
    if not items:
        log.warning("summer %d: no scenes", year)
        return
    log_scenes(items, writer)
    ds = load_masked(items, bbox)
    bands = []
    for b in SUMMER_BANDS:
        m = ds[b].chunk({"time": -1}).median("time", skipna=True)
        v = ((m + 0.1) * 10000).round().clip(1, 65535)
        bands.append(v.fillna(REFL_NODATA).astype("uint16"))
    stack = xr.concat(bands, dim=pd.Index(SUMMER_BANDS, name="band"))
    stack, day_red, day_clear = dask.compute(
        stack, ds["B04"].mean(["y", "x"], skipna=True), ds["clear"].mean(["y", "x"])
    )
    check_offsets(day_red, day_clear, f"summer {year}")
    stack.attrs["long_name"] = tuple(SUMMER_BANDS)
    stack.attrs["reflectance"] = "rho = DN * 0.0001 - 0.1"
    write_tif(stack, path, REFL_NODATA)
    log.info("summer %d: %d dates", year, ds.sizes["time"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sentinel-2 L2A monthly index composites from a STAC catalogue"
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="small window and the test months of the configuration",
    )
    ap.add_argument("--start", default=START, help="first month, YYYY-MM")
    ap.add_argument("--end", default=END, help="last month, YYYY-MM")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-summer", action="store_true", help="skip summer composites")
    args = ap.parse_args()

    bbox, out = BBOX, args.out
    months = list(pd.period_range(args.start, args.end, freq="M"))
    if args.test:
        bbox, out = TEST_BBOX, args.out / "test"
        months = [pd.Period(m, freq="M") for m in CFG["test_months"]]
    start, end = str(months[0]), str(months[-1])

    (out / "monthly").mkdir(parents=True, exist_ok=True)
    (out / "summer").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "download.log")],
    )
    setup_gdal()
    log.info("window %s, %s to %s, %d m, %s", bbox, start, end, RES, CRS)

    old = out / "scenes.csv"
    if old.exists() and "collection" not in old.open().readline():
        old.replace(out / "scenes_v1.csv")
        log.info("old scenes.csv kept as scenes_v1.csv")
    new_csv = not old.exists()
    with open(out / "scenes.csv", "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_csv:
            writer.writerow(
                [
                    "id",
                    "date",
                    "tile",
                    "cloud_cover",
                    "baseline",
                    "collection",
                    "boa_offset",
                ]
            )
        tasks: list[tuple[str, object]] = [("month", m) for m in months]
        if not args.no_summer and not args.test:
            years = range(int(start[:4]), int(end[:4]) + 1)
            tasks += [("summer", y) for y in years]
        for attempt in range(1 + RETRY_PASSES):
            if attempt:
                log.info("retry pass %d: %d item(s) failed before", attempt, len(tasks))
            failed = []
            for kind, what in tasks:
                wait_for_network()
                try:
                    if isinstance(what, pd.Period):
                        monthly_indices(what, bbox, out, writer)
                    elif isinstance(what, int):
                        summer_composite(what, bbox, out, writer)
                except Exception as exc:
                    log.error("%s %s failed: %s", kind, what, exc)
                    failed.append((kind, what))
                fh.flush()
            tasks = failed
            if not tasks:
                break
        if tasks:
            log.error(
                "still missing after retries: %s - re-run the same command",
                ", ".join(str(w) for _, w in tasks),
            )

    df = pd.read_csv(out / "scenes.csv").drop_duplicates("id").sort_values("date")
    df.to_csv(out / "scenes.csv", index=False)
    log.info("done: %d scenes listed in scenes.csv", len(df))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
