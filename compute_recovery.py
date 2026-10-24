#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

import rsconfig

CFG = rsconfig.load()

BREAK_DROP = 0.20
BREAK_MONTHS = 4
MIN_WINDOW = 18
RRI_RECOVERED = 0.8
RRI_STALLED = 0.25
MIN_MONTHS = 24
K_GRID = np.geomspace(0.01, 5.0, 160)
FIT_FROM = CFG["fit_from"]
BURN = rsconfig.burn_date(CFG)
BASE_LAST = int(CFG["baseline_last_year"])
HORIZON = rsconfig.horizon(CFG)
Y0 = BURN.year
Y_END = HORIZON[-1].year
M_EARLY = pd.Period(FIT_FROM, "M") + 2
M_LATE = pd.Period(FIT_FROM, "M") + 8
CLASSES = {2: "Low", 3: "Moderate-low", 4: "Moderate-high", 5: "High"}
PIXEL_HA = 1.0
STAGES = {1: "Recovered", 2: "Recovering", 3: "Stalled", 4: "Secondarily disturbed"}


def read_stack(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    src = rasterio.open(path)
    a = src.read().astype("float32")
    a[a == -32768] = np.nan
    return a / 1e4, src


def fit_exp(
    y: np.ndarray, t: np.ndarray, ybase: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.isfinite(y)
    r = np.where(w, ybase[None, :] - y, 0.0)
    best_sse = np.full(y.shape[1], np.inf)
    best_k = np.full(y.shape[1], np.nan)
    best_d = np.full(y.shape[1], np.nan)
    for k in K_GRID:
        e = np.exp(-k * t)[:, None] * w
        d = (r * e).sum(0) / np.maximum((e * e).sum(0), 1e-12)
        sse = (((r - d * e) * w) ** 2).sum(0)
        better = sse < best_sse
        best_sse[better], best_k[better], best_d[better] = sse[better], k, d[better]
    return best_k, best_d, w.sum(0)


def first_run(mask: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    N = mask.shape[1]
    out = np.full(N, -1)
    for j in range(N):
        idx = np.flatnonzero(valid[:, j])
        m = mask[idx, j]
        run = 0
        for p, v in enumerate(m):
            run = run + 1 if v else 0
            if run == n:
                out[j] = idx[p - n + 1]
                break
    return out


def main() -> None:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    nbr, src = read_stack(data / "reduced" / "NBR_100m.tif")
    ndvi, _ = read_stack(data / "reduced" / "NDVI_100m.tif")
    ndmi, _ = read_stack(data / "reduced" / "NDMI_100m.tif")
    months = pd.PeriodIndex(list(src.descriptions), freq="M")
    H, W = nbr.shape[1:]

    with rasterio.open(data / "severity_classes.tif") as s:
        sev = np.zeros((H, W), "uint8")
        reproject(
            s.read(1),
            sev,
            src_transform=s.transform,
            src_crs=s.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.mode,
            src_nodata=0,
            dst_nodata=0,
        )

    pre = months.year <= BASE_LAST
    clim = np.stack(
        [np.nanmean(nbr[pre & (months.month == m)], axis=0) for m in range(1, 13)]
    )
    ybase = np.nanmean(clim, axis=0)
    y = nbr - (clim[months.month - 1] - ybase)

    burned = np.isin(sev, list(CLASSES)) & np.isfinite(ybase)
    iy, ix = np.nonzero(burned)
    post = months >= pd.Period(FIT_FROM, "M")
    m_post = months[post]
    tm = np.array(
        [(p.start_time + pd.Timedelta(days=14) - BURN).days / 365.25 for p in months]
    )
    t, Y = tm[post], y[post][:, iy, ix]
    yb = ybase[iy, ix]
    sm = pd.DataFrame(Y[:MIN_WINDOW]).rolling(3, center=True, min_periods=2).mean()
    i0 = np.nan_to_num(sm.to_numpy(), nan=np.inf).argmin(0)
    ypost = sm.to_numpy()[i0, np.arange(Y.shape[1])]
    Yfit = np.where(np.arange(len(t))[:, None] >= i0[None, :], Y, np.nan)
    k, d, nval = fit_exp(Yfit, t, yb)

    raw = nbr[post][:, iy, ix]
    yoy = np.full_like(raw, np.nan)
    yoy[12:] = raw[12:] - raw[:-12]
    valid = np.isfinite(yoy)
    below = (yoy < -BREAK_DROP) & (np.arange(len(t))[:, None] >= i0[None, :] + 12)
    br = first_run(below, valid, BREAK_MONTHS)
    dist = br >= 0
    for j in np.flatnonzero(dist):
        seg = Yfit[: br[j], j : j + 1]
        if np.isfinite(seg).sum() >= 6:
            kk, dd, _ = fit_exp(seg, t[: br[j]], yb[j : j + 1])
            k[j], d[j] = kk[0], dd[0]

    yr5 = (m_post >= HORIZON[0]) & (m_post <= HORIZON[-1])
    drop = yb - ypost
    rri5 = (np.nanmean(Y[yr5], axis=0) - ypost) / np.where(drop > 0.05, drop, np.nan)
    ok = (nval >= MIN_MONTHS) & np.isfinite(rri5)
    k[~ok] = np.nan
    t80 = np.log(5.0) / k
    ok = (nval >= MIN_MONTHS) & np.isfinite(rri5)
    k[~ok] = np.nan
    t80[~ok | (k <= K_GRID[0])] = np.nan

    stage = np.where(rri5 >= RRI_RECOVERED, 1, np.where(rri5 < RRI_STALLED, 3, 2))
    stage[dist] = 4
    stage[~ok & ~dist] = 0

    def grid(v: np.ndarray, fill: float = np.nan) -> np.ndarray:
        g = np.full((H, W), fill, "float32")
        g[iy, ix] = v
        return g

    bmonth = np.where(
        dist, [int(m_post[b].strftime("%Y%m")) if b >= 0 else 0 for b in br], 0
    )
    bands = {
        "k": grid(k),
        "T80": grid(t80),
        "RRI5": grid(rri5),
        "stage": grid(stage, 0),
        "break_month": grid(bmonth, 0),
        "y_base": ybase.astype("float32"),
        "y_post": grid(ypost),
        "severity": sev.astype("float32"),
        "min_month": grid(np.array([int(m_post[i].strftime("%Y%m")) for i in i0]), 0),
    }
    prof = src.profile | {
        "count": len(bands),
        "dtype": "float32",
        "nodata": np.nan,
        "predictor": 3,
    }
    with rasterio.open(out / "recovery_100m.tif", "w", **prof) as dst:
        for b, (name, g) in enumerate(bands.items(), start=1):
            dst.write(g, b)
            dst.set_band_description(b, name)

    rows = []
    groups = {1: "Unburned"} | CLASSES
    for ci, cname in groups.items():
        msk = sev == ci
        for name, arr in (("NBR", nbr), ("NDVI", ndvi), ("NDMI", ndmi)):
            v = arr[:, msk]
            q = np.nanpercentile(v, [25, 50, 75], axis=1)
            for i, p in enumerate(months):
                rows.append([cname, name, str(p), *np.round(q[:, i], 4)])
    pd.DataFrame(rows, columns=["class", "index", "month", "q25", "q50", "q75"]).to_csv(
        out / "curves_by_class.csv", index=False
    )

    sv = sev[iy, ix]
    with open(out / "rri_by_year.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["year_n", "window", "rri_median", "rri_q25", "rri_q75"])
        for n in range(1, Y_END - Y0 + 1):
            a = pd.Period(f"{Y0 - 1 + n}-{BURN.month:02d}", "M")
            b = a + 11
            win = (m_post >= a) & (m_post <= b)
            rri_n = (np.nanmean(Y[win], axis=0) - ypost) / np.where(
                drop > 0.05, drop, np.nan
            )
            v = rri_n[ok & ~dist]
            wr.writerow(
                [n, f"{a}/{b}", *np.round(np.nanpercentile(v, [50, 25, 75]), 3)]
            )

    with open(out / "metrics_by_class.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "class",
                "area_ha",
                "k_median",
                "T80_median",
                "RRI5_median",
                "recovered_pct",
                "recovering_pct",
                "stalled_pct",
                "disturbed_pct",
            ]
        )
        for ci, cname in [*CLASSES.items(), (0, "All burned")]:
            m = (sv == ci) if ci else np.ones_like(sv, bool)
            n = m.sum()
            wr.writerow(
                [
                    cname,
                    int(n * PIXEL_HA),
                    round(float(np.nanmedian(k[m & ~dist])), 3),
                    round(float(np.nanmedian(t80[m & ~dist])), 2),
                    round(float(np.nanmedian(rri5[m & ~dist])), 3),
                    *[
                        round(100 * float((stage[m] == s).mean()), 1)
                        for s in (1, 2, 3, 4)
                    ],
                ]
            )
    with open(out / "stage_area.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stage", "code", "area_ha", "pct"])
        for s, name in STAGES.items():
            wr.writerow(
                [
                    name,
                    s,
                    int((stage == s).sum() * PIXEL_HA),
                    round(100 * float((stage == s).mean()), 1),
                ]
            )
        wr.writerow(
            [
                "Not fitted",
                0,
                int((stage == 0).sum()),
                round(100 * float((stage == 0).mean()), 1),
            ]
        )

    jja = np.isin(months.month, [6, 7, 8])
    base_jja = np.nanmean(nbr[jja & pre], axis=0)
    base_sep = np.nanmean(
        nbr[(months.month == pd.Period(FIT_FROM, "M").month) & pre], axis=0
    )
    an = []
    for yr in range(Y0, Y_END + 1):
        if yr == Y0:
            an.append(nbr[months == pd.Period(FIT_FROM, "M")][0] - base_sep)
        else:
            an.append(np.nanmean(nbr[jja & (months.year == yr)], axis=0) - base_jja)
    prof6 = src.profile | {
        "count": len(an),
        "dtype": "float32",
        "nodata": np.nan,
        "predictor": 3,
    }
    with rasterio.open(out / "summer_anomaly_100m.tif", "w", **prof6) as dst:
        for b, a in enumerate(an, start=1):
            dst.write(a.astype("float32"), b)
            dst.set_band_description(b, str(Y0 - 1 + b))
    with open(out / "summer_anomaly_stats.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["year", "median_burned", "pct_below_-0.1", "median_unburned"])
        for b, a in enumerate(an):
            wr.writerow(
                [
                    Y0 + b,
                    round(float(np.nanmedian(a[burned])), 3),
                    round(100 * float(np.nanmean(a[burned] < -0.1)), 1),
                    round(float(np.nanmedian(a[sev == 1])), 3),
                ]
            )

    ex = []
    cand = np.flatnonzero(~dist & ok & (sv == 5) & (nval >= 55))
    if cand.size:
        j = cand[
            np.argmin(
                np.abs(rri5[cand] - np.nanmedian(rri5[cand]))
                + np.abs(np.log(k[cand] / np.nanmedian(k[cand])))
            )
        ]
        ex.append(("natural", j))
    late = (i0 >= 9) & ((Y[0] - ypost) > 0.15) & ~dist & ok & (nval >= 50) & (sv <= 3)
    cand = np.flatnonzero(late)
    if cand.size:
        res = np.sqrt(
            np.nanmean((Yfit - (yb - d * np.exp(-k * t[:, None]))) ** 2, axis=0)
        )
        j = cand[np.nanargmin(res[cand])]
        ex.append(("delayed", j))
    with open(out / "examples.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "case",
                "row",
                "col",
                "month",
                "nbr",
                "y_des",
                "y_base",
                "k",
                "delta",
                "rri5",
                "t80",
                "break_month",
                "min_month",
                "y_post",
            ]
        )
        for case, j in ex:
            for i, p in enumerate(months):
                wr.writerow(
                    [
                        case,
                        iy[j],
                        ix[j],
                        str(p),
                        round(float(nbr[i, iy[j], ix[j]]), 4),
                        round(float(y[i, iy[j], ix[j]]), 4),
                        round(float(yb[j]), 4),
                        round(float(k[j]), 4),
                        round(float(d[j]), 4),
                        round(float(rri5[j]), 3),
                        round(float(t80[j]), 2),
                        str(m_post[br[j]]) if br[j] >= 0 else "",
                        str(m_post[i0[j]]),
                        round(float(ypost[j]), 4),
                    ]
                )
    with open(out / "minimum_timing.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "class",
                "min_by_early_pct",
                "min_early_to_late_pct",
                "min_after_late_pct",
                "median_extra_drop",
            ]
        )
        for ci, cname in [*CLASSES.items(), (0, "All burned")]:
            m = (sv == ci) if ci else np.ones_like(sv, bool)
            mm = np.array([m_post[i] for i in i0[m]])
            extra = np.nanmedian(Y[0, m] - ypost[m])
            wr.writerow(
                [
                    cname,
                    round(100 * float(np.mean(mm <= M_EARLY)), 1),
                    round(
                        100 * float(np.mean((mm > M_EARLY) & (mm <= M_LATE))),
                        1,
                    ),
                    round(100 * float(np.mean(mm > M_LATE)), 1),
                    round(float(extra), 3),
                ]
            )
    print((out / "minimum_timing.csv").read_text())
    print((out / "metrics_by_class.csv").read_text())
    print((out / "stage_area.csv").read_text())
    print((out / "summer_anomaly_stats.csv").read_text())


if __name__ == "__main__":
    main()
