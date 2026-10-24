#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_fn

import rsconfig

CFG = rsconfig.load()
BURN = rsconfig.burn_date(CFG)
END_MONTH = (BURN.month - 2) % 12 + 1
Y_END = rsconfig.horizon(CFG)[-1].year
REF0, REF1 = CFG["climate_reference"]

CLASSES = {2: "Low", 3: "Moderate-low", 4: "Moderate-high", 5: "High"}
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def loglogistic_cdf(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    r = np.sort(ref)
    n = r.size
    fi = (np.arange(1, n + 1) - 0.35) / n
    w0, w1, w2 = (np.mean((1 - fi) ** s * r) for s in range(3))
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    g = gamma_fn(1 + 1 / beta) * gamma_fn(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gam = w0 - alpha * g
    cdf = 1 / (1 + (alpha / np.maximum(x - gam, 1e-9)) ** beta)
    return np.clip(cdf, 1e-4, 1 - 1e-4)


def main() -> None:
    cdir, rdir, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(cdir / "controls_pixels.csv")
    d = d[(d.stage < 4) & d.k.notna() & (d.slope >= 2)]
    d["sector"] = pd.Categorical(
        np.array(SECTORS)[((d.aspect + 22.5) % 360 // 45).astype(int)], SECTORS
    )
    lines = [f"pixels used: {len(d)} (stage < 4, slope >= 2 deg)"]

    rows = []
    for c, name in CLASSES.items():
        g = d[d.severity == c]
        h, p = stats.kruskal(*[g.k[g.sector == s] for s in SECTORS])
        lines.append(f"Kruskal-Wallis k ~ aspect, {name}: H = {h:.1f}, p = {p:.2g}")
        for s in SECTORS:
            v = g.k[g.sector == s]
            rows.append([name, s, round(float(v.median()), 3), len(v)])
    pd.DataFrame(rows, columns=["class", "sector", "k_median", "n"]).to_csv(
        out / "aspect_k.csv", index=False
    )
    h, p = stats.kruskal(*[d.k[d.severity == c] for c in CLASSES])
    lines.append(f"Kruskal-Wallis k ~ severity: H = {h:.1f}, p = {p:.2g}")
    south = d.k[d.sector.isin(["SE", "S", "SW"])].median()
    north = d.k[d.sector.isin(["NW", "N", "NE"])].median()
    lines.append(f"median k south-facing {south:.3f}, north-facing {north:.3f}")

    for var, edges, fname in (
        (
            "elev",
            np.arange(0, np.ceil(d.elev.max() / 100) * 100 + 1, 100),
            "elev_k.csv",
        ),
        ("slope", np.arange(0, 46, 5), "slope_k.csv"),
    ):
        rho, p = stats.spearmanr(d[var], d.k)
        lines.append(f"Spearman k ~ {var}: rho = {rho:.3f}, p = {p:.2g}")
        b = pd.cut(d[var], edges)
        rows = []
        for iv, g in d.groupby(b, observed=True):
            if len(g) < 100:
                continue
            q = np.percentile(g.k, [25, 50, 75])
            per = [g.k[g.severity == c].median() for c in CLASSES]
            rows.append([iv.mid, *np.round(q, 3), *np.round(per, 3), len(g)])
        pd.DataFrame(
            rows, columns=["mid", "q25", "q50", "q75", *CLASSES.values(), "n"]
        ).to_csv(out / fname, index=False)

    c = pd.read_csv(cdir / "climate_monthly.csv", index_col=0, parse_dates=True)
    roll = c.rolling(12).sum()
    jul = roll[roll.index.month == END_MONTH].dropna()
    jul.index = jul.index.year
    ref = jul.loc[REF0:REF1]
    spei = pd.Series(
        stats.norm.ppf(loglogistic_cdf(jul.D.values, ref.D.values)), index=jul.index
    )
    pa = 100 * (jul.P / ref.P.mean() - 1)
    r = pd.read_csv(rdir / "rri_by_year.csv")
    inc = r.rri_median.diff().fillna(r.rri_median)
    yrs = list(range(BURN.year, Y_END + 1))
    clim = pd.DataFrame(
        {
            "year_end": yrs,
            "window": [f"{y - 1}-{BURN.month:02d} to {y}-{END_MONTH:02d}" for y in yrs],
            "P_mm": jul.P.loc[yrs].round(0).values,
            "P_anom_pct": pa.loc[yrs].round(1).values,
            "SPEI12": spei.loc[yrs].round(2).values,
            "RRI_median": [np.nan, *r.rri_median.values],
            "RRI_increment": [np.nan, *inc.round(3).values],
        }
    )
    clim.to_csv(out / "climate_years.csv", index=False)
    lines.append(f"reference P ({REF0}-{REF1}): {ref.P.mean():.0f} mm")
    lines.append(clim.to_string(index=False))
    (out / "tests.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
