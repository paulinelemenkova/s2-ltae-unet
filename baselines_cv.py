#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import rsconfig

CFG = rsconfig.load()

BURN = rsconfig.burn_date(CFG)
K_GRID = np.geomspace(0.01, 5.0, 160)


def scores(p: np.ndarray, o: np.ndarray) -> dict[str, float]:
    e = p - o
    return {
        "r2": float(1 - (e**2).sum() / ((o - o.mean()) ** 2).sum()),
        "rmse": float(np.sqrt((e**2).mean())),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "n": int(o.size),
    }


def main() -> None:
    d = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    valid, fold, yobs = d["valid"], d["fold"], d["y"]
    iy, ix = np.nonzero(valid & (fold >= 0))
    y_in = d["y_in"].astype("float32")[:, iy, ix]
    yb = d["y_base"][iy, ix]
    t = np.array(
        [
            (pd.Period(m, "M").start_time + pd.Timedelta(days=14) - BURN).days / 365.25
            for m in d["months"]
        ]
    )
    sm = pd.DataFrame(y_in).rolling(3, center=True, min_periods=2).mean().to_numpy()
    i0 = np.nan_to_num(sm, nan=np.inf).argmin(0)
    ypost = sm[i0, np.arange(sm.shape[1])]
    drop = np.where(yb - ypost > 0.05, yb - ypost, np.nan)
    last = np.nanmean(y_in[-3:], axis=0)
    pers = (last - ypost) / drop

    yfit = np.where(np.arange(len(t))[:, None] >= i0[None, :], y_in, np.nan)
    w = np.isfinite(yfit)
    r = np.where(w, yb[None, :] - yfit, 0.0)
    best = np.full(r.shape[1], np.inf)
    kk = np.full(r.shape[1], np.nan)
    dd = np.full(r.shape[1], np.nan)
    for k in K_GRID:
        e = np.exp(-k * t)[:, None] * w
        dk = (r * e).sum(0) / np.maximum((e * e).sum(0), 1e-12)
        sse = (((r - dk * e) * w) ** 2).sum(0)
        b = sse < best
        best[b], kk[b], dd[b] = sse[b], k, dk[b]
    t5 = np.array(
        [
            (p.start_time + pd.Timedelta(days=14) - BURN).days / 365.25
            for p in rsconfig.horizon(CFG)
        ]
    )
    y5 = yb - dd * np.exp(-kk[:, None] * t5[None, :]).mean(1)
    expo = (y5 - ypost) / drop

    x = d["x"].astype("float32")[:, :3, iy, ix] / 10000
    xm = d["x"][:, 3, iy, ix] > 0
    x = np.where(xm[:, None], x, np.nan).reshape(-1, len(iy)).T
    feats = np.c_[x, ypost, last, d["s"].astype("float32")[:, iy, ix].T]
    f_of = fold[iy, ix]
    blk = d["block"][iy, ix]
    yv = yobs[iy, ix]
    pred = {
        "persistence": pers,
        "exponential": expo,
        "lightgbm": np.full(len(iy), np.nan),
    }
    params = {
        "objective": "huber",
        "alpha": 0.1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 2,
        "verbose": -1,
        "seed": 0,
    }
    folds: list[dict[str, object]] = []
    log: dict[str, object] = {"params": params, "folds": folds}
    for k in range(int(fold.max()) + 1):
        tr = f_of != k
        rng = np.random.default_rng(k)
        ub = np.unique(blk[tr])
        vb = rng.choice(ub, max(1, round(0.15 * len(ub))), False)
        va = tr & np.isin(blk, vb)
        tr = tr & ~va
        dtr = lgb.Dataset(feats[tr], yv[tr])
        dva = lgb.Dataset(feats[va], yv[va], reference=dtr)
        bst = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        te = f_of == k
        pred["lightgbm"][te] = bst.predict(feats[te], num_iteration=bst.best_iteration)
        rec = {"fold": k, "rounds": bst.best_iteration}
        for name, p in pred.items():
            ok = te & np.isfinite(p)
            rec[name] = scores(p[ok], yv[ok]) | {"coverage": float(ok.sum() / te.sum())}
        folds.append(rec)
        print(json.dumps(rec))
    imp = bst.feature_importance("gain")
    log["importance_last_fold"] = imp.tolist()
    for name, p in pred.items():
        g = np.full(valid.shape, np.nan, "float32")
        g[iy, ix] = p
        np.save(out / f"pred_{name}.npy", g)
    (out / "log_baselines.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
