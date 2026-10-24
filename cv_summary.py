#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "tae": "Temporal-attention U-Net",
    "plain": "U-Net (no attention)",
    "convlstm": "ConvLSTM",
    "lightgbm": "Gradient boosting",
    "exponential": "Exponential extrapolation",
    "persistence": "Persistence",
}


def load(pdir: Path, name: str, shape: tuple[int, ...]) -> np.ndarray:
    single = pdir / f"pred_{name}.npy"
    if single.exists():
        return np.load(single)
    out = np.full(shape, np.nan, "float32")
    for k in range(5):
        p = np.load(pdir / f"pred_{name}_f{k}.npy")
        out = np.where(np.isfinite(p), p, out)
    return out


def main() -> None:
    d = np.load(sys.argv[1])
    pdir, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    valid, fold, blk, y = d["valid"], d["fold"], d["block"], d["y"]
    preds = {m: load(pdir, m, valid.shape) for m in MODELS}
    ok = valid & (fold >= 0)
    for p in preds.values():
        ok &= np.isfinite(p)
    print("pixels scored", int(ok.sum()), "of", int((valid & (fold >= 0)).sum()))

    rows = []
    for k in range(5):
        m = ok & (fold == k)
        o = y[m]
        mse_p = float(((preds["persistence"][m] - o) ** 2).mean())
        for name, p in preds.items():
            e = p[m] - o
            mse = float((e**2).mean())
            rows.append(
                {
                    "model": name,
                    "fold": k + 1,
                    "r2": 1 - mse / float(o.var()),
                    "rmse": np.sqrt(mse),
                    "mae": float(np.abs(e).mean()),
                    "bias": float(e.mean()),
                    "skill": 1 - mse / mse_p,
                    "n": int(m.sum()),
                }
            )
    sc = pd.DataFrame(rows)
    sc.round(4).to_csv(out / "cv_scores.csv", index=False)
    tab = sc.groupby("model", sort=False)[["r2", "rmse", "mae", "bias", "skill"]].agg(
        ["mean", "std"]
    )
    tab.index = [MODELS[i] for i in tab.index]
    tab.round(3).to_csv(out / "cv_table.csv")
    print(tab.round(3).to_string())

    ids = blk[ok]
    ub = np.unique(ids)
    se = {n: (p[ok] - y[ok]) ** 2 for n, p in preds.items()}
    sums = {
        n: pd.Series(v).groupby(ids).sum().reindex(ub).to_numpy() for n, v in se.items()
    }
    cnt = pd.Series(np.ones(ids.size)).groupby(ids).sum().reindex(ub).to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(ub), (2000, len(ub)))
    bt = []
    for n in MODELS:
        if n == "tae":
            continue
        dr = np.sqrt(sums["tae"][draws].sum(1) / cnt[draws].sum(1)) - np.sqrt(
            sums[n][draws].sum(1) / cnt[draws].sum(1)
        )
        obs = np.sqrt(sums["tae"].sum() / cnt.sum()) - np.sqrt(
            sums[n].sum() / cnt.sum()
        )
        bt.append(
            {
                "comparison": f"tae - {n}",
                "d_rmse": obs,
                "ci_low": np.percentile(dr, 2.5),
                "ci_high": np.percentile(dr, 97.5),
                "p_two_sided": float(
                    min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))
                ),
            }
        )
    bt_df = pd.DataFrame(bt)
    bt_df.round(4).to_csv(out / "cv_bootstrap.csv", index=False)
    print(bt_df.round(4).to_string(index=False))
    np.savez_compressed(
        out / "cv_predictions.npz",
        **{n: np.where(ok, p, np.nan).astype("float16") for n, p in preds.items()},
        observed=np.where(ok, y, np.nan).astype("float16"),
        fold=fold,
    )


if __name__ == "__main__":
    main()
