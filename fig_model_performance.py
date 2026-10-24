#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.labelpad": 2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans"])).lower()

FOLD_COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
WIN = 100


def ticks(ax: Axes, major: float) -> None:
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(MultipleLocator(major))
        a.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in", pad=2)
    ax.tick_params(which="major", length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.6)


def tag(ax: Axes, t: str) -> None:
    ax.text(
        0,
        1.02,
        t,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )


def main() -> None:
    npz, pdir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(npz)
    valid, fold, yobs = d["valid"], d["fold"], d["y"]
    h, w = valid.shape
    tr = d["transform"]
    pred = np.full((h, w), np.nan, "float32")
    for k in range(5):
        p = np.load(pdir / f"pred_tae_f{k}.npy")
        pred = np.where(np.isfinite(p), p, pred)
    ok = valid & np.isfinite(pred)
    o, f = yobs[ok], pred[ok]
    e = f - o
    r2 = 1 - (e**2).sum() / ((o - o.mean()) ** 2).sum()
    rmse = float(np.sqrt((e**2).mean()))

    i_el = list(d["static_names"]).index("elevation")
    mu, sd = d["static_stats"][i_el]
    elev = d["s"][i_el].astype("float32") * sd + mu
    shade = LightSource(315, 45).hillshade(elev, vert_exag=0.05, dx=100, dy=100)

    x0, y0 = tr[2] / 1000, tr[5] / 1000
    ext = (x0, x0 + w * 0.1, y0 - h * 0.1, y0)

    fold_r2 = []
    for k in range(5):
        m = valid & (fold == k) & np.isfinite(pred)
        ek = pred[m] - yobs[m]
        fold_r2.append(1 - (ek**2).sum() / ((yobs[m] - yobs[m].mean()) ** 2).sum())
    kmed = int(np.argsort(fold_r2)[2])
    cnt = (valid & (fold == kmed)).astype(float)
    cs = cnt.cumsum(0).cumsum(1)
    best, rc = -1.0, (0, 0)
    for r in range(0, h - WIN, 5):
        for c in range(0, w - WIN, 5):
            tot = cs[r + WIN - 1, c + WIN - 1] - (cs[r - 1, c + WIN - 1] if r else 0)
            tot -= cs[r + WIN - 1, c - 1] if c else 0
            tot += cs[r - 1, c - 1] if r and c else 0
            if tot > best:
                best, rc = tot, (r, c)
    r0, c0 = rc
    sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
    wext = (x0 + c0 * 0.1, x0 + (c0 + WIN) * 0.1, y0 - (r0 + WIN) * 0.1, y0 - r0 * 0.1)
    held = valid & (fold == kmed)

    fig = plt.figure(figsize=(17 / 2.54, 13.2 / 2.54))
    gs = fig.add_gridspec(
        2,
        6,
        left=0.07,
        right=0.985,
        top=0.95,
        bottom=0.075,
        wspace=0.9,
        hspace=0.32,
        height_ratios=[1.05, 1],
    )
    axa = fig.add_subplot(gs[0, :2])
    hb = axa.hexbin(
        o,
        f,
        gridsize=45,
        extent=(0, 1.6, 0, 1.6),
        bins="log",
        cmap="turbo",
        mincnt=1,
        lw=0,
    )
    axa.plot([0, 1.6], [0, 1.6], color="0.1", lw=0.8, ls="--")
    axa.set_xlim(0, 1.6)
    axa.set_ylim(0, 1.6)
    axa.set_aspect("equal")
    axa.set_xlabel("Observed RRI")
    axa.set_ylabel("Forecast RRI")
    ticks(axa, 0.4)
    axa.text(
        1.0,
        1.02,
        f"$R^2$ = {r2:.2f}, RMSE = {rmse:.3f}",
        transform=axa.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    cb = fig.colorbar(hb, ax=axa, fraction=0.046, pad=0.03)
    cb.set_label("Pixels per cell", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    tag(axa, "(a)")

    axb = fig.add_subplot(gs[0, 2:])
    axb.imshow(
        shade, extent=ext, cmap="gray", vmin=-0.6, vmax=1.15, interpolation="none"
    )
    fg = np.where(valid & (fold >= 0), fold, np.nan)
    axb.imshow(
        fg,
        extent=ext,
        cmap=ListedColormap(FOLD_COLS),
        norm=BoundaryNorm(np.arange(-0.5, 5.5), 5),
        interpolation="none",
        alpha=0.9,
    )
    axb.add_patch(
        Rectangle(
            (wext[0], wext[2]), WIN * 0.1, WIN * 0.1, fill=False, lw=1.2, ec="black"
        )
    )
    axb.set_xlabel("Easting (km)")
    axb.set_ylabel("Northing (km)")
    ticks(axb, 10)
    for k, col in enumerate(FOLD_COLS):
        axb.plot([], [], "s", color=col, ms=6, label=f"Fold {k + 1}")
    axb.legend(
        loc="lower left",
        ncol=5,
        frameon=True,
        fontsize=7.5,
        handletextpad=0.2,
        columnspacing=0.7,
        borderpad=0.3,
        framealpha=0.9,
        edgecolor="0.6",
    )
    tag(axb, "(b)")

    rri_cmap = plt.get_cmap("turbo")
    panels = [
        ("(c)", np.where(held, yobs, np.nan), rri_cmap, (0, 1.4), "Observed RRI"),
        ("(d)", np.where(held, pred, np.nan), rri_cmap, (0, 1.4), "Forecast RRI"),
        (
            "(e)",
            np.where(held, pred - yobs, np.nan),
            plt.get_cmap("RdBu_r"),
            (-0.5, 0.5),
            "Residual",
        ),
    ]
    for j, (t, arr, cm, lim, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 2 * j : 2 * j + 2])
        ax.imshow(
            shade[sl],
            extent=wext,
            cmap="gray",
            vmin=-0.6,
            vmax=1.15,
            interpolation="none",
        )
        im = ax.imshow(
            arr[sl],
            extent=wext,
            cmap=cm,
            vmin=lim[0],
            vmax=lim[1],
            interpolation="none",
        )
        ticks(ax, 5)
        ax.set_xlabel("Easting (km)")
        if j == 0:
            ax.set_ylabel("Northing (km)")
        cbar = fig.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2, extend="both"
        )
        cbar.set_label(lab, fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)
        tag(ax, t)
    fig.savefig(out / "fig_model_performance.pdf")
    fig.savefig(out / "fig_model_performance.png", dpi=600)
    print(
        f"R2 {r2:.3f} RMSE {rmse:.4f} n {ok.sum()} window fold {kmed + 1} "
        f"at row {r0} col {c0}; fold R2 {np.round(fold_r2, 3)}"
    )


if __name__ == "__main__":
    main()
