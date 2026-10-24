#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.labelpad": 2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "savefig.pad_inches": 0.02,
    }
)
assert "dejavu" not in findfont(FontProperties(family=["Nimbus Sans"])).lower()

CLASSES = {
    "Low": ("#c9a400", "s"),
    "Moderate-low": ("#f08c2e", "^"),
    "Moderate-high": ("#d1362a", "D"),
    "High": ("#67001f", "v"),
}
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
C_ALL = "0.15"


def ticks(ax: Axes, ymajor: float) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(MultipleLocator(ymajor))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in", pad=2)
    ax.tick_params(which="major", length=4.5, width=0.8)
    ax.tick_params(which="minor", length=2.5, width=0.6)
    ax.grid(axis="y", which="major", lw=0.5, color="0.85")
    ax.set_axisbelow(True)


def tag(ax: Axes, t: str, x: float = -0.13, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        t,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def binned(ax: Axes, df: pd.DataFrame, xlabel: str, xmax: float) -> None:
    x = df["mid"].to_numpy(float)
    ax.fill_between(x, df.q25, df.q75, color="0.8", lw=0, label="All burned, IQR")
    ax.plot(x, df.q50, color=C_ALL, lw=2.0, label="All burned, median")
    for name, (col, mk) in CLASSES.items():
        ax.plot(
            x,
            df[name],
            color=col,
            lw=1.2,
            marker=mk,
            ms=3.5,
            mec="white",
            mew=0.3,
            label=name,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$k$ (yr$^{-1}$)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ticks(ax, 0.2)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    asp = pd.read_csv(src / "aspect_k.csv")
    elev = pd.read_csv(src / "elev_k.csv")
    slope = pd.read_csv(src / "slope_k.csv")
    clim = pd.read_csv(src / "climate_years.csv")

    fig = plt.figure(figsize=(17 / 2.54, 14 / 2.54))
    gs = fig.add_gridspec(
        2, 2, left=0.075, right=0.915, top=0.88, bottom=0.085, wspace=0.34, hspace=0.42
    )
    axa = fig.add_subplot(gs[0, 0], projection="polar")
    th = np.deg2rad(np.arange(0, 360, 45))
    for name, (col, mk) in CLASSES.items():
        v = asp[asp["class"] == name].set_index("sector").loc[SECTORS, "k_median"]
        axa.plot(
            np.r_[th, th[0]],
            np.r_[v.values, v.values[0]],
            color=col,
            lw=1.2,
            marker=mk,
            ms=3.5,
            mec="white",
            mew=0.3,
            label=name,
        )
    axa.set_theta_zero_location("N")
    axa.set_theta_direction(-1)
    axa.set_xticks(th)
    axa.set_xticklabels(SECTORS)
    axa.set_ylim(0, 0.45)
    axa.set_yticks([0.1, 0.2, 0.3, 0.4])
    axa.set_yticklabels([])
    for rv in (0.1, 0.2, 0.3, 0.4):
        axa.text(
            np.deg2rad(157.5),
            rv,
            f"{rv:.1f}",
            fontsize=7.5,
            ha="center",
            va="center",
            zorder=10,
            bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none"},
        )
    axa.tick_params(pad=1)
    axa.grid(lw=0.5, color="0.85")
    axa.text(
        np.deg2rad(157.5), 0.55, r"$k$ (yr$^{-1}$)", fontsize=8, ha="left", va="center"
    )
    tag(axa, "(a)", x=-0.13, y=1.03)
    axb = fig.add_subplot(gs[0, 1])
    binned(axb, elev, "Elevation (m)", 1400)
    tag(axb, "(b)")
    axb.xaxis.set_major_locator(MultipleLocator(200))
    axc = fig.add_subplot(gs[1, 0])
    binned(axc, slope, "Slope (°)", 45)
    tag(axc, "(c)")
    axc.xaxis.set_major_locator(MultipleLocator(10))
    axd = fig.add_subplot(gs[1, 1])
    x = np.arange(len(clim))
    cols = ["#b2182b" if v < 0 else "#2166ac" for v in clim.P_anom_pct]
    axd.bar(
        x - 0.2,
        clim.P_anom_pct,
        width=0.38,
        color=cols,
        alpha=0.55,
        lw=0,
        label="Precipitation",
    )
    axd.axhline(0, color="0.3", lw=0.7)
    axd.set_ylabel("Precipitation anomaly (%)")
    axd.set_ylim(-60, 60)
    axd.set_xticks(x)
    axd.set_xticklabels([f"{y - 1}/\n{str(y)[2:]}" for y in clim.year_end])
    axd.set_xlim(-0.6, len(clim) - 0.4)
    ticks(axd, 20)
    axd.xaxis.set_minor_locator(plt.NullLocator())
    axd.tick_params(axis="x", which="both", top=False)
    axs = axd.twinx()
    axs.plot(
        x,
        clim.SPEI12,
        color="#5e3c99",
        lw=1.2,
        marker="o",
        ms=4,
        label="SPEI-12",
    )
    axs.bar(
        x + 0.2,
        clim.RRI_increment * 5,
        width=0.38,
        color="#1b7837",
        alpha=0.85,
        lw=0,
        label=r"$\Delta$RRI $\times$5",
    )
    axs.set_ylim(-3, 3)
    axs.set_ylabel(r"SPEI-12;  $\Delta$RRI $\times$5")
    axs.yaxis.set_major_locator(MultipleLocator(1))
    axs.yaxis.set_minor_locator(AutoMinorLocator())
    axs.tick_params(which="both", direction="in", pad=2)
    tag(axd, "(d)")
    axd.text(0, 3, "fire\nyear", ha="center", va="bottom", fontsize=8)

    h, lab = axb.get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.8,
        columnspacing=1.0,
        fontsize=8.5,
    )
    h1, l1 = axd.get_legend_handles_labels()
    h2, l2 = axs.get_legend_handles_labels()
    axs.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        bbox_to_anchor=(0.07, 1.0),
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        ncol=2,
        columnspacing=0.8,
    )

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in (axb, axc, axd, axs):
        for t in ax.texts:
            bb = Text.get_window_extent(t, renderer=r)
            hits = 0
            for ln in ax.get_lines():
                xy = np.asarray(ln.get_xydata(), float)
                if xy.shape[0] <= 2:
                    continue
                p = ax.transData.transform(xy)
                hits += int(
                    (
                        (p[:, 0] >= bb.x0)
                        & (p[:, 0] <= bb.x1)
                        & (p[:, 1] >= bb.y0)
                        & (p[:, 1] <= bb.y1)
                    ).sum()
                )
            if hits:
                print("overlap", repr(t.get_text()), hits)
    leg = axs.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(r)
        n = 0
        for ax in (axd, axs):
            for patch in ax.patches:
                pb = patch.get_window_extent(r)
                n += int(pb.overlaps(bb))
            for ln in ax.get_lines():
                p = ax.transData.transform(np.asarray(ln.get_xydata(), float))
                n += int(
                    (
                        (p[:, 0] >= bb.x0)
                        & (p[:, 0] <= bb.x1)
                        & (p[:, 1] >= bb.y0)
                        & (p[:, 1] <= bb.y1)
                    ).sum()
                )
        print("legend (d) overlaps:", n)
    fig.savefig(out / "fig_topographic_controls.pdf")
    fig.savefig(out / "fig_topographic_controls.png", dpi=600)


if __name__ == "__main__":
    main()
