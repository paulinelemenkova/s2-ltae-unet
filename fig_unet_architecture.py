from __future__ import annotations

import sys
from itertools import combinations, pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
_fp = FontProperties(family=list(plt.rcParams["font.sans-serif"]))
assert "dejavu" not in findfont(_fp).lower(), "DejaVu fallback: Nimbus Sans missing"

W, H = 17.0, 8.6
HEAD_PT, BODY_PT = 9.0, 8.0
ORANGE, SKY, PINK, GREEN, VERM = "#E69F00", "#56B4E9", "#CC79A7", "#009E73", "#D55E00"
GREY = "0.25"

T_MONTHS, N_CH, SIZE = 12, 4, 32
N_STATIC = 9
LEVELS = [32, 64, 128]
BOTTLENECK = 256

PROTECTED: list[tuple[float, float, float, float]] = []
INSIDE: list[Text] = []
PATHS: list[list[tuple[float, float]]] = []


def mix(color: str, white: float) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r + (1 - r) * white, g + (1 - g) * white, b + (1 - b) * white)


def dark(color: str, k: float = 0.55) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r * k, g * k, b * k)


def bar(ax: Axes, xc: float, yc: float, w: float, h: float, color: str) -> None:
    ax.add_patch(
        Rectangle(
            (xc - w / 2, yc - h / 2),
            w,
            h,
            fc=mix(color, 0.55),
            ec=dark(color, 0.8),
            lw=0.8,
            zorder=2,
        )
    )
    PROTECTED.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))


def stack(
    ax: Axes, x0: float, y0: float, size: float, n: int, color: str, dxy: float
) -> None:
    for i in reversed(range(n)):
        ax.add_patch(
            Rectangle(
                (x0 + i * dxy, y0 + i * dxy),
                size,
                size,
                fc=mix(color, 0.75),
                ec=dark(color, 0.8),
                lw=0.7,
                zorder=2 + (n - i) * 0.01,
            )
        )
    PROTECTED.append((x0, y0, x0 + size + (n - 1) * dxy, y0 + size + (n - 1) * dxy))


def arrow(
    ax: Axes,
    verts: list[tuple[float, float]],
    color: str = GREY,
    ls: str | tuple[int, tuple[int, int]] = "-",
    lw: float = 1.0,
) -> None:
    PATHS.append(verts)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(
        FancyArrowPatch(
            path=MplPath(verts, codes),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def txt(
    ax: Axes, x: float, y: float, s: str, inside: bool = False, **kw: object
) -> Text:
    kw.setdefault("fontsize", BODY_PT)
    kw.setdefault("color", "0.10")
    t = ax.text(x, y, s, zorder=5, linespacing=1.15, **kw)
    if inside:
        INSIDE.append(t)
    return t


def main(out: Path) -> None:
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    y_lev = [6.55, 4.75, 3.25, 2.05]
    h_lev = [2.3, 1.6, 1.05, 0.62]
    w_ch = dict(zip([*LEVELS, BOTTLENECK], [0.24, 0.32, 0.42, 0.54], strict=True))
    x_enc = [6.05, 7.25, 8.45]
    x_bot = 9.65
    x_dec = [10.85, 12.05, 13.25][::-1]

    sq, d = 1.25, 0.06
    in_x0 = 0.45
    stack(ax, in_x0, y_lev[0] - sq / 2 - 0.3, sq, T_MONTHS, SKY, d)
    in_x1 = in_x0 + sq + (T_MONTHS - 1) * d
    txt(
        ax,
        0.25,
        y_lev[0] + 1.27,
        "Input image time series",
        ha="left",
        va="bottom",
        fontsize=HEAD_PT,
        fontweight="bold",
        color=dark(SKY),
    )
    txt(
        ax,
        0.25,
        y_lev[0] - 1.2,
        f"{T_MONTHS} post-fire months\n{N_CH} channels: NBR, NDVI,\nNDMI, clear mask\n{SIZE} × {SIZE} pixels",
        ha="left",
        va="top",
    )

    lx0, lx1 = 2.85, 5.6
    ly0, ly1 = y_lev[0] - 1.0, y_lev[0] + 1.0
    ax.add_patch(
        FancyBboxPatch(
            (lx0, ly0),
            lx1 - lx0,
            ly1 - ly0,
            boxstyle="round,pad=0,rounding_size=0.12",
            fc=mix(VERM, 0.82),
            ec=VERM,
            lw=1.0,
            zorder=2,
        )
    )
    PROTECTED.append((lx0, ly0, lx1, ly1))
    lcx = (lx0 + lx1) / 2
    txt(
        ax,
        lcx,
        ly1 - 0.15,
        "L-TAE",
        inside=True,
        ha="center",
        va="top",
        fontsize=HEAD_PT,
        fontweight="bold",
        color=dark(VERM),
    )
    txt(
        ax,
        lcx,
        ly1 - 0.62,
        "temporal attention\nper pixel; 4 heads;\n"
        f"{T_MONTHS} × {N_CH} → {LEVELS[0]} features\n+ {N_STATIC} static layers",
        inside=True,
        ha="center",
        va="top",
    )
    arrow(ax, [(in_x1 + 0.05, y_lev[0]), (lx0, y_lev[0])])

    for i, c in enumerate(LEVELS):
        bar(ax, x_enc[i], y_lev[i], w_ch[c], h_lev[i], VERM)
        bar(ax, x_dec[i], y_lev[i], w_ch[c], h_lev[i], VERM)
        for xx in (x_enc[i], x_dec[i]):
            txt(
                ax, xx, y_lev[i] + h_lev[i] / 2 + 0.06, str(c), ha="center", va="bottom"
            )
    bar(ax, x_bot, y_lev[3], w_ch[BOTTLENECK], h_lev[3], VERM)
    txt(
        ax,
        x_bot,
        y_lev[3] + h_lev[3] / 2 + 0.06,
        f"{SIZE // 8} × {SIZE // 8}\n{BOTTLENECK} + {N_STATIC}",
        ha="center",
        va="bottom",
    )
    txt(
        ax,
        (x_enc[0] + x_dec[0]) / 2,
        y_lev[0] + 1.53,
        "U-Net encoder–decoder",
        ha="center",
        va="bottom",
        fontsize=HEAD_PT,
        fontweight="bold",
        color=dark(VERM),
    )
    arrow(ax, [(lx1, y_lev[0]), (x_enc[0] - w_ch[LEVELS[0]] / 2, y_lev[0])])

    xs_down = [*x_enc, x_bot]
    ws_down = [w_ch[c] for c in LEVELS] + [w_ch[BOTTLENECK]]
    ys, hs = y_lev, h_lev
    for i in range(3):
        arrow(
            ax,
            [
                (xs_down[i], ys[i] - hs[i] / 2),
                (xs_down[i], ys[i + 1]),
                (xs_down[i + 1] - ws_down[i + 1] / 2, ys[i + 1]),
            ],
            color=VERM,
        )
    xs_up = [x_bot, x_dec[2], x_dec[1], x_dec[0]]
    ws_up = [w_ch[BOTTLENECK], w_ch[LEVELS[2]], w_ch[LEVELS[1]], w_ch[LEVELS[0]]]
    lev_up = [3, 2, 1, 0]
    for i in range(3):
        lo, hi = lev_up[i], lev_up[i + 1]
        arrow(
            ax,
            [
                (xs_up[i] + ws_up[i] / 2, ys[lo]),
                (xs_up[i + 1], ys[lo]),
                (xs_up[i + 1], ys[hi] - hs[hi] / 2),
            ],
            color=GREEN,
        )

    for i, c in enumerate(LEVELS):
        x0 = x_enc[i] + w_ch[c] / 2
        x1 = x_dec[i] - w_ch[c] / 2
        arrow(ax, [(x0, ys[i]), (x1, ys[i])], color="0.45", ls=(0, (3, 2)), lw=0.9)
        s = SIZE // 2**i
        txt(
            ax,
            (x0 + x1) / 2,
            ys[i] + 0.06,
            f"{s} × {s}",
            ha="center",
            va="bottom",
            color="0.30",
        )

    sx0, sy0, ssq = 10.4, 0.22, 0.85
    stack(ax, sx0, sy0, ssq, N_STATIC, SKY, 0.05)
    arrow(
        ax,
        [(sx0, sy0 + ssq / 2), (x_bot, sy0 + ssq / 2), (x_bot, ys[3] - hs[3] / 2)],
        color=GREY,
    )
    txt(
        ax,
        sx0 + ssq + 8 * 0.05 + 0.15,
        sy0 + (ssq + 8 * 0.05) / 2,
        f"{N_STATIC} static covariates ({SIZE} × {SIZE}): joined\nwith the L-TAE features and, average-\n"
        f"pooled to {SIZE // 8} × {SIZE // 8}, at the bottleneck",
        ha="left",
        va="center",
    )

    ox0, osq = 14.45, 1.5
    oy0 = ys[0] - osq / 2
    arrow(ax, [(x_dec[0] + w_ch[LEVELS[0]] / 2, ys[0]), (ox0, ys[0])], color=GREY)
    txt(
        ax, (x_dec[0] + ox0) / 2 + 0.05, ys[0] + 0.06, "1 × 1", ha="center", va="bottom"
    )
    ax.add_patch(
        Rectangle(
            (ox0, oy0),
            osq,
            osq,
            fc=mix(PINK, 0.75),
            ec=dark(PINK, 0.8),
            lw=0.8,
            zorder=2,
        )
    )
    PROTECTED.append((ox0, oy0, ox0 + osq, oy0 + osq))
    ocx = ox0 + osq / 2
    txt(
        ax,
        ocx,
        oy0 + osq + 0.12,
        "Forecast RRI",
        ha="center",
        va="bottom",
        fontsize=HEAD_PT,
        fontweight="bold",
        color=dark(PINK),
    )
    txt(ax, ocx, oy0 - 0.1, f"{SIZE} × {SIZE} × 1", ha="center", va="top")
    txt(
        ax,
        ocx,
        oy0 - 0.62,
        "masked Huber loss\nagainst observed\nRRI; AdamW",
        ha="center",
        va="top",
    )

    lgx, lgy = 0.25, 2.55
    items: list[tuple[str, str]] = [
        ("bar", "feature map: 2 × (conv 3 × 3, BN, ReLU)"),
        (VERM, "max-pooling 2 × 2"),
        (GREEN, "up-convolution 2 × 2"),
        ("skip", "skip connection: copy and concatenate"),
        (GREY, "1 × 1 convolution, data flow"),
    ]
    for k, (kind, label) in enumerate(items):
        y = lgy - k * 0.48
        if kind == "bar":
            ax.add_patch(
                Rectangle(
                    (lgx + 0.2, y - 0.2),
                    0.16,
                    0.4,
                    fc=mix(VERM, 0.55),
                    ec=dark(VERM, 0.8),
                    lw=0.8,
                )
            )
        elif kind == "skip":
            arrow(ax, [(lgx, y), (lgx + 0.62, y)], color="0.45", ls=(0, (3, 2)), lw=0.9)
        else:
            arrow(ax, [(lgx, y), (lgx + 0.62, y)], color=kind)
        txt(ax, lgx + 0.8, y, label, ha="left", va="center")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes = [(t, t.get_window_extent(r).transformed(inv)) for t in ax.texts]
    tt = sum(
        1
        for (_, a), (_, b) in combinations(boxes, 2)
        if a.x1 > b.x0 and a.x0 < b.x1 and a.y1 > b.y0 and a.y0 < b.y1
    )
    tp = 0
    for t, bb in boxes:
        if t in INSIDE:
            continue
        for x0, y0, x1, y1 in PROTECTED:
            if bb.x1 > x0 and bb.x0 < x1 and bb.y1 > y0 and bb.y0 < y1:
                tp += 1
    ta = 0
    for t, bb in boxes:
        for poly in PATHS:
            hit = False
            for (xa, ya), (xb, yb) in pairwise(poly):
                for k in range(41):
                    px, py = xa + (xb - xa) * k / 40, ya + (yb - ya) * k / 40
                    if (
                        bb.x0 - 0.03 < px < bb.x1 + 0.03
                        and bb.y0 - 0.03 < py < bb.y1 + 0.03
                    ):
                        hit = True
            ta += hit
            if hit:
                print("  text on arrow:", repr(t.get_text()[:30]))
    edge = sum(1 for _, bb in boxes if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H)
    for t, bb in boxes:
        if t in INSIDE:
            x0, y0, x1, y1 = next(
                p for p in PROTECTED if p[0] <= t.get_position()[0] <= p[2]
            )
            if bb.x0 < x0 or bb.x1 > x1:
                print("  text wider than its box:", repr(t.get_text()[:30]))
    print(
        f"text-text: {tt}; text on bars/boxes: {tp}; text on arrows: {ta}; off canvas: {edge}"
    )

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_unet_architecture.pdf")
    fig.savefig(out / "fig_unet_architecture.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
