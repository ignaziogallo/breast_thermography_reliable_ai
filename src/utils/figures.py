"""Shared drawing conventions for every figure in the manuscript.

One place for the palette, the rcParams and the small helpers, so that two figure scripts
cannot drift into two visual languages. Every figure in this repository is generated from a
file under `results/`; nothing here reads a number, and nothing anywhere types one.

**Colour.** The categorical slots below are the validated default palette in its fixed
order. They are assigned per *entity* -- a channel family, an encoder, an arm -- and never
cycled: a fifth series needs a fifth slot, not a repeated one, and `series_colour` raises
rather than silently giving one hue two meanings. Identity is always carried a second time
by marker shape, so a greyscale print and a colourblind reader lose nothing.

**Intervals.** Every point that has a bootstrap interval is drawn with it. A bare point
estimate on a cohort of this size would misrepresent what the study can resolve, which is
the paper's own argument.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL = "#52514e"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#ffffff"
MARKERS = ["o", "s", "^", "D", "v"]
CHANCE = 0.5


def style() -> None:
    """Journal-figure rcParams: small type, no chartjunk, recessive axes."""
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def series_colour(index: int, total: int) -> str:
    if total > len(SERIES):
        raise SystemExit(f"{total} series would need {total} hues; the palette provides "
                         f"{len(SERIES)}. Facet the figure rather than reusing a hue.")
    return SERIES[index]


def chance_line(ax, x_text: float = 0.0, horizontal: bool = True) -> None:
    """The 0.5 reference every discrimination figure is read against."""
    (ax.axhline if horizontal else ax.axvline)(CHANCE, color=GRID, lw=1.2, ls="--", zorder=0)
    if horizontal:
        ax.text(x_text, CHANCE, " chance", color=MUTED, fontsize=7, va="bottom", ha="left")
    else:
        ax.text(CHANCE, 1.0, " chance", color=MUTED, fontsize=7, va="top", ha="left",
                transform=ax.get_xaxis_transform())


def legend_below(fig, axes, legend, gap: float = 0.06) -> None:
    """Lift `axes` until their x labels clear a legend pinned to the bottom of the canvas.

    `fig.legend(loc="lower center")` and `ax.set_xlabel` are laid out independently, so a
    bottom margin chosen by eye holds only until a label wraps or a font size changes --
    after which the key sits on top of the axis label. This measures both, in the geometry
    that will actually be written, and moves the axes, which is the one of the two that can
    move. `gap` is in inches.
    """
    axes = [axes] if not isinstance(axes, (list, tuple)) else list(axes)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    top = legend.get_window_extent(r).transformed(inv).y1
    bottom = min(ax.get_tightbbox(r).transformed(inv).y0 for ax in axes)
    shift = top + gap / fig.get_figheight() - bottom
    if shift <= 0:
        return
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + shift, box.width,
                         max(box.height - shift, 0.05)])


def save(fig, path, also_png: bool = True, pad: float | None = None) -> list:
    r"""Write the PDF the manuscript includes, and a PNG for quick inspection.

    `pad` overrides the 0.1in `savefig` adds around the tight bounding box. A figure drawn
    at exactly the column width to be included at `\linewidth` needs that padding down to
    a hairline, or the file is wider than the column and LaTeX scales the type down.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if pad is None else {"pad_inches": pad}
    written = [path]
    fig.savefig(path, **kw)
    if also_png:
        fig.savefig(path.with_suffix(".png"), **kw)
        written.append(path.with_suffix(".png"))
    plt.close(fig)
    return written
