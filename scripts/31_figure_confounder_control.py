#!/usr/bin/env python3
"""Draw the confounder-control figures from stored results.

Two figures, both read entirely from files under `results/`. No value is typed into this
script: if a number is not in `caliper_series.csv` or `ladder.csv`, it does not appear in the
manuscript.

  fig_conditional_caliper   discrimination as the evaluation is restricted to patient pairs
                            the confounder cannot separate, one panel per cohort. The
                            confounder's own curve is drawn alongside as the control: it must
                            fall to chance, and reading the imaging curve is only meaningful
                            where it has.
  fig_matching_ladder       what each rung of the matched-cohort ladder costs and buys --
                            patients retained against the discrimination the confounder still
                            has after the control.

Both are drawn against chance rather than against each other, because once the confounder is
controlled its own curve sits at 0.5 and "better than the confounder" stops being informative.

Example
-------
    python scripts/31_figure_confounder_control.py
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.figures import GRID, INK, MUTED, NEUTRAL, SERIES, legend_below

# `SERIES` holds the categorical slots of the validated default palette, in its fixed
# order. These are line series, which the palette validates on its adjacent pairlist, and
# identity is carried three times over -- hue, marker shape and line style -- so a greyscale
# print and a colourblind reader both keep the series apart without the hue. Hues are
# assigned per *arm* across the whole figure and never cycled: a sixth arm would need a
# sixth slot, not a repeated one.
CONTROL_COLOUR = NEUTRAL
# Four shapes, cycled independently of the hue; the control always takes the first.
MARKERS = ["o", "s", "^", "D"]

PANELS = [
    {"results": Path("results/confounders/dmr_session_conditional"),
     "title": "DMR-IR", "unit": "days", "confounder": "acquisition date"},
    {"results": Path("results/confounders/verification_conditional"),
     "title": "Second cohort", "unit": "years", "confounder": "age"},
]
# `rungs` names the arm whose ladder is drawn. Three arms share `results/matching/dmr`
# (the frontal probe and the two all-frames configurations); the figure shows the
# frontal one, which is the arm the full ladder was run on. Naming it here means the panel
# cannot change identity because another arm's report was regenerated last.
LADDERS = [
    {"path": Path("results/matching/dmr/ladder.csv"),
     "rungs": Path("results/matching/dmr/rung_vs_control_dmr.csv"),
     "title": "DMR-IR (acquisition date)", "unit": "days"},
    {"path": Path("results/matching/verification/ladder.csv"),
     "rungs": Path("results/matching/verification/rung_vs_control_verification.csv"),
     "title": "Second cohort (age)", "unit": "years"},
]
CONTROL = "confounder_alone_control"

# Two-column page geometry: \textwidth is 6.875in and \columnwidth 3.28in. Both figures here are
# two-panel and are placed in `figure*`, so they are drawn at the full text width and LaTeX
# includes them at 1:1 -- the point sizes set in `style()` are the point sizes on the page.
# Drawn wider than the text and scaled down to fit, 9pt type lands at 3pt and the figure is
# unreadable at print size, which is the failure these two had.
TEXT_WIDTH = 6.875


def style() -> None:
    """Small type and a light line.

    Both figures here are two-panel and full text width, so the type has to leave the panels
    room: at 9pt labels and 1.8pt series the strokes carried more of the panel than the data
    did. The sizes below are what these two need; the single-column figures of
    `scripts/33_paper_figures.py` keep the larger set in `src.utils.figures.style`.
    """
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def is_number(v) -> bool:
    """Whether a caliper label is a width (`180.0`) rather than a name (`none`)."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def order_calipers(values: pd.Series) -> list:
    """Unrestricted first, then narrowing, then the exact-stratum rung last.

    The axis reads as increasing control from left to right. `same_<column>` is not a caliper
    width and cannot be placed on a numeric scale, so it goes at the right-hand end, where it
    belongs as the strictest restriction the data admit.
    """
    labels = values.astype(str)
    numeric = sorted({float(v) for v in labels if is_number(v)}, reverse=True)
    strata = sorted({v for v in labels if v != "none" and not is_number(v)})
    return (["none"] if "none" in set(labels) else []) + numeric + strata


LABELS = {
    CONTROL: "confounder alone (control)",
    "patient_level": "ConvNeXt-T, all frames",
    "experiment_dmr_01_leakage_combined::patient_level": "ConvNeXt-T, all frames",
    "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224::patient_level":
        "Swin-T, all frames",
    "experiment_dmr_11_allframes_resnet50::patient_level": "ResNet-50, all frames",
    "convnext_tiny_linear_probe": "ConvNeXt-T, frozen probe",
    "convnext_tiny_finetune_discriminative_lr": "ConvNeXt-T, fine-tuned",
    "swin_tiny_patch4_window7_224_linear_probe": "Swin-T, frozen probe",
    "swin_tiny_patch4_window7_224_finetune_discriminative_lr": "Swin-T, fine-tuned",
}

# At most three arms are drawn beside the control. The palette's first three categorical
# slots are the ones validated for colourblind separation as an all-pairs set, and six
# overlapping interval series in one panel are unreadable regardless of colour. The full set
# is in `caliper_series.csv`; this is a selection for legibility, fixed in advance rather
# than chosen by which arms came out well, and the order below is the priority.
PREFERRED = ("::patient_level", "patient_level", "finetune_discriminative_lr", "linear_probe")
MAX_ARMS = 3


# The rung names as they are stored. Spelled out ("acquisition days matched 180") they are
# 1.5in of tick column on a 6.9in canvas, which is what squeezed the ladder panels to an
# inch and a half; each panel's title already names its confounder, so the caliper alone
# identifies the rung, in the notation the caliper figure uses for the same quantity.
CALIPER_RUNG = re.compile(r"^(?:acquisition_days|age)_matched_(\d+)$")
COHORT_LABELS = {"year_balanced": "year balanced",
                 "propensity_clinical": "propensity, clinical",
                 "propensity_with_acquisition": "propensity + date",
                 "propensity_with_age": "propensity + age"}


def cohort_label(name: str, unit: str) -> str:
    """A rung name as a short tick label: a caliper as its width, the rest spelled out."""
    m = CALIPER_RUNG.match(name)
    if m:
        return f"$\\pm${m.group(1)} {unit}"
    return COHORT_LABELS.get(name, name.replace("_", " "))


def label_of(name: str) -> str:
    """A readable series name; the model names carry the encoder and the arm."""
    return LABELS.get(name, name.replace("_", " "))


def select_arms(series: pd.DataFrame) -> list[str]:
    """The control plus at most `MAX_ARMS` imaging arms, by the fixed priority above."""
    arms = [a for a in series["score"].unique() if a != CONTROL]
    ordered = sorted(arms, key=lambda a: next((i for i, p in enumerate(PREFERRED) if p in a),
                                              len(PREFERRED)))
    return [CONTROL] + ordered[:MAX_ARMS]


def encoding(labels: list[str]) -> dict[str, dict]:
    """One colour, marker and line style per *arm*, shared by every panel.

    Assigned globally rather than per panel: the same hue must mean the same arm in both
    cohorts, or a reader comparing the panels is misled by the legend. The control keeps a
    neutral grey and a dashed line in every figure.
    """
    arms = sorted(name for name in labels if name != label_of(CONTROL))
    if len(arms) > len(SERIES):
        raise SystemExit(
            f"{len(arms)} arms would need {len(arms)} hues but the palette provides "
            f"{len(SERIES)}: {arms}. Cycling a hue would make one colour mean two arms. "
            "Lower MAX_ARMS, or facet the figure.")
    out = {label_of(CONTROL): {"color": CONTROL_COLOUR, "marker": "o", "ls": "--", "z": 3}}
    for i, name in enumerate(arms):
        out[name] = {"color": SERIES[i], "marker": MARKERS[(i + 1) % len(MARKERS)],
                     "ls": ["-", "-.", ":", (0, (3, 1, 1, 1))][i % 4], "z": 4}
    return out


def draw_caliper_panel(ax, series: pd.DataFrame, panel: dict, enc: dict[str, dict]) -> list:
    """One cohort's caliper series. Returns the legend handles, which are shared.

    Identity is carried three times over -- hue, marker shape and line style -- because this
    figure is printed, and a reader in greyscale must still be able to tell the control curve
    from the imaging curves. Direct labels are deliberately not used: at the narrow end of
    the axis the curves converge to within a hundredth of each other, and labels placed there
    collide rather than clarify.
    """
    order = order_calipers(series["caliper"])
    x = np.arange(len(order))
    ax.axhline(0.5, color=GRID, lw=0.9, ls="--", zorder=0)
    ax.text(-0.5, 0.5, "chance", color=MUTED, fontsize=6, va="bottom", ha="left")

    names = select_arms(series)
    handles = []
    for i, name in enumerate(names):
        g = series[series["score"] == name].copy()
        g = g.set_index(g["caliper"].astype(str)).reindex([str(o) for o in order])
        e = enc[label_of(name)]
        # A horizontal offset per series so that overlapping intervals stay readable. Four
        # series at four calipers put sixteen intervals on one panel; at the previous 0.07
        # the caps of neighbouring arms touched and the panel read as one smear.
        dx = (i - (len(names) - 1) / 2) * 0.12
        err = np.vstack([g["c_statistic"] - g["ci_lower"], g["ci_upper"] - g["c_statistic"]])
        # A thin line and a small marker: sixteen intervals on one panel, drawn at 1.8pt
        # with 5pt markers, put more ink on the panel than the data warrant and the series
        # merged into a smear. The offsets above keep them apart; the weight need not.
        h = ax.errorbar(x + dx, g["c_statistic"], yerr=err, color=e["color"],
                        marker=e["marker"], ms=3.2, lw=1.0, capsize=1.6, elinewidth=0.7,
                        ls=e["ls"], zorder=e["z"], label=label_of(name))
        handles.append(h)

    pairs = series.drop_duplicates("caliper").copy()
    pairs = pairs.set_index(pairs["caliper"].astype(str)).reindex([str(o) for o in order])
    # "same group", never "same session": the stratum is equality of the recorded
    # acquisition date, which is all the metadata guarantees. Calling it a session would
    # assert a documented physical setup DMR-IR does not record. One line, because the
    # count printed under it makes every tick two lines deep and a three-line tick here
    # drops this panel's axis label into the shared legend.
    ticks = ["all pairs" if o == "none"
             else f"$\\pm${o:g}" if is_number(o)
             else "same group" for o in order]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{int(n)}" for t, n in zip(ticks, pairs["n_pairs"])],
                       fontsize=6.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.25, 1.02)
    # One line, not two. The second line ("admissible pairs below") explained the counts
    # already printed under each tick, and on a canvas this short it pushed the axis label
    # into the shared legend beneath it. The caption carries the explanation instead.
    ax.set_xlabel(f"caliper on {panel['confounder']} ({panel['unit']})")
    ax.set_title(panel["title"], loc="left")
    return handles


def figure_caliper(out_dir: Path) -> Path | None:
    """Conditional discrimination against the caliper, one panel per cohort."""
    panels = [(p, pd.read_csv(p["results"] / "caliper_series.csv"))
              for p in PANELS if (p["results"] / "caliper_series.csv").exists()]
    if not panels:
        return None
    enc = encoding(sorted({label_of(a) for _, series in panels
                           for a in select_arms(series)}))
    # Full text width and short. The manuscript includes this at \linewidth inside a
    # `figure*`, so the panel is reproduced 1:1 and the 8pt tick labels print at 8pt; the
    # earlier 0.74\linewidth inclusion of a figure drawn at this width printed them at
    # 5.9pt. Width bought that way is free, height is not -- a full-width float costs the
    # page twice its own height in column inches -- so the extra room the crowded left
    # panel needed is taken horizontally and the canvas stays about as tall as before.
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 2.45), sharey=True,
                             gridspec_kw={"width_ratios": [6, 5]})
    axes = list(np.atleast_1d(axes))
    handles = []
    for ax, (panel, series) in zip(axes, panels):
        handles += draw_caliper_panel(ax, series, panel, enc)
    # Two lines: the single-line form is taller than the shortened axes and is clipped.
    axes[0].set_ylabel("discrimination within\nadmissible pairs")
    # One legend for both panels: the arms differ between cohorts, so every entry is kept and
    # duplicates (the shared control) are collapsed by label.
    seen, entries = set(), []
    for h in handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            entries.append(h)
    legend = fig.legend(entries, [h.get_label() for h in entries], loc="lower center",
                        ncol=min(3, len(entries)), frameon=False, bbox_to_anchor=(0.5, 0.0),
                        columnspacing=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.93, wspace=0.07)
    # The gap under the axes has to hold two-line x ticks and then the key below them, and
    # the key is three entries wide on one row and four on two. Measured, not guessed: a
    # bottom margin chosen by eye put the key on top of the axis label the moment either
    # the font size or the number of entries changed.
    legend_below(fig, axes, legend, gap=0.07)
    path = out_dir / "fig_conditional_caliper.pdf"
    fig.savefig(path, pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), pad_inches=0.02)
    plt.close(fig)
    return path


def figure_ladder(out_dir: Path) -> Path | None:
    """Each rung: what the confounder still discriminates, what the image does, and what an
    equally small cohort with the confounding left intact does.

    The third element is what makes the first two readable. A matched rung removes the
    confounder and shrinks the cohort together, so imaging discrimination falling on it means
    nothing until an equally small cohort with the confounding still in place has been shown
    not to fall as far. The size-control band is that comparison, drawn as the range of three
    random draws.
    """
    ladders = [(spec, pd.read_csv(spec["rungs"])) for spec in LADDERS
               if spec["rungs"].exists()]
    if not ladders:
        return None
    # Same reasoning as the caliper figure: full text width so that LaTeX includes it at
    # 1:1 and the `n=` column prints at 7pt rather than 5.2pt, and no taller than it must
    # be, because a full-width float is charged to the page twice over.
    fig, axes = plt.subplots(1, len(ladders), figsize=(TEXT_WIDTH, 2.55))
    axes = list(np.atleast_1d(axes))
    for ax, (spec, table) in zip(axes, ladders):
        table = table.iloc[::-1].reset_index(drop=True)          # original at the top
        y = np.arange(len(table))
        ax.axvline(0.5, color=GRID, lw=0.9, ls="--", zorder=0)

        for yi, row in zip(y, table.itertuples()):
            if np.isfinite(row.control_auroc_min) and row.n_size_controls:
                ax.plot([row.control_auroc_min, row.control_auroc_max], [yi + 0.22] * 2,
                        color=SERIES[1], lw=3, solid_capstyle="butt", alpha=0.4, zorder=2)
                ax.plot(row.control_auroc_median, yi + 0.22, marker="|", color=SERIES[1],
                        ms=6, mew=1.1, zorder=3)
            ax.errorbar(row.image_auroc, yi, xerr=[[row.image_auroc - row.ci_lower],
                                                   [row.ci_upper - row.image_auroc]],
                        color=SERIES[0], marker="o", ms=4, capsize=1.8, elinewidth=0.7,
                        ls="", zorder=4)
            ax.plot(row.confounder_auroc, yi - 0.22, marker="s", color=CONTROL_COLOUR, ms=4,
                    ls="", zorder=4)
            # The patients this arm *scored*, not the cohort's nominal size. The frontal
            # ladder drawn here is fitted on the 231 patients holding a static frontal
            # view, so its reference rung is 231 of 236 and its overlap rung 58 of 60;
            # printing the cohort size would credit it with patients no image reached.
            ax.text(1.03, yi, f"n={int(row.n_scored)}", fontsize=6.5, va="center", ha="left",
                    color=MUTED, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels([cohort_label(c, spec["unit"]) for c in table["cohort"]],
                           fontsize=7)
        # The axis runs past 1.0 to hold the `n=` column inside the panel. Outside it, that
        # column had to be paid for twice over -- once in the gap between the panels and
        # once in the right margin -- and both came out of the panels themselves.
        ax.set_xlim(0.15, 1.18)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("AUROC")
        ax.set_title(spec["title"], loc="left")

    handles = [plt.Line2D([], [], color=SERIES[0], marker="o", ls="", ms=5,
                          label="thermogram on this cohort [95% CI]"),
               plt.Line2D([], [], color=SERIES[1], lw=3.5, alpha=0.4,
                          label="thermogram on size-matched controls (range of 3 draws)"),
               plt.Line2D([], [], color=CONTROL_COLOUR, marker="s", ls="", ms=5,
                          label="confounder alone, after the control")]
    # Two rows, not three columns: these three labels on one line are 8.3in wide, which is
    # wider than the canvas, and `savefig(bbox="tight")` then widens the *file* to hold them
    # -- so LaTeX scales the whole figure down to \textwidth and the type shrinks with it.
    legend = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                        bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handletextpad=0.5)
    # The gap between the panels carries the right panel's rung names *and* the left panel's
    # `n=` column, so it stays wider than a normal inter-panel gap -- but only as wide as
    # those two need. At the previous 1.15 the two panels together held a quarter of the
    # canvas and every stroke in them was drawn over an inch and a half of axis.
    fig.subplots_adjust(wspace=0.60, left=0.175, right=0.995, bottom=0.20, top=0.92)
    legend_below(fig, axes, legend, gap=0.07)
    path = out_dir / "fig_matching_ladder.pdf"
    fig.savefig(path, pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), pad_inches=0.02)
    plt.close(fig)
    return path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    figures = (figure_caliper(args.output_dir), figure_ladder(args.output_dir))
    written = [f for f in figures if f]
    if not written:
        raise SystemExit("No result files found. Run scripts/29 and scripts/30 first.")
    for path in written:
        print(f"Wrote {path} and {path.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
