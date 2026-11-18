#!/usr/bin/env python3
"""The manuscript's figures, generated from stored results.

Five figures, each reading only files under `results/`. An arm that has not been run is
skipped and named in the output rather than being drawn from a remembered number.

  A  audit    the question the study asks, as a diagram: the confounder reaches the
              diagnosis and the image, and four measurable arrows follow from that.
  B  channels every information channel on the same folds, with intervals.
  D  shortcut acquisition-era decodability against diagnostic discrimination, one point per
              encoder and regime. The adversarial sweep of the protocol would populate this
              same axis pair; until it exists, the encoder family is what occupies it.
  E  split    the split unit crossed with the unit of analysis, with the contamination each
              cell carries.
  F  cohorts  both cohorts side by side: the dominant non-image channel, the image, and the
              two combined.

Figure C -- the confounder-control ladder and the caliper series -- is drawn by
`scripts/31_figure_confounder_control.py`, which owns the matched-cohort results.

Example
-------
    python scripts/33_paper_figures.py                 # all of them
    python scripts/33_paper_figures.py --figures B F
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.quality_gate import QUARANTINE_SUFFIX
from src.evaluation.summary import stored_auroc_interval as auroc_ci
from src.utils.figures import (
    GRID,
    INK,
    MARKERS,
    MUTED,
    NEUTRAL,
    SERIES,
    chance_line,
    legend_below,
    save,
    style,
)

R = Path("results")

# Two-column page geometry and type. \columnwidth is 3.28in; a figure drawn at that width and
# included at `\linewidth` is reproduced 1:1. The manuscript is set in 10pt Times and the
# figures in DejaVu Sans, matplotlib's face, which carries a taller x height and a heavier
# stroke at the same nominal size: 8.5pt of it reads as the 10pt Times beside it, which is
# what an axis label set at LABEL_PT is meant to do. Matching the nominal size instead
# would print a label visibly larger than the running text.
COLUMN_WIDTH, LABEL_PT = 3.28, 8.5

# (label, family, path, model). `family` drives colour, so a hue means a kind of evidence
# rather than a row position.
CHANNELS = [
    ("Acquisition date", "non-image", R / "experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
     "acquisition_date_only"),
    ("Tabular record, all fields", "non-image",
     R / "experiment_dmr_04_metadata_leakage/bootstrap_ci.csv", "as_published_all_fields"),
    ("Age pair (reconstructs the calendar)", "non-image",
     R / "experiment_dmr_04_metadata_leakage/bootstrap_ci.csv", "age_pair_only"),
    ("Clinical text, TF-IDF", "non-image", R / "experiment_dmr_05_text_probe/bootstrap_ci.csv",
     "all_text"),
    ("Age alone", "non-image", R / "experiment_dmr_07_metadata/bootstrap_ci.csv",
     "logistic_age_only"),
    ("Thermogram, image-level split", "contaminated",
     R / "experiment_dmr_01_leakage_combined/metrics.csv", "image_level__per_patient"),
    ("Thermogram, all frames (Swin-T)", "image",
     R / "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224/metrics.csv",
     "patient_level__per_patient"),
    ("Thermogram, all frames (ConvNeXt-T)", "image",
     R / "experiment_dmr_01_leakage_combined/metrics.csv", "patient_level__per_patient"),
    ("Thermogram, frontal, fine-tuned", "image",
     R / "experiment_dmr_11_encoder_convnext_tiny_finetune/bootstrap_ci.csv",
     "convnext_tiny_finetune_discriminative_lr"),
    ("Thermogram, frontal, frozen probe", "image",
     R / "experiment_dmr_02_frontal_static/bootstrap_ci.csv", "convnext_tiny_linear_probe"),
    ("Date + image (fixed combination)", "combined",
     R / "experiment_dmr_13_incremental_swin_allframes/bootstrap_ci.csv",
     "visit_date_plus_image_fixed"),
]
FAMILY_COLOUR = {"non-image": SERIES[1], "image": SERIES[0], "combined": SERIES[2],
                 "contaminated": NEUTRAL}
FAMILY_MARKER = {"non-image": "s", "image": "o", "combined": "^", "contaminated": "x"}

# Encoder -> (era decoder among benign patients, diagnostic arms). The era decoder is the
# x axis of figure D and the diagnostic arm the y axis.
def _ci(run: str) -> Path:
    return R / run / "bootstrap_ci.csv"


SWIN = "swin_tiny_patch4_window7_224"
ENCODERS = {
    "ConvNeXt-T": {
        "era": _ci("experiment_dmr_12_epoch_convnext_tiny_within_benign"),
        "arms": {"frozen probe": (_ci("experiment_dmr_11_encoder_convnext_tiny_probe"),
                                  "convnext_tiny_linear_probe"),
                 "fine-tuned": (_ci("experiment_dmr_11_encoder_convnext_tiny_finetune"),
                                "convnext_tiny_finetune_discriminative_lr")}},
    "Swin-T": {
        "era": _ci(f"experiment_dmr_12_epoch_{SWIN}_within_benign"),
        "arms": {"frozen probe": (_ci(f"experiment_dmr_11_encoder_{SWIN}_probe"),
                                  f"{SWIN}_linear_probe"),
                 "fine-tuned": (_ci(f"experiment_dmr_11_encoder_{SWIN}_finetune"),
                                f"{SWIN}_finetune_discriminative_lr")}},
    "ResNet-50": {
        "era": _ci("experiment_dmr_12_epoch_resnet50_within_benign"),
        "arms": {"frozen probe (failed gate)":
                 (_ci("experiment_dmr_11_encoder_resnet50_probe" + QUARANTINE_SUFFIX),
                  "resnet50_linear_probe")}},
}

INCREMENTAL_DMR = R / "experiment_dmr_13_incremental_swin_allframes/bootstrap_ci.csv"

COHORTS = [
    {"title": "DMR-IR", "confounder": "acquisition date",
     "arms": [("confounder alone", INCREMENTAL_DMR, "visit_date_only_reference"),
              ("image alone", INCREMENTAL_DMR, "image_only_reference"),
              ("confounder + image", INCREMENTAL_DMR, "visit_date_plus_image_fixed")]},
    {"title": "Second cohort", "confounder": "age",
     "arms": [("confounder alone", R / "experiment_08_multimodal/bootstrap_ci.csv",
               "age_only_reference"),
              ("image alone", R / "experiment_08_multimodal/bootstrap_ci.csv",
               "image_only_reference"),
              ("confounder + image", R / "experiment_08_multimodal/bootstrap_ci.csv",
               "age_plus_image_fixed")]},
]


# --------------------------------------------------------------------------------------
# A -- Figure 1. Left: the structure a patient-level split cannot touch, drawn from the
# manifest rather than asserted. Right: the five audit questions, each with the answer this
# benchmark returns, every number read from the file that produced it.
#
# Sized for a two-column `figure*`: at 7.0in it is reproduced near 1:1 across both columns.
# The previous version was drawn at 7.2in and included at a single column's \linewidth,
# which scaled its 8pt labels to under 4pt.
# --------------------------------------------------------------------------------------
#
# Q1 is the *fitted* calendar arm of the channels table -- a logistic model of the visit
# date on the same patient-level folds as every other channel -- so that this panel and
# Table 1 report one estimator. The raw date scored directly reads 0.981 and is the
# statistic the matched ladder and the conditional analysis use; the two are different
# estimators and the manuscript keeps both.
#
# Q2 is the per-encoder era decoder of the encoder-robustness table
# (`experiment_dmr_12_*`), not the earlier pilot of the same configuration
# (`experiment_dmr_03_epoch_within_benign`, 0.814). One family of runs supplies every era
# number the paper prints.
AUDIT_ROWS = [
    (r"$C \rightarrow Y$", "Does the calendar predict $Y$?",
     R / "experiment_dmr_04_metadata_leakage/bootstrap_ci.csv", "acquisition_date_only",
     "fitted on the folds of Table 1; above every imaging arm"),
    (r"$X \rightarrow C \mid Y$", "Is it legible in the pixels?",
     R / "experiment_dmr_12_epoch_convnext_tiny_within_benign/bootstrap_ci.csv",
     "epoch_decoder_within_benign", "diagnosis held fixed; benign stratum"),
    (r"score $\rightarrow C \mid Y$", "Do diagnostic scores carry it?",
     None, None, None),
    (r"$\Delta(X \mid C)$", "Does the image add to it?",
     None, None, "paired on identical folds; CI crosses zero"),
    (r"$X \rightarrow Y \mid C$", "What survives control?",
     R / "experiment_dmr_10_matched_overlap/bootstrap_ci.csv", None,
     r"date-overlap cohort, $n=60$"),
]


def _exploitation_note(df: pd.DataFrame) -> str:
    """`1 of 9 arms (10 runs): the all-frames probe`, counted from the table itself.

    Arms and runs are different units and the sublabel has to say which it means. The file
    holds one row per *run*; two of those rows are the same configuration trained twice, so
    the number of distinct arms is the number of distinct (frames, regime, encoder) recipes.
    The arm's own name is read from the row rather than typed, because "five views" was
    wrong for it: the all-frames configuration passes `protocol=all`, which is every static
    view *and* the dynamic sequence of the visit.
    """
    top = df.sort_values("era_auroc_benign", ascending=False).iloc[0]
    positive = int((df["era_auroc_benign_perm_p"] < 0.05).sum())
    n_arms = df["arm"].str.replace(r"\s*\(encoder run\)$", "", regex=True).nunique()
    scope = str(top["views"]).replace("all frames", "all-frames").replace("frontal", "frontal")
    return f"{positive} of {n_arms} arms ({len(df)} runs): the {scope} probe"


def _audit_answers() -> list[tuple[str, str, str, str]]:
    """The five answers, each pulled from the result file that established it."""
    rows = []
    for symbol, question, path, model, note in AUDIT_ROWS:
        if path is not None:
            got = auroc_ci(path, model)
            value = f"{got[0]:.3f}" if got else "not run"
        elif symbol.startswith("score"):
            df = pd.read_csv(R / "confounders/exploitation_interval.csv")
            top = df.sort_values("era_auroc_benign", ascending=False).iloc[0]
            value = f"{top['era_auroc_benign']:.3f}"
            note = _exploitation_note(df)
        else:
            df = pd.read_csv(R / "experiment_dmr_13_incremental_swin_allframes/paired.csv")
            d = df[df["model_a"] == "visit_date_plus_image_fixed"].iloc[0]
            value = f"{d['delta_auroc']:+.3f}"
        rows.append((symbol, question, value, note))
    return rows


# The right panel is a legend for the audit, not a data panel, so its accent must not be a
# hue that means something in the left panel. Orange is malignant there; blue is benign.
# This is a dark teal that sits outside the categorical palette entirely, prints dark in
# greyscale, and stays separable from both under deuteranopia and protanopia.
ACCENT = "#14746f"


def figure_audit(out: Path) -> Path:
    # \textwidth is 6.875in and the float is a `figure*` included at \linewidth, so drawing
    # at exactly that width makes LaTeX reproduce the panel 1:1 and the point sizes below
    # are the point sizes on the page.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.875, 2.55),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})

    # ---- left: the acquisition structure itself -------------------------------------
    pat = pd.read_csv(R / "dmr_patient_manifest.csv")
    pat["dt"] = pd.to_datetime(pat["visit_date"], errors="coerce")
    pat = pat.dropna(subset=["dt"])
    rng = np.random.default_rng(0)
    for label, y0, name, colour in ((0, 1.0, "benign", SERIES[0]),
                                    (1, 0.0, "malignant", SERIES[1])):
        g = pat[pat["patient_binary_label"] == label]
        axL.scatter(g["dt"], y0 + rng.uniform(-0.17, 0.17, len(g)), s=7, alpha=0.65,
                    color=colour, linewidths=0, zorder=3)
        axL.text(pat["dt"].min(), y0 + 0.36, f"{name}  ($n={len(g)}$)",
                 fontsize=8, color=colour, va="bottom")

    lo = pat[pat["patient_binary_label"] == 1]["dt"].min()
    hi = pat[pat["patient_binary_label"] == 0]["dt"].max()
    axL.axvspan(lo, hi, color=GRID, alpha=0.55, zorder=0)
    axL.text(lo, -0.55, " the window both classes occupy", fontsize=7, color=MUTED,
             va="center", ha="left")

    axL.set_ylim(-0.75, 1.75)
    axL.set_yticks([])
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="x", labelsize=7)
    axL.set_title("What a patient-level split cannot move", fontsize=8, loc="left")
    # One tick per calendar year the cohort spans, bounded by the years themselves rather
    # than by matplotlib's data margins. The default margins pushed the first tick to 2013
    # and invented a 2021 one, which hid the fact that the earliest benign patients were
    # photographed in 2012 -- the 2012/2013 contrast the era analysis rests on.
    first, last = pat["dt"].min().year, pat["dt"].max().year
    axL.set_xlim(pd.Timestamp(year=first, month=1, day=1),
                 pd.Timestamp(year=last, month=12, day=31))
    axL.set_xticks([pd.Timestamp(year=y, month=1, day=1) for y in range(first, last + 1)])
    axL.set_xticklabels([str(y) for y in range(first, last + 1)])
    # "Date group", never "session": DMR-IR records the calendar date and not the setup a
    # physical acquisition session would fix, and the paper's terminology section says so.
    n_groups = pat["dt"].nunique()
    pure = int((pat.groupby("dt")["patient_binary_label"].nunique() == 1).sum())
    axL.set_xlabel(f"acquisition date  ({pure} of {n_groups} date groups are label-pure)",
                   fontsize=7.5)

    # ---- right: the five questions and what this benchmark answers -------------------
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)
    axR.axis("off")
    axR.set_title("The audit, and DMR-IR's answers", fontsize=8, loc="left")
    for i, (symbol, question, value, note) in enumerate(_audit_answers()):
        y = 0.88 - 0.185 * i
        axR.text(0.0, y, symbol, fontsize=8.0, color=ACCENT, va="center")
        axR.text(0.34, y, question, fontsize=7.6, color=INK, va="center")
        axR.text(1.0, y, value, fontsize=8.0, color=INK, va="center", ha="right",
                 fontweight="bold")
        axR.text(0.34, y - 0.075, note, fontsize=6.4, color=MUTED, va="center")

    axR.axhline(0.0, xmin=0.0, xmax=1.0, color=GRID, lw=0.8)
    axR.text(0.0, -0.07, "Patient-level splitting answers none of these.",
             fontsize=7.8, color=ACCENT, va="center", style="italic")

    fig.tight_layout(w_pad=2.2)
    return save(fig, out / "fig_audit_questions.pdf")[0]


# --------------------------------------------------------------------------------------
# B -- every channel on one axis.
# --------------------------------------------------------------------------------------
def figure_channels(out: Path) -> Path | None:
    rows = [(label, family, auroc_ci(path, model)) for label, family, path, model in CHANNELS]
    rows = [row for row in rows if row[2] is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r[2][0])

    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))
    chance_line(ax, horizontal=False)
    for i, (_label, family, (est, lo, hi)) in enumerate(rows):
        colour = FAMILY_COLOUR[family]
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], color=colour, capsize=3,
                    elinewidth=1.2, marker=FAMILY_MARKER[family], ms=6, zorder=3)
        ax.text(hi + 0.006, i, f"{est:.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.33, 1.06)
    ax.set_xlabel("AUROC [95% CI], patient-level bootstrap")
    handles = [plt.Line2D([], [], color=FAMILY_COLOUR[f], marker=FAMILY_MARKER[f], ls="",
                          ms=6, label=name)
               for f, name in [("non-image", "non-image channel"), ("image", "thermogram"),
                               ("combined", "combined"),
                               ("contaminated", "contaminated by construction")]]
    # Below the axes: the lower-right corner of the panel holds the shortest row label
    # and its value, and a legend placed there covers them.
    fig.legend(handles=handles, loc="lower center", frameon=False, ncol=4,
               bbox_to_anchor=(0.5, -0.05))
    return save(fig, out / "fig_channels.pdf")[0]


# --------------------------------------------------------------------------------------
# D -- shortcut availability against diagnostic discrimination.
# --------------------------------------------------------------------------------------
def figure_shortcut(out: Path) -> Path | None:
    points = []
    for i, (encoder, spec) in enumerate(ENCODERS.items()):
        era = auroc_ci(spec["era"], "epoch_decoder_within_benign")
        if era is None:
            continue
        for j, (arm, (path, model)) in enumerate(spec["arms"].items()):
            diag = auroc_ci(path, model)
            if diag is not None:
                points.append((encoder, arm, era, diag, i, j))
    if not points:
        return None

    # One column wide and deliberately low: the manuscript includes this at
    # `\linewidth` in a single-column float, so 3.28in of drawing is reproduced 1:1 and the
    # point sizes below are the point sizes on the page. Drawn at this width but included at
    # 0.66\linewidth, as it was, everything printed at 0.64 scale -- a 6.5pt axis label
    # landed at 4.2pt. The data occupy a band 0.15 AUROC tall, so height beyond what that
    # band needs is empty page: the canvas is a wide rectangle, not a square.
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    xs = [p[2][0] for p in points] + [p[2][1] for p in points] + [p[2][2] for p in points]
    ys = [p[3][0] for p in points] + [p[3][1] for p in points] + [p[3][2] for p in points]
    # The axes are cropped to the data. Every encoder sits far to the right of chance, so a
    # panel spanning 0.5 to 1.0 would spend two thirds of its width on empty space and
    # squeeze the contrast that matters -- that diagnostic discrimination varies while era
    # decodability does not -- into a corner. The padding is symmetric: a wide right margin
    # left the last tenth of the axis empty, which reads as data that stops rather than as
    # an axis that ends.
    xlo, xhi = min(xs) - 0.02, max(xs) + 0.02
    # The upper margin is the note's, not the data's: every interval ends below max(ys), so
    # the band above it is the only place a two-line note fits without crossing a point.
    ylo, yhi = min(ys) - 0.02, max(ys) + 0.085
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    # The qualifier the x label used to carry as a second line. Inside the panel it costs
    # nothing: the upper left is the one region of this axis no interval reaches.
    ax.text(xlo + 0.004, yhi - 0.004,
            "era AUROC is among benign patients only;\n"
            "chance (0.5) is off the left of this panel",
            fontsize=6, color=MUTED, va="top", linespacing=1.25)
    # Five ticks, not the nine the default locator puts on an axis this narrow: at column
    # width their labels run into each other.
    ax.locator_params(axis="x", nbins=5)
    ax.tick_params(labelsize=6.5)

    # A legend rather than direct labels. Two of these points very nearly coincide, so any
    # offset that keeps one label clear of its own marker drives it into its neighbour's;
    # hue identifies the encoder and marker the regime, which leaves the labels nothing to
    # disambiguate that the key does not already carry.
    for _encoder, _arm, era, diag, i, j in points:
        ax.errorbar(era[0], diag[0],
                    xerr=[[era[0] - era[1]], [era[2] - era[0]]],
                    yerr=[[diag[0] - diag[1]], [diag[2] - diag[0]]],
                    color=SERIES[i], marker=MARKERS[j % len(MARKERS)], ms=5, capsize=2,
                    elinewidth=0.8, ls="", zorder=3)
    # Two short keys rather than one entry per point: the encoder-by-regime product needs
    # five lines of text, which at column width covers the points it explains. Hue and
    # marker are independent here, so they can be read off separately.
    seen_enc, seen_arm = {}, {}
    for encoder, arm, _, _, i, j in points:
        seen_enc.setdefault(encoder, i)
        seen_arm.setdefault(j, arm)
    # The failed gate belongs on the encoder, not on its regime: it is the ResNet-50 run
    # that did not pass, and it is the only ResNet-50 point here.
    keys = [plt.Line2D([], [], color=SERIES[i], marker="o", ls="", ms=5,
                       label=e + (" (failed gate)" if e == "ResNet-50" else ""))
            for e, i in seen_enc.items()]
    keys += [plt.Line2D([], [], color=MUTED, marker=MARKERS[j % len(MARKERS)], ls="", ms=5,
                        label=a.split(" (")[0]) for j, a in sorted(seen_arm.items())]
    # Below the axes: every horizontal interval spans most of the panel, so there is no
    # rectangle inside it wide enough to hold a key without covering an interval.
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.20, top=0.96)
    legend = fig.legend(handles=keys, loc="lower center", frameon=False, fontsize=6.5,
                        ncol=2, handletextpad=0.4, columnspacing=1.1,
                        bbox_to_anchor=(0.5, 0.0))
    # Both labels on as few lines as the width allows: at body size every extra line is
    # taken out of a panel that is only about an inch tall.
    ax.set_xlabel("acquisition era decodable from the image", fontsize=LABEL_PT)
    ax.set_ylabel("diagnostic\nAUROC", fontsize=LABEL_PT)
    # The axis labels are set once the axes have their final size, and the axes are then
    # lifted off the key by a measured margin rather than by a guessed bottom margin.
    legend_below(fig, ax, legend, gap=0.05)
    return save(fig, out / "fig_shortcut_tradeoff.pdf", pad=0.02)[0]


# --------------------------------------------------------------------------------------
# E -- the split unit crossed with the unit of analysis.
# --------------------------------------------------------------------------------------
def figure_split(out: Path) -> Path | None:
    metrics = R / "experiment_dmr_01_leakage_combined/metrics.csv"
    contamination = R / "experiment_dmr_01_leakage_combined/contamination.csv"
    cells = [("Patient-level split", "per patient", "patient_level__per_patient"),
             ("Patient-level split", "per frame", "patient_level__per_frame"),
             ("Image-level split", "per patient", "image_level__per_patient"),
             ("Image-level split", "per frame", "image_level__per_frame")]
    values = [(split, unit, auroc_ci(metrics, model)) for split, unit, model in cells]
    values = [v for v in values if v[2] is not None]
    if not values:
        return None

    contam = {}
    if contamination.exists():
        c = pd.read_csv(contamination)
        if "arm" in c.columns:
            contam = c.groupby("arm")["fraction_contaminated"].mean().to_dict()

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(values))[::-1]
    chance_line(ax, horizontal=False)
    for i, (split, _unit, (est, lo, hi)) in enumerate(values):
        leaky = split.startswith("Image")
        colour = NEUTRAL if leaky else SERIES[0]
        ax.errorbar(est, y[i], xerr=[[est - lo], [hi - est]], color=colour, capsize=3,
                    elinewidth=1.2, marker="x" if leaky else "o", ms=7, zorder=3)
        ax.text(hi + 0.004, y[i], f"{est:.3f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}\n{u}" for s, u, _ in values], fontsize=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUROC [95% CI]")
    note = ("Contamination: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(contam.items()))
            if contam else
            "Contamination is the mean fraction of test frames whose patient is also in training.")
    fig.text(0.5, -0.02, note, ha="center", fontsize=7, color=MUTED)
    return save(fig, out / "fig_split_leakage.pdf")[0]


# --------------------------------------------------------------------------------------
# F -- the two cohorts, each against its own dominant non-image channel.
# --------------------------------------------------------------------------------------
def figure_cohorts(out: Path) -> Path | None:
    panels = []
    for spec in COHORTS:
        arms = [(label, auroc_ci(path, model)) for label, path, model in spec["arms"]]
        arms = [arm for arm in arms if arm[1] is not None]
        if arms:
            panels.append((spec, arms))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(3.8 * len(panels), 3.2), sharey=True)
    for ax, (spec, arms) in zip(np.atleast_1d(axes), panels):
        x = np.arange(len(arms))
        ax.axhline(0.5, color=GRID, lw=1.2, ls="--", zorder=0)
        for i, (_label, (est, lo, hi)) in enumerate(arms):
            ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], color=SERIES[i],
                        marker=MARKERS[i], ms=7, capsize=3, elinewidth=1.2, ls="", zorder=3)
            ax.text(i, hi + 0.02, f"{est:.3f}", ha="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("confounder", spec["confounder"]) for name, _ in arms],
                           fontsize=7.5, rotation=18, ha="right")
        ax.set_ylim(0.35, 1.05)
        ax.set_title(f"{spec['title']}\ndominant channel: {spec['confounder']}", loc="left",
                     fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("AUROC [95% CI]")
    fig.tight_layout()
    return save(fig, out / "fig_two_cohorts.pdf")[0]


FIGURES = {"A": figure_audit, "B": figure_channels, "D": figure_shortcut,
           "E": figure_split, "F": figure_cohorts}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES))
    p.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    style()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for key in a.figures:
        path = FIGURES[key](a.output_dir)
        print(f"{key}: {'wrote ' + str(path) if path else 'skipped, no result file yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
