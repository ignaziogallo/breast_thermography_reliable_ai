#!/usr/bin/env python3
"""How much cross-label acquisition support does DMR-IR contain, and how little is that?

The conditional statistic of the paper answers "does the score still discriminate between
patients the confounder cannot separate" by restricting to admissible pairs. Its last column
-- pairs drawn from the *same acquisition-date group* -- is the only cell in which
acquisition is held exactly constant, and it holds eight pairs. This script establishes that
the eight is a property of how the cohort was recruited rather than of its size, by counting
the same statistic under the null in which diagnosis is independent of the date group:

    the same 236 patients, the same 86 date groups with the same sizes, labels permuted.

A *date group* is the set of patients sharing one recorded `visit_date`, which is the only
thing DMR-IR's metadata guarantees; it is a proxy for an acquisition session and is never
called one here, because no physical setup is documented. Everything is combinatorics over
`results/dmr_patient_manifest.csv` -- no pixels, no model, no fold -- so it runs anywhere and
is deterministic given the seed. It reports, and draws, two quantities:

    mixed date groups           date groups containing both classes (support in the sense of
                                P(Y=0 | C=c) > 0 and P(Y=1 | C=c) > 0)
    cross-label pairs           benign/malignant patient pairs sharing a date group, which is
                                the number of comparisons in which the label varies while
                                acquisition does not

The permutation is a description of the recruitment design, not a hypothesis test about a
model: it says how far the observed collinearity of label and date group is from what the
group sizes alone would produce.

Example
-------
    python scripts/41_cross_label_support.py
    python scripts/41_cross_label_support.py --permutations 10000 --output-dir paper/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.figures import GRID, MUTED, SERIES, save, style

R = Path("results")
MANIFEST = R / "dmr_patient_manifest.csv"
OUT_CSV = R / "confounders/cross_label_support.csv"


def date_groups(patients: pd.DataFrame) -> pd.DataFrame:
    """One row per acquisition-date group: how many benign and how many malignant patients."""
    g = patients.groupby("visit_date")["patient_binary_label"].agg(["size", "sum"])
    g = g.rename(columns={"size": "n", "sum": "malignant"})
    g["benign"] = g["n"] - g["malignant"]
    return g.sort_index()


def support(benign: np.ndarray, malignant: np.ndarray) -> tuple[int, int]:
    """(mixed date groups, cross-label pairs sharing a date group) for one assignment."""
    return int(((benign > 0) & (malignant > 0)).sum()), int((benign * malignant).sum())


def permute(labels: np.ndarray, sizes: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """`draws` x 2 array of (mixed date groups, cross-label pairs) under label permutation.

    Group sizes are held fixed and the labels are shuffled across patients, which is the
    null "the diagnosis a patient turned out to have is unrelated to the day they were
    photographed" with everything else about the cohort left as collected.
    """
    rng = np.random.default_rng(seed)
    cuts = np.cumsum(sizes)[:-1]
    out = np.empty((draws, 2), dtype=int)
    for i in range(draws):
        chunks = np.split(rng.permutation(labels), cuts)
        m = np.array([c.sum() for c in chunks])
        out[i] = support(np.array([len(c) for c in chunks]) - m, m)
    return out


def representative_draw(labels: np.ndarray, sizes: np.ndarray, target: int, seed: int):
    """The first permutation whose cross-label pair count is exactly the null median.

    A single draw has to stand for the null in the figure, and the honest choice is the
    typical one rather than the first: drawing until the statistic equals its median makes
    the panel's own annotation and the number quoted in the text the same number.
    """
    rng = np.random.default_rng(seed)
    cuts = np.cumsum(sizes)[:-1]
    for _ in range(10000):
        chunks = np.split(rng.permutation(labels), cuts)
        malignant = np.array([int(c.sum()) for c in chunks])
        if support(sizes - malignant, malignant)[1] == target:
            return sizes - malignant, malignant
    raise SystemExit(f"no draw in 10000 reached the median of {target}")


def figure(g: pd.DataFrame, labels: np.ndarray, median_pairs: int, seed: int,
           out: Path) -> Path:
    """Observed group composition beside a typical draw of the same cohort under the null.

    Upper and lower hold the same 236 patients, the same 86 date groups with the same sizes
    and the same class balance; only which patient was photographed on which day differs.
    What differs visibly is the quantity the paper's last question needs: date groups in
    which the label varies while acquisition does not.
    """
    drawn_b, drawn_m = representative_draw(labels, g["n"].values, median_pairs, seed)
    panels = [("DMR-IR as collected", g["benign"].values, g["malignant"].values),
              ("the same cohort, diagnosis independent of date group", drawn_b, drawn_m)]

    # One column wide and stacked, so the two designs are read against a common x axis and
    # the float costs the manuscript one column rather than two.
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.05), sharex=True, sharey=True)
    for ax, (title, benign, malignant) in zip(axes, panels):
        x = np.arange(len(benign))
        ax.bar(x, benign, width=1.0, color=SERIES[0], linewidth=0)
        ax.bar(x, malignant, width=1.0, bottom=benign, color=SERIES[1], linewidth=0)
        mixed, pairs = support(benign, malignant)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        ax.set_ylabel("patients", fontsize=7)
        ax.set_xlim(-1, len(benign))
        # Headroom for the panel's two numbers, which no bar can then reach.
        ax.set_ylim(0, g["n"].max() + 3.2)
        ax.set_yticks(np.arange(0, g["n"].max() + 1, 3))
        # Inside the panel, above the bars and hard against the right edge, where the
        # cohort's late date groups are one and two patients tall: the callout then sits
        # over blank axes rather than over the columns it is describing.
        ax.text(len(benign) - 1, g["n"].max() + 1.9,
                f"{len(benign) - mixed} of {len(benign)} date groups hold a single class; "
                f"{pairs} cross-label pairs", fontsize=6.6, color=MUTED, ha="right",
                va="center")
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["bottom"].set_color(GRID)
    axes[1].set_xlabel("acquisition-date groups, ordered by date", fontsize=7, color=MUTED)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[0], label="benign"),
                        plt.Rectangle((0, 0), 1, 1, color=SERIES[1], label="malignant")],
               loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False,
               fontsize=7, handlelength=1.1, handleheight=0.8, columnspacing=1.4)
    fig.tight_layout(h_pad=1.0)
    return save(fig, out / "fig_cross_label_support.pdf")[0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    patients = pd.read_csv(MANIFEST)
    patients = patients[patients["role"] == "Patient"]
    g = date_groups(patients)
    labels = patients["patient_binary_label"].to_numpy()
    n_benign, n_malignant = int((labels == 0).sum()), int(labels.sum())

    mixed, pairs = support(g["benign"].values, g["malignant"].values)
    null = permute(labels, g["n"].values, a.permutations, a.seed)

    rows = []
    for j, (name, observed) in enumerate((("mixed_date_groups", mixed),
                                          ("cross_label_date_group_pairs", pairs))):
        draw = null[:, j]
        rows.append({
            "statistic": name,
            "observed": observed,
            "null_median": float(np.median(draw)),
            "null_ci_lower": float(np.percentile(draw, 2.5)),
            "null_ci_upper": float(np.percentile(draw, 97.5)),
            "p_le_observed": float((draw <= observed).mean()),
            "n_permutations": a.permutations,
        })
    rows.append({"statistic": "cross_label_pairs_in_cohort",
                 "observed": n_benign * n_malignant, "null_median": float("nan"),
                 "null_ci_lower": float("nan"), "null_ci_upper": float("nan"),
                 "p_le_observed": float("nan"), "n_permutations": 0})
    rows.append({"statistic": "patients_in_mixed_date_groups",
                 "observed": int(g.loc[(g["benign"] > 0) & (g["malignant"] > 0), "n"].sum()),
                 "null_median": float("nan"), "null_ci_lower": float("nan"),
                 "null_ci_upper": float("nan"), "p_le_observed": float("nan"),
                 "n_permutations": 0})
    rows.append({"statistic": "date_groups", "observed": len(g), "null_median": float("nan"),
                 "null_ci_lower": float("nan"), "null_ci_upper": float("nan"),
                 "p_le_observed": float("nan"), "n_permutations": 0})
    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(table.to_string(index=False))
    print(f"\n{n_benign} benign / {n_malignant} malignant over {len(g)} date groups")
    print(f"wrote {OUT_CSV}")

    if not a.no_figure:
        style()
        a.output_dir.mkdir(parents=True, exist_ok=True)
        median_pairs = int(table.loc[table["statistic"] == "cross_label_date_group_pairs",
                                     "null_median"].iloc[0])
        print(f"wrote {figure(g, labels, median_pairs, a.seed, a.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
