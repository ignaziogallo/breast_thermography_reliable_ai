#!/usr/bin/env python3
"""Was it the confounder that was removed, or just the patients?

Every rung of `scripts/29_build_matched_cohorts.py` does two things at once: it takes the
confounder away and it makes the cohort smaller. A fall in imaging discrimination on a
matched rung is therefore ambiguous on its own, and the ambiguity is not small -- an
imaging model fitted on 46 training patients may fail for reasons that have nothing to do
with acquisition.

The size controls resolve it. For each rung, three random subsets of the *same class sizes*
were drawn from the full cohort with the confounding left intact, and an identical model was
fitted on each. Sample size is then held fixed and only confounder control varies:

    matched rung        n patients, confounder removed        -> AUROC_matched
    size controls       n patients, confounder intact         -> AUROC_control x3

If AUROC_control sits well above AUROC_matched, the fall belongs to the confounder. If the
two coincide, the rung was measuring its own sample size and cannot support any claim about
confounding.

The comparison is between *independent* cohorts, not paired: a matched rung and its control
contain different patients, so no patient-level pairing exists and no paired interval can be
computed. What is reported is each arm's own interval and the spread of the three control
draws, and the verdict is deliberately coarse -- whether the control draws lie above the
matched rung's interval -- because that is all an unpaired comparison on cohorts this small
can support.

Reads only stored results. Writes `rung_vs_control_<dataset>.csv` next to each ladder and
prints the LaTeX table. The file name carries the dataset because three of the datasets below
(the frontal arm and the two all-frames arms) share one `ladder.csv`: they are the same
cohorts scored by different arms. Writing them all to a single `rung_vs_control.csv`, as an
earlier version did, left only the last-run arm on disk and silently changed what
`scripts/31_figure_confounder_control.py` would draw.

Example
-------
    python scripts/32_matched_ladder_report.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.quality_gate import QUARANTINE_SUFFIX

DATASETS = {
    "dmr": {"ladder": Path("results/matching/dmr/ladder.csv"),
            "prefix": "experiment_dmr_10_matched_",
            "confounder": "acquisition date"},
    # The same cohorts, the same controls, fitted on the all-frames configuration
    # instead of the frontal one. The matched design and the conditional design were
    # otherwise measuring different arms, which is the like-for-like objection the
    # frontal-only ladder cannot answer. Same ladder.csv -- only the arm changes.
    "dmr_allframes_swin": {"ladder": Path("results/matching/dmr/ladder.csv"),
                           "prefix": "experiment_dmr_14_matched_allframes_swin_",
                           "confounder": "acquisition date, all frames (Swin-T)"},
    "dmr_allframes_convnext": {"ladder": Path("results/matching/dmr/ladder.csv"),
                               "prefix": "experiment_dmr_14_matched_allframes_convnext_",
                               "confounder": "acquisition date, all frames (ConvNeXt-T)"},
    "verification": {"ladder": Path("results/matching/verification/ladder.csv"),
                     "prefix": "experiment_ver_10_matched_",
                     "confounder": "age"},
}
CONTROL_PREFIX = "sizecontrol"


def scored_patients(root: Path) -> int:
    """How many distinct patients this run actually produced a score for.

    Not the same as the rung's cohort size, and the gap is the paper's 236-versus-231. A
    matched cohort is defined on the patient manifest, but a *frontal* arm can only score
    the members of it that hold a static frontal view; five patients of DMR-IR do not. The
    figure and the supplement table report this number, so that no rung claims to have been
    fitted on patients no image reached.
    """
    path = root / "predictions.csv"
    if not path.exists():
        return 0
    return int(pd.read_csv(path, usecols=["patient_id"])["patient_id"].nunique())


def arm_auroc(prefix: str, cohort: str) -> tuple[float, float, float, bool, int] | None:
    """The stored AUROC interval of one rung's experiment, if it was run.

    A run quarantined by the quality gate lives under the `QUARANTINE_SUFFIX` name and is
    returned with that flag set, because it must be visible in the table rather than
    silently absent or silently counted.
    """
    for suffix, failed in (("", False), (QUARANTINE_SUFFIX, True)):
        root = Path("results") / f"{prefix}{cohort}{suffix}"

        path = root / "bootstrap_ci.csv"
        if path.exists():
            ci = pd.read_csv(path)
            ci = ci[ci["metric"] == "roc_auc"]
            if ci.empty:
                continue
            r = ci.iloc[0]
            return (float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"]), failed,
                    scored_patients(root))

        # The all-frames trainer (scripts/03_leakage_experiment.py) reports one row per
        # aggregation unit in metrics.csv rather than writing a separate bootstrap file.
        # Only the per-patient row is admissible here: the per-frame row scores frames as
        # if they were independent, which is the very inflation the ladder is controlling
        # for elsewhere in this paper.
        path = root / "metrics.csv"
        if path.exists():
            m = pd.read_csv(path)
            m = m[m["model"].astype(str).str.endswith("__per_patient")]
            if m.empty or "estimate" not in m.columns:
                continue
            r = m.iloc[0]
            return (float(r["estimate"]), float(r["ci_lower"]), float(r["ci_upper"]), failed,
                    scored_patients(root))
    return None


def build(dataset: str, spec: dict) -> pd.DataFrame | None:
    """One row per matched rung: its own interval beside the spread of its size controls."""
    if not spec["ladder"].exists():
        return None
    ladder = pd.read_csv(spec["ladder"])
    if "control_for" not in ladder.columns:
        ladder["control_for"] = np.nan

    controls = ladder[ladder["cohort"].str.startswith(CONTROL_PREFIX)]
    rows = []
    for r in ladder[~ladder["cohort"].str.startswith(CONTROL_PREFIX)].itertuples():
        got = arm_auroc(spec["prefix"], r.cohort)
        if got is None:
            continue
        auroc, lo, hi, failed, n_scored = got

        mine = controls[controls["control_for"].astype(str).str.split(";").apply(
            lambda names, cohort=r.cohort: cohort in names)]
        values = []
        for c in mine.itertuples():
            g = arm_auroc(spec["prefix"], c.cohort)
            if g is not None:
                values.append(g[0])

        row = {
            "dataset": dataset, "cohort": r.cohort, "n_patients": int(r.n_patients),
            "n_scored": n_scored,
            "n_benign": int(r.n_benign), "n_malignant": int(r.n_malignant),
            "confounder_auroc": r.confounder_auroc,
            "image_auroc": round(auroc, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "quality_gate_failed": failed,
            "n_size_controls": len(values),
            "control_auroc_min": round(min(values), 4) if values else np.nan,
            "control_auroc_median": round(float(np.median(values)), 4) if values else np.nan,
            "control_auroc_max": round(max(values), 4) if values else np.nan,
            "control_aurocs": ";".join(f"{v:.4f}" for v in values),
        }
        # The coarse verdict an unpaired comparison supports: did every control draw of the
        # same size, with the confounding left in, land above this rung's interval?
        row["all_controls_above_rung_ci"] = bool(values) and min(values) > hi
        rows.append(row)
    return pd.DataFrame(rows)


def latex(table: pd.DataFrame, confounder: str, label: str) -> str:
    """The rung-versus-control table as LaTeX."""
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Confounder-controlled cohorts and their size-matched controls "
             f"({confounder}). Each control column is three random subsets of the same class "
             f"sizes drawn from the full cohort with the confounding left intact, so that "
             f"sample size is held fixed and only confounder control varies. The comparison "
             f"is unpaired: the two cohorts contain different patients.}}",
             f"\\label{{{label}}}",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             "Cohort & $n$ & scored & Confounder & Image AUROC [95\\% CI] & Size controls "
             "& Above CI \\\\",
             "\\midrule"]
    for r in table.itertuples():
        controls = ("--" if not r.n_size_controls
                    else f"{r.control_auroc_min:.3f}--{r.control_auroc_max:.3f}")
        verdict = ("yes" if r.all_controls_above_rung_ci
                   else "--" if not r.n_size_controls else "no")
        name = r.cohort.replace("_", "\\_") + ("$^{\\dagger}$" if r.quality_gate_failed else "")
        lines.append(f"{name} & {r.n_patients} & {r.n_scored} & {r.confounder_auroc:.3f} & "
                     f"{r.image_auroc:.3f} [{r.ci_lower:.3f}, {r.ci_upper:.3f}] & "
                     f"{controls} & {verdict} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS) + ["all"], default="all")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    names = sorted(DATASETS) if a.dataset == "all" else [a.dataset]
    any_built = False
    for name in names:
        spec = DATASETS[name]
        table = build(name, spec)
        if table is None or table.empty:
            print(f"{name}: no ladder or no experiments run yet, skipped")
            continue
        any_built = True
        out = spec["ladder"].parent / f"rung_vs_control_{name}.csv"
        table.to_csv(out, index=False)
        print(f"\n% Generated by: python scripts/32_matched_ladder_report.py --dataset {name}")
        print(latex(table, spec["confounder"], f"tab:ladder_{name}"))
        print(table[["cohort", "n_patients", "n_scored", "confounder_auroc", "image_auroc",
                     "control_auroc_median", "all_controls_above_rung_ci"]]
              .to_string(index=False))
        print(f"Wrote {out}")
    if not any_built:
        raise SystemExit("Nothing to report: run scripts/29 and the matched experiments first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
