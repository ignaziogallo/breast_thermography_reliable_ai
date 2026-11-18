#!/usr/bin/env python3
"""The matching diagnostics the methods section claims are computed.

`scripts/29_build_matched_cohorts.py` writes a `balance.csv` beside every rung: for each
covariate, the two class means and their standardised mean difference. The manuscript states
that matching was checked with SMDs but never tabulates them, which is exactly the kind of
claim a statistics reviewer asks to see.

The table this prints is the audit trail for the ladder. It should be read for two things,
and they are opposite in sign:

  * the confounder's own row must move towards zero on the rungs that control it -- that is
    what "controlled" means, and it is what separates the overlap rung (1.20, still large,
    which is why the rung is described as weakening rather than removing the confounder) from
    the day-calipers (|SMD| < 0.06);
  * the null-control rung must leave it alone. `propensity_clinical` balances the clinical
    covariates and leaves acquisition at 2.39 against the original cohort's 2.46, which is
    the arithmetic behind calling it a null control.

Size controls are excluded from the printed table: they are random draws with the confounding
left intact by construction, so their SMDs describe the draw, not a design choice. They are
kept in the CSV.

Reads only stored results. No GPU, no training.

Example
-------
    python scripts/38_matching_balance_report.py
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import ensure_parent

DATASETS = {
    "dmr": {"root": Path("results/matching/dmr"),
            "confounder": "acquisition_days",
            "order": ["original", "overlap", "year_balanced",
                      "acquisition_days_matched_7", "acquisition_days_matched_30",
                      "acquisition_days_matched_90", "acquisition_days_matched_180",
                      "propensity_clinical", "propensity_with_acquisition"],
            "caption": "Standardised mean differences (malignant minus benign, pooled SD) "
                       "for every rung of the DMR-IR ladder. The confounder is "
                       "\\texttt{acquisition\\_days}; a rung controls it when its SMD "
                       "approaches zero. The null control (\\texttt{propensity\\_clinical}) "
                       "balances the clinical covariates and leaves the acquisition axis "
                       "essentially where the original cohort has it."},
    "verification": {"root": Path("results/matching/verification"),
                     "confounder": "age",
                     "order": ["original", "overlap", "age_matched_2", "age_matched_5",
                               "age_matched_10", "propensity_clinical",
                               "propensity_with_age"],
                     "caption": "Standardised mean differences for the verification "
                                "cohort's ladder. The dominant channel is \\texttt{age}."},
}


def collect(root: Path) -> pd.DataFrame:
    """Every `balance.csv` under one ladder, stacked."""
    rows = [pd.read_csv(p) for p in sorted(glob.glob(str(root / "*" / "balance.csv")))]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def latex(table: pd.DataFrame, spec: dict, label: str) -> str:
    """The SMD table as LaTeX, the confounder's own column first."""
    covariates = [spec["confounder"]] + [c for c in table.columns if c != spec["confounder"]]
    header = " & ".join(c.replace("_", "\\_") for c in covariates)
    lines = ["\\begin{table}[h]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{l{'c' * len(covariates)}}}", "\\toprule",
             f"Cohort & {header} \\\\", "\\midrule"]
    for cohort, row in table.iterrows():
        cells = " & ".join("--" if pd.isna(row[c]) else f"{row[c]:+.2f}" for c in covariates)
        lines.append(f"\\texttt{{{cohort.replace('_', chr(92) + '_')}}} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{spec['caption']}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("results/matching/smd_all_rungs.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    everything = []
    for name, spec in DATASETS.items():
        raw = collect(spec["root"])
        if raw.empty:
            print(f"{name}: no balance.csv found, skipped")
            continue
        raw.insert(0, "dataset", name)
        everything.append(raw)

        wide = raw.pivot(index="cohort", columns="covariate", values="smd")
        keep = [c for c in spec["order"] if c in wide.index]
        wide = wide.loc[keep]
        print(f"\n% Generated by: python scripts/38_matching_balance_report.py  ({name})")
        print(latex(wide, spec, f"tab:smd_{name}"))
        worst = raw[~raw["cohort"].str.startswith("sizecontrol")]
        worst = worst.loc[worst["smd"].abs().groupby(worst["cohort"]).idxmax()]
        print(worst[["cohort", "covariate", "smd"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if everything:
        ensure_parent(a.output)
        pd.concat(everything, ignore_index=True).to_csv(a.output, index=False)
        print(f"\nWrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
