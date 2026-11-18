#!/usr/bin/env python3
"""A multiplicity guard for the conditional caliper grid.

The conditional table reports one c-statistic per imaging arm per caliper and marks the
cells whose bootstrap interval excludes 0.5. That is five arms times five calipers of
marks, and nothing in the table tells the reader how many of them a grid of that size
would produce by chance. The surviving-signal claim is carried by exactly those marks, so
the grid needs a correction and the paper needs to say which one it used.

The test per cell: permute the patient scores, leaving labels and acquisition dates --- and
therefore the admissible pair set --- untouched. Under the null that the score orders
date-comparable pairs no better than chance, the conditional c-statistic is centred on 0.5.
Two-sided, on |c - 0.5|.

Holm rather than Bonferroni: the cells are strongly dependent (nested calipers over shared
pairs, correlated arms), Holm is valid under arbitrary dependence for the family-wise error
rate, and it is uniformly more powerful than Bonferroni. A permutation-based step-down that
exploited the dependence would be sharper still, but it would also require a joint null
across arms that the stored per-arm scores do not support.

The family is declared here, not chosen after seeing the results: all imaging arms in the
encoder conditional series, at every caliper, including the unrestricted one. The
confounder-alone control is excluded --- it is a control that must fail, not a hypothesis
under test --- and reported separately.

Reads only stored scores. No GPU, no retraining.

Example
-------
    python scripts/35_caliper_multiplicity.py
    python scripts/35_caliper_multiplicity.py --n-permutation 50000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confounding.conditional import conditional_c_statistic, conditional_permutation_p
from src.evaluation.statistical_tests import holm
from src.utils.io import ensure_parent

CONTROL = "score__confounder_alone_control"
CALIPERS = [None, 180.0, 90.0, 30.0, 7.0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path,
                   default=Path("results/confounders/dmr_encoders_conditional/patient_scores.csv"))
    p.add_argument("--confounder", default="visit_date")
    p.add_argument("--output", type=Path,
                   default=Path("results/confounders/caliper_multiplicity.csv"))
    p.add_argument("--n-permutation", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.scores)
    y = df["patient_binary_label"].to_numpy().astype(int)
    conf = df[args.confounder].to_numpy(dtype=float)
    arms = [c for c in df.columns if c.startswith("score__") and c != CONTROL]

    rows = []
    for arm in arms:
        s = df[arm].to_numpy(dtype=float)
        for caliper in CALIPERS:
            stat = conditional_c_statistic(y, s, conf, caliper)
            if not np.isfinite(stat["c_statistic"]):
                continue
            p = conditional_permutation_p(y, s, conf, caliper, stat["c_statistic"],
                                          args.n_permutation,
                                          np.random.default_rng(args.seed))
            rows.append({"arm": arm.removeprefix("score__"),
                         "caliper": "none" if caliper is None else f"{caliper:.0f}d",
                         "n_pairs": stat["n_pairs"], "c_statistic": stat["c_statistic"],
                         "p_raw": p})
            print(f"  {rows[-1]['arm'][:46]:46s} {rows[-1]['caliper']:>5s} "
                  f"c={stat['c_statistic']:.3f} p={p:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    out["survives_holm_05"] = out["p_holm"] < 0.05

    ensure_parent(args.output)
    out.to_csv(args.output, index=False)

    print(f"\nFamily: {len(out)} tests ({out['arm'].nunique()} imaging arms x "
          f"{out['caliper'].nunique()} calipers), Holm-adjusted.\n")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSurviving Holm at 0.05: {int(out['survives_holm_05'].sum())} of {len(out)}")

    # The control is not part of the family: it is required to fail, and its failure is
    # what licenses reading the imaging rows at all.
    print("\nConfounder-alone control (excluded from the family):")
    s = df[CONTROL].to_numpy(dtype=float)
    for caliper in CALIPERS:
        stat = conditional_c_statistic(y, s, conf, caliper)
        if not np.isfinite(stat["c_statistic"]):
            continue
        p = conditional_permutation_p(y, s, conf, caliper, stat["c_statistic"],
                                      args.n_permutation, np.random.default_rng(args.seed))
        lbl = "none" if caliper is None else f"{caliper:.0f}d"
        print(f"  {lbl:>5s}  c={stat['c_statistic']:.3f}  p_raw={p:.4f}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
