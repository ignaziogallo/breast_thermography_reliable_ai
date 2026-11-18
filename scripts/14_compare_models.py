#!/usr/bin/env python3
"""Paired patient-level comparison between two experiments.

The question an image model must answer is not "what AUROC did it reach" but "did it improve on
the reference baselines". Because both models were evaluated on identical patient folds,
every patient has a prediction from each, and the comparison can be made paired -- which is
far more powerful than comparing two overlapping confidence intervals, and is the only
correct way to read two intervals that overlap.

Reported for each pair:
  * the difference in AUROC, with a bootstrap 95% confidence interval on the *difference*
    resampled at patient level (the paired analogue of a DeLong test; DeLong itself is not
    implemented here, and a paired bootstrap makes fewer distributional assumptions);
  * McNemar's exact test on the paired hard decisions;
  * the absolute difference, always, alongside the p-value.

A confidence interval on the difference that contains zero means the experiment did not
demonstrate an improvement. It does not mean the models are equivalent: with 35 events this
study cannot resolve differences below roughly 0.10 AUROC, and that limit should be stated
whenever a null result is reported.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistical_tests import mcnemar_pvalue
from src.utils.io import read_results_csv


def pooled(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    """Repeat-averaged probability and majority decision per patient."""
    g = preds[preds["model"] == model]
    if g.empty:
        raise SystemExit(f"No predictions for model {model!r}; available: {sorted(preds['model'].unique())}")
    out = g.groupby("patient_id").agg(
        y_true=("y_true", "first"),
        y_prob=("y_prob", "mean"),
        y_pred=("y_pred", lambda s: int(s.mean() >= 0.5)),
    ).reset_index()
    return out


def paired_auroc_difference(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                            n_bootstrap: int, seed: int) -> dict:
    observed = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)          # one index set for both models: the pairing
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, pa[idx]) - roc_auc_score(ys, pb[idx]))
    diffs = np.asarray(diffs)
    return {
        "auroc_a": roc_auc_score(y, pa),
        "auroc_b": roc_auc_score(y, pb),
        "delta_auroc": observed,
        "delta_ci_lower": float(np.percentile(diffs, 2.5)) if diffs.size else np.nan,
        "delta_ci_upper": float(np.percentile(diffs, 97.5)) if diffs.size else np.nan,
        # Two-sided bootstrap p-value: how often the resampled difference crosses zero.
        # Capped at 1: for two identical models both tail fractions are 1 and twice the
        # minimum would exceed unity.
        "bootstrap_p": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))) if diffs.size else np.nan,
        "n_bootstrap_valid": int(diffs.size),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, nargs="+", required=True,
                   help="one or more results/<experiment>/predictions.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="restrict to these model names (default: all found)")
    p.add_argument("--reference", default=None,
                   help="compare every model against this one instead of all pairs")
    p.add_argument("--output", type=Path, default=Path("results/final_statistics.csv"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    frames = []
    for path in a.predictions:
        if not path.exists():
            raise SystemExit(f"{path} does not exist; run that experiment first.")
        frames.append(read_results_csv(path))
    preds = pd.concat(frames, ignore_index=True)

    models = a.models or sorted(preds["model"].unique())
    tables = {m: pooled(preds, m) for m in models}

    # Every model must have been scored on the same patients, or the pairing is invalid.
    reference_ids = set(tables[models[0]]["patient_id"])
    for m, t in tables.items():
        if set(t["patient_id"]) != reference_ids:
            raise SystemExit(
                f"Model {m!r} was scored on a different patient set than {models[0]!r} "
                f"({len(set(t['patient_id']))} vs {len(reference_ids)}). "
                "Paired comparison requires identical folds.")

    pairs = ([(m, a.reference) for m in models if m != a.reference] if a.reference
             else list(itertools.combinations(models, 2)))
    if a.reference and a.reference not in tables:
        raise SystemExit(f"Reference {a.reference!r} not among {models}")

    rows = []
    for ma, mb in pairs:
        ta = tables[ma].sort_values("patient_id").reset_index(drop=True)
        tb = tables[mb].sort_values("patient_id").reset_index(drop=True)
        assert (ta["patient_id"] == tb["patient_id"]).all()
        assert (ta["y_true"] == tb["y_true"]).all(), f"{ma} and {mb} disagree on the ground truth"
        y = ta["y_true"].to_numpy()
        stats = paired_auroc_difference(y, ta["y_prob"].to_numpy(), tb["y_prob"].to_numpy(),
                                        a.n_bootstrap, a.seed)
        rows.append({
            "model_a": ma, "model_b": mb, "n_patients": len(y), **stats,
            "mcnemar_p": mcnemar_pvalue(y, ta["y_pred"].to_numpy(), tb["y_pred"].to_numpy()),
        })

    out = pd.DataFrame(rows).sort_values("delta_auroc", ascending=False)
    # Pre-specified decision rule, fixed before the confirmatory run: model A improves on
    # model B only if the lower bound of the bootstrap interval on the difference exceeds
    # zero. Recorded as a column so the verdict is in the artifact, not only in the prose.
    out["improvement_demonstrated"] = out["delta_ci_lower"] > 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)

    show = out[["model_a", "model_b", "auroc_a", "auroc_b", "delta_auroc",
                "delta_ci_lower", "delta_ci_upper", "bootstrap_p", "mcnemar_p",
                "improvement_demonstrated"]].round(4)
    print(show.to_string(index=False))
    print(f"\nWrote {a.output}")

    if a.reference:
        print(f"\nPre-specified criterion: improvement over {a.reference!r} requires "
              "delta_ci_lower > 0.")
        for _, r in out.iterrows():
            verdict = "IMPROVEMENT DEMONSTRATED" if r["improvement_demonstrated"] else \
                      "no improvement demonstrated"
            print(f"  {r['model_a']:<42s} delta {r['delta_auroc']:+.3f} "
                  f"[{r['delta_ci_lower']:+.3f}, {r['delta_ci_upper']:+.3f}]  {verdict}")

    inconclusive = out[(out["delta_ci_lower"] < 0) & (out["delta_ci_upper"] > 0)]
    if len(inconclusive):
        print(f"\n{len(inconclusive)}/{len(out)} comparisons have a difference interval containing "
              "zero: no improvement demonstrated. With 35 events this study cannot resolve "
              "differences below roughly 0.10 AUROC, so these are inconclusive rather than null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
