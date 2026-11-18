"""Hypothesis tests on paired decisions, and the multiplicity correction applied to grids.

McNemar is straightforward for paired hard decisions. Paired differences in AUROC are
handled by resampling instead (`src.evaluation.resampling`, `scripts/14_compare_models.py`):
a paired patient-level bootstrap makes fewer distributional assumptions than DeLong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_pvalue(y_true, pred_a, pred_b) -> float:
    """Exact McNemar test on the discordant decisions of two models over the same patients."""
    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = a == y_true, b == y_true
    table = [[int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    return float(mcnemar(table, exact=True).pvalue)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Step-down Holm adjustment, monotonicity enforced, clipped at 1.

    Valid for the family-wise error rate under arbitrary dependence between the tests, and
    uniformly more powerful than Bonferroni.
    """
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def within_patient_test(breast_preds: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of the two breasts of the same patient.

    Restricted to patients whose two breasts carry different labels. Because both images
    come from one person in one session, age, BMI, body temperature, camera and posture are
    held exactly constant, and the only difference is which breast is diseased. Under the
    null that the model reads nothing about the breast itself, the diseased side is ranked
    higher in half the patients.
    """
    rows = []
    for contrast, hi, lo in (("PM vs contralateral N", "PM", "N"),
                             ("PB vs contralateral N", "PB", "N")):
        ordered, ties, total = 0, 0, 0
        for _, g in breast_preds.groupby("patient_id"):
            classes = set(g["breast_class"])
            if classes != {hi, lo}:
                continue
            p_hi = float(g.loc[g["breast_class"] == hi, "y_prob"].iloc[0])
            p_lo = float(g.loc[g["breast_class"] == lo, "y_prob"].iloc[0])
            total += 1
            if p_hi > p_lo:
                ordered += 1
            elif p_hi == p_lo:
                ties += 1
        if total == 0:
            continue
        # Sign test: ties are conservatively counted against the hypothesis.
        p = stats.binomtest(ordered, total, 0.5, alternative="greater").pvalue
        ci = stats.binomtest(ordered, total, 0.5).proportion_ci(confidence_level=0.95)
        rows.append({"contrast": contrast, "n_patients": total,
                     "correctly_ordered": ordered, "ties": ties,
                     "proportion": ordered / total,
                     "ci_lower": ci.low, "ci_upper": ci.high, "p_one_sided": p})
    return pd.DataFrame(rows)
