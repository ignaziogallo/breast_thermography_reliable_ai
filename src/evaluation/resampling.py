"""Resampling inference for a rank statistic: bootstrap intervals and permutation tests.

Every routine here works on the Mann-Whitney form of the AUROC, computed from ranks. That
is what makes the loops affordable -- a permutation of the labels leaves the ranks
untouched, so ten thousand relabellings are one vectorised gather -- and it is why the
shortcut-usage columns of both cohorts and the era probes can share one implementation and
therefore mean exactly the same thing.

`labels` is always a binary vector marking the "positive" group of the contrast at hand: the
later acquisition era, the older age band, the malignant class. Nothing here knows which.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc_from_ranks(ranks: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney form of the AUROC. Identical to roc_auc_score, ties included, but it
    takes the ranks as given -- which is what makes the resampling loops affordable."""
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(score: np.ndarray, labels: np.ndarray, n_boot: int,
                 rng: np.random.Generator) -> tuple[float, float, int]:
    """Percentile interval over patients resampled with replacement.

    A resample changes the score vector, so its ranks have to be recomputed each time;
    `rankdata` on a few hundred values is cheap enough for that to be the honest choice
    rather than an approximation that ignores the ties duplicates introduce.

    Returns `(lower, upper, n_valid)`; a resample holding one group only is skipped.
    """
    n = len(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        positive = labels[idx].astype(bool)
        if positive.all() or not positive.any():
            continue
        vals.append(auc_from_ranks(rankdata(score[idx]), positive))
    if not vals:
        return np.nan, np.nan, 0
    a = np.asarray(vals)
    return float(np.quantile(a, .025)), float(np.quantile(a, .975)), len(a)


def permutation_p(score: np.ndarray, labels: np.ndarray, observed: float, n_perm: int,
                  rng: np.random.Generator) -> float:
    """Two-sided: how often does a random relabelling separate at least as well?

    Permuting the labels leaves the scores untouched, so the ranks are computed once and
    every permutation is a sum over a random subset of them -- the whole test becomes one
    vectorised gather. The statistic is |AUROC - 0.5| and the p-value is the conservative
    `(hits + 1) / (n_perm + 1)`.
    """
    ranks = rankdata(score)
    n, n1 = len(score), int(labels.sum())
    n0 = n - n1
    picks = rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)[:, :n1]
    aucs = (ranks[picks].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    hits = int((np.abs(aucs - 0.5) >= abs(observed - 0.5) - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def paired_auc_difference(score_a: np.ndarray, score_b: np.ndarray, positive: np.ndarray,
                          n_boot: int, rng: np.random.Generator) -> dict:
    """AUROC of `score_a` minus that of `score_b` on the same patients, with its interval.

    One index set per draw serves both scores. That pairing is what gives the difference a
    proper interval, which reading two independent rows of a table cannot.

    `bootstrap_p` is twice the smaller tail fraction and is not capped: for two identical
    scores both tails are 1, so a caller that reports it should clip it at 1.
    """
    auc_a = auc_from_ranks(rankdata(score_a), positive)
    auc_b = auc_from_ranks(rankdata(score_b), positive)
    n, diffs = len(positive), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        p = positive[i]
        if p.all() or not p.any():
            continue
        diffs.append(auc_from_ranks(rankdata(score_a[i]), p)
                     - auc_from_ranks(rankdata(score_b[i]), p))
    d = np.asarray(diffs)
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": auc_a - auc_b,
            "delta_lo": float(np.quantile(d, .025)), "delta_hi": float(np.quantile(d, .975)),
            "bootstrap_p": 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))}
