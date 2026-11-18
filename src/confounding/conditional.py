"""Discrimination restricted to confounder-comparable patient pairs.

The AUROC of a score is the fraction of benign/malignant patient pairs it ranks correctly.
On DMR-IR that statistic is uninterpretable, because the two classes were acquired in
different years: a pair drawn at random is usually a 2012 benign against a 2019 malignant,
and any feature that tracks the calendar orders it correctly without reading pathology.

This module computes the same statistic over the *subset of pairs the confounder cannot
separate*: pairs whose acquisition dates (or ages) differ by at most a caliper W.

    c_W(score) = P(score_malignant > score_benign | |C_malignant - C_benign| <= W)

Two properties make this the right primary estimator for this cohort rather than 1:1
matching, and both are measured rather than assumed (see scripts/29_build_matched_cohorts.py):

*   **Efficiency.** Greedy 1:1 date matching on DMR-IR retains 10 pairs at a 30-day caliper
    and 14 at 180 days. The conditional statistic retains 101 and 410 respectively, because
    a benign patient may serve in every admissible pair rather than being consumed by one
    match. With 96 malignant patients that difference decides whether anything is estimable.
*   **Interpretation.** Within an admissible pair the confounder is, by construction, nearly
    constant, so a score that still orders the pair is not ordering it by the calendar. The
    claim is conditional and local: it says nothing about pairs the caliper excluded.

The control that makes the whole design legible is to run the *confounder's own* score
through the identical statistic. `c_W(date)` must fall towards 0.5 as W narrows -- that is
arithmetic, not a finding -- and reporting it alongside `c_W(image)` is what distinguishes
"the image survives confounder control" from "the caliper was too wide to control anything".

Everything here consumes stored per-patient predictions and a metadata column. No pixels,
no model, no fold loop: an experiment that has already been run can be re-read under
confounder control without being repeated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE = 0.5


def pair_matrix(conf_pos: np.ndarray, conf_neg: np.ndarray, caliper: float | None,
                stratum_pos: np.ndarray | None = None,
                stratum_neg: np.ndarray | None = None) -> np.ndarray:
    """Boolean (n_pos, n_neg) mask of pairs the confounder cannot separate.

    Two constraints, and they control different things. The `caliper` bounds how far apart
    two patients may be on a *continuous* confounder. The `stratum` requires them to share a
    *categorical* one exactly -- the same acquisition session, the same device, the same site.

    Both are needed because a caliper on a date does not control the session. On DMR-IR two
    patients photographed five days apart usually belong to two different sessions, and 83 of
    86 sessions contain a single diagnostic class, so a model reading a session signature can
    order almost every date-comparable pair correctly while the date's own value orders none
    of them. Only the exact-stratum constraint closes that route.
    """
    mask = (np.ones((len(conf_pos), len(conf_neg)), dtype=bool) if caliper is None
            else np.abs(conf_pos[:, None] - conf_neg[None, :]) <= caliper)
    if stratum_pos is not None and stratum_neg is not None:
        mask &= stratum_pos[:, None] == stratum_neg[None, :]
    return mask


def conditional_c_statistic(y_true, score, confounder, caliper: float | None,
                            stratum=None) -> dict:
    """Fraction of admissible discordant pairs ranked correctly.

    `caliper=None` with no stratum admits every pair, which recovers the ordinary AUROC
    exactly; that equivalence is what makes the caliper series readable as one curve whose
    left end is the familiar number.

    `stratum` additionally requires the two patients of a pair to share a categorical value,
    which is how an acquisition session is held constant rather than merely approximated by
    a date caliper.

    Ties count as half a correctly ordered pair, the same convention the Mann-Whitney form
    of AUROC uses. A degenerate arm -- no positives, no negatives, or no admissible pair --
    returns NaN with the counts that explain why, rather than raising: a caliper series is
    expected to run out of pairs at its narrow end, and that is a result.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    c = np.asarray(confounder, dtype=float)
    k = None if stratum is None else np.asarray(stratum).astype(str)
    ok = np.isfinite(s) & np.isfinite(c)
    y, s, c = y[ok], s[ok], c[ok]
    if k is not None:
        k = k[ok]

    pos, neg = y == 1, y == 0
    admissible = pair_matrix(c[pos], c[neg], caliper,
                             None if k is None else k[pos], None if k is None else k[neg])
    n_pairs = int(admissible.sum())
    out = {
        "n_pairs": n_pairs,
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "n_positive_in_pairs": int(admissible.any(axis=1).sum()) if n_pairs else 0,
        "n_negative_in_pairs": int(admissible.any(axis=0).sum()) if n_pairs else 0,
        "c_statistic": np.nan,
    }
    if n_pairs == 0:
        return out
    diff = s[pos][:, None] - s[neg][None, :]
    won = (diff > 0).astype(float) + TIE * (diff == 0)
    out["c_statistic"] = float(won[admissible].sum() / n_pairs)
    return out


def _resample_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A stratified patient bootstrap draw.

    Resampling the cohort as a whole occasionally yields a draw with no admissible pair at a
    narrow caliper, which silently shortens the interval it is meant to describe.
    Resampling within each class keeps the class sizes fixed, so every draw supports the
    same pairing structure and the interval describes uncertainty about the ranking rather
    than about the class balance, which was fixed by the cohort anyway.
    """
    idx = []
    for label in (0, 1):
        members = np.flatnonzero(y == label)
        if len(members):
            idx.append(rng.choice(members, len(members), replace=True))
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def bootstrap_conditional(y_true, score, confounder, caliper: float | None,
                          n_bootstrap: int = 2000, seed: int = 42, stratum=None) -> dict:
    """Point estimate and patient-level percentile interval for `conditional_c_statistic`.

    Patients are resampled, and the admissible pair set is recomputed inside each draw: a
    patient drawn twice contributes twice to the pairing, which is the correct propagation
    of the fact that pairs are not independent observations.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    c = np.asarray(confounder, dtype=float)
    k = None if stratum is None else np.asarray(stratum).astype(str)
    point = conditional_c_statistic(y, s, c, caliper, k)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(y, rng)
        r = conditional_c_statistic(y[idx], s[idx], c[idx], caliper,
                                    None if k is None else k[idx])
        if np.isfinite(r["c_statistic"]):
            draws.append(r["c_statistic"])
    arr = np.asarray(draws)
    lower = float(np.percentile(arr, 2.5)) if arr.size else np.nan
    return {
        **point,
        "ci_lower": lower,
        "ci_upper": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
        # The readout that carries content at a narrow caliper. Once the confounder has been
        # controlled its own c_W sits at chance and may fall below it, so "better than the
        # confounder" becomes an easy and uninformative comparison; what has to be
        # demonstrated there is discrimination against chance itself.
        "exceeds_chance": bool(np.isfinite(lower) and lower > 0.5),
        "n_bootstrap_valid": int(arr.size),
    }


def paired_conditional_difference(y_true, score_a, score_b, confounder,
                                  caliper: float | None, n_bootstrap: int = 2000,
                                  seed: int = 42, stratum=None) -> dict:
    """Difference between two scores on the *same* admissible pairs, with its interval.

    One index set per draw serves both scores; that pairing is what makes the interval on
    the difference far tighter than the overlap of two marginal intervals, and it is the
    only correct way to compare two arms evaluated on one cohort.

    The decision rule is the repository's pre-specified one: A improves on B only when
    `delta_ci_lower > 0`. An interval containing zero means no improvement was demonstrated,
    not that the two scores are equivalent.
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    c = np.asarray(confounder, dtype=float)
    k = None if stratum is None else np.asarray(stratum).astype(str)

    ca = conditional_c_statistic(y, a, c, caliper, k)
    cb = conditional_c_statistic(y, b, c, caliper, k)
    observed = ca["c_statistic"] - cb["c_statistic"]

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(y, rng)
        ks = None if k is None else k[idx]
        ra = conditional_c_statistic(y[idx], a[idx], c[idx], caliper, ks)
        rb = conditional_c_statistic(y[idx], b[idx], c[idx], caliper, ks)
        if np.isfinite(ra["c_statistic"]) and np.isfinite(rb["c_statistic"]):
            diffs.append(ra["c_statistic"] - rb["c_statistic"])
    arr = np.asarray(diffs)
    lower = float(np.percentile(arr, 2.5)) if arr.size else np.nan
    upper = float(np.percentile(arr, 97.5)) if arr.size else np.nan
    return {
        "n_pairs": ca["n_pairs"],
        "c_statistic_a": ca["c_statistic"],
        "c_statistic_b": cb["c_statistic"],
        "delta": observed,
        "delta_ci_lower": lower,
        "delta_ci_upper": upper,
        # Capped at 1: for two identical scores both tail fractions are 1.
        "bootstrap_p": (float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean())))
                        if arr.size else np.nan),
        "n_bootstrap_valid": int(arr.size),
        "improvement_demonstrated": bool(np.isfinite(lower) and lower > 0),
    }


def conditional_permutation_p(y: np.ndarray, score: np.ndarray, confounder: np.ndarray,
                              caliper: float | None, observed: float, n_perm: int,
                              rng: np.random.Generator, chunk: int = 200) -> float:
    """Two-sided permutation p-value for one conditional c-statistic.

    The patient scores are permuted while labels and confounder values -- and therefore the
    admissible pair set -- stay untouched. Under the null that the score orders
    confounder-comparable pairs no better than chance, the statistic is centred on 0.5, and
    the test is on |c - 0.5|. `chunk` bounds the permutations held in memory at once: the
    unrestricted grid on DMR-IR is 12878 pairs.
    """
    pos, neg = y == 1, y == 0
    admissible = pair_matrix(confounder[pos], confounder[neg], caliper, None, None)
    pi, nj = np.nonzero(admissible)
    n_pairs = len(pi)
    if n_pairs == 0:
        return np.nan

    obs = abs(observed - 0.5)
    hits, done = 0, 0
    while done < n_perm:
        batch = min(chunk, n_perm - done)
        perm = rng.permuted(np.tile(score, (batch, 1)), axis=1)
        sp, sn = perm[:, pos], perm[:, neg]
        diff = sp[:, pi] - sn[:, nj]
        c = ((diff > 0).sum(axis=1) + TIE * (diff == 0).sum(axis=1)) / n_pairs
        hits += int((np.abs(c - 0.5) >= obs - 1e-12).sum())
        done += batch
    return (hits + 1) / (n_perm + 1)


def caliper_series(patients: pd.DataFrame, score_columns: dict[str, str], confounder: str,
                   calipers: list[float | None], n_bootstrap: int = 2000,
                   seed: int = 42, stratum_column: str | None = None) -> pd.DataFrame:
    """One row per (score, caliper): the confounder-control ladder as a table.

    `patients` holds one row per patient with `patient_binary_label`, the confounder, and one
    column per score. `score_columns` maps a reported arm name to its column, so the
    confounder's own score can be included under a name that identifies it as the control.

    `stratum_column` adds one further row per score, in which the two patients of a pair must
    share that column exactly. It is the strictest rung of the ladder and usually the emptiest:
    on DMR-IR only eight benign/malignant pairs share an acquisition date at all, so the row
    exists to report that the cohort cannot answer the question rather than to answer it.
    """
    required = {"patient_binary_label", confounder} | set(score_columns.values())
    if stratum_column:
        required.add(stratum_column)
    missing = sorted(required - set(patients.columns))
    if missing:
        raise ValueError(f"caliper_series: columns absent from the patient table: {missing}")
    if "patient_id" in patients.columns and patients["patient_id"].duplicated().any():
        raise ValueError("caliper_series expects one row per patient; aggregate repeats first")

    y = patients["patient_binary_label"].astype(int).to_numpy()
    c = patients[confounder].astype(float).to_numpy()
    k = patients[stratum_column].to_numpy() if stratum_column else None
    rows = []
    for name, column in score_columns.items():
        s = patients[column].astype(float).to_numpy()
        for caliper in calipers:
            r = bootstrap_conditional(y, s, c, caliper, n_bootstrap, seed)
            rows.append({"score": name, "score_column": column, "confounder": confounder,
                         "caliper": "none" if caliper is None else caliper, **r})
        if stratum_column:
            r = bootstrap_conditional(y, s, c, None, n_bootstrap, seed, stratum=k)
            rows.append({"score": name, "score_column": column, "confounder": confounder,
                         "caliper": f"same_{stratum_column}", **r})
    return pd.DataFrame(rows)
