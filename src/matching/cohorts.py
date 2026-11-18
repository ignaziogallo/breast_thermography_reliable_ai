"""Confounder-controlled cohort construction, from metadata only.

The ladder this module builds runs from the cohort as distributed to the strictest control
the data can carry, and each rung is a *patient subset* that the existing training scripts
can consume unchanged through a restricted split file:

    original          every eligible patient.
    overlap           patients acquired in the period where both classes occur.
    year_balanced     within each acquisition year, the classes down-sampled to a common
                      size, so the year distribution of the two classes is identical.
    date_matched      1:1 nearest-neighbour matching on acquisition date within a caliper.
    propensity        1:1 matching on a propensity score fitted to pre-diagnostic covariates.

Three rules govern every rung, and they are the reason this lives in its own module rather
than inside a script:

*   **No prediction may enter.** A cohort is defined from metadata and the label alone.
    Selecting patients with any knowledge of how a model scored them would make the
    resulting AUROC a statement about the selection, and the leakage tests could not see it.
    The public functions here accept no score argument at all, which is the enforcement.
*   **Only pre-diagnostic covariates may be matched on.** Matching on `biopsy` or
    `mastectomy` would balance the classes on records of what the clinic did *because* it
    had already suspected the diagnosis, removing signal that is not confounding.
*   **The cost is reported, never hidden.** Each rung records how many patients it retained,
    and `standardized_mean_differences` records how well it balanced what it aimed at. A
    rung that retains 28 patients is a rung on which nothing can be demonstrated, and the
    table has to say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL = "patient_binary_label"


def acquisition_days(dates: pd.Series) -> pd.Series:
    """Days elapsed since the cohort's earliest acquisition.

    Affine and monotone, so it changes no AUROC and no matching distance; it exists so that
    a caliper can be stated in days and a coefficient read without a datetime in it.
    """
    parsed = pd.to_datetime(dates, errors="coerce")
    return (parsed - parsed.min()).dt.days.astype(float)


def standardized_mean_differences(cohort: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Per-covariate SMD between the classes, the standard balance diagnostic.

    SMD = (mean_malignant - mean_benign) / sqrt((var_malignant + var_benign) / 2), computed
    on non-missing values. |SMD| < 0.1 is the conventional threshold for adequate balance.
    It is reported per rung rather than tested, because a balance test on 28 patients has no
    power and its non-significance would be misread as balance achieved.
    """
    rows = []
    pos = cohort[cohort[LABEL] == 1]
    neg = cohort[cohort[LABEL] == 0]
    for col in covariates:
        if col not in cohort.columns:
            continue
        a = pd.to_numeric(pos[col], errors="coerce").dropna()
        b = pd.to_numeric(neg[col], errors="coerce").dropna()
        pooled = (np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                  if len(a) > 1 and len(b) > 1 else np.nan)
        smd = ((a.mean() - b.mean()) / pooled
               if pooled and np.isfinite(pooled) and pooled > 0 else np.nan)
        rows.append({
            "covariate": col,
            "n_malignant": int(len(a)), "n_benign": int(len(b)),
            "mean_malignant": float(a.mean()) if len(a) else np.nan,
            "mean_benign": float(b.mean()) if len(b) else np.nan,
            "smd": float(smd) if np.isfinite(smd) else np.nan,
            "balanced_at_0.1": bool(np.isfinite(smd) and abs(smd) < 0.1),
        })
    return pd.DataFrame(rows)


def overlap_cohort(patients: pd.DataFrame, confounder: str) -> pd.DataFrame:
    """Patients whose confounder value lies in the range both classes occupy.

    On DMR-IR this is the 2013-2015 window: it removes the 2012 benign block and the
    2018-2020 malignant block, and leaves a cohort on which the date still scores 0.914,
    which is precisely why the stricter rungs below are needed.
    """
    v = pd.to_numeric(patients[confounder], errors="coerce")
    pos, neg = v[patients[LABEL] == 1], v[patients[LABEL] == 0]
    lo = max(pos.min(), neg.min())
    hi = min(pos.max(), neg.max())
    return patients[(v >= lo) & (v <= hi)].reset_index(drop=True)


def stratum_balanced_cohort(patients: pd.DataFrame, stratum: str,
                            seed: int = 42) -> pd.DataFrame:
    """Within each stratum, down-sample the larger class to the size of the smaller.

    The result has, by construction, an identical stratum distribution in the two classes,
    so the stratum carries no marginal information about the label. Strata holding only one
    class contribute nobody: they are exactly the sessions that make the confounder
    predictive, and keeping them would defeat the rung.

    Down-sampling is a random draw at a fixed seed, not a choice among patients, and it
    happens before any model sees the cohort.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in patients.groupby(stratum, dropna=True):
        pos = g[g[LABEL] == 1]
        neg = g[g[LABEL] == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            continue
        for part in (pos, neg):
            members = part.index.to_numpy()
            idx = rng.choice(members, n, replace=False) if len(part) > n else members
            keep.extend(idx.tolist())
    return patients.loc[sorted(keep)].reset_index(drop=True)


def nearest_neighbour_pairs(patients: pd.DataFrame, distance_column: str,
                            caliper: float) -> pd.DataFrame:
    """Greedy 1:1 matching of each malignant patient to one benign patient within a caliper.

    Malignant patients are processed in order of how isolated they are -- the one whose
    nearest admissible control is furthest away goes first -- so that a patient with a
    single admissible control is not left unmatched because that control was consumed by
    somebody with several. This is the standard greedy remedy; optimal matching would gain
    little on a cohort where the binding constraint is that the classes barely coexist.

    Returns one row per matched pair. On DMR-IR this rung retains roughly a dozen pairs,
    which the caller is expected to report rather than round away.
    """
    v = pd.to_numeric(patients[distance_column], errors="coerce")
    pos = patients[(patients[LABEL] == 1) & v.notna()]
    neg = patients[(patients[LABEL] == 0) & v.notna()]
    if pos.empty or neg.empty:
        return pd.DataFrame(columns=["pair_id", "malignant_id", "benign_id", "distance"])

    pv = v.loc[pos.index].to_numpy()
    nv = v.loc[neg.index].to_numpy()
    dist = np.abs(pv[:, None] - nv[None, :])

    nearest = np.where(dist.min(axis=1) <= caliper, dist.min(axis=1), np.inf)
    order = np.argsort(-nearest)                      # most isolated first; unmatchable last
    taken: set[int] = set()
    rows = []
    for i in order:
        if not np.isfinite(nearest[i]):
            continue
        candidates = [j for j in np.argsort(dist[i]) if dist[i, j] <= caliper and j not in taken]
        if not candidates:
            continue
        j = candidates[0]
        taken.add(int(j))
        rows.append({"pair_id": len(rows) + 1,
                     "malignant_id": pos.iloc[i]["patient_id"],
                     "benign_id": neg.iloc[j]["patient_id"],
                     "distance": float(dist[i, j])})
    if not rows:
        return pd.DataFrame(columns=["pair_id", "malignant_id", "benign_id", "distance"])
    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def size_control_cohort(patients: pd.DataFrame, n_benign: int, n_malignant: int,
                        seed: int) -> pd.DataFrame:
    """A random subset of the given class sizes, with the confounding left intact.

    Every rung above removes the confounder *and* shrinks the cohort at the same time, so a
    fall in imaging discrimination on a matched rung has two candidate explanations: the
    shortcut was taken away, or there was no longer enough data to fit anything. Those are
    not separable from the matched rung alone.

    This draws the same number of patients of each class from the whole cohort, at random
    and without regard to the confounder. It therefore holds the sample size fixed and
    varies only whether the confounder was controlled -- which is exactly the contrast the
    matched rung cannot make on its own. Several seeds should be run and read together:
    with 26 patients a single draw is itself noisy, and the point of the control is the
    spread of the draws, not any one of them.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label, n in ((0, n_benign), (1, n_malignant)):
        members = patients.index[patients[LABEL] == label].to_numpy()
        if n > len(members):
            raise ValueError(f"cannot draw {n} patients of class {label} from {len(members)}")
        keep.extend(rng.choice(members, n, replace=False).tolist())
    return patients.loc[sorted(keep)].reset_index(drop=True)


def propensity_scores(patients: pd.DataFrame, covariates: list[str], seed: int = 42) -> pd.Series:
    """P(malignant | pre-diagnostic covariates), fitted on the whole cohort.

    Fitted on every patient because a propensity score is a nuisance model over covariates,
    not a predictor evaluated out of fold: it enters only as a matching distance, and the
    imaging arm that is eventually evaluated on the matched cohort is fitted and scored
    under the ordinary patient-level protocol. Using the label here is what makes the score
    a propensity; it does not put label information into any reported estimate, because the
    matched cohort is a set of patients rather than a score.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    present = [c for c in covariates if c in patients.columns]
    if not present:
        raise ValueError(f"none of the covariates {covariates} are in the patient table")
    X = patients[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = patients[LABEL].astype(int).to_numpy()
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
    pipe.fit(X, y)
    return pd.Series(pipe.predict_proba(X)[:, 1], index=patients.index, name="propensity")


def pairs_to_cohort(patients: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """The matched patients as a cohort table carrying their pair identifier.

    The pair id travels with the patient so that a downstream analysis can respect the
    matched structure -- a matched-pair sign test, or a split that keeps a pair together --
    rather than discarding the design once the subset has been taken.
    """
    if pairs.empty:
        return patients.iloc[0:0].assign(pair_id=pd.Series(dtype=int))
    mapping = {}
    for r in pairs.itertuples(index=False):
        mapping[r.malignant_id] = r.pair_id
        mapping[r.benign_id] = r.pair_id
    out = patients[patients["patient_id"].isin(mapping)].copy()
    out["pair_id"] = out["patient_id"].map(mapping).astype(int)
    return out.sort_values(["pair_id", LABEL]).reset_index(drop=True)


def describe_cohort(cohort: pd.DataFrame, name: str, confounder: str | None = None) -> dict:
    """The counts a matched analysis must report before any metric is read."""
    label = cohort[LABEL].astype(int) if len(cohort) else pd.Series(dtype=int)
    out = {
        "cohort": name,
        "n_patients": int(len(cohort)),
        "n_benign": int((label == 0).sum()),
        "n_malignant": int((label == 1).sum()),
    }
    if confounder and confounder in cohort.columns and len(cohort):
        v = pd.to_numeric(cohort[confounder], errors="coerce")
        for cls, key in ((0, "benign"), (1, "malignant")):
            s = v[label == cls]
            out[f"{confounder}_{key}_min"] = float(s.min()) if len(s) else np.nan
            out[f"{confounder}_{key}_max"] = float(s.max()) if len(s) else np.nan
            out[f"{confounder}_{key}_mean"] = float(s.mean()) if len(s) else np.nan
    return out
