#!/usr/bin/env python3
"""Which patient-level experimental designs does DMR-IR actually support?

`scripts/18_audit_dmr_ir.py` audits the *download* -- roles, ages, duplicate pixels,
protocol counts. This script audits the *design space*: given the cohort we kept, how many
patients carry five static views, how many carry a complete dynamic sequence, how many
carry both, which clinical fields survive, and which of them are consequences of the
diagnosis rather than inputs to it.

It reads only the committed manifests in `results/`, never the pixels, so it runs on a
development workstation with no dataset mounted -- and answers the questions that decide
what can be trained before any GPU time is spent.

Three findings drive the design and are computed here rather than asserted:

*   **Acquisition epoch is a near-perfect proxy for the label.** `visit_date` scores AUROC
    0.98, and label-pure acquisition sessions cover 226 of 236 patients. Restricting to the
    2013-2015 overlap window does not repair this. Every reported metric must therefore be
    accompanied by the epoch control, not caveated in prose.
*   **`biopsy` is post-diagnostic.** It scores AUROC 0.85 on its own and its *missingness*
    scores 0.59. Clinical metadata must be partitioned into pre- and post-diagnostic fields
    before any multimodal ablation, or the ablation measures record-keeping.
*   **The "static" frame is often a member of the dynamic sequence.** For ~97 patients the
    static frontal is byte-identical to a dynamic frame, at a sequence index spread roughly
    uniformly over 0-19. A static-vs-dynamic comparison on those patients compares a set
    with one of its own elements.

Example
-------
    python scripts/21_audit_dmr_design.py
    python scripts/21_audit_dmr_design.py --min-dynamic-frames 19
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

# Clinical fields recorded by DMR-IR, split by whether they can precede the diagnosis.
# `biopsy`, `mastectomy` and `mammography` record what the clinic did *because* a lesion was
# suspected or found; using them as model inputs measures the care pathway, not the
# thermogram. They are the mechanism behind metadata-only accuracies in this literature.
PRE_DIAGNOSTIC = ("age_at_visit", "menarche", "menopause", "body_temperature")
POST_DIAGNOSTIC = ("biopsy", "mastectomy", "mammography")

VIEW_CODES = (0, 1, 2, 3, 4)
FRONTAL = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--output", type=Path, default=Path("results/dmr_design_audit.md"))
    p.add_argument("--min-dynamic-frames", type=int, default=20,
                   help="frames required for a 'complete' frontal dynamic sequence")
    return p.parse_args()


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    # patient_id must stay a string: 'DMR0090' is safe, but a bare '0090' would be reread as
    # 90 and every manifest/split join would silently drop rows.
    pat = pd.read_csv(args.patients, dtype={"patient_id": str})
    img = pd.read_csv(args.images, dtype={"patient_id": str})
    img["view_code"] = pd.to_numeric(img["view_code"], errors="coerce").astype("Int64")
    stem = img["image_path"].str.split("/").str[-1].str.rsplit(".", n=1).str[0]
    img["record"] = stem.str.split("_").str[0]
    img["seq"] = pd.to_numeric(stem.str.split("_").str[-1], errors="coerce").astype("Int64")
    return pat, img


def safe_auroc(y: pd.Series, x: pd.Series) -> float:
    """AUROC over rows where both are present; NaN when a class or a value is missing."""
    x = pd.to_numeric(x, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10 or y[m].nunique() < 2 or x[m].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[m], x[m]))


def md_table(df: pd.DataFrame, index_name: str = "") -> list[str]:
    body = df.reset_index() if df.index.name or index_name else df
    if index_name:
        body = body.rename(columns={body.columns[0]: index_name})
    head = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "|" + "|".join("---" for _ in body.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in body.itertuples(index=False)]
    return [head, rule, *rows, ""]


def section_cohort(pat: pd.DataFrame) -> list[str]:
    y = pat["patient_binary_label"]
    return [
        "## Cohort",
        "",
        f"- patients: **{len(pat)}** ({int((y == 0).sum())} benign / {int((y == 1).sum())} malignant)",
        f"- roles: {dict(pat['role'].value_counts())}",
        "",
        "Labels are **patient-level only**. DMR-IR records no per-breast diagnosis, so a",
        "breast-level task and any left/right supervised asymmetry model are out of scope;",
        "laterality can enter only as an unsupervised feature of a patient-level model.",
        "",
    ]


def section_epoch(pat: pd.DataFrame) -> list[str]:
    y = pat["patient_binary_label"]
    visit = pd.to_datetime(pat["visit_date"], errors="coerce")
    days = visit.astype("int64") / 86_400e9

    out = ["## Acquisition epoch", "",
           f"`visit_date` alone scores **AUROC {safe_auroc(y, days):.3f}**.", ""]

    by_year = pd.crosstab(visit.dt.year.rename("year"), y.rename("label"))
    by_year.columns = [f"label={c}" for c in by_year.columns]
    out += md_table(by_year, "year")

    windows = [("full cohort", None, None), ("2013-2015", "2013-01-01", "2015-12-31"),
               ("2013H2-2015", "2013-06-01", "2015-12-31")]
    rows = []
    for name, lo, hi in windows:
        m = pd.Series(True, index=pat.index)
        if lo:
            m &= visit >= lo
        if hi:
            m &= visit <= hi
        rows.append({"window": name, "n": int(m.sum()),
                     "benign": int((y[m] == 0).sum()), "malignant": int((y[m] == 1).sum()),
                     "date AUROC": round(safe_auroc(y[m], days[m]), 3)})
    out += ["Restricting to the overlap window shrinks the cohort without clearing the date:", ""]
    out += md_table(pd.DataFrame(rows))

    # A session is one acquisition date. If sessions are label-pure, no split scheme can
    # decorrelate batch from label -- there is nothing left to hold constant.
    g = pat.groupby("visit_date")["patient_binary_label"].agg(["size", "mean"])
    pure = (g["mean"] == 0) | (g["mean"] == 1)
    out += [
        f"- distinct acquisition dates: **{len(g)}** (median {g['size'].median():.0f} patients each)",
        f"- label-pure sessions: **{int(pure.sum())}/{len(g)}**, covering "
        f"**{int(g.loc[pure, 'size'].sum())}/{len(pat)}** patients",
        f"- mixed sessions: {int((~pure).sum())}, covering {int(g.loc[~pure, 'size'].sum())} patients",
        "",
        "Session membership is therefore an almost perfect proxy for the label. No split",
        "scheme repairs this; the confound has to be measured and reported, not removed.",
        "",
    ]
    return out


def section_metadata(pat: pd.DataFrame) -> list[str]:
    y = pat["patient_binary_label"]
    rows = []
    for col in PRE_DIAGNOSTIC + POST_DIAGNOSTIC:
        if col not in pat.columns:
            continue
        s = pd.to_numeric(pat[col].replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}),
                          errors="coerce")
        rows.append({
            "field": col,
            "kind": "pre" if col in PRE_DIAGNOSTIC else "POST-DIAGNOSTIC",
            "non-null": f"{int(s.notna().sum())}/{len(pat)}",
            "AUROC": round(safe_auroc(y, s), 3),
            "AUROC of missingness": round(safe_auroc(y, s.isna().astype(int)), 3),
        })
    out = ["## Clinical metadata", "", *md_table(pd.DataFrame(rows))]

    if "biopsy" in pat.columns:
        ct = pd.crosstab(pat["biopsy"].fillna("missing").rename("biopsy"),
                         y.rename("label"), dropna=False)
        ct.columns = [f"label={c}" for c in ct.columns]
        out += ["`biopsy` in detail -- a record of what the clinic already suspected:", ""]
        out += md_table(ct, "biopsy")

    issues = []
    age = pd.to_numeric(pat.get("age_at_visit"), errors="coerce")
    if age is not None and (age <= 0).any():
        issues.append(f"- `age_at_visit` <= 0 for {int((age <= 0).sum())} patient(s): "
                      f"{sorted(age[age <= 0].tolist())}")
    temp = pd.to_numeric(pat.get("body_temperature"), errors="coerce")
    if temp is not None and (temp <= 0).any():
        issues.append(f"- `body_temperature` == 0 for {int((temp <= 0).sum())} patients "
                      "(missing recorded as zero, not NaN)")
    if issues:
        out += ["Data-quality defects to repair before any tabular model:", "", *issues, ""]
    return out


def section_views(pat: pd.DataFrame, img: pd.DataFrame, min_frames: int) -> list[str]:
    lab = pat.set_index("patient_id")["patient_binary_label"]

    def counts(ids) -> str:
        v = lab.reindex(sorted(ids)).value_counts()
        return f"{int(v.get(0, 0))} benign / {int(v.get(1, 0))} malignant"

    out = ["## Views and protocols", ""]
    ct = pd.crosstab(img["view_code"].rename("view_code"), img["protocol"])
    out += md_table(ct, "view_code")

    static = img[img["protocol"] == "Static"]
    n_views = static.groupby("patient_id")["view_code"].nunique().reindex(
        pat["patient_id"], fill_value=0)
    dist = n_views.value_counts().sort_index().rename("patients").to_frame()
    out += ["Static views held per patient:", ""]
    out += md_table(dist, "n static views")

    full5 = set(n_views[n_views == len(VIEW_CODES)].index)

    dyn = img[(img["protocol"] == "Dinamic") & (img["view_code"] == FRONTAL)]
    per = dyn.groupby("patient_id").size()
    complete = set(per[per >= min_frames].index)
    gaps = [pid for pid, g in dyn.groupby("patient_id")
            if sorted(g["seq"].dropna().tolist()) != list(range(len(g)))]

    out += [
        "",
        f"- patients with **all 5 static views**: **{len(full5)}** ({counts(full5)})",
        f"- patients with a frontal dynamic sequence of >= {min_frames} frames: "
        f"**{len(complete)}** ({counts(complete)})",
        f"- sequences whose `seq` is not contiguous 0..n-1: **{len(gaps)}**",
        f"- patients with **both**: **{len(full5 & complete)}** ({counts(full5 & complete)})",
        "",
        "The last line is the complete-case cohort for a model using every modality.",
        "",
    ]

    # For patients whose static frame is a member of the dynamic sequence, "static vs
    # dynamic" is not a comparison between two acquisitions.
    rows = []
    for v in VIEW_CODES:
        s = img[(img["protocol"] == "Static") & (img["view_code"] == v)][["patient_id", "sha256"]]
        d = img[(img["protocol"] == "Dinamic") & (img["view_code"] == v)][["patient_id", "sha256", "seq"]]
        shared = s.merge(d, on=["patient_id", "sha256"])
        rows.append({"view_code": v, "patients with static image": s["patient_id"].nunique(),
                     "static frame is a dynamic frame": shared["patient_id"].nunique()})
    out += ["Overlap between the static and dynamic protocols (byte-identical files):", ""]
    out += md_table(pd.DataFrame(rows))

    front = img[(img["protocol"] == "Static") & (img["view_code"] == FRONTAL)][["patient_id", "sha256"]]
    fdyn = img[(img["protocol"] == "Dinamic") & (img["view_code"] == FRONTAL)][["patient_id", "sha256", "seq"]]
    shared = front.merge(fdyn, on=["patient_id", "sha256"])
    if len(shared):
        idx = shared["seq"].dropna()
        out += [
            f"For the {shared['patient_id'].nunique()} frontal cases the matched sequence index "
            f"spans {int(idx.min())}-{int(idx.max())} (median {idx.median():.0f}), i.e. the "
            "static image is an arbitrary member of the sequence rather than its first frame.",
            "",
        ]
    return out


def section_duplicates(img: pd.DataFrame) -> list[str]:
    g = img.groupby("sha256")["patient_id"].agg(["size", "nunique"])
    dup = g[g["size"] > 1]
    cross = dup[dup["nunique"] > 1]
    masks = img["mask_path"].notna() & img["mask_path"].astype(str).str.len().gt(0)
    return [
        "## Duplicates and masks",
        "",
        f"- duplicate-hash groups: **{len(dup)}** covering {int(dup['size'].sum())} images",
        f"- **cross-patient** duplicate groups: **{len(cross)}** "
        f"({'patient-level splitting is sufficient' if not len(cross) else 'MUST be resolved'})",
        f"- images carrying a mask: **{int(masks.sum())}/{len(img)}**",
        "",
        "Mask coverage of 100% confirms the masks are TransUNet *predictions* generated over",
        "the whole mirror, not annotations. Because that segmenter was fitted on this cohort,",
        "a mask may carry label information; run the shape-only baseline before using masks.",
        "",
    ]


def main() -> int:
    args = parse_args()
    pat, img = load(args)

    lines = ["# DMR-IR design audit", "",
             f"Sources: `{args.patients}`, `{args.images}`.",
             "Reads manifests only -- no pixels -- so it runs without the dataset mounted.", ""]
    lines += section_cohort(pat)
    lines += section_epoch(pat)
    lines += section_metadata(pat)
    lines += section_views(pat, img, args.min_dynamic_frames)
    lines += section_duplicates(img)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
