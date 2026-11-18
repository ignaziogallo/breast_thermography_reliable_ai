#!/usr/bin/env python3
"""Audit the downloaded DMR-IR cohort before it is used for anything.

`scripts/17_download_dmr_ir.py` reports what arrived; this script reports whether it can be
trusted. Four questions, in decreasing order of how badly a wrong answer would hurt:

1.  **Is `role` a shortcut?** DMR-IR mixes healthy volunteers with mastology-clinic
    patients -- two populations recruited separately. If volunteer/patient status predicts
    the label, a network can learn the recruitment channel instead of the pathology. The
    AUROC of `role` alone quantifies exactly how much label is available for free.
2.  **Is the age confound the same as in our own cohort?** There, `logistic_age_only`
    reaches AUROC 0.758 and no image model beats it. If DMR-IR carries the same confound,
    the age-only baseline is again the bar to clear -- not chance.
3.  **Are there duplicate images?** Perez-Martin & Sanchez-Cauce (2023) found 365 anomalies
    in DMR-IR, including patients sharing complete image sets (0090/0091, 0153/0154) and
    111 patients whose frontal static and dynamic images are identical. Cross-patient
    duplicates break the patient-level split that the whole protocol rests on.
4.  **Does the acquisition protocol hold?** The expected per-patient composition is 1 static
    frontal + 20 dynamic frontal + 2 lateral 90 + 2 lateral 45 = 27 images.

Writes `<root>/audit_report.md`. Nothing here modifies the data.

Example
-------
    python scripts/18_audit_dmr_ir.py --root data/DMR-IR
    python scripts/18_audit_dmr_ir.py --root data/DMR-IR --no-hashes   # skip step 3
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

PATIENT_KEY = "patient_dir"
LABEL_COL = "label_name"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--no-hashes", action="store_true", help="skip duplicate detection (reads ~9 GB)")
    p.add_argument("--output", type=Path, default=None, help="default: <root>/audit_report.md")
    return p.parse_args()


def patient_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient; the experimental unit is the patient, not the frame."""
    agg = {LABEL_COL: "first", "n_images": "size"}
    for col in ("role", "age_at_visit", "age_current", "body_temperature", "hf_split", "record"):
        if col in meta.columns:
            agg[col] = "first"
    out = meta.assign(n_images=1).groupby(PATIENT_KEY).agg(agg)
    out["y"] = (out[LABEL_COL] == "malignant").astype(int)
    return out


def section_role(pat: pd.DataFrame) -> list[str]:
    if "role" not in pat.columns:
        return ["", "## 1. `role` shortcut", "", "No `role` column -- skipped."]
    tab = pd.crosstab(pat["role"], pat[LABEL_COL])
    lines = ["", "## 1. `role` as a shortcut (patient level)", "", "```", tab.to_string(), "```", ""]

    rate = pat.groupby("role")["y"].mean()
    for role, r in rate.items():
        lines.append(f"- `{role}`: {r:.1%} malignant ({int(pat['role'].eq(role).sum())} patients)")

    # One binary feature, scored the same way any model would be.
    is_patient = (pat["role"].astype(str).str.lower() == "patient").astype(int)
    if pat["y"].nunique() > 1 and is_patient.nunique() > 1:
        auroc = roc_auc_score(pat["y"], is_patient)
        lines += ["", f"**AUROC of `role` alone: {auroc:.3f}** "
                  f"(our own cohort's age-only baseline reaches 0.758).", ""]
        if auroc >= 0.65:
            lines.append("`role` carries a large share of the label. Any image model trained across")
            lines.append("both groups may be reading the recruitment channel -- different sessions,")
            lines.append("room conditions and operators -- rather than pathology. Either restrict to")
            lines.append("`role == Patient`, or report `role`-stratified results alongside pooled ones.")

    odd = pat[(pat["role"].astype(str).str.lower() == "volunteer") & (pat["y"] == 1)]
    if len(odd):
        lines += ["", f"Volunteers labelled malignant: **{len(odd)}** -> {list(odd.index)}",
                  "Healthy volunteers carrying a malignant label are either genuine incidental",
                  "findings or label errors; inspect before including them."]
    return lines


def section_age(pat: pd.DataFrame) -> list[str]:
    lines = ["", "## 2. Age confound", ""]
    found = False
    for col in ("age_at_visit", "age_current"):
        if col not in pat.columns:
            continue
        age = pd.to_numeric(pat[col], errors="coerce")
        ok = age.notna() & pat["y"].notna()
        if ok.sum() < 10 or pat.loc[ok, "y"].nunique() < 2:
            continue
        found = True
        auroc = roc_auc_score(pat.loc[ok, "y"], age[ok])
        lines += [f"- `{col}`: n={int(ok.sum())}, missing={int((~age.notna()).sum())}, "
                  f"benign mean {age[ok & (pat['y'] == 0)].mean():.1f}, "
                  f"malignant mean {age[ok & (pat['y'] == 1)].mean():.1f}, "
                  f"**AUROC {auroc:.3f}**"]
    if not found:
        lines.append("- no usable age column.")
    else:
        lines += ["", "Use `age_at_visit`, not `age_current`: the latter is age at database export,",
                  "so it folds in how long ago the visit happened.",
                  "", "This is the bar an image model has to clear on this cohort."]
    return lines


def section_duplicates(meta: pd.DataFrame, root: Path) -> list[str]:
    lines = ["", "## 3. Duplicate images (exact SHA-256)", ""]
    digests: dict[str, list[str]] = defaultdict(list)
    owners: dict[str, list[str]] = defaultdict(list)

    for _, row in meta.iterrows():
        path = root / row["image_path"]
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[digest].append(row["image_path"])
        owners[digest].append(row[PATIENT_KEY])

    groups = {d: paths for d, paths in digests.items() if len(paths) > 1}
    cross = {d: paths for d, paths in groups.items() if len(set(owners[d])) > 1}

    lines += [f"- images hashed: **{sum(len(v) for v in digests.values())}**",
              f"- distinct image contents: **{len(digests)}**",
              f"- duplicate groups: **{len(groups)}**",
              f"- **cross-patient** duplicate groups: **{len(cross)}**", ""]
    if cross:
        lines += ["Cross-patient duplicates break the patient-level split: the same picture would",
                  "sit in train and test under two different patient ids. Resolve before use.", ""]
        for d, paths in list(cross.items())[:15]:
            lines.append(f"  - patients {sorted(set(owners[d]))}: {paths[:4]}")
        if len(cross) > 15:
            lines.append(f"  - ... and {len(cross) - 15} more groups")
    else:
        lines.append("No cross-patient duplicates -- the patient-level split is safe on this axis.")

    within = len(groups) - len(cross)
    lines += ["", f"Within-patient duplicate groups: **{within}**. Expected: the literature reports",
              "111 patients whose frontal static and dynamic images are identical. These do not",
              "break the split, but they inflate any image-level count."]
    return lines


def section_protocol(meta: pd.DataFrame, pat: pd.DataFrame) -> list[str]:
    lines = ["", "## 4. Acquisition protocol", ""]
    if {"view", "protocol"} <= set(meta.columns):
        lines += ["```", pd.crosstab(meta["view"], meta["protocol"]).to_string(), "```", "",
                  "Expected per patient: 1 static frontal + 20 dynamic frontal + 2 lateral 90",
                  "+ 2 lateral 45 = 27 images.", ""]
    counts = pat["n_images"].value_counts().sort_index()
    lines += ["Images per patient:", "", "```", counts.to_string(), "```", ""]
    thin = pat[pat["n_images"] < 5]
    if len(thin):
        lines.append(f"Patients with fewer than 5 images: **{len(thin)}** -> {list(thin.index)[:20]}")
        lines.append("Decide explicitly whether these enter the cohort; do not let them in by default.")
    return lines


def main() -> int:
    args = parse_args()
    meta_path = args.root / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found; run scripts/17_download_dmr_ir.py first.", file=sys.stderr)
        return 1

    # patient_dir must stay a string: pandas would read "0090" as 90 and lose the zero
    # padding that matches the directory names on disk.
    meta = pd.read_csv(meta_path, low_memory=False, dtype={PATIENT_KEY: str, "patient_id": str})
    if LABEL_COL not in meta.columns:
        print(f"ERROR: no `{LABEL_COL}` column in {meta_path}.", file=sys.stderr)
        return 1

    pat = patient_table(meta)
    lines = ["# DMR-IR audit", "",
             f"Source: `{meta_path}`", "",
             f"- images: **{len(meta)}**", f"- patients: **{len(pat)}**",
             f"- patient-level labels: "
             f"{ {k: int(v) for k, v in pat[LABEL_COL].value_counts().items()} }"]

    lines += section_role(pat)
    lines += section_age(pat)
    if not args.no_hashes:
        lines += section_duplicates(meta, args.root)
    else:
        lines += ["", "## 3. Duplicate images", "", "Skipped (`--no-hashes`)."]
    lines += section_protocol(meta, pat)

    out = args.output or (args.root / "audit_report.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
