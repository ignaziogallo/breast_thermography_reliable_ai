#!/usr/bin/env python3
"""Turn `data/DMR-IR/metadata.csv` into manifests with this repository's schema.

Emits a patient manifest carrying `patient_id` and `patient_binary_label`, which is all
`scripts/02_create_patient_splits.py` and `src/data/splits.py` require, plus an image
manifest with one row per frame. The DMR-IR cohort then flows through the existing
patient-level machinery unchanged.

Three cohort decisions are made here rather than silently downstream, each reversible by a
flag, each recorded in the report:

*   **Volunteers are excluded** (`--keep-volunteers` to keep them). DMR-IR mixes healthy
    volunteers with mastology-clinic patients: 3.1% vs 40.7% malignant, so volunteer status
    alone scores AUROC 0.585. That is a recruitment-channel shortcut, not pathology, and no
    choice of architecture can undo it. Dropping 32 volunteers leaves 140/96 -- a *better*
    balanced cohort than the 171/97 pool.
*   **Duplicate patients are collapsed** (`--keep-duplicate-patients` to keep them). Patients
    sharing byte-identical images are one person entered twice; left in place they would sit
    in train and test simultaneously under two ids, which is precisely the leakage this
    protocol exists to prevent. Patients are grouped by shared image hashes and only the
    representative (most images, then lowest id) is kept.
*   **Patients below `--min-images` are dropped** (default 1, i.e. keep everything with at
    least one image). The distribution is long-tailed -- 196 patients have the protocol's 27
    images, but some have 1 or 2 -- so this is an explicit knob, not an accident.

Image hashes are cached in `results/dmr_image_hashes.csv`; delete it to recompute.

Example
-------
    python scripts/19_build_dmr_manifest.py --root data/DMR-IR
    python scripts/02_create_patient_splits.py \
        --manifest results/dmr_patient_manifest.csv \
        --output   results/dmr_patient_splits.csv
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# DMR-IR encodes `view` as 0-4. The 45/90 grouping is *measured*: the dynamic protocol adds
# two 90-degree laterals per patient, so views 2 and 4 carry ~250 dynamic frames each while
# views 1 and 3 carry ~30. The left/right assignment below follows the enumeration in
# Figure 7 of Silva et al. (2014) -- Front, Right 45, Right 90, Left 45, Left 90 -- and is
# INFERRED, not verified. Before relying on laterality, run the mirror-correlation check
# described in data/README.md: corr(left, fliplr(right)) should exceed corr(left, right).
VIEW_GROUP = {0: "frontal", 1: "lateral45", 2: "lateral90", 3: "lateral45", 4: "lateral90"}
VIEW_SIDE = {0: "both", 1: "right", 2: "right", 3: "left", 4: "left"}

PATIENT_KEY = "patient_dir"
CARRY = ("role", "label_name", "age_at_visit", "age_current", "body_temperature",
         "visit_date", "registration_date", "menopause", "menarche", "mastectomy",
         "biopsy", "mammography", "hf_split", "record", "patient_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--prefix", default="dmr", help="output filename prefix (default: dmr)")
    p.add_argument("--id-prefix", default="DMR",
                   help="prepended to every patient id (default: DMR, mirroring this cohort's "
                        "IIR ids). Keeps ids non-numeric so pandas cannot reread '0090' as 90, "
                        "and makes cohort membership visible in any joined table.")
    p.add_argument("--keep-volunteers", action="store_true")
    p.add_argument("--keep-duplicate-patients", action="store_true")
    p.add_argument("--min-images", type=int, default=1)
    p.add_argument("--no-dedup", action="store_true",
                   help="skip hashing entirely; implies --keep-duplicate-patients")
    return p.parse_args()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk_bytes := f.read(chunk):
            h.update(chunk_bytes)
    return h.hexdigest()


def image_hashes(meta: pd.DataFrame, root: Path, cache: Path) -> pd.Series:
    """sha256 per image_path, cached because it reads ~9 GB."""
    if cache.exists():
        cached = pd.read_csv(cache).drop_duplicates("image_path").set_index("image_path")["sha256"]
        if set(meta["image_path"]) <= set(cached.index):
            print(f"Reusing hashes from {cache}")
            return meta["image_path"].map(cached)
        print(f"{cache} is stale; recomputing")

    print(f"Hashing {len(meta)} images (cached afterwards in {cache})...")
    values = []
    for i, rel in enumerate(meta["image_path"], start=1):
        path = root / rel
        values.append(sha256_file(path) if path.exists() else "")
        if i % 1000 == 0:
            print(f"  {i}/{len(meta)}")
    series = pd.Series(values, index=meta.index)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": meta["image_path"], "sha256": series}).to_csv(cache, index=False)
    return series


def duplicate_patient_groups(meta: pd.DataFrame) -> list[list[str]]:
    """Patients linked by at least one byte-identical image, via union-find."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_hash: dict[str, set[str]] = defaultdict(set)
    for digest, patient in zip(meta["sha256"], meta[PATIENT_KEY]):
        if digest:
            by_hash[digest].add(patient)
    for patients in by_hash.values():
        if len(patients) > 1:
            ordered = sorted(patients)
            for other in ordered[1:]:
                union(ordered[0], other)

    groups: dict[str, list[str]] = defaultdict(list)
    for patient in {p for ps in by_hash.values() if len(ps) > 1 for p in ps}:
        groups[find(patient)].append(patient)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main() -> int:
    args = parse_args()
    meta_path = args.root / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found; run scripts/17_download_dmr_ir.py first.", file=sys.stderr)
        return 1

    meta = pd.read_csv(meta_path, low_memory=False,
                       dtype={PATIENT_KEY: str, "patient_id": str})
    if "label_name" not in meta.columns:
        print(f"ERROR: no `label_name` column in {meta_path}.", file=sys.stderr)
        return 1

    report = ["# DMR-IR manifest", "", f"Source: `{meta_path}`", "",
              f"Start: **{len(meta)}** images, **{meta[PATIENT_KEY].nunique()}** patients", ""]
    dropped: dict[str, list[str]] = {}

    if not args.no_dedup:
        meta["sha256"] = image_hashes(meta, args.root, args.out_dir / f"{args.prefix}_image_hashes.csv")
    else:
        meta["sha256"] = ""

    # --- exclusion 1: volunteers -------------------------------------------------------
    if "role" in meta.columns and not args.keep_volunteers:
        is_volunteer = meta["role"].astype(str).str.lower().eq("volunteer")
        dropped["volunteer"] = sorted(meta.loc[is_volunteer, PATIENT_KEY].unique())
        meta = meta[~is_volunteer]

    # --- exclusion 2: duplicate patients -----------------------------------------------
    groups: list[list[str]] = []
    if not args.no_dedup and not args.keep_duplicate_patients:
        groups = duplicate_patient_groups(meta)
        counts = meta[PATIENT_KEY].value_counts()
        discard: list[str] = []
        for group in groups:
            # Representative: most images, ties broken by lowest id -- deterministic.
            keep = sorted(group, key=lambda p: (-counts.get(p, 0), p))[0]
            discard.extend(p for p in group if p != keep)
        dropped["duplicate_patient"] = sorted(discard)
        meta = meta[~meta[PATIENT_KEY].isin(discard)]

    # --- exclusion 3: too few images ---------------------------------------------------
    counts = meta[PATIENT_KEY].value_counts()
    thin = sorted(counts[counts < args.min_images].index)
    if thin:
        dropped[f"fewer_than_{args.min_images}_images"] = thin
        meta = meta[~meta[PATIENT_KEY].isin(thin)]

    if meta.empty:
        print("ERROR: every patient was excluded.", file=sys.stderr)
        return 1

    # --- image manifest ----------------------------------------------------------------
    view = pd.to_numeric(meta.get("view"), errors="coerce")
    images = pd.DataFrame({
        "patient_id": args.id_prefix + meta[PATIENT_KEY],
        "image_path": meta["image_path"],
        "mask_path": meta.get("mask_path", ""),
        "view_code": view,
        "view_group": view.map(VIEW_GROUP),
        "view_side_inferred": view.map(VIEW_SIDE),
        "protocol": meta.get("protocol"),
        "patient_binary_label": meta["label_name"].eq("malignant").astype(int),
        "sha256": meta["sha256"],
    }).reset_index(drop=True)

    # --- patient manifest --------------------------------------------------------------
    agg = {col: "first" for col in CARRY if col in meta.columns}
    patients = meta.groupby(PATIENT_KEY).agg(agg)
    # The canonical id is the zero-padded directory slug; keep the raw Hugging Face
    # `patient_id` only as provenance, under a name that cannot collide with it.
    patients = patients.rename(columns={"patient_id": "hf_patient_id"})
    patients.index.name = "patient_dir"
    patients = patients.reset_index()
    patients.insert(0, "patient_id", args.id_prefix + patients["patient_dir"])
    patients["patient_binary_label"] = patients["label_name"].eq("malignant").astype(int)
    patients["n_images"] = patients["patient_dir"].map(meta[PATIENT_KEY].value_counts())

    # Aliases under the names src/models/metadata_model.py expects, so this cohort is a
    # drop-in for scripts/09_metadata_baseline.py. `age_at_visit`, not `age_current`: the
    # latter is age at database export and folds in how long ago the visit happened.
    # DMR-IR records no weight or height, so there is no `bmi` and `logistic_metadata` /
    # `gradient_boosting_metadata` cannot run on this cohort -- only `logistic_age_only`.
    if "age_at_visit" in patients.columns:
        patients["age"] = pd.to_numeric(patients["age_at_visit"], errors="coerce")
    if "body_temperature" in patients.columns:
        patients["temperature"] = pd.to_numeric(patients["body_temperature"], errors="coerce")
    for group in ("frontal", "lateral45", "lateral90"):
        per = images[images["view_group"] == group].groupby("patient_id").size()
        patients[f"n_{group}"] = patients["patient_id"].map(per).fillna(0).astype(int)

    # One row per patient is the invariant every later stage assumes; a duplicated row
    # mispairs labels with predictions instead of failing loudly.
    dup = patients.loc[patients["patient_id"].duplicated(keep=False), "patient_id"].unique()
    if len(dup):
        print(f"ERROR: {len(dup)} duplicated patient rows (e.g. {sorted(dup)[:5]}).", file=sys.stderr)
        return 1
    # A patient whose images disagree on the label would silently pick `first` above.
    mixed = meta.groupby(PATIENT_KEY)["label_name"].nunique()
    mixed = sorted(mixed[mixed > 1].index)
    if mixed:
        print(f"ERROR: {len(mixed)} patients carry more than one label (e.g. {mixed[:5]}).", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pat_path = args.out_dir / f"{args.prefix}_patient_manifest.csv"
    img_path = args.out_dir / f"{args.prefix}_image_manifest.csv"
    patients.sort_values("patient_id").to_csv(pat_path, index=False)
    images.to_csv(img_path, index=False)

    # --- report ------------------------------------------------------------------------
    for reason, ids in dropped.items():
        report.append(f"- excluded `{reason}`: **{len(ids)}** patients"
                      + (f" -> {ids[:12]}{' ...' if len(ids) > 12 else ''}" if ids else ""))
    if groups:
        report += ["", "Duplicate-patient groups (byte-identical images shared):", ""]
        report += [f"  - {g}" for g in groups]
    report += ["", "## Final cohort", "",
               f"- patients: **{len(patients)}**", f"- images: **{len(images)}**",
               f"- labels: { {k: int(v) for k, v in patients['patient_binary_label'].value_counts().items()} }"
               + "  (0 = benign, 1 = malignant)"]
    if "role" in patients.columns:
        report.append(f"- roles: { {k: int(v) for k, v in patients['role'].value_counts().items()} }")
    report += ["", "```", images["view_group"].value_counts(dropna=False).to_string(), "```", "",
               "`view_side_inferred` follows Figure 7 of Silva et al. (2014) and is UNVERIFIED;",
               "confirm with the mirror-correlation check in data/README.md before using laterality.",
               "", "## Next", "",
               "```bash", "python scripts/02_create_patient_splits.py \\",
               f"    --manifest {pat_path} \\", f"    --output   {args.out_dir / (args.prefix + '_patient_splits.csv')}",
               "```"]

    rep_path = args.out_dir / f"{args.prefix}_manifest_report.md"
    rep_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nWrote {pat_path}, {img_path}, {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
