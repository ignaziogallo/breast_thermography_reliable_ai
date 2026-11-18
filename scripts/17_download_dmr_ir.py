#!/usr/bin/env python3
"""Download the SemilleroCV/DMR-IR mirror into a patient-organised directory tree.

The Hugging Face mirror ships raw-temperature TIFFs, per-row clinical metadata and,
crucially for us, a ``patient_id`` column -- so unlike the flattened Kaggle mirrors it can
be split at patient level.

Two properties of this script matter scientifically:

1.  **Image bytes are copied verbatim.** The value of this mirror is that the pixels are
    temperatures, not an auto-ranged 8-bit render. Decoding and re-encoding through PIL
    can silently change dtype or scaling, so the original file bytes are written straight
    to disk and never round-tripped.
2.  **The published train/validation/test split is recorded as provenance only.** It is
    almost certainly an image-level split; DMR-IR has ~5 static views plus 20 sequential
    dynamic frames per patient, so an image-level split puts near-identical frames of the
    same patient on both sides of the boundary. The download report measures exactly that
    overlap. Build splits with ``scripts/02_create_patient_splits.py`` instead.

Requires ``pip install datasets huggingface_hub`` (not in requirements.txt: this is an
optional external cohort). Authenticate once with ``hf auth login`` (``huggingface-cli`` is
deprecated), or export ``HF_TOKEN``. Do not paste a token into this file.

Example
-------
    python scripts/17_download_dmr_ir.py --out data/DMR-IR --limit 20   # smoke test
    python scripts/17_download_dmr_ir.py --out data/DMR-IR             # full ~9 GB
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ID = "SemilleroCV/DMR-IR"
# Decoded size reported by the datasets-server (~8.8 GB) plus the parquet cache.
ESTIMATED_BYTES = 15 * 1024**3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/DMR-IR"),
                   help="destination root; must be git-ignored (default: data/DMR-IR)")
    p.add_argument("--hf-cache", type=Path, default=None,
                   help="Hugging Face cache dir; point it at a disk with ~6 GB free")
    p.add_argument("--token", default=None,
                   help="HF token; prefer `huggingface-cli login` or the HF_TOKEN env var")
    p.add_argument("--limit", type=int, default=None,
                   help="write only the first N rows of each split (smoke test)")
    p.add_argument("--no-masks", action="store_true",
                   help="skip the segmentation_mask column")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite files that already exist (default: skip, so runs resume)")
    p.add_argument("--drop-cache", action="store_true",
                   help="delete the HF parquet cache once extraction succeeds")
    return p.parse_args()


def slug(value, default: str = "na") -> str:
    """Filesystem-safe token; integer-looking patient ids are zero-padded to DMR-IR style."""
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        return f"{int(text):04d}"
    except ValueError:
        pass
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return cleaned[:60] or default


def image_bytes(cell: dict) -> bytes:
    """Bytes of an undecoded datasets.Image cell, reading from the cache if inlined bytes are absent."""
    if cell.get("bytes"):
        return cell["bytes"]
    path = cell.get("path")
    if not path:
        raise ValueError("image cell carries neither bytes nor a path")
    return Path(path).read_bytes()


def suffix_of(cell: dict, fallback: str) -> str:
    return Path(cell.get("path") or "").suffix.lower() or fallback


def main() -> int:
    args = parse_args()

    try:
        import datasets
        from datasets import ClassLabel, Value, load_dataset
        from datasets import Image as HFImage
    except ImportError:
        print("ERROR: `pip install datasets huggingface_hub` first (optional dependency).", file=sys.stderr)
        return 1

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(out_root).free
    if args.limit is None and free < ESTIMATED_BYTES:
        print(f"ERROR: {free / 1024**3:.1f} GB free at {out_root}; the full download needs "
              f"~{ESTIMATED_BYTES / 1024**3:.0f} GB (images + parquet cache).", file=sys.stderr)
        print("Free space, pass --hf-cache to a roomier disk, or try --limit first.", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Loading {REPO_ID} (this downloads the parquet shards on first run)...")
    dsdict = load_dataset(REPO_ID, cache_dir=str(args.hf_cache) if args.hf_cache else None, token=token)

    first = dsdict[next(iter(dsdict))]
    features = first.features
    (out_root / "dataset_info.json").write_text(json.dumps(
        {"repo_id": REPO_ID,
         "splits": {name: len(split) for name, split in dsdict.items()},
         "features": {name: str(feat) for name, feat in features.items()}},
        indent=2), encoding="utf-8")

    image_cols = [c for c, f in features.items() if isinstance(f, HFImage)]
    if not image_cols:
        print(f"ERROR: no datasets.Image column found; columns are {list(features)}", file=sys.stderr)
        return 1
    primary_image = "image" if "image" in image_cols else image_cols[0]

    mask_col = None
    mask_is_image = False
    if not args.no_masks and "segmentation_mask" in features:
        mask_col = "segmentation_mask"
        mask_is_image = isinstance(features[mask_col], HFImage)

    scalar_cols = [c for c, f in features.items()
                   if isinstance(f, (Value, ClassLabel)) and c not in image_cols]
    # text_embedding and any other array column: megabytes per row, no use here.
    array_cols = [c for c in features if c not in scalar_cols and c not in image_cols and c != mask_col]

    if "patient_id" not in scalar_cols:
        print(f"ERROR: no `patient_id` column -- this mirror cannot be split at patient level.\n"
              f"       Columns present: {list(features)}", file=sys.stderr)
        return 1

    print(f"  image column      : {primary_image}")
    print(f"  segmentation mask : {mask_col or '(skipped)'}"
          f"{' [Image]' if mask_is_image else ' [array]' if mask_col else ''}")
    print(f"  metadata columns  : {len(scalar_cols)}")
    print(f"  dropped (arrays)  : {array_cols or 'none'}")

    records: list[dict] = []
    per_patient_seq: dict[tuple, int] = defaultdict(int)

    for split_name, split in dsdict.items():
        if args.limit is not None:
            split = split.select(range(min(args.limit, len(split))))

        keep = scalar_cols + [primary_image] + ([mask_col] if mask_col and mask_is_image else [])
        raw = split.select_columns(keep)
        raw = raw.cast_column(primary_image, HFImage(decode=False))
        if mask_col and mask_is_image:
            raw = raw.cast_column(mask_col, HFImage(decode=False))

        label_feature = features.get("label")
        n_written = n_skipped = 0

        for idx, row in enumerate(raw):
            patient = slug(row.get("patient_id"))
            record = slug(row.get("record"), "r0")
            protocol = slug(row.get("protocol"), "proto")
            view = slug(row.get("view"), "view")

            key = (patient, record, protocol, view)
            seq = per_patient_seq[key]
            per_patient_seq[key] += 1
            stem = f"{record}_{protocol}_{view}_{seq:03d}"

            cell = row[primary_image]
            rel_image = Path("images") / patient / f"{stem}{suffix_of(cell, '.tiff')}"
            dest = out_root / rel_image
            if args.overwrite or not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(image_bytes(cell))
                n_written += 1
            else:
                n_skipped += 1

            rel_mask = ""
            if mask_col and mask_is_image:
                mcell = row[mask_col]
                if mcell and (mcell.get("bytes") or mcell.get("path")):
                    rel_mask = Path("masks") / patient / f"{stem}{suffix_of(mcell, '.png')}"
                    mdest = out_root / rel_mask
                    if args.overwrite or not mdest.exists():
                        mdest.parent.mkdir(parents=True, exist_ok=True)
                        mdest.write_bytes(image_bytes(mcell))
                    rel_mask = str(rel_mask)

            entry = {c: row.get(c) for c in scalar_cols}
            if isinstance(label_feature, ClassLabel) and entry.get("label") is not None:
                entry["label_name"] = label_feature.int2str(int(entry["label"]))
            entry.update({
                "patient_dir": patient,
                "image_path": str(rel_image),
                "mask_path": rel_mask,
                "hf_split": split_name,
                "hf_row_index": idx,
            })
            records.append(entry)

            if (idx + 1) % 500 == 0:
                print(f"  [{split_name}] {idx + 1}/{len(raw)}")

        print(f"  [{split_name}] done: {n_written} written, {n_skipped} already present")

    if mask_col and not mask_is_image:
        write_array_masks(dsdict, out_root, mask_col, records, args)

    import pandas as pd

    meta = pd.DataFrame.from_records(records)
    meta_path = out_root / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    print(f"\nWrote {len(meta)} rows to {meta_path}")

    write_report(meta, out_root, dsdict)

    if args.drop_cache:
        cache = args.hf_cache or Path(datasets.config.HF_DATASETS_CACHE)
        target = Path(cache) / "SemilleroCV___dmr-ir"
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed parquet cache {target}")

    return 0


def write_array_masks(dsdict, out_root: Path, mask_col: str, records: list, args) -> None:
    """Masks stored as uint8 arrays rather than images: materialise them as PNGs."""
    import numpy as np
    from PIL import Image as PILImage

    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_split[rec["hf_split"]].append(rec)

    for split_name, rows in by_split.items():
        split = dsdict[split_name].select_columns([mask_col]).with_format("numpy")
        print(f"  [{split_name}] extracting {len(rows)} segmentation masks")
        for rec in rows:
            arr = split[rec["hf_row_index"]][mask_col]
            arr = np.asarray(arr).squeeze()
            if arr.ndim != 2:
                continue
            rel = Path("masks") / rec["patient_dir"] / (Path(rec["image_path"]).stem + ".png")
            dest = out_root / rel
            if args.overwrite or not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                PILImage.fromarray(arr.astype(np.uint8)).save(dest)
            rec["mask_path"] = str(rel)


def write_report(meta, out_root: Path, dsdict) -> None:
    """Audit what was downloaded, and measure the leakage in the published splits."""
    lines: list[str] = ["# DMR-IR download report", "",
                        f"Source: `{REPO_ID}` (Hugging Face)", ""]

    n_patients = meta["patient_dir"].nunique()
    lines += ["## Counts", "",
              f"- image rows: **{len(meta)}**",
              f"- distinct patients: **{n_patients}**",
              f"- masks written: **{(meta['mask_path'].astype(str) != '').sum()}**"]

    per_patient = meta.groupby("patient_dir").size()
    lines.append(f"- images per patient: min {per_patient.min()}, median "
                 f"{per_patient.median():.0f}, max {per_patient.max()}")

    label_col = "label_name" if "label_name" in meta.columns else "label"
    if label_col in meta.columns:
        pat_labels = meta.groupby("patient_dir")[label_col].nunique()
        inconsistent = sorted(pat_labels[pat_labels > 1].index)
        pat_label = meta.groupby("patient_dir")[label_col].first()
        lines += ["", "## Labels", "",
                  f"- image-level: {dict(Counter(meta[label_col].dropna()))}",
                  f"- patient-level (first row per patient): {dict(Counter(pat_label.dropna()))}",
                  f"- patients with more than one distinct label: **{len(inconsistent)}**"
                  + (f" -> {inconsistent[:20]}" if inconsistent else "")]

    # The headline check: do the published splits share patients?
    splits = {name: set(meta.loc[meta["hf_split"] == name, "patient_dir"])
              for name in meta["hf_split"].unique()}
    lines += ["", "## Published split: patient overlap", "",
              "The `train`/`validation`/`test` split shipped with the mirror is recorded in",
              "`metadata.csv` as provenance only. If the sets below intersect, that split is",
              "image-level and unusable for evaluation -- near-identical frames of one patient",
              "sit on both sides of the boundary.", ""]
    for name, ids in splits.items():
        lines.append(f"- `{name}`: {len(meta[meta['hf_split'] == name])} images, {len(ids)} patients")
    names = list(splits)
    overlapping = False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = splits[a] & splits[b]
            overlapping |= bool(shared)
            lines.append(f"- `{a}` ∩ `{b}`: **{len(shared)} patients**")
    lines.append("")
    lines.append("**Verdict: the published split leaks patients -- rebuild it with "
                 "`scripts/02_create_patient_splits.py`.**" if overlapping else
                 "Verdict: no patient overlap detected in the published split.")

    for col in ("view", "protocol", "role"):
        if col in meta.columns:
            lines += ["", f"## `{col}`", "", "```",
                      meta[col].value_counts(dropna=False).to_string(), "```"]

    # `role` and `label` should say the same thing; if they disagree, one of them is wrong.
    if "role" in meta.columns and label_col in meta.columns:
        lines += ["", "## `role` x label", "", "```",
                  meta.groupby(["role", label_col], dropna=False).size().to_string(), "```"]

    path = out_root / "download_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote audit to {path}")
    if overlapping:
        print("\n*** The published train/val/test split shares patients across sets. ***")
        print("*** Ignore it. Build splits with scripts/02_create_patient_splits.py. ***")


if __name__ == "__main__":
    raise SystemExit(main())
