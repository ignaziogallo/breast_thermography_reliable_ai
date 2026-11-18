#!/usr/bin/env python3
"""Pivot the long DMR-IR image manifest into the wide per-patient table `04` expects.

`scripts/04_train_single_view.py` and `src.data.dataset.SingleViewDataset` read one row per
patient with a `<view>_path` column; `results/dmr_image_manifest.csv` holds one row per
frame. This writes the bridge, so the DMR cohort reaches the existing single-view machinery
without a second implementation of the fold loop.

The view names are `frontal`, `oblique45_a`, `oblique90_a`, `oblique45_b`, `oblique90_b`.
They are deliberately not called left/right: the side assignment follows Figure 7 of Silva
et al. (2014) and is inferred, not verified (see `data/README.md` and `src/data/dmr.py`).

`--require` restricts the cohort to patients holding every named view, which is what a
multi-view model needs; without it a missing view is an empty string and the patient is
kept, which is what a view-dropout robustness study needs. Both counts are printed, because
the cohort size is part of the result and must not be discovered later from a log.

Example
-------
    python scripts/22_build_dmr_view_table.py --protocol Static \\
        --output results/dmr_view_table_static.csv

    python scripts/22_build_dmr_view_table.py --protocol Static --require all \\
        --output results/dmr_view_table_static_complete.csv

    CUDA_VISIBLE_DEVICES=2 python scripts/04_train_single_view.py \\
        --data-root data/DMR-IR --view frontal \\
        --manifest results/dmr_view_table_static.csv \\
        --splits   results/dmr_patient_splits.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dmr import ALL_VIEWS, DYNAMIC, STATIC, build_view_table
from src.utils.io import read_results_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--output", type=Path, default=Path("results/dmr_view_table_static.csv"))
    p.add_argument("--protocol", choices=[STATIC, DYNAMIC], default=STATIC)
    p.add_argument("--views", nargs="+", default=list(ALL_VIEWS), choices=list(ALL_VIEWS))
    p.add_argument("--require", nargs="+", default=[],
                   help="views a patient must hold to be kept; 'all' for every selected view")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label"])
    images = read_results_csv(args.images)

    views = tuple(args.views)
    require = views if args.require == ["all"] else tuple(args.require)
    unknown = set(require) - set(views)
    if unknown:
        raise SystemExit(f"--require names views that are not selected: {sorted(unknown)}")

    table = build_view_table(images, patients, protocol=args.protocol, views=views,
                             require=require)
    if table.empty:
        raise SystemExit(f"No patient holds {require} under protocol {args.protocol}")

    counts = table["patient_binary_label"].value_counts()
    print(f"protocol={args.protocol} views={list(views)} require={list(require) or 'none'}")
    print(f"patients: {len(table)} of {len(patients)} "
          f"({int(counts.get(0, 0))} benign / {int(counts.get(1, 0))} malignant)")
    incomplete = {}
    for view in views:
        held = int((table[f"{view}_path"].astype(str).str.len() > 0).sum())
        if held < len(table):
            incomplete[view] = len(table) - held
        print(f"  {view:<12} {held:>4} patients")

    # An empty path survives this file as "" but pd.read_csv reads it back as NaN, which
    # reaches Path() as a float. A multi-view model handles the gap; a single-view run
    # cannot, so say so here rather than let it fail inside the fold loop.
    if incomplete:
        print("\nWARNING: this table has gaps and is NOT usable for single-view training:")
        for view, n in sorted(incomplete.items()):
            print(f"  {view}: {n} patient(s) without the view")
        first = sorted(incomplete)[0]
        print("  For scripts/04_train_single_view.py, build a restricted table instead:\n"
              f"    python scripts/22_build_dmr_view_table.py --views {first} --require all \\\n"
              f"        --output results/dmr_view_table_{first}.csv")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")
    dropped = set(patients["patient_id"]) - set(table["patient_id"])
    if dropped:
        print(f"Excluded {len(dropped)} patients lacking a required view, e.g. {sorted(dropped)[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
