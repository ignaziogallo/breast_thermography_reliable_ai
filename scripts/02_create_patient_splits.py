#!/usr/bin/env python3
"""Build the patient-level cross-validation folds once, and store them.

Repeated stratified group k-fold with the patient as the grouping variable, plus a
stratified validation subset inside each fold's training portion. Every experiment reads
the stored file; none re-derives its folds, which is what makes two arms comparable on
identical patients. The folds are checked for patient overlap before they are written.

Example
-------
    python scripts/02_create_patient_splits.py --folds 5 --repeats 5
    python scripts/02_create_patient_splits.py \\
        --manifest results/dmr_patient_manifest.csv \\
        --output results/dmr_patient_splits.csv --folds 5 --repeats 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import make_repeated_patient_splits, validate_split_integrity


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--output", type=Path, default=Path("results/patient_splits.csv"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=.15)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest = pd.read_csv(a.manifest)
    splits = make_repeated_patient_splits(manifest, a.folds, a.repeats,
                                          val_fraction=a.val_fraction)
    errors = validate_split_integrity(splits)
    if errors:
        raise RuntimeError("Leakage detected:\n" + "\n".join(errors))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(a.output, index=False)
    print(f"Wrote {len(splits)} assignments to {a.output}; no patient overlap detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
