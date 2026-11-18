#!/usr/bin/env python3
"""Build the second cohort's manifests and audit the dataset before anything is fitted.

Walks `<data-root>/{Benign,Malignant}/IIR####/` and `Diagnostics.xlsx`, writes the patient
and image manifests, and audits what it found: missing views, disagreements between the
folder class and the per-breast labels, byte-identical and near-duplicate images, and the
association of every covariate with the label. The audit describes the cohort; it removes
nobody from it.

Writes `results/patient_manifest.csv`, `results/image_manifest.csv`, the audit tables beside
them, the covariate figures, and `reports/data_audit.md`.

Example
-------
    python scripts/00_audit_dataset.py --data-root data/Breast-Thermal-Image
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import run_audit
from src.data.manifest import build_patient_manifest


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--image-ext", default=".jpg",
                   help="'.jpg' for the renders, '.tiff' for the radiometric tree")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--paper-figures", type=Path, default=Path("paper/figures"))
    p.add_argument("--near-duplicate-threshold", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    manifest, images = build_patient_manifest(a.data_root, compute_hashes=True,
                                              image_ext=a.image_ext)
    a.results_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(a.results_dir / "patient_manifest.csv", index=False)
    images.to_csv(a.results_dir / "image_manifest.csv", index=False)
    summary = run_audit(manifest, images, a.reports_dir, a.results_dir, a.paper_figures,
                        a.near_duplicate_threshold)
    print(summary)
    print(f"Audit report: {a.reports_dir / 'data_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
