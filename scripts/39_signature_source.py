#!/usr/bin/env python3
"""Where in the image is the acquisition era written?

The audit establishes that the era is legible in the pixels of DMR-IR among patients who
share a diagnosis (0.813--0.823 for every backbone). That is the fact the paper needs, and
it is silent about *why*. The difference matters for how the finding should be read:

  background only   the signature is in the room, the camera or the framing -- physically
                    non-pathological, and no reading of it as physiology survives;
  breast ROI only   the signature is entangled with the tissue that carries the diagnosis,
                    which does not contradict the confounding account but sharpens the
                    identifiability problem: era and pathology would then be written in the
                    same pixels;
  global statistics  a single number per image -- mean temperature and its spread -- would
                    mean the shortcut needs no spatial structure at all, and that any model
                    with a global pooling layer gets it for free;
  histogram only    the same, one step richer: the distribution without any geometry.

Every representation is tabular, so all four are answered by a linear probe on a few dozen
numbers rather than by a network, and the comparison between them is not confounded by
architecture. The masks distributed with the mirror define the breast region, so
"background" is its exact complement and the three pixel selections partition the frame.

Two stages, because the pixels live on the GPU host and the probe does not:

    --extract   read every selected frame and its mask, write the feature table
    (default)   fit the within-benign era probes on that table and report

The probe is the same protocol as everything else in the paper: patient-level grouped folds,
five repeats, one prediction per patient per repeat, patient bootstrap intervals. The
contrast is benign 2012 vs 2013 -- diagnosis held constant -- so a positive result cannot be
the label in disguise.

Example
-------
    python scripts/39_signature_source.py --extract --data-root data/DMR-IR   # on the host
    python scripts/39_signature_source.py                                     # anywhere
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dmr import ERA_CONTRASTS
from src.data.splits import make_repeated_patient_splits, validate_split_integrity
from src.evaluation.summary import summarize_predictions
from src.training.tabular_cv import run_patient_cv
from src.utils.io import ensure_parent, read_results_csv, read_thermal_image, write_json
from src.utils.seed import seed_everything

# The within-benign contrast of the era decoders: diagnosis held constant, both years
# well populated.
BENIGN, (EARLY,), (LATE,) = ERA_CONTRASTS["within_benign"]

PERCENTILES = (5, 25, 50, 75, 95)
N_BINS = 32
# A fixed range in degrees Celsius, so that a histogram bin means the same temperature in
# every image. Per-image ranges would normalise away exactly the between-session differences
# this script is looking for. The range must cover the cohort: the first run used 25-40 degC
# and the extraction then reported an observed range of 18.7-36.8, so every pixel below 25 --
# the cold end of the background, about the coldest twentieth of the frame -- fell outside
# the bins and was silently dropped by `np.histogram`. That handicapped the histogram row
# against the percentile rows, which is not a comparison worth reporting.
HIST_RANGE = (18.0, 38.0)

BLOCKS = ("global", "roi", "background", "histogram")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract", action="store_true", help="read the pixels and write features")
    p.add_argument("--data-root", type=Path, default=Path("data/DMR-IR"))
    p.add_argument("--images", type=Path, default=Path("results/dmr_image_manifest.csv"))
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--features", type=Path, default=Path("results/dmr_signature_features.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("results/confounders/signature_source.csv"))
    p.add_argument("--protocol", default="Static", help="Static or Dinamic")
    p.add_argument("--view-group", default="frontal")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def summarise(values: np.ndarray, prefix: str) -> dict:
    """Location, spread and shape of one pixel selection, in degrees."""
    if values.size < 10:
        return {}
    out = {f"{prefix}__mean": float(values.mean()), f"{prefix}__sd": float(values.std()),
           f"{prefix}__min": float(values.min()), f"{prefix}__max": float(values.max())}
    for q in PERCENTILES:
        out[f"{prefix}__p{q:02d}"] = float(np.percentile(values, q))
    out[f"{prefix}__iqr"] = out[f"{prefix}__p75"] - out[f"{prefix}__p25"]
    centred = values - values.mean()
    sd = values.std()
    if sd > 1e-6:
        out[f"{prefix}__skew"] = float((centred ** 3).mean() / sd ** 3)
        out[f"{prefix}__kurtosis"] = float((centred ** 4).mean() / sd ** 4)
    return out


def frame_features(image_path: Path, mask_path: Path) -> dict:
    """Statistics of one frame under three pixel selections.

    Two details of this dataset are load-bearing here. The frames are read with
    `read_thermal_image`, not with a bare `convert("L")`: a float32 degrees-Celsius TIFF
    collapses to 8-bit integers under conversion, which would quantise away exactly the
    between-session temperature differences this script is looking for. And the mirror
    stores masks at 224x224 while frames are 640x480, so the mask is upsampled to the frame
    rather than the frame downsampled to the mask -- the same choice
    `src/data/dmr_datasets.py` makes for training, and for the same reason: the
    alternative silently changes what is
    being measured. Nearest-neighbour upsampling leaves the breast boundary blocky, which is
    immaterial to percentile statistics over tens of thousands of pixels but means the two
    selections are exact complements only up to that boundary.
    """
    from PIL import Image

    array = read_thermal_image(image_path).astype(np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    with Image.open(mask_path) as im:
        if im.size != (array.shape[1], array.shape[0]):
            im = im.resize((array.shape[1], array.shape[0]), Image.NEAREST)
        raw = np.asarray(im)
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    # Any non-zero pixel is breast. This is the test `src/data/dmr_datasets.py` applies, and it is
    # the one that matters: the mirror's masks are binary but not necessarily 0/255, so a
    # mid-grey threshold selects nothing on a 0/1 mask -- which is exactly what it did here.
    mask = raw > 0

    flat = array.reshape(-1)
    features = summarise(flat, "global")
    features |= summarise(array[mask], "roi")
    features |= summarise(array[~mask], "background")
    counts, _ = np.histogram(flat, bins=N_BINS, range=HIST_RANGE)
    # Normalised by the pixels that fell inside the range, and the fraction that did is kept
    # as a feature of its own: if a frame sits partly outside HIST_RANGE, that is information
    # about the frame, and it must be visible rather than absorbed into the normalisation.
    features["histogram__in_range"] = float(counts.sum()) / flat.size
    total = max(int(counts.sum()), 1)
    features |= {f"histogram__bin{i:02d}": float(c) / total for i, c in enumerate(counts)}
    features["roi_fraction"] = float(mask.mean())
    features["_mask_values"] = np.unique(raw)[:6]
    return features


def extract(args, cohort: pd.DataFrame) -> pd.DataFrame:
    """Read every selected frame and its mask; write one row of descriptors per patient."""
    frames = read_results_csv(args.images)
    frames = frames[(frames["protocol"] == args.protocol)
                    & (frames["view_group"] == args.view_group)
                    & (frames["patient_id"].isin(set(cohort["patient_id"])))]
    if frames.empty:
        raise SystemExit(f"No {args.protocol}/{args.view_group} frames for these patients")
    print(f"{len(frames)} frames over {frames['patient_id'].nunique()} patients", flush=True)

    rows, ranges, degenerate = [], [], []
    for n, r in enumerate(frames.itertuples(), start=1):
        image, mask = args.data_root / r.image_path, args.data_root / r.mask_path
        if not image.exists() or not mask.exists():
            print(f"  [skip] missing {image if not image.exists() else mask}", flush=True)
            continue
        f = frame_features(image, mask)
        if not f:
            continue
        values = f.pop("_mask_values")
        if n == 1:
            print(f"  first mask {Path(r.mask_path).name}: values {list(values)}, "
                  f"breast fraction {f['roi_fraction']:.3f}", flush=True)
        # A mask that selects everything or nothing yields no ROI/background split. One such
        # frame is a defect of that file; many mean the convention above is wrong, and the
        # run is stopped rather than reported over whatever survived.
        if f["roi_fraction"] in (0.0, 1.0):
            degenerate.append(r.mask_path)
            continue
        ranges.append((f["global__min"], f["global__max"]))
        rows.append({"patient_id": r.patient_id, **f})
        if n % 100 == 0:
            print(f"  {n}/{len(frames)}", flush=True)

    if degenerate:
        print(f"\n{len(degenerate)} of {len(frames)} masks select everything or nothing, "
              f"e.g. {degenerate[:3]}")
        if len(degenerate) > 0.1 * len(frames):
            raise SystemExit("More than a tenth of the masks are degenerate: the foreground "
                             "test in frame_features() does not match what this mirror "
                             "stores. Inspect one mask before rerunning.")
    if not rows:
        raise SystemExit(
            f"No frame under --data-root {args.data_root} could be read, so there is "
            f"nothing to extract. {len(frames)} frames were selected from {args.images}; "
            "each needs its image and its mask on disk at the path the manifest records, "
            "relative to --data-root (see 'Dataset preparation' in the README). Without "
            "the dataset, run this script with no --extract: it then fits the probes on "
            f"the cached descriptors in {args.features}, which ship with this package.")
    table = pd.DataFrame(rows).groupby("patient_id").mean().reset_index()
    lo = min(a for a, _ in ranges)
    hi = max(b for _, b in ranges)
    print(f"\npixel value range across frames: [{lo:.2f}, {hi:.2f}]")
    if not (0.0 < lo < 45.0 and 20.0 < hi < 60.0):
        print("WARNING: these do not look like degrees Celsius. The histogram bins assume "
              f"{HIST_RANGE} degC; check what the mirror stores before reading the histogram "
              "row, and rerun with a corrected HIST_RANGE if needed.")
    ensure_parent(args.features)
    table.to_csv(args.features, index=False)
    print(f"Wrote {args.features}: {len(table)} patients x {table.shape[1] - 1} features")
    return table


def era_cohort(patients: pd.DataFrame) -> pd.DataFrame:
    """Benign patients acquired in the two well-populated years, labelled by era."""
    df = patients.copy()
    df["year"] = pd.to_datetime(df["visit_date"], errors="coerce").dt.year
    df = df[(df["patient_binary_label"].astype(int) == BENIGN) & df["year"].isin((EARLY, LATE))]
    df = df.copy()
    df["diagnosis_label"] = BENIGN
    df["patient_binary_label"] = (df["year"] == LATE).astype(int)
    return df[["patient_id", "year", "diagnosis_label", "patient_binary_label"]]


def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label"])
    cohort = era_cohort(patients)
    print(f"within-benign era cohort: {len(cohort)} patients, "
          f"{int((cohort['patient_binary_label'] == 0).sum())} in {EARLY} / "
          f"{int((cohort['patient_binary_label'] == 1).sum())} in {LATE}")

    if args.extract:
        extract(args, cohort)
        print("\nNow rerun without --extract to fit the probes.")
        return 0

    if not args.features.exists():
        raise SystemExit(f"{args.features} does not exist: run --extract on the host that "
                         "holds data/DMR-IR first (see jobs/f3_o1_signature_source.txt)")
    features = read_results_csv(args.features)
    table = cohort.merge(features, on="patient_id", how="inner")
    missing = set(cohort["patient_id"]) - set(table["patient_id"])
    if missing:
        print(f"{len(missing)} cohort patients have no features and are dropped, "
              f"e.g. {sorted(missing)[:5]}")

    splits = make_repeated_patient_splits(table, args.folds, args.repeats)
    errors = validate_split_integrity(splits)
    if errors:
        raise SystemExit("Leakage in the signature-source splits:\n" + "\n".join(errors))

    estimator = Pipeline([("scale", StandardScaler()),
                          ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}

    predictions = []
    for block in BLOCKS:
        cols = [c for c in table.columns if c.startswith(f"{block}__")]
        if not cols:
            print(f"[skip] {block}: no columns in {args.features}")
            continue
        print(f"{block}: {len(cols)} features", flush=True)
        predictions.append(run_patient_cv(table, cols, splits, clone(estimator), grid, block))
    if not predictions:
        raise SystemExit("No feature block was present in the table")

    preds = pd.concat(predictions, ignore_index=True)
    _, ci, _ = summarize_predictions(preds, n_bootstrap=args.n_bootstrap, seed=args.seed)
    ci = ci[ci["metric"] == "roc_auc"].rename(columns={"model": "representation"})

    ensure_parent(args.output)
    ci.to_csv(args.output, index=False)
    preds.to_csv(args.output.with_name("signature_source_predictions.csv"), index=False)
    write_json({
        "question": "where in the image is the acquisition era legible, within benign patients",
        "contrast": f"benign {EARLY} vs {LATE}",
        "n_patients": int(len(table)),
        "protocol": args.protocol, "view_group": args.view_group,
        "histogram_bins": N_BINS, "histogram_range_degC": list(HIST_RANGE),
        "estimator": "standardised logistic regression, C chosen on the validation split",
    }, args.output.parent / "signature_source_config.json")

    print("\nEra decodability within benign patients, by representation:")
    print(ci[["representation", "estimate", "ci_lower", "ci_upper"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
