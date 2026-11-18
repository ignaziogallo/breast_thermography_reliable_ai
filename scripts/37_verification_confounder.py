#!/usr/bin/env python3
"""The two audit questions the verification cohort had not been asked.

On DMR-IR all five questions of the audit are answered: how strongly the confounder
predicts the label, whether it is legible in the pixels with the diagnosis held constant,
whether a diagnostic model's own score carries it, what the image adds over it, and what
survives when it is controlled. On the verification cohort only three of the five had been
run -- strength, incremental value, control -- while the paper says "the same audit". This
script closes the gap for the two that were missing, for a confounder that is continuous
(age) rather than a two-level era.

**Exploitation** (this script, no GPU). Within a diagnosis stratum, does an arm's own
malignancy score separate older from younger patients? Age is dichotomised at the
*within-stratum* median, which makes the statistic an AUROC directly comparable to DMR-IR's
2012-vs-2013 contrast and balances the two groups by construction. The Spearman correlation
between score and age is reported beside it because it uses the whole continuous variable
and does not depend on where the split falls; the two are allowed to disagree, and if they
do, that is reported. The bootstrap and the permutation test are the routines
`scripts/34_exploitation_interval.py` uses on DMR-IR (`src.evaluation.resampling`), not a
reimplementation, so the number means exactly what the DMR-IR column means.

**Legibility** (`--build-probe-cohorts`, then a GPU run). A frozen probe trained to recover
the age band from the pixels *among patients who share a diagnosis*. This needs training, so
the script only builds the cohorts and folds -- one self-contained experiment directory per
stratum, the construction `scripts/24_epoch_control.py` uses on DMR-IR -- and prints the
commands. Training happens on the A100 host; see `jobs/f2_ver_age_legibility.txt`.

A negative result here is a result: age legible in the metadata but not in the thermogram
would say that this cohort's dominant channel is a clinical variable the image does not
encode, which is the distinction between a nuisance confounder and a legitimate predictor
the audit is built on. It is reported either way.

Example
-------
    python scripts/37_verification_confounder.py
    python scripts/37_verification_confounder.py --build-probe-cohorts
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import make_repeated_patient_splits, validate_split_integrity
from src.evaluation.predictions import pooled_scores
from src.evaluation.resampling import bootstrap_ci, permutation_p
from src.utils.io import ensure_parent, read_results_csv

# arm label -> (experiment directory, model filter, regime)
ARMS = {
    "ConvNeXt-T anterior, frozen":
        ("experiment_02_single_view_linear_probe", "convnext_tiny_linear_probe", "frozen"),
    "ConvNeXt-T anterior, fine-tuned":
        ("experiment_02_single_view_disc_lr", "convnext_tiny_finetune_discriminative_lr",
         "fine-tuned"),
    "ConvNeXt-T anterior, fine-tuned (encoder run)":
        ("experiment_ver_11_encoder_convnext_tiny_finetune",
         "convnext_tiny_finetune_discriminative_lr", "fine-tuned"),
    "Swin-T anterior, fine-tuned":
        ("experiment_ver_11_encoder_swin_tiny_patch4_window7_224_finetune",
         "swin_tiny_patch4_window7_224_finetune_discriminative_lr", "fine-tuned"),
    "ResNet-50 anterior, fine-tuned":
        ("experiment_ver_11_encoder_resnet50_finetune_FAILED_QUALITY_GATE",
         "resnet50_finetune_discriminative_lr", "fine-tuned (quarantined)"),
}

STRATA = {"benign": 0, "malignant": 1}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--patients", type=Path, default=Path("results/patient_manifest.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("results/confounders/verification_exploitation.csv"))
    p.add_argument("--build-probe-cohorts", action="store_true",
                   help="write the within-stratum age-band cohorts and folds for the probe")
    p.add_argument("--probe-cohorts", type=Path,
                   default=Path("results/confounders/verification_age_probe"),
                   help="where the derived cohorts and folds are written")
    p.add_argument("--probe-prefix", type=Path,
                   default=Path("results/experiment_ver_12_age"),
                   help="where each probe *run* will write its predictions")
    p.add_argument("--probe-encoders", nargs="+",
                   default=["convnext_tiny", "swin_tiny_patch4_window7_224"],
                   help="the pre-specified backbones, as on DMR-IR; no new architectures")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutation", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def age_band(meta: pd.DataFrame, label: int) -> pd.Series:
    """1 = older than the median of this diagnosis stratum, 0 = younger; ties dropped.

    Splitting inside the stratum, not across the cohort, is what keeps the contrast free of
    the diagnosis: malignant patients are twelve years older on average here, so a cohort-wide
    median would put most benign patients on one side and most malignant on the other and the
    statistic would be measuring the label again.
    """
    stratum = meta[meta["patient_binary_label"].astype(int) == label]
    cut = stratum["age"].median()
    band = (stratum["age"] > cut).astype(int)
    band[stratum["age"] == cut] = -1        # exactly at the median: excluded, not assigned
    return band


def analyse(label: str, spec, args, meta: pd.DataFrame) -> dict | None:
    """One arm: its diagnostic AUROC and how well its score separates the age bands."""
    directory, model, regime = spec
    path = args.results_root / directory / "predictions.csv"
    if not path.exists():
        print(f"  [skip] {label}: {path} absent", flush=True)
        return None
    pooled = pooled_scores(path, model)

    common = pooled.index.intersection(meta.index)
    p = pooled.loc[common]
    row = {"arm": label, "regime": regime,
           "diagnostic_auroc": roc_auc_score(p["y_true"], p["score"])}

    for name, y in STRATA.items():
        band = age_band(meta, y)
        ids = band.index.intersection(p.index)
        band = band.loc[ids]
        keep = band >= 0
        score = p.loc[ids[keep], "score"].to_numpy()
        older = band[keep].to_numpy()
        if len(score) < 20 or len(np.unique(older)) < 2:
            continue
        auc = roc_auc_score(older, score)
        lo, hi, valid = bootstrap_ci(score, older, args.n_bootstrap,
                                     np.random.default_rng(args.seed))
        pval = permutation_p(score, older, auc, args.n_permutation,
                             np.random.default_rng(args.seed))
        # Spearman against the continuous age, on the same patients, so that the
        # dichotomisation is visibly not doing the work.
        rho, rho_p = spearmanr(score, meta.loc[ids[keep], "age"].to_numpy())
        row |= {f"n_{name}": len(score),
                f"age_auroc_{name}": auc,
                f"age_auroc_{name}_lo": lo,
                f"age_auroc_{name}_hi": hi,
                f"age_auroc_{name}_perm_p": pval,
                f"age_auroc_{name}_boot_valid": valid,
                f"spearman_{name}": rho,
                f"spearman_{name}_p": rho_p}
    return row


def build_probe_cohorts(args, meta: pd.DataFrame) -> list[str]:
    """One experiment directory per stratum: manifest with the age band as the target.

    The trainer reads `patient_binary_label`, so the band is written into that column and the
    diagnosis is preserved beside it. Folds are rebuilt on the new target -- reusing the
    diagnosis folds would stratify on the wrong variable.
    """
    commands = []
    for name, y in STRATA.items():
        band = age_band(meta, y)
        cohort = meta.loc[band.index[band >= 0]].copy()
        cohort["diagnosis_label"] = cohort["patient_binary_label"].astype(int)
        cohort["patient_binary_label"] = band[band >= 0].astype(int)
        cohort["age_cut"] = float(meta.loc[band.index, "age"].median())
        cohort = cohort.reset_index()

        splits = make_repeated_patient_splits(cohort, args.folds, args.repeats)
        errors = validate_split_integrity(splits)
        if errors:
            raise SystemExit(f"Leakage in the age-probe splits ({name}):\n" +
                             "\n".join(errors))

        cohort_dir = args.probe_cohorts / name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(cohort_dir / "manifest.csv", index=False)
        splits.to_csv(cohort_dir / "splits.csv", index=False)
        counts = cohort["patient_binary_label"].value_counts().to_dict()
        print(f"{cohort_dir}: {len(cohort)} patients, younger/older = "
              f"{counts.get(0, 0)}/{counts.get(1, 0)}, median age "
              f"{cohort['age_cut'].iloc[0]:.0f}")

        for encoder in args.probe_encoders:
            short = "convnext" if encoder.startswith("convnext") else encoder.split("_")[0]
            run = Path(f"{args.probe_prefix}_within_{name}_{short}")
            commands.append(
                "python scripts/04_train_single_view.py "
                "--data-root data/Breast-Thermal-Image "
                f"--manifest {cohort_dir}/manifest.csv --splits {cohort_dir}/splits.csv "
                f"--view anterior --encoder {encoder} --freeze-backbone "
                "--no-early-stopping --epochs 15 --batch-size 16 --num-workers 4 "
                f"--output-dir {run} > logs/f2_ver_age_within_{name}_{short}.log 2>&1")
    return commands


def latex_table(df: pd.DataFrame) -> str:
    """The second cohort's shortcut-usage table as LaTeX, one row per arm."""
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Arm & Diagn. & Age within benign & Age within malignant \\",
             r" & AUROC & AUROC [95\% CI], $p$ & AUROC [95\% CI], $p$ \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for stratum in ("benign", "malignant"):
            key = f"age_auroc_{stratum}"
            if pd.isna(r.get(key)):
                cells.append("--")
                continue
            p = r[f"{key}_perm_p"]
            ptxt = f"{p:.2f}" if p >= 0.01 else (f"{p:.3f}" if p >= 0.001 else "<0.001")
            cells.append(f"{r[key]:.3f} [{r[f'{key}_lo']:.3f}, {r[f'{key}_hi']:.3f}], {ptxt}")
        lines.append(f"{r['arm']} & {r['diagnostic_auroc']:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    patients = read_results_csv(args.patients).dropna(subset=["patient_binary_label", "age"])
    meta = patients.set_index("patient_id")

    if args.build_probe_cohorts:
        commands = build_probe_cohorts(args, meta)
        print("\nRun these on the GPU host (they are also in jobs/f2_ver_age_legibility.txt):\n")
        print("\n".join(commands))
        return 0

    rows = [r for label, spec in ARMS.items() if (r := analyse(label, spec, args, meta))]
    df = pd.DataFrame(rows).sort_values("age_auroc_benign", ascending=False)

    ensure_parent(args.output)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}\n")

    show = ["arm", "regime", "diagnostic_auroc", "n_benign", "age_auroc_benign",
            "age_auroc_benign_lo", "age_auroc_benign_hi", "age_auroc_benign_perm_p",
            "spearman_benign", "spearman_benign_p"]
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(df[[c for c in show if c in df.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + latex_table(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
