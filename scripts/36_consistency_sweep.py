#!/usr/bin/env python3
"""Does every number printed in the paper still exist in `results/`?

Two checks, deliberately different in kind, because they fail in different ways.

**Anchors (semantic).** A hand-written map from a claim in the manuscript to the exact file,
row and column of `results/` that produces it. This is the check that catches a number which
is right in the abstract and stale in the table, or a table regenerated from a different arm
than the text describes. It is the only check that knows what a number *means*, and it is
therefore the only one whose failure is unambiguous. Anchors cover the main-text tables and
every headline figure quoted in prose.

**Sweep (syntactic).** Every numeric literal in `main.tex` (and in `supplementary.tex`, for
a version of the manuscript that has one) is looked up in an index of every numeric cell
of every CSV under `results/`. A value found nowhere in the results tree cannot have been
produced by this pipeline: either it is stale, it is
derived (a difference, a count, a page number), or it was typed. The sweep cannot tell those
apart -- it reports, it does not judge, and a hit is evidence of existence, not of
provenance. Its purpose is to bound the manual reading: the reviewer of this report only has
to look at the unmatched list.

Tolerance is 5e-4 throughout: the manuscript prints three decimals and several CSVs store
four, so exact equality would flag correct roundings.

Example
-------
    python scripts/36_consistency_sweep.py
    python scripts/36_consistency_sweep.py --strict     # non-zero exit if an anchor fails
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

TOL = 5e-4


def _tex_files() -> list[Path]:
    """The manuscript sources, wherever this tree keeps them.

    The development repository nests them under the submission directory; the
    reproducibility package ships them flat beside the code. Resolving both means one
    script serves both trees, and a missing manuscript degrades to "anchors only" rather
    than to a crash. The arXiv version is a single `main.tex` with the appendices inside
    it; a version with a separate `supplementary.tex` has both files swept.
    """
    for parent in (Path("paper/paper_arxiv"), Path("paper")):
        main = parent / "main.tex"
        if main.exists():
            supplementary = parent / "supplementary.tex"
            return [main, supplementary] if supplementary.exists() else [main]
    return []


TEX_FILES = _tex_files()
REPORT = Path("results/paper_consistency.md")

# Files whose cells are raw per-frame or per-patient scores, identifiers, hashes or split
# assignments. Indexing them would make the sweep match almost any three-decimal number by
# coincidence and would say nothing about where a manuscript number came from.
SKIP_BASENAMES = ("predictions", "patient_scores", "splits", "manifest", "hashes",
                  "view_subset_predictions", "fold_log", "asymmetry_features",
                  "near_duplicate", "exact_duplicates", "data_audit")


# ---------------------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------------------
def printed_tol(value: float) -> float:
    """Half a unit in the last decimal the manuscript actually prints, floored at TOL."""
    text = repr(float(value))
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return max(TOL, 0.5 * 10 ** (-decimals)) + 1e-9


@dataclass
class Anchor:
    """One claim, its printed values, and the cells of `results/` that must produce them."""
    claim: str
    printed: list[float]
    path: str
    where: dict[str, object]
    columns: list[str]
    note: str = ""
    got: list[float] = field(default_factory=list)
    status: str = ""

    def check(self) -> None:
        p = Path(self.path)
        if not p.exists():
            self.status, self.note = "MISSING FILE", f"{self.path} does not exist"
            return
        df = pd.read_csv(p)
        for col, val in self.where.items():
            if col not in df.columns:
                self.status, self.note = "MISSING COLUMN", f"no column {col!r} in {self.path}"
                return
            df = df[df[col].astype(str) == str(val)]
        if len(df) != 1:
            self.status = "AMBIGUOUS ROW"
            self.note = f"{len(df)} rows match {self.where} in {self.path}"
            return
        row = df.iloc[0]
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            self.status, self.note = "MISSING COLUMN", f"no column(s) {missing} in {self.path}"
            return
        self.got = [float(row[c]) for c in self.columns]
        # A printed value is only claimed to as many decimals as it shows: 0.47 is a correct
        # rendering of 0.4669, and demanding 5e-4 of it would report a rounding as an error.
        bad = [(c, p_, g) for c, p_, g in zip(self.columns, self.printed, self.got)
               if abs(p_ - g) > printed_tol(p_)]
        self.status = "OK" if not bad else "MISMATCH"
        if bad:
            self.note = "; ".join(f"{c}: paper {p_} vs results {g:.6f}" for c, p_, g in bad)


CI = ["estimate", "ci_lower", "ci_upper"]
LEAK = "results/experiment_dmr_01_leakage_combined/metrics.csv"
EXPL = "results/confounders/exploitation_interval.csv"
COND = "results/confounders/dmr_session_conditional/caliper_series.csv"
MULT = "results/confounders/caliper_multiplicity.csv"
LADD = "results/matching/dmr/rung_vs_control_dmr.csv"
LADS = "results/matching/dmr/rung_vs_control_dmr_allframes_swin.csv"
LADC = "results/matching/dmr/rung_vs_control_dmr_allframes_convnext.csv"
VLAD = "results/matching/verification/rung_vs_control_verification.csv"
VCOND = "results/confounders/verification_conditional/caliper_series.csv"
SIG = "results/confounders/signature_source.csv"
VEXP = "results/confounders/verification_exploitation.csv"
SMD = "results/matching/smd_all_rungs.csv"
SUPP = "results/confounders/cross_label_support.csv"
SEL = "results/confounders/selection_diagnostics.csv"
RUNG = ["image_auroc", "ci_lower", "ci_upper"]
SWIN_AF = "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224::patient_level"
CNX_AF = "experiment_dmr_01_leakage_combined::patient_level"

ANCHORS: list[Anchor] = [
    # --- Table 1, every information channel -------------------------------------------
    Anchor("Tab. channels: acquisition date alone", [0.980, 0.967, 0.992],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "acquisition_date_only", "metric": "roc_auc"}, CI),
    Anchor("Tab. channels: tabular record, all fields", [0.952, 0.921, 0.976],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "as_published_all_fields", "metric": "roc_auc"}, CI),
    Anchor("Tab. channels: clinical text, TF-IDF", [0.927, 0.888, 0.960],
           "results/experiment_dmr_05_text_probe/bootstrap_ci.csv",
           {"model": "all_text", "metric": "roc_auc"}, CI),
    Anchor("Tab. channels: thermogram, image-level split", [0.968, 0.945, 0.985],
           LEAK, {"model": "image_level__per_patient"}, CI),
    Anchor("Tab. channels: thermogram, all frames, Swin-T", [0.949, 0.918, 0.973],
           "results/experiment_dmr_11_allframes_swin_tiny_patch4_window7_224/metrics.csv",
           {"model": "patient_level__per_patient"}, CI),
    Anchor("Tab. channels: thermogram, all frames, ConvNeXt-T", [0.922, 0.882, 0.955],
           LEAK, {"model": "patient_level__per_patient"}, CI),
    Anchor("Tab. channels: thermogram, frontal, fine-tuned", [0.923, 0.887, 0.954],
           "results/experiment_dmr_11_encoder_convnext_tiny_finetune/bootstrap_ci.csv",
           {"model": "convnext_tiny_finetune_discriminative_lr", "metric": "roc_auc"}, CI),
    Anchor("Tab. channels: thermogram, frontal, frozen probe", [0.815, 0.760, 0.867],
           "results/experiment_dmr_02_frontal_static/bootstrap_ci.csv",
           {"model": "convnext_tiny_linear_probe", "metric": "roc_auc"}, CI),
    Anchor("Tab. channels: age alone", [0.416, 0.342, 0.498],
           "results/experiment_dmr_07_metadata/bootstrap_ci.csv",
           {"model": "logistic_age_only", "metric": "roc_auc"}, CI),

    # --- Table 2 and supplement, shortcut usage ----------------------------------------
    Anchor("Tab. exploitation: ConvNeXt-T all frames", [0.922, 0.636, 0.540, 0.729, 0.005],
           EXPL, {"arm": "ConvNeXt-T all frames, frozen"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),
    Anchor("Tab. exploitation: Swin-T all frames", [0.949, 0.593, 0.495, 0.691, 0.063],
           EXPL, {"arm": "Swin-T all frames, frozen"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),
    Anchor("Tab. exploitation: ConvNeXt-T frontal, fine-tuned", [0.923, 0.537, 0.438, 0.630, 0.47],
           EXPL, {"arm": "ConvNeXt-T frontal, fine-tuned"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),
    Anchor("Tab. exploitation: Swin-T frontal, fine-tuned", [0.920, 0.502, 0.402, 0.599, 0.97],
           EXPL, {"arm": "Swin-T frontal, fine-tuned"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),
    Anchor("Tab. exploitation: ConvNeXt-T frontal, frozen", [0.815, 0.509, 0.413, 0.610, 0.85],
           EXPL, {"arm": "ConvNeXt-T frontal, frozen"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),
    Anchor("Supp. exploitation: ResNet-50 all frames (quarantined arm reported as failure)",
           [0.898, 0.525, 0.431, 0.627, 0.62], EXPL, {"arm": "ResNet-50 all frames, frozen"},
           ["diagnostic_auroc", "era_auroc_benign", "era_auroc_benign_lo",
            "era_auroc_benign_hi", "era_auroc_benign_perm_p"]),

    # --- Paired view contrast (the mechanism sentence) ---------------------------------
    Anchor("View contrast, ConvNeXt-T frontal vs all frames",
           [0.509, 0.631, 0.122, 0.039, 0.202],
           "results/confounders/exploitation_view_contrast.csv", {"encoder": "ConvNeXt-T"},
           ["era_auroc_frontal", "era_auroc_all_frames", "delta", "delta_lo", "delta_hi"]),
    Anchor("View contrast, Swin-T frontal vs all frames",
           [0.524, 0.582, 0.058, -0.071, 0.178],
           "results/confounders/exploitation_view_contrast.csv", {"encoder": "Swin-T"},
           ["era_auroc_frontal", "era_auroc_all_frames", "delta", "delta_lo", "delta_hi"]),
    Anchor("View contrast, ResNet-50 (quarantined)", [0.403, 0.520, 0.116, 0.043, 0.194],
           "results/confounders/exploitation_view_contrast.csv",
           {"encoder": "ResNet-50 (quarantined)"},
           ["era_auroc_frontal", "era_auroc_all_frames", "delta", "delta_lo", "delta_hi"]),

    # --- Table 3, incremental value ----------------------------------------------------
    Anchor("Tab. incremental: Swin all frames alone vs date",
           [0.945, 0.980, -0.036, -0.068, -0.010],
           "results/experiment_dmr_13_incremental_swin_allframes/paired.csv",
           {"model_a": "image_only_reference", "model_b": "visit_date_only_reference"},
           ["auroc_a", "auroc_b", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + Swin all frames (fixed)",
           [0.984, 0.003, -0.008, 0.015],
           "results/experiment_dmr_13_incremental_swin_allframes/paired.csv",
           {"model_a": "visit_date_plus_image_fixed", "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + Swin all frames (stacked)",
           [0.984, 0.004, -0.007, 0.014],
           "results/experiment_dmr_13_incremental_swin_allframes/paired.csv",
           {"model_a": "visit_date_plus_image_stacked_exploratory",
            "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: frontal image alone vs date",
           [0.811, -0.169, -0.222, -0.120],
           "results/experiment_dmr_08_epoch_frontal/paired.csv",
           {"model_a": "image_only_reference", "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + frontal (fixed)", [0.926, -0.054, -0.084, -0.028],
           "results/experiment_dmr_08_epoch_frontal/paired.csv",
           {"model_a": "visit_date_plus_image_fixed", "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + frontal (stacked)", [0.977, -0.003, -0.010, 0.004],
           "results/experiment_dmr_08_epoch_frontal/paired.csv",
           {"model_a": "visit_date_plus_image_stacked_exploratory",
            "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: ConvNeXt all frames alone vs date",
           [0.920, -0.060, -0.098, -0.028],
           "results/experiment_dmr_08_epoch_allframes/paired.csv",
           {"model_a": "image_only_reference", "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + ConvNeXt all frames (fixed)",
           [0.967, -0.013, -0.033, 0.003],
           "results/experiment_dmr_08_epoch_allframes/paired.csv",
           {"model_a": "visit_date_plus_image_fixed", "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Tab. incremental: date + ConvNeXt all frames (stacked)",
           [0.977, -0.003, -0.015, 0.009],
           "results/experiment_dmr_08_epoch_allframes/paired.csv",
           {"model_a": "visit_date_plus_image_stacked_exploratory",
            "model_b": "visit_date_only_reference"},
           ["auroc_a", "delta_auroc", "delta_ci_lower", "delta_ci_upper"]),

    # --- How many patients each ladder rung actually scored ------------------------------
    # The cohort size and the scored size differ on every frontal rung, because the frozen
    # frontal probe needs a static anterior view and five DMR-IR patients hold none. The
    # figure prints the scored size and the supplement table prints both, so both are
    # anchored: this is the 236-versus-231 the manuscript is asked to keep straight.
    Anchor("Ladder (frontal): cohort and scored size at the reference rung", [236, 231],
           LADD, {"cohort": "original"}, ["n_patients", "n_scored"]),
    Anchor("Ladder (frontal): cohort and scored size at the overlap rung", [60, 58],
           LADD, {"cohort": "overlap"}, ["n_patients", "n_scored"]),
    Anchor("Ladder (frontal): cohort and scored size at the null control", [174, 170],
           LADD, {"cohort": "propensity_clinical"}, ["n_patients", "n_scored"]),
    Anchor("Ladder (all frames, Swin): every patient of each cohort is scored", [236, 236],
           LADS, {"cohort": "original"}, ["n_patients", "n_scored"]),
    Anchor("Ladder (all frames, ConvNeXt): overlap rung scores all 60", [60, 60],
           LADC, {"cohort": "overlap"}, ["n_patients", "n_scored"]),

    # --- Era legibility, the per-encoder decoders quoted in the text and Fig. 1 ----------
    # These exist because the manuscript once quoted 0.814 for ConvNeXt-T from an earlier
    # pilot of the identical configuration (`experiment_dmr_03_epoch_within_benign`) while
    # the encoder table quoted 0.813 from the run below. Two runs, one recipe, and nothing
    # tying either to the sentence that used it. The anchors now name the run.
    Anchor("Era legibility: ConvNeXt-T within benign", [0.813, 0.740, 0.887],
           "results/experiment_dmr_12_epoch_convnext_tiny_within_benign/bootstrap_ci.csv",
           {"model": "epoch_decoder_within_benign", "metric": "roc_auc"}, CI,
           note="canonical run; the dmr_03 pilot of the same recipe reads 0.814 and is "
                "quoted only as rerun variance"),
    Anchor("Era legibility: Swin-T within benign", [0.821, 0.748, 0.885],
           "results/experiment_dmr_12_epoch_swin_tiny_patch4_window7_224_within_benign/"
           "bootstrap_ci.csv",
           {"model": "epoch_decoder_within_benign", "metric": "roc_auc"}, CI),
    Anchor("Era legibility: ResNet-50 within benign", [0.823, 0.747, 0.891],
           "results/experiment_dmr_12_epoch_resnet50_within_benign/bootstrap_ci.csv",
           {"model": "epoch_decoder_within_benign", "metric": "roc_auc"}, CI),
    Anchor("Era legibility: the pilot rerun of the ConvNeXt-T decoder",
           [0.814, 0.739, 0.886],
           "results/experiment_dmr_03_epoch_within_benign/bootstrap_ci.csv",
           {"model": "epoch_decoder_within_benign", "metric": "roc_auc"}, CI,
           note="identical manifest, splits and seed as dmr_12; the difference is "
                "run-to-run variance and the supplement says so"),

    # --- Matched ladder, frontal arm ---------------------------------------------------
    Anchor("Ladder (frontal): original reference", [0.833, 0.780, 0.882],
           LADD, {"cohort": "original"}, RUNG),
    Anchor("Ladder (frontal): overlap rung + size controls",
           [0.544, 0.390, 0.705, 0.731, 0.828],
           LADD, {"cohort": "overlap"}, RUNG + ["control_auroc_min", "control_auroc_max"]),
    Anchor("Ladder (frontal): null control (propensity, clinical only)",
           [0.823, 0.757, 0.879, 0.834, 0.848],
           LADD, {"cohort": "propensity_clinical"},
           RUNG + ["control_auroc_min", "control_auroc_max"]),
    Anchor("Ladder (frontal): confounder on the overlap cohort", [0.720],
           LADD, {"cohort": "overlap"}, ["confounder_auroc"]),

    # --- Matched ladder, all-frames arms ------------------------------------------
    Anchor("Ladder (Swin all frames): original", [0.950, 0.920, 0.974],
           LADS, {"cohort": "original"}, RUNG),
    Anchor("Ladder (Swin all frames): overlap + controls", [0.757, 0.628, 0.878, 0.853, 0.894],
           LADS, {"cohort": "overlap"}, RUNG + ["control_auroc_min", "control_auroc_max"]),
    Anchor("Ladder (Swin all frames): null control", [0.954, 0.922, 0.977, 0.939, 0.952],
           LADS, {"cohort": "propensity_clinical"},
           RUNG + ["control_auroc_min", "control_auroc_max"]),
    # Read from the run itself, not from the ladder report: the report stores four decimals,
    # and 0.9225 is exactly the value where a third-decimal rounding could go either way.
    Anchor("Ladder (ConvNeXt all frames): original", [0.922, 0.883, 0.956],
           "results/experiment_dmr_14_matched_allframes_convnext_original/metrics.csv",
           {"model": "patient_level__per_patient"}, CI),
    Anchor("Ladder (Swin all frames): original, from the run", [0.950, 0.920, 0.974],
           "results/experiment_dmr_14_matched_allframes_swin_original/metrics.csv",
           {"model": "patient_level__per_patient"}, CI),
    Anchor("Ladder (ConvNeXt all frames): overlap + controls",
           [0.788, 0.666, 0.898, 0.807, 0.871],
           LADC, {"cohort": "overlap"}, RUNG + ["control_auroc_min", "control_auroc_max"]),
    Anchor("Ladder (ConvNeXt all frames): null control", [0.916, 0.869, 0.955, 0.902, 0.934],
           LADC, {"cohort": "propensity_clinical"},
           RUNG + ["control_auroc_min", "control_auroc_max"]),

    # --- Conditional caliper -----------------------------------------------------------
    Anchor("Conditional: date control, unrestricted", [0.981, 12878],
           COND, {"score": "confounder_alone_control", "caliper": "none"},
           ["c_statistic", "n_pairs"]),
    Anchor("Conditional: date control, +-90 d (must fall to chance)", [0.379, 256],
           COND, {"score": "confounder_alone_control", "caliper": "90.0"},
           ["c_statistic", "n_pairs"]),
    Anchor("Conditional: date control, +-7 d", [0.578, 45],
           COND, {"score": "confounder_alone_control", "caliper": "7.0"},
           ["c_statistic", "n_pairs"]),
    Anchor("Conditional: Swin all frames, +-180 d", [0.779, 390],
           COND, {"score": SWIN_AF, "caliper": "180.0"}, ["c_statistic", "n_pairs"]),
    Anchor("Conditional: Swin all frames, +-90 d", [0.746, 256],
           COND, {"score": SWIN_AF, "caliper": "90.0"}, ["c_statistic", "n_pairs"]),
    Anchor("Conditional: Swin all frames, +-30 d", [0.860, 93],
           COND, {"score": SWIN_AF, "caliper": "30.0"}, ["c_statistic", "n_pairs"]),
    Anchor("Conditional: Swin all frames, +-7 d", [0.911, 45],
           COND, {"score": SWIN_AF, "caliper": "7.0"}, ["c_statistic", "n_pairs"]),
    Anchor("Conditional: Swin all frames, same session (the 8 pairs)", [0.875, 8],
           COND, {"score": SWIN_AF, "caliper": "same_visit_date"},
           ["c_statistic", "n_pairs"]),
    Anchor("Conditional: ConvNeXt all frames, same session", [0.375, 8],
           COND, {"score": CNX_AF, "caliper": "same_visit_date"}, ["c_statistic", "n_pairs"]),

    # --- Multiplicity guard -------------------------------------------------------
    Anchor("Holm: Swin all frames +-180 d", [0.779, 0.023],
           MULT, {"arm": SWIN_AF, "caliper": "180d"}, ["c_statistic", "p_holm"]),
    Anchor("Holm: Swin all frames +-90 d (does NOT survive)", [0.746, 0.194],
           MULT, {"arm": SWIN_AF, "caliper": "90d"}, ["c_statistic", "p_holm"]),
    Anchor("Holm: Swin all frames +-30 d", [0.860, 0.023],
           MULT, {"arm": SWIN_AF, "caliper": "30d"}, ["c_statistic", "p_holm"]),
    Anchor("Holm: Swin all frames +-7 d", [0.911, 0.008],
           MULT, {"arm": SWIN_AF, "caliper": "7d"}, ["c_statistic", "p_holm"]),

    # --- Split protocol x unit of analysis (supplement) --------------------------------
    Anchor("Split ladder: patient split, per patient", [0.922, 0.882, 0.955],
           LEAK, {"model": "patient_level__per_patient"}, CI),
    Anchor("Split ladder: patient split, per frame", [0.881, 0.873, 0.890],
           LEAK, {"model": "patient_level__per_frame"}, CI),
    Anchor("Split ladder: image split, per patient", [0.968, 0.945, 0.985],
           LEAK, {"model": "image_level__per_patient"}, CI),
    Anchor("Split ladder: image split, per frame", [0.928, 0.922, 0.935],
           LEAK, {"model": "image_level__per_frame"}, CI),

    # --- Attribution of the metadata channel (supplement) ------------------------------
    Anchor("Attribution: pre-diagnostic fields", [0.668, 0.595, 0.737],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "pre_diagnostic", "metric": "roc_auc"}, CI),
    Anchor("Attribution: care-pathway fields", [0.811, 0.753, 0.866],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "post_diagnostic", "metric": "roc_auc"}, CI),
    Anchor("Attribution: biopsy alone", [0.714, 0.639, 0.785],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "biopsy_only", "metric": "roc_auc"}, CI),
    Anchor("Attribution: field presence only", [0.555, 0.478, 0.631],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "missingness_only", "metric": "roc_auc"}, CI),
    Anchor("Attribution: age pair reconstructs the calendar", [0.946, 0.910, 0.978],
           "results/experiment_dmr_04_metadata_leakage/bootstrap_ci.csv",
           {"model": "age_pair_only", "metric": "roc_auc"}, CI),
    Anchor("Attribution: text minus care-pathway clause", [0.908, 0.864, 0.944],
           "results/experiment_dmr_05_text_probe/bootstrap_ci.csv",
           {"model": "minus_all_post_diagnostic", "metric": "roc_auc"}, CI),
    Anchor("Attribution: care-pathway clause only", [0.855, 0.802, 0.900],
           "results/experiment_dmr_05_text_probe/bootstrap_ci.csv",
           {"model": "only_medical_history", "metric": "roc_auc"}, CI),
    Anchor("Attribution: pre-diagnostic clauses", [0.899, 0.854, 0.938],
           "results/experiment_dmr_05_text_probe/bootstrap_ci.csv",
           {"model": "pre_diagnostic_clauses", "metric": "roc_auc"}, CI),

    # --- Where the era signature lives ---------------------------------------------
    Anchor("Signature source: background only", [0.938, 0.892, 0.976],
           SIG, {"representation": "background", "metric": "roc_auc"}, CI),
    Anchor("Signature source: whole frame", [0.924, 0.875, 0.966],
           SIG, {"representation": "global", "metric": "roc_auc"}, CI),
    Anchor("Signature source: temperature histogram", [0.899, 0.841, 0.951],
           SIG, {"representation": "histogram", "metric": "roc_auc"}, CI),
    Anchor("Signature source: breast region only", [0.806, 0.729, 0.875],
           SIG, {"representation": "roi", "metric": "roc_auc"}, CI),

    # --- Era legibility under fine-tuning ------------------------------------------
    Anchor("Era probe: ConvNeXt-T frozen representation", [0.953],
           "results/confounders/finetuned_era_probe_convnext.csv",
           {"run": "experiment_dmr_15_features_convnext_frozen"}, ["era_auroc_benign"]),
    Anchor("Era probe: ConvNeXt-T fine-tuned representation", [0.945],
           "results/confounders/finetuned_era_probe_convnext.csv",
           {"run": "experiment_dmr_15_features_convnext_finetune"}, ["era_auroc_benign"]),
    Anchor("Era probe: ConvNeXt-T paired difference", [-0.008, -0.027, 0.007],
           "results/confounders/finetuned_era_probe_convnext_paired.csv",
           {"run": "experiment_dmr_15_features_convnext_finetune"},
           ["delta", "delta_lo", "delta_hi"]),
    Anchor("Era probe: Swin-T frozen representation", [0.937],
           "results/confounders/finetuned_era_probe_swin.csv",
           {"run": "experiment_dmr_15_features_swin_frozen"}, ["era_auroc_benign"]),
    Anchor("Era probe: Swin-T fine-tuned representation", [0.940],
           "results/confounders/finetuned_era_probe_swin.csv",
           {"run": "experiment_dmr_15_features_swin_finetune"}, ["era_auroc_benign"]),
    Anchor("Era probe: Swin-T paired difference", [0.003, -0.023, 0.031],
           "results/confounders/finetuned_era_probe_swin_paired.csv",
           {"run": "experiment_dmr_15_features_swin_finetune"},
           ["delta", "delta_lo", "delta_hi"]),
    Anchor("Feature-dump rerun reproduces the frozen ConvNeXt-T arm", [0.815, 0.760, 0.867],
           "results/experiment_dmr_15_features_convnext_frozen/bootstrap_ci.csv",
           {"model": "convnext_tiny_linear_probe", "metric": "roc_auc"}, CI),
    Anchor("Feature-dump rerun reproduces the frozen Swin-T arm", [0.874, 0.822, 0.919],
           "results/experiment_dmr_15_features_swin_frozen/bootstrap_ci.csv",
           {"model": "swin_tiny_patch4_window7_224_linear_probe", "metric": "roc_auc"}, CI),

    # --- Age legibility on the second cohort --------------------------------------------
    Anchor("Cohort 2: age band decoded from the pixels, within benign", [0.476, 0.344, 0.608],
           "results/experiment_ver_12_age_within_benign_convnext/bootstrap_ci.csv",
           {"model": "convnext_tiny_linear_probe", "metric": "roc_auc"}, CI),
    Anchor("Cohort 2: age band decoded from the pixels, within malignant",
           [0.602, 0.394, 0.804],
           "results/experiment_ver_12_age_within_malignant_convnext/bootstrap_ci.csv",
           {"model": "convnext_tiny_linear_probe", "metric": "roc_auc"}, CI),

    # --- Age scored directly, which is what answers the direction question -------------
    Anchor("Age raw on DMR-IR (at chance, not inverted)", [0.495, 0.415, 0.576],
           "results/confounders/age_direction.csv",
           {"cohort": "dmr_ir", "statistic": "age_raw_auroc"}, CI),
    Anchor("Age raw on DMR-IR: mean ages by diagnosis", [54.8, 54.5],
           "results/confounders/age_direction.csv",
           {"cohort": "dmr_ir", "statistic": "age_raw_auroc"},
           ["mean_age_benign", "mean_age_malignant"]),
    Anchor("Age raw on cohort 2 (reproduces the quoted 0.769)", [0.769, 0.676, 0.850],
           "results/confounders/age_direction.csv",
           {"cohort": "verification", "statistic": "age_raw_auroc"}, CI),
    Anchor("Age raw on cohort 2: mean ages by diagnosis", [45.1, 57.4],
           "results/confounders/age_direction.csv",
           {"cohort": "verification", "statistic": "age_raw_auroc"},
           ["mean_age_benign", "mean_age_malignant"]),

    # --- Second cohort ------------------------------------------------------------------
    Anchor("Cohort 2: age alone (fitted, patient-level CV)", [0.759, 0.665, 0.844],
           "results/experiment_08_multimodal/bootstrap_ci.csv",
           {"model": "age_only_reference", "metric": "roc_auc"}, CI),
    Anchor("Cohort 2: thermogram, fine-tuned (channels sentence)", [0.654, 0.543, 0.756],
           "results/experiment_02_single_view_disc_lr/bootstrap_ci.csv",
           {"model": "convnext_tiny_finetune_discriminative_lr", "metric": "roc_auc"}, CI,
           note="second run of the same configuration as ver_11 (0.658); both are quoted"),
    Anchor("Cohort 2: conditional, ConvNeXt-T fine-tuned unrestricted", [0.658, 2940],
           VCOND, {"score": "convnext_tiny_finetune_discriminative_lr", "caliper": "none"},
           ["c_statistic", "n_pairs"]),
    Anchor("Cohort 2: conditional, Swin-T fine-tuned unrestricted", [0.669, 2940],
           VCOND, {"score": "swin_tiny_patch4_window7_224_finetune_discriminative_lr",
                   "caliper": "none"}, ["c_statistic", "n_pairs"]),
    Anchor("Cohort 2: conditional, ConvNeXt-T at the five-year caliper", [0.552, 678],
           VCOND, {"score": "convnext_tiny_finetune_discriminative_lr", "caliper": "5.0"},
           ["c_statistic", "n_pairs"]),
    Anchor("Cohort 2: conditional, Swin-T at the five-year caliper", [0.524, 678],
           VCOND, {"score": "swin_tiny_patch4_window7_224_finetune_discriminative_lr",
                   "caliper": "5.0"}, ["c_statistic", "n_pairs"]),
    Anchor("Cohort 2: conditional control (age) at five years", [0.544, 678],
           VCOND, {"score": "confounder_alone_control", "caliper": "5.0"},
           ["c_statistic", "n_pairs"]),
    Anchor("Cohort 2: image vs age, paired", [-0.140, -0.262, -0.018],
           "results/final_statistics.csv",
           {"model_a": "image_only_reference", "model_b": "age_only_reference"},
           ["delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Cohort 2: age + image (fixed) vs age", [-0.028, -0.096, 0.037],
           "results/final_statistics.csv",
           {"model_a": "age_plus_image_fixed", "model_b": "age_only_reference"},
           ["delta_auroc", "delta_ci_lower", "delta_ci_upper"]),
    Anchor("Cohort 2 ladder: original reference (ladder recipe)", [0.582, 0.473, 0.691],
           VLAD, {"cohort": "original"}, RUNG),
    Anchor("Cohort 2 ladder: age raw statistic on the full cohort", [0.769],
           VLAD, {"cohort": "original"}, ["confounder_auroc"]),
    Anchor("Cohort 2 ladder: five-year age matching", [0.501, 0.353, 0.657, 0.503],
           VLAD, {"cohort": "age_matched_5"}, RUNG + ["confounder_auroc"]),
    Anchor("Cohort 2 ladder: propensity rung, age uncontrolled", [0.581, 0.429, 0.723, 0.780],
           VLAD, {"cohort": "propensity_clinical"}, RUNG + ["confounder_auroc"]),

    # --- Cohort 2, exploitation of age (phase 2) ----------------------------------------
    Anchor("Cohort 2 exploitation: Swin-T fine-tuned", [0.669, 0.630, 0.504, 0.760, 0.048],
           VEXP, {"arm": "Swin-T anterior, fine-tuned"},
           ["diagnostic_auroc", "age_auroc_benign", "age_auroc_benign_lo",
            "age_auroc_benign_hi", "age_auroc_benign_perm_p"]),
    Anchor("Cohort 2 exploitation: ConvNeXt-T fine-tuned (encoder run)",
           [0.658, 0.642, 0.513, 0.765, 0.030],
           VEXP, {"arm": "ConvNeXt-T anterior, fine-tuned (encoder run)"},
           ["diagnostic_auroc", "age_auroc_benign", "age_auroc_benign_lo",
            "age_auroc_benign_hi", "age_auroc_benign_perm_p"]),
    Anchor("Cohort 2 exploitation: ConvNeXt-T fine-tuned",
           [0.654, 0.641, 0.510, 0.764, 0.030],
           VEXP, {"arm": "ConvNeXt-T anterior, fine-tuned"},
           ["diagnostic_auroc", "age_auroc_benign", "age_auroc_benign_lo",
            "age_auroc_benign_hi", "age_auroc_benign_perm_p"]),
    Anchor("Cohort 2 exploitation: ResNet-50 (quarantined)",
           [0.581, 0.589, 0.455, 0.720, 0.18],
           VEXP, {"arm": "ResNet-50 anterior, fine-tuned"},
           ["diagnostic_auroc", "age_auroc_benign", "age_auroc_benign_lo",
            "age_auroc_benign_hi", "age_auroc_benign_perm_p"]),
    Anchor("Cohort 2 exploitation: frozen probe sits at chance",
           [0.535, 0.501, 0.370, 0.630, 0.99],
           VEXP, {"arm": "ConvNeXt-T anterior, frozen"},
           ["diagnostic_auroc", "age_auroc_benign", "age_auroc_benign_lo",
            "age_auroc_benign_hi", "age_auroc_benign_perm_p"]),
    Anchor("Cohort 2 exploitation: Spearman against untransformed age (ConvNeXt-T)",
           [0.29], VEXP, {"arm": "ConvNeXt-T anterior, fine-tuned (encoder run)"},
           ["spearman_benign"]),
    Anchor("Cohort 2 exploitation: Spearman against untransformed age (Swin-T)",
           [0.35], VEXP, {"arm": "Swin-T anterior, fine-tuned"}, ["spearman_benign"]),

    # --- Matching diagnostics ------------------------------------------------------
    Anchor("SMD: DMR-IR original, acquisition axis", [2.46],
           SMD, {"dataset": "dmr", "cohort": "original", "covariate": "acquisition_days"},
           ["smd"]),
    Anchor("SMD: DMR-IR overlap rung weakens but does not remove it", [1.20],
           SMD, {"dataset": "dmr", "cohort": "overlap", "covariate": "acquisition_days"},
           ["smd"]),
    Anchor("SMD: DMR-IR null control leaves the acquisition axis alone", [2.39],
           SMD, {"dataset": "dmr", "cohort": "propensity_clinical",
                 "covariate": "acquisition_days"}, ["smd"]),
    Anchor("SMD: cohort 2 original, age", [1.05],
           SMD, {"dataset": "verification", "cohort": "original", "covariate": "age"},
           ["smd"]),
    Anchor("SMD: cohort 2 five-year rung balances age", [0.04],
           SMD, {"dataset": "verification", "cohort": "age_matched_5", "covariate": "age"},
           ["smd"]),
    Anchor("SMD: cohort 2 null control leaves age alone", [1.00],
           SMD, {"dataset": "verification", "cohort": "propensity_clinical",
                 "covariate": "age"}, ["smd"]),

    # --- Cross-label acquisition support (the identifiability section) ------------------
    Anchor("Support: cross-label pairs sharing a date group, observed and under the null",
           [8, 155, 136, 171], SUPP, {"statistic": "cross_label_date_group_pairs"},
           ["observed", "null_median", "null_ci_lower", "null_ci_upper"]),
    Anchor("Support: date groups holding both classes, observed and under the null",
           [3, 44, 38, 50], SUPP, {"statistic": "mixed_date_groups"},
           ["observed", "null_median", "null_ci_lower", "null_ci_upper"]),
    Anchor("Support: benign/malignant pairs the cohort forms", [13440],
           SUPP, {"statistic": "cross_label_pairs_in_cohort"}, ["observed"]),
    Anchor("Support: patients in the three label-mixed date groups", [10],
           SUPP, {"statistic": "patients_in_mixed_date_groups"}, ["observed"]),

    # --- Cross-cohort transfer (supplement, not load-bearing) ---------------------------
    Anchor("Transfer, per-image standardisation: in-domain reference", [0.798, 0.728, 0.860],
           "results/experiment_transfer_dmr_to_cali_perimage/metrics.csv",
           {"model": "source_internal"}, CI,
           note="single source fold under the transfer pipeline, not the 5x5 CV arm"),
    Anchor("Transfer, per-image standardisation: transferred", [0.519, 0.410, 0.624],
           "results/experiment_transfer_dmr_to_cali_perimage/metrics.csv",
           {"model": "transfer"}, CI),
    Anchor("Transfer, fixed temperature scale: in-domain reference", [0.775, 0.696, 0.846],
           "results/experiment_transfer_dmr_to_cali_fixed/metrics.csv",
           {"model": "source_internal"}, CI),
    Anchor("Transfer, fixed temperature scale: transferred", [0.457, 0.351, 0.564],
           "results/experiment_transfer_dmr_to_cali_fixed/metrics.csv",
           {"model": "transfer"}, CI),

    # --- Model selection on a small validation partition (supplement) --------------------
    # The four diagnostics on which the epoch-selecting run was quarantined. They were
    # computed by hand when the run was discarded; scripts/43_selection_diagnostics.py now
    # recomputes them from that run's fold_log.csv and predictions.csv, which is what these
    # anchors check.
    Anchor("Selection: validation vs test correlation", [0.06],
           SEL, {"statistic": "corr_selected_val_vs_test"}, ["value"]),
    Anchor("Selection: mean optimism", [0.14],
           SEL, {"statistic": "mean_optimism"}, ["value"]),
    Anchor("Selection: median selected epoch", [4],
           SEL, {"statistic": "median_selected_epoch"}, ["value"]),
    Anchor("Selection: folds emitting a near-constant probability", [14],
           SEL, {"statistic": "collapsed_folds"}, ["value"]),
    Anchor("Selection: validation partition size", [15],
           SEL, {"statistic": "n_validation_patients"}, ["value"]),
]


# Counts the manuscript states about the cohorts themselves. They live in the manifests,
# which the sweep deliberately does not index (per-image files would match any literal by
# coincidence), so they are checked here as row counts instead.
def _rows(df: pd.DataFrame) -> int:
    return len(df)


def _label(value: int):
    """How many patients carry one binary diagnosis label."""
    return lambda df: int((df["patient_binary_label"] == value).sum())


def _median_frames(protocol: str):
    """Median frames per patient under one acquisition protocol.

    The manuscript describes what an all-frames arm reads, and that description is a
    property of the image manifest rather than of any run, so it is checked here rather
    than anchored to a result CSV.
    """
    def f(df: pd.DataFrame) -> int:
        sub = df[df["protocol"] == protocol]
        return int(sub.groupby("patient_id").size().median())
    return f


def _label_pure_date_groups(df: pd.DataFrame) -> int:
    return int((df.groupby("visit_date")["patient_binary_label"].nunique() == 1).sum())


def _date_groups(df: pd.DataFrame) -> int:
    return int(df["visit_date"].nunique())


DMRP, DMRI = "results/dmr_patient_manifest.csv", "results/dmr_image_manifest.csv"
C2P = "results/patient_manifest.csv"

# (claim, value printed in the manuscript, file, how to recompute it)
COUNT_ANCHORS: list[tuple[str, int, str, object]] = [
    ("DMR-IR thermograms", 6051, DMRI, _rows),
    ("DMR-IR patients after exclusions", 236, DMRP, _rows),
    ("DMR-IR benign patients", 140, DMRP, _label(0)),
    ("DMR-IR malignant patients", 96, DMRP, _label(1)),
    ("DMR-IR acquisition-date groups", 86, DMRP, _date_groups),
    ("DMR-IR label-pure date groups", 83, DMRP, _label_pure_date_groups),
    ("DMR-IR median static frames per patient", 5, DMRI, _median_frames("Static")),
    ("DMR-IR median dynamic frames per patient", 22, DMRI, _median_frames("Dinamic")),
    ("Cohort 2 thermograms", 357, "results/image_manifest.csv", _rows),
    ("Cohort 2 patients", 119, C2P, _rows),
    ("Cohort 2 benign patients", 84, C2P, _label(0)),
    ("Cohort 2 malignant patients", 35, C2P, _label(1)),
]


def check_counts() -> list[tuple[str, str, int, int]]:
    out = []
    for claim, printed, path, how in COUNT_ANCHORS:
        p = Path(path)
        if not p.exists():
            # The manifests hold one row per patient and are not part of the public
            # release; a count that cannot be recomputed is reported as such, not failed.
            out.append((claim, "NOT DISTRIBUTED", printed, -1))
            continue
        got = int(how(pd.read_csv(p)))
        out.append((claim, "OK" if got == printed else "MISMATCH", printed, got))
    return out


# ---------------------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------------------
NUMBER = re.compile(r"(?<![\w.\\])(-?\d+(?:\.\d+)?)(?![\w.])")
# Lines that carry typesetting magnitudes rather than measurements.
LAYOUT = re.compile(r"includegraphics|renewcommand|setcounter|definecolor|usepackage|"
                    r"geometry|documentclass|linewidth|textwidth|graphicspath|newcommand|"
                    r"hyperref|def\\wacv|vspace|hspace")


def strip_comment(line: str) -> str:
    out, prev = [], ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def harvest(path: Path) -> list[dict]:
    found, in_body = [], False
    for i, raw in enumerate(path.read_text().splitlines(), start=1):
        if "\\begin{document}" in raw:
            in_body = True
            continue
        if not in_body:
            continue
        line = strip_comment(raw)
        if LAYOUT.search(line):
            continue
        for m in NUMBER.finditer(line):
            text = m.group(1)
            # `0.813--0.823` is a range, not a negative number: a leading minus counts as a
            # sign only when what precedes it is neither another dash nor a digit.
            if text.startswith("-") and m.start() > 0 and line[m.start() - 1] in "-0123456789":
                text = text[1:]
            # `10^{-6}` is an exponent, not a measurement.
            if line[max(0, m.start() - 2):m.start()].endswith(("^", "^{")):
                continue
            value = float(text)
            decimals = len(text.split(".")[1]) if "." in text else 0
            found.append({"file": path.name, "line": i, "text": text, "value": value,
                          "decimals": decimals, "context": line.strip()[:110]})
    return found


def build_index() -> dict:
    """value (rounded to 3 dp) -> list of "path::column" that contain it."""
    index: dict[float, list[str]] = {}
    files = [p for p in glob.glob("results/**/*.csv", recursive=True)
             if not any(s in os.path.basename(p) for s in SKIP_BASENAMES)]
    for p in sorted(files):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            for v in s.round(3).unique():
                index.setdefault(float(v), []).append(f"{p}::{col}")
    return index


def lookup(index: dict, value: float) -> list[str]:
    hits: list[str] = []
    for delta in (-0.001, 0.0, 0.001):
        hits += index.get(round(value + delta, 3), [])
    return sorted(set(h for h in hits
                      if any(abs(value - u) <= TOL + 1e-9
                             for u in [round(value, 3), round(value + 0.001, 3),
                                       round(value - 0.001, 3)])))[:4] if hits else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any anchor fails")
    args = ap.parse_args()

    for a in ANCHORS:
        a.check()
    failed = [a for a in ANCHORS if a.status != "OK"]
    counted = check_counts()
    skipped_counts = [c for c in counted if c[1] == "NOT DISTRIBUTED"]
    failed_counts = [c for c in counted if c[1] not in ("OK", "NOT DISTRIBUTED")]

    index = build_index()
    measurements, counts, unmatched_m, unmatched_c = [], [], [], []
    for tex in TEX_FILES:
        if not tex.exists():
            print(f"missing {tex}", file=sys.stderr)
            continue
        for item in harvest(tex):
            hits = lookup(index, item["value"])
            item["hits"] = hits
            if item["decimals"] >= 3:
                measurements.append(item)
                if not hits:
                    unmatched_m.append(item)
            elif item["decimals"] == 0 and abs(item["value"]) >= 2:
                counts.append(item)
                if not hits:
                    unmatched_c.append(item)

    lines = ["# Paper consistency sweep", "",
             "Generated by `python scripts/36_consistency_sweep.py`. Anchors are semantic "
             "(claim -> file/row/column); the sweep is syntactic (does this literal exist "
             "anywhere in `results/`?). Tolerance 5e-4.", "",
             f"- anchors checked: **{len(ANCHORS)}**, failing: **{len(failed)}**",
             f"- three-decimal measurements in the .tex: **{len(measurements)}**, "
             f"unmatched: **{len(unmatched_m)}**",
             f"- integer counts in the .tex: **{len(counts)}**, unmatched: "
             f"**{len(unmatched_c)}**", "",
             "## Anchors", "", "| Status | Claim | Paper | Results | Note |",
             "|---|---|---|---|---|"]
    for a in ANCHORS:
        printed = ", ".join(f"{v:.3f}" for v in a.printed)
        got = ", ".join(f"{v:.3f}" for v in a.got) if a.got else "--"
        lines.append(f"| {a.status} | {a.claim} | {printed} | {got} | {a.note} |")

    lines += ["", "## Cohort counts (checked as manifest row counts)", "",
              "| Status | Claim | Paper | Results |", "|---|---|---|---|"]
    for claim, status, printed, got in counted:
        lines.append(f"| {status} | {claim} | {printed} | {got} |")

    lines += ["", "## Unmatched measurements (found in no results CSV)", ""]
    if unmatched_m:
        lines += ["| File | Line | Value | Context |", "|---|---|---|---|"]
        for it in unmatched_m:
            ctx = it["context"].replace("|", "\\|")
            lines.append(f"| {it['file']} | {it['line']} | {it['text']} | `{ctx}` |")
    else:
        lines.append("None: every three-decimal literal in the manuscript exists in `results/`.")

    lines += ["", "## Unmatched counts", "",
              "Cohort sizes and frame counts live in the manifests, which the sweep does "
              "not index; they are checked in the section above instead, so their presence "
              "here is expected.", ""]
    if unmatched_c:
        lines += ["| File | Line | Value | Context |", "|---|---|---|---|"]
        for it in unmatched_c:
            ctx = it["context"].replace("|", "\\|")
            lines.append(f"| {it['file']} | {it['line']} | {it['text']} | `{ctx}` |")
    else:
        lines.append("None.")

    REPORT.write_text("\n".join(lines) + "\n")

    print(f"anchors: {len(ANCHORS) - len(failed)}/{len(ANCHORS)} OK")
    for a in failed:
        print(f"  {a.status}: {a.claim} -- {a.note}")
    checked = len(counted) - len(skipped_counts)
    print(f"cohort counts: {checked - len(failed_counts)}/{checked} OK"
          + (f" ({len(skipped_counts)} need the manifests, which are rebuilt from the "
             "datasets and not distributed)" if skipped_counts else ""))
    for claim, status, printed, got in failed_counts:
        print(f"  {status}: {claim} -- paper {printed} vs results {got}")
    print(f"measurements: {len(measurements) - len(unmatched_m)}/{len(measurements)} traced")
    for it in unmatched_m:
        print(f"  unmatched {it['file']}:{it['line']}  {it['text']}  {it['context'][:80]}")
    print(f"counts: {len(counts) - len(unmatched_c)}/{len(counts)} traced")
    print(f"Wrote {REPORT}")
    return 1 if (args.strict and (failed or failed_counts)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
