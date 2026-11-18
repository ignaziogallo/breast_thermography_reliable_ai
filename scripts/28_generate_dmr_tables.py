#!/usr/bin/env python3
"""Print the DMR-IR LaTeX tables from completed experiments.

The tables compare *channels* -- calendar, tabular record, clinical text, thermogram --
rather than model variants, and the quantity reported is an AUROC with a patient-level
bootstrap interval rather than a mean and standard deviation.

Nothing is written into `paper/`. The tables are printed to stdout, so a number cannot
reach the manuscript without existing under `results/` first.

Where an arm has no `metrics.csv` row -- the patient-level arm of the leakage experiment was
produced by a run that scored only the other arm -- the interval is recomputed here from
`predictions.csv` with the same bootstrap the experiments use, rather than copied from a
log.

Usage:
    python scripts/28_generate_dmr_tables.py            # list which tables are available
    python scripts/28_generate_dmr_tables.py --table channels
    python scripts/28_generate_dmr_tables.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap import patient_bootstrap
from src.evaluation.predictions import pool_repeats
from src.evaluation.quality_gate import QUARANTINE_SUFFIX
from src.evaluation.summary import stored_auroc_interval as ci_from_file

R = Path("results")
LEAK = R / "experiment_dmr_01_leakage"
FRONTAL = R / "experiment_dmr_02_frontal_static"
EPOCH_B = R / "experiment_dmr_03_epoch_within_benign"
EPOCH_M = R / "experiment_dmr_03_epoch_within_malignant"
META = R / "experiment_dmr_04_metadata_leakage"
TEXT = R / "experiment_dmr_05_text_probe"
AGE = R / "experiment_dmr_07_metadata"
INC_FRONTAL = R / "experiment_dmr_08_epoch_frontal"
INC_ALL = R / "experiment_dmr_08_epoch_allframes"


def ci_from_predictions(path: Path, model: str | None = None, n_bootstrap: int = 2000,
                        seed: int = 42) -> tuple[float, float, float] | None:
    """Recompute the interval from stored predictions, pooling repeats per patient."""
    if not path.exists():
        return None
    preds = pd.read_csv(path, dtype={"patient_id": str})
    if model is not None:
        preds = preds[preds["model"] == model]
    if preds.empty:
        return None
    pooled = pool_repeats(preds).reset_index()
    ci, _ = patient_bootstrap(pooled["y_true"], pooled["y_prob"], pooled["patient_id"],
                              n_bootstrap=n_bootstrap, seed=seed)
    row = ci[ci["metric"] == "roc_auc"].iloc[0]
    return float(row["estimate"]), float(row["ci_lower"]), float(row["ci_upper"])


def fmt(v: tuple[float, float, float] | None) -> str:
    """An estimate with its interval as a table cell; a dash for an arm that was not run."""
    return "--" if v is None else f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def table(rows: list[list[str]], columns: list[str], label: str, caption: str,
          align: str | None = None) -> str:
    """A booktabs table; a row equal to `["MIDRULE"]` becomes a rule."""
    spec = align or ("l" + "c" * (len(columns) - 1))
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(columns) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append("\\midrule" if row == ["MIDRULE"]
                     else " & ".join(str(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def metrics_cell(metrics: pd.DataFrame, model: str) -> str:
    """One model's interval from a `metrics.csv` that carries it merged into the summary."""
    if metrics.empty or model not in set(metrics["model"]):
        return "--"
    r = metrics[metrics["model"] == model].iloc[0]
    return f"{r['estimate']:.3f} [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"


def build_cohort() -> str | None:
    """The DMR-IR cohort as actually constituted, counted from the manifests."""
    pm, im = R / "dmr_patient_manifest.csv", R / "dmr_image_manifest.csv"
    if not (pm.exists() and im.exists()):
        return None
    pat = pd.read_csv(pm, dtype={"patient_id": str})
    img = pd.read_csv(im, dtype={"patient_id": str})
    img["view_code"] = pd.to_numeric(img["view_code"], errors="coerce")
    lab = pat["patient_binary_label"].astype(int)

    static = img[img["protocol"] == "Static"]
    n_views = static.groupby("patient_id")["view_code"].nunique().reindex(pat["patient_id"],
                                                                         fill_value=0)
    dyn = img[(img["protocol"] == "Dinamic") & (img["view_code"] == 0)]
    per = dyn.groupby("patient_id").size()
    full5, dyn20 = set(n_views[n_views == 5].index), set(per[per >= 20].index)

    sessions = pat.groupby("visit_date")["patient_binary_label"].agg(["size", "mean"])
    pure = (sessions["mean"] == 0) | (sessions["mean"] == 1)
    hashes = img.groupby("sha256")["patient_id"].agg(["size", "nunique"])
    cross = hashes[(hashes["size"] > 1) & (hashes["nunique"] > 1)]
    masks = img["mask_path"].notna() & img["mask_path"].astype(str).str.len().gt(0)
    years = pd.to_datetime(pat["visit_date"], errors="coerce").dt.year

    rows = [
        ["Patients", f"{len(pat)}"],
        ["Benign / malignant", f"{int((lab == 0).sum())} / {int((lab == 1).sum())}"],
        ["Thermograms", f"{len(img)}"],
        ["Static frames / dynamic frames",
         f"{int((img['protocol'] == 'Static').sum())} / "
         f"{int((img['protocol'] == 'Dinamic').sum())}"],
        ["Patients with all five static views",
         f"{len(full5)} ({int(lab[pat['patient_id'].isin(full5)].eq(0).sum())} / "
         f"{int(lab[pat['patient_id'].isin(full5)].eq(1).sum())})"],
        ["Patients with a 20-frame frontal dynamic sequence", f"{len(dyn20)}"],
        ["Patients with both", f"{len(full5 & dyn20)}"],
        ["MIDRULE"],
        ["Distinct acquisition dates", f"{len(sessions)}"],
        ["Label-pure acquisition sessions",
         f"{int(pure.sum())} / {len(sessions)}, covering "
         f"{int(sessions.loc[pure, 'size'].sum())} patients"],
        ["Acquisition years, benign",
         f"{int(years[lab == 0].min())}--{int(years[lab == 0].max())}"],
        ["Acquisition years, malignant",
         f"{int(years[lab == 1].min())}--{int(years[lab == 1].max())}"],
        ["MIDRULE"],
        ["Cross-patient byte-identical frames", f"{len(cross)}"],
        ["Frames carrying a segmentation mask", f"{int(masks.sum())} / {len(img)}"],
    ]
    return table(rows, ["Characteristic", "Value"], "tab:dmr_cohort",
                 "The DMR-IR cohort after excluding healthy volunteers and collapsing patients "
                 "entered twice, counted from the manifests by "
                 "\\texttt{scripts/21\\_audit\\_dmr\\_design.py}. A session is one acquisition "
                 "date; a label-pure session contains patients of a single class.")


def build_verification() -> str | None:
    """The verification cohort, where the dominant confounder is age rather than the calendar."""
    spec = [
        ("Age alone", ci_from_file(R / "experiment_07_metadata" / "bootstrap_ci.csv",
                                   "logistic_age_only")),
        ("Bilateral asymmetry descriptors",
         ci_from_file(R / "experiment_07b_asymmetry" / "bootstrap_ci.csv", "asymmetry")),
        ["MIDRULE"],
        ("Thermogram, linear probe",
         ci_from_file(R / "experiment_02_single_view_linear_probe" / "bootstrap_ci.csv",
                      "convnext_tiny_linear_probe")),
        ("Thermogram, fine-tuned",
         ci_from_file(R / "experiment_02_single_view_disc_lr" / "bootstrap_ci.csv",
                      "convnext_tiny_finetune_discriminative_lr")),
        ["MIDRULE"],
        ("Age + thermogram (fixed combination)",
         ci_from_file(R / "experiment_08_multimodal" / "bootstrap_ci.csv",
                      "age_plus_image_fixed")),
    ]
    rows = [r if r == ["MIDRULE"] else [r[0], fmt(r[1])] for r in spec]
    if all(r == ["MIDRULE"] or r[1] == "--" for r in rows):
        return None
    return table(rows, ["Channel", "AUROC [95\\% CI]"], "tab:verification",
                 "The verification cohort under the same patient-level protocol. Its dominant "
                 "non-image channel is age rather than the acquisition calendar, and the "
                 "thermogram again fails to exceed it.")


def build_channels() -> str | None:
    """Every channel that predicts malignancy on DMR-IR, image and non-image alike."""
    spec = [
        ("Acquisition date alone",
         ci_from_file(META / "bootstrap_ci.csv", "acquisition_date_only")),
        ("Tabular record, all fields",
         ci_from_file(META / "bootstrap_ci.csv", "as_published_all_fields")),
        ("Age pair (\\texttt{age\\_at\\_visit} + \\texttt{age\\_current})",
         ci_from_file(META / "bootstrap_ci.csv", "age_pair_only")),
        ("Clinical text, TF--IDF", ci_from_file(TEXT / "bootstrap_ci.csv", "all_text")),
        ["MIDRULE"],
        ("Thermogram, image-level split, per patient",
         ci_from_file(LEAK / "metrics.csv", "image_level__per_patient")
         or ci_from_predictions(LEAK / "predictions_image_level_LEAKY.csv")),
        ("Thermogram, all frames, Swin-T",
         ci_from_file(R / "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224/metrics.csv",
                      "patient_level__per_patient")),
        ("Thermogram, all frames, ConvNeXt-T",
         ci_from_predictions(LEAK / "predictions.csv", "patient_level")),
        ("Thermogram, frontal static, fine-tuned",
         ci_from_file(R / "experiment_dmr_11_encoder_convnext_tiny_finetune/bootstrap_ci.csv",
                      "convnext_tiny_finetune_discriminative_lr")),
        ("Thermogram, frontal static, frozen probe",
         ci_from_file(FRONTAL / "bootstrap_ci.csv", "convnext_tiny_linear_probe")),
        ["MIDRULE"],
        ("Age alone", ci_from_file(AGE / "bootstrap_ci.csv", "logistic_age_only")),
    ]
    rows = [r if r == ["MIDRULE"] else [r[0], fmt(r[1])] for r in spec]
    if all(r == ["MIDRULE"] or r[1] == "--" for r in rows):
        return None
    return table(rows, ["Channel", "AUROC [95\\% CI]"], "tab:dmr_channels",
                 "Every information channel available on DMR-IR, evaluated on the same frozen "
                 "patient-level folds. Intervals are patient-level bootstrap percentiles. "
                 "The image-level row is reported for comparability with the published "
                 "protocol and is contaminated by construction; see "
                 "Table~\\ref{tab:dmr_ladder}.")


def build_ladder() -> str | None:
    """Split unit crossed with unit of analysis, plus the contamination each produces."""
    if not (LEAK / "contamination.csv").exists():
        return None
    cont = (pd.read_csv(LEAK / "contamination.csv")
            .groupby("arm")["fraction_contaminated"].mean())
    metrics = (pd.read_csv(LEAK / "metrics.csv") if (LEAK / "metrics.csv").exists()
               else pd.DataFrame())

    rows = [
        ["Patient-level", "per patient", fmt(ci_from_predictions(LEAK / "predictions.csv",
                                                                 "patient_level")),
         f"{cont.get('patient_level', float('nan')):.4f}"],
        ["Patient-level", "per frame", metrics_cell(metrics, "patient_level__per_frame"),
         f"{cont.get('patient_level', float('nan')):.4f}"],
        ["Image-level", "per patient", metrics_cell(metrics, "image_level__per_patient"),
         f"{cont.get('image_level', float('nan')):.4f}"],
        ["Image-level", "per frame", metrics_cell(metrics, "image_level__per_frame"),
         f"{cont.get('image_level', float('nan')):.4f}"],
    ]
    return table(rows, ["Split unit", "Unit of analysis", "AUROC [95\\% CI]", "Contamination"],
                 "tab:dmr_ladder",
                 "The split protocol and the unit of analysis are separate choices. Encoder, "
                 "hyperparameters, epoch budget and fold count are identical across rows; only "
                 "what constitutes a fold changes. Contamination is the mean fraction of test "
                 "frames whose patient also appears in training.")


def build_epoch() -> str | None:
    """Is acquisition era legible from the thermogram, with the diagnosis held constant?"""
    rows = []
    for path, name in ((EPOCH_B, "within_benign"), (EPOCH_M, "within_malignant")):
        v = ci_from_file(path / "bootstrap_ci.csv", f"epoch_decoder_{name}")
        if v is None:
            continue
        cfg = json.loads((path / "config.json").read_text())

        def era(years: list[int]) -> str:
            return str(years[0]) if len(set(years)) == 1 else f"{min(years)}--{max(years)}"

        rows.append([name.replace("_", " ").replace("within ", "Within "),
                     f"{cfg['n_patients']} ({cfg['n_early']}/{cfg['n_late']})",
                     f"{era(cfg['early_years'])} vs {era(cfg['late_years'])}", fmt(v)])
    if not rows:
        return None
    return table(rows, ["Stratum", "$n$ (early/late)", "Eras", "AUROC [95\\% CI]"],
                 "tab:dmr_epoch",
                 "An encoder trained to predict the acquisition era rather than the diagnosis, "
                 "among patients who share a diagnosis. Chance is 0.5. The within-malignant "
                 "contrast is not attributable: no benign patient was acquired after 2015, so "
                 "a protocol change and a change of case mix cannot be separated there.")


def build_attribution() -> str | None:
    """Where the non-image score comes from: tabular partitions and text clause ablations."""
    rows = []
    for model, name in [("pre_diagnostic", "Pre-diagnostic fields"),
                        ("post_diagnostic", "Care-pathway fields"),
                        ("biopsy_only", "\\texttt{biopsy} alone"),
                        ("missingness_only", "Field presence only, no values"),
                        ("as_published_all_fields", "All fields")]:
        v = ci_from_file(META / "bootstrap_ci.csv", model)
        if v is not None:
            rows.append([name, "Tabular", fmt(v)])
    if rows:
        rows.append(["MIDRULE"])
    for model, name in [("all_text", "Full prompt"),
                        ("minus_medical_history", "Minus the care-pathway clause"),
                        ("only_medical_history", "Care-pathway clause alone"),
                        ("pre_diagnostic_clauses", "Pre-diagnostic clauses only")]:
        v = ci_from_file(TEXT / "bootstrap_ci.csv", model)
        if v is not None:
            rows.append([name, "Text", fmt(v)])
    if not rows:
        return None
    return table(rows, ["Subset", "Representation", "AUROC [95\\% CI]"], "tab:dmr_attribution",
                 "Attribution of the metadata-only score. Upper block: the tabular record "
                 "partitioned by whether a field can precede any suspicion of disease. Lower "
                 "block: the distributed natural-language prompt with one clause removed at a "
                 "time. Text arms use TF--IDF bigrams fitted inside each fold, so the absolute "
                 "level is not comparable with a sentence-transformer probe; the comparison "
                 "that carries content is between subsets.")


def build_incremental() -> str | None:
    """Does the thermogram add anything once the calendar is known?"""
    rows = []
    arms = ((INC_FRONTAL, "Frontal static (ConvNeXt-T)", "final_statistics.csv"),
            (INC_ALL, "All frames (ConvNeXt-T)", "final_statistics.csv"),
            (R / "experiment_dmr_13_incremental_swin_allframes", "All frames (Swin-T)",
             "paired.csv"))
    for path, name, filename in arms:
        f = path / filename
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for model, label in [("visit_date_plus_image_fixed", "Date + image (fixed combination)"),
                             ("visit_date_plus_image_stacked_exploratory",
                              "Date + image (stacked, exploratory)"),
                             ("image_only_reference", "Image alone")]:
            r = df[df["model_a"] == model]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([name, label, f"{r['auroc_a']:.3f}",
                         f"{r['delta_auroc']:+.3f} [{r['delta_ci_lower']:+.3f}, "
                         f"{r['delta_ci_upper']:+.3f}]"])
    if not rows:
        return None
    return table(rows, ["Image arm", "Model", "AUROC", "$\\Delta$ vs date alone [95\\% CI]"],
                 "tab:dmr_incremental",
                 "Incremental value of the thermogram given the acquisition date, paired on "
                 "identical folds. The fixed combination fits nothing to labels and is the "
                 "leakage-free estimate; the stacked variant is optimistic because its "
                 "meta-model sees image scores that depend on the test fold. An interval "
                 "excluding zero from above would be required to demonstrate improvement.")


def build_transfer() -> str | None:
    """Does a DMR-IR model discriminate on an independent cohort?"""
    rows = []
    for suffix, name in (("perimage", "Per-image standardisation"),
                         ("fixed", "Fixed temperature scale")):
        path = R / f"experiment_transfer_dmr_to_cali_{suffix}"
        metrics = path / "metrics.csv"
        if not metrics.exists():
            continue
        df = pd.read_csv(metrics)
        rows.append([name, metrics_cell(df, "source_internal"), metrics_cell(df, "transfer")])
    if not rows:
        return None
    return table(rows,
                 ["Normalisation", "Held-out DMR-IR patients", "Transfer to the primary cohort"],
                 "tab:dmr_transfer",
                 "Zero-shot transfer of a DMR-IR model to the primary cohort, which was "
                 "acquired in another country, on another camera, and in another decade. Both "
                 "cohorts are expressed in degrees Celsius. One model per repeat trains on "
                 "DMR-IR patients and scores every patient of the target cohort once; the "
                 "left column is the same models on their own held-out source patients.")



# Fixed in advance: the encoders and regimes the robustness arm reports, in this order.
# Every one that ran is tabulated, including the one that failed its quality gate -- the
# purpose of the arm is to show whether the conclusion depends on the encoder, which a
# table showing only the encoders that worked could not do.
ENCODER_ARMS = [
    ("ConvNeXt-T", "frozen probe",
     R / "experiment_dmr_11_encoder_convnext_tiny_probe", "convnext_tiny_linear_probe"),
    ("ConvNeXt-T", "fine-tuned",
     R / "experiment_dmr_11_encoder_convnext_tiny_finetune",
     "convnext_tiny_finetune_discriminative_lr"),
    ("Swin-T", "frozen probe",
     R / "experiment_dmr_11_encoder_swin_tiny_patch4_window7_224_probe",
     "swin_tiny_patch4_window7_224_linear_probe"),
    ("Swin-T", "fine-tuned",
     R / "experiment_dmr_11_encoder_swin_tiny_patch4_window7_224_finetune",
     "swin_tiny_patch4_window7_224_finetune_discriminative_lr"),
    ("ResNet-50", "frozen probe",
     R / "experiment_dmr_11_encoder_resnet50_probe", "resnet50_linear_probe"),
    ("ResNet-50", "fine-tuned",
     R / "experiment_dmr_11_encoder_resnet50_finetune", "resnet50_finetune_discriminative_lr"),
]
ALLFRAMES_ARMS = [
    ("ConvNeXt-T", R / "experiment_dmr_01_leakage_combined", "patient_level__per_patient"),
    ("Swin-T", R / "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224",
     "patient_level__per_patient"),
    ("ResNet-50", R / "experiment_dmr_11_allframes_resnet50", "patient_level__per_patient"),
]
ERA_BY_ENCODER = {
    "ConvNeXt-T": R / "experiment_dmr_12_epoch_convnext_tiny_within_benign",
    "Swin-T": R / "experiment_dmr_12_epoch_swin_tiny_patch4_window7_224_within_benign",
    "ResNet-50": R / "experiment_dmr_12_epoch_resnet50_within_benign",
}


def _arm_cell(directory: Path, model: str) -> tuple[str, bool]:
    """One arm's interval, and whether its run was quarantined by the quality gate."""
    for suffix, failed in (("", False), (QUARANTINE_SUFFIX, True)):
        base = directory.parent / (directory.name + suffix)
        for filename in ("bootstrap_ci.csv", "metrics.csv"):
            got = ci_from_file(base / filename, model)
            if got is not None:
                return fmt(got), failed
    return "--", False


def build_encoders() -> str | None:
    """Does the conclusion depend on which encoder reads the image?"""
    rows = []
    for encoder, regime, directory, model in ENCODER_ARMS:
        cell, failed = _arm_cell(directory, model)
        if cell == "--":
            continue
        era = ERA_BY_ENCODER.get(encoder)
        era_cell = fmt(ci_from_file(era / "bootstrap_ci.csv", "epoch_decoder_within_benign")) \
            if era is not None else "--"
        rows.append([f"{encoder}, {regime}" + ("$^{\\dagger}$" if failed else ""),
                     "frontal", cell, era_cell])
    if rows:
        rows.append(["MIDRULE"])
    for encoder, directory, model in ALLFRAMES_ARMS:
        cell, failed = _arm_cell(directory, model)
        if cell == "--":
            continue
        era = ERA_BY_ENCODER.get(encoder)
        era_cell = fmt(ci_from_file(era / "bootstrap_ci.csv", "epoch_decoder_within_benign")) \
            if era is not None else "--"
        rows.append([f"{encoder}, frozen probe" + ("$^{\\dagger}$" if failed else ""),
                     "all frames", cell, era_cell])
    if not rows:
        return None
    return table(rows, ["Encoder and regime", "Frames", "Diagnosis AUROC [95\\% CI]",
                        "Acquisition era AUROC [95\\% CI]"],
                 "tab:dmr_encoders",
                 "Three encoders in two regimes, on identical folds. The right-hand column is "
                 "the same encoder trained to recover the acquisition era among benign "
                 "patients only, so the two columns say what an encoder can read about the "
                 "diagnosis and about the calendar. $^{\\dagger}$ quarantined by the "
                 "pre-specified quality gate (more than 5 of 25 folds emitting a near-constant "
                 "prediction) and reported as a failed configuration, not as a result.")


def build_conditional() -> str | None:
    """Discrimination restricted to pairs the acquisition date cannot separate."""
    path = R / "confounders/dmr_session_conditional/caliper_series.csv"
    if not path.exists():
        return None
    series = pd.read_csv(path)
    names = {
        "confounder_alone_control": "Acquisition date alone (control)",
        "experiment_dmr_01_leakage_combined::patient_level":
            "Thermogram, all frames (ConvNeXt-T)",
        "experiment_dmr_11_allframes_swin_tiny_patch4_window7_224::patient_level":
            "Thermogram, all frames (Swin-T)",
        "convnext_tiny_finetune_discriminative_lr": "Thermogram, frontal (ConvNeXt-T)",
        "swin_tiny_patch4_window7_224_finetune_discriminative_lr":
            "Thermogram, frontal (Swin-T)",
    }
    order = ["none", "180.0", "90.0", "30.0", "7.0", "same_visit_date"]
    header = ["Score", "All pairs", "$\\pm$180 d", "$\\pm$90 d", "$\\pm$30 d",
              "$\\pm$7 d", "Same session"]
    rows = []
    for key, label in names.items():
        g = series[series["score"] == key]
        if g.empty:
            continue
        cells = []
        for caliper in order:
            r = g[g["caliper"].astype(str) == caliper]
            if r.empty or not np.isfinite(r.iloc[0]["c_statistic"]):
                cells.append("--")
                continue
            r = r.iloc[0]
            star = "$^{*}$" if bool(r["exceeds_chance"]) else ""
            cells.append(f"{r['c_statistic']:.3f}{star}")
        rows.append([label] + cells)
    if not rows:
        return None
    pairs = series.drop_duplicates("caliper").set_index(series["caliper"].astype(str)
                                                        .drop_duplicates())
    counts = ["Admissible pairs"] + [
        str(int(pairs.loc[c, "n_pairs"])) if c in pairs.index else "--" for c in order]
    rows += [["MIDRULE"], counts]
    return table(rows, header, "tab:dmr_conditional",
                 "Discrimination restricted to benign/malignant pairs whose acquisition dates "
                 "differ by at most the caliper, and finally to pairs sharing an acquisition "
                 "session. Chance is 0.5; $^{*}$ marks an interval excluding it. The control "
                 "row is the acquisition date scored through the identical statistic: it must "
                 "fall to chance as the caliper narrows, and that it does is what shows the "
                 "restriction controls what it claims to.")


BUILDERS = {"cohort": build_cohort, "channels": build_channels, "ladder": build_ladder,
            "epoch": build_epoch,
            "attribution": build_attribution, "incremental": build_incremental,
            "transfer": build_transfer, "verification": build_verification,
            "encoders": build_encoders, "conditional": build_conditional}

# Availability is decided on file existence, never by building. A builder may recompute a
# bootstrap over 30k frame predictions, which takes minutes; listing what is available must
# stay instant.
REQUIRES = {
    "cohort": [R / "dmr_patient_manifest.csv", R / "dmr_image_manifest.csv"],
    "channels": [META / "bootstrap_ci.csv", LEAK / "predictions.csv"],
    "ladder": [LEAK / "contamination.csv", LEAK / "predictions.csv"],
    "epoch": [EPOCH_B / "bootstrap_ci.csv"],
    "attribution": [META / "bootstrap_ci.csv"],
    "incremental": [INC_FRONTAL / "final_statistics.csv"],
    "transfer": [R / "experiment_transfer_dmr_to_cali_perimage" / "metrics.csv"],
    "verification": [R / "experiment_07_metadata" / "bootstrap_ci.csv"],
    "encoders": [R / "experiment_dmr_11_encoder_convnext_tiny_probe" / "bootstrap_ci.csv"],
    "conditional": [R / "confounders/dmr_session_conditional" / "caliper_series.csv"],
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", choices=sorted(BUILDERS))
    p.add_argument("--all", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    if not a.table and not a.all:
        print("Available DMR-IR tables:\n")
        for key in BUILDERS:
            missing = [str(x) for x in REQUIRES[key] if not x.exists()]
            print(f"  {key:<12} {'ready' if not missing else 'missing ' + missing[0]}")
        print("\nRun with --table <key> or --all.")
        return 0

    keys = sorted(BUILDERS) if a.all else [a.table]
    for key in keys:
        out = BUILDERS[key]()
        if out is None:
            print(f"% {key}: inputs missing; experiment not yet run.", file=sys.stderr)
            continue
        print(f"% Generated by: python scripts/28_generate_dmr_tables.py --table {key}")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
