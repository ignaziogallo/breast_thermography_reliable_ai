#!/usr/bin/env python3
"""A direct replication of Pinto et al. Table III, and a clause-level attribution.

`data/DMR-IR/metadata.csv` ships a `text` column holding the natural-language rendering of
each patient's record -- the same textual representation their multimodal model consumes:

    Patient is 63 years old and identifies as multiracial. Menarche occurred at age 13.
    Her last menstrual period was at age 46. No complaints were reported. Observed signs
    include: yes. Has eating habits fat-free. Has a medical history of: mammography.
    Regarding the exam protocol, body temperature was 34.6 degrees Celsius.

Table III reports 0.918 accuracy from a linear probe on embeddings of that text alone, with
no image. This script probes the same strings under a *patient-level* protocol and then
removes one clause at a time, so the accuracy can be attributed to a clause rather than
credited to "clinical information" as a whole.

The clause that matters is `Has a medical history of: ...`, which enumerates biopsy,
mastectomy, radiotherapy and mammography -- records of what the clinic did *because* a
lesion was suspected or found. On the tabular fields, `biopsy` alone reaches AUROC 0.854 and
its absence 0.590. If dropping that one sentence collapses the text probe, Table III is
measuring the care pathway, not the patient.

Two arms are the experiment; the rest are controls:

  all_text                 every sentence: the replication.
  minus_medical_history    the same text with that one clause deleted.
  only_medical_history     that clause alone.
  pre_diagnostic_clauses   only clauses that can precede any suspicion.

A note on scope. The prompt carries a single age, `age_at_visit`. The `age_pair` channel
found in `scripts/25_dmr_metadata_leakage.py` -- where `age_current - age_at_visit`
reconstructs the acquisition era and reaches AUROC 0.946 -- is therefore **not** available
to Table III. It is a defect of the distributed metadata, not an explanation of their
number, and the two must not be conflated.

TF-IDF and the classifier are fitted inside each fold: fitting the vectoriser once over the
cohort would let the test partition's vocabulary and document frequencies inform training.

Example
-------
    python scripts/26_dmr_text_probe.py --metadata data/DMR-IR/metadata.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.summary import select_threshold, summarize_predictions
from src.utils.io import input_fingerprint, read_results_csv
from src.utils.seed import seed_everything

# Leading pattern -> (clause name, is it a record of the diagnostic pathway?). Order matters
# only for reporting; each sentence is assigned to the first pattern it matches.
CLAUSES = [
    (r"^Patient is .*years old", "demographics", False),
    (r"^Menarche", "menarche", False),
    (r"^(Her|His)? ?last menstrual period|menopaus", "menstrual", False),
    (r"complaint", "complaints", False),
    (r"^Observed signs|^Reported symptoms|symptom", "signs_symptoms", False),
    (r"eating habits", "eating_habits", False),
    (r"family", "family_history", False),
    (r"^Has a medical history of", "medical_history", True),
    (r"nipple|wart|prosthes|plastic surgery|hormone replacement", "breast_findings", True),
    (r"exam protocol|body temperature", "exam_protocol", False),
]
GRID = [0.01, 0.1, 1.0, 10.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=Path("data/DMR-IR/metadata.csv"))
    p.add_argument("--patients", type=Path, default=Path("results/dmr_patient_manifest.csv"))
    p.add_argument("--splits", type=Path, default=Path("results/dmr_patient_splits.csv"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/experiment_dmr_05_text_probe"))
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", str(text)) if s.strip()]


def classify(sentence: str) -> str:
    for pattern, name, _ in CLAUSES:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return name
    return "UNMATCHED"


def patient_text(metadata: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """One text per patient, after checking that the record really is per patient."""
    md = metadata.copy()
    md["patient_dir"] = md["patient_dir"].astype(str).str.zfill(4)
    md["patient_id"] = "DMR" + md["patient_dir"]
    md = md[md["patient_id"].isin(set(patients["patient_id"]))]

    varying = md.groupby("patient_id")["text"].nunique()
    if (varying > 1).any():
        bad = varying[varying > 1]
        raise SystemExit(f"{len(bad)} patients carry more than one distinct text, e.g. "
                         f"{bad.index[:3].tolist()}. The prompt is supposed to describe the "
                         "patient, not the frame; refusing to pick one arbitrarily.")

    out = md.groupby("patient_id", as_index=False)["text"].first()
    return out.merge(patients[["patient_id", "patient_binary_label"]], on="patient_id")


def clause_report(table: pd.DataFrame) -> pd.DataFrame:
    """Which clauses exist, how often, and what each predicts on its own."""
    y = table["patient_binary_label"].to_numpy(dtype=int)
    rows, seen = [], {}
    for text in table["text"]:
        for s in sentences(text):
            seen.setdefault(classify(s), []).append(s)

    for name in list(dict.fromkeys([c[1] for c in CLAUSES] + ["UNMATCHED"])):
        if name not in seen:
            continue
        present = np.array([any(classify(s) == name for s in sentences(t))
                            for t in table["text"]], dtype=int)
        auc = (roc_auc_score(y, present) if len(np.unique(present)) > 1 else np.nan)
        post = next((p for _, n, p in CLAUSES if n == name), False)
        rows.append({"clause": name, "post_diagnostic": post,
                     "patients_with_clause": int(present.sum()), "n": len(table),
                     "auroc_of_presence": round(float(auc), 4) if np.isfinite(auc) else np.nan,
                     "example": seen[name][0][:110]})
    return pd.DataFrame(rows)


def filtered(table: pd.DataFrame, keep=None, drop=()) -> pd.Series:
    """Rebuild each text from the clauses selected."""
    def build(text: str) -> str:
        out = [s for s in sentences(text)
               if classify(s) not in drop and (keep is None or classify(s) in keep)]
        return " ".join(out) if out else "empty"
    return table["text"].map(build)


def run_text_cv(table: pd.DataFrame, text: pd.Series, splits: pd.DataFrame,
                model_name: str, seed: int) -> pd.DataFrame:
    """The repository's fold protocol, with the vectoriser fitted inside each fold."""
    data = table.assign(_text=text).set_index("patient_id")
    rows = []
    for (repeat, fold), g in splits.groupby(["repeat", "fold"]):
        ids = {s: g.loc[g["split"] == s, "patient_id"].tolist() for s in ("train", "val", "test")}
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            overlap = set(ids[a]) & set(ids[b])
            assert not overlap, f"{a}/{b} overlap at r{repeat} f{fold}: {sorted(overlap)[:5]}"

        X = {s: data.loc[ids[s], "_text"].tolist() for s in ids}
        y = {s: data.loc[ids[s], "patient_binary_label"].to_numpy(dtype=int) for s in ids}

        best, best_auc = None, -np.inf
        for C in GRID:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=5000, C=C, class_weight="balanced",
                                           random_state=seed)),
            ])
            pipe.fit(X["train"], y["train"])
            p_val = pipe.predict_proba(X["val"])[:, 1]
            auc = roc_auc_score(y["val"], p_val) if len(np.unique(y["val"])) == 2 else np.nan
            if np.isfinite(auc) and auc > best_auc:
                best, best_auc = (pipe, C, p_val), auc
        if best is None:
            continue

        pipe, C, p_val = best
        threshold = select_threshold(y["val"], p_val)
        p_test = pipe.predict_proba(X["test"])[:, 1]
        for pid, yt, pp in zip(ids["test"], y["test"], p_test):
            rows.append({"patient_id": pid, "repeat": int(repeat), "fold": int(fold),
                         "y_true": int(yt), "y_prob": float(pp),
                         "y_pred": int(pp >= threshold), "threshold": threshold,
                         "model": model_name, "config": json.dumps({"C": C})})
    return pd.DataFrame(rows)


def main() -> int:
    a = parse_args()
    seed_everything(a.seed)
    if not a.metadata.exists():
        raise SystemExit(f"{a.metadata} not found. This script needs the dataset, so it runs "
                         "on the compute server, not on a development workstation.")

    patients = read_results_csv(a.patients).dropna(subset=["patient_binary_label"])
    splits = read_results_csv(a.splits)
    metadata = pd.read_csv(a.metadata, dtype={"patient_dir": str})
    if "text" not in metadata.columns:
        raise SystemExit(f"{a.metadata} has no `text` column; available: "
                         f"{sorted(metadata.columns)}")

    table = patient_text(metadata, patients)
    absent = sorted(set(splits["patient_id"]) - set(table["patient_id"]))
    if absent:
        print(f"Restricting splits to the {len(table)} patients with text: "
              f"{len(absent)} dropped, e.g. {absent[:5]}")
        splits = splits[splits["patient_id"].isin(set(table["patient_id"]))]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    report = clause_report(table)
    report.to_csv(a.output_dir / "clauses.csv", index=False)
    print(f"\n{len(table)} patients with a prompt. Clauses found:")
    print(report.to_string(index=False))
    if "UNMATCHED" in set(report["clause"]):
        print("\nWARNING: some sentences matched no clause pattern and are kept in every arm; "
              "see the UNMATCHED row above and extend CLAUSES if it carries content.")

    post = {n for _, n, p in CLAUSES if p}
    arms = {
        "all_text": filtered(table),
        "minus_medical_history": filtered(table, drop={"medical_history"}),
        "only_medical_history": filtered(table, keep={"medical_history"}),
        "minus_all_post_diagnostic": filtered(table, drop=post),
        "pre_diagnostic_clauses": filtered(table, drop=post | {"UNMATCHED"}),
    }

    all_preds = []
    for name, text in arms.items():
        empty = int((text == "empty").sum())
        print(f"\n{name}: {len(text) - empty}/{len(text)} patients have content", flush=True)
        all_preds.append(run_text_cv(table, text, splits, name, a.seed))

    preds = pd.concat(all_preds, ignore_index=True)
    per_repeat, ci, summary = summarize_predictions(preds, a.n_bootstrap, a.seed)

    preds.to_csv(a.output_dir / "predictions.csv", index=False)
    per_repeat.to_csv(a.output_dir / "metrics_per_repeat.csv", index=False)
    summary.to_csv(a.output_dir / "metrics.csv", index=False)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    (a.output_dir / "config.json").write_text(json.dumps({
        "experiment": a.output_dir.name,
        "research_question": "which clause of the DMR-IR prompt carries the metadata-only score",
        "comparison": "Pinto et al. 2025 Table III: 0.918 accuracy from a linear probe on "
                      "embeddings of this same text, image-level split, no test set",
        "protocol_difference": "patient-level 5x5 folds; TF-IDF and classifier fitted inside "
                               "each fold so the test partition's vocabulary never informs training",
        "encoder_difference": "TF-IDF bigrams, not a sentence-transformer; the comparison is "
                              "between clauses, not against their absolute number",
        "arms": sorted(arms),
        "post_diagnostic_clauses": sorted(post),
        "age_pair_scope_note": "the prompt carries only age_at_visit, so the age_current pair "
                               "channel of scripts/25 (AUROC 0.946) is NOT available here and "
                               "does not explain Table III",
        "n_patients": len(table),
        "inputs": input_fingerprint(a.patients, a.splits),
        "n_bootstrap": a.n_bootstrap, "seed": a.seed,
    }, indent=2), encoding="utf-8")

    table_out = ci[ci["metric"] == "roc_auc"][["model", "estimate", "ci_lower", "ci_upper"]].merge(
        summary[["model", "accuracy_at_selected_threshold_mean", "balanced_accuracy_mean"]],
        on="model", how="left")
    print("\n" + table_out.round(3).to_string(index=False))
    print("\nIf all_text is high and minus_medical_history collapses towards chance, Table III "
          "is reading the care pathway rather than the patient.")
    print(f"Wrote {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
