"""Transparent tabular baselines for the metadata-only experiment.

These models exist to answer a scientific question, not to win a leaderboard: how much
patient-level malignancy is predictable from clinical covariates alone? The audit found
age to be strongly associated with the target, so an image model must be shown to add
information beyond this baseline before its representations can be read as thermal
evidence. The models are therefore kept deliberately simple and interpretable.

Every transformation that learns from data (imputation, scaling) lives inside the
pipeline so that it is fitted on training patients only and never sees validation or
test patients.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# name -> (features, hyperparameter grid). The grid is searched on the patient-level
# validation split of each fold, never on the test fold.
MODELS: dict[str, dict] = {
    "logistic_metadata": {
        "features": ["age", "bmi", "temperature"],
        "grid": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "description": "L2 logistic regression on the three pre-specified covariates.",
    },
    "logistic_age_only": {
        "features": ["age"],
        "grid": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "description": "Minimal confounder model: age alone.",
    },
    "gradient_boosting_metadata": {
        "features": ["age", "bmi", "temperature"],
        "grid": {"clf__max_depth": [1, 2], "clf__n_estimators": [50, 150]},
        "description": "Non-linearity check; not a primary comparison.",
    },
}


def build_pipeline(kind: str, seed: int = 42) -> Pipeline:
    """Assemble a tabular pipeline. `kind` is 'logistic' or 'gradient_boosting'.

    Imputation and scaling live inside the pipeline so that they are fitted on training
    patients only and never see validation or test patients.
    """
    if kind == "logistic":
        # L2 is the default; naming it explicitly is deprecated from scikit-learn 1.8.
        clf = LogisticRegression(solver="lbfgs", max_iter=5000,
                                 class_weight="balanced", random_state=seed)
    elif kind == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Unknown estimator kind {kind!r}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def build_estimator(name: str, seed: int = 42) -> Pipeline:
    """The pipeline of one named model of `MODELS`."""
    if name not in MODELS:
        raise ValueError(f"Unknown metadata model {name!r}; available: {sorted(MODELS)}")
    kind = "logistic" if name.startswith("logistic") else "gradient_boosting"
    return build_pipeline(kind, seed)


def feature_names(name: str) -> list[str]:
    return list(MODELS[name]["features"])
