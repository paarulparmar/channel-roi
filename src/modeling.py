"""
Model training, comparison and evaluation.

Trains three classifiers inside a shared preprocessing pipeline:
    * Logistic Regression  (interpretable baseline)
    * Random Forest        (bagged non-linear ensemble)
    * XGBoost              (gradient-boosted trees)

Class imbalance is handled explicitly (class weights / scale_pos_weight). Models
are compared with stratified cross-validated ROC-AUC and a held-out test set is
used for the final metric table. The best model is selected on test ROC-AUC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessor, get_feature_frame, get_feature_names


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    roc_curve: tuple = ()  # (fpr, tpr)
    confusion: np.ndarray = None


def _build_models(pos_weight: float) -> dict[str, object]:
    """Instantiate the three estimators with imbalance-aware settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def split_data(df: pd.DataFrame):
    X = get_feature_frame(df)
    y = df[config.TARGET].astype(int)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def train_and_compare(df: pd.DataFrame):
    """Train all models and return (results, best_name, split tuple)."""
    X_train, X_test, y_train, y_test = split_data(df)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    estimators = _build_models(pos_weight)
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    results: dict[str, TrainedModel] = {}
    for name, est in estimators.items():
        pipe = Pipeline(
            steps=[("prep", build_preprocessor()), ("clf", est)]
        )
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        results[name] = TrainedModel(
            name=name,
            pipeline=pipe,
            metrics=metrics,
            cv_auc_mean=float(cv_scores.mean()),
            cv_auc_std=float(cv_scores.std()),
            roc_curve=(fpr, tpr),
            confusion=confusion_matrix(y_test, pred),
        )

    best_name = max(results, key=lambda n: results[n].metrics["roc_auc"])
    return results, best_name, (X_train, X_test, y_train, y_test)


def metrics_table(results: dict[str, TrainedModel]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"model": name, **r.metrics,
               "cv_auc_mean": r.cv_auc_mean, "cv_auc_std": r.cv_auc_std}
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("model")
    return tbl.sort_values("roc_auc", ascending=False).round(4)


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    """Return a tidy feature-importance frame for the given trained pipeline.

    Uses model-native importances (coef magnitude for logistic regression,
    impurity/gain importance for tree ensembles).
    """
    prep = model.pipeline.named_steps["prep"]
    clf = model.pipeline.named_steps["clf"]
    names = get_feature_names(prep)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).ravel()
    else:  # pragma: no cover
        importances = np.zeros(len(names))

    imp = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp


def save_best_model(results: dict[str, TrainedModel], best_name: str) -> None:
    joblib.dump(results[best_name].pipeline, config.BEST_MODEL_PATH)


def save_metrics(tbl: pd.DataFrame) -> None:
    tbl.to_csv(config.METRICS_PATH)
