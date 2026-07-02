"""
Feature engineering and preprocessing.

Builds a leakage-safe feature matrix and a scikit-learn ColumnTransformer that
scales numeric features and one-hot encodes categoricals. Crucially,
`total_revenue_generated` is *excluded* from the feature set because the target
is derived from it -- using it would leak the label. Only acquisition-time and
early-behaviour signals are used, matching the real deployment scenario where a
customer must be scored shortly after signup.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features available at (or shortly after) acquisition time."""
    out = df.copy()
    out["signup_date"] = pd.to_datetime(out["signup_date"])
    out["signup_month"] = out["signup_date"].dt.month
    out["signup_dayofweek"] = out["signup_date"].dt.dayofweek

    # Early spend efficiency: value captured per dollar of acquisition cost.
    out["value_per_cost"] = (
        out["first_order_value"] / out["acquisition_cost"].clip(lower=1e-6)
    ).round(4)

    # Behavioural stickiness signal: did the customer come back quickly?
    median_repeat = out["days_to_second_purchase"].median()
    out["fast_repeat_flag"] = (
        out["days_to_second_purchase"] <= median_repeat
    ).astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Construct the ColumnTransformer for numeric + categorical features."""
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns used as model inputs (post feature engineering)."""
    cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    return df[cols].copy()


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recover human-readable feature names after fitting the preprocessor."""
    num = config.NUMERIC_FEATURES
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat = ohe.get_feature_names_out(config.CATEGORICAL_FEATURES).tolist()
    return list(num) + cat
