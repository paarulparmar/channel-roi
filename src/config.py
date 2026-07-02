"""
Central configuration for the Marketing Channel ROI Prediction &
Budget Optimization project.

All paths are resolved relative to the project root so the pipeline runs
identically regardless of the working directory it is launched from.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
# "olist"     -> REAL anonymized data (Olist Marketing Funnel + E-Commerce).
#                Acquisition channel, dates, behaviour and revenue are all real;
#                only per-channel CAC is overlaid from published benchmarks
#                (Olist did not publish per-lead ad spend).
# "synthetic" -> fully simulated dataset (kept as a high-volume fallback).
DATA_SOURCE: str = "olist"

RAW_DATA_PATH: Path = DATA_DIR / "customers.csv"
OLIST_CACHE_DIR: Path = DATA_DIR / "olist_raw"
OLIST_BASE_URL: str = (
    "https://raw.githubusercontent.com/"
    "Ganesh7699/Brazilian-E-Commerce-OList/main"
)
OLIST_FILES: list[str] = [
    "olist_marketing_qualified_leads_dataset.csv",
    "olist_closed_deals_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_orders_dataset.csv",
]

# Per-channel CAC benchmark overlay (USD). Real Olist data has no per-lead ad
# spend, so channel-level acquisition cost is applied from typical paid/earned
# channel economics (earned channels cheap, paid channels expensive). This is
# the ONLY non-native field in the real dataset and is clearly labelled as an
# assumption throughout.
OLIST_CHANNEL_CAC: dict[str, float] = {
    "Referral": 18.0,
    "Email": 15.0,
    "Organic Search": 25.0,
    "Direct": 22.0,
    "Unknown": 40.0,
    "Other": 45.0,
    "Display": 60.0,
    "Social": 55.0,
    "Paid Search": 70.0,
}

SCORED_DATA_PATH: Path = OUTPUTS_DIR / "scored_customers.csv"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH: Path = OUTPUTS_DIR / "model_metrics.csv"
SCORECARD_PATH: Path = OUTPUTS_DIR / "business_scorecard.csv"
BUDGET_PATH: Path = OUTPUTS_DIR / "budget_optimization.csv"

for _d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Dataset generation parameters
# --------------------------------------------------------------------------- #
N_CUSTOMERS: int = 4200

# Each acquisition channel has a distinct economic and behavioural profile.
#   share            : fraction of customers acquired through the channel
#   cac_mean/cac_std : cost of acquiring one customer (USD)
#   hv_propensity    : baseline latent propensity toward becoming high value
#   aov_mean/aov_std : first-order value distribution (USD)
#   repeat_speed     : mean days to a second purchase (lower = stickier)
CHANNELS: dict[str, dict[str, float]] = {
    "Referral": {
        "share": 0.14, "cac_mean": 12.0, "cac_std": 4.0,
        "hv_propensity": 0.55, "aov_mean": 92.0, "aov_std": 34.0,
        "repeat_speed": 22.0,
    },
    "Organic Search": {
        "share": 0.20, "cac_mean": 8.0, "cac_std": 3.0,
        "hv_propensity": 0.42, "aov_mean": 78.0, "aov_std": 30.0,
        "repeat_speed": 30.0,
    },
    "Email": {
        "share": 0.12, "cac_mean": 6.0, "cac_std": 2.5,
        "hv_propensity": 0.38, "aov_mean": 70.0, "aov_std": 26.0,
        "repeat_speed": 26.0,
    },
    "Google Ads": {
        "share": 0.22, "cac_mean": 34.0, "cac_std": 10.0,
        "hv_propensity": 0.30, "aov_mean": 82.0, "aov_std": 33.0,
        "repeat_speed": 34.0,
    },
    "Facebook Ads": {
        "share": 0.18, "cac_mean": 28.0, "cac_std": 9.0,
        "hv_propensity": 0.22, "aov_mean": 64.0, "aov_std": 25.0,
        "repeat_speed": 42.0,
    },
    "Instagram Ads": {
        "share": 0.14, "cac_mean": 31.0, "cac_std": 9.5,
        "hv_propensity": 0.18, "aov_mean": 58.0, "aov_std": 24.0,
        "repeat_speed": 48.0,
    },
}

DEVICE_TYPES: dict[str, float] = {"Mobile": 0.58, "Desktop": 0.34, "Tablet": 0.08}

# Device effect on the latent high-value propensity (desktop shoppers convert
# to loyal buyers slightly more often in this simulated business).
DEVICE_HV_EFFECT: dict[str, float] = {"Mobile": -0.03, "Desktop": 0.06, "Tablet": -0.02}

# High-value business definition: lifetime revenue at or above this threshold.
HIGH_VALUE_REVENUE_THRESHOLD: float = 320.0

# --------------------------------------------------------------------------- #
# Modelling parameters
# --------------------------------------------------------------------------- #
TEST_SIZE: float = 0.25
CV_FOLDS: int = 5

# Feature groups used by the preprocessing pipeline.
NUMERIC_FEATURES: list[str] = [
    "acquisition_cost",
    "first_order_value",
    "days_to_second_purchase",
    "signup_month",
    "signup_dayofweek",
    "value_per_cost",
    "fast_repeat_flag",
]
# The secondary categorical differs by source: device type is available in the
# synthetic funnel; the real Olist data provides the seller's business type.
SECONDARY_CATEGORICAL: str = (
    "business_type" if DATA_SOURCE == "olist" else "device_type"
)
CATEGORICAL_FEATURES: list[str] = ["acquisition_channel", SECONDARY_CATEGORICAL]
TARGET: str = "is_high_value"

# --------------------------------------------------------------------------- #
# Business / optimization parameters
# --------------------------------------------------------------------------- #
# Total monthly acquisition budget to allocate across channels (USD).
TOTAL_BUDGET: float = 250_000.0

# Guard-rails so the optimizer never fully starves or over-concentrates a channel.
MIN_CHANNEL_BUDGET_SHARE: float = 0.04
MAX_CHANNEL_BUDGET_SHARE: float = 0.35

# Expected revenue contribution of a customer, conditioned on value tier.
# Used to translate predicted high-value rates into expected revenue per acquisition.
HIGH_VALUE_EXPECTED_REVENUE: float = 480.0
LOW_VALUE_EXPECTED_REVENUE: float = 95.0
