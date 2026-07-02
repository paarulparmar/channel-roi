"""
Business layer -- translating model predictions into budget decisions.

The classifier outputs a per-customer probability of becoming high value. This
module aggregates those probabilities to the channel level, builds a decision
scorecard, and runs a constrained budget-reallocation that shifts spend toward
channels with the strongest predicted quality-adjusted return. A before/after
simulation quantifies the expected revenue and ROI lift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .modeling import TrainedModel
from .preprocessing import get_feature_frame


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_customers(df: pd.DataFrame, model: TrainedModel) -> pd.DataFrame:
    """Attach a predicted high-value probability to every customer."""
    X = get_feature_frame(df)
    proba = model.pipeline.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["predicted_hv_proba"] = proba.round(4)
    scored["predicted_high_value"] = (proba >= 0.5).astype(int)
    return scored


# --------------------------------------------------------------------------- #
# Channel scorecard
# --------------------------------------------------------------------------- #
def _expected_revenue_per_customer(hv_rate: float) -> float:
    """Blend high/low value expected revenue by predicted high-value rate."""
    return (
        hv_rate * config.HIGH_VALUE_EXPECTED_REVENUE
        + (1.0 - hv_rate) * config.LOW_VALUE_EXPECTED_REVENUE
    )


def channel_scorecard(scored: pd.DataFrame) -> pd.DataFrame:
    """Build the channel decision scorecard the growth team acts on."""
    grp = scored.groupby("acquisition_channel")
    sc = grp.agg(
        customers=("customer_id", "count"),
        avg_cac=("acquisition_cost", "mean"),
        actual_hv_rate=("is_high_value", "mean"),
        predicted_hv_rate=("predicted_hv_proba", "mean"),
        avg_revenue=("total_revenue_generated", "mean"),
    )
    total_spend = grp["acquisition_cost"].sum()
    total_revenue = grp["total_revenue_generated"].sum()
    sc["roi"] = (total_revenue - total_spend) / total_spend

    # Efficiency score: expected revenue per acquisition dollar, driven by the
    # model's predicted high-value rate. This is the ranking signal.
    exp_rev = sc["predicted_hv_rate"].apply(_expected_revenue_per_customer)
    sc["expected_value_per_customer"] = exp_rev
    sc["efficiency_score"] = exp_rev / sc["avg_cac"]

    median_eff = sc["efficiency_score"].median()
    top = sc["efficiency_score"].quantile(0.66)
    bottom = sc["efficiency_score"].quantile(0.33)

    def _recommend(score: float) -> str:
        if score >= top:
            return "Scale Up"
        if score <= bottom:
            return "Reduce / Optimize"
        return "Maintain"

    sc["recommendation"] = sc["efficiency_score"].apply(_recommend)
    sc = sc.sort_values("efficiency_score", ascending=False)
    return sc.round(4)


# --------------------------------------------------------------------------- #
# Budget optimization
# --------------------------------------------------------------------------- #
def optimize_budget(
    scorecard: pd.DataFrame,
    total_budget: float = config.TOTAL_BUDGET,
) -> pd.DataFrame:
    """Reallocate budget proportionally to each channel's efficiency score.

    Allocation is proportional to the model-derived efficiency score, then
    clipped to configured per-channel min/max shares and renormalised so the
    total budget is preserved. The 'before' allocation mirrors the historical
    spend mix implied by the observed cohort.
    """
    sc = scorecard.copy()

    # Historical / current allocation = share of observed acquisition spend.
    hist_spend = sc["avg_cac"] * sc["customers"]
    current_share = hist_spend / hist_spend.sum()

    # Optimized allocation proportional to efficiency, with guard-rails.
    raw = sc["efficiency_score"].clip(lower=0)
    opt_share = raw / raw.sum()
    opt_share = opt_share.clip(
        lower=config.MIN_CHANNEL_BUDGET_SHARE,
        upper=config.MAX_CHANNEL_BUDGET_SHARE,
    )
    opt_share = opt_share / opt_share.sum()

    out = pd.DataFrame(index=sc.index)
    out["avg_cac"] = sc["avg_cac"]
    out["predicted_hv_rate"] = sc["predicted_hv_rate"]
    out["efficiency_score"] = sc["efficiency_score"]
    out["current_share"] = current_share.round(4)
    out["optimized_share"] = opt_share.round(4)
    out["current_budget"] = (current_share * total_budget).round(2)
    out["optimized_budget"] = (opt_share * total_budget).round(2)
    out["budget_delta"] = (out["optimized_budget"] - out["current_budget"]).round(2)
    return out.sort_values("optimized_budget", ascending=False)


def simulate_scenario(
    allocation: pd.Series,
    budget: pd.Series,
    scorecard: pd.DataFrame,
) -> dict:
    """Simulate customers acquired, expected revenue and ROI for an allocation.

    Customers acquired per channel = channel budget / channel CAC. Expected
    revenue uses the predicted-high-value-blended per-customer revenue.
    """
    cac = scorecard["avg_cac"]
    exp_rev = scorecard["predicted_hv_rate"].apply(_expected_revenue_per_customer)

    customers = budget / cac
    revenue = (customers * exp_rev).sum()
    spend = budget.sum()
    roi = (revenue - spend) / spend
    return {
        "customers_acquired": float(customers.sum()),
        "expected_revenue": float(revenue),
        "spend": float(spend),
        "roi": float(roi),
    }


def simulate_before_after(
    scorecard: pd.DataFrame, budget_df: pd.DataFrame
) -> pd.DataFrame:
    """Return a tidy before/after comparison of the two allocations."""
    before = simulate_scenario(
        budget_df["current_share"], budget_df["current_budget"], scorecard
    )
    after = simulate_scenario(
        budget_df["optimized_share"], budget_df["optimized_budget"], scorecard
    )
    comp = pd.DataFrame([before, after], index=["Before (current mix)", "After (optimized)"])
    comp.loc["Lift"] = comp.loc["After (optimized)"] - comp.loc["Before (current mix)"]
    comp.loc["Lift %"] = (
        (comp.loc["After (optimized)"] - comp.loc["Before (current mix)"])
        / comp.loc["Before (current mix)"].abs()
        * 100.0
    )
    return comp.round(2)
