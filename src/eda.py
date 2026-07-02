"""
Exploratory Data Analysis -- business-focused.

Rather than generic descriptive statistics, this module computes the unit
economics a growth team actually reasons about: Customer Acquisition Cost (CAC),
average revenue, Return on Investment (ROI) and the high-value customer rate,
all sliced by acquisition channel.
"""
from __future__ import annotations

import pandas as pd

from . import config


def channel_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-channel unit economics.

    ROI is defined as (total revenue - total acquisition spend) / total spend,
    i.e. the profit generated per dollar invested in acquiring the cohort.
    """
    grp = df.groupby("acquisition_channel")
    econ = grp.agg(
        customers=("customer_id", "count"),
        total_spend=("acquisition_cost", "sum"),
        total_revenue=("total_revenue_generated", "sum"),
        avg_cac=("acquisition_cost", "mean"),
        avg_revenue=("total_revenue_generated", "mean"),
        avg_first_order=("first_order_value", "mean"),
        high_value_rate=("is_high_value", "mean"),
    )
    econ["roi"] = (
        (econ["total_revenue"] - econ["total_spend"]) / econ["total_spend"]
    )
    econ["revenue_per_cac_dollar"] = econ["total_revenue"] / econ["total_spend"]
    econ = econ.sort_values("roi", ascending=False)
    return econ.round(3)


def overall_summary(df: pd.DataFrame) -> dict:
    """High-level dataset summary used in the console report and README."""
    return {
        "n_customers": int(len(df)),
        "high_value_rate": float(df["is_high_value"].mean()),
        "avg_cac": float(df["acquisition_cost"].mean()),
        "avg_revenue": float(df["total_revenue_generated"].mean()),
        "blended_roi": float(
            (df["total_revenue_generated"].sum() - df["acquisition_cost"].sum())
            / df["acquisition_cost"].sum()
        ),
        "n_channels": int(df["acquisition_channel"].nunique()),
    }


def print_eda_report(df: pd.DataFrame) -> pd.DataFrame:
    econ = channel_economics(df)
    summary = overall_summary(df)
    print("=" * 68)
    print("EXPLORATORY DATA ANALYSIS  --  CHANNEL UNIT ECONOMICS")
    print("=" * 68)
    print(f"Customers            : {summary['n_customers']:,}")
    print(f"Channels             : {summary['n_channels']}")
    print(f"High-value rate      : {summary['high_value_rate']:.1%}")
    print(f"Average CAC          : ${summary['avg_cac']:.2f}")
    print(f"Average revenue      : ${summary['avg_revenue']:.2f}")
    print(f"Blended ROI          : {summary['blended_roi']:.2f}x")
    print("-" * 68)
    show = econ[["customers", "avg_cac", "avg_revenue", "high_value_rate", "roi"]]
    print(show.to_string())
    print("=" * 68)
    return econ
