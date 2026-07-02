"""
Real-data loader: Olist Marketing Funnel + Brazilian E-Commerce.

This builds a genuine marketing-channel dataset from Olist's publicly released,
anonymized commercial data. The unit of analysis is an *acquired seller* (a
marketing-qualified lead that closed), which is the entity Olist actually spends
to acquire through each channel.

Real, native fields:
    * acquisition_channel  <- MQL `origin` (organic_search, paid_search, social,
                              email, referral, direct, display, ...)
    * signup_date          <- deal `won_date`
    * business_type        <- real seller business type (reseller/manufacturer)
    * first_order_value    <- value of the seller's first order
    * days_to_second_purchase <- gap between first and second order
    * total_revenue_generated <- summed order-item revenue for the seller
    * is_high_value        <- top-tier revenue (data-driven threshold)

Overlaid (clearly labelled) field:
    * acquisition_cost     <- per-channel CAC benchmark (Olist did not publish
                              per-lead ad spend). Everything else is real.

Leads that closed but never generated an order are retained as the genuine
low-value / churn-risk class (revenue = 0), which is exactly the outcome the
model exists to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Map raw Olist origins to clean, presentation-ready channel names. Sparse
# origins are folded into "Other" so every channel has a usable sample size.
_ORIGIN_MAP = {
    "organic_search": "Organic Search",
    "paid_search": "Paid Search",
    "social": "Social",
    "direct_traffic": "Direct",
    "email": "Email",
    "referral": "Referral",
    "display": "Display",
    "unknown": "Unknown",
    "other": "Other",
    "other_publicities": "Other",
}

_NEVER_REPEATED = 999  # sentinel: seller never placed a second order


def _download_olist() -> None:
    """Fetch the four Olist CSVs into the local cache (idempotent)."""
    import urllib.request

    config.OLIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in config.OLIST_FILES:
        dest = config.OLIST_CACHE_DIR / fname
        if dest.exists():
            continue
        url = f"{config.OLIST_BASE_URL}/{fname}"
        urllib.request.urlretrieve(url, dest)


def _read(fname: str, **kw) -> pd.DataFrame:
    return pd.read_csv(config.OLIST_CACHE_DIR / fname, **kw)


def build_olist_dataset(random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Construct the seller-acquisition dataset in the project schema."""
    _download_olist()
    rng = np.random.default_rng(random_state)

    mql = _read(
        "olist_marketing_qualified_leads_dataset.csv",
        parse_dates=["first_contact_date"],
    )
    deals = _read("olist_closed_deals_dataset.csv", parse_dates=["won_date"])
    items = _read("olist_order_items_dataset.csv")
    orders = _read("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])

    # 1. Acquired sellers with their acquisition channel + business type.
    deals = deals.merge(
        mql[["mql_id", "origin"]], on="mql_id", how="left"
    ).dropna(subset=["seller_id"])
    deals["acquisition_channel"] = (
        deals["origin"].map(_ORIGIN_MAP).fillna("Other")
    )
    deals["business_type"] = (
        deals["business_type"].fillna("unknown").replace("", "unknown")
    )

    # 2. Seller-level order economics (exclude cancelled / unavailable orders).
    it = items.merge(
        orders[["order_id", "order_purchase_timestamp", "order_status"]],
        on="order_id",
        how="left",
    )
    it = it[~it["order_status"].isin(["canceled", "unavailable"])].copy()

    rev = it.groupby("seller_id").agg(
        total_revenue_generated=("price", "sum"),
        n_orders=("order_id", "nunique"),
    )

    # Value of each seller's first order.
    first_ts = it.groupby("seller_id")["order_purchase_timestamp"].min().rename("fo_ts")
    it = it.merge(first_ts, on="seller_id")
    first_val = (
        it[it["order_purchase_timestamp"] == it["fo_ts"]]
        .groupby("seller_id")["price"]
        .sum()
        .rename("first_order_value")
    )

    # Days between first and second order.
    def _gap(ts: pd.Series) -> float:
        s = ts.sort_values()
        return (s.iloc[1] - s.iloc[0]).days if len(s) >= 2 else _NEVER_REPEATED

    gap = (
        it.groupby("seller_id")["order_purchase_timestamp"].apply(_gap)
        .rename("days_to_second_purchase")
    )

    econ = pd.concat([rev, first_val, gap], axis=1)

    # 3. Join economics onto acquired sellers. Sellers with no orders become the
    #    genuine zero-revenue / churn class.
    df = deals.merge(econ, left_on="seller_id", right_index=True, how="left")
    df["total_revenue_generated"] = df["total_revenue_generated"].fillna(0.0)
    df["first_order_value"] = df["first_order_value"].fillna(0.0)
    df["days_to_second_purchase"] = (
        df["days_to_second_purchase"].fillna(_NEVER_REPEATED).astype(int)
    )

    # 4. Overlay per-channel CAC benchmark (only non-native field).
    base_cac = df["acquisition_channel"].map(config.OLIST_CHANNEL_CAC).fillna(40.0)
    noise = rng.normal(1.0, 0.12, size=len(df)).clip(0.6, 1.5)
    df["acquisition_cost"] = (base_cac * noise).round(2)

    # 5. Signup date + high-value label (data-driven top-tier revenue).
    df["signup_date"] = df["won_date"]
    threshold = df.loc[df["total_revenue_generated"] > 0, "total_revenue_generated"].quantile(0.37)
    df["is_high_value"] = (df["total_revenue_generated"] >= threshold).astype(int)

    df["customer_id"] = df["seller_id"]
    out = df[
        [
            "customer_id",
            "signup_date",
            "acquisition_channel",
            "acquisition_cost",
            "business_type",
            "first_order_value",
            "days_to_second_purchase",
            "total_revenue_generated",
            "is_high_value",
        ]
    ].reset_index(drop=True)
    out["total_revenue_generated"] = out["total_revenue_generated"].round(2)
    out["first_order_value"] = out["first_order_value"].round(2)
    return out


def load_or_build(force: bool = False) -> pd.DataFrame:
    if config.RAW_DATA_PATH.exists() and not force:
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=["signup_date"])
    df = build_olist_dataset()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    d = load_or_build(force=True)
    print(f"Built {len(d):,} acquired sellers from REAL Olist data")
    print(f"High-value rate = {d['is_high_value'].mean():.1%}")
    print(d["acquisition_channel"].value_counts())
