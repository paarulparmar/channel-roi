"""
Synthetic dataset generation.

Produces a realistic customer-acquisition dataset in which acquisition channels
carry genuinely different economics (CAC) and customer quality (high-value
propensity). The target `is_high_value` is derived from simulated lifetime
revenue so that it is *causally consistent* with the observable features rather
than assigned at random -- this is what allows the downstream models to learn a
signal that mirrors a real growth-analytics problem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(
    n: int = config.N_CUSTOMERS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a synthetic customer-level acquisition dataset.

    Returns a DataFrame with the columns required by the project brief plus the
    derived binary target `is_high_value`.
    """
    rng = np.random.default_rng(random_state)

    channels = list(config.CHANNELS.keys())
    shares = np.array([config.CHANNELS[c]["share"] for c in channels])
    shares = shares / shares.sum()
    channel = rng.choice(channels, size=n, p=shares)

    devices = list(config.DEVICE_TYPES.keys())
    dev_p = np.array([config.DEVICE_TYPES[d] for d in devices])
    dev_p = dev_p / dev_p.sum()
    device_type = rng.choice(devices, size=n, p=dev_p)

    # Signup dates spread across a ~2 year window.
    start = np.datetime64("2023-01-01")
    day_offsets = rng.integers(0, 730, size=n)
    signup_date = start + day_offsets.astype("timedelta64[D]")

    # Vectorised per-channel parameter lookup.
    cac_mean = np.array([config.CHANNELS[c]["cac_mean"] for c in channel])
    cac_std = np.array([config.CHANNELS[c]["cac_std"] for c in channel])
    aov_mean = np.array([config.CHANNELS[c]["aov_mean"] for c in channel])
    aov_std = np.array([config.CHANNELS[c]["aov_std"] for c in channel])
    hv_prop = np.array([config.CHANNELS[c]["hv_propensity"] for c in channel])
    repeat_speed = np.array([config.CHANNELS[c]["repeat_speed"] for c in channel])
    dev_effect = np.array([config.DEVICE_HV_EFFECT[d] for d in device_type])

    acquisition_cost = np.clip(rng.normal(cac_mean, cac_std), 1.5, None).round(2)
    first_order_value = np.clip(rng.normal(aov_mean, aov_std), 5.0, None).round(2)

    # Days to second purchase: exponential-ish, faster for stickier channels.
    days_to_second_purchase = rng.gamma(
        shape=2.0, scale=repeat_speed / 2.0
    ).round().astype(int)
    days_to_second_purchase = np.clip(days_to_second_purchase, 1, 365)

    # Latent high-value score combines channel propensity, device, spend signal
    # and repeat speed, plus non-linear interactions and idiosyncratic noise.
    # The interactions (spend x speed, and a fast-repeat threshold effect) give
    # tree-based models a genuine edge over the linear baseline, mirroring real
    # behavioural data where loyalty emerges from combinations of signals.
    aov_z = (first_order_value - aov_mean) / (aov_std + 1e-9)
    repeat_z = (repeat_speed - days_to_second_purchase) / (repeat_speed + 1e-9)

    # Threshold / step effect: a very fast second purchase is a strong loyalty
    # signal, but only weakly informative once it is "slow enough".
    fast_repeat = (days_to_second_purchase <= 14).astype(float)

    # Interaction: high early spend *and* quick return compound into loyalty.
    spend_speed_interaction = np.clip(aov_z, -3, 3) * np.clip(repeat_z, -3, 3)

    latent = (
        -0.65
        + 3.0 * (hv_prop - 0.30)
        + dev_effect * 6.0
        + 0.55 * aov_z
        + 1.05 * repeat_z
        + 0.9 * fast_repeat
        + 0.6 * spend_speed_interaction
        + rng.normal(0, 0.5, size=n)
    )
    hv_probability = _sigmoid(latent)

    # Simulate lifetime revenue: high-value customers repeat and spend more.
    repeat_multiplier = np.where(
        rng.random(n) < hv_probability,
        rng.uniform(3.2, 7.5, size=n),   # loyal customers
        rng.uniform(0.6, 2.2, size=n),   # low-value / churn-risk
    )
    total_revenue_generated = np.clip(
        first_order_value * repeat_multiplier + rng.normal(0, 25, size=n),
        5.0,
        None,
    ).round(2)

    is_high_value = (
        total_revenue_generated >= config.HIGH_VALUE_REVENUE_THRESHOLD
    ).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],
            "signup_date": pd.to_datetime(signup_date),
            "acquisition_channel": channel,
            "acquisition_cost": acquisition_cost,
            "device_type": device_type,
            "first_order_value": first_order_value,
            "days_to_second_purchase": days_to_second_purchase,
            "total_revenue_generated": total_revenue_generated,
            "is_high_value": is_high_value,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, path=config.RAW_DATA_PATH) -> None:
    df.to_csv(path, index=False)


def load_or_generate(force: bool = False) -> pd.DataFrame:
    """Load the dataset from disk, generating it first if necessary."""
    if config.RAW_DATA_PATH.exists() and not force:
        return pd.read_csv(config.RAW_DATA_PATH, parse_dates=["signup_date"])
    df = generate_customers()
    save_dataset(df)
    return df


if __name__ == "__main__":
    data = generate_customers()
    save_dataset(data)
    rate = data["is_high_value"].mean()
    print(f"Generated {len(data):,} customers | high-value rate = {rate:.1%}")
