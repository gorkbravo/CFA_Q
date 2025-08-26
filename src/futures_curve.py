"""
 futures_curve.py - Calculation module for term structure features.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# Note: The _expiry and MONTH_CODES logic would be needed if we were dynamically
# determining expiries from contract codes. For now, we assume expiries are known.

def calculate_term_structure_features(daily_curve_df: pd.DataFrame, front_month_expiry: pd.Timestamp) -> dict:
    """
    Computes the Curve-Term-Structure Index (CTSI) and related metrics.

    Args:
        daily_curve_df: A DataFrame for a single day with columns ['contract', 'price', 'expiration'].
        front_month_expiry: The expiration date of the true front-month contract.

    Returns:
        A dictionary containing the CTSI and its component features.
    """
    df = daily_curve_df.sort_values("expiration").reset_index(drop=True)
    if df.empty:
        return {}

    try:
        front_price = df[df["expiration"] == front_month_expiry]["price"].iloc[0]
    except IndexError:
        # Fallback if the exact front month is not in the curve for some reason
        front_price = df.iloc[0]["price"]
        front_month_expiry = df.iloc[0]["expiration"]

    # Define time anchors for spread calculations
    anchors = {"1M": 30, "3M": 90, "1Y": 365}
    spreads = {}
    for tag, days in anchors.items():
        time_delta = pd.Timedelta(days=days)
        mask = df["expiration"].sub(front_month_expiry).ge(time_delta)
        # Get the first contract that meets the time delta requirement
        idx = mask.idxmax() if mask.any() else len(df) - 1
        price = df.loc[idx, "price"]
        spreads[tag] = (price - front_price) / front_price

    # Normalize spreads based on typical volatility at that tenor
    sev = {
        "1M": spreads["1M"] / 0.01,
        "3M": spreads["3M"] / 0.03,
        "1Y": spreads["1Y"] / 0.05
    }
    norm = {k: np.tanh(v) for k, v in sev.items()}

    # Calculate persistence and its impact on the index
    persistence = np.mean(np.diff(df["price"]) < 0) if len(df["price"]) > 1 else 0.5
    impact = (2 * persistence - 1) * min(abs(spreads["1Y"] / 0.05), 1)

    # Calculate the final CTSI
    ctsi = np.clip(
        0.25 * norm["1M"] + 0.35 * norm["3M"] + 0.25 * norm["1Y"] + 0.15 * impact, -1, 1
    )

    features = {
        "term_structure_index": round(ctsi, 4),
        "market_state": "Contango" if ctsi >= 0 else "Backwardation",
        "month1_spread": spreads["1M"],
        "month3_spread": spreads["3M"],
        "year1_spread": spreads["1Y"],
        "norm_month1": norm["1M"],
        "norm_month3": norm["3M"],
        "norm_year1": norm["1Y"],
        "persistence": persistence,
        "persistence_impact": impact,
    }
    
    return {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in features.items()}