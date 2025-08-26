"""
feature_engine.py
=========================
Module for generating time-series features from the daily data.
"""

import pandas as pd

def add_lagged_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """Adds lagged columns to a DataFrame."""
    df_out = df.copy()
    for col in columns:
        for lag in lags:
            df_out[f'{col}_lag_{lag}'] = df_out[col].shift(lag)
    return df_out

def add_moving_average_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    """Adds moving average columns to a DataFrame."""
    df_out = df.copy()
    for col in columns:
        for window in windows:
            df_out[f'{col}_ma_{window}'] = df_out[col].rolling(window=window).mean()
    return df_out

def add_momentum_features(df: pd.DataFrame, columns: list[str], periods: list[int]) -> pd.DataFrame:
    """Adds momentum (difference) columns to a DataFrame."""
    df_out = df.copy()
    for col in columns:
        for period in periods:
            df_out[f'{col}_mom_{period}'] = df_out[col].diff(periods=period)
    return df_out

def generate_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a predefined set of feature engineering steps to the daily data.

    Args:
        df: The daily SVI analysis DataFrame, with a DatetimeIndex.

    Returns:
        The DataFrame with added time-series features.
    """
    # Define the core features to engineer
    features_to_engineer = [
        'atm_iv',
        'rho',
        'implied_variance',
        'implied_skew',
        'implied_kurtosis'
    ]

    # Generate features with smaller windows suitable for limited data
    df = add_lagged_features(df, features_to_engineer, lags=[1])
    df = add_moving_average_features(df, features_to_engineer, windows=[2])
    df = add_momentum_features(df, features_to_engineer, periods=[1])

    return df
