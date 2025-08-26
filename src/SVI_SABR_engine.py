
"""
SVI_SABR_engine.py
=========================
Main engine to iterate through a date range, assemble market data using handlers,
and calibrate the SVI-SABR model for each day.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math
from datetime import datetime
import scipy.optimize as spo
from scipy.integrate import trapezoid
from scipy.stats import norm

# Import our new data handlers and calculation modules
import data_handlers
import futures_curve as fcm

# --- Constants --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results_testing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PARAM_BOUNDS_SVI = [(1e-5, 10.0), (1e-5, 10.0), (-0.999, 0.999)]
ARB_CAP = 4.0
EXTRAPOLATION_FACTOR = 4.0
EXT_GRID_SIZE = 500

# --- Core SVI and Black-76 Calculations (preserved from old engine) --- #

def call_price_black76(F: float, K: float, sigma: float, T: float, r: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (norm.cdf(d1) * F - norm.cdf(d2) * K)

def implied_volatility_black76(price: float, F: float, K: float, T: float, r: float) -> float:
    def objective(sigma):
        return call_price_black76(F, K, sigma, T, r) - price
    try:
        return spo.brentq(objective, 1e-6, 5.0)
    except (ValueError, RuntimeError):
        return np.nan

def svi_sabr_vol(F: float, K: float, T: float, alpha: float, nu: float, rho: float) -> float:
    k = math.log(K / F)
    term1 = 1 + rho * (nu / alpha) * k
    term2 = math.sqrt((nu / alpha * k + rho)**2 + (1 - rho**2))
    w = 0.5 * alpha**2 * (term1 + term2)
    return math.sqrt(w / T)

def calibrate_svi_sabr(df: pd.DataFrame, F: float, T: float, r: float, initial_guess: list) -> dict:
    df_calls = df[df['type'] == 'c'].copy()
    df_calls['iv'] = df_calls.apply(lambda row: implied_volatility_black76(row['price'], F, row['strike'], T, r), axis=1)
    df_calls = df_calls.dropna(subset=['iv'])
    
    if df_calls.empty:
        return None # Cannot calibrate if no valid IVs

    strikes = df_calls.strike.astype(float).to_numpy()
    ivs = df_calls.iv.astype(float).to_numpy()
    weights = (df_calls.volume.astype(float) + df_calls.open_interest.astype(float))
    weights = (weights / weights.sum()) if weights.sum() > 0 else (np.ones_like(weights) / len(weights))

    def obj(x):
        sim = np.array([svi_sabr_vol(F, K, T, *x) for K in strikes])
        penalty = 1e6 if not (x[0] * x[1] * T * (1 + abs(x[2])) < ARB_CAP) else 0
        return float(np.sum(weights * (sim - ivs)**2)) + penalty

    res = spo.minimize(obj, initial_guess, bounds=PARAM_BOUNDS_SVI, method='L-BFGS-B')
    alpha, nu, rho = res.x
    iv_rmse = math.sqrt(res.fun)
    warn = '' if (alpha * nu * T * (1 + abs(rho)) < ARB_CAP) else 'ARB_FAIL'
    return {'alpha': alpha, 'nu': nu, 'rho': rho, 'iv_rmse': iv_rmse, 'warn_code': warn}

def calculate_implied_moments(F: float, T: float, r: float, alpha: float, nu: float, rho: float) -> dict:
    try:
        K_grid = np.linspace(F / EXTRAPOLATION_FACTOR, F * EXTRAPOLATION_FACTOR, EXT_GRID_SIZE * 2)
        k_grid = np.log(K_grid / F)
        C = np.array([call_price_black76(F, K, svi_sabr_vol(F, K, T, alpha, nu, rho), T, r) for K in K_grid])
        pdf = np.gradient(np.gradient(C, K_grid), K_grid)
        pdf[pdf < 0] = 0
        mass = trapezoid(pdf, K_grid)
        if mass > 1e-6: pdf = pdf / mass
        mean_k = trapezoid(k_grid * pdf, K_grid)
        var_k = trapezoid(((k_grid - mean_k)**2) * pdf, K_grid)
        if var_k < 1e-9: return {'implied_variance': 0.0, 'implied_skew': 0.0, 'implied_kurtosis': 0.0}
        std_k = np.sqrt(var_k)
        skew_k = trapezoid(((k_grid - mean_k)**3) * pdf, K_grid) / (std_k**3)
        kurt_k = trapezoid(((k_grid - mean_k)**4) * pdf, K_grid) / (std_k**4)
        return {'implied_variance': var_k, 'implied_skew': skew_k, 'implied_kurtosis': kurt_k}
    except Exception:
        return {'implied_variance': 0.0, 'implied_skew': 0.0, 'implied_kurtosis': 0.0}

# --- Main Processing Loop --- #

def process_daily_data(start_date: str, end_date: str, expiry_symbol: str, expiry_date_str: str):
    """
    Main loop to process market data for each day in a date range.
    """
    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='B'))
    expiry_date = pd.to_datetime(expiry_date_str)
    all_results = []
    last_params = [0.2, 0.5, -0.3] # Initial guess for SVI parameters

    # Get the full futures curve for the expiry to determine the underlying contract
    # This is a simplification; a more robust method would map option expiry to futures contract
    underlying_contract_symbol = 'CLV25' # Hardcoded for now based on user info

    for date in date_range:
        print(f"Processing {date.strftime('%Y-%m-%d')}...")
        try:
            # 1. Get data using handlers
            r = data_handlers.get_risk_free_rate(date)
            option_chain = data_handlers.get_option_chain(date, expiry_symbol)
            futures_contracts = data_handlers._load_futures_contracts()
            F = futures_contracts[underlying_contract_symbol]['close'].asof(date)
            T = (expiry_date - date).days / 365.0

            if T <= 0 or pd.isna(F):
                continue

            # 2. Calibrate SVI-SABR model
            cal_result = calibrate_svi_sabr(option_chain, F, T, r, initial_guess=last_params)
            if not cal_result:
                continue
            last_params = [cal_result['alpha'], cal_result['nu'], cal_result['rho']]

            # 3. Calculate moments and ATM IV using the core SVI parameters
            svi_params = {'alpha': cal_result['alpha'], 'nu': cal_result['nu'], 'rho': cal_result['rho']}
            moments = calculate_implied_moments(F, T, r, **svi_params)
            atm_iv = svi_sabr_vol(F, F, T, **svi_params)

            # 4. Consolidate and store results
            daily_result = {
                'date': date,
                'F': F, 'T': T, 'r': r,
                'atm_iv': atm_iv,
                **cal_result,
                **moments
            }
            all_results.append(daily_result)

        except (ValueError, IndexError) as e:
            print(f"  -> Skipping {date.strftime('%Y-%m-%d')}: {e}")
            continue

    # 5. Save final DataFrame
    results_df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / "daily_svi_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully processed {len(results_df)} days.")
    print(f"Results saved to {output_path}")

# --- CLI --- #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SVI-SABR calibration over a date range.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Based on user info, the expiry is fixed for this dataset
    EXPIRY_SYMBOL = 'clv5'
    EXPIRY_DATE = '2025-09-17'

    process_daily_data(args.start, args.end, EXPIRY_SYMBOL, EXPIRY_DATE)
