"""
vol_engine_v6.py
=========================
Adapted from vol_engine_v5.py: implements full analytic Breeden–Litzenberger second derivative for risk-neutral PDF,
adds static-arb check and filters out non-positive strikes to avoid domain errors.
"""
from __future__ import annotations
import argparse, math, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.optimize as spo
from scipy.integrate import trapezoid
from scipy.stats import norm

# ───── parameters ─────────────────────────────────────────────────────────────
IV_WIDTH_CUTOFF = 0.50
RISK_FREE_RATE = 0.0
# SABR-SVI β=1 bounds: alpha, nu, rho
PARAM_BOUNDS = [(1e-5, 5.0), (1e-5, 5.0), (-0.999, 0.999)]
DATE_TOKEN_RE = re.compile(r"\d{2}-\d{2}-\d{4}")

# tail extrapolation settings
EXTRAPOLATION_FACTOR = .0  # extend grid to [K_min/2, K_max*2]
EXT_GRID_SIZE = 500         # number of points in extended grid

# ───── utils ──────────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = df.columns.str.strip().str.lower(); return df

@lru_cache(maxsize=None)
def _fut_row(curve_dir: Path, date_tok: str) -> pd.Series:
    f = next(curve_dir.glob(f"*{date_tok}*.csv"))
    return normalise_columns(pd.read_csv(f)).iloc[1]


def forward_price(curve_dir: Path, date_tok: str, T: float) -> float:
    F = float(_fut_row(curve_dir, date_tok)["last"])
    return F * math.exp(RISK_FREE_RATE * T)

# ───── SVI-SABR β=1 total variance & vol ────────────────────────────────────────
def _svi_sabr_w(k: float, alpha: float, nu: float, rho: float) -> float:
    term1 = 1 + rho * (nu / alpha) * k
    term2 = math.sqrt((nu/alpha * k + rho)**2 + (1 - rho**2))
    return (alpha**2 / 2) * (term1 + term2)


def svi_sabr_vol(F: float, K: float, T: float, alpha: float, nu: float, rho: float) -> float:
    k = math.log(K / F)
    w = _svi_sabr_w(k, alpha, nu, rho)
    return math.sqrt(w / T)

# ───── Breeden–Litzenberger PDF ─────────────────────────────────────────────────
def breeden_pdf(
    F: float,
    K: np.ndarray,
    T: float,
    alpha: float,
    nu: float,
    rho: float
) -> np.ndarray:
    # [Implementation unchanged from v5]
    sigma = np.zeros_like(K)
    dsig_dK = np.zeros_like(K)
    d2sig_dK2 = np.zeros_like(K)

    for i, Ki in enumerate(K):
        if Ki <= 0: continue
        k = math.log(Ki / F)
        term2 = math.sqrt((nu/alpha * k + rho)**2 + (1 - rho**2))
        w = (alpha**2 / 2) * (1 + rho*(nu/alpha)*k + term2)
        dw_dk = (alpha * nu / 2) * (rho + (nu/alpha * k + rho)/term2)
        d2w_dk2 = (nu**2 * (1 - rho**2)) / (2 * term2**3)
        sigma_i = math.sqrt(w / T)
        sigma[i] = sigma_i
        dsig_dk = dw_dk / (2 * sigma_i * T)
        dsig_dK[i] = dsig_dk / Ki
        d2sig_dk2 = d2w_dk2 / (2 * sigma_i * T) - (dw_dk * dsig_dk) / (2 * sigma_i**2 * T)
        d2sig_dK2[i] = (d2sig_dk2 - dsig_dk) / (Ki * Ki)

    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vega = F * norm.pdf(d1) * np.sqrt(T)
    vomma = vega * d1 * d2 / sigma
    C_KK = norm.pdf(d2) / (sigma * K * np.sqrt(T))
    C_Ksigma = vega * d1 / (K * sigma * np.sqrt(T))

    pdf = (
        C_KK
        + 2 * C_Ksigma * dsig_dK
        + vomma * dsig_dK**2
        + vega * d2sig_dK2
    ) * math.exp(RISK_FREE_RATE * T)
    return pdf

# ───── calibration and bfly arb check ─────────────────────────
def calibrate_svi_sabr(df: pd.DataFrame, F: float, T: float) -> Dict[str, float]:
    
    strikes = df.strike.astype(float).to_numpy()
    # Use mid price as proxy for implied volatility. Needs proper IV calculation.
    vols = df.mid.astype(float).to_numpy() / F # Placeholder: needs proper IV calculation

    mask = strikes > 0
    strikes = strikes[mask]
    vols = vols[mask]
    
    def obj(x: List[float]) -> float:
        a, n, r = x
        model = np.array([svi_sabr_vol(F, K, T, a, n, r) for K in strikes])
        return np.mean((model - vols)**2)

    x0 = [0.2, 0.5, -0.3]
    res = spo.minimize(obj, x0, bounds=PARAM_BOUNDS, method='L-BFGS-B')
    alpha, nu, rho = res.x
    rmse = math.sqrt(res.fun)

    # stat arb check 
    cond1 = alpha * nu * T * (1 + abs(rho)) < 4
    cond2 = nu**2 * T * (1 + abs(rho))   < 4
    arb_free = cond1 and cond2
    warn_code = '' if (res.success and arb_free) else 'ARB_FAIL'

    
    K_min, K_max = strikes.min(), strikes.max()
    K_ext = np.linspace(K_min/EXTRAPOLATION_FACTOR, K_max*EXTRAPOLATION_FACTOR, EXT_GRID_SIZE)

    
    pdf_vals = breeden_pdf(F, K_ext, T, alpha, nu, rho)
    mass = trapezoid(pdf_vals, K_ext)

    
    band = np.nan

    return {
        'alpha': alpha, 'nu': nu, 'rho': rho,
        'iv_rmse': rmse, 'pdf_mass': mass,
        'band_pct_within': band,
        'bid_iv_mean': np.nan, 'ask_iv_mean': np.nan,
        'iv_band_p95': np.nan,
        'warn_code': warn_code
    }

# ───── driver ──────────────────────────────────────────────────────────────────
def token(name: str) -> str:
    m = DATE_TOKEN_RE.search(name)
    if not m: raise ValueError(name)
    return m.group(0)


def yearfrac(tok: str, expiry: datetime) -> float:
    return (expiry - datetime.strptime(tok, "%m-%d-%Y")).days / 365.0


def process(opt_dir: Path, fut_dir: Path, expiry: str, out_csv: Path, show: bool, pdf_dir: Path | None = None) -> None:
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    rows = []
    for f in sorted(opt_dir.glob('*_cleaned.csv')):
        dt = token(f.name)
        T = yearfrac(dt, expiry_dt)
        if T <= 0: continue
        df = pd.read_csv(f) # Read already cleaned data
        if df.empty: continue
        F = forward_price(fut_dir, dt, T)
        res = calibrate_svi_sabr(df, F, T)
        res['date'] = dt
        rows.append(res)
    pd.DataFrame(rows)[['date','alpha','nu','rho','iv_rmse','pdf_mass','band_pct_within','warn_code']].to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} slices to {out_csv}")


def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('opt_dir', type=Path)
    p.add_argument('--fut_dir', type=Path, default=Path('Data/Futures_data'))
    p.add_argument('--expiry', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--show', action='store_true')
    p.add_argument('--save-pdf', type=Path)
    args = p.parse_args()
    if args.save_pdf: args.save_pdf.mkdir(exist_ok=True)
    process(args.opt_dir, args.fut_dir, args.expiry, args.out, args.show, args.save_pdf)

if __name__ == '__main__':
    cli()