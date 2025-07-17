"""
vol_engine_hybrid.py
=========================
Reverted hybrid engine to SVI–SABR only (per your request), pending further refinements.
"""
from __future__ import annotations
import argparse, math, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import scipy.optimize as spo
from scipy.integrate import trapezoid
from scipy.stats import norm

# ───── parameters ─────────────────────────────────────────────────────────────
IV_WIDTH_CUTOFF   = 0.50
RISK_FREE_RATE    = 0.0
# SABR-SVI β=1 bounds: alpha, nu, rho
PARAM_BOUNDS_SVI  = [(1e-5, 5.0), (1e-5, 5.0), (-0.999, 0.999)]
# static-arbitrage cap for variance-space
ARB_CAP           = 4.0
# extrapolation grid for PDF mass
EXTRAPOLATION_FACTOR = 4.0
EXT_GRID_SIZE        = 500
# regex for date tokens
DATE_TOKEN_RE     = re.compile(r"\d{2}-\d{2}-\d{4}")

# ───── I/O utils ─────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

@lru_cache(maxsize=None)
def _fut_row(curve_dir: Path, date_tok: str) -> pd.Series:
    f = next(curve_dir.glob(f"*{date_tok}*.csv"))
    return normalise_columns(pd.read_csv(f)).iloc[1]


def forward_price(curve_dir: Path, date_tok: str, T: float) -> float:
    F = float(_fut_row(curve_dir, date_tok)["last"])
    return F * math.exp(RISK_FREE_RATE * T)

# ───── Black-76 call price ─────────────────────────────────────────────────────
def call_price_black76(F: float, K: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    d1 = (math.log(F/K) + 0.5 * sigma*sigma*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return norm.cdf(d1)*F - norm.cdf(d2)*K

# ───── SVI-SABR β=1 vol ────────────────────────────────────────────────────────
def svi_sabr_vol(F: float, K: float, T: float, alpha: float, nu: float, rho: float) -> float:
    k = math.log(K/F)
    term1 = 1 + rho * (nu/alpha) * k
    term2 = math.sqrt((nu/alpha * k + rho)**2 + (1 - rho*rho))
    w = 0.5 * alpha*alpha * (term1 + term2)
    return math.sqrt(w / T)

# ───── PDF mass via Breeden-Litzenberger ────────────────────────────────────────
def compute_pdf_mass(F: float, T: float, alpha: float, nu: float, rho: float) -> float:
    def vol_fn(K): return svi_sabr_vol(F, K, T, alpha, nu, rho)
    Kmin, Kmax = F/EXTRAPOLATION_FACTOR, F*EXTRAPOLATION_FACTOR
    Kgrid = np.linspace(Kmin, Kmax, EXT_GRID_SIZE)
    C = [call_price_black76(F, K, vol_fn(K), T) for K in Kgrid]
    pdf = np.gradient(np.gradient(C, Kgrid), Kgrid)
    return float(trapezoid(pdf, Kgrid))

# ───── SVI-SABR calibrator ─────────────────────────────────────────────────────
def calibrate_svi_sabr(df: pd.DataFrame, F: float, T: float) -> Dict[str, float]:
    strikes = df.strike.astype(float).to_numpy()
    # Use mid price as proxy for implied volatility. Needs proper IV calculation.
    vols = df.mid.astype(float).to_numpy() / F # Placeholder: needs proper IV calculation

    def obj(x):
        sim = np.array([svi_sabr_vol(F, K, T, *x) for K in strikes])
        return float(np.mean((sim - vols)**2))
    res = spo.minimize(obj, [0.2,0.5,-0.3], bounds=PARAM_BOUNDS_SVI, method='L-BFGS-B')
    alpha, nu, rho = res.x
    iv_rmse = math.sqrt(res.fun)
    mass    = compute_pdf_mass(F, T, alpha, nu, rho)
    warn    = '' if (alpha*nu*T*(1+abs(rho))<ARB_CAP and nu*nu*T*(1+abs(rho))<ARB_CAP) else 'ARB_FAIL'
    return {'alpha': alpha, 'nu': nu, 'rho': rho,
            'iv_rmse': iv_rmse, 'pdf_mass': mass, 'warn_code': warn}

# ───── main process ────────────────────────────────────────────────────────────
def process(opt_dir: Path, fut_dir: Path, expiry: str, out_csv: Path) -> None:
    exp_dt = datetime.strptime(expiry, '%Y-%m-%d')
    rows: List[Dict] = []
    for f in sorted(Path(opt_dir).glob('*_cleaned.csv')):
        date_tok = DATE_TOKEN_RE.search(f.name).group(0)
        T = (exp_dt - datetime.strptime(date_tok, '%m-%d-%Y')).days/365.0
        if T <= 0: continue
        df_s = pd.read_csv(f) # Read already cleaned data
        if df_s.empty: continue
        F = forward_price(Path(fut_dir), date_tok, T)
        rec = calibrate_svi_sabr(df_s, F, T)
        rec.update({'date': date_tok, 'model': 'SVI-SABR'})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows")

# ───── CLI ─────────────────────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument('opt_dir', type=Path)
    p.add_argument('--fut_dir', type=Path, default=Path('Data/Futures_data'))
    p.add_argument('--expiry', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    process(args.opt_dir, args.fut_dir, args.expiry, args.out)

if __name__=='__main__':
    cli()