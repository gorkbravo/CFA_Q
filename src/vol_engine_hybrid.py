"""
vol_engine_hybrid.py
=========================
Hybrid volatility engine: per-slice calibration using SVI–SABR (short-dated) and two-step variance-space SSVI (long-dated).
"""
from __future__ import annotations
import argparse, math, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import scipy.optimize as spo
from scipy.integrate import trapezoid
from scipy.stats import norm


IV_WIDTH_CUTOFF   = 0.50
RISK_FREE_RATE    = 0.0

PARAM_BOUNDS_SVI  = [(1e-5, 5.0), (1e-5, 5.0), (-0.999, 0.999)]   #HELP :( 

PARAM_BOUNDS_SSVI = [(1e-8, None), (-0.999, 0.999)] 

ARB_CAP           = 4.0

ATM_K0            = 0.05

T_SWITCH          = 30.0 / 365.0

EXTRAPOLATION_FACTOR = 4.0
EXT_GRID_SIZE        = 500

DATE_TOKEN_RE     = re.compile(r"\d{2}-\d{2}-\d{4}")

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

# ───── SSVI vol helpers ─────────────────────────────────────────────────────────
def ssvi_total_variance(k: np.ndarray, theta: float, phi: float, rho: float) -> np.ndarray:
    inner = phi * k + rho
    return 0.5 * theta * (1 + rho*phi*k + np.sqrt(inner*inner + (1 - rho*rho)))

def ssvi_vol(F: float, K: float, T: float, theta: float, phi: float, rho: float) -> float:
    k = math.log(K/F)
    w = ssvi_total_variance(np.array([k]), theta, phi, rho)[0]
    return math.sqrt(w / T)

# ───── slice filter ────────────────────────────────────────────────────────────
def filter_slice(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)
    df = df[df.type.str.startswith('c')]
    df[['bid','ask']] = df[['bid','ask']].apply(pd.to_numeric, errors='coerce')
    df = df[df.strike.astype(float) > 0]
    mid = (df.bid + df.ask)/2
    ok  = (df.bid>0) & (df.ask>df.bid) & (((df.ask-df.bid)/(mid+1e-12)) <= IV_WIDTH_CUTOFF)
    return df[ok] if ok.sum() >= 8 else df[df.bid>0]

# ───── PDF mass via Breeden-Litzenberger ────────────────────────────────────────
def compute_pdf_mass(
    F: float, T: float, model: str,
    alpha: Optional[float]=None, nu: Optional[float]=None, rho: Optional[float]=None,
    theta: Optional[float]=None, phi: Optional[float]=None
) -> float:
    def vol_fn(K):
        if model=='SVI': return svi_sabr_vol(F, K, T, alpha, nu, rho)
        return ssvi_vol(F, K, T, theta, phi, rho)
    Kmin, Kmax = F/EXTRAPOLATION_FACTOR, F*EXTRAPOLATION_FACTOR
    Kgrid = np.linspace(Kmin, Kmax, EXT_GRID_SIZE)
    C = [call_price_black76(F, K, vol_fn(K), T) for K in Kgrid]
    pdf = np.gradient(np.gradient(C, Kgrid), Kgrid)
    return float(trapezoid(pdf, Kgrid))

# ───── calibrators ────────────────────────────────────────────────────────────
def calibrate_svi_sabr(df: pd.DataFrame, F: float, T: float) -> Dict[str, float]:
    strikes = df.strike.astype(float).to_numpy()
    vols    = df.impliedvolatility.astype(float).to_numpy()
    def obj_svi(x: List[float]) -> float:
        sim = np.array([svi_sabr_vol(F, K, T, *x) for K in strikes])
        return float(np.mean((sim - vols)**2))
    res = spo.minimize(obj_svi, [0.2, 0.5, -0.3], bounds=PARAM_BOUNDS_SVI, method='L-BFGS-B')
    alpha, nu, rho = res.x
    iv_rmse = math.sqrt(res.fun)
    mass    = compute_pdf_mass(F, T, 'SVI', alpha=alpha, nu=nu, rho=rho)
    warn    = '' if (alpha*nu*T*(1+abs(rho))<ARB_CAP and nu*nu*T*(1+abs(rho))<ARB_CAP) else 'ARB_FAIL'
    return {'alpha': alpha, 'nu': nu, 'rho': rho,
            'iv_rmse': iv_rmse, 'pdf_mass': mass, 'warn_code': warn}


def calibrate_ssvi_two_step(df: pd.DataFrame, F: float, T: float) -> Dict[str, float]:
    strikes = df.strike.astype(float).to_numpy()
    ivs     = df.impliedvolatility.astype(float).to_numpy()
    k       = np.log(strikes / F)
    w_mkt   = ivs * ivs * T

    atm_mask = np.abs(k) < ATM_K0
    # Step1: ATM
    theta0 = float(np.mean(w_mkt[atm_mask]))

    # Step2: wings
    def obj_phi(x: List[float]) -> float:
        phi, rho = x
        w_hat = ssvi_total_variance(k[~atm_mask], theta0, phi, rho)
        return float(np.mean((w_hat - w_mkt[~atm_mask])**2))
    cons = {'type':'ineq', 'fun': lambda x: ARB_CAP - theta0*x[0]*(1+abs(x[1]))}
    res2 = spo.minimize(obj_phi, [0.5, 0.0], bounds=PARAM_BOUNDS_SSVI, constraints=cons, method='SLSQP')
    phi, rho = res2.x

    # full-slice
    w_all   = ssvi_total_variance(k, theta0, phi, rho)
    iv_hat  = np.sqrt(w_all / T)
    iv_rmse = float(np.sqrt(np.mean((iv_hat - ivs)**2)))
    mass    = compute_pdf_mass(F, T, 'SSVI', theta=theta0, phi=phi, rho=rho)
    warn    = '' if (theta0*phi*T*(1+abs(rho))<ARB_CAP and phi*phi*T*(1+abs(rho))<ARB_CAP) else 'ARB_FAIL'
    return {'alpha': np.nan, 'nu': np.nan, 'rho': rho,
            'iv_rmse': iv_rmse, 'pdf_mass': mass, 'warn_code': warn,
            'theta': theta0, 'phi': phi}

# ───── main hybrid process ────────────────────────────────────────────────────
def process(opt_dir: Path, fut_dir: Path, expiry: str, out_csv: Path,
            show: bool, t_switch: float = T_SWITCH) -> None:
    exp_dt = datetime.strptime(expiry, '%Y-%m-%d')
    out: List[Dict[str, float]] = []
    for f in sorted(Path(opt_dir).glob('*_cleaned.csv')):
        date_tok = DATE_TOKEN_RE.search(f.name).group(0)
        T = (exp_dt - datetime.strptime(date_tok, '%m-%d-%Y')).days / 365.0
        if T <= 0: continue
        df_s = filter_slice(pd.read_csv(f))
        if df_s.empty: continue
        F = forward_price(Path(fut_dir), date_tok, T)
        if T <= t_switch:
            rec = calibrate_svi_sabr(df_s, F, T)
            rec['model'] = 'SVI-SABR'
        else:
            rec = calibrate_ssvi_two_step(df_s, F, T)
            rec['model'] = 'SSVI'
        rec['date'] = date_tok
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows to {out_csv}")

# ───── CLI ─────────────────────────────────────────────────────────────────────
def cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('opt_dir', type=Path)
    p.add_argument('--fut_dir', type=Path, default=Path('Data/Futures_data'))
    p.add_argument('--expiry', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--show', action='store_true')
    p.add_argument('--t-switch', type=float, default=T_SWITCH)
    args = p.parse_args()
    process(args.opt_dir, args.fut_dir, args.expiry, args.out, args.show, args.t_switch)

if __name__ == '__main__':
    cli()
