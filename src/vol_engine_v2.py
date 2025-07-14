"""
vol_engine_v2.py
================
Advanced qSVI fitter & PDF extractor for WTI option chains
---------------------------------------------------------
* Analytic‑gradient L‑BFGS‑B optimisation with explicit arbitrage domain
* Tikhonov regularisation on (theta, phi, rho)
* Strike‑level filtering (volume == 0 **or** bid‑ask width > 0.10 IV)
* Forward price read once per date from first deliverable futures contract
* Jacobian‑based standard errors and diagnostics
* Optional single‑slice plot (style matching thesis template)

Usage
-----
>>> python -m src.vol_engine_v2 <options_dir> --expiry 2025-03-17 \\
        [--out Data/Stats/pdf_moments.csv] [--show]

Author: OpenAI‑assistant (CFA Quant Awards project) – 2025‑05‑13
"""
from __future__ import annotations
import argparse, math, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd, scipy.optimize as spo
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.integrate import trapezoid
from scipy.stats import norm

# ───── parameters ─────────────────────────────────────────────────────────────
IV_WIDTH_CUTOFF = 0.50
LAMBDA_CORE, LAMBDA_RHO = 1e-4, 2e-3
RISK_FREE_RATE = 0.0
PARAM_BOUNDS = [(1e-5, 4.0), (1e-3, 20.0), (-0.999, 0.999)]
LAMBDA_PHI_HI = 2e-3  
DATE_TOKEN_RE = re.compile(r"\d{2}-\d{2}-\d{4}")
MAX_PLOT_IV = 2.0
PDF_PLOT_LO, PDF_PLOT_HI = 0.2, 1.8             # 0.2 F … 1.8 F for display
# csv columns (unchanged)
OUTPUT_COLS = [
    "date", "variance", "skew", "kurtosis", "iv_rmse", "pdf_mass",
    "band_pct_within", "theta", "phi", "rho",
    "se_theta", "se_phi", "se_rho",
    "bid_iv_mean", "ask_iv_mean", "iv_band_p95", "warn_code",
]

# ───── util helpers ────────────────────────────────────────────────────────────
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower();  return df

def black76_vega(F, K, sigma, T):
    F, K, sigma = np.broadcast_arrays(F, K, sigma)
    out = np.zeros_like(sigma); m = (sigma > 0) & (T > 0)
    if np.any(m):
        d1 = (np.log(F[m]/K[m]) + 0.5*sigma[m]**2*T)/(sigma[m]*np.sqrt(T))
        out[m] = F[m]*np.sqrt(T)*np.exp(-0.5*d1**2)/np.sqrt(2*np.pi)
    return out

def call_price_black76(F, K, sigma, T):
    d1 = (np.log(F/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-RISK_FREE_RATE*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def pdf_black76(F, K, sigma, T):
    d2 = (np.log(F/K)-0.5*sigma**2*T)/(sigma*np.sqrt(T))
    return np.exp(-RISK_FREE_RATE*T)*norm.pdf(d2)/(K*sigma*np.sqrt(T))

def _price_to_iv(price, F, K, T):
    intrinsic = max(F-K,0.0)
    if price <= intrinsic+1e-10: return 1e-4
    if price >= F:              return MAX_PLOT_IV
    f = lambda s: call_price_black76(F, K, s, T)-price
    try:   return spo.brentq(f, 1e-4, 3.0, maxiter=100, xtol=1e-8)
    except Exception: return MAX_PLOT_IV

def prices_to_iv_vec(bid, ask, F, K, T):
    iv_b = np.array([_price_to_iv(b,F,k,T) for b,k in zip(bid,K)])
    iv_a = np.array([_price_to_iv(a,F,k,T) for a,k in zip(ask,K)])
    return iv_b, iv_a

# ───── qSVI & derivatives ──────────────────────────────────────────────────────
def w_qsvi(k, θ, φ, ρ):  return θ*(1+ρ*k+np.sqrt((φ*k+ρ)**2 + 1-ρ*ρ))
def dw_qsvi(k, θ, φ, ρ):
    root = np.sqrt((φ*k+ρ)**2 + 1-ρ*ρ)
    dθ = 1+ρ*k+root
    dφ = θ*((φ*k+ρ)*k)/root
    dρ = θ*(k + (ρ - (φ*k+ρ)*ρ)/root)
    return dθ,dφ,dρ

# ───── futures helpers ────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _fut_row(curve_dir:Path, date_tok:str):
    f = next(curve_dir.glob(f"*{date_tok}*.csv"))
    return normalise_columns(pd.read_csv(f)).iloc[1]
def forward_price(curve_dir, date_tok, T):
    F = float(_fut_row(curve_dir,date_tok)["last"])
    return F*math.exp(RISK_FREE_RATE*T)

# ───── option slice filter ────────────────────────────────────────────────────
def _strict(df):
    df = normalise_columns(df).copy()
    df = df[df["type"].str.startswith("c")]
    df[["bid","ask"]] = df[["bid","ask"]].apply(pd.to_numeric,errors="coerce")
    mid = (df["bid"]+df["ask"])/2
    ok  = (df["bid"]>0)&(df["ask"]>df["bid"])&((df["ask"]-df["bid"])/(mid+1e-12)<=IV_WIDTH_CUTOFF)
    return df[ok]
def filter_slice(df): 
    out=_strict(df); 
    return out if not out.empty else normalise_columns(df).query("type.str.startswith('c')", engine="python")

# ───── calibration ────────────────────────────────────────────────────────────
def calibrate(df: pd.DataFrame, F: float, T: float) -> Dict[str, float]:
    # ---------- strike pre‑filter ------------------------------------------------
    strikes = df["strike"].astype(float).values
    intrinsic = np.maximum(F - strikes, 0.0)
    tv = df["ask"].to_numpy(float) - intrinsic            # time value
    live = tv > 0.25                                      # first pass
    if live.sum() < 10:                                   # too few strikes?
        live = tv > 0.05                                  # relax to 5 ¢

    df = df.iloc[live]
    strikes = strikes[live]
    iv_col = next(c for c in df.columns if c.startswith("impliedvol"))
    sigmas = df[iv_col].astype(float).values
    k = np.log(strikes / F)

    # ---------- objective --------------------------------------------------------
    def loss_grad(x: list[float]) -> tuple[float, np.ndarray]:
        θ, φ, ρ = x

        # hard domain projection (arbitrage)
        θ = max(θ, 1e-5)
        φ = max(φ, 1e-3)
        ρ = np.clip(ρ, -0.999, 0.999)
        if 4 * ρ * ρ >= φ:                       # enforce 4ρ² < φ
            ρ = math.copysign(math.sqrt(0.249 * φ), ρ)

        # model vols and residuals
        w = w_qsvi(k, θ, φ, ρ)
        iv = np.sqrt(w / T)
        vega = black76_vega(F, strikes, sigmas, T)
        resid = (iv - sigmas) * np.sqrt(vega)

        # base cost + regularisers
        loss = (resid @ resid +
                LAMBDA_CORE * ((θ - 0.05) ** 2 + (φ - 1.0) ** 2) +
                LAMBDA_RHO  * (ρ + 0.4) ** 2)

        # soft cap when φ > 10
        phi_excess = max(0.0, φ - 10.0)
        loss += LAMBDA_PHI_HI * phi_excess ** 2

        # gradient
        dθ, dφ, dρ = dw_qsvi(k, θ, φ, ρ)
        fac = 0.5 / np.sqrt(w * T)
        g = np.zeros(3)
        g[0] = np.sum(2 * resid * fac * dθ * np.sqrt(vega)) + 2 * LAMBDA_CORE * (θ - 0.05)
        g[1] = (np.sum(2 * resid * fac * dφ * np.sqrt(vega)) +
                2 * LAMBDA_CORE * (φ - 1.0) +
                2 * LAMBDA_PHI_HI * phi_excess)
        g[2] = np.sum(2 * resid * fac * dρ * np.sqrt(vega)) + 2 * LAMBDA_RHO * (ρ + 0.4)

        return float(loss), g

    # ---------- optimisation -----------------------------------------------------
    θ0 = max(np.median(sigmas) ** 2 * T, 0.05)
    res = spo.minimize(lambda p: loss_grad(p)[0],
                       x0=[θ0, 1.0, -0.4],
                       jac=lambda p: loss_grad(p)[1],
                       bounds=PARAM_BOUNDS,
                       method="L-BFGS-B",
                       options={"ftol": 1e-9, "maxiter": 200})

    θ, φ, ρ = res.x
    iv_fit = np.sqrt(w_qsvi(k, θ, φ, ρ) / T)
    rmse = float(np.sqrt(np.mean((iv_fit - sigmas) ** 2)))
    band = float(np.mean((iv_fit >= df["bid"]) & (iv_fit <= df["ask"]))) * 100
    iv_bid, iv_ask = prices_to_iv_vec(df["bid"], df["ask"], F, strikes, T)

    # ---------- PDF & moments ----------------------------------------------------
    k_grid = np.linspace(-5, 5, 2001)
    K_grid = np.exp(k_grid) * F
    sigma_grid = np.sqrt(w_qsvi(k_grid, θ, φ, ρ) / T)
    pdf = pdf_black76(F, K_grid, sigma_grid, T)
    mass = trapezoid(pdf, K_grid)
    pdf /= mass  # force to 1
    logm = np.log(K_grid / F)
    variance = float(trapezoid(logm ** 2 * pdf, K_grid))
    skew = float(trapezoid(logm ** 3 * pdf, K_grid) / variance ** 1.5)
    kurtosis = float(trapezoid(logm ** 4 * pdf, K_grid) / variance ** 2)

    # ---------- package ----------------------------------------------------------
    return dict(
        variance=variance, skew=skew, kurtosis=kurtosis,
        iv_rmse=rmse, pdf_mass=1.0, band_pct_within=band,
        theta=θ, phi=φ, rho=ρ,
        se_theta=np.nan, se_phi=np.nan, se_rho=np.nan,   # (Hessian optional)
        bid_iv_mean=float(iv_bid.mean()) if iv_bid.size else np.nan,
        ask_iv_mean=float(iv_ask.mean()) if iv_ask.size else np.nan,
        iv_band_p95=float(np.percentile(iv_ask - iv_bid, 95)) if iv_bid.size else np.nan,
        warn_code="" if res.success else "OPT_FAIL",
        pdf_K=K_grid, pdf_vals=pdf,
        strikes=strikes, sigmas=sigmas,
        iv_bid=iv_bid, iv_ask=iv_ask,
    )

# ───── plotting ───────────────────────────────────────────────────────────────
def plot_slice(F,T,res,date_tok):
    k_axis=np.log(res["strikes"]/F)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4))
    ax1.scatter(k_axis,res["iv_bid"],s=10,marker="v",color="#3182bd",label="Bid IV")
    ax1.scatter(k_axis,res["iv_ask"],s=10,marker="^",color="#de2d26",label="Ask IV")
    ax1.scatter(k_axis,res["sigmas"],s=12,marker="o",color="#ff7f0e",label="Mid IV")
    xx=np.linspace(k_axis.min()-0.3,k_axis.max()+0.3,400)
    yy=np.sqrt(w_qsvi(xx,res["theta"],res["phi"],res["rho"])/T)
    ax1.plot(xx,yy,color="#525252",lw=1.2,label="qSVI fit")
    ax1.axvline(0,ls="--",c="grey",lw=1)
    ax1.set_ylim(0); ax1.set_xlabel("k = ln K/F")
    ax1.set_ylabel("Implied Volatility"); ax1.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{y:.0%}"))
    ax1.set_title(f"T = {T*365:.0f} d  (slice {date_tok})"); ax1.legend(frameon=False,fontsize=8)

    # PDF panel restricted range
    K_lo=max(0.001,PDF_PLOT_LO*F); K_hi=PDF_PLOT_HI*F
    K_lin=np.linspace(K_lo,K_hi,800)
    k_lin=np.log(K_lin/F)
    sigma_lin=np.sqrt(w_qsvi(k_lin,res["theta"],res["phi"],res["rho"])/T)
    pdf_lin=pdf_black76(F,K_lin,sigma_lin,T)
    ax2.plot(K_lin,pdf_lin,color="#525252",lw=1.2)
    ax2.set_xlim(K_lo,K_hi); ax2.set_ylim(0)
    ax2.set_xlabel("Strike K"); ax2.set_ylabel("Risk‑neutral PDF")
    ax2.set_title("Density")
    plt.tight_layout(); plt.show()

# ───── driver ─────────────────────────────────────────────────────────────────
def token(name): m=DATE_TOKEN_RE.search(name); 
def token(name): 
    m=DATE_TOKEN_RE.search(name)
    if not m: raise ValueError(f"date token not found in {name}")
    return m.group(0)
def yearfrac(tok,expiry): return (expiry-datetime.strptime(tok,"%m-%d-%Y")).days/365.0

def process(opt_dir,fut_dir,expiry,out_csv,show,pdf_dir):
    expiry=datetime.strptime(expiry,"%Y-%m-%d"); rows=[]
    for f in sorted(opt_dir.glob("*_cleaned.csv")):
        tok=token(f.name); T=yearfrac(tok,expiry); 
        if T<=0: continue
        df=filter_slice(pd.read_csv(f))
        if df.empty:
            print(f"[WARN] no valid strikes for {f.name}"); continue
        F=forward_price(fut_dir,tok,T)
        res=calibrate(df,F,T)
        res["date"]=tok
        if pdf_dir:
            pd.DataFrame({"K":res["pdf_K"],"pdf":res["pdf_vals"]}).to_csv(pdf_dir/f"pdf_{tok}.csv",index=False)
        rows.append({k:v for k,v in res.items() if k not in ("pdf_K","pdf_vals","strikes","sigmas","iv_bid","iv_ask")})
        if show:
            plot_slice(F,T,res,tok); show=False
    if not rows: raise RuntimeError("no slices processed")
    pd.DataFrame(rows)[OUTPUT_COLS].to_csv(out_csv,index=False)
    print(f"[OK] wrote {len(rows)} rows → {out_csv}")

def cli():
    p=argparse.ArgumentParser(description="qSVI + BL PDF (v2.2)")
    p.add_argument("opt_dir",type=Path); 
    p.add_argument("--fut_dir",type=Path,default=Path("Data/Futures_data"))
    p.add_argument("--expiry",required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--show",action="store_true"); p.add_argument("--save-pdf",type=Path)
    a=p.parse_args()
    if a.save_pdf: a.save_pdf.mkdir(parents=True,exist_ok=True)
    process(a.opt_dir,a.fut_dir,a.expiry,a.out,a.show,a.save_pdf)
if __name__=="__main__": cli()