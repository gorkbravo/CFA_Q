# src/vol_engine.py  – v2.3.1  (vol‑space first, working demo)

from __future__ import annotations
from pathlib import Path
import re, numpy as np, pandas as pd
from scipy.optimize import minimize, root_scalar
from scipy.stats   import norm
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parents[1]
FUTURES_DIR = ROOT / "Data" / "Futures_data"
STATS_DIR   = ROOT / "Data" / "Stats"; STATS_DIR.mkdir(exist_ok=True)

DATE_RE = re.compile(r"(\d{2}-\d{2}-\d{4})")        # mm‑dd‑yyyy
UND_RE  = re.compile(r"(cl[a-z]\d{2})", re.I)

# ---------- Black‑76 helpers -------------------------------------------
def _bs_call(F, K, T, σ, r=0.0):
    σ = np.maximum(σ, 1e-8)
    d1 = (np.log(F/K) + 0.5*σ**2*T) / (σ*np.sqrt(T))
    d2 = d1 - σ*np.sqrt(T)
    return np.exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def _imp_vol(F, K, T, price, r=0.0):
    f = lambda σ: _bs_call(F, K, T, σ, r) - price
    try:
        return root_scalar(f, bracket=[1e-4, 5.0], method="brentq").root
    except ValueError:
        return np.nan

# ---------- qSVI --------------------------------------------------------
def _qtot(k, θ, φ, ρ):  return 0.5*θ*(1 + ρ*φ*k + np.sqrt((φ*k+ρ)**2 + 1 - ρ**2))
def _qsvi_iv(k, θ, φ, ρ, T):  return np.sqrt(_qtot(k, θ, φ, ρ)/T)

# ---------- forward -----------------------------------------------------
def _forward(date_str, und) -> float:
    fut = next(FUTURES_DIR.glob(f"*{date_str}*.csv"))
    df  = pd.read_csv(fut)
    df  = df[~df["Contract"].str.contains("(Cash)", regex=False)]
    row = df[df["Contract"].str.startswith(und.upper())]
    if row.empty: row = df.iloc[[0]]
    return float(row["Last"].values[0])

# ---------- fit one slice ----------------------------------------------
def fit_slice(df, F, T, r=0.0):
    calls = df[(df["type"].str.lower()=="call") & (df["smoothed_midprice"]>0)].copy()
    if calls.empty: return None

    strikes = calls["strike"].to_numpy()
    bid, ask = calls["bid"].to_numpy(), calls["ask"].to_numpy()
    mid_px   = 0.5*(bid+ask)

    iv_mid = np.vectorize(_imp_vol)(F, strikes, T, mid_px, r)
    iv_bid = np.vectorize(_imp_vol)(F, strikes, T, bid,    r)
    iv_ask = np.vectorize(_imp_vol)(F, strikes, T, ask,    r)

    msk = np.isfinite(iv_mid)
    strikes, iv_mid, iv_bid, iv_ask = strikes[msk], iv_mid[msk], iv_bid[msk], iv_ask[msk]
    if strikes.size < 4: return None
    k = np.log(strikes/F)

    bounds = [(0.005,0.08),(0.03,0.4),(-0.8,-0.05)]
    x0     = [0.02,0.1,-0.3]

    # Stage 1 – vol RMSE
    vol_rmse = lambda p: np.mean((_qsvi_iv(k,*p,T)-iv_mid)**2)
    θ,φ,ρ = minimize(vol_rmse, x0, method="L-BFGS-B", bounds=bounds).x

    # Stage 2 – band polish
    def band_obj(p):
        θ,φ,ρ=p
        iv = _qsvi_iv(k,θ,φ,ρ,T)
        mp = _bs_call(F,strikes,T,iv,r)
        over  = np.maximum(0, mp - _bs_call(F,strikes,T,iv_ask,r))
        under = np.maximum(0, _bs_call(F,strikes,T,iv_bid,r) - mp)
        centre= np.abs(mp - _bs_call(F,strikes,T,iv_mid,r))
        return np.mean((centre+0.5*(over+under))**2)
    θ,φ,ρ = minimize(band_obj,[θ,φ,ρ],method="L-BFGS-B",bounds=bounds,
                     options={"maxiter":40}).x

    iv_fit = _qsvi_iv(k,θ,φ,ρ,T)
    iv_err = float(np.sqrt(np.mean((iv_fit-iv_mid)**2)))

    # PDF (K‑space)
    K_grid  = np.linspace(0.6*F,1.4*F,601)
    iv_g    = _qsvi_iv(np.log(K_grid/F),θ,φ,ρ,T)
    C_g     = _bs_call(F,K_grid,T,iv_g,r)
    dK      = K_grid[1]-K_grid[0]
    pdf_raw = np.exp(r*T)*np.gradient(np.gradient(C_g,dK),dK)
    pdf     = np.clip(pdf_raw,0,None)
    mass    = np.trapz(pdf,K_grid); pdf/=mass

    x=K_grid/F-1
    mean=np.trapz(x*pdf,K_grid)
    var =np.trapz(((x-mean)**2)*pdf,K_grid)
    skew=np.trapz(((x-mean)**3)*pdf,K_grid)/var**1.5
    kurt=np.trapz(((x-mean)**4)*pdf,K_grid)/var**2

    return dict(variance=var,skew=skew,kurtosis=kurt,
                iv_rmse=iv_err,pdf_mass=mass,
                theta=θ,phi=φ,rho=ρ,
                k=k,iv_mid=iv_mid,iv_fit=iv_fit,
                K_grid=K_grid,pdf=pdf)

# ---------- batch calibration -------------------------------------------
def batch_calibrate(opt_dir, expiry, out_csv):
    rows=[]
    for fp in Path(opt_dir).glob("*.csv"):
        md, mu = DATE_RE.search(fp.name), UND_RE.search(fp.name)
        if not (md and mu): continue
        dt, und = md.group(1), mu.group(1)
        F = _forward(dt,und)
        T = (pd.to_datetime(expiry)-pd.to_datetime(dt,format="%m-%d-%Y")).days/365
        res = fit_slice(pd.read_csv(fp),F,T)
        if res is None: print(f"[SKIP] {fp.name}"); continue
        rows.append({"date":pd.to_datetime(dt,format="%m-%d-%Y"),
                     **{k:res[k] for k in ("variance","skew","kurtosis",
                                           "iv_rmse","pdf_mass",
                                           "theta","phi","rho")}})
        print(f"[VOL] {fp.name:44s}  θ={res['theta']:.4f} φ={res['phi']:.3f} ρ={res['rho']:.3f}  RMSE={res['iv_rmse']:.3f}")
    pd.DataFrame(rows).sort_values("date").to_csv(out_csv,index=False)
    print(f"[VOL] wrote {out_csv}")

# ---------- demo ---------------------------------------------------------
def demo(opt_dir, expiry):
    fp   = next(Path(opt_dir).glob("*.csv"))
    dt   = DATE_RE.search(fp.name).group(1)
    und  = UND_RE.search(fp.name).group(1)
    F    = _forward(dt,und)
    T    = (pd.to_datetime(expiry)-pd.to_datetime(dt,format="%m-%d-%Y")).days/365
    res  = fit_slice(pd.read_csv(fp),F,T)
    if res is None: print("no data"); return

    plt.figure(); plt.scatter(res["k"],res["iv_mid"],s=12,label="Market iv")
    plt.plot   (res["k"],res["iv_fit"],lw=1.5,label="qSVI fit")
    plt.axvline(0,color='k',ls=":"); plt.xlabel("k = ln(K/F)"); plt.ylabel("σ")
    plt.title("Smile fit"); plt.legend()

    plt.figure(); plt.scatter(res["k"],res["iv_fit"]-res["iv_mid"],s=12)
    plt.axhline(0); plt.xlabel("k"); plt.ylabel("Model‑Market σ"); plt.title("Residuals")

    plt.figure(); plt.plot(res["K_grid"],res["pdf"])
    plt.title(f"Risk‑neutral PDF (mass={res['pdf_mass']:.4f})")
    plt.xlabel("Strike  K"); plt.ylabel("f(K)")
    plt.show()

# ---------- CLI ----------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap
    p=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent("""
        qSVI calibration:
          python -m src.vol_engine <option_dir> --expiry YYYY-MM-DD
        Demo (plots one file):
          python -m src.vol_engine <option_dir> --expiry YYYY-MM-DD --demo
        """))
    p.add_argument("option_dir")
    p.add_argument("--expiry", required=True)
    p.add_argument("--out", default="Data/Stats/pdf_moments.csv")
    p.add_argument("--demo", action="store_true")
    args=p.parse_args()
    if args.demo:
        demo(args.option_dir,args.expiry)
    else:
        batch_calibrate(args.option_dir,args.expiry,args.out)
