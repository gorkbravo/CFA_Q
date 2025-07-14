"""
futures_curve.py  – production‑ready prototype
──────────────────────────────────────────────
Reads every WTI curve CSV in   Data/Futures_data/
Computes the Curve‑Term‑Structure Index (CTSI)
Appends (or updates)            Data/Stats/stats.csv
▪ Skips the “CLY00 (Cash)” spot row
▪ Uses true front‑month futures as anchor
▪ Includes raw & normalised spreads + persistence
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import re, sys

# ---------- folder constants --------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[1]
FUTURES_DIR= ROOT_DIR / "Data" / "Futures_data"
STATS_DIR  = ROOT_DIR / "Data" / "Stats"
STATS_CSV  = STATS_DIR / "stats.csv"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers -----------------------------------------
MONTH_CODES = dict(zip("FGHJKMNQUVXZ", range(1, 13)))
DATE_RE     = re.compile(r"(\d{2}-\d{2}-\d{4})")   # mm-dd-yyyy in filename

def _expiry(code: str):
    m = re.match(r"CL([FGHJKMNQUVXZ])(\d{2})", code)
    if not m: return pd.NaT
    month = MONTH_CODES[m.group(1)]
    year  = 2000 + int(m.group(2))
    return pd.Timestamp(year, month, 1)

def _snapshot_date(fname: str) -> str:
    m = DATE_RE.search(fname)
    if m:
        return datetime.strptime(m.group(1), "%m-%d-%Y").strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")

def _ctsi_components(df: pd.DataFrame):
    front_price, front_exp = df.iloc[0]["Last"], df.iloc[0]["Expiration"]

    anchors = {"1M":30, "3M":90, "1Y":365}          # calendar days
    spreads = {}
    for tag, days in anchors.items():
        td   = pd.Timedelta(days=days)
        mask = df["Expiration"].sub(front_exp).ge(td)
        idx  = mask.idxmax() if mask.any() else len(df) - 1   # fallback last contract
        price = df.loc[idx, "Last"]
        spreads[tag] = (price - front_price) / front_price

    sev   = {"1M":spreads["1M"]/0.01,
             "3M":spreads["3M"]/0.03,
             "1Y":spreads["1Y"]/0.05}
    norm  = {k: np.tanh(v) for k, v in sev.items()}

    persistence = np.mean(np.diff(df["Last"]) < 0)
    impact = (2*persistence - 1) * min(abs(spreads["1Y"]/0.05), 1)

    ctsi = np.clip(
        0.25*norm["1M"] + 0.35*norm["3M"] +
        0.25*norm["1Y"] + 0.15*impact, -1, 1
    )

    comp = {
        "month1_spread": spreads["1M"],
        "month3_spread": spreads["3M"],
        "year1_spread":  spreads["1Y"],
        "norm_month1":   norm["1M"],
        "norm_month3":   norm["3M"],
        "norm_year1":    norm["1Y"],
        "persistence":   persistence,
        "persistence_impact": impact,
    }
    return float(ctsi), comp

def _upsert_row(row: dict):
    if STATS_CSV.exists():
        df = pd.read_csv(STATS_CSV)
        mask = (df["calculation_date"] == row["calculation_date"]) & \
               (df["source_file"]      == row["source_file"])
        df = df.loc[~mask]
    else:
        df = pd.DataFrame()
    pd.concat([df, pd.DataFrame([row])]).to_csv(STATS_CSV, index=False)

# ---------- core processing ---------------------------------
def process_curve(fp: Path, plot: bool=False):
    df = pd.read_csv(fp)

    # 1) drop spot “Cash” row
    df = df[df["Contract"].str.contains("(Cash)") == False]

    # 2) keep only CL codes like CLH25
    df = df[df["Contract"].str.match(r"CL[A-Z]\d{2}")]
    df["Expiration"] = df["Contract"].apply(_expiry)
    df = df.dropna(subset=["Expiration"]).sort_values("Expiration")
    if df.empty: return

    ctsi, comp = _ctsi_components(df)
    row = {
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "source_file": fp.name,
        "term_structure_index": round(ctsi,4),
        "calculation_date": _snapshot_date(fp.name),
        "market_state": "Contango" if ctsi >= 0 else "Backwardation",
        **{k: round(v,4) for k,v in comp.items()}
    }
    _upsert_row(row)
    print(f"[CTSI] {fp.name:35s}  {ctsi:+.3f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(df["Expiration"], df["Last"], marker="o")
        plt.title(f"{fp.name} – CTSI {ctsi:+.3f}")
        plt.ylabel("Price ($)")
        plt.show()

# ---------- CLI entry ---------------------------------------
def main():
    if len(sys.argv) > 1:
        process_curve(Path(sys.argv[1]), plot="--plot" in sys.argv)
    else:
        for fp in FUTURES_DIR.glob("*.csv"):
            process_curve(fp)

if __name__ == "__main__":
    main()
