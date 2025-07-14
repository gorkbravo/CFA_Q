# src/hmm_engine.py
# ------------------------------------------------------------
# Fit a 2‑state Gaussian Markov‑switching model on the daily
# CTSI series  (Data/Stats/stats.csv)  and write the smoothed
# backwardation probability to  Data/Stats/stats_hmm.csv
# ------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---------- dynamic project‑relative paths ------------------
ROOT_DIR  = Path(__file__).resolve().parents[1]        # CFA_Quant_Awards/
STATS_DIR = ROOT_DIR / "Data" / "Stats"
STATS_CSV = STATS_DIR / "stats.csv"        # input  (written by futures_curve.py)
HMM_CSV   = STATS_DIR / "stats_hmm.csv"    # output
# ------------------------------------------------------------
PLOT = True    # flip to False if running headless

def main():
    # 1. Load CTSI history
    if not STATS_CSV.exists():
        raise FileNotFoundError(f"{STATS_CSV} not found – run futures_curve.py first.")
    df = pd.read_csv(STATS_CSV, parse_dates=["calculation_date"])
    df = df.sort_values("calculation_date").reset_index(drop=True)

    y = df["term_structure_index"]

    # 2. Two‑state Markov‑switching mean model (constant variance)
    mod = sm.tsa.MarkovRegression(y, k_regimes=2, trend="c", switching_variance=False)
    res = mod.fit(em_iter=400, show_warning=False)

    # Identify backwardation regime (lower CTSI mean)
    means = np.array([res.params["const[0]"], res.params["const[1]"]])
    back_state = int(np.argmin(means))
    back_prob  = res.smoothed_marginal_probabilities[back_state]
    viterbi    = res.smoothed_marginal_probabilities.idxmax(axis=1).astype(int)

    # 3. Save results
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date":     df["calculation_date"],
        "hmm_prob": back_prob,
        "state":    viterbi
    }).to_csv(HMM_CSV, index=False)
    print(f"[HMM] wrote {len(df)} rows → {HMM_CSV.name}")

    # 4. Optional quick‑look plot
    if PLOT:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df["calculation_date"], y, lw=1, label="CTSI")
        ax1.set_ylabel("CTSI"); ax1.set_xlabel("Date")

        ax2 = ax1.twinx()
        ax2.fill_between(df["calculation_date"], 0, back_prob,
                         color="orange", alpha=0.3, label="Pr(backwardation)")
        ax2.set_ylim(0, 1); ax2.set_ylabel("HMM probability")

        ax1.set_title("CTSI & Markov‑smoothed backwardation probability")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout(); plt.show()

    # 5. Console diagnostics
    trans = res.transition_probabilities.filtered
    spell = 1 / (1 - np.diag(trans))
    print("[HMM] regime means (CTSI):", means.round(3))
    print("[HMM] filtered transition matrix:\n", trans.round(3))
    print("[HMM] expected spell length (days):", spell.round(1))

if __name__ == "__main__":
    main()
