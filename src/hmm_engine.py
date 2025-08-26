"""
hmm_engine.py - Calculation module for the Hidden Markov Model.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm

def calculate_hmm_probabilities(ctsi_series: pd.Series) -> pd.DataFrame:
    """
    Fits a 2-state Gaussian Markov-switching model on a CTSI series.

    Args:
        ctsi_series: A pandas Series of the daily Curve-Term-Structure Index.

    Returns:
        A DataFrame with the original index, the HMM probabilities, and the predicted state.
    """
    if ctsi_series.empty:
        return pd.DataFrame(columns=['hmm_prob', 'state'])

    # Two-state Markov-switching mean model (constant variance)
    mod = sm.tsa.MarkovRegression(ctsi_series, k_regimes=2, trend="c", switching_variance=False)
    res = mod.fit(em_iter=400, show_warning=False)

    # Identify backwardation regime (lower CTSI mean)
    means = np.array([res.params["const[0]"], res.params["const[1]"]])
    back_state = int(np.argmin(means))
    
    # Get the smoothed probabilities for the backwardation state
    back_prob = res.smoothed_marginal_probabilities[back_state]
    viterbi_state = res.smoothed_marginal_probabilities.idxmax(axis=1).astype(int)

    results_df = pd.DataFrame({
        "hmm_prob": back_prob,
        "state": viterbi_state
    }, index=ctsi_series.index)
    
    return results_df