"""
IM_engine.py
=========================
Uses the trained model (which predicts log-ratios) to generate the final
IM correction factor.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# --- Configuration --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_DATA_DIR = ROOT_DIR / "Data_act" / "Final"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results_testing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = FINAL_DATA_DIR / "model_ready_dataset.csv"
OUTPUT_PATH = RESULTS_DIR / "im_correction_factor.csv"

# --- Main Script --- #

def main():
    print("--- Generating IM Correction Factor from Log-Ratio Model ---")
    # Load the dataset that was used for training
    df = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col='date')

    # Drop non-numeric columns to match the feature set used for training
    df = df.drop(columns=['market_state'], errors='ignore')

    # Load the trained model and scaler
    mlp = joblib.load(MODEL_DIR / "nn_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    # Define the features (top 20 features used for training)
    top_20_features = [
        'atm_iv_mom_1', 'implied_kurtosis_mom_1', 'atm_iv_ma_2', 'implied_skew_mom_1',
        'implied_skew_ma_2', 'rho_lag_1', 'F', 'atm_iv_lag_1', 'rho_ma_2',
        'implied_variance_mom_1', 'implied_skew_lag_1', 'nu', 'rho_mom_1', 'iv_rmse',
        'T', 'rho', 'implied_kurtosis', 'r', 'implied_kurtosis_lag_1', 'implied_skew'
    ]
    X = df[top_20_features]

    # Scale features and predict the log-ratio
    X_scaled = scaler.transform(X)
    predicted_log_ratios = mlp.predict(X_scaled)

    # Convert the predicted log-ratio back to the correction factor
    # Factor = exp(log(IV_t+1 / IV_t)) = IV_t+1 / IV_t
    df['im_correction_factor'] = np.exp(predicted_log_ratios)

    # Add original ATM IV back for context in the output file
    df['atm_iv'] = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col='date')['atm_iv']

    # Save the results
    df_output = df[['atm_iv', 'im_correction_factor']].copy()
    df_output.to_csv(OUTPUT_PATH)
    print(f"IM correction factors calculated and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
