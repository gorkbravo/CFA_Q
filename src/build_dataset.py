import pandas as pd
from pathlib import Path
import re

# Import our modules
from src.config import ROOT_DIR, TABLES_DIR, MODEL_READY_DATASET_PATH, START_DATE, END_DATE, EXPIRY_SYMBOL, EXPIRY_DATE
from src import data_handlers
from src import futures_curve as fcm
from src import hmm_engine
from src import feature_engine
from src import SVI_SABR_engine

SVI_OUTPUT_PATH = TABLES_DIR / "daily_svi_analysis.csv"
FINAL_DATASET_PATH = MODEL_READY_DATASET_PATH

def main():
    # --- Step 1: Run SVI Engine --- #
    print("--- Running SVI Engine ---")
    SVI_SABR_engine.run_svi_engine(START_DATE, END_DATE, EXPIRY_SYMBOL, EXPIRY_DATE)
    df_svi = pd.read_csv(SVI_OUTPUT_PATH, parse_dates=['date'], index_col='date')
    print(f"SVI Engine finished. Shape: {df_svi.shape}")

    # --- Step 2: Calculate Term Structure & HMM Features --- #
    print("\n--- Calculating Term Structure and HMM Features ---")
    all_features = []
    date_range = pd.to_datetime(pd.date_range(start=START_DATE, end=END_DATE, freq='B'))

    month_codes = dict(zip("FGHJKMNQUVXZ", range(1, 13)))
    def get_expiry(code):
        m = re.match(r"CL([FGHJKMNQUVXZ])(\d{2})", code)
        if not m: return pd.NaT
        return pd.Timestamp(2000 + int(m.group(2)), month_codes[m.group(1)], 1)

    for date in date_range:
        try:
            curve_df = data_handlers.get_futures_curve(date)
            curve_df['expiration'] = curve_df['contract'].apply(get_expiry)
            front_month_expiry = curve_df['expiration'].min()
            features = fcm.calculate_term_structure_features(curve_df, front_month_expiry)
            features['date'] = date
            all_features.append(features)
        except (ValueError, IndexError):
            continue
    df_ts = pd.DataFrame(all_features).set_index('date')

    if not df_ts.empty and 'term_structure_index' in df_ts.columns:
        df_hmm = hmm_engine.calculate_hmm_probabilities(df_ts['term_structure_index'])
        df_ts = df_ts.join(df_hmm)

    print(f"Term Structure and HMM features calculated. Shape: {df_ts.shape}")

    # --- Step 3: Merge SVI, Term Structure, and OVX Data --- #
    print("\n--- Merging Data Sources ---")
    df_ovx = data_handlers.get_ovx_data()
    df_merged = df_svi.join(df_ts, how='inner').join(df_ovx, how='inner')
    
    # Drop the problematic, non-numeric column before feature engineering
    df_merged = df_merged.drop(columns=['warn_code'], errors='ignore')
    print(f"Merged data shape after cleaning: {df_merged.shape}")

    # --- Step 4: Generate Time-Series Features --- #
    print("\n--- Generating Time-Series Features ---")
    # Using reduced complexity features for the limited dataset
    df_final = feature_engine.generate_timeseries_features(df_merged.drop(columns=['state', 'hmm_prob']))
    df_final['hmm_regime'] = df_merged['state']

    # Drop rows with NaN values created by feature engineering
    df_final = df_final.dropna()

    print(f"Final dataset created with {len(df_final)} rows and {len(df_final.columns)} columns.")

    # --- Step 5: Save Final Dataset --- #
    df_final.to_csv(FINAL_DATASET_PATH)
    print(f"\nModel-ready dataset saved to {FINAL_DATASET_PATH}")

