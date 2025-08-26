import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_DATA_DIR = ROOT_DIR / "Data_act" / "Final"
RESULTS_DIR = ROOT_DIR / "results_testing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = FINAL_DATA_DIR / "model_ready_dataset.csv"
OUTPUT_PLOT_PATH = RESULTS_DIR / "feature_importance.png"

# --- Main Script --- #

def main():
    print("--- Analyzing Feature Importance --- ")
    # Load the dataset
    df = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col='date')

    # Define the target variable (log-ratio of next day's ATM IV)
    df['target'] = np.log(df['atm_iv'].shift(-1) / df['atm_iv'])
    df = df.dropna() # Drop rows with NaN target or NaNs from feature engineering

    # Define features (X) and target (y)
    features_to_drop = ['market_state', 'target', 'atm_iv'] # Drop non-numeric and target-related columns
    features = [col for col in df.columns if col not in features_to_drop]
    X = df[features]
    y = df['target']

    print(f"Analyzing {len(features)} features on {len(X)} data points.")

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # --- Print Full Report --- #
    print("\n--- Feature Importance Ranking ---")
    print(feature_importance_df.to_string())
    print("----------------------------------")

    # --- Generate and Save Visualization --- #
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot only the top 20 features for clarity
        top_20_features = feature_importance_df.head(20)
        
        sns.barplot(x='importance', y='feature', data=top_20_features, palette='viridis', ax=ax)
        
        ax.set_title('Top 20 Most Important Features', fontsize=16)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        fig.savefig(OUTPUT_PLOT_PATH)
        print(f"\nFeature importance plot saved to: {OUTPUT_PLOT_PATH}")

    except Exception as e:
        print(f"\nCould not generate plot. Please ensure matplotlib and seaborn are installed. Error: {e}")

if __name__ == "__main__":
    main()