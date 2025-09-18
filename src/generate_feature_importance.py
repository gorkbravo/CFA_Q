import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import MODEL_READY_DATASET_PATH, FIGURES_DIR

# --- Main Script --- #

def main() -> None:
    """Generate and save feature importance plot."""
    print("--- Analyzing Feature Importance --- ")
    # Load the dataset
    df = pd.read_csv(MODEL_READY_DATASET_PATH, parse_dates=['date'], index_col='date')

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
        
        output_plot_path = FIGURES_DIR / "feature_importance.png"
        fig.savefig(output_plot_path)
        print(f"\nFeature importance plot saved to: {output_plot_path}")

    except Exception as e:
        print(f"\nCould not generate plot. Please ensure matplotlib and seaborn are installed. Error: {e}")
