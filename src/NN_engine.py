import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_DATA_DIR = ROOT_DIR / "Data_act" / "Final"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = FINAL_DATA_DIR / "model_ready_dataset.csv"

# --- Main Script --- #

def main():
    print("--- Hyperparameter Tuning for Neural Network --- ")
    # Load the dataset
    df = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col='date')

    # Define the target variable (log-ratio of next day's ATM IV)
    df['target'] = np.log(df['atm_iv'].shift(-1) / df['atm_iv'])
    df = df.dropna(subset=['target'])

    # Use the Top 20 features identified from our analysis
    top_20_features = [
        'atm_iv_mom_1', 'implied_kurtosis_mom_1', 'atm_iv_ma_2', 'implied_skew_mom_1',
        'implied_skew_ma_2', 'rho_lag_1', 'F', 'atm_iv_lag_1', 'rho_ma_2',
        'implied_variance_mom_1', 'implied_skew_lag_1', 'nu', 'rho_mom_1', 'iv_rmse',
        'T', 'rho', 'implied_kurtosis', 'r', 'implied_kurtosis_lag_1', 'implied_skew'
    ]
    
    X = df[top_20_features]
    y = df['target']

    print(f"Tuning model on {len(X)} data points with {len(X.columns)} features.")

    # Scale the features before the search
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the parameter grid to search
    param_grid = {
        'hidden_layer_sizes': [(50, 25), (100, 50), (50, 25, 10)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01], # L2 regularization
    }

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Set up GridSearchCV
    mlp = MLPRegressor(max_iter=2000, random_state=42, early_stopping=True)
    grid_search = GridSearchCV(mlp, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=2)

    print("\nStarting Grid Search...")
    grid_search.fit(X_scaled, y)

    # --- Report Results and Save Best Model ---
    print("\nGrid Search Complete.")
    print(f"Best R^2 Score: {grid_search.best_score_:.4f}")
    print("Best Hyperparameters:")
    print(grid_search.best_params_)

    # Save the best model found by the grid search
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, MODEL_DIR / "nn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib") # Save the same scaler
    print(f"\nBest model and scaler saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()