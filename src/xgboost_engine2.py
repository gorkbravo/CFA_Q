import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

def custom_gamma_objective(y_pred, dtrain):
    y_true = dtrain.get_label()

    # MSE component
    grad_mse = (y_pred - y_true)
    hess_mse = np.ones_like(y_pred)

    

    # Initialize gradients and Hessians for extreme penalty
    grad_extreme = np.zeros_like(y_pred)
    hess_extreme = np.zeros_like(y_pred)

    # Apply conditions for extreme penalty
    # y_pred > 1.3
    mask_upper = y_pred > 1.3
    grad_extreme[mask_upper] = 2 * (y_pred[mask_upper] - 1.3)
    hess_extreme[mask_upper] = 2 * np.ones_like(y_pred[mask_upper])

    # y_pred < 0.7
    mask_lower = y_pred < 0.7
    grad_extreme[mask_lower] = 2 * (y_pred[mask_lower] - 0.7)
    hess_extreme[mask_lower] = 2 * np.ones_like(y_pred[mask_lower])

    # New: Penalty for higher im_correction_factor values
    penalty_coeff = 0.05 # Fixing the penalty coefficient as optimal
    grad_avg_margin = penalty_coeff * np.ones_like(y_pred)
    hess_avg_margin = np.zeros_like(y_pred)

    # Total gradient and Hessian
    grad = grad_mse + grad_extreme + grad_avg_margin
    hess = hess_mse + hess_extreme + hess_avg_margin

    return grad, hess

def calculate_optimal_gamma_targets(df, lookback_window=22):
    """
    Calculate what gamma should have been based on realized volatility
    This creates training targets for direct gamma prediction
    """
    # Calculate realized volatility over next period
    df['realized_vol'] = df['atm_iv'].shift(-1)  # Next day's actual IV
    df['vol_forecast'] = df['atm_iv']  # Current IV (baseline forecast)
    
    # Optimal gamma would be the ratio that minimizes forecast error
    # while staying within reasonable bounds
    optimal_gamma = df['realized_vol'] / df['vol_forecast']
    
    # Clip to reasonable range and smooth
    optimal_gamma = np.clip(optimal_gamma, 0.7, 1.5)
    
    # Apply smoothing to reduce noise
    df['optimal_gamma'] = optimal_gamma.rolling(3, center=True).mean()
    
    return df.dropna(subset=['optimal_gamma'])

def create_enhanced_features(df):
    """Add interaction terms and regime features"""
    
    # Volatility regime indicators
    df['vol_regime_high'] = (df['atm_iv'] > df['atm_iv'].rolling(22).quantile(0.75)).astype(int)
    df['vol_regime_low'] = (df['atm_iv'] < df['atm_iv'].rolling(22).quantile(0.25)).astype(int)
    
    # Interaction terms
    df['skew_kurt_interaction'] = df['implied_skew'] * df['implied_kurtosis']
    df['vol_momentum'] = df['atm_iv_mom_1'] * df['implied_variance_mom_1']
    df['term_structure_interaction'] = df['T'] * df['F']
    df['rho_vol_interaction'] = df['rho'] * df['atm_iv']
    
    # Market stress indicators
    df['iv_vol_ratio'] = df['atm_iv'] / df['atm_iv'].rolling(10).std()
    df['skew_level'] = np.abs(df['implied_skew'])
    
    return df

def main():
    print("--- Direct Gamma Training with XGBoost (Custom Objective) ---")

    # Load and prepare data
    df = pd.read_csv(DATASET_PATH, parse_dates=['date'], index_col='date')

    # Create enhanced features
    df = create_enhanced_features(df)

    # Calculate optimal gamma targets
    df = calculate_optimal_gamma_targets(df)

    # Enhanced feature set
    enhanced_features = [
        # Original top features
        'atm_iv_mom_1', 'implied_kurtosis_mom_1', 'atm_iv_ma_2', 'implied_skew_mom_1',
        'implied_skew_ma_2', 'rho_lag_1', 'F', 'atm_iv_lag_1', 'rho_ma_2',
        'implied_variance_mom_1', 'implied_skew_lag_1', 'nu', 'rho_mom_1', 'iv_rmse',
        'T', 'rho', 'implied_kurtosis', 'r', 'implied_kurtosis_lag_1', 'implied_skew',
        # New enhanced features
        'vol_regime_high', 'vol_regime_low', 'skew_kurt_interaction', 'vol_momentum',
        'term_structure_interaction', 'rho_vol_interaction', 'iv_vol_ratio', 'skew_level',
        # OVX features
        'OVX_Close_ma_2', 'OVX_Close_mom_1'
    ]

    # Ensure all features and target are non-NaN before splitting
    df = df.dropna(subset=enhanced_features + ['optimal_gamma'])

    # Prepare training data
    X = df[enhanced_features]
    y = df['optimal_gamma']

    # Scale features
    scaler = joblib.load(SCALER_PATH) if (MODEL_DIR / "scaler.joblib").exists() else None
    if scaler:
        original_features = enhanced_features[:20]
        X_scaled = X.copy()
        X_scaled[original_features] = scaler.transform(X[original_features])
    else:
        X_scaled = X

    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_scaled, label=y)

    # Gamma-optimized XGBoost parameters
    gamma_params = {
        'n_estimators': 200, # This will be num_boost_round
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1 # Not directly used by xgb.train, but good to keep for consistency
    }

    # Train model using xgb.train with custom objective
    print("Training Gamma-Optimized XGBoost with custom objective...")
    # Remove n_estimators from params as it's passed as num_boost_round
    num_boost_round = gamma_params.pop('n_estimators')
    xgb_gamma_booster = xgb.train(
        gamma_params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=custom_gamma_objective
    )
    print("XGBoost model training complete.")

    # Save the trained XGBoost Booster model
    joblib.dump(xgb_gamma_booster, MODEL_DIR / "xgboost_gamma_model.joblib")
    print(f"\nBest XGBoost model saved to {MODEL_DIR / 'xgboost_gamma_model.joblib'}")

    # Make predictions using the Booster model
    gamma_predictions = xgb_gamma_booster.predict(dtrain)

    # Bound predictions to reasonable range
    gamma_predictions = np.clip(gamma_predictions, 0.7, 1.5)

    # Calculate R² for gamma predictions (using sklearn's r2_score)
    from sklearn.metrics import r2_score
    gamma_r2 = r2_score(y, gamma_predictions)
    print(f"Gamma Prediction R²: {gamma_r2:.4f}")

    # Feature importance analysis (Booster has get_score for feature importance)
    feature_importance_dict = xgb_gamma_booster.get_score(importance_type='weight')
    feature_importance = pd.DataFrame({
        'feature': list(feature_importance_dict.keys()),
        'importance': list(feature_importance_dict.values())
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features for Gamma:")
    print(feature_importance.head(10))

    # --- Save Feature Importance --- #
    feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nFeature importance data saved to C:\\Users\\User\\Desktop\\Me\\Coding Projects\\CFA_Quant_Awards\\results_testing\\feature_importance.csv")

    # Save results
    df_results = df.copy()
    df_results['im_correction_factor'] = gamma_predictions
    df_results['optimal_gamma_target'] = y

    output_df = df_results[['atm_iv', 'im_correction_factor', 'optimal_gamma_target']].dropna()
    output_df.to_csv(OUTPUT_PATH)

    print(f"\nGamma model and results saved.")
    print(f"Mean Gamma: {np.mean(gamma_predictions):.3f}")
    print(f"Gamma Std: {np.std(gamma_predictions):.3f}")
    print(f"Gamma Range: [{np.min(gamma_predictions):.3f}, {np.max(gamma_predictions):.3f}]")

if __name__ == "__main__":
   
    ROOT_DIR = Path(__file__).resolve().parents[1]
    FINAL_DATA_DIR = ROOT_DIR / "Data_act" / "Final"
    MODEL_DIR = ROOT_DIR / "models"
    RESULTS_DIR = ROOT_DIR / "results_testing"
    
    DATASET_PATH = FINAL_DATA_DIR / "model_ready_dataset.csv"
    OUTPUT_PATH = RESULTS_DIR / "im_correction_factor.csv"
    SCALER_PATH = MODEL_DIR / "scaler.joblib"
    
    main()