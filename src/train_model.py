import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score

from src.config import MODEL_READY_DATASET_PATH, MODELS_DIR, TABLES_DIR, GAMMA_PARAMS
from src.model_utils import (
    custom_gamma_objective,
    calculate_optimal_gamma_targets,
    create_enhanced_features,
)


def main() -> None:
    """Train the XGBoost model for gamma prediction and write outputs."""
    print("---Direct Gamma Training with XGBoost (Custom Objective)---")

    # Load and prepare data
    df = pd.read_csv(MODEL_READY_DATASET_PATH, parse_dates=['date'], index_col='date')

    # Create enhanced features and targets
    df = create_enhanced_features(df)
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

    # Optional scaling of a subset of features
    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        original_features = enhanced_features[:20]
        X_scaled = X.copy()
        X_scaled[original_features] = scaler.transform(X[original_features])
    else:
        X_scaled = X

    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_scaled, label=y)

    # Gamma-optimized XGBoost parameters (copy to avoid mutation)
    params = dict(GAMMA_PARAMS)
    num_boost_round = params.pop('n_estimators')

    # Train model using xgb.train with custom objective
    print("Training Gamma-Optimized XGBoost with custom objective...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=custom_gamma_objective,
    )
    print("XGBoost model training complete.")

    # Save the trained XGBoost Booster model
    joblib.dump(booster, MODELS_DIR / "xgboost_gamma_model.joblib")
    print(f"\nBest XGBoost model saved to {MODELS_DIR / 'xgboost_gamma_model.joblib'}")

    # Make predictions using the Booster model
    preds = booster.predict(dtrain)
    preds = np.clip(preds, 0.7, 1.5)

    # Calculate R^2 for gamma predictions
    gamma_r2 = r2_score(y, preds)
    print(f"Gamma Prediction R^2: {gamma_r2:.4f}")

    # Feature importance analysis (Booster has get_score for feature importance)
    feature_importance_dict = booster.get_score(importance_type='weight')
    feature_importance = pd.DataFrame({
        'feature': list(feature_importance_dict.keys()),
        'importance': list(feature_importance_dict.values())
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features for Gamma:")
    print(feature_importance.head(10))

    # --- Save Feature Importance --- #
    feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    print(f"\nFeature importance data saved to {TABLES_DIR / 'feature_importance.csv'}")

    # Save results
    df_results = df.copy()
    df_results['im_correction_factor'] = preds
    df_results['optimal_gamma_target'] = y

    output_df = df_results[['atm_iv', 'im_correction_factor', 'optimal_gamma_target']].dropna()
    output_df.to_csv(TABLES_DIR / "im_correction_factor.csv")


if __name__ == "__main__":
    main()

