
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score

from src.config import MODELS_DIR, TABLES_DIR, MODEL_READY_DATASET_PATH, GAMMA_PARAMS, ENHANCED_FEATURES
from src.model_utils import (
    custom_gamma_objective,
    calculate_optimal_gamma_targets,
    create_enhanced_features,
)

def run_ablation_analysis() -> None:
    """
    Performs ablation analysis on the XGBoost model by removing one feature at a time
    and evaluating the impact on performance.
    """
    print("--- Running Ablation Analysis ---")

    # Load and prepare data
    df = pd.read_csv(MODEL_READY_DATASET_PATH, parse_dates=['date'], index_col='date')
    df = create_enhanced_features(df)
    df = calculate_optimal_gamma_targets(df)

    df = df.dropna(subset=ENHANCED_FEATURES + ['optimal_gamma'])
    y = df['optimal_gamma']

    # --- Train Baseline Model --- #
    X_base = df[ENHANCED_FEATURES]
    dtrain_base = xgb.DMatrix(X_base, label=y)

    baseline_model = xgb.train(
        dict(GAMMA_PARAMS),
        dtrain_base,
        num_boost_round=200,
        obj=custom_gamma_objective
    )
    baseline_preds = baseline_model.predict(dtrain_base)
    baseline_r2 = r2_score(y, np.clip(baseline_preds, 0.7, 1.5))
    print(f"Baseline R²: {baseline_r2:.4f}")

    # --- Run Ablation --- #
    ablation_results = []
    for feature in ENHANCED_FEATURES:
        print(f"Ablating feature: {feature}")
        features_ablated = [f for f in ENHANCED_FEATURES if f != feature]
        
        X_ablated = df[features_ablated]
        dtrain_ablated = xgb.DMatrix(X_ablated, label=y)
        
        model_ablated = xgb.train(
            dict(GAMMA_PARAMS),
            dtrain_ablated,
            num_boost_round=200,
            obj=custom_gamma_objective
        )
        
        preds_ablated = model_ablated.predict(dtrain_ablated)
        r2_ablated = r2_score(y, np.clip(preds_ablated, 0.7, 1.5))
        
        performance_drop = baseline_r2 - r2_ablated
        
        ablation_results.append({
            'removed_feature': feature,
            'r2_score': r2_ablated,
            'performance_drop': performance_drop
        })

    # --- Save Results --- #
    df_ablation = pd.DataFrame(ablation_results).sort_values('performance_drop', ascending=False)
    output_path = TABLES_DIR / "ablation_results.csv"
    df_ablation.to_csv(output_path, index=False)
    print(f"\nAblation analysis results saved to {output_path}")
    print(df_ablation)

    # --- Save IM Correction Factor Predictions --- #
    df_results = df.copy()
    df_results['im_correction_factor'] = np.clip(baseline_preds, 0.7, 1.5)
    output_im_path = TABLES_DIR / "im_correction_factor.csv"
    df_results[['im_correction_factor']].to_csv(output_im_path)
    print(f"IM correction factor predictions saved to {output_im_path}")

def run_sensitivity_analysis() -> None:
    """
    Performs sensitivity analysis on the XGBoost model by varying one feature
    and observing the impact on the prediction.
    """
    print("--- Running Sensitivity Analysis ---")

    # Load the trained model
    model_path = MODELS_DIR / "xgboost_gamma_model.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model = joblib.load(model_path)

    # Load data to get feature means
    df = pd.read_csv(MODEL_READY_DATASET_PATH, parse_dates=['date'], index_col='date')
    df = create_enhanced_features(df)
    
    # Feature to analyze (most important from ablation)
    feature_to_analyze = 'iv_vol_ratio'
    
    # Create a range of values for the feature
    feature_range = np.linspace(df[feature_to_analyze].min(), df[feature_to_analyze].max(), 100)
    
    # Create synthetic data
    synthetic_data = []
    mean_features = df[ENHANCED_FEATURES].mean().to_dict()

    for val in feature_range:
        new_point = mean_features.copy()
        new_point[feature_to_analyze] = val
        synthetic_data.append(new_point)
        
    df_synthetic = pd.DataFrame(synthetic_data)
    dmatrix_synthetic = xgb.DMatrix(df_synthetic[ENHANCED_FEATURES])
    
    # Make predictions
    predictions = model.predict(dmatrix_synthetic)
    
    # --- Save Results --- #
    df_sensitivity = pd.DataFrame({
        feature_to_analyze: feature_range,
        'predicted_gamma': np.clip(predictions, 0.7, 1.5)
    })
    
    output_path = TABLES_DIR / "sensitivity_analysis_iv_vol_ratio.csv"
    df_sensitivity.to_csv(output_path, index=False)
    print(f"Sensitivity analysis results saved to {output_path}")
    print(df_sensitivity.head())


if __name__ == "__main__":
    run_ablation_analysis()
    run_sensitivity_analysis()
