
import pandas as pd
import numpy as np
import xgboost as xgb

from src.config import TABLES_DIR, MODEL_READY_DATASET_PATH, GAMMA_PARAMS, ENHANCED_FEATURES
from src.backtest_engine import run_backtest
from src.model_utils import (
    custom_gamma_objective,
    calculate_optimal_gamma_targets,
    create_enhanced_features,
)


def run_backtest_ablation_analysis() -> None:
    """
    Performs ablation analysis on the XGBoost model by removing one feature at a time
    and evaluating the impact on backtest performance.
    """
    print("--- Running Backtest-Driven Ablation Analysis ---")

    # Load and prepare data
    df = pd.read_csv(MODEL_READY_DATASET_PATH, parse_dates=['date'], index_col='date')
    df = create_enhanced_features(df)
    df = calculate_optimal_gamma_targets(df)

    df = df.dropna(subset=ENHANCED_FEATURES + ['optimal_gamma'])
    y = df['optimal_gamma']

    backtest_ablation_results = []

    # --- Run Ablation --- #
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
        
        baseline_preds = model_ablated.predict(dtrain_ablated)

        # --- Save IM Correction Factor Predictions --- #
        df_results = df.copy()
        df_results['im_correction_factor'] = np.clip(baseline_preds, 0.7, 1.5)
        output_im_path = TABLES_DIR / "im_correction_factor.csv"
        df_results[['im_correction_factor']].to_csv(output_im_path)

        # --- Run Backtest --- # 
        # The backtest engine will read the file we just saved
        breaches, avg_margin, procyclicality = run_backtest()

        backtest_ablation_results.append({
            'removed_feature': feature,
            'breaches': breaches,
            'avg_margin_percent': avg_margin,
            'procyclicality': procyclicality
        })

    # --- Save Results --- #
    df_backtest_ablation = pd.DataFrame(backtest_ablation_results)
    output_path = TABLES_DIR / "backtest_ablation_results.csv"
    df_backtest_ablation.to_csv(output_path, index=False)
    print(f"\nBacktest ablation analysis results saved to {output_path}")
    print(df_backtest_ablation)

if __name__ == "__main__":
    run_backtest_ablation_analysis()
