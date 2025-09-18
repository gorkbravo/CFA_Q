
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score

from src.config import MODEL_READY_DATASET_PATH, MODELS_DIR, TABLES_DIR, GAMMA_PARAMS, PENALTY_COEFF

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
    penalty_coeff = PENALTY_COEFF # Fixing the penalty coefficient as optimal
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
