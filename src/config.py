from pathlib import Path

# --- Base Paths --- #
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# --- Data Paths --- #
RAW_DATA_DIR = DATA_DIR / "raw"
FUTURES_DIR = RAW_DATA_DIR / "Futures_curve_time_series"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_READY_DATASET_PATH = PROCESSED_DATA_DIR / "model_ready_dataset.csv"

# --- Results Paths --- #
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# --- Model Parameters (XGBoost) --- #
GAMMA_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

# Penalty coefficient used in custom objective (fixed to preserve results)
PENALTY_COEFF = 0.05

ENHANCED_FEATURES = [
    'atm_iv_mom_1', 'implied_kurtosis_mom_1', 'atm_iv_ma_2', 'implied_skew_mom_1',
    'implied_skew_ma_2', 'rho_lag_1', 'F', 'atm_iv_lag_1', 'rho_ma_2',
    'implied_variance_mom_1', 'implied_skew_lag_1', 'nu', 'rho_mom_1', 'iv_rmse',
    'T', 'rho', 'implied_kurtosis', 'r', 'implied_kurtosis_lag_1', 'implied_skew',
    'vol_regime_high', 'vol_regime_low', 'skew_kurt_interaction', 'vol_momentum',
    'term_structure_interaction', 'rho_vol_interaction', 'iv_vol_ratio', 'skew_level',
    'OVX_Close_ma_2', 'OVX_Close_mom_1'
]


# --- Backtest Parameters --- #
TEST_DATE = '2025-08-22' # Consistent test date for data handlers
FUTURES_CONTRACT_SYMBOL = 'CLV25'
CONFIDENCE_LEVELS = [0.95, 0.99, 0.995]
GARCH_P = 2
GARCH_Q = 2
GARCH_VOL = 'EGARCH'


# SVI Engine Parameters
START_DATE = '2023-02-07'
END_DATE = '2025-08-22'
EXPIRY_SYMBOL = 'clv5'
EXPIRY_DATE = '2025-09-17'

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
