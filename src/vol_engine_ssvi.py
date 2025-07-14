
"""
vol_engine_ssvi.py - Fits the SSVI model to cleaned options data.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# --- Constants ---
ROOT_DIR = Path(__file__).resolve().parents[1]
CLEANED_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Cleaned"
STATS_DIR = ROOT_DIR / "Data" / "Stats"
PDF_MOMENTS_CSV = STATS_DIR / "pdf_moments.csv"

# --- SSVI Model ---
def ssvi_surface(k, theta, phi, rho):
    """ SSVI volatility smile parameterization. """
    return theta * (1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + 1 - rho**2))

def objective_function(params, k, market_vols):
    """ Objective function to minimize for SSVI calibration. """
    theta, phi, rho = params
    model_vols = ssvi_surface(k, theta, phi, rho)
    return np.sum((model_vols - market_vols)**2)

# --- Core Processing ---
def process_cleaned_options(file_path: Path, futures_price: float):
    """
    Fits the SSVI model to a single cleaned options file.
    """
    df = pd.read_csv(file_path)

    # Calculate log-moneyness (k)
    df["k"] = np.log(df["Strike"] / futures_price)

    # Calculate implied volatility (simple approximation)
    # Note: A more robust calculation would use a proper library (e.g., py_vollib)
    # For now, we'll use a placeholder calculation.
    # This is a critical point for improvement.
    df["Implied_Vol"] = np.sqrt(2 * np.pi / (df["Mid"] / futures_price)) * (df["Mid"] / futures_price)

    # --- Fit SSVI ---
    initial_params = [0.1, 0.1, -0.5] # theta, phi, rho
    bounds = [(1e-3, None), (1e-3, None), (-0.999, 0.999)]

    result = minimize(objective_function, initial_params, args=(df["k"], df["Implied_Vol"]),
                      bounds=bounds, method='L-BFGS-B')

    theta_fit, phi_fit, rho_fit = result.params

    # --- Calculate Moments (placeholders) ---
    # These would be derived from the fitted SSVI parameters.
    # This is a complex step and would require further implementation.
    variance = theta_fit**2
    skewness = 0 # Placeholder
    kurtosis = 3 # Placeholder

    return {
        "file": file_path.name,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis
    }

def main():
    """
    Processes all cleaned options files and saves the moments.
    """
    # This is a placeholder for getting the daily futures price.
    # In a real implementation, this would be read from the futures data.
    futures_price = 100.0

    all_moments = []
    for file_path in CLEANED_OPTIONS_DIR.glob("*.csv"):
        moments = process_cleaned_options(file_path, futures_price)
        all_moments.append(moments)

    df_moments = pd.DataFrame(all_moments)
    df_moments.to_csv(PDF_MOMENTS_CSV, index=False)
    print(f"[SSVI] Processed {len(df_moments)} files and saved moments to {PDF_MOMENTS_CSV}")

if __name__ == "__main__":
    main()
