import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import t, chi2

from src.config import (
    TABLES_DIR,
    FUTURES_CONTRACT_SYMBOL,
    CONFIDENCE_LEVELS,
    GARCH_P,
    GARCH_Q,
    GARCH_VOL,
)
from src.data_handlers import get_daily_closing_prices
from src.garch_engine import garch_volatility_forecast

def kupiec_uc_test(n_breaches, n_obs, confidence_level):
    """Kupiec's Unconditional Coverage (UC) test."""
    p = 1 - confidence_level
    if n_breaches == 0:
        # Handle case where no breaches occur to avoid log(0)
        if p == 0: # If expected breaches are also 0, it's a perfect match
            return 1.0 # p-value of 1.0
        else: # If expected breaches > 0 but observed is 0, it's a strong rejection
            return 0.0 # p-value of 0.0
    
    lr_uc = -2 * (np.log(((1 - p)**(n_obs - n_breaches)) * (p**n_breaches)) - \
                  np.log(((1 - n_breaches/n_obs)**(n_obs - n_breaches)) * ((n_breaches/n_obs)**n_breaches)))
    p_value = 1 - chi2.cdf(lr_uc, df=1)
    return p_value

def christoffersen_ind_test(breaches):
    """Christoffersen's Independence test."""
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for i in range(1, len(breaches)):
        if breaches[i-1] == 0 and breaches[i] == 0:
            n00 += 1
        elif breaches[i-1] == 0 and breaches[i] == 1:
            n01 += 1
        elif breaches[i-1] == 1 and breaches[i] == 0:
            n10 += 1
        elif breaches[i-1] == 1 and breaches[i] == 1:
            n11 += 1
    
    # Handle cases where denominators might be zero
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return 1.0 # Cannot compute, assume independence

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi == 0 or pi == 1: # Avoid log(0) or log(1) issues if pi is 0 or 1
        return 1.0

    lr_ind = -2 * (np.log(((1 - pi01)**n00) * (pi01**n01) * \
                          ((1 - pi11)**n10) * (pi11**n11)) - \
                   np.log(((1 - pi)**(n00 + n10)) * (pi**(n01 + n11))))
    
    p_value = 1 - chi2.cdf(lr_ind, df=1)
    return p_value



def run_backtest():
    # --- Load Data ---
    df_im = pd.read_csv(TABLES_DIR / "im_correction_factor.csv", parse_dates=['date'])
    df_im.set_index('date', inplace=True)

    df_prices = get_daily_closing_prices(FUTURES_CONTRACT_SYMBOL)
    if df_prices.empty:
        print("Could not load futures prices. Aborting backtest.")
        return

    # --- Calculate P&L and New VaR-based Baseline Margin ---
    df_prices['returns'] = df_prices['price'].pct_change()

    # Calculate GARCH volatility and degrees of freedom
    df_prices['garch_volatility'], dof = garch_volatility_forecast(df_prices['returns'], p=GARCH_P, q=GARCH_Q, vol=GARCH_VOL)

    # Calculate parametric VaR using GARCH volatility and Student's t-distribution for multiple confidence levels
    for cl in CONFIDENCE_LEVELS:
        df_prices[f'var_{int(cl*1000)}_garch_t'] = df_prices['garch_volatility'] * t.ppf(1 - cl, df=dof)
        df_prices[f'margin_{int(cl*1000)}'] = df_prices[f'var_{int(cl*1000)}_garch_t'].abs() * df_prices['price']

    # The margin is the absolute VaR percentage multiplied by the current price
    df_prices['baseline_margin'] = df_prices['margin_990'] # Default baseline is 99%

    df_prices['pnl'] = df_prices['price'].diff().shift(-1)

    # --- Join and Clean ---
    # We only need the price, pnl, and the new baseline margin from the prices df
    df_backtest = df_im.join(
        df_prices[
            ['price', 'pnl', 'baseline_margin', 'garch_volatility']
            + [f'margin_{int(cl*1000)}' for cl in CONFIDENCE_LEVELS]
        ],
        how='inner',
    )
    df_backtest.dropna(inplace=True)

    if df_backtest.empty:
        print("Backtest dataframe is empty. This is likely due to a date mismatch between the options data and the futures price data.")
        return

    # --- Calculations ---
    # The dynamic margin now modifies the new VaR-based baseline
    df_backtest['dynamic_margin'] = df_backtest['baseline_margin'] * df_backtest['im_correction_factor']
    
    # Calculate breaches for all confidence levels
    for cl in CONFIDENCE_LEVELS:
        df_backtest[f'baseline_breach_{int(cl*1000)}'] = (df_backtest['pnl'].abs() > df_backtest[f'margin_{int(cl*1000)}']).astype(int)
    
    df_backtest['baseline_breach'] = df_backtest['baseline_breach_990'] # Default for reporting
    df_backtest['dynamic_breach'] = (df_backtest['pnl'].abs() > df_backtest['dynamic_margin']).astype(int)

    # Calculate breach magnitudes
    df_backtest['baseline_breach_magnitude'] = np.where(df_backtest['baseline_breach'] == 1, df_backtest['pnl'].abs() - df_backtest['baseline_margin'], 0)
    df_backtest['dynamic_breach_magnitude'] = np.where(df_backtest['dynamic_breach'] == 1, df_backtest['pnl'].abs() - df_backtest['dynamic_margin'], 0)

    # Run formal tests
    n_obs = len(df_backtest)
    formal_test_results = {}
    for cl in CONFIDENCE_LEVELS:
        n_breaches_baseline = df_backtest[f'baseline_breach_{int(cl*1000)}'].sum()
        n_breaches_dynamic = df_backtest['dynamic_breach'].sum() # Dynamic model always uses 99% baseline for correction

        # Kupiec UC Test
        p_value_uc_baseline = kupiec_uc_test(n_breaches_baseline, n_obs, cl)
        p_value_uc_dynamic = kupiec_uc_test(n_breaches_dynamic, n_obs, 0.99) # Dynamic is always 99% target

        # Christoffersen Independence Test
        p_value_ind_baseline = christoffersen_ind_test(df_backtest[f'baseline_breach_{int(cl*1000)}'].values)
        p_value_ind_dynamic = christoffersen_ind_test(df_backtest['dynamic_breach'].values)

        formal_test_results[f'baseline_{int(cl*1000)}'] = {'UC_p_value': p_value_uc_baseline, 'Ind_p_value': p_value_ind_baseline}
        if cl == 0.99: # Only report dynamic for its target CL
            formal_test_results[f'dynamic_{int(cl*1000)}'] = {'UC_p_value': p_value_uc_dynamic, 'Ind_p_value': p_value_ind_dynamic}
    
    # Store formal test results in a DataFrame for saving
    df_formal_tests = pd.DataFrame.from_dict({(i,j): formal_test_results[i][j] 
                                               for i in formal_test_results.keys() 
                                               for j in formal_test_results[i].keys()}, 
                                              orient='index')
    df_formal_tests.index = pd.MultiIndex.from_tuples(df_formal_tests.index, names=['Model_CL', 'Test'])
    
    # --- Save Backtest Results ---
    df_backtest.to_csv(TABLES_DIR / "backtest_results.csv")
    print(f"Backtest results saved to {TABLES_DIR / 'backtest_results.csv'}")
    
    # Save formal test results
    df_formal_tests.to_csv(TABLES_DIR / "formal_test_results.csv")
    print(f"Formal test results saved to {TABLES_DIR / 'formal_test_results.csv'}")

    # --- Reporting ---
    print("--- Backtesting Summary (99% Historical VaR Baseline) ---")
    print(f"Data points: {len(df_backtest)}")
    print(f"Date Range: {df_backtest.index.min().date()} to {df_backtest.index.max().date()}")
    print("\n")

    print("Margin Breaches:")
    baseline_breaches = df_backtest['baseline_breach'].sum()
    dynamic_breaches = df_backtest['dynamic_breach'].sum()
    print(f"  - Baseline Model (Hist. VaR): {baseline_breaches} ({baseline_breaches / len(df_backtest):.2%})")
    print(f"  - Dynamic Model:              {dynamic_breaches} ({dynamic_breaches / len(df_backtest):.2%})")
    
    baseline_breach_dates = df_backtest[df_backtest['baseline_breach'] == 1].index.strftime('%Y-%m-%d').tolist()
    dynamic_breach_dates = df_backtest[df_backtest['dynamic_breach'] == 1].index.strftime('%Y-%m-%d').tolist()

    print("\nBreach Dates:")
    print(f"  - Baseline Model: {baseline_breach_dates}")
    print(f"  - Dynamic Model:  {dynamic_breach_dates}")
    print("\n")

    print("Average Margin Size (% of Price):")
    avg_baseline_margin = (df_backtest['baseline_margin'] / df_backtest['price']).mean()
    avg_dynamic_margin = (df_backtest['dynamic_margin'] / df_backtest['price']).mean()
    print(f"  - Baseline Model (Hist. VaR): {avg_baseline_margin:.2%}")
    print(f"  - Dynamic Model:              {avg_dynamic_margin:.2%}")
    print(f"  - Change:                     {((avg_dynamic_margin - avg_baseline_margin) / avg_baseline_margin):.2%}")
    print("\n")
    
    print("Margin Procyclicality (Std Dev of Daily Margin/Price Ratio Changes):")
    pro_baseline = (df_backtest['baseline_margin'] / df_backtest['price']).diff().std()
    pro_dynamic = (df_backtest['dynamic_margin'] / df_backtest['price']).diff().std()
    print(f"  - Baseline Model (Hist. VaR): {pro_baseline:.4f}")
    print(f"  - Dynamic Model:              {pro_dynamic:.4f}")
    print(f"  - Change:                     {((pro_dynamic - pro_baseline) / pro_baseline):.2%}")
    print("----------------------------------------------------------")

    return df_backtest['dynamic_breach'].sum(), (df_backtest['dynamic_margin'] / df_backtest['price']).mean(), (df_backtest['dynamic_margin'] / df_backtest['price']).diff().std()

    print("\n--- Breach Analysis ---")
    for breach_date_str in baseline_breach_dates:
        breach_date = pd.to_datetime(breach_date_str)
        start_date = breach_date - pd.Timedelta(days=3)
        end_date = breach_date + pd.Timedelta(days=3)
        print(f"\nData around baseline breach date: {breach_date_str}")
        print(df_backtest.loc[start_date:end_date, ['price', 'pnl', 'baseline_margin', 'garch_volatility']])

    for breach_date_str in dynamic_breach_dates:
        breach_date = pd.to_datetime(breach_date_str)
        start_date = breach_date - pd.Timedelta(days=3)
        end_date = breach_date + pd.Timedelta(days=3)
        print(f"\nData around dynamic breach date: {breach_date_str}")
        print(df_backtest.loc[start_date:end_date, ['price', 'pnl', 'dynamic_margin', 'im_correction_factor']])
