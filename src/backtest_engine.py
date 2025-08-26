import pandas as pd
import numpy as np
from pathlib import Path
import sys

def get_daily_closing_prices(futures_dir: Path, contract_symbol: str) -> pd.DataFrame:
    try:
        contract_file = next(futures_dir.glob(f"{contract_symbol}*.csv"))
    except StopIteration:
        print(f"Error: Futures contract file for {contract_symbol} not found.")
        return pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))

    df = pd.read_csv(contract_file, header=1)
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns={'date time': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df = df.set_index('date')

    if 'close' in df.columns:
        return df[['close']].rename(columns={'close': 'price'})
    else:
        print(f"Error: 'close' column not found in {contract_file.name}.")
        return pd.DataFrame(columns=['price']).set_index(pd.to_datetime([]))

def run_backtest():
    # --- Load Data ---
    df_im = pd.read_csv(RESULTS_DIR / "im_correction_factor.csv", parse_dates=['date'])
    df_im.set_index('date', inplace=True)

    df_prices = get_daily_closing_prices(FUTURES_DIR, 'CLV25')
    if df_prices.empty:
        print("Could not load futures prices. Aborting backtest.")
        return

    # --- Calculate P&L and New VaR-based Baseline Margin ---
    df_prices['returns'] = df_prices['price'].pct_change()
    # 10-day, 99% VaR calculation
    df_prices['var_99'] = df_prices['returns'].rolling(window=10).quantile(0.01)
    # The margin is the absolute VaR percentage multiplied by the current price
    df_prices['baseline_margin'] = df_prices['var_99'].abs() * df_prices['price']
    df_prices['pnl'] = df_prices['price'].diff().shift(-1)

    # --- Join and Clean ---
    # We only need the price, pnl, and the new baseline margin from the prices df
    df_backtest = df_im.join(df_prices[['price', 'pnl', 'baseline_margin']], how='inner')
    df_backtest.dropna(inplace=True)

    if df_backtest.empty:
        print("Backtest dataframe is empty. This is likely due to a date mismatch between the options data and the futures price data.")
        return

    # --- Calculations ---
    # The dynamic margin now modifies the new VaR-based baseline
    df_backtest['dynamic_margin'] = df_backtest['baseline_margin'] * df_backtest['im_correction_factor']
    df_backtest['baseline_breach'] = (df_backtest['pnl'].abs() > df_backtest['baseline_margin']).astype(int)
    df_backtest['dynamic_breach'] = (df_backtest['pnl'].abs() > df_backtest['dynamic_margin']).astype(int)

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

if __name__ == "__main__":
    # Define paths at the top level for the script to run
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT_DIR / "Data_act"
    RESULTS_DIR = ROOT_DIR / "results_testing"
    FUTURES_DIR = DATA_DIR / "Futures_curve_time_series"
    run_backtest()
