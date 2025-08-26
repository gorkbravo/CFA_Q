"""
data_handlers.py
=========================
Central handlers for ingesting and preparing time-series data for the margin model pipeline.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# --- Constants and Caching --- #

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data_act"
SOFR_FILE = DATA_DIR / "SORF" / "SOFR.csv"

_sofr_cache = None

def _load_sofr_data() -> pd.DataFrame:
    """Loads the SOFR data from CSV, caching it after the first read."""
    global _sofr_cache
    if _sofr_cache is None:
        df = pd.read_csv(SOFR_FILE)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df = df.set_index('observation_date')
        # The rates are in percent, so convert to decimal
        df['SOFR'] = df['SOFR'] / 100.0
        _sofr_cache = df
    return _sofr_cache

def get_risk_free_rate(date: str | pd.Timestamp) -> float:
    """
    Provides the risk-free rate for a given date.

    Args:
        date: The date for which to retrieve the rate.

    Returns:
        The risk-free rate as a float.
    """
    sofr_data = _load_sofr_data()
    target_date = pd.to_datetime(date)
    
    # Use 'asof' to get the most recent rate if the target date is not a trading day
    rate = sofr_data['SOFR'].asof(target_date)
    
    if pd.isna(rate):
        raise ValueError(f"Risk-free rate not available for date {target_date} or earlier.")
        
    return rate

# --- Futures Data Handling --- #

FUTURES_DIR = DATA_DIR / "Futures_curve_time_series"

_futures_contracts_cache = None

def _load_futures_contracts() -> dict[str, pd.DataFrame]:
    """Loads all futures contract time series into a cached dictionary."""
    global _futures_contracts_cache
    if _futures_contracts_cache is None:
        contracts = {}
        for f in FUTURES_DIR.glob("*.csv"):
            # Extract contract symbol, e.g., 'CLV25'
            symbol = f.name.split("_")[0]
            df = pd.read_csv(f, header=1) # Skip the first row which contains the symbol
            df.columns = [col.strip().lower() for col in df.columns]
            df = df.rename(columns={"date time": "date"})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.set_index('date')
            contracts[symbol] = df
        _futures_contracts_cache = contracts
    return _futures_contracts_cache

def get_futures_curve(date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Constructs the full futures term structure for a given date.

    Args:
        date: The date for which to construct the curve.

    Returns:
        A DataFrame representing the futures curve for that day, 
        with columns ['contract', 'price'].
    """
    contracts = _load_futures_contracts()
    target_date = pd.to_datetime(date)
    
    curve_data = []
    for symbol, df in contracts.items():
        price_at_date = df['close'].asof(target_date)
        if pd.notna(price_at_date):
            curve_data.append({'contract': symbol, 'price': price_at_date})
            
    if not curve_data:
        raise ValueError(f"Futures curve data is not available for date {target_date}.")
        
    return pd.DataFrame(curve_data)

# --- Options Data Handling --- #

OPTIONS_DIR = DATA_DIR / "Options_time_series"

def get_option_chain(date: str | pd.Timestamp, expiry_symbol: str) -> pd.DataFrame:
    """
    Assembles a full option chain for a given date and expiry.

    Args:
        date: The date for which to assemble the chain.
        expiry_symbol: The symbol for the option expiry (e.g., 'clv5').

    Returns:
        A DataFrame representing the option chain for that day, with columns
        ['strike', 'type', 'price', 'volume', 'open_interest'].
    """
    target_date = pd.to_datetime(date)
    chain_data = []

    # Process both Calls and Puts
    for opt_type in ['Calls', 'Puts']:
        type_char = 'c' if opt_type == 'Calls' else 'p'
        directory = OPTIONS_DIR / opt_type
        
        for f in directory.glob(f"{expiry_symbol}*.csv"):
            try:
                # Extract strike from filename, e.g., clv5_605c -> 605
                strike_str = f.name.split("_")[1][:-1]
                strike = float(strike_str) / 10 if len(strike_str) > 2 else float(strike_str)
            except (IndexError, ValueError):
                continue # Skip files with unexpected naming conventions

            df = pd.read_csv(f)
            df.columns = [col.strip().lower() for col in df.columns]
            df = df.rename(columns={"time": "date"})
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            # Find the data for the target date
            day_data = df[df['date'] == target_date]
            if day_data.empty:
                continue

            # Extract relevant metrics
            # Assuming 'last' is the price, and checking for existence of other columns
            price = day_data['last'].iloc[0]
            volume = day_data['volume'].iloc[0] if 'volume' in day_data.columns else 0
            open_interest = day_data['open int'].iloc[0] if 'open int' in day_data.columns else 0

            if pd.notna(price) and price > 0: # Only include options with a valid price
                chain_data.append({
                    'strike': strike,
                    'type': type_char,
                    'price': price,
                    'volume': volume,
                    'open_interest': open_interest
                })

    if not chain_data:
        raise ValueError(f"Option chain data is not available for date {target_date} and expiry {expiry_symbol}.")

    return pd.DataFrame(chain_data).sort_values('strike').reset_index(drop=True)
