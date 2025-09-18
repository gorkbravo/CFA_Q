import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

import data_handlers
import futures_curve as fcm

# --- Consistent Test Date --- #
TEST_DATE = '2025-08-22'

def run_test(test_name, test_function):
    """Helper function to run a test and print the result."""
    print(f"--- Running Test: {test_name} ---")
    try:
        test_function()
        print(f"[PASS] {test_name}")
    except Exception as e:
        print(f"[FAIL] {test_name}")
        print(f"  -> Error: {e}")
        raise e
    finally:
        print("-" * (len(test_name) + 20))
        print()

def test_get_risk_free_rate():
    rate = data_handlers.get_risk_free_rate(TEST_DATE)
    assert isinstance(rate, float), "Rate is not a float"
    print(f"  -> Found rate for {TEST_DATE}: {rate:.4f}")

def test_get_futures_curve():
    curve_df = data_handlers.get_futures_curve(TEST_DATE)
    assert isinstance(curve_df, pd.DataFrame), "Did not return a DataFrame"
    assert not curve_df.empty, "Returned an empty DataFrame"
    assert len(curve_df) == 9, f"Expected 9 contracts, found {len(curve_df)}"
    print(f"  -> Found {len(curve_df)} contracts for {TEST_DATE}")

def test_get_option_chain():
    chain_df = data_handlers.get_option_chain(TEST_DATE, 'clv5')
    assert isinstance(chain_df, pd.DataFrame), "Did not return a DataFrame"
    assert not chain_df.empty, "Returned an empty DataFrame"
    assert 'c' in chain_df['type'].unique(), "Missing call options"
    assert 'p' in chain_df['type'].unique(), "Missing put options"
    print(f"  -> Found {len(chain_df)} options in chain for {TEST_DATE}")

def test_calculate_term_structure_features():
    curve_df = data_handlers.get_futures_curve(TEST_DATE)
    month_codes = dict(zip("FGHJKMNQUVXZ", range(1, 13)))
    def get_expiry(code):
        m = re.match(r"CL([FGHJKMNQUVXZ])(\d{2})", code)
        if not m: return pd.NaT
        month = month_codes[m.group(1)]
        year  = 2000 + int(m.group(2))
        return pd.Timestamp(year, month, 1)

    curve_df['expiration'] = curve_df['contract'].apply(get_expiry)
    front_month_expiry = curve_df['expiration'].min()
    features = fcm.calculate_term_structure_features(curve_df, front_month_expiry)
    assert isinstance(features, dict), "Did not return a dictionary"
    assert 'term_structure_index' in features, "Missing 'term_structure_index'"
    print(f"  -> Calculated CTSI: {features['term_structure_index']:.4f}")

if __name__ == "__main__":
    print("="*50)
    print("Starting Pipeline Test Suite (Parts 1 & 2)")
    print("="*50)
    
    run_test("Risk-Free Rate Handler", test_get_risk_free_rate)
    run_test("Futures Curve Handler", test_get_futures_curve)
    run_test("Option Chain Assembler", test_get_option_chain)
    run_test("Term Structure Feature Calculation", test_calculate_term_structure_features)
    
    print("="*50)
    print("Test Suite Finished Successfully.")
    print("Proceeding to Part 3: End-to-End Engine Test")
    print("="*50)