
"""
options_data_cleaner.py - Cleans raw options data.
"""
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Raw"
CLEANED_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Cleaned"
CLEANED_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# IV_WIDTH_CUTOFF = 0.50 # Not applicable with current raw data (no bid/ask)
# MONEYNESS_RANGE = (0.8, 1.2) # To be handled in vol_engine scripts

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def clean_options_file(file_path: Path):
    """
    Reads a raw options CSV, cleans it, and saves it to the Cleaned directory.
    """
    df = pd.read_csv(file_path)
    df = normalise_columns(df)

    try:
        # 1. Basic Filtering
        df = df[(df["volume"] > 0) & (df["open_int"] > 0)]
        df = df[df.type.str.startswith('c')]
        df['strike'] = df['strike'].astype(str).str.extract(r'(\d+\.?\d*)')[0] # Extract numeric part
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df = df[df.strike > 0]

        # 2. No bid-ask spread filtering as bid/ask columns are not in raw data.
        #    The 'Mid' column is directly available.

        # 3. Select and rename columns for consistency with vol_engines
        df = df[["strike", "mid", "volume", "openinterest", "type"]]

        # 4. Save cleaned file
        cleaned_file_path = CLEANED_OPTIONS_DIR / file_path.name
        df.to_csv(cleaned_file_path, index=False)
        print(f"[CLEAN] {file_path.name} -> {cleaned_file_path.name}")
    except KeyError as e:
        print(f"[ERROR] Skipping file {file_path.name} due to missing column: {e}. Columns available: {df.columns.tolist()}")

def main():
    """
    Cleans all raw options files.
    """
    for file_path in RAW_OPTIONS_DIR.glob("*.csv"):
        clean_options_file(file_path)

if __name__ == "__main__":
    main()
