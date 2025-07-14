
"""
options_data_cleaner.py - Cleans raw options data.
"""
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Raw"
CLEANED_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Cleaned"
CLEANED_OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

def clean_options_file(file_path: Path):
    """
    Reads a raw options CSV, cleans it, and saves it to the Cleaned directory.
    """
    df = pd.read_csv(file_path)

    # 1. Basic Filtering
    df = df[(df["Volume"] > 0) & (df["Open Int"] > 0)]

    # 2. Use mid-price
    df["Mid"] = (df["Bid"] + df["Ask"]) / 2

    # 3. Remove unnecessary columns
    df = df[["Strike", "Mid", "Volume", "Open Int", "Type"]]

    # 4. Save cleaned file
    cleaned_file_path = CLEANED_OPTIONS_DIR / file_path.name
    df.to_csv(cleaned_file_path, index=False)
    print(f"[CLEAN] {file_path.name} -> {cleaned_file_path.name}")

def main():
    """
    Cleans all raw options files.
    """
    for file_path in RAW_OPTIONS_DIR.glob("*.csv"):
        clean_options_file(file_path)

if __name__ == "__main__":
    main()
