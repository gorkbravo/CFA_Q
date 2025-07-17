import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_OPTIONS_DIR = ROOT_DIR / "Data" / "Options_data" / "Raw"
RAW_BACKUP_DIR = ROOT_DIR / "Data" / "Options_data" / "Raw_backup"

for file_path in RAW_BACKUP_DIR.glob("*.csv"):
    if "_cleaned.csv" not in file_path.name:
        try:
            shutil.move(str(file_path), str(RAW_OPTIONS_DIR / file_path.name))
            print(f"Moved: {file_path.name}")
        except Exception as e:
            print(f"Error moving {file_path.name}: {e}")
