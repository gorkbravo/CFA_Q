import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_DIR = RESULTS_DIR / "baseline"


FILES_TO_TRACK = [
    TABLES_DIR / "im_correction_factor.csv",
    TABLES_DIR / "backtest_results.csv",
    TABLES_DIR / "formal_test_results.csv",
    TABLES_DIR / "feature_importance.csv",
]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_metrics(backtest_csv: Path) -> dict:
    df = pd.read_csv(backtest_csv, parse_dates=[0])
    if not {"price", "dynamic_margin"}.issubset(df.columns):
        return {}
    metrics = {}
    if "dynamic_breach" in df.columns:
        metrics["dynamic_breaches"] = int(df["dynamic_breach"].sum())
    ratio = (df["dynamic_margin"] / df["price"]).astype(float)
    metrics["avg_margin_ratio"] = float(ratio.mean())
    metrics["procyclicality_std"] = float(ratio.diff().std())
    return metrics


def init_baseline() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for p in FILES_TO_TRACK:
        if p.exists():
            checksums[str(p.relative_to(ROOT))] = {
                "sha256": sha256sum(p),
                "size": p.stat().st_size,
            }
    (BASELINE_DIR / "checksums.json").write_text(json.dumps(checksums, indent=2))

    # Metrics
    bt_csv = TABLES_DIR / "backtest_results.csv"
    if bt_csv.exists():
        metrics = compute_metrics(bt_csv)
        (BASELINE_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Baseline initialized under {BASELINE_DIR}")


def check_against_baseline() -> int:
    cfile = BASELINE_DIR / "checksums.json"
    if not cfile.exists():
        print("No baseline found. Run: python scripts/repro_check.py init")
        return 2
    baseline = json.loads(cfile.read_text())
    diffs = []
    for rel, meta in baseline.items():
        path = ROOT / rel
        if not path.exists():
            diffs.append((rel, "missing"))
            continue
        cur_hash = sha256sum(path)
        if cur_hash != meta.get("sha256"):
            diffs.append((rel, "hash_mismatch"))

    if diffs:
        print("Repro check FAILED. Differences detected:")
        for rel, kind in diffs:
            print(f" - {rel}: {kind}")
        return 1
    else:
        print("Repro check PASSED. Outputs match baseline.")
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducibility checker")
    parser.add_argument("command", choices=["init", "check"], help="Initialize or check baseline")
    args = parser.parse_args(argv)
    if args.command == "init":
        init_baseline()
        return 0
    else:
        return check_against_baseline()


if __name__ == "__main__":
    sys.exit(main())

