import sys
import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

# Import modules
from src.config import MODEL_READY_DATASET_PATH, TABLES_DIR, FIGURES_DIR, RESULTS_DIR
from src import build_dataset
from src import train_model
from src import backtest_engine
from src import generate_feature_importance
from src import create_visuals


def capture_run_metadata(steps, force):
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "steps": steps,
        "force": force,
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    # pip freeze
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        meta["pip_freeze"] = out.strip().splitlines()
    except Exception:
        meta["pip_freeze"] = []
    # git commit (if available)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))


def run_steps(steps: list[str], force: bool = False):
    print("--- Starting Pipeline ---")

    # Build
    if "build" in steps:
        print("\n--- Building Dataset ---")
        if MODEL_READY_DATASET_PATH.exists() and not force:
            print(f"Dataset exists at {MODEL_READY_DATASET_PATH}. Skipping rebuild.")
        else:
            build_dataset.main()

    # Train
    if "train" in steps:
        print("\n--- Training Model ---")
        im_cf_path = TABLES_DIR / "im_correction_factor.csv"
        if im_cf_path.exists() and not force:
            print(f"IM correction factor exists at {im_cf_path}. Skipping training.")
        else:
            train_model.main()

    # Backtest
    if "backtest" in steps:
        print("\n--- Running Backtest ---")
        bt_path = TABLES_DIR / "backtest_results.csv"
        ft_path = TABLES_DIR / "formal_test_results.csv"
        if bt_path.exists() and ft_path.exists() and not force:
            print("Backtest outputs exist. Skipping.")
        else:
            backtest_engine.run_backtest()

    # Visuals
    if "visuals" in steps:
        print("\n--- Generating Visuals ---")
        fi_plot = FIGURES_DIR / "feature_importance.png"
        if not fi_plot.exists() or force:
            generate_feature_importance.main()
        else:
            print("Feature importance plot exists. Skipping.")

        ablation_plot = FIGURES_DIR / "ablation_analysis.png"
        if not ablation_plot.exists() or force:
            create_visuals.create_ablation_plot()
        else:
            print("Ablation plot exists. Skipping.")

        sens_plot = FIGURES_DIR / "sensitivity_analysis_iv_vol_ratio.png"
        if not sens_plot.exists() or force:
            create_visuals.create_sensitivity_plot()
        else:
            print("Sensitivity plot exists. Skipping.")

    print("\n--- Pipeline Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Run the CFA Quant pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["build", "train", "backtest", "visuals", "all"],
        default=["all"],
        help="Steps to run in order",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if outputs exist",
    )
    args = parser.parse_args()

    steps = args.steps
    if "all" in steps:
        steps = ["build", "train", "backtest", "visuals"]

    capture_run_metadata(steps, args.force)
    run_steps(steps, force=args.force)


if __name__ == "__main__":
    main()
