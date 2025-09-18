# Initial Margin Models for Energy Markets: A WTI Case Study

Code and data for the paper “Initial Margin Models for Energy Markets: A WTI Case Study.” The project develops a forward‑looking, option‑implied overlay for initial margin models for WTI futures.

## Installation

1) Clone and enter the repo
```bash
git clone https://github.com/your-repo/CFA_Quant_Awards.git
cd CFA_Quant_Awards
```
2) Create and activate a virtual env
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3) Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline (skips steps if outputs exist):
```bash
python run_pipeline.py --steps all
```

Run specific steps:
```bash
python run_pipeline.py --steps build train backtest visuals
```

Force recompute even if outputs exist:
```bash
python run_pipeline.py --steps train backtest --force
```

## Reproducibility

Initialize the baseline (captures current outputs):
```bash
python scripts/repro_check.py init
```

Run the pipeline, then verify outputs match the baseline:
```bash
python run_pipeline.py --steps all
python scripts/repro_check.py check
```

Baseline files live under `results/baseline/` and store checksums and key metrics.

## Project Structure

```
CFA_Quant_Awards/
├─ data/
│  ├─ raw/
│  └─ processed/                  # model_ready_dataset.csv
├─ models/                        # trained models and scalers
├─ paper/                         # PDF of the paper
├─ results/
│  ├─ figures/
│  ├─ tables/
│  └─ baseline/                   # repro baseline checksums/metrics
├─ scripts/
│  └─ repro_check.py              # init/check reproducibility baseline
├─ src/
│  ├─ analysis/
│  │  ├─ analysis_toolkit.py
│  │  └─ backtest_ablation.py
│  ├─ SVI_SABR_engine.py
│  ├─ backtest_engine.py
│  ├─ build_dataset.py
│  ├─ config.py
│  ├─ create_visuals.py
│  ├─ data_handlers.py
│  ├─ feature_engine.py
│  ├─ futures_curve.py
│  ├─ garch_engine.py
│  ├─ generate_feature_importance.py
│  ├─ hmm_engine.py
│  ├─ model_utils.py
│  └─ train_model.py
├─ tests/
│  └─ test_data_processing.py
├─ README.md
├─ requirements.txt
└─ run_pipeline.py
```

