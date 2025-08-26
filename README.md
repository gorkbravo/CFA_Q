# Dynamic Initial Margin using a Neural Network Correction Factor

This project implements and evaluates a dynamic initial margin (IM) model for WTI Crude Oil futures options. The core of the model is a Neural Network that predicts a correction factor based on market dynamics, aiming to create a more risk-sensitive margining system.

## Project Objective

The goal is to develop a forward-looking IM framework that adjusts to predicted near-term volatility. This is achieved by calculating a dynamic correction factor, defined as the ratio of the next day's predicted at-the-money (ATM) implied volatility (IV) to the current day's ATM IV.

## Methodology

The project follows a multi-step pipeline:

1.  **SVI-SABR Model Calibration & Data Preparation:** An enhanced Stochastic Volatility Inspired (SVI) model is calibrated to the options data for each trading day. This process, along with the calculation of term structure and Hidden Markov Model (HMM) features, and subsequent feature engineering, culminates in a model-ready dataset (`Data_act/Final/model_ready_dataset.csv`).

2.  **Neural Network Training:** An MLP Regressor is trained to predict the log-ratio of the next day's ATM IV to the current day's ATM IV using the prepared dataset. The trained model (`models/nn_model.joblib`) and its feature scaler (`models/scaler.joblib`) are saved for inference.

3.  **IM Correction Factor Generation:** The trained Neural Network model is used to generate daily IM correction factors based on its predictions. These factors are stored in `results_testing/im_correction_factor.csv`.

4.  **Backtesting:** A backtest is performed to compare the performance of the dynamic IM model against a static baseline model. The backtest evaluates effectiveness based on metrics such as margin breaches, average margin size, and procyclicality.

## Project Structure

```
CFA_Quant_Awards/
│
├── Data_act/              # Raw and processed data (Futures, Options, OVX, SOFR)
│   └── Final/
│       └── model_ready_dataset.csv # Consolidated dataset for modeling
│
├── src/                   # Source code for the pipeline components
│   ├── SVI_SABR_engine.py # SVI-SABR model calibration and related calculations
│   ├── build_dataset.py   # Orchestrates data preparation, SVI, HMM, and feature engineering
│   ├── data_handlers.py   # Handles data loading and initial processing
│   ├── feature_engine.py  # Generates time-series features
│   ├── futures_curve.py   # Futures curve related calculations
│   ├── hmm_engine.py      # Hidden Markov Model calculations
│   ├── IM_engine.py       # Generates IM correction factors using the NN model
│   ├── NN_engine.py       # Neural network training and hyperparameter tuning
│   └── backtest_engine.py # Backtesting and performance analysis
│
├── models/                # Saved trained models and scalers
│   ├── nn_model.joblib
│   └── scaler.joblib
│
├── results_testing/       # Quantitative results and analysis outputs
│   ├── daily_svi_analysis.csv # Output from SVI-SABR engine
│   ├── feature_importance.png # Example visualization
│   └── im_correction_factor.csv # Output from IM engine
│
├── Other/                 # Miscellaneous analysis and visualization scripts
│   ├── analyze_fits.py
│   ├── generate_fit_visuals.py
│   └── visualize_parameters.py
│
├── Data_Requirements.md   # Document detailing data needs for a full implementation
├── NN_IM_Proposal.md      # Proposal document for the Neural Network IM approach
├── PROJECT_BRIEF.md       # High-level project brief
├── README.md              # This file
└── test_pipeline.py       # Unit tests for data handling and feature calculation
```

## How to Run the Pipeline

To run the full pipeline and generate the backtest results, execute the following Python scripts in sequence from the project root directory:

1.  **Prepare the Dataset:**
    ```bash
    python src/build_dataset.py
    ```
    This script orchestrates the SVI-SABR engine, HMM, and feature engineering to create `Data_act/Final/model_ready_dataset.csv`.

2.  **Generate IM Correction Factors:**
    ```bash
    python src/IM_engine.py
    ```
    This script loads the trained Neural Network model, makes predictions on the prepared dataset, and generates `results_testing/im_correction_factor.csv`.
    *(Note: Ensure `models/nn_model.joblib` and `models/scaler.joblib` exist. If not, run `python src/NN_engine.py` first to train the model.)*

3.  **Run the Backtest:**
    ```bash
    python src/backtest_engine.py
    ```
    This script performs the backtest using the generated IM correction factors and prints the summary results to the console.

## Current Status & Backtest Results (as of last run)

The baseline VaR model is currently configured as a **10-day simple historical VaR**.

Here are the latest backtesting summary results:

```
--- Backtesting Summary (99% Historical VaR Baseline) ---
Data points: 286
Date Range: 2024-07-05 to 2025-08-22


Margin Breaches:
  - Baseline Model (Hist. VaR): 49 (17.13%)
  - Dynamic Model:              49 (17.13%)


Average Margin Size (% of Price):
  - Baseline Model (Hist. VaR): 2.50%
  - Dynamic Model:              2.49%
  - Change:                     -0.21%


Margin Procyclicality (Std Dev of Daily Margin/Price Ratio Changes):
  - Baseline Model (Hist. VaR): 0.0066
  - Dynamic Model:              0.0067
  - Change:                     2.32%
----------------------------------------------------------
```

## Data Requirements

The data used in this repository is a limited sample for testing and demonstration purposes. For a full, robust implementation, a more extensive dataset is required, as detailed in the `Data_Requirements.md` file.