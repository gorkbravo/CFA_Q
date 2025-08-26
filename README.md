# Dynamic Initial Margin using a Neural Network Correction Factor

This project implements and evaluates a dynamic initial margin (IM) model for WTI Crude Oil futures options. The core of the model is a Neural Network that predicts a correction factor based on market dynamics, aiming to create a more risk-sensitive margining system.

## Project Objective

The goal is to develop a forward-looking IM framework that adjusts to predicted near-term volatility. This is achieved by calculating a dynamic correction factor, defined as the ratio of the next day's predicted at-the-money (ATM) implied volatility (IV) to the current day's ATM IV.

## Methodology

The project follows a multi-step pipeline:

1.  **SVI-SABR Model Calibration:** An enhanced Stochastic Volatility Inspired (SVI) model is calibrated to the options data for each trading day. This model is used to extract key features, including the ATM Implied Volatility, Forward Price (F), and Time to Expiry (T). The enhancements include a weighted MSE loss function, an arbitrage violation penalty, and a sequential parameter initialization scheme.

2.  **Feature Engineering & Data Consolidation:** The calibrated SVI-SABR parameters are combined with market regime probabilities from a Hidden Markov Model (HMM) to create a daily consolidated dataset (`Data/Stats/consolidated_data.csv`).

3.  **Neural Network Prediction:** An MLP Regressor is trained to predict the next day's ATM IV using the consolidated data. The trained model (`models/nn_model.joblib`) and its feature scaler (`models/scaler.joblib`) are saved for inference.

4.  **IM Correction Factor Calculation:** The trained NN model is used to generate daily IM correction factors, which are stored in `Data/Stats/im_correction_factor.csv`.

5.  **Backtesting:** A backtest is performed to compare the performance of the dynamic IM model against a static baseline model, evaluating its effectiveness in different market scenarios.

## Project Structure

```
CFA_Quant_Awards/
│
├── Data/                  # Raw and processed data
├── src/                   # Source code for the pipeline
│   ├── SVI_SABR_engine.py       # SVI-SABR model calibration
│   ├── data_consolidator.py   # Feature engineering and aggregation
│   ├── NN_engine.py           # Neural network training
│   ├── IM_engine.py           # IM correction factor calculation
│   └── backtest_engine.py     # Backtesting and performance analysis
│
├── models/                # Saved trained models
│   ├── nn_model.joblib
│   └── scaler.joblib
│
├── results_testing/       # Quantitative results and analysis
├── fit_visuals/           # Visualizations of model fits
│
├── run.py                 # Main execution script for the pipeline
├── analyze_fits.py        # Script for analyzing SVI-SABR fit quality
├── Data_Requirements.md   # Document detailing data needs for a full implementation
└── README.md              # This file
```

## How to Run

The entire pipeline can be executed by running the main script from the project root directory:

```bash
python run.py
```

This will execute the following modules in sequence:
1.  `SVI_SABR_engine.py`
2.  `data_consolidator.py`
3.  `NN_engine.py`
4.  `IM_engine.py`
5.  `backtest_engine.py`

## Data Requirements

The data used in this repository is a limited sample for testing and demonstration purposes. For a full, robust implementation, a more extensive dataset is required, as detailed in the `Data_Requirements.md` file.
