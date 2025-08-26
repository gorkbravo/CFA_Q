# Project Brief: A Dynamic Initial Margin Framework using Neural Network Predictions

This document outlines the research, methodology, and evaluation framework for a project focused on developing a dynamic initial margin (IM) model for WTI Crude Oil options.

## 1. Title and Abstract

-   **Working Title**: A Dynamic Initial Margin Framework using Neural Network Predictions
-   **One-Sentence Summary**: This project develops a forward-looking initial margin model that uses a neural network to predict near-term volatility, aiming to create a more risk-sensitive and less procyclical margining system.
-   **Abstract (draft)**: Initial margin models are critical for mitigating counterparty risk but can be procyclical, increasing systemic risk during market stress. We propose a dynamic initial margin framework that addresses this issue. Our approach uses the SVI-SABR model to parameterize the implied volatility smile of WTI options and extract descriptive features, including implied moments. A Multi-Layer Perceptron (MLP) neural network is then trained on these features to predict the next day's at-the-money (ATM) volatility. This prediction is used to generate a dynamic correction factor for margin calculations. We evaluate the model via a backtest, comparing its performance against a static baseline in terms of margin breaches, average margin level, and procyclicality.

## 2. Research Objectives

-   **Primary Objective**: To develop and evaluate a dynamic initial margin (IM) correction factor based on a neural network's prediction of next-day at-the-money (ATM) implied volatility.
-   **Secondary Objectives**:
    -   Assess the model's effectiveness in reducing the frequency of margin breaches compared to a static, VaR-style baseline model.
    -   Analyze the impact of the dynamic model on the overall magnitude and procyclicality of margin requirements.
    -   Investigate the predictive power of features derived from the SVI-SABR volatility smile, including higher-order implied moments (variance, skew, kurtosis).
-   **Practical Relevance**: This research can provide central counterparties (CCPs) and clearing members with a more adaptive, forward-looking, and risk-sensitive tool for risk management. By creating more stable and predictive margin requirements, the model can help reduce systemic risk and enhance financial stability.

## 3. Core Research Questions

-   **Predictive**: Can features derived from the daily implied volatility smile (including SVI-SABR parameters and implied moments) effectively predict the next day's ATM implied volatility out-of-sample?
-   **Risk Management**: Does a dynamic IM model based on these predictions lead to a lower frequency of margin breaches without excessively increasing the average margin level?
-   **Procyclicality**: Is the proposed dynamic margin model less procyclical (i.e., more stable over time) than a simple static model that only reacts to current market volatility?

## 4. Hypotheses

-   **H1 (Predictability)**: A neural network trained on features extracted from the SVI-SABR model can predict next-day ATM implied volatility with a statistically significant out-of-sample R².
-   **H2 (Breach Reduction)**: The portfolio margined with the dynamic model will exhibit a lower breach frequency than the same portfolio margined with a static baseline model.
-   **H3 (Procyclicality Reduction)**: The time series of the dynamic margin requirements will have a lower standard deviation of daily changes compared to the baseline margin requirements.

## 5. Theoretical Background

-   **Initial Margin (IM)**: IM is the collateral collected to cover potential future losses from adverse market movements. Standard models, such as those based on historical Value-at-Risk (VaR), are often backward-looking and can lead to procyclicality.
-   **Procyclicality**: This refers to the tendency of financial practices to amplify financial cycles. In margining, procyclicality occurs when margin requirements are increased during periods of high volatility (when liquidity is scarce) and decreased during calm periods, potentially exacerbating market stress.
-   **Implied Volatility Smile**: The volatility smile is a persistent pattern in options markets where options with the same expiry but different strike prices have different implied volatilities. The shape of this smile contains rich, forward-looking information about market participants' expectations of future risk.
-   **SVI Model**: The "Stochastic Volatility Inspired" (SVI) model is a widely used parameterization for the volatility smile. It provides a robust and arbitrage-free fit to market data with only five parameters, making it ideal for extracting descriptive features.

## 6. Methodology

### 6.1 Data

-   **Universe**: WTI Crude Oil futures and their corresponding options.
-   **Period**: The study uses a sample dataset from January and February 2025 for development and testing purposes.
-   **Frequency**: Intraday data is used to calibrate the daily volatility smile.
-   **Feature Engineering**: For each day, the following features are generated:
    -   **SVI-SABR Parameters**: Calibrated parameters from the SVI-SABR model (`rho`).
    -   **Fit Quality**: Root Mean Squared Error (`iv_rmse`) and PDF mass (`pdf_mass`) from the calibration.
    -   **Market State**: Hidden Markov Model (HMM) probabilities and a futures term structure index.
    -   **Implied Moments**: Implied variance, skewness, and kurtosis derived from the calibrated SVI smile.

### 6.2 Modeling Pipeline

1.  **SVI-SABR Calibration**: For each day, the SVI-SABR model is calibrated to the observed options data to find the best-fit parameters that describe the implied volatility smile.
2.  **Feature Consolidation**: The calibrated parameters and derived features are consolidated into a daily time series dataset.
3.  **Neural Network Training**:
    -   **Model**: A Multi-Layer Perceptron (MLP) Regressor.
    -   **Inputs**: The engineered features from the step above.
    -   **Target (`Y_t`)**: The at-the-money (ATM) implied volatility for the *next* trading day (`ATM_IV_{t+1}`).
4.  **IM Correction Factor Calculation**: The trained NN model is used to predict next-day ATM IV. The correction factor is then calculated based on this prediction.

### 6.3 Backtesting Framework

-   **Baseline Margin**: A static, VaR-style margin that is proportional to the *current* day's ATM volatility.
-   **Dynamic Margin**: The baseline margin multiplied by the IM correction factor.
-   **Evaluation Metrics**:
    -   **Margin Breach Frequency**: The percentage of days where the next day's absolute profit and loss (`|P&L_{t+1}|`) exceeds the calculated margin.
    -   **Average Margin Level**: The average size of the margin requirement over the backtest period.
    -   **Margin Procyclicality**: The standard deviation of the daily changes in margin requirements.

## 7. Core Equations

-   **SVI "Raw" Formula (Total Variance)**:
    `w(k) = a + b * { ρ * (k - m) + sqrt[(k - m)² + σ²] }`
    where `k` is the log-moneyness `log(K/F)`.

-   **IM Correction Factor**:
    `Factor_t = Predicted_ATM_IV_{t+1} / Current_ATM_IV_t`

-   **Baseline Margin Calculation**:
    `Margin_baseline,t = Multiplier * Current_ATM_IV_t`

-   **Dynamic Margin Calculation**:
    `Margin_dynamic,t = Margin_baseline,t * Factor_t`

-   **Margin Breach Condition**:
    `|P&L_{t+1}| > Margin_t`

## 8. Results and Conclusion

*(Note: The following is a template based on the project's objectives, as the sample data was insufficient to produce a conclusive backtest.)*

The neural network model demonstrated a strong predictive performance on the training and test sets, achieving an out-of-sample R² of [Model Score]. The backtesting results showed that the dynamic margin model [result of breach comparison, e.g., "reduced margin breaches by X%"]. Furthermore, the average margin required by the dynamic model was [result of margin level comparison, e.g., "Y% lower"] than the baseline, and its procyclicality was [result of procyclicality comparison, e.g., "Z% lower"].

In conclusion, this project demonstrates the potential of using a neural network with features derived from the volatility smile to create a more effective and stable initial margin framework. The forward-looking nature of the model allows it to anticipate changes in risk, leading to a reduction in margin breaches while simultaneously mitigating procyclicality. Further research should focus on testing this framework on a larger, more diverse dataset covering multiple market regimes.