# Data Requirements for Full Implementation

This document outlines the data required for the full implementation, training, and generalization of the Neural Network-based Initial Margin (IM) correction factor model. The data listed below is essential for building a robust model that can adapt to various market conditions.

## 1. WTI Crude Oil Futures Data

*   **Data Type:** Historical Futures Prices
*   **Description:** This data is crucial for constructing the futures curve, which is a primary input for the SVI-SABR model to determine the forward price (F) for different option expiries.
*   **Granularity:** Intraday (e.g., 1-minute, 5-minute, or tick-level data). High-frequency data is necessary to accurately capture the term structure dynamics throughout the trading day.
*   **Historical Depth:** A minimum of **2-3 years** of continuous historical data. This timeframe should ideally cover diverse market environments (e.g., high and low volatility, different structural regimes, economic cycles) to ensure the model's generalizability.
*   **Contracts:** All listed monthly and quarterly futures contracts for WTI Crude Oil. Access to the full term structure is essential.

## 2. WTI Crude Oil Options Data

*   **Data Type:** Historical Options Prices (Calls and Puts)
*   **Description:** This is the most critical dataset, as it forms the basis for calibrating the SVI-SABR volatility model. The model fits the implied volatility smile, which is the source of the ATM IV feature for the neural network.
*   **Granularity:** Intraday, with timestamps synchronized with the futures data.
*   **Historical Depth:** A minimum of **2-3 years**, matching the futures data history.
*   **Contracts:** All available option expiries and strike prices for WTI Crude Oil. A rich set of strikes is required for a stable and accurate calibration of the volatility smile.

## 3. Market Regime Data

*   **Data Type:** Market State Probabilities or Indicators
*   **Description:** The model uses features derived from a Hidden Markov Model (HMM) to represent the prevailing market regime (e.g., calm, volatile, trending). While the HMM can be trained on historical price data, alternative sources for market state are also suitable.
*   **Source:** Can be generated from the historical futures data or sourced from a third-party provider specializing in market analytics.
*   **Granularity:** Daily.
*   **Historical Depth:** A minimum of **2-3 years**, aligned with the other datasets.

## 4. Risk-Free Interest Rate Data

*   **Data Type:** Historical Risk-Free Rates
*   **Description:** A risk-free interest rate curve is a required input for the options pricing and SVI-SABR calibration models.
*   **Source:** US Treasury yields or SOFR (Secured Overnight Financing Rate) are standard proxies.
*   **Granularity:** Daily.
*   **Historical Depth:** A minimum of **2-3 years**, aligned with the other datasets.
