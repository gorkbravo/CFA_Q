# Dynamic Initial Margin Correction Factor for CCPs

This project develops a dynamic initial margin correction factor for CCPs (Central Counterparties) that can be recalculated intraday to complement existing VaR-based margin models.

## Core Concept

The core concept is to use the relationship between forward curve steepness and option-implied risk-neutral density moments to create a real-time multiplier that adjusts baseline VaR margins more dynamically than traditional methods.

In WTI crude oil markets, steeper backwardation correlates with higher implied variance, more negative skew, and lower kurtosis (fatter tails) in the risk-neutral distribution derived from options prices.

## Implementation Approach

This project does NOT aim to replace VaR models, but to create a lightweight correction factor. The workflow is as follows:

1.  Risk departments run overnight VaR simulations as usual.
2.  During the trading day, the correction factor is continuously recalculated based on observable forward curve steepness.
3.  This multiplier is applied to adjust initial margins in real-time.

## Target Outcome

The target outcome is to demonstrate that this approach improves margin performance (better risk coverage, reduced procyclicality) while being computationally feasible for real-time CCP operations.

## To Do

*   **Refine Options Data Cleaning:** Improve the options data cleaning methodology to ensure high-quality inputs for the volatility models.
*   **Consolidate Data:** Create a script to merge the forward curve steepness data with the corresponding daily risk-neutral moments.
*   **Model Correction Factor:** Perform a statistical analysis to model the relationship between forward curve steepness and the moments, and define the margin correction factor.
*   **Simulation and Backtesting:** Create a simulation to demonstrate the factor's behavior and its potential to improve risk coverage.
