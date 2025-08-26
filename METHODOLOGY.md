# Methodology

This document details the data sources and quantitative procedures underpinning the dynamic initial margin framework for WTI crude oil futures options. The discussion mirrors a methodology section of an academic paper.

## 1. Data

### 1.1 Risk‑Free Rate
Daily overnight risk‑free rates are sourced from the Secured Overnight Financing Rate (SOFR) time series. Let \( r_t \) denote the SOFR on date \( t \). For non‑trading days, the most recent available rate is carried forward. Rates in the CSV file are given in percentage points and converted to decimals.

### 1.2 Futures Term Structure
Historical settlement prices for nine consecutive WTI futures contracts are used to construct the daily term structure. For a target date \( t \), the handler collects a panel
\[
\{ (C_i, P_i(t)) \}_{i=1}^N,
\]
where \( C_i \) is the contract code and \( P_i(t) \) its closing price as of \( t \).

The term‑structure features are derived as follows. Let \( P_0 \) be the true front‑month price with expiry date \( T_0 \). For maturities of one month, three months and one year beyond \( T_0 \), the spreads are
\[
S_{\Delta} = \frac{P_{\Delta} - P_0}{P_0}, \qquad \Delta \in \{1\text{M},3\text{M},1\text{Y}\}.
\]
These spreads are scaled by empirical volatilities \( \sigma_{\Delta} \in \{0.01,0.03,0.05\} \) and passed through a hyperbolic tangent normalisation
\[
N_{\Delta} = \tanh\!\left(\frac{S_{\Delta}}{\sigma_{\Delta}}\right).
\]
Price persistence is measured by
\[
\text{persistence} = \frac{1}{M-1} \sum_{j=2}^M \mathbb{1}\{P_j < P_{j-1}\},
\]
where prices are ordered by increasing expiry and \(M\) is the number of contracts. The persistence impact term is
\[
I = (2\,\text{persistence} - 1)\,\min\!\left(\left|\frac{S_{1\text{Y}}}{0.05}\right|,1\right).
\]
The Curve‑Term‑Structure Index (CTSI) combines these components
\[
\text{CTSI} = 0.25 N_{1\text{M}} + 0.35 N_{3\text{M}} + 0.25 N_{1\text{Y}} + 0.15 I,
\]
clipped to \([-1,1]\). Positive values indicate contango, negative values backwardation.

### 1.3 Options Chain
For each business day \( t \) and a fixed option expiry \( T \), a complete call and put chain is assembled. Filenames encode strike levels; valid contracts are retained when a non‑zero price is reported. The resulting dataset contains strike \( K \), option type \( \mathsf{c}/\mathsf{p} \), mid price, traded volume and open interest.

## 2. Methodology

### 2.1 SVI‑SABR Volatility Surface
The forward price \( F_t \) for expiry \( T \) is obtained from the matching futures contract. Under Black–76, the price of a call option with strike \( K \), volatility \( \sigma \) and maturity \( \tau = (T-t)/365 \) is
\[
C(F,K,\sigma,\tau,r_t) = e^{-r_t \tau}\left(F\Phi(d_1) - K\Phi(d_2)\right),
\]
where
\[
d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2\tau}{\sigma\sqrt{\tau}}, \qquad d_2 = d_1 - \sigma\sqrt{\tau}.
\]
Implied volatilities are obtained by solving \( C(F,K,\sigma,\tau,r_t) = P_{\text{obs}} \).

The stochastic‑volatility‑inspired (SVI) surface with SABR‑style parameterisation models total variance as
\[
w(k) = \tfrac{1}{2}\alpha^2 \Big(1 + \rho \frac{\nu}{\alpha}k + \sqrt{\big(\tfrac{\nu}{\alpha}k + \rho\big)^2 + 1 - \rho^2}\Big),
\]
where \( k = \ln(K/F) \) and \( \alpha,\nu,\rho \) are the curvature, vol‑of‑vol and correlation parameters. Instantaneous volatility is \( \sigma(k) = \sqrt{w(k)/\tau} \).

Calibration minimises a weighted least‑squares objective
\[
\min_{\alpha,\nu,\rho} \sum_i w_i\big(\sigma(k_i) - \hat{\sigma}_i\big)^2,
\]
subject to the arbitrage constraint \( \alpha\nu\tau(1+|\rho|) < 4 \). Weights \( w_i \) are proportional to option volume plus open interest.

### 2.2 Implied Moments
The SVI surface is integrated to extract risk‑neutral moments of the log return
\[
k = \ln\frac{S_T}{F_t}.
\]
The risk‑neutral density \( f(k) \) is inferred via the second derivative of the call price with respect to strike, and moments are computed as
\[
\text{Var}(k) = \int (k-\mu)^2 f(k)\,dK, \quad \text{Skew}(k) = \frac{\int (k-\mu)^3 f(k)\,dK}{\text{Var}(k)^{3/2}}, \quad \text{Kurt}(k) = \frac{\int (k-\mu)^4 f(k)\,dK}{\text{Var}(k)^{2}}.
\]
The at‑the‑money implied volatility \( \sigma_{\text{ATM}} = \sigma(k=0) \) is also recorded.

### 2.3 Hidden Markov Model
The CTSI series \( \{x_t\} \) is modelled with a two‑state Gaussian Markov‑switching regression
\[
x_t = \mu_{s_t} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0, \sigma^2),
\]
where \( s_t \in \{0,1\} \) follows a first‑order Markov chain. Smoothed probabilities of the low‑mean (backwardation) regime are appended as a feature.

### 2.4 Feature Engineering
For each daily observation the dataset is expanded with lag, moving‑average and momentum transforms. For any base series \( z_t \):
- Lagged value: \( z_{t-1} \)
- Two‑day moving average: \( \tfrac{1}{2}(z_t + z_{t-1}) \)
- One‑day momentum: \( z_t - z_{t-1} \)

Features are generated for \( \sigma_{\text{ATM}}, \rho, \text{Var}, \text{Skew}, \text{Kurt} \) and merged with \(F_t, r_t, T, \alpha, \nu\) and HMM outputs. Rows with missing values are removed, yielding the model‑ready dataset.

### 2.5 Neural‑Network Forecasting
The objective is to predict the next‑day change in ATM implied volatility. The target is the log ratio
\[
y_t = \ln\frac{\sigma_{\text{ATM},t+1}}{\sigma_{\text{ATM},t}}.
\]
A feed‑forward neural network (MLPRegressor) is tuned via grid search over activation \( \{\text{ReLU}, \tanh\} \), hidden‑layer sizes \([(50,25),(100,50),(50,25,10)]\) and L2 penalties \(\alpha \in \{10^{-4},10^{-3},10^{-2}\}\). TimeSeriesSplit with five folds preserves temporal order. The best model and accompanying standard‑scaler are saved.

### 2.6 Correction Factor Generation
The trained network produces predictions \( \hat{y}_t \). The implied margin correction factor is
\[
\phi_t = e^{\hat{y}_t} = \frac{\widehat{\sigma}_{\text{ATM},t+1}}{\sigma_{\text{ATM},t}}.
\]
The factor is paired with contemporaneous \( \sigma_{\text{ATM},t} \) for downstream usage.

### 2.7 Backtesting Framework
Let \( P_t \) denote the settlement price of the front‑month futures. Ten‑day 99\% historical Value‑at‑Risk is
\[
\text{VaR}_{0.99,t} = \text{quantile}_{0.01}\left(\{r_{t-j}\}_{j=1}^{10}\right), \quad r_t = \frac{P_t - P_{t-1}}{P_{t-1}}.
\]
The baseline margin is \( M^{\text{base}}_t = |\text{VaR}_{0.99,t}| P_t \). Applying the correction factor yields the dynamic margin
\[
M^{\text{dyn}}_t = M^{\text{base}}_t \phi_t.
\]
Out‑of‑sample performance is evaluated through margin breaches \( |\Delta P_t| > M_t \), average margin levels and procyclicality—defined as the standard deviation of daily margin‑to‑price changes.

---
This methodology enables a forward‑looking initial margin that adapts to projected changes in implied volatility while respecting term‑structure dynamics and regime shifts.

