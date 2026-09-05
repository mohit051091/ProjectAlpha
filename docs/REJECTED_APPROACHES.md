# Rejected Ideas Registry

This registry documents rejected approaches, parameters, and methodologies to prevent repeated proposals.

---

## 2026-06-11 — Premature Parameter Sweeps & Threshold Optimization on Single-Day Data

* **Rejected Approach**: Grid-sweeping 5 hold horizons and 6 alert windows on the current 2-day/single-day dataset.
* **Reason for Rejection**: 
  * **Overfitting Risk**: With only 1-2 days of historical data, any "optimized" parameters (e.g. 120-minute hold with 180-minute alert expiration) are statistically guaranteed to be fit to the specific noise of those specific days (May 29 / June 3 / June 9 / June 10).
  * **Regime Generalization Failure**: Single-day data shares the same market-wide volatility, trend, and news regimes. Cross-sectional splits do not measure temporal generalization.
* **Alternative/Correct Path**: Halt all parameter optimization and feature sweeps. Focus exclusively on acquiring a minimum of 30+ days (target 60+ days) of walk-forward historical data first.

---

## 2026-06-11 — Uncontrolled ROC Momentum Trigger Implementation

* **Rejected Approach**: Implementing and optimizing the Alert-Trigger framework (ML Alert + 9-period ROC Trigger >= 1% / <= -1%) as a core model assumption without a baseline control.
* **Reason for Rejection**: 
  * **Entry/Selection Bias**: Entering a trade only after the price has already moved 1% in the desired direction introduces a strong selection bias. If the backtest does not align the return window and labels exactly with the trigger time ($T_{\text{entry}}$), it will output artificially inflated results.
  * **Attribution Lack**: If the ROC filter is required to make the signal viable, the ROC filter might be doing all the predictive work, rendering the complex ML model redundant.
* **Alternative/Correct Path**: 
  * Keep the raw ML model predictions as the baseline.
  * When more data is available, run a strict control test: **ROC >= 1% alone** vs. **ML Alert + ROC >= 1%**.
  * The return horizons and labels for the combined system must start strictly at the trigger time ($T_{\text{entry}}$).

---

## 2026-06-13 — Fixed Holding Horizon (e.g. 180-Minute Hold)

* **Rejected Approach**: Evaluating backtest and model performance using a fixed holding window (e.g., 180 minutes).
* **Reason for Rejection**: 
  * **Premature Exits**: Exited trades prematurely during strong intraday trend days, capping gains (e.g., ZEEL on June 11, where a 180m exit cut off the rally to the day's high).
  * **Mismatched Objectives**: The primary objective is to evaluate the model's ability to identify high-mover stocks early in the day (out of 437 candidates). A fixed exit penalizes early entries that subsequently trend all day, complicating the evaluation of selection accuracy.
* **Alternative/Correct Path**: Use an infinite-till-EOD hold horizon, and track:
  1. Entry timing and price.
  2. Maximum Favorable Excursion (MFE) to measure the peak move.
  3. Maximum Adverse Excursion (MAE) to measure drawdown risk.
  4. EOD close return.

---

## 2026-06-13 — Z-Score Cross-Sectional Normalization of Turnover (Value Delta)

* **Rejected Approach**: Applying Z-score cross-sectional normalization to `delta_value_1m` (net turnover).
* **Reason for Rejection**: 
  * **Scale Degradation**: Z-scoring standardizes volume to relative standard deviation units, removing the absolute size of the trade. However, in equities, a stock-day rally requires an **absolute minimum size of capital** (e.g. $\ge 30$ Lakhs INR) to sustain the move. 
  * **Noisy Breaks**: A stock with a +3.0 Z-score on a total volume of 1 Lakh INR is still extremely illiquid and will mean-revert. Raw turnover in INR has 4x higher signal separation (0.111) compared to Z-scored turnover (0.029).
* **Alternative/Correct Path**: Keep `delta_value_1m` in raw INR to filter on absolute capital entry. Use Z-scores only for scale-free features like `aggressor_ratio` and `imbalance_top5` (which see 75% and 138% separation boosts respectively).


