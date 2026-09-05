# Expert Review: Order Flow & Pre-Move Detection System
**Reviewed:** 2026-06-11 | **Reviewer:** Senior Quant / 35+ yr perspective

---

## TLDR — Honest Verdict

> **You are doing surprisingly well for a first attempt.** The pipeline discipline, the contamination incident handling, the review/rebuttal cycle, and the methodological choices (stock-level CV, LR-first, shuffle test) are genuinely professional. But there are 7–8 issues that if left unaddressed will cause your model to fail in production or give you false confidence. I'll rank them by severity.

---

## 1. CRITICAL: The Single-Day Problem Is Bigger Than You Think

Your documents acknowledge "we only have 1 day." What you may not fully appreciate is **how fundamentally broken** this makes your current validation.

**What you're actually measuring:**
- 420 stocks, all on the same day (May 29, 2026)
- Your 5-fold "stock-level CV" separates stocks, not time
- Every bar from fold 1 is temporally adjacent to every bar from fold 4
- All stocks are subject to the same market regime that day

**What this means concretely:**
- Your RF MCC=0.31 is measuring **cross-sectional stock characteristics on a single day**, not a **temporal prediction signal**
- `spread` and `volatility_5m` being top features is a red flag — these are **static stock properties** (some stocks are always more volatile, always have wider spreads). On 1 day, these are essentially stock IDs.
- The model may have learned: "HDFC Bank has tight spreads → predict NO_TRADE; IFCI has wide spreads → predict LONG or SHORT"
- This **looks like signal but is not alpha**

**The professional test:** Run the same model but replace every feature with just the stock ticker encoded as a number. If MCC stays ≥ 0.20, you've proven the model is learning stock identity, not flow patterns.

**Bottom line:** Get more days. Until then, **don't trust MCC=0.31.**

---

## 2. CRITICAL: Label Leakage Through Overlapping Forward Returns

This is the most dangerous bug you haven't caught yet.

**The problem:**
```
Bar at 10:00 → label_60m_5pct = return from 10:00 to 11:00
Bar at 10:01 → label_60m_5pct = return from 10:01 to 11:01
```

These two labels share 59 minutes of price data. When you train on bar 10:00 and validate on bar 10:01, the labels are **98% overlapping**.

**Why your stock-level split doesn't fix this:**
Your split separates stocks (HDFC vs TATA), but within each stock, ALL bars from that stock either go to train OR to val. So no temporal overlap exists within stocks. ✅

**But there's a subtler leak you haven't checked:**
- The features at 10:01 include `volatility_5m` (std of prices from 9:57–10:01) and `delta_5m` (rolling 5min delta)
- These features look backward → they're fine
- But `vwap_distance` uses `vwap_1m` which is computed from inferred trades in that minute
- Those inferred trades come from DOM snapshots at 21µs resolution
- **Are the DOM snapshots at time T forward-contaminated by any snapshot-level computation that uses T+1 data?** Check the alignment engine carefully.

**The deeper issue — your current approach at 1 day:**
With 374 bars per stock-day and 60-min horizon, you lose 60 bars per stock (last hour). You have ~314 usable bars per stock. That's ~130k rows total after NaN drop. This is enough for a baseline but not enough to trust any per-class metric.

---

## 3. HIGH: The Backtest Script Has a Severe Lookahead Bug

Look at [`50_backtest_signals.py`](file:///c:/Users/MOHIT/.gemini/antigravity/playground/ProjectAlpha/scripts/50_backtest_signals.py), lines 19–36:

```python
q10 = df[FEATURES].quantile(0.10)   # computed on FULL dataset
q80 = df[FEATURES].quantile(0.80)   # computed on FULL dataset
q90 = df[FEATURES].quantile(0.90)   # computed on FULL dataset

rules.append((feature, 'top_decile_long', df[feature] >= q90[feature], ...))
```

**This is a textbook lookahead bias.** The quantile thresholds are computed on the ENTIRE dataset (including future bars), then used to generate signals at each bar. In live trading, at 10:00 AM you don't know what q90 will be at 3:30 PM.

**Impact:** Every signal from this backtest is invalid. The precision/recall/win-rate numbers are meaningless.

**Fix:** For each bar `t`, compute quantiles using only bars 1..t-1 (expanding window or rolling window). This is the first lesson in any serious quant firm's onboarding.

---

## 4. HIGH: Trade Inference Engine Has Multiple Silent Failures

Reading [`trade_inference.py`](file:///c:/Users/MOHIT/.gemini/antigravity/playground/ProjectAlpha/features/trade_inference.py):

**Bug 1 — Fallback quantity = 1 is wrong:**
```python
inferred_records.append((t, ltp_curr, 1, direction))  # Fallback quantity = 1
```
When LTP changes without a DOM signal, you assign `quantity=1`. This contaminates:
- `large_trade_ratio` (1 is never a large trade → always 0 for fallback bars)  
- `volume_burst` (adds 1 to volume vs actual trade which may be 10,000)
- `delta_1m` (adds ±1 vs actual ±10,000)

A better fallback: use the median trade size of the last 10 observed trades, or use `ltp_change * typical_quantity` estimation.

**Bug 2 — BUY inference from ask price up is overly aggressive:**
```python
if a1_curr > a1_prev and a1_prev > 0.0:
    for k in range(20):
        p_level = ask_prices[i-1, k]
        if p_level < a1_curr and p_level > 0.0 and q_level > 0:
            inferred_records.append(...)  # ALL levels below new ask
```
This assumes ALL quantity at every level below the new ask was consumed by trades. In reality, market makers may reprice those levels (cancel + re-post at new price). You're confusing **price improvement** with **trade execution**.

**Bug 3 — No cancellation vs trade disambiguation:**
DOM snapshot diffs can't distinguish:
- Trade: quantity at ask1 drops from 5000 → 3000 (2000 shares sold)
- Cancel: quantity at ask1 drops from 5000 → 3000 (2000 shares cancelled)
- Reprice: market maker moves ask1 from 100.50 → 100.55

All three look identical in your snapshot diff logic. This is a known limitation but it means your `order_cancel_rate` is permanently zero AND your trade quantities are systematically biased.

**Mitigation:** The fact that `large_trade_ratio` is your #1 feature despite this noise is actually encouraging — it suggests the signal is robust. But you should quantify the accuracy: for each stock-day where you have real ticks, compare total inferred volume vs total real tick volume per minute. If they're within 20%, your inference is acceptable.

---

## 5. HIGH: Feature Engineering — Serious Flaws

### 5.1 `volatility_5m` is computed wrong

```python
df['volatility_5m'] = df['vwap_1m'].rolling(window=5, min_periods=1).std()
```

**Problems:**
- Uses `vwap_1m` (trade-weighted price), not LTP. For bars with no trades, `vwap_1m=0.0`, which will massively distort std.
- `min_periods=1` means the first 4 bars compute std on 1–4 observations — statistically meaningless std.
- Use `df['ltp'].rolling(5, min_periods=5).std()` or compute realized vol properly as `sqrt(sum(r_t^2))` where `r_t = log(ltp_t / ltp_{t-1})`.

### 5.2 `imbalance` is computed on resampled (mean) DOM, not point-in-time

```python
dom_1m = dom_numeric.resample('1min', label='left', closed='left').mean()
dom_1m['imbalance_top5'] = FeatureFactory._compute_imbalance(dom_1m, levels=5)
```

You're computing bid/ask imbalance on the **mean** of 21µs snapshots over a 1-minute bar. The mean of bid quantities is not the same as the bid quantity at any point in time. For microstructure research, imbalance should be computed at the **last snapshot** of each minute (or time-weighted). Mean-resampled imbalance loses all the intra-minute dynamics.

**Fix:** Use `.last()` instead of `.mean()` for order book quantities, and use `.mean()` only for continuous price features.

### 5.3 `bid_replenishment_rate` has a logical gap

```python
return np.where(
    (df['bid1'] >= df['bid1'].shift(1)) & (prev_bid_qty > 0),
    np.minimum(1.0, np.maximum(0.0, bid_qty_change / prev_bid_qty)),
    0.0,
)
```

When `bid1` stays same and bid qty increases, this returns the fractional replenishment — fine. But it's clipped to [0,1] which means a 200% replenishment looks identical to 50%. The signal is weaker than it should be. Consider using log-scale or uncapped values.

### 5.4 `spread` has negative values — this is a data bug, not a range issue

```
spread = ask1 - bid1 shows negative for 6 stocks
```

A negative spread means ask < bid — this is physically impossible in a properly cleaned order book. This indicates the DOM cleaning step (`raw/data_cleaner.py`) is not enforcing `ask1 > bid1 > 0`. These rows should be filtered out as corrupted data, not passed downstream as features.

---

## 6. MEDIUM: The Logistic Regression Has a Parameter Bug

```python
lr = LogisticRegression(
    l1_ratio=1,       # ← This is for ElasticNet, NOT LogisticRegression
    solver="saga",
    C=1.0,
    ...
)
```

`l1_ratio` is only valid when `penalty='elasticnet'`. In `LogisticRegression` with default `penalty='l2'`, `l1_ratio` is **silently ignored**. So you're actually running L2, not L1.

**To get L1:**
```python
lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, ...)
```

This means your "LR(L1)" results are actually LR(L2). Re-run with correct parameters. L1 would give you sparsity (some features zeroed out), which is useful for feature selection analysis.

---

## 7. MEDIUM: Labeling — The 1-min LTP Used as Price Is Wrong

```python
future_price = df[price_col].shift(-horizon_min)
return (future_price - df[price_col]) / df[price_col]
```

Here `ltp` is the **last traded price at the end of that 1-minute bar** (it's a mean of ~21µs snapshots actually — see feature_factory's `.mean()` resample). 

**The correct approach for intraday prediction:**
- Entry price = `ask1` at time T (you're buying at the ask)
- Exit price = `bid1` at time T+horizon` (you're selling at the bid)
- Forward return = `(bid1[T+horizon] - ask1[T]) / ask1[T]`

Using `ltp` ignores bid-ask spread costs. For a 5% threshold on NSE stocks with 0.05–0.2% spreads this may be minor, but for 3% threshold scenarios it's material. Professional shops always compute returns with crossing the spread factored in.

---

## 8. MEDIUM: What You Should Actually Measure (Metrics)

### Currently Missing — These Are Non-Negotiable in Professional Settings:

| Metric | Why It Matters | How to Compute |
|--------|---------------|----------------|
| **Precision per class** | Precision of LONG class = how often your BUY signal is actually a up move. False positives cost money | `classification_report` (you have this but don't surface it prominently) |
| **Hit Rate** | % of LONG signals that hit the threshold. The fundamental trading metric | `(true_LONG) / (pred_LONG)` |
| **Avg return on signals** | Mean forward return conditional on predicting LONG/SHORT | `df[pred==LONG]['return_60m'].mean()` |
| **IC (Information Coefficient)** | Spearman correlation between predicted probability and actual return. The industry standard for alpha measurement | `scipy.stats.spearmanr(pred_proba[:,LONG_idx], actual_return)` |
| **ICIR (IC / std(IC))** | Consistency of IC over time. Any IC > 0.05 sustained is real alpha | Compute IC per day/week, take mean/std |
| **Decile analysis** | Sort stocks by predicted probability, compute mean return in each decile. A monotone increase = real alpha | Group by predicted prob decile, compute mean return |
| **Calibration** | Does P(LONG)=0.7 actually precede LONG 70% of the time? | `sklearn.calibration.calibration_curve` |
| **Sharpe of hypothetical strategy** | If you traded every LONG signal with 60m hold: Sharpe ratio | Simple backtest with realistic costs |

### Metrics You Have That Are Actually Good:
- **MCC** — correct for multi-class imbalanced (keep it as primary)
- **Shuffle test** — keeps you honest on leakage (keep it)
- **Cross-fold std** — measures stability (keep it)
- **Feature importance** — good signal about what's working (keep it)

---

## 9. What HFT / Hedge Funds Actually Do Differently

This is where I'll be most direct:

### 9.1 Data
- Real firms use **exchange-certified ITCH/PITCH feeds** with nanosecond timestamps, not snapshot data at 21µs intervals. Your DOM is already 100x slower than what HFT uses.
- For a quant hedge fund at 60-min horizon (not HFT), NSE data from vendors like **True Data, Global Data Feed, Tick Data LLC** is standard. They provide L2 data.
- **You should buy at minimum 1 full year of NSE data** (NSE itself sells historical data). 252 trading days × 438 stocks = 110,000+ stock-days. That's what you need.

### 9.2 Labels
- Professional shops **never use fixed-horizon discrete labels** for this type of signal. They use:
  - **Continuous regression targets** (predict the return, not the class)
  - **Optimally-stopped returns** (label = max favorable excursion before 5% adverse excursion, like your EOD plan)
  - **Risk-adjusted targets** (return / realized vol over the horizon)
  - **Event-based labels** (did price cross X level within T minutes, yes/no)
- The LONG/SHORT/NO_TRADE structure throws away information. A 5.1% return is the same label as a 50% return. A -4.9% return gets NO_TRADE while -5.1% gets SHORT.

### 9.3 Validation
- **Purged k-fold** (López de Prado's approach): when splitting train/val, remove bars within H minutes of the boundary (where H = label horizon), to prevent label overlap between train and val. Your stock-level split avoids this, but multi-day data requires purging.
- **Walk-forward out-of-sample**: train on months 1-6, test on month 7. Expand window. Never look back.
- **Combinatorial Purged CV** (CPCV): even more conservative, used by top quant funds.

### 9.4 Features That Actually Work (Published Research)
These are academically validated, not just "might work":

1. **Order Flow Imbalance (OFI)** — Cont, Kukanov, Stoikov (2014). Exactly your `delta_1m`. ✅ You have this.
2. **VPIN** — Easley, de Prado, O'Hara (2012). Volume-synchronized prob of informed trading. Your `aggressor_ratio` is a crude proxy. ❌ Implement properly.
3. **Queue imbalance** — ratio of queue depth on each side at best bid/ask. ✅ Your `imbalance_top5` is close.
4. **Trade sign autocorrelation** — are buyers clustering? Computed as autocorrelation of sign(trade) series.  ❌ Not in your pipeline.
5. **Amihud illiquidity** — |return| / volume. Proxy for price impact. ❌ Not in your pipeline.
6. **Kyle's lambda** — slope of price impact vs signed order flow via OLS on 5-min windows. ❌ Not in your pipeline.
7. **Micro-price** — `bid1 * aqty1/(bqty1+aqty1) + ask1 * bqty1/(bqty1+aqty1)`. Better mid-price estimate. ❌ Not in your pipeline.

### 9.5 Model Selection Reality

For 60-minute predictive signal on NSE microstructure data, the honest ranking by expected performance with sufficient data:

| Model | Expected Rank | Reason |
|-------|:---:|-------|
| **LightGBM/XGBoost** | 1 | Handles non-linear feature interactions, native L1 regularization, faster than RF |
| **Random Forest** | 2 | Good baseline, robust to outliers, but no gradient boosting |
| **Linear Regression (on returns)** | 3 | Often underrated; for IC measurement, linear is the honest floor |
| **Neural Net (LSTM on raw features)** | 4 | Needs 10x more data, overfit risk is high |
| **Logistic Regression (L1)** | 5 | Good for interpretability and checking sparsity |

**BUT: With only 1 day of data, model choice is irrelevant. Get more data first.**

### 9.6 Observability — What You're Missing

Professional-grade systems track:
```
Per-day:
- Feature drift (KL-divergence from training distribution)
- Label distribution shift
- Model prediction distribution shift
- IC decay (how fast does signal decay?)

Per-model:
- Calibration curve over time
- Feature importance stability
- Confusion matrix trend
- SHAP value distribution per feature

Infrastructure:
- Data pipeline SLA (latency, completeness)
- Feature computation latency
- Model inference latency
- Alert on NaN spikes in any feature
```

You have none of this. For research it's acceptable. For production it's mandatory.

---

## 10. Prioritized Action List

### Tier 1 — Do These Before Any More Modeling (Critical)

1. **Fix the LogisticRegression L1 bug** (30 min): Add `penalty='l1'` explicitly
2. **Fix the backtest lookahead bias** in `50_backtest_signals.py` (2 hours): Use expanding-window quantiles
3. **Fix `volatility_5m`** computation (1 hour): Use LTP returns std, not VWAP std
4. **Fix negative spread** handling in data cleaner (1 hour): Filter rows where `ask1 <= bid1`
5. **Run the stock-identity test** (1 hour): Replace features with stock one-hot encoding, see if MCC stays high

### Tier 2 — High Value, Do Before Production

6. **Validate trade inference accuracy** (1 day): Compare inferred volume vs real tick volume per minute for all 438 BOTH stocks
7. **Switch DOM imbalance to `.last()` snapshot** instead of `.mean()` (2 hours)
8. **Add IC/ICIR computation** to baseline model output (1 day)
9. **Add precision/recall per class** as primary metrics (2 hours)
10. **Implement decile analysis** on predicted probabilities (4 hours)

### Tier 3 — Data First, Then Come Back

11. **Acquire 30+ days of NSE data** before doing anything else in this list
12. **Implement purged walk-forward validation** once multi-day data exists
13. **Implement VPIN** as an additional feature
14. **Implement micro-price** as a feature
15. **Switch labels to continuous regression target** (predict return, not class)

---

## 11. Specific Bug Summary Table

| # | File | Line(s) | Bug | Severity |
|---|------|---------|-----|----------|
| 1 | `50_baseline_model.py` | 225–232 | `l1_ratio=1` doesn't enable L1 without `penalty='l1'` | HIGH |
| 2 | `50_backtest_signals.py` | 19–22 | Quantiles computed on full data = lookahead bias | CRITICAL |
| 3 | `features/tick_features.py` | 176 | `volatility_5m` uses vwap_1m (wrong) and min_periods=1 (noisy) | HIGH |
| 4 | `features/feature_factory.py` | 134 | DOM resampled to mean before imbalance — should use `.last()` | MEDIUM |
| 5 | `features/trade_inference.py` | 145 | Fallback qty=1 massively underestimates trade size | MEDIUM |
| 6 | `features/trade_inference.py` | 91–106 | Ask level clearing assumes all qty is traded (vs repriced) | MEDIUM |
| 7 | `raw/data_cleaner.py` | — | No enforcement of `ask1 > bid1 > 0` | MEDIUM |
| 8 | `labels/label_generator.py` | 19–20 | Uses LTP (mean), not ask/bid for true cost-aware return | LOW |

---

## 12. Honest Summary

**What you've built:**
A well-engineered, well-documented research pipeline that correctly handles edge cases, has good test coverage, survived a critical data corruption incident, and produced genuine signal (MCC > 0.3 is real, not noise).

**What you haven't built:**
A trading system. You have a research artifact that works on 1 day of data for 420 stocks cross-sectionally. The features may be measuring stock characteristics (spread size, volatility regime) rather than predictive flow signals.

**The path forward:**
1. Fix the 5 critical bugs listed above (1–2 days of work)
2. Acquire 30+ days of data (the only thing that matters)
3. Re-run the pipeline on multi-day data
4. Implement walk-forward validation
5. Add IC/ICIR as primary metrics
6. Then, and only then, start comparing LightGBM vs RF vs Neural Nets

**Is MCC=0.31 "too good to be true"?**
Yes and no. It's real signal, but it's likely partially driven by static stock properties rather than temporal prediction. On multi-day data with proper purged walk-forward validation, expect MCC to drop to 0.05–0.15 for genuine out-of-time prediction. If it stays at 0.31, you've found something extraordinary. If it drops to 0.03, that's still tradeable with enough volume.

**You are on the right track. The discipline and rigor are professional-grade. Now go get more data.**
