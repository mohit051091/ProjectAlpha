# MASTER REVIEW & EXECUTION PLAN
## Order Flow & Pre-Move Detection System — NSE Equities

**Compiled:** 2026-06-11
**Sources:** 4 independent AI consultant reviews (Claude Sonnet 4.6 Agentic, Claude Sonnet Chat 4.6, GPT-4o, Gemini 2.5 Pro) + deep code/graph analysis
**Purpose:** Single source of truth for what's wrong, what to fix, what to build, and in what order. Designed to be handed to an execution agent.

---

## PART 1: CONSOLIDATED VERDICT

> [!IMPORTANT]
> **All 4 consultants agree on 3 things:**
> 1. The pipeline engineering and documentation discipline are professional-grade
> 2. MCC=0.31 is almost certainly inflated by stock-level characteristics, not temporal prediction signal
> 3. More data is the #1 priority — everything else is secondary

### Scorecard (Unanimous Across All 4 Reviews)

| Dimension | Status | Consensus |
|-----------|:------:|-----------|
| Pipeline integrity | ✅ PASS | Contamination fix, rebuild, validation — all done correctly |
| Documentation | ✅ PASS | Review/rebuttal cycle, RCA, decision logs — professional |
| Data depth | 🔴 CRITICAL | 1 day = stock characterization, not alpha. Need 60+ days minimum |
| Label design | 🟡 REVISE | Fixed-horizon discrete labels lose information. Switch to MFE/Triple-Barrier |
| Validation strategy | 🟡 REVISE | Stock-level CV correct for 1 day, but walk-forward required for multi-day |
| Trade inference | 🟠 UNVALIDATED | 2 strongest features built on unvalidated inferred trades |
| Feature engineering | 🟡 PARTIAL | Good foundation but 4 bugs + missing institutional-grade features |
| Model selection | ✅ CORRECT | RF baseline appropriate. LightGBM next. No neural nets yet |
| Metrics | 🟡 INCOMPLETE | MCC + shuffle test are good. Missing IC, calibration, P&L simulation |
| Observability | 🔴 MISSING | Zero production monitoring. Acceptable for research, not for production |

---

## PART 2: ALL BUGS FOUND (Cross-Referenced Across Reviews)

Every bug below was identified by at least 2 of 4 reviewers. Severity is the consensus maximum.

### BUG TABLE — Ordered by Severity

| # | Severity | File | Line(s) | Bug | Found By | Fix |
|---|:--------:|------|---------|-----|----------|-----|
| B1 | 🔴 CRITICAL | `scripts/50_backtest_signals.py` | 19–36 | **Lookahead bias**: quantiles computed on full dataset including future bars | Sonnet Agentic, Sonnet Chat | Use expanding-window quantiles: at bar `t`, compute q90 using bars `1..t-1` only |
| B2 | 🔴 HIGH | `scripts/50_baseline_model.py` | 225–232 | **LR penalty bug**: `l1_ratio=1` is silently ignored without `penalty='l1'`. Running L2, not L1 | Sonnet Agentic | Add `penalty='l1'` explicitly |
| B3 | 🔴 HIGH | `features/tick_features.py` | 176 | **`volatility_5m` computed on VWAP instead of LTP**: when `volume_1m=0`, `vwap_1m=0.0` → massive distortion. Also `min_periods=1` → noisy | Sonnet Agentic, Sonnet Chat | Use `df['ltp'].rolling(5, min_periods=5).std()` |
| B4 | 🔴 HIGH | `features/tick_features.py` | 171–174 | **`vwap_distance` = 296.9**: division by near-zero VWAP in opening minutes before trades execute | Sonnet Chat | Gate: `vwap_distance = NaN` until cumulative volume exceeds minimum threshold |
| B5 | 🟠 MEDIUM | `features/feature_factory.py` | 134 | **DOM resampled to `.mean()` before imbalance**: mean of 21µs snapshots is meaningless for bid/ask depth | Sonnet Agentic, Gemini | Use `.last()` for order book quantities, `.mean()` only for price |
| B6 | 🟠 MEDIUM | `features/trade_inference.py` | 145 | **Fallback qty=1**: when LTP changes without DOM signal, assigns qty=1 vs actual trade which may be 10,000 | All 4 | Use median of last 10 observed trade sizes |
| B7 | 🟠 MEDIUM | `features/trade_inference.py` | 91–106 | **Level clearing = trade**: assumes all qty below new ask was consumed by trades (vs repriced by market makers) | Sonnet Agentic, Gemini | Known limitation — document and quantify error rate |
| B8 | 🟠 MEDIUM | `raw/data_cleaner.py` | — | **No `ask1 > bid1 > 0` enforcement**: 6 stocks have negative spread (crossed book). Physically impossible. Corrupts imbalance, iceberg, all depth features | All 4 | Add filter: drop rows where `ask1 <= bid1` or `bid1 <= 0`. Quarantine the 6 affected stocks until investigated |
| B9 | 🟡 LOW | `labels/label_generator.py` | 19–20 | **Labels use LTP (which is mean of snapshots), not ask/bid for cost-aware return** | Sonnet Agentic, Sonnet Chat | Entry = `ask1[T]`, exit = `bid1[T+H]` for realistic cost accounting. Minor impact at 5% threshold |

---

## PART 3: THE SINGLE-DAY PROBLEM — CONSENSUS ANALYSIS

### 3.1 What MCC=0.31 Actually Measures

All 4 consultants flagged this as the #1 risk. The consensus:

- **420 stocks, 1 day (May 29, 2026)**
- `spread` (importance 0.196) and `volatility_5m` (0.165) are the top RF features
- These are **static stock properties**: HDFC Bank always has tight spreads; IFCI always has wide spreads
- The model likely learned: "volatile small-cap stocks with wide spreads move more" — which is trivially true and completely unexploitable

**Realistic MCC bounds** (from Sonnet Chat, most specific):
| Model | Expected MCC on Generalized Signal |
|-------|:---:|
| LR baseline | 0.04–0.10 |
| Tree ensemble (RF) | 0.08–0.18 |
| Tuned LightGBM | 0.12–0.22 |
| Regime-aware ensemble | 0.15–0.28 |

MCC=0.31 sits **above** the realistic upper bound → almost certainly inflated by stock characterization.

### 3.2 Three Diagnostic Tests to Run Immediately

All 4 consultants converge on these (listed by cheapest → most expensive):

| Test | Time | Pass Criterion | What It Proves |
|------|:----:|-----------------|----------------|
| **Stock-identity test**: Use only per-stock `mean(spread)` + `mean(volatility_5m)` as features in fresh LR | 20 min | MCC < 0.15 | Model is NOT just a stock classifier |
| **Z-score neutralization** (GPT): Z-score all features within each stock, retrain | 30 min | MCC stays > 0.15 | Signal is temporal, not stock-level |
| **Within-stock temporal CV**: For each stock, train on bars 1–250, test on bars 251–374 | 30 min | MCC > 0.05 | Signal is temporal within each stock |

> [!CAUTION]
> **If all 3 tests fail** (MCC collapses), the current model has zero temporal prediction signal and everything beyond bug fixes + data acquisition is premature.

### 3.3 Is 10 Days of Data Enough?

**Short answer: It's a minimum starting point, not a destination.**

| Days | Stock-Days | What You Can Do | What You Cannot Do |
|:----:|:----------:|-----------------|-------------------|
| 1 (current) | 438 | Cross-sectional baseline | Any temporal validation |
| **10** | **4,380** | Walk-forward on 2–3 test days. Feature stability check. Regime diversity: unlikely (10 consecutive days ≈ same market regime) | Regime generalization. Production-ready model. Meaningful ICIR |
| 20–30 | 8,760–13,140 | Walk-forward with purging. Feature importance stability. Basic regime testing if the period spans volatility shifts | Full regime coverage (bull/bear/sideways) |
| **60** (recommended minimum) | **26,280** | LightGBM vs RF comparison. CPCV. Meaningful walk-forward across regime changes | Out-of-time validation across seasons |
| 252 (1 year) | 110,000+ | Full regime-aware ensemble. Production deployment ready | Nothing — this is the target |

> [!IMPORTANT]
> **10 days is enough to validate whether temporal signal exists at all.** It is NOT enough for production modeling.
> 
> **Priority with 10 days:** Run walk-forward (train on days 1–7, test on days 8–10 with 60-bar purge gap). If MCC > 0.05 on the held-out days, signal is real. If MCC ≈ 0, acquire 60+ days before continuing.
>
> **Data regime coverage matters more than raw volume.** 10 days from the same calm market is worse than 5 days spanning a volatile event + 5 calm days. Ask vendor for days from different VIX/INDIA VIX regimes.

---

## PART 4: FEATURE ENGINEERING — CONSOLIDATED RECOMMENDATIONS

### 4.1 Current Features — Status

| Feature | Status | Consensus Action |
|---------|:------:|-----------------|
| `large_trade_ratio` | ✅ Keep | ES=0.571. Strongest signal. **But**: validate against real ticks first |
| `iceberg_score` | ✅ Keep | ES=0.266. Unique microstructure signal. Same validation requirement |
| `delta_1m` | ✅ Keep | ES=0.488. Core OFI signal. Correct as computed |
| `imbalance_top5` | ✅ Keep | ES=0.121. Best of three imbalance features |
| `aggressor_ratio` | ✅ Keep | ES=0.111. Crude VPIN proxy — keep until proper VPIN added |
| `spread` | ⚠️ Investigate | RF importance=0.196 but likely stock-type feature. Check temporal signal after per-stock mean subtraction |
| `volatility_5m` | 🔴 Fix | Wrong computation (uses VWAP). After fix, keep as regime context only |
| `vwap_distance` | 🔴 Fix | Gate on minimum cumulative volume. Current 296.9 values physically impossible |
| `bid_replenishment_rate` | ✅ Keep | Clip to [0,1] loses signal. Consider log-scale or uncapped |
| `imbalance_top10/20` | ✅ Keep | Lower ES but add book depth perspective |
| `depth_drop_bid/ask` | ⚠️ Low priority | ES=0.009/0.006. May improve as rate-of-change instead of level |
| `price_acceleration` | ✅ Keep | ES=0.011. Weak but computationally free |
| `delta_5m` | ❌ Excluded | Correct — r=0.944 with delta_1m |
| `order_cancel_rate` | 🔴 Rebuild | All zeros. Implement DOM-diff proxy (see 4.3 below) |

### 4.2 New Features to Add — Priority Order

All 4 consultants agree on the top 5. Ordered by consensus priority:

| Priority | Feature | Formula / Description | Source |
|:--------:|---------|----------------------|--------|
| 🔴 **P1** | **Micro-price** | `(ask_qty × bid1 + bid_qty × ask1) / (bid_qty + ask_qty)`. Better mid-price than LTP. Standard institutional reference price | All 4 |
| 🔴 **P1** | **Cumulative delta since open** | `cumsum(delta_1m)` from 9:15 AM. Persistent directional pressure. Single most important addition for ≥5% detection | Sonnet Chat, GPT |
| 🔴 **P1** | **Book pressure** | `imbalance_top5 × total_depth`. 60% imbalance with 1000 lots ≠ 60% with 10 lots | Sonnet Chat, Gemini |
| 🟠 **P2** | **Time since open** (minutes) | Linear: `(ts - 9:15 AM) / total_session_minutes`. Captures intraday seasonality. Opening auction vs lunch lull vs close-of-day effects | Sonnet Chat |
| 🟠 **P2** | **Distance from day high/low** | `(ltp - running_high) / running_high` and `(ltp - running_low) / running_low`. Measures retracement depth | Sonnet Chat |
| 🟠 **P2** | **Order arrival rate** | DOM updates per second within 1-min bar. Acceleration signals institutional order placement | Sonnet Chat, Gemini |
| 🟠 **P2** | **Depth slope** | Gradient of quantity across bid levels L1→L10. Steep drop = thin book = breakout setup | Sonnet Chat, Gemini |
| 🟡 **P3** | **VPIN proxy** | Volume-sync buckets, measure buy/sell imbalance per bucket. More robust than aggressor_ratio | All 4 |
| 🟡 **P3** | **Ask replenishment absence** | After ask consumed, refill speed. No refill = weak supply = bullish signal | Sonnet Chat |
| 🟡 **P3** | **Trade sign autocorrelation** | Are buyers clustering? ACF of sign(trade) series | Sonnet Agentic |
| 🟡 **P3** | **Order book slope/curvature** | Δ(Quantity) / Δ(Price level) across 20 levels. Convex = robust support, concave = liquidity trap | Gemini |
| ⚪ **P4** | **Amihud illiquidity** | |return| / volume. Price impact proxy | Sonnet Agentic |
| ⚪ **P4** | **Kyle's lambda** | Slope of price impact vs signed order flow (OLS on 5-min windows) | Sonnet Agentic |
| ⚪ **P4** | **Relative strength vs Nifty** | `stock_return_since_open - nifty_return_since_open`. Requires Nifty index feed | Sonnet Chat |

### 4.3 Rebuilding `order_cancel_rate` (DOM-diff approach)

All reviewers agree the current implementation is broken (all zeros). The Sonnet Chat review provides the best implementation spec:

```
For each DOM snapshot pair (t-1, t):
  For each price level L:
    qty_change = qty[t, L] - qty[t-1, L]
    if qty_change < 0:   # quantity decreased
      if LTP did NOT change within ~1 tick window:
        → classify as CANCELLATION
      else:
        → classify as EXECUTION (trade)

cancel_rate_proxy = sum(qty_drops_without_trade) / sum(total_qty_changes)
```

**Expected accuracy:** ~60–70% on NSE 20-level DOM. Confounds partial fills with cancellations at the same level. Still significantly better than all-zeros.

### 4.4 Feature Normalization (Gemini's Critical Point)

> [!WARNING]
> **All price-derived features must be normalized to log-return space** or divided by the asset's running VWAP. A ₹10 move on Maruti (₹5,000) is noise; the same ₹10 on IFCI (₹50) is a regime shift. Raw unscaled features compromise all tree-split decisions.

```python
# Instead of: delta_1m = buy_vol - sell_vol
# Use:        delta_1m_normalized = (buy_vol - sell_vol) / rolling_avg_volume
# Instead of: spread = ask1 - bid1  
# Use:        spread_bps = (ask1 - bid1) / mid * 10000  # in basis points
```

---

## PART 5: LABELING STRATEGY — CONSENSUS

### 5.1 Current Labels: Wrong for the Stated Goal

All 4 reviewers agree the current fixed-horizon labels are a mismatch:

**Current question:** "Did price exceed ±5% at exactly t+60 minutes?"
**Actual question:** "Will price reach +5% at any point between now and market close?"

These are fundamentally different. The current label:
- Misses moves that hit 5% at 35 minutes then reversed
- Misses moves that hit 5% at 90 minutes
- Treats 5.1% the same as 50%
- Treats -4.9% as NO_TRADE while -5.1% is SHORT

### 5.2 Switch to MFE Triple-Barrier Labels

**Unanimous recommendation across all 4 reviews:**

```
Upper barrier: MFE_to_close = max(high[t : close] - entry[t]) / entry[t]
  → LONG if MFE_to_close >= 0.05

Lower barrier: min(low[t : close] - entry[t]) / entry[t]
  → SHORT if <= -0.05

Time barrier: market close
  → NO_TRADE if price stays within [-5%, +5%] until close

Label = whichever barrier is hit first
```

**Why this is better:**
- Maps directly to your trade parameters (entry, target, stop-loss, expiry)
- No information loss from fixed-horizon sampling
- Economically meaningful — models the actual decision you'd make as a trader
- Standard institutional practice (López de Prado, Chapter 3)

### 5.3 Also Keep Continuous Return Targets (GPT)

In parallel with classification labels, add continuous return targets:
```python
return_5m  = forward_return(5)
return_15m = forward_return(15)
return_30m = forward_return(30)
return_60m = forward_return(60)
```
These allow regression-based IC measurement — the industry standard for alpha evaluation.

### 5.4 Expected Class Imbalance with MFE Labels

On any given day, ~15–40 of 438 NSE stocks move ≥5% intraday. Within those stocks, only bars **before** the move are meaningful positive examples. **Expect 2–5% positive class rate.**

**How to handle (consensus order of preference):**
1. **Precision-focused threshold**: optimize for precision ≥ 60%. Do NOT use SMOTE.
2. **Class weights**: `class_weight='balanced'` in sklearn, `scale_pos_weight` in LightGBM
3. **Calibration**: Platt scaling or isotonic regression after training
4. **Asymmetric cost matrix**: false positive (bad trade) vs false negative (missed opportunity)

---

## PART 6: VALIDATION STRATEGY

### 6.1 Current Approach (Single Day)

Stock-level 5-fold CV is **correct for single-day data** (all 4 agree). No changes needed here.

### 6.2 Single-Day Honest Test (Do Now)

Within-stock temporal split (from Sonnet Chat):
- Train on bars 1–250 of each stock
- Test on bars 251–374 of each stock
- This tests if the model predicts the afternoon from the morning — actual temporal generalization

### 6.3 Multi-Day Walk-Forward (Required for 10+ Days)

```
Train on days 1–7, purge last 60 bars, test on day 8
Train on days 1–8, purge last 60 bars, test on day 9
Train on days 1–9, purge last 60 bars, test on day 10
...
```

**Purge gap:** exclude last 60 bars of training set (= label horizon) to prevent forward-return leakage.
**Embargo gap:** exclude first 30 bars of each test day to prevent serial correlation at boundary.

### 6.4 Combinatorial Purged CV (CPCV) — For 60+ Days

López de Prado Chapter 12. For each fold boundary: purge N bars where N = label horizon. Embargo a gap of 30 bars after each training set. This is the gold standard for financial ML validation.

### 6.5 Out-of-Time Validation — Required Before Production

Train on first 80% of days, test on last 20%. Target: MCC in OOT should be within 30% of in-sample MCC.

---

## PART 7: METRICS — COMPLETE STACK

### 7.1 What You Have (Keep These)

| Metric | Status | Notes |
|--------|:------:|-------|
| MCC | ✅ Keep | Correct for imbalanced multi-class. Primary metric |
| Shuffle test | ✅ Keep | Leakage detection. Keep forever |
| Cross-fold std | ✅ Keep | Stability measurement |
| Feature importance (RF) | ✅ Keep | Directional, but switch to SHAP with LightGBM later |
| Classification report | ✅ Keep | Already computed but not prominently surfaced |

### 7.2 What You Must Add (All 4 Agree)

| Metric | Priority | What It Tells You | How to Compute |
|--------|:--------:|-------------------|----------------|
| **Precision per class** (LONG/SHORT separately) | 🔴 P1 | How often your BUY signal actually goes up. False positives = lost money | `classification_report` → surface LONG precision prominently |
| **IC (Information Coefficient)** | 🔴 P1 | Spearman correlation between predicted probability and actual forward return. **Industry standard** for alpha | `scipy.stats.spearmanr(pred_proba[:, LONG_idx], actual_return)` |
| **Hit rate** | 🔴 P1 | % of LONG signals that hit the threshold | `true_LONG / pred_LONG` |
| **Avg return on signals** | 🔴 P1 | Mean forward return conditional on predicting LONG/SHORT. The actual P&L proxy | `df[pred==LONG]['return_60m'].mean()` |
| **Calibration curve** | 🟠 P2 | Does P(LONG)=0.70 → 70% LONG frequency? Required for position sizing | `sklearn.calibration.calibration_curve` |
| **Simulated P&L** | 🟠 P2 | Enter at LONG signal, hold until target hit or close, apply 0.05% slippage each way. Sharpe, max drawdown, win rate | Simple backtest loop |
| **ICIR** (IC / std(IC)) | 🟠 P2 | Consistency of IC over time. IC > 0.05 sustained = real alpha | Compute IC per day, take mean/std |
| **Decile analysis** | 🟠 P2 | Sort by predicted probability, compute mean return per decile. Monotone increase = real alpha | Group by pred prob decile |
| **Sharpe ratio** | 🟠 P2 | Risk-adjusted return of the strategy. The universal fund metric | `mean(daily_returns) / std(daily_returns) * sqrt(252)` |
| **Sortino ratio** | 🟠 P2 | Like Sharpe but penalizes only downside volatility. Better for asymmetric strategies | `mean(excess_return) / std(negative_returns) * sqrt(252)` |
| **Max drawdown** | 🟠 P2 | Largest peak-to-trough decline. Determines survival risk | `max(running_max - equity) / running_max` |
| **Turnover** | 🟡 P3 | How often positions change. High turnover = high cost drag | `sum(abs(position_change)) / sum(abs(position))` per period |
| **Feature importance stability** | 🟡 P3 | Do top features change across folds? Instability → production degradation | Compare importance rankings across CV folds |
| **Deflated Sharpe Ratio** | 🟡 P3 | Accounts for multiple testing. True significance of best result | Track all experiments, compute DSR per López de Prado |

---

## PART 8: MODEL SELECTION — CONSENSUS PROGRESSION

| Phase | Data Volume | Model | Rationale |
|-------|:----------:|-------|-----------|
| Current (1 day) | 438 stock-days | **Logistic Regression (L1)** | Linear floor. Interpretable. Fix the penalty bug first |
| Phase 2 (10–30 days) | 4,000–13,000 | **Random Forest** | Current approach. Controls overfit with depth. Feature importance |
| Phase 3 (60+ days) | 26,000+ | **LightGBM** | Gradient boosting > bagging. Native NaN handling. SHAP importance. Better calibration | 
| Phase 4 (6+ months) | 100,000+ | **Regime-aware ensemble** | Separate models for opening (9:15–10:00), midday (10:00–13:00), afternoon (13:00–15:30). Meta-classifier |

> [!CAUTION]
> **Do NOT use:** LSTM, Transformers, or neural networks until 2+ years of data. Do NOT use SMOTE. Do NOT build live infrastructure before offline signal validated on 6+ months walk-forward.

---

## PART 9: OBSERVABILITY (For Production)

### 9.1 Research Phase (Now)

| Monitor | Implementation |
|---------|---------------|
| Feature drift per fold | Compare feature distributions (mean, std, min, max) across CV folds |
| Label distribution check | Log class balance per fold |
| NaN audit | Count NaN per feature per stock. Alert if > 5% |
| Feature range violations | Already implemented. Fix the tight thresholds for Indian markets |

### 9.2 Production Phase (After Multi-Day Validation)

| Domain | Metric | Alert Trigger |
|--------|--------|---------------|
| Data ingestion | Feed latency (exchange ts vs local ts) | Drop feeds with > 100ms delta |
| Feature stability | Population Stability Index (PSI) daily | PSI > 0.25 → pause model |
| Prediction health | Calibration vs empirical frequency | Platt scaling if warp detected |
| Prediction distribution | Daily mean(P(LONG)) | Shift from 0.35 → 0.55 → regime change alert |
| Real-time IC | Rolling 10-day IC on LONG signals | IC < 0.03 → investigate. IC < 0.01 for 3+ days → suspend |
| Slippage | Actual entry slippage vs assumed | Actual > 2× assumed → re-estimate P&L |
| Infrastructure | Peak memory during transforms | Migrate to PyArrow streaming if > 32GB |

---

## PART 10: EXECUTION PLAN — PHASED TASKS

> [!NOTE]
> Each task has acceptance criteria. Agent should execute in order. Tasks marked 🧑 require human decision/input before proceeding.

---

### PHASE 0: CRITICAL BUG FIXES (Estimated: 1–2 days)
*No new data needed. Fix what's broken before anything else.*

#### Task 0.1: Fix Logistic Regression penalty bug
- **File:** `scripts/50_baseline_model.py` line 225
- **Change:** Add `penalty='l1'` to LogisticRegression constructor
- **Acceptance:** `lr = LogisticRegression(penalty='l1', solver='saga', C=1.0, class_weight='balanced', max_iter=2000, random_state=42)`
- **Status:** `[ ]`

#### Task 0.2: Fix backtest lookahead bias
- **File:** `scripts/50_backtest_signals.py` lines 19–36
- **Change:** Replace `df[FEATURES].quantile(0.90)` with expanding-window quantile
- **Acceptance:** For each bar `t`, quantiles computed on bars `0..t-1` only. No future data used.
- **Status:** `[ ]`

#### Task 0.3: Fix `volatility_5m` computation
- **File:** `features/tick_features.py` line 176
- **Change:** Replace `df['vwap_1m'].rolling(5, min_periods=1).std()` with `df['ltp'].rolling(5, min_periods=5).std()`
- **Acceptance:** No bars with `vwap_1m=0` contributing to volatility. First 4 bars are NaN (not fake values).
- **Status:** `[ ]`

#### Task 0.4: Fix `vwap_distance` near-zero division
- **File:** `features/tick_features.py` lines 171–174
- **Change:** Gate: set `vwap_distance = NaN` until cumulative volume exceeds a minimum threshold
- **Explicit gate condition (from Gemini feedback):**
  ```python
  cumulative_vol = df['volume_1m'].cumsum()
  median_trade = df['volume_1m'][df['volume_1m'] > 0].median()
  gate = cumulative_vol >= 100 * median_trade
  df['vwap_distance'] = np.where(gate, (df['ltp'] - df['vwap_1m']) / df['vwap_1m'], np.nan)
  ```
- **Acceptance:** No `vwap_distance` values > 1.0 or < -1.0 (physically impossible intraday). Opening-minute bars produce NaN, not extreme values.
- **Status:** `[ ]`

#### Task 0.5: Fix negative spread / crossed book
- **File:** `raw/data_cleaner.py`
- **Change:** Add filter: drop rows where `ask1 <= bid1` or `bid1 <= 0` or `ask1 <= 0`
- **Acceptance:** Zero rows in cleaned output where `spread < 0`. Log how many rows dropped per stock.
- **Status:** `[ ]`

#### Task 0.6: Fix DOM resampling for imbalance features
- **File:** `features/feature_factory.py` line 134
- **Change:** Use `.last()` for order book quantity columns (`bqty*`, `aqty*`, `bid*`, `ask*`, `total_bid_qty`, `total_ask_qty`). Keep `.mean()` for `ltp` only.
- **Acceptance:** Imbalance computed on end-of-minute snapshot, not averaged snapshots.
- **Status:** `[ ]`

#### Task 0.7: Fix trade inference fallback quantity
- **File:** `features/trade_inference.py` line 145
- **Change:** Replace `1` with median of last 10 observed trade sizes (rolling median)
- **Acceptance:** Fallback trades have quantity > 1 and approximate recent trade sizes.
- **Status:** `[ ]`

#### 🧑 CHECKPOINT 0: Re-run full pipeline (Stages 0–5) on corrected code. Compare MCC before/after. Report changes to human.

---

### PHASE 1: DIAGNOSTIC TESTS (Estimated: 0.5 days)
*Determine if current MCC=0.31 is real temporal signal or stock characterization.*

#### Task 1.1: Stock-identity test
- **What:** Use only per-stock `mean(spread)` and `mean(volatility_5m)` as features in fresh LR
- **Acceptance:** Report MCC. If MCC > 0.15 → model is stock classifier.
- **Status:** `[ ]`

#### Task 1.2: Z-score neutralization (GPT recommendation)
- **What:** Z-score all features within each stock (subtract per-stock mean, divide by per-stock std), then retrain RF
- **Acceptance:** Report MCC. If MCC collapses → signal was stock-level only.
- **Status:** `[ ]`

#### Task 1.3: Within-stock temporal CV
- **What:** For each stock, train on bars 1–250 (morning), test on bars 251–374 (afternoon). Aggregate MCC.
- **Acceptance:** Report MCC. If MCC > 0.05 → temporal signal exists. If ≈ 0 → no temporal signal.
- **Status:** `[ ]`

> [!CAUTION]
> **HALT_AND_REPORT GATE (from GPT + Gemini feedback):**
> 
> | Test | PASS | FAIL | AMBIGUOUS |
> |------|:----:|:----:|:---------:|
> | Task 1.1 (Stock-identity) | MCC < 0.15 | MCC ≥ 0.15 | — |
> | Task 1.2 (Z-score) | MCC stays > 0.15 | MCC collapses to < 0.05 | MCC 0.05–0.15 |
> | Task 1.3 (Temporal CV) | MCC > 0.05 | MCC ≈ 0 | MCC 0.01–0.05 |
>
> **If ANY test FAILs:** Agent must **HALT immediately** and report to human. Do NOT auto-pivot to feature redesign. Do NOT continue to Phase 2. Present the exact numbers and let the human decide.
>
> **If all 3 PASS:** Continue to Phase 2.
>
> **If AMBIGUOUS (borderline values):** Report all values with a recommendation, but let human decide.

#### 🧑 CHECKPOINT 1: Present all 3 test results to human with the HALT gate table filled in. Human decides: if no temporal signal, pause modeling and focus exclusively on data acquisition. If temporal signal exists, continue to Phase 2.

---

### PHASE 2: TRADE INFERENCE VALIDATION (Estimated: 0.5 days)
*Your 2 strongest features (large_trade_ratio, iceberg_score) are built on unvalidated inferred trades.*

#### Task 2.1: Bar-level trade inference accuracy
- **What:** For each of the 438 BOTH stocks, compare per-minute:
  - Inferred trade count vs real tick count
  - Inferred buy volume vs real buy volume
  - Inferred sell volume vs real sell volume
- **Metrics:** Pearson r and MAPE for each
- **Acceptance:** r > 0.85 on trade counts, r > 0.70 on directional split. If below these thresholds → features need rebuild before acquiring more data.
- **Status:** `[ ]`

#### 🧑 CHECKPOINT 2: Report inference accuracy. Human decides if accuracy is sufficient or if trade inference engine needs rework.

---

### PHASE 3: NEW FEATURES + LABEL REDESIGN (Estimated: 2–3 days)
*Add institutional-grade features. Switch to MFE labels.*

#### Task 3.1: Implement P1 features
- **Micro-price:** `(ask_qty × bid1 + bid_qty × ask1) / (bid_qty + ask_qty)`
- **Cumulative delta:** `cumsum(delta_1m)` from 9:15 AM
- **Book pressure:** `imbalance_top5 × (total_bid_qty + total_ask_qty)`
- **File:** `features/feature_factory.py` or new module
- **Acceptance:** All 3 features computed for all 438 stocks. No NaN in steady-state bars.
- **Status:** `[ ]`

#### Task 3.2: Implement P2 features
- **Time since open:** `(ts - market_open) / session_duration`
- **Distance from day high/low:** `(ltp - running_max) / running_max`
- **Order arrival rate:** Count DOM snapshots per 1-min bar
- **Depth slope:** Linear regression coefficient of qty vs level index across L1–L10
- **Status:** `[ ]`

#### Task 3.3: Rebuild `order_cancel_rate`
- **Logic:** DOM-diff: qty decrease without subsequent LTP change = cancellation
- **Formula:** `cancel_rate = sum(qty_drops_without_trade) / sum(total_qty_changes)`
- **Acceptance:** Feature no longer all-zeros. Distribution checked for reasonableness.
- **Status:** `[ ]`

#### Task 3.4: Implement MFE Triple-Barrier Labels
- **Upper barrier:** `max(ltp[t:close]) >= entry[t] * 1.05` → LONG
- **Lower barrier:** `min(ltp[t:close]) <= entry[t] * 0.95` → SHORT
- **Time barrier:** market close → NO_TRADE
- **Label = whichever barrier hit first**
- **Also add:** continuous return targets (5m, 15m, 30m, 60m) for IC measurement
- **Acceptance:** New label columns added alongside existing ones. Distribution reported.
- **Status:** `[ ]`

#### Task 3.5: Feature normalization
- **Spread:** Convert to basis points: `(ask1 - bid1) / mid * 10000`
- **Price-derived features:** Normalize to log-return space or divide by running VWAP
- **Acceptance:** No raw price values in feature matrix (all are ratios, bps, or z-scores)
- **Status:** `[ ]`

#### 🧑 CHECKPOINT 3: Human reviews new feature set and label distribution. Approve before re-running baseline.

---

### PHASE 4: RE-RUN BASELINE WITH FIXES (Estimated: 1 day)
*Re-run baseline model with all fixes, new features, new labels.*

#### Task 4.1: Full pipeline re-run (Stages 1–5)
- **Status:** `[ ]`

#### Task 4.2: Add IC, calibration, P&L metrics
- **IC:** `spearmanr(pred_proba[:, LONG], actual_return)` per fold
- **Calibration:** `sklearn.calibration.calibration_curve`
- **P&L simulation:** Enter at LONG signal, hold until 5% hit or close, 0.05% slippage each way
- **Status:** `[ ]`

#### Task 4.3: Report comprehensive results
- MCC (old labels vs new MFE labels)
- IC per fold
- Precision per class
- Hit rate
- Avg return on signals
- Calibration curve
- Simulated Sharpe ratio
- **Status:** `[ ]`

#### 🧑 CHECKPOINT 4: Human reviews results. Decides whether to proceed to data acquisition or iterate on features.

---

### PHASE 5: DATA ACQUISITION + MULTI-DAY VALIDATION (Estimated: depends on vendor)

#### 🧑 Task 5.1: Human acquires additional data
- **Minimum:** 10 days of tick + DOM (same format as current vendor)
- **Recommended:** 60 days across different market regimes
- **Target:** Days from different VIX regimes (calm + volatile)
- **Data sources (in order of preference):**
  1. Current vendor — ask for historical backfill in same L2 DOM format
  2. TickerPlant India — institutional NSE L2 tick data (~₹15–30K/mo)
  3. True Data (TrueBeacon) — NSE tick and DOM (~₹10–25K/mo)
  4. Refinitiv — institutional grade but USD pricing
- **Agent should ask human:** "Please provide additional day-level parquet files in the same hive-style format. How many days can you provide?"
- **Status:** `[ ]`

#### Task 5.2: Ingest new data through pipeline
- Run Stages 0–4 on new data
- Verify schema compatibility with existing pipeline
- **Status:** `[ ]`

#### Task 5.3: Implement walk-forward validation
- Train on first N-3 days, purge 60 bars, test on next day
- Expanding window
- Embargo gap: 30 bars at start of each test day
- **Acceptance:** Per-day MCC reported. Overall MCC ± std. IC per day.
- **Status:** `[ ]`

#### Task 5.4: Regime analysis
- Compute features and model performance separately for:
  - High-volatility days vs low-volatility days
  - Gap-up open days vs gap-down open days
- **Acceptance:** Feature importance stability across regimes reported
- **Status:** `[ ]`

#### 🧑 CHECKPOINT 5: Human reviews multi-day validation results. Decides next steps (more data, model upgrade to LightGBM, or pivot strategy).

---

### PHASE 6: MODEL UPGRADE (Only After Phase 5 ✅)

#### Task 6.1: LightGBM implementation
- `max_depth=4–6`, `num_leaves≤31`, `min_data_in_leaf` controlled
- SHAP feature importance
- Native NaN handling
- **Status:** `[ ]`

#### Task 6.2: LightGBM vs RF comparison
- Same walk-forward CV, same purging
- Compare MCC, IC, Sharpe, calibration
- **Status:** `[ ]`

---

## PART 11: REFERENCE READING (Consensus)

These books were cited by multiple reviewers as non-negotiable:

1. **López de Prado (2018)** — *Advances in Financial Machine Learning*. Chapter 3 (Triple Barrier), Chapter 7 (Feature Importance), Chapter 12 (CPCV)
2. **Cartea, Jaimungal, Penalva (2015)** — *Algorithmic and High-Frequency Trading*. Chapters 6–8 on order flow
3. **Easley, López de Prado, O'Hara (2012)** — *Flow Toxicity and Liquidity in a High-Frequency World*. The VPIN paper
4. **Avellaneda & Stoikov (2008)** — *High-frequency trading in a limit order book*. Foundation for microstructure signals

---

## PART 12: EFFORT ALLOCATION (Professional Benchmark)

From GPT review, confirmed by all others:

| Activity | % of Total Effort |
|----------|:-----------------:|
| **Data acquisition & cleaning** | **60%** |
| **Validation & testing** | **25%** |
| **Infrastructure** | **10%** |
| **Modeling** | **5%** |

> Most performance improvements come from better data and validation, not more sophisticated models.

---

## PART 13: TRADING ECONOMICS & CAPACITY ANALYSIS

*Missing from all 4 consultant reviews. Added per GPT follow-up.*

Everything above is ML-centric. Real firms care about: **Signal Quality → Execution Quality → P&L**, not just MCC.

### 13.1 Questions That Must Be Answered Before Production

| Question | How to Measure | When to Measure |
|----------|---------------|----------------|
| **Expected edge per trade** | Mean return on LONG signals minus slippage (both sides) | Phase 4 (after baseline re-run) |
| **Expected holding period** | Median time from entry to barrier hit (using MFE labels) | Phase 3 (after MFE label implementation) |
| **Slippage kill threshold** | At what slippage level does edge → 0? Compute breakeven slippage | Phase 4 |
| **Strategy turnover** | How many trades per day? Per stock? | Phase 4 |
| **Capacity limits** | At what position size does market impact eat the alpha? | Phase 5+ (requires multi-day data) |

### 13.2 Capacity Analysis Framework

The same signal can be:
```
Amazing at ₹5 lakh per trade
Dead at ₹5 crore per trade
```

**Methodology (implement in Phase 4+):**

For each LONG signal:
1. Look at available liquidity at `ask1` through `ask5` at signal time
2. Compute how many shares you can buy without moving the price beyond level 1
3. That's your **instantaneous capacity per signal**
4. Aggregate across all signals per day: `daily_capacity = sum(per_signal_capacity)`

| Capital Level | Test | Acceptance |
|:-------------:|------|:---------:|
| ₹1 lakh | Can fill within L1 depth? | Almost always |
| ₹10 lakh | Can fill within L1–L3? | Usually for liquid stocks |
| ₹1 crore | Market impact < 20% of expected edge? | Test empirically |
| ₹10 crore | Market impact model required | Needs full impact model |

**Impact cost formula:**
```python
impact_cost = (execution_price - arrival_price) / arrival_price
# where execution_price = volume-weighted avg across all levels consumed
```

### 13.3 Slippage Sensitivity

In Phase 4, compute:
```python
for slippage_bps in [5, 10, 15, 20, 30, 50]:
    net_return = gross_signal_return - 2 * slippage_bps / 10000
    sharpe = compute_sharpe(net_return)
    print(f"Slippage={slippage_bps}bps → Sharpe={sharpe:.2f}")
```

Plot the decay curve. Find the breakeven slippage where Sharpe = 0.

---

## PART 14: SIGNAL DECAY ANALYSIS

*One of the most important HFT metrics. Missing from all consultant reviews.*

### 14.1 What Is Signal Decay?

After generating a signal at time T, how quickly does the expected return materialize and then dissipate?

```python
# For each LONG signal at time T, compute:
return_1m  = (ltp[T+1]  - ltp[T]) / ltp[T]
return_3m  = (ltp[T+3]  - ltp[T]) / ltp[T]
return_5m  = (ltp[T+5]  - ltp[T]) / ltp[T]
return_10m = (ltp[T+10] - ltp[T]) / ltp[T]
return_15m = (ltp[T+15] - ltp[T]) / ltp[T]
return_30m = (ltp[T+30] - ltp[T]) / ltp[T]
return_60m = (ltp[T+60] - ltp[T]) / ltp[T]
```

### 14.2 What to Plot

Plot `mean_return_on_signal` vs `minutes_after_signal`.

**Example of what good looks like:**
```
Signal strongest at 3 minutes   → edge builds quickly
Peak at 15 minutes              → maximum alpha capture
Dead by 45 minutes              → alpha fully decayed
```

**Example of what bad looks like:**
```
Signal flat from 1 to 60 minutes → not a timing signal, it's a stock classifier
```

### 14.3 Implementation

Add to Phase 4 (Task 4.2):
```python
def compute_signal_decay(df, signal_mask, horizons=[1,3,5,10,15,30,60]):
    results = {}
    for h in horizons:
        future_ret = df['ltp'].shift(-h) / df['ltp'] - 1
        results[f'{h}m'] = future_ret[signal_mask].mean()
    return results
```

**Acceptance:** Decay curve plotted. Peak identified. If flat → flag as stock classifier.

---

## PART 15: ALPHA ATTRIBUTION & REGIME DETECTION

### 15.1 Alpha Attribution (SHAP + Feature Interactions)

You know RF works. You don't know **why**.

**Required analysis (implement in Phase 6 with LightGBM):**

| Analysis | What It Reveals | Tool |
|----------|----------------|------|
| **SHAP values** | Per-prediction feature contribution. "This trade was triggered because iceberg_score was 3.2 AND imbalance was 0.7" | `shap.TreeExplainer(lgbm_model)` |
| **Permutation importance** | True feature importance without bias from cardinality | `sklearn.inspection.permutation_importance` |
| **SHAP interaction matrix** | Does `imbalance` matter only when `spread` is tight? Does `iceberg_score` matter only in high volume? | `shap.TreeExplainer(model).shap_interaction_values(X)` |
| **Partial dependence plots** | Non-linear relationship between each feature and prediction | `shap.dependence_plot` or `sklearn.inspection.PartialDependenceDisplay` |

**Key interaction hypotheses to test:**
- `iceberg_score × volume_burst` → iceberg detection more predictive during high-volume bars?
- `imbalance_top5 × spread_bps` → imbalance signal stronger in tight-spread stocks?
- `large_trade_ratio × time_since_open` → large trades more predictive in the morning?

### 15.2 Regime Detection

Most alphas are regime-dependent. A signal that works during trending markets may fail during mean-reversion.

**Intraday regimes (implement as features in Phase 3 and as analysis segments in Phase 5):**

| Regime | Time Window (IST) | Characteristics |
|--------|:-----------------:|-----------------|
| Opening auction | 9:15–9:20 | High volatility, gap resolution, institutional flows |
| Morning trend | 9:20–11:00 | Strongest directional moves. Most 5%+ moves start here |
| Lunch lull | 11:00–13:30 | Low volume, mean-reverting, thin books |
| Afternoon positioning | 13:30–14:30 | Institutional positioning for close |
| Closing auction | 14:30–15:30 | High volume, gamma effects, forced execution |

**Cross-day regimes (implement in Phase 5 with multi-day data):**

| Regime | Proxy | How to Segment |
|--------|-------|---------------|
| High VIX | INDIA VIX > 20 | Wider spreads, more 5%+ moves, shorter signal decay |
| Low VIX | INDIA VIX < 14 | Tighter spreads, fewer 5%+ moves, longer holding periods |
| Trending | Nifty 5-day return > 2% | Directional signals stronger |
| Mean-reverting | Nifty 5-day return within ±0.5% | Imbalance signals stronger |

**Acceptance:** Run separate model evaluations per regime. Report MCC/IC per regime. If variance across regimes > 2×, regime-aware ensemble is justified.

---

## PART 16: RESEARCH INFRASTRUCTURE

### 16.1 Experiment Registry

Professional firms never re-run everything from scratch. Build an experiment tracking system.

**Minimum viable experiment registry (implement as JSON/CSV before Phase 4):**

Each experiment stores:
```json
{
  "experiment_id": "EXP-007",
  "timestamp": "2026-06-11T14:30:00",
  "git_hash": "abc123",
  "features": ["list of features used"],
  "label": "label_mfe_5pct",
  "data_range": "2026-05-29",
  "n_stock_days": 438,
  "n_rows": 88321,
  "model": "RandomForest(n_estimators=200, max_depth=10)",
  "validation": "5-fold stock-level CV",
  "mcc_mean": 0.3137,
  "mcc_std": 0.0150,
  "ic_mean": null,
  "sharpe": null,
  "notes": "Baseline run on corrected data"
}
```

**File location:** `results/experiment_registry.jsonl` (one JSON per line, append-only)

**For later (Phase 6+):** Migrate to MLflow or Weights & Biases for full experiment tracking with artifacts.

### 16.2 Research Kill Criteria

> [!CAUTION]
> **Define failure conditions BEFORE investing more time. Professional firms kill projects aggressively.**

| Kill Condition | Threshold | When to Evaluate |
|---------------|:---------:|:-----------------:|
| No temporal signal after diagnostic tests | All 3 Phase 1 tests FAIL | After Phase 1 |
| Trade inference accuracy too low | Pearson r < 0.50 on trade counts | After Phase 2 |
| MCC on multi-day walk-forward near zero | MCC < 0.03 on held-out days | After Phase 5 |
| IC consistently below noise | IC < 0.02 across 60+ trading days | After Phase 5 |
| Strategy Sharpe < 0 after realistic costs | Sharpe < 0 with 10bps slippage | After Phase 4/5 |
| Feature importance unstable across regimes | Top-3 features change in > 50% of regime splits | After Phase 5 |

**If a kill condition is met:**
1. Document in `docs/REJECTED_APPROACHES.md` with full evidence
2. Analyze what went wrong (stock characterization? data quality? feature design?)
3. Decide: pivot to different approach, acquire different data, or stop

### 16.3 Data Vendor Risk Documentation

**Document for every data source (implement immediately):**

| Risk | What to Track | Impact |
|------|--------------|--------|
| Vendor corrections | Does vendor issue retroactive corrections to historical data? | Invalidates trained models |
| Missing packets | % of expected DOM snapshots per minute actually received | Features built on sparse data are noisy |
| Feed resets | Does feed restart mid-day? Do sequence numbers reset? | Creates artificial gaps in features |
| Clock drift | Is vendor timestamp synchronized to exchange timestamp? | Misaligned Tick+DOM join |
| Exchange outages | NSE circuit breakers, trading halts, market-wide halts | Labels are meaningless during halts |
| Schema changes | Does vendor change column names/types between batches? | Pipeline breaks silently |

**File location:** `docs/DATA_VENDOR_RISK.md` — create before ingesting any new data.

---

## PART 17: PRODUCTION READINESS SCORECARD

*Evaluate before any capital deployment decision.*

| Area | Weight | Current Score | Target | Notes |
|------|:------:|:------------:|:------:|-------|
| **Data quality** | 25% | 8/10 | 9/10 | Good after contamination fix. Deduct for 6 crossed-book stocks and unvalidated trade inference |
| **Signal quality** | 25% | ?/10 | 7/10 | Unknown until Phase 1 diagnostic tests complete |
| **Validation rigor** | 20% | 7/10 | 9/10 | Stock-level CV is correct but needs walk-forward + CPCV on multi-day data |
| **Capacity analysis** | 10% | 0/10 | 6/10 | Zero capacity work done. Not blocking for research but blocking for production |
| **Monitoring** | 10% | 1/10 | 7/10 | Feature range checks exist but no PSI, IC tracking, or calibration monitoring |
| **Infrastructure** | 10% | 6/10 | 8/10 | Pipeline works but lacks resumability, parallel safety, and experiment tracking |
| **Overall** | 100% | **~5.5/10** | **8/10** | Not production-ready. Research-ready with caveats |

**Minimum production threshold: 7.5/10 across all areas, with no single area below 5/10.**

---

## PART 18: PIPELINE SAFETY — HIDDEN OPERATIONAL GAPS

*Identified by Gemini in follow-up review. Not covered by any original consultant.*

### 18.1 Cross-Symbol State Leakage (Stage 2 & 3 Boundary)

**The Problem:** When running parallel jobs across 438 symbols in Stages 2 and 3, sliding feature windows (`volatility_5m`, `volume_burst`) use rolling lookback buffers. If execution reuses buffers across different symbol files within the same thread pool, data from a high-volume stock can bleed into the initial rows of the next stock.

**The Fix:** Enforce stateless feature workers:
```python
# BAD: shared state across symbols
for symbol_file in all_files:
    features = compute_features(symbol_file, shared_buffer)  # ← leaks

# GOOD: isolated workers
for symbol_file in all_files:
    features = compute_features(symbol_file)  # fresh state per symbol
    assert features.iloc[0]['symbol'] == expected_symbol  # identity check
```

**Acceptance:** Add a post-condition to Stage 3: for each output file, sample 10 random rows and verify `symbol` column matches filename. (Same pattern that would have caught the Stage 0 contamination bug.)

**Status:** `[ ]` — Add to Phase 0 as Task 0.8.

### 18.2 Order Book Clock Synchronization

**The Problem:** DOM feed operates at ~21µs update scale. Tick files have separate timestamps. When the pipeline joins these in Stage 3, what happens when timestamps disagree within the same 1-minute bar?

**The Fix:** Define an explicit clock synchronization threshold:
- DOM-to-tick matching window: ±50ms (configurable)
- If no tick falls within ±50ms of a DOM snapshot, mark as `unmatched`
- Log `unmatched_rate` per stock. If > 20%, flag for investigation.

**Status:** `[ ]` — Add to Phase 0 as Task 0.9.

### 18.3 Market Impact in MFE Label Computation

**The Problem:** MFE labels assume instantaneous execution at `ask1[T]`. For mid-cap and small-cap NSE stocks, executing institutional-sized orders clears through multiple depth levels.

**The Fix:** When computing MFE labels, add a market impact penalty:
```python
# Instead of: entry_price = ask1[T]
# Use:        entry_price = ask1[T] + impact_estimate
# where impact_estimate scales with order book slope:
impact_estimate = target_shares / available_depth_L1_to_L5 * avg_spread
```

For initial research, use a flat 5bps impact assumption. Refine later with actual depth data.

**Status:** `[ ]` — Address in Phase 3 (Task 3.4).

---

## PART 19: PROJECT RATINGS (External Assessment)

*From GPT follow-up review, representing a research committee perspective.*

| Dimension | Rating | Notes |
|-----------|:------:|-------|
| Engineering | **9/10** | Contamination RCA especially impressive |
| Research process | **8.5/10** | Better than many small prop desks |
| Statistical rigor | **7/10** | Good but needs multi-day validation |
| Data | **3/10** | Single day. Everything bottlenecked here |
| Production readiness | **5.5/10** | Not because code is bad — because 1 day of data means nothing is proven yet |

**Committee verdict: GO** — with one condition:

> **Freeze feature expansion and model experimentation until the temporal-signal diagnostics (Phase 1) and trade-inference validation (Phase 2) are complete.** Those two phases will tell you whether you have discovered an actual alpha source or merely a sophisticated stock classifier. The entire future direction of the project depends on that answer.

---

## SUMMARY — What the Agent Should Do

1. **Read this document fully before starting any work**
2. **Execute Phase 0 (bug fixes) first — no exceptions**
3. **Execute Phase 1 (diagnostic tests) to determine if temporal signal exists**
4. **Respect HALT_AND_REPORT gates — if any diagnostic test FAILs, stop and report to human**
5. **At every 🧑 CHECKPOINT, stop and present results to the human**
6. **Ask the human for additional data when you reach Phase 5**
7. **Never skip ahead to model upgrades before validation confirms signal**
8. **Track all experiments in `results/experiment_registry.jsonl` for Deflated Sharpe Ratio accountability**
9. **Evaluate kill criteria at each checkpoint — document killed approaches in `docs/REJECTED_APPROACHES.md`**
10. **Update docs/DECISION_LOG.md, docs/TASKS.md, docs/CURRENT_STATE.md after every significant change**
11. **Create `docs/DATA_VENDOR_RISK.md` before ingesting any new data**
12. **Compute signal decay curve and capacity analysis in Phase 4 — not optional**
