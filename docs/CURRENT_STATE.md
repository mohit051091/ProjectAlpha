# Current State

**Date**: 2026-07-21

## 2026-07-21 Refresh — Jul 20-21 Added, 38 Dates Complete (Final Coverage)

### Pipeline
- **Jul 20-21 pipeline** completed (Stages -1→4) in 12,380s (~3.4h)
  - Stage -1: 1,080s (3 unified tick files)
  - Stage 1: 500s
  - Stage 2: 260s
  - Stage 3: 10,136s (1,946 symbols, ~5.1s per symbol)
  - Stage 4: 395s (1,010 new labeled files)
- **Total coverage**: 38 dates (May 26 – Jul 21), 510 symbols

### Final Model
- **Retrained** `models/lgbm_model_60m_1pct_final.txt` on all 5,929,963 rows, 38 dates
- **OOF rebuilt**: `results/oof_predictions_lgbm_label_60m_1pct.parquet` — 5,929,963 rows, 510 symbols, 38 dates

### Sweeps — Active Config L30_S25
| Metric | 36 dates | 38 dates | Change |
|--------|----------|----------|--------|
| Total Trades | 306 | 308 | +2 |
| Win Rate | 57.19% | 56.49% | -0.70pp |
| Cumulative Return | +202.53% | +196.16% | -6.37pp |
| LONG / SHORT | 182 / 124 | 177 / 131 | -5 / +7 |

## 2026-07-17 Refresh — Jul 15-17 Added, 36 Dates Complete

### Pipeline
- **Jul 15-17 pipeline** completed (Stages -1→4) in 18,605s (~5.2h)
  - Stage -1: 1,144s (4 unified tick files converted)
  - Stage 1: 681s
  - Stage 2: 397s
  - Stage 3: 15,991s (2,379 symbols processed, ~6.6s per symbol)
  - Stage 4: 382s (1,443 new labeled files)
- **Total coverage**: 36 dates (May 26 – Jul 17), 510 symbols

### Final Model
- **Retrained** `models/lgbm_model_60m_1pct_final.txt` on all 5,553,130 rows, 36 dates
- **OOF rebuilt**: `results/oof_predictions_lgbm_label_60m_1pct.parquet` — 5,553,130 rows, 510 symbols, 36 dates

### Sweeps — Active Config L30_S25
| Metric | 33 dates | 36 dates | Change |
|--------|----------|----------|--------|
| Total Trades | 287 | 306 | +19 |
| Win Rate | 57.49% | 57.19% | -0.30pp |
| Cumulative Return | +212.32% | +202.53% | -9.79pp |
| Mean MFE | +2.54% | +2.54% | — |
| MFE ≥ 5% | 15.68% | 15.69% | — |
| LONG / SHORT | 172 / 115 | 182 / 124 | +10 / +9 |

## 2026-07-15 Refresh — Jul 13-14 Added, 33 Dates Complete

### Pipeline
- **Jul 13-14 pipeline** completed (Stages -1→4) with DuckDB `TIMESTAMPTZ→TIME` fix in `raw/data_cleaner.py`
- **Stage 3** processed 1246 files in 205s (3.4 min) — idempotent resume
- **Stage 4** labeled 1012 new feature files (16 CHOLAFIN files skipped, pre-existing `ltp` issue)
- **Total coverage**: 33 dates (May 26 – Jul 14), 506–510 symbols

### Final Model
- **Retrained** `models/lgbm_model_60m_1pct_final.txt` on all 5,014,998 rows, 33 dates (vs prior data through Jul 3 only)
- **OOF rebuilt**: `results/oof_predictions_lgbm_label_60m_1pct.parquet` — 5,014,998 rows, 510 symbols, 33 dates

### Sweeps — Active Config L30_S25
| Metric | 31 dates (old model) | 33 dates (retrained) | Change |
|--------|---------------------|---------------------|--------|
| Total Trades | 324 | 287 | -37 |
| Win Rate | 55.25% | **57.49%** | +2.24pp |
| Cumulative Return | +230.40% | **+212.32%** | -18.08pp |
| Mean Return | — | +0.74% | — |
| Mean MFE | — | +2.54% | — |
| MFE ≥ 5% | — | 15.68% | — |
| LONG / SHORT | — | 172 / 115 | — |

### DuckDB Fix
- `raw/data_cleaner.py` lines 387, 388, 455: replaced `(ts + INTERVAL 5 HOUR + INTERVAL 30 MINUTE)::TIME` with `(ts AT TIME ZONE 'Asia/Kolkata')::TIME` for DuckDB 1.5.3 compatibility (TIMESTAMPTZ cast removed in v1.5.0+)

## 2026-07-11 Refresh — Discard Pre-Market Data in Backtest for 100% Parity
- **Pre-Market Time Filtering**:
  - Discarded all ticks and DOM snapshots outside market hours (09:15:00 AM to 15:29:59.999999 PM IST) during the Stage 1 cleaning process in `raw/data_cleaner.py`.
  - This eliminates the pre-market warm-up discrepancy (e.g. `SONATSOFTW` on July 10) where the backtest armed early at `09:15:00` because it had pre-market history, while the live scanner had none and correctly skipped the first bar.
  - Now, `ROC_3` and all other rolling features are guaranteed to start with NaNs for the first 3 bars (09:15, 09:16, 09:17) in both the backtest and live scanner, aligning them to the letter.

## 2026-07-11 Refresh — Live Scanner Self-Healing Watchdog and Scheduled Login Fix
- **Live Scanner Self-Healing Watchdog**:
  - Implemented an individual symbol-level watchdog inside `Clean_Scanner/collector.py` that audits tick frequencies every 60 seconds.
  - If any NSE Equity (`EQ` series) symbol goes silent for more than 120 seconds during market hours, it triggers up to 3 sequential re-subscription attempts to the Dhan WebSocket gateway.
  - This solves the pre-market drop issue where Dhan silently dropped symbol subscriptions (like `SWIGGY` on July 9) during rapid watchdog reconnections.
- **Dhan Token Invalidation Fix**:
  - Identified that the embedded `pmocata` server's Daily 9 AM IST Login Thread generated a new token at `09:00:26 AM IST` (15 minutes after container boot at 08:45 AM).
  - This invalidated the startup token used by the active WebSocket workers. When they disconnected and tried to reconnect, they were locked out with `HTTP 429` (too many requests) for the rest of the day.
  - Updated `pmocata-main/server.py` to track the token's refresh time and **skip** the scheduled 9 AM login if the token was refreshed recently (under 4 hours ago) at container startup.

## 2026-07-11 Refresh — EQ Series Only Filtering and Online Dhan Scrip Master Update
- **EQ Series Only Alignment**:
  - Aligned both the live scanner (`Clean_Scanner/collector.py`) and the offline pipeline (`ProjectAlpha/scripts/update_whitelist.py`) to strictly trade the standard **`EQ`** series cash equities.
  - This prevents intraday trading of **`BE`** series (Trade-to-Trade) stocks, which block intraday squaring-off and carry delivery/auction penalties.
- **Dynamic Online Scrip Master Ingestion**:
  - The offline pipeline now downloads the latest `api-scrip-master.csv` online from Dhan on launch, filters it for NSE Segment `E` and Series `EQ`, and updates `config/equity_symbols.parquet` against your manual Nifty 500 watchlist.
  - Discovered that the VOGL trade omission offline was due to it listing on June 15 under the `BE` series, meaning it was filtered out by the stale June 22 offline whitelist, but was processed live (after transitioning to `EQ` on June 29). Now, the online update keeps the whitelist fresh.

## 2026-07-08 Refresh — Jul 6-7 Added
- **Data coverage extended through Jul 07** via idempotent pipeline resume.
- **Pipeline stages complete for Jul 6-7**:
  - `features_1m`: 498/498 symbols per day
  - `labeled_features_1m`: 498/498 symbols per day
  - 2 symbols per day had no data (no cleaned_dom files)
- **OOF extended**: 3,876,659 → 4,232,421 rows (+355,762, 28 total dates)
  - Jul 6: 177,373 rows, long_avg=0.0518, short_avg=0.0408
  - Jul 7: 178,389 rows, long_avg=0.0518, short_avg=0.0411
- **Sweeps regenerated**: `results/performance_sweeps.xlsx` now covers 28 dates (May 26–Jul 7)
  - L30_S25 (active config): 292 trades (165 LONG + 127 SHORT), 54.79% win rate
  - All 8 threshold pairs regenerated in same workbook
  - Jul 6-7 prob_long>=0.30 signals: 3 (Jul 6) + 12 (Jul 7) = 15 total
  - Jul 6-7 prob_short>=0.25 signals: 18 (Jul 6) + 32 (Jul 7) = 50 total
- **Jul 6-7 trades vs prior**: Net trade count changed from earlier snapshot due to intervening OOF extensions (Jun 15–Jul 3 were added after original 323-trade report). Current counts supersede all prior snapshots.
- **CHOLAFIN files across all dates**: 16 labeled feature files missing `ltp` column — skipped during label generation (pre-existing issue, not Jul 6-7 specific).

## 2026-07-05 Refresh Snapshot (Supersedes prior 20-day snapshot)
- **Data coverage now complete through Jul 03** using a DuckDB-first, date-targeted incremental flow.
- **Stage completeness validated** for `29_Jun_26`, `30_Jun_26`, `01_Jul_26`, `02_Jul_26`, `03_Jul_26`:
  - `cleaned_dom=500`, `inferred=500`, `features=500`, `labeled=500` for each day.
- **New OOF source built from labeled parquet via DuckDB**:
  - `results/oof_source_full_2026_07_03.parquet`
  - Coverage: `2026-05-26` to `2026-07-03`
  - Rows: `3,876,659`
- **Auto-research (execute mode, 4 target families) rerun on updated dataset** with target-specific sweeps regenerated:
  - `results/performance_sweeps.xlsx`
  - `results/performance_sweeps_label_30m_1pct.xlsx`
  - `results/performance_sweeps_label_60m_3pct.xlsx`
  - `results/performance_sweeps_label_120m_3pct.xlsx`
  - All include dates up to `2026-07-03`.
- **Current best recommendation (latest)**:
  - Target label: `label_30m_1pct` (`early_momentum_30m_1pct`)
  - Pair: `L25_S30` (`LONG_TH=0.25`, `SHORT_TH=0.30`)
  - Score: `0.6140`
  - Executed trades: `172` across `21` days (`8.19/day`)
  - Win rate: `51.74%`
  - Mover precision: `86.90%`
  - Mover recall: `13.11%`
- **Latest recommendation artifacts**:
  - `research/mover_auto_researcher_v1/reports/live_recommended_settings.json`
  - `research/mover_auto_researcher_v1/reports/latest_result_snapshot.json`
  - `research/mover_auto_researcher_v1/proposals/auto_generated_candidates.json`
- **Model channel separation (new)**:
  - Auto-research releases: `models/auto_research/releases/2026-07-05_jul03_refresh/`
  - Live-production reference: `models/live_production/current/`
  - Channel guide: `models/MODEL_CHANNELS.md`

## Status
**Final strategy deployed**: L30_S25 (LONG θ=0.30, SHORT θ=0.25) with Double ROC(3), daily open filter, LONG bypass exceptions (prob ≥ 0.35). Model retrained on 20 days (May 26–Jun 25) with median fix + depth-5 features. Live ML scanner running on Railway: VIRTUAL mode, posting signals to pmocata with full trade management params (transactionType, sl/tp, capitalReservePct, maxStocksPerDay, maxActiveLongs/Shorts).

### Final Sweep Results (20 trading days, 2.3M OOF rows)
**L30_S25**: 148 trades, 58.11% win rate, +146.97% cum return, 0.99% mean return, 18.24% MFE≥5%.

| Config | LONG θ | SHORT θ | Total Exec | LONG | SHORT | Win Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| L30_S25 | 0.30 | 0.25 | 148 | 184 | 139 | 58.11% |

## Live Scanner Deployment (Clean_Scanner/)
- **3 services on Railway**: questdb, dhan-collector-mohit, dhan-collector-rahul (pmocata standalone deleted — embedded into collector_mohit)
- **ML lives in features/**: bar_accumulator, live_computer, state_machine, ml_engine
- **VIRTUAL mode**: All signals posted to pmocata with `"mode":"VIRTUAL"` + full trade management payload
- **Env vars for ML**: LGBM_MODEL_PATH, LONG_TH/SHORT_TH, ML_MODE, PMOCATA_URL, SL_PCT/TP_PCT, ML_USE_SUPER_ORDER, ML_CAPITAL_RESERVE_PCT, ML_MAX_STOCKS_PER_DAY, ML_MAX_ACTIVE_LONGS, ML_MAX_ACTIVE_SHORTS
- **Signal latency**: ~1-3s from bar completion to pmocata POST
- **EQ filter**: Uses SEM_SEGMENT from scrip master (not symbol name heuristics)
- **Daily open filter**: ENFORCED — failed signals skipped (no POST); LONG bypass at prob ≥ 0.35, SHORT strict
- **pmocata embedded**: Copied into `Clean_Scanner/pmocata/`, started in background thread when `COLLECTOR_ROLE=mohit`. Uses collector's existing `DHAN_ACCESS_TOKEN` (reads env first, skips auto-login). Runs on Railway `$PORT` (8080). Deployed as single Railway service — no separate pmocata service.
- **Domain port fix**: Railway domain `targetPort` was 5001 but Flask listens on `$PORT` (8080). Updated to 8080 via dashboard. `/status` (200) and `/webhook` (400 for bad payloads) now functional.

## Key Fixes Deployed
- **Jul-05 dataset completion + refresh**: Completed missing date processing through Jul-03 with DuckDB-first staged execution; rebuilt OOF source from labeled parquet (`oof_source_full_2026_07_03.parquet`); reran multi-label auto-research and regenerated all target sweep workbooks.
- **Auto-research label-family update (Jul 4)**: Removed EOD label-family from auto-research loop and switched to curated intuitive families (`30m_1pct`, `60m_1pct`, `60m_3pct`, `120m_3pct`) with target-aware train/refresh/sweep scripts.
- **transactionType in payload**: SHORT now sends `"transactionType":"SELL"` (was missing, would default to BUY)
- **Trade management params**: capitalReservePct, maxStocksPerDay, maxActiveLongs/Shorts sent from ml_engine env vars
- **Symbol name resolution**: sid_to_symbol dict in collector.py, resolved in pmocata logs
- **403 retry**: pmocata server.py retries on HTTP 403 once
- **WebSocket reconnect**: exponential backoff death spiral fix
- **Median/lookahead fix**: large_trade_count running median, depth limited to 5
- **5 live-vs-batch fixes (Jul 2)**: trade_qty→volume, removed bid1=0 filter, day boundary clear, RingBuffer(5) volatility, iceberg_score=0. All committed to `main`.
- **pmocata embed**: TokenManager reads DHAN_ACCESS_TOKEN from env first. No more rate-limit race at 08:45 startup.
- **build_subscriptions return fix**: Was accidentally inside `_start_pmocata_webhook`, causing `NoneType` crash.
- **Domain targetPort fix**: 5001→8080 resolved 502 Bad Gateway on public URL.
- **Cross-boundary trade inference fix (Jul 4)**: `_prev_dom` saved before `flush()/reset()`, restored after. Bar queued as `_pending_bar`, returned on next tick. Eliminated ~14% volume mismatch (now <0.5%). Pushed as `7e70b74`.
- **ML Engine constructor fix (Jul 4)**: Updated constructor signature to take `symbol_segments` and `sid_to_symbol` to resolve initialization TypeError crash on startup.
- **WebSocket Reconnection & HTTP 429 backoff (Jul 4)**: Monitored the background `feed_thread` directly, and added a 30s+ retry delay if rate limited (HTTP 429) to allow gateway session cleanup.
- **Bar Accumulator direct return & skipped ticks bug fix (Jul 4)**: Replaced `_pending_bar` mechanism with direct `flushed_bar` return, resolving the critical skipped tick bug where the second tick of every new minute was dropped. Achieved 100.0% perfect feature parity.
- **Zero-trade VWAP and Depth Drop defaults (Jul 4)**: Defaulted `vwap_1m` to `0.0` when trade quantity is 0, and initialized `depth_drop_bid` and `depth_drop_ask` to `0.0` on the first bar (when previous is 0.0) to match offline pandas calculation behavior.
- **Daily Open dynamic initialization (Jul 4)**: Initialized daily open from the first tick's `"open"` field of Dhan ticks, making the daily open filter robust to collector restarts.

## Deployment Instructions (for next LLM)
```bash
# Push code
cd Clean_Scanner && git add -A && git commit -m "<msg>" && git push

# Deploy (pulls latest git commit — faster than railway up which hangs)
railway service redeploy -s dhan-collector-mohit --from-source -y
railway service redeploy -s dhan-collector-rahul --from-source -y

# Check status
railway service list
```

## Pipeline Commands (for adding more days)
```bash
# Full pipe
python run_pipeline.py

# Skip split (if raw ticks already split)
python run_pipeline.py --skip-split

# Retrain only
python scripts/80_train_final_models.py

# Refresh OOF predictions + generate sweeps
python scripts/99_refresh_oof_predictions.py
python scripts/76_generate_v2_sweeps.py
```

## Data Quality
- **Jun 5 excluded**: Tick file contains May 29–30 data, not Jun 5.
- **Depth-5 consistency**: All features use depth 5 only. Unified feed auto-detected.
- **Trade inference depth-agnostic**: `_detect_depth()` auto-detects available bid/bqty columns.
- **Walk-forward killed**: DuckDB approach unusably slow on 2.3M rows (11+ hrs per fold). Use OOF refresh instead.

## Known Issue
- `scripts/77_generate_unified_predictions.py` had LONG/SHORT pred indices inverted (fix applied Jul 1). Pipeline script only, not deployed.
- **AEGISLOG Jun 11**: +8.91% net P&L across 2 LONG trades (Open filter blocked morning SHORT fakes)
- **ZEEL Jun 9–11**: Model captured SHORT on plunge day (+5.49% MFE) and LONG on rally day (+6.03% MFE), avoided flat day
- **Z-Score gates (v1)**: Threshold 0.30 achieved 44.55% selection accuracy, 18.3 trades/day — deferred in v2
- **Hybrid Strategy (v1)**: LONG bypass enabled, SHORT bypass disabled — slashed 122 fake trades, retained 95 movers

## Model
- Retrained on 1,541,300 rows (Jun 3–12 + Jun 15–16, depth 5 features)
- 18 features, 3-class LightGBM. `prob_long = preds[:, 2]`, `prob_short = preds[:, 0]`
- Walk-forward OOF: 1,319,475 rows, 10 dates. Folds 1–7 (Jun 3–12), folds 8–9 (Jun 15–16)
- Jun 15 avg prob_long=0.064, Jun 16 avg prob_long=0.059 (fair out-of-sample estimates via walk-forward)

## Key Output Files
- **v2_performance_sweeps.xlsx**: 7 sheets (3 Strategy_Sweep_*, 3 Sweep_*, All_Movers)
- **oof_predictions_lgbm_label_60m_1pct.parquet**: 1,319,475 rows
- **lgbm_model_60m_1pct_final.txt**: 18-feature model weights

## Priorities & Next Steps
1. **✅ Live vs batch comparison testing**: COMPLETED — replay comparison shows **0.000000** error across all features, achieving **100.0% perfect feature parity**. All discrepancies fixed (including MLEngine initialization, reconnection thread checking, HTTP 429 cool-down, zero-trade VWAP guard, dynamic daily open initialization, first-bar depth drops default, and the critical skipped ticks boundary bug fix).
2. **Multi-day data acquisition**: Need 30+ additional days for robust validation
3. **Decouple strategy**: Move `simulate_double_roc()` into its own module (`strategy/double_roc.py`)
4. **Granular P&L analysis**: Per-trade MFE/MAE distributions across configs
5. **Z-score & turnover gate re-integration**: Activate on top of Double ROC(3) state machine
6. **Regime analysis**: VIX, gap-up/gap-down segmentation
7. **Live execution bridge**: Connect ML signals to broker API (pmocata embed + 502 fix complete)
8. **Live market verification**: Wait for Monday to confirm Railway deployment produces correct signals for non-ZEEL symbols



