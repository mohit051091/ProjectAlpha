# Live ML Scanner — Architecture & Plan

## Objective
Generate identical signals to the backtest sweep (L20_S25) in real-time, using live QuestDB tick data, with zero impact on the existing collector.

## Guiding Principles
- **Zero risk to collector**: ML scanner is a completely separate process. Reads QuestDB only (read-only). No collector code changes.
- **Backtest parity**: Same 18 features, same LightGBM model, same Double ROC(3) state machine, same data filter (09:15-15:29 IST).
- **Rollback-able**: Separate Railway service. Delete the service = full rollback. Collector untouched.
- **Backtrace-able**: Every feature vector, prediction, and state transition logged to QuestDB `ml_features` + `ml_signals` tables.

## Architecture

```
DhanHQ WebSocket
       │
       ▼
┌──────────────────────┐
│  Collector (existing)│  Two Railway services (mohit + rahul roles)
│  No changes needed   │  Writes ticks + delta_bars to QuestDB via ILP
└──────────┬───────────┘
           │  (ILP write, ~500ms batch delay)
           ▼
┌──────────────────────┐
│  QuestDB (existing)  │
│  Tables: ticks,      │
│  delta_bars, signals │
└──────────┬───────────┘
           │  Read-only (PGWire)
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  ML Scanner (NEW — separate Railway service)                    │
│                                                                  │
│  python ml_scanner.py                                           │
│                                                                  │
│  Startup:                                                        │
│    1. Backfill rolling buffers from last 20 min of ticks         │
│    2. Load LightGBM model                                        │
│    3. Initialize per-symbol accumulators + state machines        │
│                                                                  │
│  Every 1 second:                                                 │
│    1. Query QuestDB: SELECT * FROM ticks WHERE ts > last_ts     │
│    2. Update per-symbol 1-minute accumulators                   │
│    3. At :00 boundary:                                           │
│       a. Flush → 1-min bar                                      │
│       b. Push to rolling buffers                                │
│       c. Compute 18 features                                    │
│       d. LightGBM predict → probs                               │
│       e. Double ROC(3) state machine step                       │
│       f. Write to ml_features table (every minute)              │
│       g. If TRIGGERED → write to ml_signals + HTTP POST pmocata │
│                                                                  │
│  Env vars:                                                      │
│    ML_MODE = VIRTUAL | LIVE                                     │
│    QUESTDB_HOST, QUESTDB_PORT, QUESTDB_USER, QUESTDB_PASSWORD   │
│    PMOCATA_URL                                                  │
│    LGBM_MODEL_PATH                                              │
└──────────────────────┬───────────────────────────────────────────┘
           │  HTTP POST /webhook (if TRIGGERED)
           ▼
┌──────────────────────┐
│  pmocata (existing)  │  Separate Railway service
│  server.py           │  Places BO via Dhan API → /v2/orders
└──────────────────────┘
```

## Signal JSON (POST to pmocata /webhook)

```json
{
  "action": "ENTRY",
  "signal": "LONG",
  "tradingSymbol": "MSUMI",
  "exchangeSegment": "NSE_EQ",
  "entryPrice": 38.90,
  "mode": "LIVE"
}
```

When `ML_MODE=VIRTUAL`, `"mode": "VIRTUAL"` is sent — pmocata logs the signal without placing any order.

## Env Vars

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `ML_MODE` | `VIRTUAL` | No | `VIRTUAL` = log only, `LIVE` = execute trades |
| `QUESTDB_HOST` | `localhost` | No | QuestDB host for tick reads |
| `QUESTDB_PORT` | `8812` | No | QuestDB PGWire port |
| `QUESTDB_USER` | `admin` | No | QuestDB username |
| `QUESTDB_PASSWORD` | `quest` | No | QuestDB password |
| `PMOCATA_URL` | — | No | URL of pmocata server (if empty, no POST) |
| `LGBM_MODEL_PATH` | `models/lgbm_model_60m_1pct_final.txt` | No | Path to LightGBM weights |
| `SYMBOLS_PATH` | — | No | CSV of symbols to scan (default: all 500) |

## Acceptance Criteria

### Must Pass Before Deployment
1. **Time filter**: Data outside 09:15:00-15:29:59 IST is NEVER used for feature computation, model prediction, or state machine. Verified by running on an archived day and checking all timestamps.
2. **Backtest parity**: Running the ML scanner on an archived day's ticks produces the **same trade signals** (same symbol, same direction, same entry time ±1s) as the batch sweep for L20_S25. Verified by comparing `ml_signals` output against sweep sheet `L20_S25`.
3. **Zero collector impact**: The collector runs unchanged. No new dependencies, no new threads, no new sockets in collector.py.
4. **Rollback**: Deleting the ML scanner Railway service restores the system to its previous state with zero side effects.
5. **VIRTUAL mode**: With `ML_MODE=VIRTUAL`, signals are written to QuestDB `ml_signals` table but NO HTTP POST is sent to pmocata.

### Performance
6. **Signal latency**: Signal fires within **<2 seconds** of bar close (1s poll + <1s feature compute).
7. **CPU**: Scanner uses **<5% of one core** during market hours.
8. **RAM**: Scanner uses **<100 MB**.

## Test Cases

### Unit Tests (features/live_computer.py)
```
test_ring_buffer_push_pop:
  Given a RingBuffer(3), after pushing [1,2,3,4],
  the buffer contains [2,3,4] (oldest evicted).

test_18_features_match_batch:
  Given one minute of archived tick data,
  the 18 features computed by LiveFeatureComputer
  match the batch pipeline's output to 6 decimal places.

test_volume_burst_rolling:
  Given 20 minutes of constant volume=100,
  volume_burst = 1.0 (equal to own rolling mean).

test_roc_3_continuous:
  Given ltp history [100, 101, 102, 103],
  roc_3 = (103-100)/100 * 100 = 3.0.
```

### Integration Tests
```
test_bar_accumulator_single_tick:
  Given one tick at ts=09:15:03,
  the bar accumulator has open=close=high=low=ltp.

test_bar_accumulator_flush_at_minute:
  Given ticks from 09:15:00 to 09:15:59,
  flush() at 09:16:00 returns a complete bar.

test_bar_accumulator_no_pre_market:
  Given ticks at 09:09:00 IST,
  the accumulator does NOT flush a bar (time filter).

test_state_machine_idle_to_arming:
  Given prob_long=0.25 (≥LONG_TH=0.20) and state=IDLE,
  step() transitions to ARMING.

test_state_machine_full_trigger:
  Given a sequence of probs and roc_3 that match
  IDLE→ARMING→MARK1→BETWEEN→TRIGGERED,
  step() returns a Signal at TRIGGERED.

test_questdb_tick_read:
  Given 10 ticks in QuestDB with ts > last_ts,
  fetch_new_ticks() returns exactly those 10 ticks.
```

### End-to-End Test
```
test_live_vs_sweep_parity:
  Given one archived day of ticks in QuestDB,
  run ML scanner on that day.
  Compare ml_signals output vs batch sweep L20_S25 sheet.
  Assert: same symbol, same direction, same ±1min entry time.
  Assert: no false triggers from pre-market data.
```

## Rollback Procedure

If the ML scanner causes any issue:
1. Stop the service: `railway service delete ml-scanner`
2. Collector continues unaffected (never shared code)
3. QuestDB `ml_features` + `ml_signals` tables can be dropped: `DROP TABLE ml_features, ml_signals`
4. OOF predictions parquet is unaffected (read-only for scanner)
5. Full restore in <2 minutes

## Files to Create (4 new, 0 modified)

| File | Purpose | Lines |
|---|---|---|
| `features/bar_accumulator.py` | Per-symbol 1-min bucket from ticks (time-filtered) | ~80 |
| `features/live_computer.py` | Rolling buffers + 18-feature formulas, matches batch exactly | ~150 |
| `features/state_machine.py` | Double ROC(3) per-symbol class, L20_S25 thresholds | ~120 |
| `ml_scanner.py` | Main loop: QuestDB poll → accumulate → compute → predict → state machine → pmocata POST | ~200 |

**Zero changes to collector.py.**
**Zero changes to pmocata server.py.**
