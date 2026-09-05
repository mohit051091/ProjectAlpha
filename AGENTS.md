# ProjectAlpha — Pre-Move Detection System

## Project Identity
Quant research: detect NSE equities before large moves using Tick + DOM (Limit Order Book) data. ~510 symbols, 38 trading days (May 26 – Jul 21, 2026). DuckDB disk-based processing, LightGBM.

## Entry Points
- **Full pipeline**: `python run_pipeline.py [--skip-split] [--skip-model] [--force]`
- **Stage scripts**: `scripts/05_split_day_level.py` → `10_prepare_data.py` → `20_infer_trades.py` → `30_compute_features.py` → `40_generate_labels.py` → `50_baseline_model.py`
- **Post-pipeline**: `50_backtest_signals.py`, `60_alert_trigger_backtest.py`, `70_walk_forward_validation.py`, `75_tune_lightgbm_threshold.py`, `80_train_final_models.py`, `76_generate_v2_sweeps.py`
- **OOF rebuild** (after adding dates): Run `80_train_final_models.py` then predict on all labeled features using final model — faster than re-running walk-forward
- **Tests**: `python -m pytest tests/`
- **No packaging**: plain Python, no requirements.txt/pyproject.toml/CI

## Architecture — Corrected Facts
- **DuckDB** processes on-disk; bursts RAM on read (not streaming as ARCHITECTURE.md says)
- **Clock sync**: ±1s tolerance between Tick and DOM streams
- **Time sync stage** (`features/alignment.py` — `EventAlignmentEngine.align_tick_dom()`): KEEP for data up to Jun 12 (separate DOM + Tick files). Auto-detect now implemented in `feature_factory.py`: if DOM columns (`bqty1`) already exist inside tick parquet (unified feed), alignment is skipped automatically.
- **DOM depth**: Removed `imbalance_top10` and `imbalance_top20` from all code, features lists, and registry. Model retrained with 18 features (depth 5 only). Past data still has 20 levels in parquet files — the extra columns are simply ignored.
- **Raw tick parquets are corrupted up to Jun 12** — trades reconstructed from DOM level changes via Trade Inference Engine (`scripts/20_infer_trades.py`)

## Final Strategy (Active)
| Component | Setting |
|---|---|
| Target label | `label_60m_1pct` (triple-barrier, 60min, 1%) |
| ML thresholds | LONG ≥ 0.30, SHORT ≥ 0.25 |
| ROC confirmation | 9-period ROC ≥ +1% (LONG) / ≤ -1% (SHORT), precalculated across day boundaries |
| Z-score gates | Cross-sectional Z of `imbalance_top5` + `aggressor_ratio`. Two paths: Steady Flow or Block Pullback (see HANDOVER_STRATEGY §4D) |
| Turnover gate | `delta_value_lakhs` = delta_1m × entry price / 100,000 |
| Daily Open filter | Close > Open for LONG, Close < Open for SHORT |
| Bypass exceptions | LONG: enabled (high prob ≥ 0.35 OR extreme Z < -1.5). SHORT: disabled |
| Deduplication | Separate per direction (not chronological lock) |
| Exit | EOD only — no intraday exits or overnight holds |
| Walk-forward folds | 7 folds, expanding window, 60-min purge |
| Selected thresholds | Probability 0.30; Z-score 0.30 |

## Final Output
- **Trades**: `results/performance_sweeps.xlsx` → sheet `L30_S25` (308 trades: 177 LONG + 131 SHORT)
- **Raw sweeps**: 33,182 symbol-date rows across 8 config sheets
- **OOF**: `results/oof_predictions_lgbm_label_60m_1pct.parquet` (5,929,963 rows, 510 symbols, 38 dates)
- **Model**: `models/lgbm_model_60m_1pct_final.txt` (retrained on 38 dates, Jul 21)

## Documentation — Read Before Changes
| Priority | File | What it contains |
|---|---|---|
| 1 | `docs/HANDOVER_STRATEGY_REPORT.md` | **Full strategy spec**: trade decision flow, Z-score math, filter rules, pending work (data gap, live bridge) |
| 2 | `docs/DECISION_LOG.md` | Every technical decision with rationale. Read this for full history. |
| 3 | `docs/CURRENT_STATE.md` | Bottlenecks, priorities, latest performance sweeps, RCA audits |
| 4 | `docs/ARCHITECTURE.md` | Pipeline mermaid diagram (notes: DuckDB not streaming; clock sync 1s not 50ms) |
| 5 | `docs/PROJECT_PLAYBOOK.md` | Project objective, subsystems, data inputs |
| 6 | `graphify-out/GRAPH_REPORT.md` | Auto-generated code dependency graph |

## Quant Research Rules
- No claims from one stock / one day / one dataset
- Validate on unseen data. Do not optimize thresholds before leakage/overlap/cross-stock audit
- Simpler features preferred when predictive power is similar
- If insufficient stock-days: stop feature/model/threshold work, document gap, request data
- More data > more features > more models > more optimization
- Verify labels validated, backtests validated, audits complete before proposing any model

## Workflow
1. Read `graphify-out/GRAPH_REPORT.md` before touching source files
2. Before changes: review doc priority list above
3. After changes: update Decision Log, Task Registry, Timeline, Current State
4. Run `graphify update .` after every session

## Live ML Scanner (4 new files in features/)
| File | Purpose | Tests |
|---|---|---|
| `features/bar_accumulator.py` | Per-symbol 1-min bucket from ticks (time-filtered 09:15-15:29 IST) | ✅ 23 pass |
| `features/live_computer.py` | 18-feature rolling buffers matching batch pipeline | ✅ |
| `features/state_machine.py` | Double ROC(3) per-symbol state machine | ✅ |
| `features/ml_engine.py` | Orchestrator: LightGBM + state machine + pgwire + pmocata POST | ✅ |

Deployment: Copy to `Clean_Scanner/features/`. Tests: `python -m pytest tests/test_live_scanner.py -v`
