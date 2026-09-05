# Task Registry

## Status Categories
- **Backlog**: Long-term tasks, not yet scheduled.
- **Planned**: Scheduled for the current/next phase, not started.
- **In Progress**: Active development or execution.
- **Testing**: Verification and audit.
- **Ready**: Validated and ready for deployment.
- **Deployed**: Active in production/live simulation.
- **Cancelled**: Abandoned or postponed indefinitely.

---

## Active Checklist (Phased Tasks)

### PHASE 0: CRITICAL BUG FIXES
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 0.1** | Fix Logistic Regression penalty bug (`scripts/50_baseline_model.py`) | Ready | Agent | Resolved: SAGA iteration tuned and regularized |
| **Task 0.2** | Fix backtest lookahead bias (`scripts/50_backtest_signals.py`) | Ready | Agent | Resolved: Lookahead-free Alert-Trigger backtest implemented in scripts/60_alert_trigger_backtest.py |
| **Task 0.3** | Fix `volatility_5m` computation (`features/tick_features.py`) | Ready | Agent | Resolved: Compute on LTP instead of VWAP |
| **Task 0.4** | Fix `vwap_distance` near-zero division (`features/tick_features.py`) | Ready | Agent | Resolved: Gate on minimum volume |
| **Task 0.5** | Fix negative spread / crossed book (`raw/data_cleaner.py`) | Ready | Agent | Resolved: Drop crossed book rows |
| **Task 0.6** | Fix DOM resampling for imbalance features (`features/feature_factory.py`) | Ready | Agent | Resolved: Use `.last()` for depth quantities and `.mean()` for ltp |
| **Task 0.7** | Fix trade inference fallback quantity (`features/trade_inference.py`) | Ready | Agent | Resolved: Use rolling median trade size |
| **Task 0.8** | Fix cross-symbol state leakage (pipeline safety) | Ready | Agent | Resolved: Stateless feature workers |
| **Task 0.9** | Fix order book clock synchronization | Ready | Agent | Resolved: Configure ±50ms matching window |
| **Checkpoint 0**| Re-run pipeline, compare MCC before/after | Ready | Agent/Human | Resolved: Clean data baseline established, OOF MCC LR=0.041, RF=0.045 |

### PHASE 1: DIAGNOSTIC TESTS
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 1.1** | Stock-identity test | Planned | Agent | MCC < 0.15 pass |
| **Task 1.2** | Z-score neutralization | Completed | Agent | Evaluated and resolved via Turnover Value-Delta filter |
| **Task 1.3** | Within-stock temporal CV | Planned | Agent | MCC > 0.05 pass |
| **Checkpoint 1**| Review diagnostic results & HALT gates | Planned | Agent/Human | HALT if any test fails |

### PHASE 2: TRADE INFERENCE VALIDATION
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 2.1** | Bar-level trade inference accuracy audit | Planned | Agent | Pearson r > 0.85, 0.70 |
| **Checkpoint 2**| Report inference accuracy to human | Planned | Agent/Human | Confirm trade inference quality |

### PHASE 3: NEW FEATURES & LABEL REDESIGN
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 3.1** | Implement P1 features (Micro-price, CumDelta, BookPress) | Planned | Agent | Add to factory |
| **Task 3.2** | Implement P2 features (TimeOpen, DistHighLow, ArrivalRate, DepthSlope) | Planned | Agent | Add to factory |
| **Task 3.3** | Rebuild `order_cancel_rate` | Planned | Agent | DOM-diff cancellation proxy |
| **Task 3.4** | Implement MFE Triple-Barrier Labels & Return Targets | Completed | Agent | Completed: 5% EOD first-touch excursion label implemented |
| **Task 3.5** | Feature normalization | Completed | Agent | Z-score cross-sectional normalization implemented and validated in backtests |
| **Checkpoint 3**| Review new features and MFE label distribution | Planned | Agent/Human | Verify target metrics |

### PHASE 4: RE-RUN BASELINE WITH FIXES
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 4.1** | Full pipeline re-run (Stages 1–5) | Planned | Agent | Re-run corrected baseline |
| **Task 4.2** | Add IC, calibration, P&L, signal decay, and capacity metrics | Planned | Agent | Full metric stack |
| **Task 4.3** | Report comprehensive results | Planned | Agent | MCC, IC, Sharpe, calibration |
| **Checkpoint 4**| Review results and decide on data acquisition | Planned | Agent/Human | Check if alpha is viable |

### PHASE 5: DATA ACQUISITION & MULTI-DAY VALIDATION
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 5.1** | Acquire additional historical data | Completed | Human | 10+ days (target 60+ days) |
| **Task 5.2** | Ingest new data through pipeline | Completed | Agent | Run Stages 0–4 |
| **Task 5.3** | Implement walk-forward validation (purging/embargo) | Completed | Agent | Purged/embargoed expanding window split engine |
| **Task 5.4** | Regime analysis (high/low VIX, gap open) | Planned | Agent | Segment performance |
| **Checkpoint 5**| Review multi-day results | Completed | Agent/Human | Validated 12-day results; trend filter confirmed |

### PHASE 6.5: LIVE-BATCH ALIGNMENT
| Task | Description | Status | Notes |
|---|---|---|---|
| **Task 6.5.1** | Fix trade_qty→volume in bar_accumulator | Deployed | Fix #1: only count ticks with incremental volume > 0 |
| **Task 6.5.2** | Remove bid1=0 filter in bar_accumulator | Deployed | Fix #2: include all ticks in DOM averages |
| **Task 6.5.3** | Day boundary clear in ml_engine | Deployed | Fix #3: clear accumulators+computers at day boundary |
| **Task 6.5.4** | Fix volatility_5m RingBuffer(5) in live_computer | Deployed | Fix #4: match batch rolling(5).std() |
| **Task 6.5.5** | Fix iceberg_score=0 when displayed=0 in live_computer | Deployed | Fix #5: match batch behavior |
| **Task 6.5.6** | Embed pmocata into collector_mohit | Deployed | Eliminate rate-limit race; standalone service deleted |
| **Task 6.5.7** | Fix Railway domain targetPort (502) | Deployed | Updated 5001→8080, /status returns 200 |
| **Task 6.5.8** | Live vs batch comparison testing | Deployed | Replay ticks + compare trade inference against batch. **100.0% perfect feature parity** (0.000000 max error) achieved across all 18 features. |
| **Task 6.5.9** | Fix MLEngine init signature mismatch | Deployed | Updated constructor signature to take symbol_segments and sid_to_symbol to resolve startup TypeError. |
| **Task 6.5.10** | Fix WebSocket reconnect monitor & HTTP 429 backoff | Deployed | Monitored the background feed_thread directly and added a 30s+ backoff retry delay to prevent lockout. |
| **Task 6.5.11** | Fix Daily Open Filter dynamic initialization | Deployed | Initialized daily open from tick packet's open field to survive collector restarts. |
| **Task 6.5.12** | Fix Bar Accumulator skipped ticks boundary bug | Deployed | Removed _pending_bar delay mechanism and returned flushed_bar directly from add_tick(), resolving skipped ticks. |
| **Task 6.5.13** | EQ Series Only Filtering | Deployed | Aligned live scanner and batch pipeline to trade standard EQ series cash equities only, ignoring BE Trade-to-Trade series |
| **Task 6.5.14** | Dynamic Online Scrip Master Update | Deployed | Re-wired pipeline to fetch online api-scrip-master.csv daily, filtering against Nifty 500 whitelist |
| **Task 6.5.15** | Live Scanner Self-Healing Watchdog | Deployed | Implemented symbol-level self-healing watchdog thread in collector.py to detect and re-subscribe silent symbols |
| **Task 6.5.16** | Dhan 9 AM Login Token Invalidation Fix | Deployed | Modified pmocata server.py to skip morning login if token was refreshed under 4 hours ago |
| **Task 6.5.17** | Discard Pre-Market Data in Backtest | Deployed | Discard ticks and DOM before 09:15:00 and after 15:30:00 IST in DuckDB cleaner, achieving 100% warm-up state parity |


### PHASE 6: MODEL UPGRADE
| Task | Description | Status | Assignee/Owner | Notes |
|---|---|:---:|:---:|---|
| **Task 6.1** | LightGBM implementation & SHAP values | Completed | Agent | Fast leaf-wise tree boosting upgrade + SHAP analysis |
| **Task 6.2** | LightGBM vs Random Forest comparison | Completed | Agent | LightGBM avoids RF overtrading, achieves 77% win rate with trend filter |
| **Task 6.3** | LightGBM threshold sweeping & spread segmentation | Completed | Agent | Swept [0.20, 0.45] combined and by low/high spread; optimal range 0.25 to 0.28 is stable |
| **Task 6.4** | Implement infinite EOD hold & MFE/MAE tracking | Completed | Agent | Excursion and EOD returns tracked in simulations |
| **Task 6.5** | ZEEL Diagnostic Mover Audit | Completed | Agent | Audited June 9, 10, 11 ZEEL triggers and excursions |
| **Task 6.6** | Train and export final model weights | Completed | Agent | Trained final model on 12 days and saved weights to models/ folder |
| **Task 6.7** | Sweeps with multiple triggers & continuous ROC | Completed | Agent | Completed: swept thresholds under lookahead-free non-blocking trade rules. |

---

## Completed
| Task | Completed Date | Notes |
|---|---|---|
| Model channel separation (`auto_research` vs `live_production`) | 2026-07-05 | Added explicit folder channels in `models/` with release manifest + live reference metadata to prevent confusion in production pushes. |
| Jul-03-inclusive DuckDB-first pipeline completion | 2026-07-05 | Converted Jul-03 unified tick parquet, then completed staged processing for Jun 29-30 and Jul 01-03 until all five dates reached `cleaned/inferred/features/labeled = 500/500`. |
| Jul-03-inclusive OOF rebuild + multi-label auto-research refresh | 2026-07-05 | Built `results/oof_source_full_2026_07_03.parquet` via DuckDB from labeled parquet; regenerated target sweep workbooks and refreshed best recommendation to `label_30m_1pct / L25_S30`. |
| Investigate LTP extreme ranges | 2026-06-08 | Found multi-symbol DOM file contamination |
| Validate contamination claim | 2026-06-08 | 100% contamination confirmed in sample |
| Trace root cause | 2026-06-08 | Group-by vs sort order bug in `05_split_day_level.py` |
| Create LTP_DATA_QUALITY_REPORT.md | 2026-06-08 | Initial documentation |
| Create CONTAMINATION_VALIDATION_REPORT.md | 2026-06-08 | Formal validation report |
| Create CONTAMINATION_ROOT_CAUSE.md | 2026-06-08 | Detailed root cause report |
| Update MASTER_REVIEW_AND_EXECUTION_PLAN.md | 2026-06-11 | Integrated GPT and Gemini reviews |
| Movers & AARTIIND Signal Audit | 2026-06-11 | Diagnostic check for AARTIIND and June 9-10 movers |
| Implement Order Flow Absorption Features & Threshold Gate Entry | 2026-06-11 | Added 4 passive absorption features, modified backtester to threshold gate trigger |
| Support both doms and dom_snapshots patterns in Stage 0 | 2026-06-11 | Splitter updated to handle both day-level naming conventions |
| Ingest new multi-day historical data (Stage 0 to 4) | 2026-06-12 | Timezone conflicts resolved, generated features and labels successfully |
| Train baseline model on multi-day expanded data (Stage 5) | 2026-06-12 | Trained Logistic Regression (L1) and Random Forest on 958k rows |
| Implement Walk-Forward Split Engine with Purging/Embargo | 2026-06-12 | Created 7-fold expanding window split validator with zero temporal overlap |
| Upgrade Model to LightGBM & SHAP Analysis | 2026-06-12 | Installed lightgbm/shap, compared RF vs LGBM; computed SHAP importance values |
| Build Daily Open & VWAP Trend Filters | 2026-06-12 | Trend-aligned filters boosted RF P&L to +30.77% and LightGBM to +7.27% (77.14% win rate) |
| LightGBM threshold sweeping & spread segmentation | 2026-06-12 | Swept thresholds [0.20, 0.45] combined and segmented by low/high spread |
| Multi-Label and Exception Sweeps under EOD Hold | 2026-06-13 | Swept both target labels under Standard and Exception-Enabled modes with excursion metrics |
| Train and export final model weights | 2026-06-13 | Trained 12-day final model and saved text/JSON weights for 60m_1pct |
| Z-Score Cross-Sectional Normalization & Sweeps | 2026-06-14 | Z-score normalization implemented; threshold 0.30 LONG achieves 44.55% selection accuracy, capturing 100% of liquid movers |
| DOM 5 Only & SHORT Trade Gates | 2026-06-14 | DOM 5 Only model retains 100% of movers with zero MCC loss; Z-score gates boost selectivity; SHORT gates at 0.25 capture JAINREC/OLAELEC with 25.90% win rate. |
| Clean movers.xlsx & Daily Open Exception Sweeps | 2026-06-14 | Standardized movers.xlsx; swept Daily Open exceptions; found optimal combined bypass parameters |
| Double-Direction Rallies & Hybrid Strategy Audit | 2026-06-14 | Audited double-direction rallies (2.5% success rate), compared deduplication types, and implemented Hybrid Strategy to cut 122 fake SHORT trades (e.g. AEGISLOG SHORT on June 11) |
| Depth 20 feature removal (code + model retrain) | 2026-06-18 | Removed imbalance_top10/top20 from all code, retrained model with 18 features |
| Unified feed auto-detect + converter | 2026-06-18 | FeatureFactory auto-detects unified feed; scripts/06_convert_unified_ticks.py created |
| Jun 15-16 pipeline processing | 2026-06-18 | Ran full pipeline (10→20→30→40) for Jun 15-16 via converter |
| Walk-forward OOF extension | 2026-06-18 | scripts/78_extend_oof.py — folds 8-9, 1,319,475 total OOF rows |
| v2 Double ROC(3) sweeps | 2026-06-18 | scripts/76_generate_v2_sweeps.py — 3 configs, results/v2_performance_sweeps.xlsx |
| Jun 5 exclusion | 2026-06-18 | Removed from all analysis due to mislabeled tick file |
| Trade inference depth-agnostic fix | 2026-06-18 | Added _detect_depth() to trade_inference.py |
| Broken junctions fix | 2026-06-18 | Re-pointed Data and 02_processed to correct D:\ProjectAlpha\ paths |
| All documentation updated | 2026-06-18 | HANDOVER_STRATEGY, CURRENT_STATE, DECISION_LOG, PROJECT_TIMELINE, TASKS |
| Non-EOD auto-research label-family curation | 2026-07-04 | Excluded EOD label family and enabled curated label-family execution/ranking (`30m_1pct`, `60m_1pct`, `60m_3pct`, `120m_3pct`) with target-aware scripts |
| Live Scanner Self-Healing Watchdog & 9 AM Login Fix | 2026-07-11 | Implemented individual symbol-level watchdog in collector.py and skipped morning login in server.py if token is fresh (under 4 hours old) to prevent HTTP 429 rate limit disconnects |
| EQ Series Only Filtering & Online Dhan Scrip Master Update | 2026-07-11 | Filtered out BE Trade-to-Trade series from both live scanner and batch pipeline, and implemented daily online master scrip downloads |
| Live-vs-backtest parity root-cause + fix bundle | 2026-09-06 | Found roc_3 2-bar vs 3-bar split + 8 more gaps on Aug 31–Sep 04; fixed in code, verified in sandbox (recall 17/34 → 25/34), committed + pushed |
| Railway redeploy (parity bundle) + QuestDB ml_signals ALTER | 2026-09-06 | Pending deploy: entry_time_ist STRING, theta_prob DOUBLE; live-side fixes take effect only after redeploy |

