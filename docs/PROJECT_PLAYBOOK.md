# Project Playbook: Order Flow & Pre-Move Detection System

## 1. Project Overview & Objective
The Order Flow & Pre-Move Detection System is designed to identify NSE stocks that are highly likely to exhibit significant directional price movement (e.g., ±5% intraday) using ultra-high-frequency Limit Order Book (DOM) data and reconstructed trade flows.
The core objective is to detect institutional pre-move signals using microstructure signatures such as trade imbalances, order flow imbalances, and hidden liquidity detection (icebergs).

## 2. Target Users & Environment
- **Target**: Intraday high-frequency and mid-frequency trading execution algorithms.
- **Assets**: NSE Equities (~510 symbols, 38 trading days May 26 – Jul 21, 2026).
- **Environment**: Offline research and backtesting, progressing toward live simulation (live ML scanner on Railway, VIRTUAL mode).

## 3. Data Inputs
- **DOM (Limit Order Book)**: Raw up to Jun 12 has 20 levels; feature engine uses top 5 only. Jun 15+ unified feed is native depth 5. `imbalance_top10/top20` removed from code, registry, and model.
- **Tick Data**: Reconstructed from DOM level changes via Trade Inference Engine for Jun 3–12 (raw tick parquets corrupted). Jun 15+ unified tick parquets contain trades + DOM depth 5 in one table (split via `scripts/06_convert_unified_ticks.py`), then same pipeline path with depth-agnostic inference.

## 4. Subsystems
- **Ingestion & Splitter (Stage 0)**: Day-level parquet files are streamed and split into per-symbol parquets.
- **Data Cleaner (Stage 1)**: Drops garbage columns, localizes timezones to UTC, and sanitizes spreads.
- **Trade Inference Engine (Stage 2)**: Reconstructs aggressive buys and sells from order book changes.
- **Feature Factory (Stage 3)**: Aggregates data into 1-minute bars and computes 18 key features across Trade Flow, Market Depth, Price-Derived, and Microstructure groups.
- **Label Generator (Stage 4)**: Generates target labels (LONG/SHORT/NO_TRADE) using triple-barrier/MFE horizons.
- **Validation & Model Pipeline (Stage 5+)**: Performs stock-level cross-validation and walk-forward modeling.

## 5. Current Project Status
- **Pipeline Rebuilt**: Data contamination bug resolved. All splitters, cleaning, inference, feature, and label steps are functional.
- **Status (Jul 21, 2026)**: Final strategy L30_S25 (LONG 0.30 / SHORT 0.25) with Double ROC(3) + Daily Open filter deployed. Model retrained on 38 dates (5.93M OOF rows). 308 trades (177 LONG + 131 SHORT), 56.49% win rate. Live scanner at 100% batch parity. See `docs/CURRENT_STATE.md` and `docs/HANDOVER_STRATEGY_REPORT.md` for active spec.
