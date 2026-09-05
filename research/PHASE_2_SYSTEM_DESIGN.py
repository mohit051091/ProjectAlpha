"""
================================================================================
PHASE 2: SYSTEM DESIGN & ARCHITECTURE
Order Flow & Pre-Move Detection System

Status: IN PROGRESS
Goal: Design complete system architecture before writing feature code
================================================================================

This document outlines the complete system design:
1. Folder structure
2. Data schemas
3. Pipeline architecture
4. Module organization
5. Data flow
6. Configuration management
7. Validation strategy
"""

# PROJECT FOLDER STRUCTURE
# ================================================================================

project_structure = """
d:/ML_June_2026/
│
├─ Data/                                    # Raw data directory
│  ├─ dom_ifci_3_Jun_26.parquet
│  ├─ dom_wockpharma_1_Jun_26.parquet
│  ├─ tick_ifci_1_Jun_26.parquet
│  └─ tick_wockpharma_1_Jun_26.parquet
│
├─ 01_raw/                                  # Raw data processing
│  ├─ __init__.py
│  ├─ data_loader.py                       # Load parquet files
│  ├─ data_validator.py                    # Validate schema, values
│  └─ data_cleaner.py                      # Remove garbage columns
│
├─ 02_processed/                            # Processed data storage
│  ├─ __init__.py
│  ├─ data_schemas.py                      # Standardized schemas
│  └─ (processed parquets go here)
│
├─ 03_features/                             # Feature engineering
│  ├─ __init__.py
│  ├─ feature_factory.py                   # Main feature engine
│  ├─ tick_features.py                     # Trade flow features
│  ├─ dom_features.py                      # Order book features
│  ├─ price_features.py                    # Price-derived features
│  ├─ trade_inference.py                   # Infer trades from DOM
│  └─ feature_registry.py                  # Track all features
│
├─ 04_labels/                               # Label generation
│  ├─ __init__.py
│  ├─ label_generator.py                   # Compute LONG/SHORT/NO_TRADE
│  └─ label_validator.py                   # Check class distribution
│
├─ 05_research/                             # Statistical analysis
│  ├─ __init__.py
│  ├─ feature_analysis.py                  # Distribution, IV, WOE
│  ├─ correlation_analysis.py              # Correlation matrix
│  ├─ predictive_power.py                  # Feature importance
│  └─ backtester_simple.py                 # Rule-based backtesting
│
├─ 06_models/                               # ML models
│  ├─ __init__.py
│  ├─ model_trainer.py                     # Train LightGBM
│  ├─ model_evaluator.py                   # Compute metrics
│  ├─ cross_validator.py                   # Walk-forward validation
│  └─ (saved models go here)
│
├─ 07_utils/                                # Utilities & helpers
│  ├─ __init__.py
│  ├─ config.py                            # Configuration parameters
│  ├─ logger.py                            # Logging setup
│  ├─ performance.py                       # Timing & profiling
│  └─ constants.py                         # Enums, constants
│
├─ 08_tests/                                # Unit tests
│  ├─ __init__.py
│  ├─ test_data_loader.py
│  ├─ test_feature_factory.py
│  └─ test_label_generator.py
│
├─ notebooks/                               # Jupyter notebooks
│  ├─ 01_eda.ipynb                         # Exploratory analysis
│  ├─ 02_feature_exploration.ipynb         # Feature validation
│  └─ 03_results_summary.ipynb             # Final results
│
├─ docs/                                    # Documentation
│  ├─ PHASE_1_SUMMARY.md
│  ├─ PHASE_2_DESIGN.md                    # (This file)
│  ├─ architecture.md
│  └─ api_reference.md
│
├─ config/                                  # Configuration files
│  ├─ default.yaml                         # Default parameters
│  ├─ development.yaml                     # Dev settings
│  └─ production.yaml                      # Prod settings
│
├─ scripts/                                 # Runnable scripts
│  ├─ 10_prepare_data.py                   # Clean raw data
│  ├─ 20_infer_trades.py                   # Generate trade records
│  ├─ 30_compute_features.py                # Calculate all features
│  ├─ 40_generate_labels.py                # Assign LONG/SHORT/NO_TRADE
│  ├─ 50_analyze_features.py               # Statistical research
│  └─ 60_train_model.py                    # ML training
│
└─ README.md                                # Project overview
"""

# DATA SCHEMAS
# ================================================================================

SCHEMA_RAW_DOM = """
Raw DOM Snapshot Schema
────────────────────────────────────────────────────────────────────

Column                Type        Non-Null    Description
────────────────────────────────────────────────────────────────────
ts                    datetime    100%        Timestamp (UTC, nanosecond)
symbol                string      100%        Stock symbol (e.g., "IFCI")
ltp                   float64     100%        Last Traded Price

total_bid_qty         int64       100%        Sum of all bid quantities
total_ask_qty         int64       100%        Sum of all ask quantities
imbalance             float64     100%        Pre-calculated imbalance

bid1-bid20            float64     100%        Bid prices (20 levels)
bqty1-bqty20          int64       100%        Bid quantities (20 levels)
ask1-ask20            float64     100%        Ask prices (20 levels)
aqty1-aqty20          int64       100%        Ask quantities (20 levels)

────────────────────────────────────────────────────────────────────
Total Columns:        86 valid + 8 garbage (to drop)
Row Example:
  ts=2026-06-03 10:00:00.000000123,
  symbol=IFCI,
  ltp=76.00,
  bid1=75.99, bqty1=5000,
  ask1=76.05, aqty1=3000,
  ...
────────────────────────────────────────────────────────────────────
"""

SCHEMA_CLEANED_DOM = """
Cleaned DOM Snapshot Schema (After removing garbage columns)
────────────────────────────────────────────────────────────────────

Column                Type        Description
────────────────────────────────────────────────────────────────────
ts                    datetime    Timestamp (UTC, nanosecond)
symbol                string      Stock symbol
ltp                   float64     Last Traded Price
total_bid_qty         int64       Sum of all bids
total_ask_qty         int64       Sum of all asks
imbalance             float64     Bid-ask imbalance

bid1-bid20            float64     Bid prices (20 levels)
bqty1-bqty20          int64       Bid quantities (20 levels)
ask1-ask20            float64     Ask prices (20 levels)
aqty1-aqty20          int64       Ask quantities (20 levels)

────────────────────────────────────────────────────────────────────
Total Columns:        86
File Format:          Parquet (compressed, efficient)
Index:                ts, symbol (multiindex for fast lookup)
────────────────────────────────────────────────────────────────────
"""

SCHEMA_INFERRED_TRADES = """
Inferred Trade Records (Derived from DOM level changes)
────────────────────────────────────────────────────────────────────

Column                Type        Description
────────────────────────────────────────────────────────────────────
ts                    datetime    Trade timestamp
symbol                string      Stock symbol
trade_price           float64     Price of trade (level crossed)
trade_qty             int64       Quantity traded (estimated)
direction             string      BUY or SELL
side                  string      AGGRESSIVE (init) or PASSIVE (respond)
level_before          int64       DOM level before crossing (1-20)
level_after           int64       DOM level after crossing (1-20)

────────────────────────────────────────────────────────────────────
Derivation Method:
  • Detect when bid/ask quantity drops unexpectedly
  • Calculate qty_consumed = expected - actual
  • Direction = side of level crossed (bid=sell, ask=buy)
  • Accuracy: ~80-90% for microstructure analysis
────────────────────────────────────────────────────────────────────
"""

SCHEMA_ONE_MINUTE_FEATURES = """
One-Minute Feature Matrix (Aggregated to 1-minute bars)
────────────────────────────────────────────────────────────────────

Column                Type        Description
────────────────────────────────────────────────────────────────────
ts                    datetime    Period start (minute boundary)
symbol                string      Stock symbol

# GROUP A: Trade Flow (6 features)
delta_1m              float64     Buy vol - Sell vol (last 1 min)
delta_5m              float64     Buy vol - Sell vol (last 5 min)
volume_burst          float64     Current vol / avg vol (rolling 20)
aggressor_ratio       float64     Aggressive buys / total trades
trade_count_burst     float64     Trade count / avg (rolling 20)
large_trade_ratio     float64     Trades > 10x median / total

# GROUP B: Market Depth (6 features)
imbalance_top5        float64     (bid-ask)/(bid+ask) top 5 levels
imbalance_top10       float64     (bid-ask)/(bid+ask) top 10 levels
imbalance_top20       float64     (bid-ask)/(bid+ask) top 20 levels
spread                float64     ask1 - bid1 (best bid-ask gap)
depth_drop_bid        float64     Change in total bid qty
depth_drop_ask        float64     Change in total ask qty

# GROUP C: Price Derived (3 features)
vwap_distance         float64     (price - VWAP) / VWAP
volatility_5m         float64     Std dev of prices (5 min window)
price_acceleration    float64     2nd derivative of price (momentum change)

# GROUP D: Microstructure (3 features)
iceberg_score         float64     Executed / Displayed at level
order_cancel_rate     float64     Cancelled / Placed (spoofing signal)
bid_replenishment_rate float64    How fast bids refill (absorption signal)

# LABEL
label                 string      LONG / SHORT / NO_TRADE (look-ahead 2h)

────────────────────────────────────────────────────────────────────
Total Columns:        22 (18 features + ts + symbol + label)
Frequency:            1-minute bars
Index:                ts, symbol (multiindex)
File Format:          Parquet
────────────────────────────────────────────────────────────────────
"""

SCHEMA_LABELED_FEATURES = """
Final Labeled Feature Dataset
────────────────────────────────────────────────────────────────────

Same as One-Minute Features (above), plus:

Column                Type        Description
────────────────────────────────────────────────────────────────────
label                 string      LONG (move>+5%), SHORT (move<-5%), 
                                  NO_TRADE (move between)

label_return          float64     Actual return in next 2 hours
label_confidence      float64     How extreme was the move (0-1)

────────────────────────────────────────────────────────────────────
Class Distribution (typical):
  • LONG:             ~15-20% of rows
  • SHORT:            ~15-20% of rows
  • NO_TRADE:         ~60-70% of rows
  
(Imbalanced class problem - will address with rebalancing)

Usage:
  • Training ML models: Use first 80% of time
  • Validation: Use middle 10% of time  
  • Testing: Use final 10% of time
  • Cross-validation: Walk-forward (time-based, not random)
────────────────────────────────────────────────────────────────────
"""

# DATA PIPELINE ARCHITECTURE
# ================================================================================

PIPELINE_ARCHITECTURE = """
Complete Data Processing Pipeline
════════════════════════════════════════════════════════════════════

┌─ STAGE 1: DATA INGESTION & CLEANING ──────────────────────────────┐
│                                                                     │
│  Input:  Raw parquet files (4 files, 159K+ DOM snapshots)         │
│  |                                                                 │
│  ├─ Load:       data_loader.load_parquet()                        │
│  │              └─ Returns: pd.DataFrame with all columns         │
│  │                                                                 │
│  ├─ Validate:   data_validator.validate_schema()                 │
│  │              └─ Check: dtypes, non-nulls, ranges              │
│  │                                                                 │
│  ├─ Clean:      data_cleaner.drop_garbage_cols()                 │
│  │              └─ Remove: 8 columns with no data                │
│  │                                                                 │
│  └─ Save:       parquet file → 02_processed/                     │
│                                                                     │
│  Output: Cleaned DOM data (86 columns, 159K rows)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       ↓

┌─ STAGE 2: TRADE INFERENCE ────────────────────────────────────────┐
│                                                                     │
│  Input:  Cleaned DOM snapshots                                    │
│  |                                                                 │
│  ├─ Detect:     trade_inference.detect_level_crossings()         │
│  │              └─ Find: qty changes at bid/ask levels           │
│  │                                                                 │
│  ├─ Estimate:   trade_inference.estimate_trade_direction()       │
│  │              └─ Determine: BUY or SELL                        │
│  │                                                                 │
│  ├─ Quantify:   trade_inference.estimate_trade_qty()             │
│  │              └─ Calculate: qty consumed                       │
│  │                                                                 │
│  └─ Save:       parquet file → 02_processed/                     │
│                                                                     │
│  Output: Inferred trade records (1-3M trades per day)             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       ↓

┌─ STAGE 3: FEATURE ENGINEERING ────────────────────────────────────┐
│                                                                     │
│  Input:  Cleaned DOM + Inferred trades                            │
│  |                                                                 │
│  ├─ Synchronize:  feature_factory.align_tick_dom()               │
│  │                └─ Merge on timestamp                          │
│  │                                                                 │
│  ├─ Aggregate:    feature_factory.aggregate_to_1min()            │
│  │                └─ Create 1-minute buckets                     │
│  │                                                                 │
│  ├─ Compute:      feature_factory.compute_all_features()         │
│  │                └─ Calculate: 18 features per group            │
│  │                                                                 │
│  │  Group A (Trade Flow):                                        │
│  │    ├─ delta_1m, delta_5m (buy-sell volume difference)        │
│  │    ├─ volume_burst (volume spike detection)                  │
│  │    ├─ aggressor_ratio (aggressive trade %)                   │
│  │    ├─ trade_count_burst (trade count spike)                  │
│  │    └─ large_trade_ratio (large trade %)                      │
│  │                                                                 │
│  │  Group B (Market Depth):                                      │
│  │    ├─ imbalance_top5, top10, top20 (bid-ask imbalance)       │
│  │    ├─ spread (best bid-ask gap)                              │
│  │    ├─ depth_drop_bid, depth_drop_ask (liquidity changes)     │
│  │    └─ (all derived from order book snapshot data)            │
│  │                                                                 │
│  │  Group C (Price Derived):                                     │
│  │    ├─ vwap_distance (VWAP deviation)                         │
│  │    ├─ volatility_5m (price volatility)                       │
│  │    └─ price_acceleration (momentum change)                   │
│  │                                                                 │
│  │  Group D (Microstructure):                                    │
│  │    ├─ iceberg_score (hidden order detection)                 │
│  │    ├─ order_cancel_rate (spoofing detection)                 │
│  │    └─ bid_replenishment_rate (absorption detection)          │
│  │                                                                 │
│  ├─ Validate:     feature_factory.validate_features()            │
│  │                └─ Check: NaN, outliers, ranges               │
│  │                                                                 │
│  └─ Save:         parquet file → 02_processed/                   │
│                                                                     │
│  Output: Feature matrix (18 features × N minutes × M stocks)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       ↓

┌─ STAGE 4: LABEL GENERATION ───────────────────────────────────────┐
│                                                                     │
│  Input:  Feature matrix + Original price data                     │
│  |                                                                 │
│  ├─ Look-ahead:   label_generator.compute_forward_returns()      │
│  │                └─ For each 1-min bar, look 2 hours forward    │
│  │                                                                 │
│  ├─ Calculate:    label_generator.calculate_max_return()         │
│  │                └─ Find: max price in next 2 hours             │
│  │                                                                 │
│  ├─ Assign:       label_generator.assign_labels(threshold=0.05)  │
│  │                ├─ LONG if return > +5%                        │
│  │                ├─ SHORT if return < -5%                       │
│  │                └─ NO_TRADE else                               │
│  │                                                                 │
│  ├─ Analyze:      label_validator.check_class_distribution()     │
│  │                └─ Report: % LONG, SHORT, NO_TRADE             │
│  │                                                                 │
│  └─ Save:         parquet file → 02_processed/                   │
│                                                                     │
│  Output: Labeled feature dataset (features + labels)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       ↓

┌─ STAGE 5: STATISTICAL RESEARCH ───────────────────────────────────┐
│                                                                     │
│  Input:  Labeled features                                         │
│  |                                                                 │
│  ├─ Distributions:  feature_analysis.compute_distributions()      │
│  │                  └─ Mean, std, skew, kurtosis                 │
│  │                                                                 │
│  ├─ Correlation:    correlation_analysis.build_matrix()          │
│  │                  └─ Feature-to-feature correlations           │
│  │                                                                 │
│  ├─ Information:    predictive_power.compute_iv()                │
│  │                  └─ Information Value per feature              │
│  │                                                                 │
│  ├─ Backtest:       backtester_simple.run_rules()                │
│  │                  └─ Test: "if imbalance > 0.7, label LONG?"   │
│  │                                                                 │
│  └─ Report:         Generate visualizations & summaries           │
│                                                                     │
│  Output: Research findings (which features predict moves)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
       ↓

┌─ STAGE 6: MODEL TRAINING & VALIDATION ────────────────────────────┐
│                                                                     │
│  Input:  Labeled features                                         │
│  |                                                                 │
│  ├─ Split:         cross_validator.create_walk_forward_splits()  │
│  │                 └─ Time-based splits (no data leakage)        │
│  │                                                                 │
│  ├─ Train:         model_trainer.train_lightgbm()                │
│  │                 └─ LGBMClassifier(n_estimators=500)           │
│  │                                                                 │
│  ├─ Evaluate:      model_evaluator.compute_all_metrics()         │
│  │                 ├─ Classification: accuracy, precision, recall │
│  │                 ├─ Ranking: AUC, Gini, Lift                   │
│  │                 └─ Trading: Sharpe, drawdown, win rate        │
│  │                                                                 │
│  ├─ Interpret:     model_evaluator.feature_importances()         │
│  │                 └─ Which features matter most?                │
│  │                                                                 │
│  └─ Save:          model file → 06_models/                       │
│                                                                     │
│  Output: Trained model + performance metrics                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

End-to-End: Raw Data → Predictions (Fully automated pipeline)
"""

# CONFIGURATION MANAGEMENT
# ================================================================================

CONFIG_EXAMPLE = """
Configuration System (config/default.yaml)
────────────────────────────────────────────────────────────────────

# Data Processing
data:
  input_path: "./Data"
  output_path: "./02_processed"
  parquet_engine: "pyarrow"
  compression: "snappy"

# Feature Engineering
features:
  aggregation_window: "1min"          # 1-minute bars
  lookback_windows: [1, 5, 20]        # 1-min, 5-min, 20-min
  dom_levels: 20                      # Use all 20 levels
  
# Label Generation
labels:
  forward_window: "2h"                # Look 2 hours ahead
  move_threshold: 0.05                # 5% move threshold
  long_threshold: 0.05                # > 5% = LONG
  short_threshold: -0.05              # < -5% = SHORT

# Model Training
model:
  type: "lightgbm"
  params:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 7
    num_leaves: 31
    min_child_samples: 20
    
# Validation
validation:
  method: "walk_forward"              # Time-based splits
  train_size: 0.7
  val_size: 0.1
  test_size: 0.2
  
# Performance
performance:
  workers: 4                          # Parallel processing
  batch_size: 10000                   # Process in chunks
  verbose: True
────────────────────────────────────────────────────────────────────
"""

# DATA QUALITY GATES
# ================================================================================

QUALITY_GATES = """
Data Quality Validation Gates
════════════════════════════════════════════════════════════════════

STAGE 1: Raw Data Ingestion
────────────────────────────────────────────────────────────────────
✓ File exists and readable
✓ Schema matches expected: 94 columns
✓ Row count > 0
✓ ts column is datetime
✓ symbol column is string
✓ Price columns (ltp, bid*, ask*) are numeric
✓ Quantity columns (bqty*, aqty*) are integers

STAGE 2: Data Cleaning
────────────────────────────────────────────────────────────────────
✓ Garbage columns identified and removed
✓ Remaining columns: exactly 86
✓ No rows dropped (quality is good)
✓ dtypes are correct
✓ Non-null count: 100% for all columns

STAGE 3: Trade Inference
────────────────────────────────────────────────────────────────────
✓ Level-crossing events detected
✓ Trade direction assigned (BUY/SELL)
✓ Trade quantity > 0
✓ Trade price within [bid1, ask1]
✓ Timestamp aligned with DOM data
✓ Max trades per minute: < 1000 (sanity check)

STAGE 4: Feature Computation
────────────────────────────────────────────────────────────────────
✓ All 18 features computed
✓ No NaN values in features
✓ Feature values within reasonable ranges:
  • delta_1m, delta_5m: [-1M, +1M]
  • volume_burst: [0.1, 100]
  • imbalance_*: [-1, +1]
  • spread: > 0
  • volatility: > 0
  • etc.
✓ Features are reproducible (deterministic)

STAGE 5: Label Generation
────────────────────────────────────────────────────────────────────
✓ Forward return calculated for all rows
✓ Labels assigned: LONG, SHORT, or NO_TRADE
✓ Class distribution checked:
  • No class represents >95% of data (okay if 60-70% NO_TRADE)
  • Minimum 100 examples per class
  • Imbalance handled (rebalancing if needed)

STAGE 6: Statistical Research
────────────────────────────────────────────────────────────────────
✓ Feature distributions analyzed
✓ Correlation matrix computed
✓ No multi-collinearity issues (corr < 0.95)
✓ Information Value computed per feature
✓ At least 1-2 features show promising predictive power

STAGE 7: Model Training
────────────────────────────────────────────────────────────────────
✓ Train/val/test splits are time-based (no leakage)
✓ Model trains without errors
✓ Validation metrics computed
✓ No overfitting detected (train acc ≈ test acc)
✓ Performance better than baseline (50% for balanced)

If any gate fails: STOP, investigate, fix, and re-run.
"""

print("\n" + "="*80)
print("PHASE 2: SYSTEM DESIGN - COMPLETE BLUEPRINT")
print("="*80)
print("\nSee this file for:")
print("  • Project folder structure")
print("  • Data schemas (input/output formats)")
print("  • Complete pipeline architecture")
print("  • Configuration management")
print("  • Data quality validation gates")
print("\nAll modules will follow this design exactly.")
print("="*80 + "\n")
