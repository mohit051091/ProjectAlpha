"""
PHASE 1: DATA EXPLORATION SCRIPT
Purpose: Understand the structure, quality, and characteristics of tick and DOM data
before designing the feature engineering pipeline.

This script will:
1. Load tick and DOM datasets
2. Inspect schemas and columns
3. Examine timestamp precision and ordering
4. Check for data quality issues
5. Generate summary statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_PATH = Path("d:/ML_June_2026/Data")

print("\n" + "="*80)
print("PHASE 1: DATA EXPLORATION & QUALITY ASSESSMENT")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATASETS
# ============================================================================
print("\n[1] LOADING DATASETS...")
print("-" * 80)

# Load all parquet files
tick_files = sorted(DATA_PATH.glob("tick_*.parquet"))
dom_files = sorted(DATA_PATH.glob("dom_*.parquet"))

print(f"Found {len(tick_files)} tick files: {[f.name for f in tick_files]}")
print(f"Found {len(dom_files)} DOM files: {[f.name for f in dom_files]}")

# Load a sample of each type
tick_df = pd.read_parquet(tick_files[0])
dom_df = pd.read_parquet(dom_files[0])

print(f"\n✓ Loaded: {tick_files[0].name}")
print(f"  Shape: {tick_df.shape}")
print(f"✓ Loaded: {dom_files[0].name}")
print(f"  Shape: {dom_df.shape}")

# ============================================================================
# SECTION 2: TICK DATA SCHEMA
# ============================================================================
print("\n[2] TICK DATA SCHEMA")
print("-" * 80)
print("\nColumn Names & Types:")
print(tick_df.dtypes)
print("\nFirst 5 rows:")
print(tick_df.head())
print("\nDataset Info:")
print(tick_df.info())

# ============================================================================
# SECTION 3: DOM DATA SCHEMA
# ============================================================================
print("\n[3] DOM DATA SCHEMA")
print("-" * 80)
print("\nColumn Names & Types:")
print(dom_df.dtypes)
print("\nFirst 5 rows:")
print(dom_df.head())
print("\nDataset Info:")
print(dom_df.info())

# ============================================================================
# SECTION 4: TIMESTAMP ANALYSIS
# ============================================================================
print("\n[4] TIMESTAMP ANALYSIS")
print("-" * 80)

# Analyze tick timestamps
if 'time' in tick_df.columns:
    tick_time_col = 'time'
elif 'timestamp' in tick_df.columns:
    tick_time_col = 'timestamp'
else:
    tick_time_col = [col for col in tick_df.columns if 'time' in col.lower()][0]

print(f"\nTick timestamp column: '{tick_time_col}'")
print(f"Data type: {tick_df[tick_time_col].dtype}")
print(f"Min: {tick_df[tick_time_col].min()}")
print(f"Max: {tick_df[tick_time_col].max()}")

# Check timestamp precision
sample_times = tick_df[tick_time_col].head(10).astype(str).tolist()
print(f"\nSample timestamps (first 10 rows):")
for i, t in enumerate(sample_times):
    print(f"  {i+1}. {t}")

# Check for duplicate timestamps
dup_timestamps = tick_df[tick_time_col].duplicated().sum()
print(f"\nDuplicate timestamps: {dup_timestamps}")

# Check ordering
is_sorted = tick_df[tick_time_col].is_monotonic_increasing
print(f"Timestamps in order: {is_sorted}")

if not is_sorted:
    out_of_order_idx = np.where(tick_df[tick_time_col].diff().dt.total_seconds() < 0)[0]
    print(f"  Out-of-order records: {len(out_of_order_idx)}")

# Analyze DOM timestamps
if 'time' in dom_df.columns:
    dom_time_col = 'time'
elif 'timestamp' in dom_df.columns:
    dom_time_col = 'timestamp'
else:
    dom_time_col = [col for col in dom_df.columns if 'time' in col.lower()][0]

print(f"\nDOM timestamp column: '{dom_time_col}'")
print(f"Data type: {dom_df[dom_time_col].dtype}")
print(f"Min: {dom_df[dom_time_col].min()}")
print(f"Max: {dom_df[dom_time_col].max()}")

# ============================================================================
# SECTION 5: MISSING VALUES & DATA QUALITY
# ============================================================================
print("\n[5] DATA QUALITY CHECK")
print("-" * 80)

print("\nTick Data - Missing Values:")
missing_tick = tick_df.isnull().sum()
if missing_tick.sum() > 0:
    print(missing_tick[missing_tick > 0])
else:
    print("  No missing values ✓")

print("\nDOM Data - Missing Values:")
missing_dom = dom_df.isnull().sum()
if missing_dom.sum() > 0:
    print(missing_dom[missing_dom > 0])
else:
    print("  No missing values ✓")

# ============================================================================
# SECTION 6: TICK DATA STATISTICS
# ============================================================================
print("\n[6] TICK DATA STATISTICS")
print("-" * 80)
print(tick_df.describe())

print("\nUnique symbols in tick data:")
if 'symbol' in tick_df.columns:
    print(f"  {tick_df['symbol'].unique()}")

# ============================================================================
# SECTION 7: DOM DATA STATISTICS
# ============================================================================
print("\n[7] DOM DATA STATISTICS")
print("-" * 80)
print(dom_df.describe())

print("\nUnique symbols in DOM data:")
if 'symbol' in dom_df.columns:
    print(f"  {dom_df['symbol'].unique()}")

# ============================================================================
# SECTION 8: DATA ALIGNMENT CHECK
# ============================================================================
print("\n[8] TICK-DOM ALIGNMENT")
print("-" * 80)

tick_count = len(tick_df)
dom_count = len(dom_df)
print(f"\nTick records: {tick_count}")
print(f"DOM records: {dom_count}")
print(f"Ratio (Tick/DOM): {tick_count/dom_count:.2f}")

# ============================================================================
# SECTION 9: MEMORY FOOTPRINT
# ============================================================================
print("\n[9] MEMORY FOOTPRINT")
print("-" * 80)
tick_memory = tick_df.memory_usage(deep=True).sum() / 1024**2
dom_memory = dom_df.memory_usage(deep=True).sum() / 1024**2
print(f"\nTick data: {tick_memory:.2f} MB")
print(f"DOM data: {dom_memory:.2f} MB")
print(f"Total (sample): {tick_memory + dom_memory:.2f} MB")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
