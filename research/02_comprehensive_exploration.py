"""
COMPREHENSIVE DATA EXPLORATION
Load all available files and create detailed documentation
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

DATA_PATH = Path("d:/ML_June_2026/Data")

print("\n" + "="*80)
print("PHASE 1: COMPREHENSIVE DATA EXPLORATION")
print("="*80)

# Load all files and track status
files_status = {}

for parquet_file in sorted(DATA_PATH.glob("*.parquet")):
    print(f"\n{'='*80}")
    print(f"FILE: {parquet_file.name}")
    print(f"{'='*80}")
    
    try:
        # Get file metadata
        pf = pq.ParquetFile(str(parquet_file))
        num_rows = pf.metadata.num_rows
        num_cols = pf.metadata.num_columns
        
        print(f"✓ Status: Valid Parquet file")
        print(f"  Rows: {num_rows:,}")
        print(f"  Columns: {num_cols}")
        print(f"  Row groups: {pf.num_row_groups}")
        print(f"  File size: {parquet_file.stat().st_size / 1024**2:.2f} MB")
        
        # Load into pandas
        df = pd.read_parquet(str(parquet_file))
        print(f"\n✓ Loaded into pandas")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Display columns
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            null_pct = (1 - non_null/len(df)) * 100
            print(f"  {i:2d}. {col:20s} | {dtype:20s} | Non-null: {non_null:,} ({100-null_pct:5.1f}%)")
        
        # Timestamp analysis
        ts_col = 'ts' if 'ts' in df.columns else None
        if ts_col:
            print(f"\nTIMESTAMP ANALYSIS (column: '{ts_col}'):")
            print(f"  Data type: {df[ts_col].dtype}")
            print(f"  Min: {df[ts_col].min()}")
            print(f"  Max: {df[ts_col].max()}")
            print(f"  Duration: {df[ts_col].max() - df[ts_col].min()}")
            
            # Check ordering
            is_sorted = df[ts_col].is_monotonic_increasing
            print(f"  Sorted: {is_sorted}")
            
            # Check resolution
            diffs = df[ts_col].diff().dropna()
            min_diff = diffs[diffs > pd.Timedelta(0)].min()
            print(f"  Min time difference (non-zero): {min_diff}")
            print(f"  Resolution: Nanoseconds (1e-9 seconds)")
        
        # Symbol analysis
        if 'symbol' in df.columns:
            print(f"\nSYMBOL ANALYSIS:")
            symbols = df['symbol'].unique()
            print(f"  Unique symbols: {len(symbols)}")
            print(f"  Symbols: {symbols}")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNUMERIC COLUMNS SUMMARY:")
            for col in numeric_cols[:5]:  # Show first 5
                print(f"\n  {col}:")
                print(f"    Min: {df[col].min()}")
                print(f"    Max: {df[col].max()}")
                print(f"    Mean: {df[col].mean():.2f}")
                print(f"    Std: {df[col].std():.2f}")
                print(f"    Nulls: {df[col].isna().sum()}")
        
        files_status[parquet_file.name] = "✓ OK"
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:200]}")
        files_status[parquet_file.name] = f"✗ {type(e).__name__}"

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for fname, status in files_status.items():
    print(f"{fname:40s} | {status}")
