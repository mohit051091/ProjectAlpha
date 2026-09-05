"""
PHASE 1: DATA EXPLORATION (REVISED)
Loading Parquet files using PyArrow engine
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_PATH = Path("d:/ML_June_2026/Data")

print("\n" + "="*80)
print("PHASE 1: DATA EXPLORATION (PARQUET LOADING)")
print("="*80)

# Try loading with PyArrow
tick_file = DATA_PATH / "tick_ifci_1_Jun_26.parquet"
dom_file = DATA_PATH / "dom_ifci_3_Jun_26.parquet"

print(f"\n[1] Loading {tick_file.name}...")
try:
    # Use PyArrow to read schema first
    parquet_file = pq.ParquetFile(str(tick_file))
    print(f"✓ Parquet file metadata loaded")
    print(f"  Number of row groups: {parquet_file.num_row_groups}")
    print(f"  Total rows: {parquet_file.metadata.num_rows}")
    print(f"\nSchema:")
    print(parquet_file.schema)
    
    # Load as pandas
    tick_df = pd.read_parquet(str(tick_file), engine='pyarrow')
    print(f"\n✓ Loaded as pandas DataFrame")
    print(f"  Shape: {tick_df.shape}")
    print(f"  Columns: {list(tick_df.columns)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n[2] Loading {dom_file.name}...")
try:
    parquet_file = pq.ParquetFile(str(dom_file))
    print(f"✓ Parquet file metadata loaded")
    print(f"  Number of row groups: {parquet_file.num_row_groups}")
    print(f"  Total rows: {parquet_file.metadata.num_rows}")
    print(f"\nSchema:")
    print(parquet_file.schema)
    
    dom_df = pd.read_parquet(str(dom_file), engine='pyarrow')
    print(f"\n✓ Loaded as pandas DataFrame")
    print(f"  Shape: {dom_df.shape}")
    print(f"  Columns: {list(dom_df.columns)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
