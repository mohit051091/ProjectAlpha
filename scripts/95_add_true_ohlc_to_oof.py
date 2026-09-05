"""
Experimental Script 95: Add True 1-Minute OHLC to OOF Dataset (Filtered Valid DOM Parquets).
======================================================================================
Filters out 0-byte corrupted parquet files, then extracts true open_price, high_price,
low_price, close_price from valid cleaned_dom_*.parquet files for all symbol-dates in
results/oof_true_5fold.parquet via DuckDB parallel engine.
Saves enriched dataset to results/oof_true_5fold_ohlc.parquet.
"""
import os
import sys
import time
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "02_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_PATH = RESULTS_DIR / "oof_true_5fold.parquet"
OUTPUT_OHLC_OOF_PATH = RESULTS_DIR / "oof_true_5fold_ohlc.parquet"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 70)
    log("STEP 1: REBUILD TRUE 1-MINUTE OHLC FROM CLEANED DOM SNAPSHOTS (DUCKDB)")
    log("=" * 70)

    log(f"Loading true 5-fold OOF predictions from {OOF_PATH}...")
    df_oof = pd.read_parquet(OOF_PATH)
    log(f"Loaded {len(df_oof):,} rows.")

    log("Scanning for valid non-empty DOM parquet files...")
    all_dom_files = [str(PROC_DIR / f) for f in os.listdir(PROC_DIR) if f.startswith("cleaned_dom_") and f.endswith(".parquet")]
    valid_dom_files = [f for f in all_dom_files if os.path.getsize(f) > 100]
    invalid_count = len(all_dom_files) - len(valid_dom_files)
    log(f"Found {len(valid_dom_files):,} valid DOM files (filtered out {invalid_count} empty 0-byte files).")

    log("Extracting true OHLC from valid DOM parquet files using DuckDB...")
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET TIME ZONE 'UTC'")
    con.execute("SET threads TO 4")
    
    query = """
    SELECT 
        symbol,
        date_trunc('minute', ts) as ts,
        FIRST(ltp) as open_price,
        MAX(ltp) as high_price,
        MIN(ltp) as low_price,
        LAST(ltp) as close_price
    FROM read_parquet(?, union_by_name=True)
    GROUP BY symbol, date_trunc('minute', ts)
    """
    df_ohlc_all = con.execute(query, [valid_dom_files]).df()
    con.close()

    log(f"Extracted {len(df_ohlc_all):,} true 1-minute OHLC bars in {time.time()-t0:.1f}s.")

    # Ensure tz-awareness for merging
    if len(df_ohlc_all) > 0 and df_ohlc_all["ts"].dt.tz is None:
        df_ohlc_all["ts"] = df_ohlc_all["ts"].dt.tz_localize("UTC")
        
    df_oof["ts"] = pd.to_datetime(df_oof["ts"])
    if df_oof["ts"].dt.tz is None:
        df_oof["ts"] = df_oof["ts"].dt.tz_localize("UTC")

    # Drop existing ltp / close_price to replace with exact OHLC
    cols_to_drop = [c for c in ["open_price", "high_price", "low_price", "close_price"] if c in df_oof.columns]
    if cols_to_drop:
        df_oof = df_oof.drop(columns=cols_to_drop)

    log("Merging true OHLC bars onto OOF dataset...")
    df_merged = df_oof.merge(df_ohlc_all, on=["symbol", "ts"], how="left")
    
    # Fill any missing high/low with ltp
    df_merged["open_price"] = df_merged["open_price"].fillna(df_merged["ltp"])
    df_merged["high_price"] = df_merged["high_price"].fillna(df_merged["ltp"])
    df_merged["low_price"] = df_merged["low_price"].fillna(df_merged["ltp"])
    df_merged["close_price"] = df_merged["close_price"].fillna(df_merged["ltp"])

    df_merged = df_merged.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df_merged.to_parquet(OUTPUT_OHLC_OOF_PATH, index=False)
    log(f"\n[SUCCESS] Saved true 5-fold OOF dataset with TRUE OHLC to {OUTPUT_OHLC_OOF_PATH} ({len(df_merged):,} rows)")

if __name__ == "__main__":
    main()
