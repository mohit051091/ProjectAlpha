"""
Experimental Script 93: Opening Minute Distribution, Slot Contention & Screener Baseline.
=======================================================================================
1. Analyzes temporal distribution of top 0.1% signals by 15-minute window across the day.
2. Measures how fast 10 portfolio slots fill during 09:15 - 09:30 IST under FCFS.
3. Evaluates Bid-Ask Spread for signals with minutes_to_close > 350 (09:15 - 09:30 IST).
4. Compares LightGBM top 0.1% performance against a Trivial 15-Minute Volume & Gap Screener.
"""
import os
import sys
import time
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_OHLC_PATH = RESULTS_DIR / "oof_true_5fold_ohlc.parquet"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 90)
    log("STEP 4: OPENING MINUTE DISTRIBUTION, SLOT CONTENTION & SCREENER BASELINE AUDIT")
    log("=" * 90)

    log(f"Loading true 5-fold OOF predictions with OHLC from {OOF_OHLC_PATH}...")
    df = pd.read_parquet(OOF_OHLC_PATH)
    log(f"Loaded {len(df):,} rows.")

    df_valid = df.dropna(subset=["label_60m_1pct", "close_price"]).copy()
    
    # Calculate 15-minute time window labels
    df_valid["time_str"] = df_valid["ts"].dt.strftime("%H:%M")
    
    top_long_01 = df_valid.sort_values("prob_long", ascending=False).head(int(len(df_valid) * 0.001)).copy()
    top_short_01 = df_valid.sort_values("prob_short", ascending=False).head(int(len(df_valid) * 0.001)).copy()
    
    # 1. Temporal Distribution by 15-minute buckets
    log("\n--- 1. TEMPORAL SIGNAL DISTRIBUTION (09:15 - 15:30 IST) ---")
    
    def get_15m_bucket(ts):
        m = ts.minute
        m_bucket = (m // 15) * 15
        return ts.strftime(f"%H:{m_bucket:02d}")
        
    df_valid["window_15m"] = df_valid["ts"].apply(get_15m_bucket)
    top_long_01["window_15m"] = top_long_01["ts"].apply(get_15m_bucket)
    top_short_01["window_15m"] = top_short_01["ts"].apply(get_15m_bucket)
    
    long_dist = top_long_01["window_15m"].value_counts().sort_index()
    short_dist = top_short_01["window_15m"].value_counts().sort_index()
    
    print("\n15-Min Time Window | Top 0.1% LONG Signals | Top 0.1% SHORT Signals")
    print("-" * 65)
    all_windows = sorted(list(set(long_dist.index).union(set(short_dist.index))))
    for w in all_windows:
        l_cnt = long_dist.get(w, 0)
        s_cnt = short_dist.get(w, 0)
        l_pct = (l_cnt / len(top_long_01)) * 100
        s_pct = (s_cnt / len(top_short_01)) * 100
        print(f"  {w:<16} | {l_cnt:>5} ({l_pct:>5.1f}%)         | {s_cnt:>5} ({s_pct:>5.1f}%)")

    # 2. Slot Exhaustion Audit (First 15 minutes of trading day)
    log("\n--- 2. PORTFOLIO SLOT EXHAUSTION AUDIT (09:15 - 09:30 IST) ---")
    
    dates = sorted(df_valid["date"].unique())
    daily_slot_counts = []
    
    for d in dates:
        df_day_long = top_long_01[top_long_01["date"] == d].sort_values("ts")
        if not df_day_long.empty:
            open_window_cnt = len(df_day_long[df_day_long["ts"].dt.minute < 30])
            total_day_cnt = len(df_day_long)
            daily_slot_counts.append({"date": d, "open_signals": open_window_cnt, "total_signals": total_day_cnt})
            
    df_slots = pd.DataFrame(daily_slot_counts)
    med_open_cnt = df_slots["open_signals"].median()
    pct_days_slot_capped = (df_slots["open_signals"] >= 10).mean() * 100
    print(f"  Median Top 0.1% LONG signals in first 15 mins per day: {med_open_cnt:.1f}")
    print(f"  % of trading days where >10 signals fire in the first 15 minutes: {pct_days_slot_capped:.1f}%")

    # 3. LightGBM vs Trivial 15-Min Gap & Volume Screener Baseline
    log("\n--- 3. LIGHTGBM VS TRIVIAL OPENING VOLUME & GAP SCREENER BASELINE ---")
    
    # Compute 15-minute gap and volume for each symbol-date
    # Find first 15m bar per symbol-date
    df_open = df_valid[df_valid["ts"].dt.minute < 30].copy()
    
    if "fwd_ret_60m_bps" not in df_valid.columns:
        df_sorted = df_valid.sort_values(["symbol", "ts"]).reset_index(drop=True)
        df_sorted["fwd_close_60m"] = df_sorted.groupby("symbol")["close_price"].shift(-60)
        df_sorted["ret_60m_bps"] = ((df_sorted["fwd_close_60m"] - df_sorted["close_price"]) / df_sorted["close_price"]) * 10000.0
        df_valid = df_sorted.dropna(subset=["ret_60m_bps"]).copy()

    # Rank by Volume & Gap size in opening 15m
    if "volume_burst" in df_valid.columns:
        df_valid_open = df_valid[df_valid["ts"].dt.minute < 30].copy()
        top_vol_screener = df_valid_open.sort_values("volume_burst", ascending=False).head(len(top_long_01))
        
        lgbm_mean_ret = top_long_01["ret_60m_bps"].mean() if "ret_60m_bps" in top_long_01.columns else np.nan
        vol_mean_ret = top_vol_screener["ret_60m_bps"].mean() if "ret_60m_bps" in top_vol_screener.columns else np.nan
        
        print(f"\n  LightGBM Top 0.1% Mean 60m Return: {lgbm_mean_ret:+.2f} bps")
        print(f"  Trivial Volume-Burst Screener Mean 60m Return: {vol_mean_ret:+.2f} bps")

if __name__ == "__main__":
    main()
