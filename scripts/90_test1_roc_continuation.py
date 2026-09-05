"""
Experimental Script 90: TEST 1 - Does ROC Momentum Continue? (No Model Involved).
===============================================================================
Evaluates pure ROC momentum continuation on the 33 Chronological Working Set Dates (May 26 - Jul 14, 2026).
Sealed Dates (Jul 15 - Jul 24) are completely excluded.

Rules:
1. Entry at OPEN of bar t+1 (never signal bar close t).
2. Max 1 position per symbol per day.
3. Costs: 30 bps round-trip for entry before 10:00 IST (04:30 UTC), 15 bps after.
4. Square off at session close (no overnight).
5. Output table per horizon (+30m, +60m, EOD):
   Event Count, Mean Gross bps, Mean Net bps, Median Net bps, Std bps, % Positive,
   Date-Clustered t-stat (N=33), Day-level mean, Day-level std, Day-level t-stat (N=33), Positive Days (N_pos/33).

Patterns Tested:
- Single ROC(3) Cross Above +1.0% (LONG) & Below -1.0% (SHORT)
- Double ROC(3) Cross Pattern (LONG & SHORT)
"""
import os
import sys
import time
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_OHLC_PATH = RESULTS_DIR / "oof_true_5fold_ohlc.parquet"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 115)
    log("TEST 1: DOES ROC MOMENTUM CONTINUE? (NO MODEL INVOLVED)")
    log("=" * 115)

    log(f"Loading OHLC dataset from {OOF_OHLC_PATH}...")
    df = pd.read_parquet(OOF_OHLC_PATH)
    log(f"Loaded {len(df):,} rows.")

    # 1. Isolate Chronological Working Set (First 33 dates)
    date_map = df.groupby("date")["ts"].min().sort_values()
    sorted_dates = date_map.index.tolist()
    working_dates = sorted_dates[:33]
    sealed_dates = sorted_dates[33:]

    log(f"Working Set: 33 dates ({working_dates[0]} to {working_dates[-1]}).")
    log(f"Sealed Set: {len(sealed_dates)} dates ({sealed_dates[0]} to {sealed_dates[-1]}) - EXCLUDED.")

    df_work = df[df["date"].isin(working_dates)].copy()
    df_work = df_work.sort_values(["symbol", "date", "ts"]).reset_index(drop=True)
    df_work["ts_np"] = df_work["ts"].dt.tz_localize(None) if df_work["ts"].dt.tz is not None else df_work["ts"]

    log("Calculating ROC(3) across 1-minute bars...")
    df_work["ltp_lag3"] = df_work.groupby(["symbol", "date"])["close_price"].shift(3)
    df_work["roc3"] = ((df_work["close_price"] - df_work["ltp_lag3"]) / df_work["ltp_lag3"]) * 100.0

    # Signal Detections
    df_work["roc3_lag1"] = df_work.groupby(["symbol", "date"])["roc3"].shift(1)
    
    df_work["long_cross"] = (df_work["roc3"] >= 1.0) & (df_work["roc3_lag1"] < 1.0)
    df_work["short_cross"] = (df_work["roc3"] <= -1.0) & (df_work["roc3_lag1"] > -1.0)

    # Double-Cross Detections
    df_work["long_cross_cumsum"] = df_work.groupby(["symbol", "date"])["long_cross"].cumsum()
    df_work["short_cross_cumsum"] = df_work.groupby(["symbol", "date"])["short_cross"].cumsum()

    df_work["long_double_cross"] = df_work["long_cross"] & (df_work["long_cross_cumsum"] >= 2)
    df_work["short_double_cross"] = df_work["short_cross"] & (df_work["short_cross_cumsum"] >= 2)

    # Process Price Paths per Symbol-Date into dict for fast lookup
    log("Building fast price path lookup index...")
    group_paths = {}
    for (sym, d_str), grp in df_work.groupby(["symbol", "date"]):
        group_paths[(sym, d_str)] = {
            "ts": grp["ts_np"].values,
            "open": grp["open_price"].values,
            "high": grp["high_price"].values,
            "low": grp["low_price"].values,
            "close": grp["close_price"].values
        }

    # Run evaluations for 4 pattern configurations
    run_pattern_evaluation("1. SINGLE ROC(3) CROSS ABOVE +1.0% (LONG MOMENTUM)", df_work, "long_cross", "LONG", group_paths)
    run_pattern_evaluation("2. SINGLE ROC(3) CROSS BELOW -1.0% (SHORT MOMENTUM)", df_work, "short_cross", "SHORT", group_paths)
    run_pattern_evaluation("3. DOUBLE ROC(3) CROSS PATTERN (LONG)", df_work, "long_double_cross", "LONG", group_paths)
    run_pattern_evaluation("4. DOUBLE ROC(3) CROSS PATTERN (SHORT)", df_work, "short_double_cross", "SHORT", group_paths)

def evaluate_signals_for_pattern(df_work, flag_col, direction, group_paths):
    sig_rows = df_work[df_work[flag_col] == True].copy()
    if sig_rows.empty:
        return []

    # Rule 5: Max one open position per symbol per day
    sig_rows = sig_rows.drop_duplicates(subset=["symbol", "date"], keep="first")
    
    trades = []
    
    for idx, row in sig_rows.iterrows():
        sym = row["symbol"]
        d_str = row["date"]
        ts_sig = row["ts_np"]
        
        path_data = group_paths.get((sym, d_str))
        if path_data is None:
            continue
            
        ts_arr = path_data["ts"]
        open_arr = path_data["open"]
        close_arr = path_data["close"]
        
        idx_sig = np.searchsorted(ts_arr, ts_sig)
        idx_entry = idx_sig + 1
        
        # Rule 1: Entry at OPEN of bar t+1 (after signal bar t)
        if idx_entry >= len(open_arr):
            continue
            
        entry_price = open_arr[idx_entry]
        entry_ts = pd.to_datetime(ts_arr[idx_entry])
        
        fwd_close = close_arr[idx_entry:]
        n_fwd = len(fwd_close)
        
        if n_fwd == 0 or entry_price <= 0:
            continue

        # Rule 3: Cost calculation (30 bps before 10:00 IST / 04:30 UTC, 15 bps after)
        entry_hour_utc = entry_ts.hour
        entry_min_utc = entry_ts.minute
        is_before_10am_ist = (entry_hour_utc < 4) or (entry_hour_utc == 4 and entry_min_utc < 30)
        cost_bps = 30.0 if is_before_10am_ist else 15.0
        
        # Forward returns at +30m, +60m, EOD
        idx_30m = min(29, n_fwd - 1)
        idx_60m = min(59, n_fwd - 1)
        idx_eod = n_fwd - 1
        
        if direction == "LONG":
            gross_30m = ((fwd_close[idx_30m] - entry_price) / entry_price) * 10000.0
            gross_60m = ((fwd_close[idx_60m] - entry_price) / entry_price) * 10000.0
            gross_eod = ((fwd_close[idx_eod] - entry_price) / entry_price) * 10000.0
        else:
            gross_30m = ((entry_price - fwd_close[idx_30m]) / entry_price) * 10000.0
            gross_60m = ((entry_price - fwd_close[idx_60m]) / entry_price) * 10000.0
            gross_eod = ((entry_price - fwd_close[idx_eod]) / entry_price) * 10000.0

        net_30m = gross_30m - cost_bps
        net_60m = gross_60m - cost_bps
        net_eod = gross_eod - cost_bps
        
        trades.append({
            "date": d_str,
            "symbol": sym,
            "entry_price": entry_price,
            "cost_bps": cost_bps,
            "gross_30m": gross_30m,
            "gross_60m": gross_60m,
            "gross_eod": gross_eod,
            "net_30m": net_30m,
            "net_60m": net_60m,
            "net_eod": net_eod,
        })
        
    return pd.DataFrame(trades)

def run_pattern_evaluation(title, df_work, flag_col, direction, group_paths):
    df_trades = evaluate_signals_for_pattern(df_work, flag_col, direction, group_paths)
    
    print("\n" + "=" * 125)
    print(f"PATTERN: {title}")
    print("=" * 125)
    
    if df_trades.empty:
        print("No valid trades executed.")
        return
        
    horizons = ["30m", "60m", "eod"]
    
    fmt = "{:<10} {:<8} {:<12} {:<12} {:<12} {:<10} {:<10} {:<14} {:<12} {:<10} {:<12} {:<10}"
    print(fmt.format(
        "Horizon", "Count", "Mean Gross", "Mean Net", "Median Net", "Std bps", "% Pos Net",
        "Clustered t", "Day-Mean Net", "Day-Std", "Day t (n=33)", "Pos Days"
    ))
    print("-" * 135)
    
    for hz in horizons:
        gross_col = f"gross_{hz}"
        net_col = f"net_{hz}"
        
        n_count = len(df_trades)
        m_gross = np.mean(df_trades[gross_col])
        m_net = np.mean(df_trades[net_col])
        med_net = np.median(df_trades[net_col])
        std_net = np.std(df_trades[net_col], ddof=1)
        pct_pos = (df_trades[net_col] > 0).mean() * 100.0
        
        # Day-level aggregation across N=33 working set dates
        day_net_means = df_trades.groupby("date")[net_col].mean()
        n_dates = len(day_net_means)
        
        bar_D = np.mean(day_net_means)
        s_D = np.std(day_net_means, ddof=1) if n_dates > 1 else np.nan
        t_day = (bar_D / (s_D / np.sqrt(n_dates))) if (s_D and s_D > 0) else np.nan
        t_clustered = t_day # Day-clustered t-stat on daily means
        pos_days = (day_net_means > 0).sum()
        pos_str = f"{pos_days}/{n_dates}"
        
        print(fmt.format(
            f"+{hz}",
            f"{n_count:,}",
            f"{m_gross:+.2f} bps",
            f"{m_net:+.2f} bps",
            f"{med_net:+.2f} bps",
            f"{std_net:.2f}",
            f"{pct_pos:.1f}%",
            f"{t_clustered:+.2f}",
            f"{bar_D:+.2f} bps",
            f"{s_D:.2f}",
            f"{t_day:+.2f}",
            pos_str
        ))
    print("=" * 135)

if __name__ == "__main__":
    main()
