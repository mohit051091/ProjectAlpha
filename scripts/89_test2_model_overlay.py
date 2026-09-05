"""
Experimental Script 89: TEST 2 - Does the Model Add Anything On Top?
====================================================================
Evaluates model overlay (Top 1% daily prob threshold) on the ROC momentum events from Test 1.
Restricted strictly to the 33 Chronological Working Set Dates (May 26 - Jul 14, 2026).
Sealed Dates (Jul 15 - Jul 24) are completely excluded.

Rules:
1. Entry at OPEN of bar t+1 (never signal bar close t).
2. Max 1 position per symbol per day.
3. Costs: 30 bps round-trip before 10:00 IST (04:30 UTC), 15 bps after.
4. Square off at session close (no overnight).
5. Group A: Model probability in Top 1% on that trading date.
6. Group B: Everything else.
7. Outputs identical stats table for Group A vs Group B, Difference (A - B), Clustered t-stat (N=33),
   and Daily Event Counts for Group A.
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
    log("TEST 2: DOES THE MODEL ADD ANYTHING ON TOP?")
    log("=" * 115)

    log(f"Loading OHLC dataset from {OOF_OHLC_PATH}...")
    df = pd.read_parquet(OOF_OHLC_PATH)
    log(f"Loaded {len(df):,} rows.")

    # Isolate Chronological Working Set (First 33 dates)
    date_map = df.groupby("date")["ts"].min().sort_values()
    sorted_dates = date_map.index.tolist()
    working_dates = sorted_dates[:33]

    log(f"Working Set: 33 dates ({working_dates[0]} to {working_dates[-1]}).")

    df_work = df[df["date"].isin(working_dates)].copy()
    df_work = df_work.sort_values(["symbol", "date", "ts"]).reset_index(drop=True)
    df_work["ts_np"] = df_work["ts"].dt.tz_localize(None) if df_work["ts"].dt.tz is not None else df_work["ts"]

    # Calculate ROC(3)
    df_work["ltp_lag3"] = df_work.groupby(["symbol", "date"])["close_price"].shift(3)
    df_work["roc3"] = ((df_work["close_price"] - df_work["ltp_lag3"]) / df_work["ltp_lag3"]) * 100.0

    df_work["roc3_lag1"] = df_work.groupby(["symbol", "date"])["roc3"].shift(1)
    
    df_work["long_cross"] = (df_work["roc3"] >= 1.0) & (df_work["roc3_lag1"] < 1.0)
    df_work["short_cross"] = (df_work["roc3"] <= -1.0) & (df_work["roc3_lag1"] > -1.0)

    # Assign Daily Top 1% Model Probability Flags
    log("Assigning Daily Top 1% Model Probability Flags...")
    df_work["prob_long_p99"] = df_work.groupby("date")["prob_long"].transform(lambda x: x.quantile(0.99))
    df_work["prob_short_p99"] = df_work.groupby("date")["prob_short"].transform(lambda x: x.quantile(0.99))

    df_work["is_top1_long"] = df_work["prob_long"] >= df_work["prob_long_p99"]
    df_work["is_top1_short"] = df_work["prob_short"] >= df_work["prob_short_p99"]

    # Build price path lookup
    group_paths = {}
    for (sym, d_str), grp in df_work.groupby(["symbol", "date"]):
        group_paths[(sym, d_str)] = {
            "ts": grp["ts_np"].values,
            "open": grp["open_price"].values,
            "high": grp["high_price"].values,
            "low": grp["low_price"].values,
            "close": grp["close_price"].values
        }

    # Run Group A vs Group B evaluation for LONG and SHORT
    run_group_ab_evaluation("LONG MOMENTUM (ROC3 >= +1.0%)", df_work, "long_cross", "is_top1_long", "LONG", group_paths)
    run_group_ab_evaluation("SHORT MOMENTUM (ROC3 <= -1.0%)", df_work, "short_cross", "is_top1_short", "SHORT", group_paths)

def evaluate_signal_events(sig_df, direction, group_paths):
    sig_rows = sig_df.drop_duplicates(subset=["symbol", "date"], keep="first")
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
        
        if idx_entry >= len(open_arr):
            continue
            
        entry_price = open_arr[idx_entry]
        entry_ts = pd.to_datetime(ts_arr[idx_entry])
        
        fwd_close = close_arr[idx_entry:]
        n_fwd = len(fwd_close)
        
        if n_fwd == 0 or entry_price <= 0:
            continue

        entry_hour_utc = entry_ts.hour
        entry_min_utc = entry_ts.minute
        is_before_10am_ist = (entry_hour_utc < 4) or (entry_hour_utc == 4 and entry_min_utc < 30)
        cost_bps = 30.0 if is_before_10am_ist else 15.0
        
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
            "gross_30m": gross_30m,
            "gross_60m": gross_60m,
            "gross_eod": gross_eod,
            "net_30m": net_30m,
            "net_60m": net_60m,
            "net_eod": net_eod,
        })
        
    return pd.DataFrame(trades)

def run_group_ab_evaluation(title, df_work, flag_col, top1_col, direction, group_paths):
    log(f"Evaluating Group A vs Group B for {title}...")
    
    events_all = df_work[df_work[flag_col] == True].copy()
    
    group_a_df = events_all[events_all[top1_col] == True].copy()
    group_b_df = events_all[events_all[top1_col] == False].copy()
    
    trades_a = evaluate_signal_events(group_a_df, direction, group_paths)
    trades_b = evaluate_signal_events(group_b_df, direction, group_paths)
    
    print("\n" + "=" * 135)
    print(f"TEST 2 RESULTS: {title}")
    print("=" * 135)
    print(f"Group A (Top 1% Model Prob): {len(trades_a):,} trades | Group B (Other): {len(trades_b):,} trades")
    print("-" * 135)
    
    # 1. Print Daily Event Counts for Group A
    log_daily_counts(title, trades_a, trades_b)
    
    # 2. Side-by-side comparison table
    horizons = ["30m", "60m", "eod"]
    
    fmt = "{:<10} {:<15} {:<15} {:<18} {:<15} {:<15}"
    print(fmt.format("Horizon", "Group A Mean Net", "Group B Mean Net", "Net Diff (A - B)", "Day Diff (bar_D)", "t (n=33 dates)"))
    print("-" * 135)
    
    for hz in horizons:
        net_col = f"net_{hz}"
        
        m_a = np.mean(trades_a[net_col]) if not trades_a.empty else np.nan
        m_b = np.mean(trades_b[net_col]) if not trades_b.empty else np.nan
        diff_pooled = m_a - m_b
        
        # Day-level diffs
        day_a = trades_a.groupby("date")[net_col].mean() if not trades_a.empty else pd.Series()
        day_b = trades_b.groupby("date")[net_col].mean() if not trades_b.empty else pd.Series()
        
        common_dates = list(set(day_a.index).intersection(set(day_b.index)))
        n_dates = len(common_dates)
        
        if n_dates > 1:
            diffs = day_a.loc[common_dates] - day_b.loc[common_dates]
            bar_D = np.mean(diffs)
            s_D = np.std(diffs, ddof=1)
            t_stat = (bar_D / (s_D / np.sqrt(n_dates))) if s_D > 0 else np.nan
        else:
            bar_D = np.nan
            t_stat = np.nan
            
        print(fmt.format(
            f"+{hz}",
            f"{m_a:+.2f} bps",
            f"{m_b:+.2f} bps",
            f"{diff_pooled:+.2f} bps",
            f"{bar_D:+.2f} bps",
            f"{t_stat:+.2f}"
        ))
    print("=" * 135)

def log_daily_counts(title, trades_a, trades_b):
    print("\n--- Daily Event Counts (Working Set: 33 Dates) ---")
    dates_a = trades_a.groupby("date").size() if not trades_a.empty else pd.Series()
    dates_b = trades_b.groupby("date").size() if not trades_b.empty else pd.Series()
    
    print(f"Group A Total: {len(trades_a):,} trades across {len(dates_a)} active dates.")
    print(f"Group A Daily Counts -> Min: {dates_a.min() if not dates_a.empty else 0}, Median: {dates_a.median() if not dates_a.empty else 0:.1f}, Max: {dates_a.max() if not dates_a.empty else 0}")
    print(f"Group B Total: {len(trades_b):,} trades across {len(dates_b)} active dates.\n")

if __name__ == "__main__":
    main()
