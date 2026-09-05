"""
Experimental Script 92: Volatility-Matched Control Group Test (Path 1).
========================================================================
Rematches control entries by:
1. Exact trading date
2. Exact minute-of-day
3. Trailing 20-minute realized volatility decile (volatility_20m)
Evaluates if MFE lift survives when controls have identical trailing volatility.
"""
import os
import sys
import time
import datetime
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_OHLC_PATH = RESULTS_DIR / "oof_true_5fold_ohlc.parquet"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

GLOBAL_GROUPS = {}

def build_group_index(df):
    t0 = time.time()
    df_sorted = df.sort_values(["symbol", "date", "ts"]).reset_index(drop=True)
    df_sorted["ts_np"] = df_sorted["ts"].dt.tz_localize(None) if df_sorted["ts"].dt.tz is not None else df_sorted["ts"]
    
    for (sym, d_str), group in df_sorted.groupby(["symbol", "date"]):
        GLOBAL_GROUPS[(sym, d_str)] = {
            "ts": group["ts_np"].values,
            "open": group["open_price"].values,
            "high": group["high_price"].values,
            "low": group["low_price"].values,
            "close": group["close_price"].values
        }
    log(f"Indexed {len(GLOBAL_GROUPS):,} symbol-date price paths with true OHLC in {time.time()-t0:.2f}s")

def compute_single_excursion(sym, d_str, ts_entry, entry_price, direction="LONG"):
    grp = GLOBAL_GROUPS.get((sym, d_str))
    if grp is None:
        return None
        
    ts_arr = grp["ts"]
    high_arr = grp["high"]
    low_arr = grp["low"]
    close_arr = grp["close"]
    
    ts_entry_naive = np.datetime64(ts_entry.tz_localize(None) if ts_entry.tzinfo is not None else ts_entry)
    idx_start = np.searchsorted(ts_arr, ts_entry_naive)
    
    if idx_start >= len(close_arr) - 1:
        return None
        
    fwd_high = high_arr[idx_start + 1:]
    fwd_low = low_arr[idx_start + 1:]
    fwd_close = close_arr[idx_start + 1:]
    n_bars = len(fwd_close)
    
    if n_bars == 0:
        return None
        
    if direction == "LONG":
        mfe_path_bps = ((fwd_high - entry_price) / entry_price) * 10000.0
        mae_path_bps = ((fwd_low - entry_price) / entry_price) * 10000.0
        ret_path_bps = ((fwd_close - entry_price) / entry_price) * 10000.0
    else:
        mfe_path_bps = ((entry_price - fwd_low) / entry_price) * 10000.0
        mae_path_bps = ((entry_price - fwd_high) / entry_price) * 10000.0
        ret_path_bps = ((entry_price - fwd_close) / entry_price) * 10000.0
        
    mfe_idx = np.argmax(mfe_path_bps)
    mfe_bps = mfe_path_bps[mfe_idx]
    mfe_min = mfe_idx + 1
    
    mae_idx = np.argmin(mae_path_bps)
    mae_bps = mae_path_bps[mae_idx]
    mae_min = mae_idx + 1
    
    ret_before_mfe = mae_path_bps[:mfe_idx + 1]
    mae_before_mfe_bps = np.min(ret_before_mfe)
    
    ret_60m_bps = ret_path_bps[min(59, n_bars - 1)]
    ret_eod_bps = ret_path_bps[-1]
    
    return {
        "mfe_bps": mfe_bps,
        "mfe_minute": mfe_min,
        "mae_bps": mae_bps,
        "mae_minute": mae_min,
        "mae_before_mfe_bps": mae_before_mfe_bps,
        "ret_60m_bps": ret_60m_bps,
        "ret_eod_bps": ret_eod_bps
    }

def run_volatility_matched_control_test(df, prob_col, direction="LONG", top_pct=0.001, n_controls=20):
    log(f"Running Volatility-Matched Control Test for {direction} Top {top_pct*100:.1f}% signals...")
    t0 = time.time()
    
    df_valid = df.dropna(subset=[prob_col, "close_price"]).copy()
    df_valid = df_valid.sort_values(["symbol", "ts"]).reset_index(drop=True)
    
    # Compute trailing 20m return volatility
    log("Computing trailing 20-minute realized volatility...")
    df_valid["ret_1m"] = df_valid.groupby("symbol")["close_price"].pct_change()
    df_valid["vol_20m"] = df_valid.groupby("symbol")["ret_1m"].transform(lambda x: x.rolling(20, min_periods=5).std()) * 10000.0
    df_valid["vol_20m"] = df_valid["vol_20m"].fillna(df_valid["vol_20m"].median())
    
    # Assign Volatility Decile per Date
    df_valid["vol_decile"] = df_valid.groupby("date")["vol_20m"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)
    )
    df_valid["time_of_day"] = df_valid["ts"].dt.strftime("%H:%M")
    
    # Extract Top Signals
    top_n = max(1, int(len(df_valid) * top_pct))
    signals = df_valid.sort_values(prob_col, ascending=False).head(top_n).copy()
    signal_symbols = set(signals["symbol"])
    
    # Fast Groupby Indexing for Volatility-Matched Controls (non-signaled symbols)
    df_non_signal = df_valid[~df_valid["symbol"].isin(signal_symbols)]
    
    date_tod_vol_dict = {}
    for (d_str, tod, v_dec), group in df_non_signal.groupby(["date", "time_of_day", "vol_decile"]):
        date_tod_vol_dict[(d_str, tod, v_dec)] = list(zip(group["symbol"].values, group["ts"].values, group["close_price"].values))
        
    log(f"Built date-time-volatility control lookup dictionary in {time.time()-t0:.2f}s")
    
    signal_excursions = []
    control_excursions = []
    
    np.random.seed(42)
    
    for idx, s_row in signals.iterrows():
        sym = s_row["symbol"]
        d_str = s_row["date"]
        tod = s_row["time_of_day"]
        v_dec = s_row["vol_decile"]
        ts_entry = s_row["ts"]
        entry_p = s_row["close_price"]
        
        # Calculate Signal Excursion
        sig_exc = compute_single_excursion(sym, d_str, ts_entry, entry_p, direction)
        if sig_exc is None:
            continue
        sig_exc["date"] = d_str
        signal_excursions.append(sig_exc)
        
        # Sample 20 Control Entries matched on (date, time_of_day, vol_decile)
        candidates = date_tod_vol_dict.get((d_str, tod, v_dec), [])
        if len(candidates) == 0:
            # Fallback to date + time_of_day if exact vol decile has no non-signaled symbols
            candidates = [item for dec in range(10) for item in date_tod_vol_dict.get((d_str, tod, dec), [])]
            
        if len(candidates) > 0:
            n_sample = min(n_controls, len(candidates))
            sampled_indices = np.random.choice(len(candidates), size=n_sample, replace=False)
            for s_i in sampled_indices:
                c_sym, c_ts_raw, c_price = candidates[s_i]
                c_ts = pd.to_datetime(c_ts_raw)
                c_exc = compute_single_excursion(c_sym, d_str, c_ts, c_price, direction)
                if c_exc is not None:
                    c_exc["date"] = d_str
                    control_excursions.append(c_exc)

    df_sig = pd.DataFrame(signal_excursions)
    df_ctrl = pd.DataFrame(control_excursions)
    log(f"Processed {len(df_sig):,} signals and {len(df_ctrl):,} volatility-matched controls in {time.time()-t0:.2f}s")

    print_control_results(f"VOLATILITY-MATCHED CONTROL GROUP TEST: {direction} (Top {top_pct*100:.1f}%)", df_sig, df_ctrl)

def print_control_results(title, df_sig, df_ctrl):
    print("\n" + "=" * 115)
    print(title)
    print("=" * 115)
    print(f"Signals Sampled: {len(df_sig):,} | Volatility-Matched Control Entries Sampled: {len(df_ctrl):,}")
    print("-" * 115)
    
    metrics = ["mfe_bps", "mfe_minute", "mae_bps", "mae_before_mfe_bps", "ret_60m_bps", "ret_eod_bps"]
    fmt = "{:<22} {:<15} {:<15} {:<18} {:<12}"
    print(fmt.format("Metric", "Signal Mean", "Control Mean", "Net Lift (Sig - Ctrl)", "t-statistic"))
    print("-" * 115)
    
    for m in metrics:
        s_vals = df_sig[m].values
        c_vals = df_ctrl[m].values
        
        s_mean = np.mean(s_vals)
        c_mean = np.mean(c_vals)
        net_lift = s_mean - c_mean
        
        # Clustered t-statistic by date (correct signed calculation)
        s_date_means = df_sig.groupby("date")[m].mean()
        c_date_means = df_ctrl.groupby("date")[m].mean()
        
        common_dates = list(set(s_date_means.index).intersection(set(c_date_means.index)))
        if len(common_dates) > 1:
            diffs = s_date_means.loc[common_dates] - c_date_means.loc[common_dates]
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            t_stat = (mean_diff / (std_diff / np.sqrt(len(common_dates)))) if std_diff > 0 else np.nan
        else:
            t_stat = np.nan
            
        print(fmt.format(m, f"{s_mean:.2f}", f"{c_mean:.2f}", f"{net_lift:+.2f} bps", f"{t_stat:+.2f}"))
    print("=" * 115)

def main():
    log(f"Loading OHLC-enriched true 5-fold OOF predictions from {OOF_OHLC_PATH}...")
    df = pd.read_parquet(OOF_OHLC_PATH)
    log(f"Loaded {len(df):,} rows.")

    build_group_index(df)

    df_valid = df.dropna(subset=["label_60m_1pct", "close_price"]).copy()

    # Run Volatility-Matched Control Test for LONG Top 0.1%
    run_volatility_matched_control_test(df_valid, "prob_long", "LONG", top_pct=0.001, n_controls=20)

    # Run Volatility-Matched Control Test for SHORT Top 0.1%
    run_volatility_matched_control_test(df_valid, "prob_short", "SHORT", top_pct=0.001, n_controls=20)

if __name__ == "__main__":
    main()
