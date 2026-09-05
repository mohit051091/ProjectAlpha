"""
Experimental Script 91: Absolute Move & Realized Variance Lift Test (Options Edge Test - Fast Vectorized).
========================================================================================================
Evaluates whether LightGBM signals predict:
1. abs_ret_60m_bps = |terminal 60-minute return| (Naked Straddle Payoff)
2. abs_ret_eod_bps = |terminal EOD return|
3. realized_var_60m = sum of squared 1-minute returns over next 60 bars in bps^2 (Delta-Hedged Straddle Payoff)
4. mfe_bps, mae_bps, ret_60m_bps, ret_eod_bps

Evaluates against Volatility-Matched Controls (matched on Date, Minute-of-Day, and 20m Trailing Realized Volatility Decile).
Reports:
- Pooled Mean Signal vs Control
- Day-Level Mean Difference (bar_D over 41 dates)
- Day-Level Standard Deviation & Date-Clustered t-statistic (n=41 dates)
- Positive Days Count (N_pos / 41)

Runs on:
1. Full Universe (510 Symbols)
2. F&O Liquid Subset (Top 35 most liquid optionable NSE equities)
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

FO_LIQUID_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL",
    "TATAMOTORS", "AXISBANK", "KOTAKBANK", "LT", "HINDUNILVR", "ITC", "BAJFINANCE",
    "MARUTI", "SUNPHARMA", "TATASTEEL", "NTPC", "POWERGRID", "TITAN", "ULTRACEMCO",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "ONGC", "JSWSTEEL", "GRASIM", "HEROMOTOCO",
    "EICHERMOT", "HCLTECH", "WIPRO", "TECHM", "BPCL", "IOC"
]

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
    
    abs_ret_60m_bps = abs(ret_60m_bps)
    abs_ret_eod_bps = abs(ret_eod_bps)
    
    # Calculate 60-bar realized variance (sum of squared 1-minute log returns in bps^2)
    path_prices = np.concatenate(([entry_price], fwd_close[:min(60, n_bars)]))
    log_returns_bps = np.diff(np.log(path_prices)) * 10000.0
    realized_var_60m = np.sum(log_returns_bps ** 2)
    
    return {
        "mfe_bps": mfe_bps,
        "mfe_minute": mfe_min,
        "mae_bps": mae_bps,
        "mae_minute": mae_min,
        "mae_before_mfe_bps": mae_before_mfe_bps,
        "ret_60m_bps": ret_60m_bps,
        "ret_eod_bps": ret_eod_bps,
        "abs_ret_60m_bps": abs_ret_60m_bps,
        "abs_ret_eod_bps": abs_ret_eod_bps,
        "realized_var_60m": realized_var_60m
    }

def run_options_test(df, prob_col, direction="LONG", top_pct=0.001, n_controls=20, subset_name="Full Universe"):
    log(f"Running Options & Realized Variance Test [{subset_name}] for {direction} Top {top_pct*100:.1f}%...")
    t0 = time.time()
    
    df_valid = df.dropna(subset=[prob_col, "close_price"]).copy()
    df_valid = df_valid.sort_values(["symbol", "ts"]).reset_index(drop=True)
    
    if subset_name == "F&O Liquid Subset":
        df_valid = df_valid[df_valid["symbol"].str.upper().isin(FO_LIQUID_SYMBOLS)].copy()
        if len(df_valid) == 0:
            log("No rows found for F&O liquid subset.")
            return

    # Fast Vectorized Trailing Volatility Calculation
    df_valid["ret_1m"] = df_valid.groupby("symbol")["close_price"].pct_change()
    df_valid["vol_20m"] = df_valid.groupby("symbol")["ret_1m"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True) * 10000.0
    df_valid["vol_20m"] = df_valid["vol_20m"].fillna(df_valid["vol_20m"].median())
    
    # Assign Volatility Decile per Date
    df_valid["vol_decile"] = df_valid.groupby("date")["vol_20m"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) if len(x) >= 10 else 0
    )
    df_valid["time_of_day"] = df_valid["ts"].dt.strftime("%H:%M")
    
    # Extract Top Signals
    top_n = max(1, int(len(df_valid) * top_pct))
    signals = df_valid.sort_values(prob_col, ascending=False).head(top_n).copy()
    signal_symbols = set(signals["symbol"])
    
    # Groupby Indexing for Controls
    df_non_signal = df_valid[~df_valid["symbol"].isin(signal_symbols)]
    
    date_tod_vol_dict = {}
    for (d_str, tod, v_dec), group in df_non_signal.groupby(["date", "time_of_day", "vol_decile"]):
        date_tod_vol_dict[(d_str, tod, v_dec)] = list(zip(group["symbol"].values, group["ts"].values, group["close_price"].values))
        
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
        
        sig_exc = compute_single_excursion(sym, d_str, ts_entry, entry_p, direction)
        if sig_exc is None:
            continue
        sig_exc["date"] = d_str
        signal_excursions.append(sig_exc)
        
        candidates = date_tod_vol_dict.get((d_str, tod, v_dec), [])
        if len(candidates) == 0:
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
    log(f"Processed {len(df_sig):,} signals and {len(df_ctrl):,} controls in {time.time()-t0:.2f}s")

    print_options_results(f"OPTIONS & REALIZED VARIANCE TEST [{subset_name}]: {direction} (Top {top_pct*100:.1f}%)", df_sig, df_ctrl)

def print_options_results(title, df_sig, df_ctrl):
    print("\n" + "=" * 135)
    print(title)
    print("=" * 135)
    print(f"Signals Sampled: {len(df_sig):,} | Matched Control Entries: {len(df_ctrl):,}")
    print("-" * 135)
    
    metrics = [
        "abs_ret_60m_bps", "abs_ret_eod_bps", "realized_var_60m",
        "mfe_bps", "mae_bps", "ret_60m_bps", "ret_eod_bps"
    ]
    
    fmt = "{:<22} {:<13} {:<13} {:<15} {:<15} {:<12} {:<12}"
    print(fmt.format("Metric", "Sig Pooled", "Ctrl Pooled", "Pooled Diff", "Day-Mean Diff", "t (n=dates)", "Pos Days"))
    print("-" * 135)
    
    for m in metrics:
        s_vals = df_sig[m].values
        c_vals = df_ctrl[m].values
        
        s_pooled = np.mean(s_vals)
        c_pooled = np.mean(c_vals)
        pooled_diff = s_pooled - c_pooled
        
        # Day-level aggregation
        s_date_means = df_sig.groupby("date")[m].mean()
        c_date_means = df_ctrl.groupby("date")[m].mean()
        
        common_dates = list(set(s_date_means.index).intersection(set(c_date_means.index)))
        n_dates = len(common_dates)
        
        if n_dates > 1:
            diffs = s_date_means.loc[common_dates] - c_date_means.loc[common_dates]
            day_mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            t_stat = (day_mean_diff / (std_diff / np.sqrt(n_dates))) if std_diff > 0 else np.nan
            pos_days = (diffs > 0).sum()
            pos_str = f"{pos_days}/{n_dates}"
        else:
            day_mean_diff = np.nan
            t_stat = np.nan
            pos_str = "N/A"
            
        print(fmt.format(m, f"{s_pooled:.2f}", f"{c_pooled:.2f}", f"{pooled_diff:+.2f}", f"{day_mean_diff:+.2f}", f"{t_stat:+.2f}", pos_str))
    print("=" * 135)

def main():
    log(f"Loading OHLC-enriched true 5-fold OOF predictions from {OOF_OHLC_PATH}...")
    df = pd.read_parquet(OOF_OHLC_PATH)
    log(f"Loaded {len(df):,} rows.")

    build_group_index(df)

    df_valid = df.dropna(subset=["label_60m_1pct", "close_price"]).copy()

    # 1. FULL UNIVERSE (510 Symbols) - LONG & SHORT
    run_options_test(df_valid, "prob_long", "LONG", top_pct=0.001, subset_name="Full Universe (510 Symbols)")
    run_options_test(df_valid, "prob_short", "SHORT", top_pct=0.001, subset_name="Full Universe (510 Symbols)")

    # 2. F&O LIQUID SUBSET (Top 35 Liquid Stock Universe) - LONG & SHORT
    run_options_test(df_valid, "prob_long", "LONG", top_pct=0.001, subset_name="F&O Liquid Subset")
    run_options_test(df_valid, "prob_short", "SHORT", top_pct=0.001, subset_name="F&O Liquid Subset")

if __name__ == "__main__":
    main()
