"""
Experimental Script 96: Full MFE / MAE / Excursion Profile Analysis (Fast Dict Indexed).
=====================================================================================
Walks candidate signals (Top 1%, Top 0.1%, Top 0.01%) from entry bar to session close.
Computes:
- mfe_bps & mfe_minute (best unrealised gain & timing)
- mae_bps & mae_minute (worst unrealised loss & timing)
- mae_before_mfe_bps (drawdown prior to peak MFE)
- ret_60m_bps & ret_eod_bps (terminal returns at 60m and EOD close)
- minutes_to_close
- Give-back gap (Median MFE - Median EOD Return)
- MFE / MAE ratio
Reports P5, P25, median, P75, P95, mean for LONG, SHORT, Overall 5-Fold, and Fold 5 only.
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
OOF_PATH = RESULTS_DIR / "oof_true_5fold.parquet"

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

GLOBAL_GROUPS = {}

def build_group_index(df):
    t0 = time.time()
    df_sorted = df.sort_values(["symbol", "date", "ts"]).reset_index(drop=True)
    # Ensure ts is standard numpy datetime64
    df_sorted["ts_np"] = df_sorted["ts"].dt.tz_localize(None) if df_sorted["ts"].dt.tz is not None else df_sorted["ts"]
    
    for (sym, d_str), group in df_sorted.groupby(["symbol", "date"]):
        GLOBAL_GROUPS[(sym, d_str)] = {
            "ts": group["ts_np"].values,
            "ltp": group["ltp"].values
        }
    log(f"Indexed {len(GLOBAL_GROUPS):,} symbol-date price paths in {time.time()-t0:.2f}s")

def extract_excursion_paths(df, prob_col, target_direction="LONG", top_pcts=[0.01, 0.001, 0.0001]):
    df_valid = df.dropna(subset=[prob_col, "ltp"]).copy()
    df_valid = df_valid.sort_values(prob_col, ascending=False).reset_index(drop=True)
    total_n = len(df_valid)
    
    results_by_bucket = {}

    for p in top_pcts:
        n_rows = max(1, int(total_n * p))
        pct_label = f"Top {p*100:.2f}%" if p >= 0.001 else f"Top {p*100:.3f}%"
        sub = df_valid.head(n_rows)
        
        excursions = []
        
        for idx, row in sub.iterrows():
            sym = row["symbol"]
            d_str = row["date"]
            ts_entry = row["ts"]
            entry_price = row["ltp"]
            
            grp = GLOBAL_GROUPS.get((sym, d_str))
            if grp is None:
                continue
                
            ts_arr = grp["ts"]
            ltp_arr = grp["ltp"]
            
            # Match naive datetime64 for searchsorted
            ts_entry_naive = np.datetime64(ts_entry.tz_localize(None) if ts_entry.tzinfo is not None else ts_entry)
            
            idx_start = np.searchsorted(ts_arr, ts_entry_naive)
            if idx_start >= len(ltp_arr) - 1:
                continue
                
            fwd_prices = ltp_arr[idx_start + 1:]
            n_bars = len(fwd_prices)
            
            if n_bars == 0:
                continue
                
            if target_direction == "LONG":
                ret_path_bps = ((fwd_prices - entry_price) / entry_price) * 10000.0
            else:
                ret_path_bps = ((entry_price - fwd_prices) / entry_price) * 10000.0
                
            mfe_idx = np.argmax(ret_path_bps)
            mfe_bps = ret_path_bps[mfe_idx]
            mfe_min = mfe_idx + 1
            
            mae_idx = np.argmin(ret_path_bps)
            mae_bps = ret_path_bps[mae_idx]
            mae_min = mae_idx + 1
            
            ret_before_mfe = ret_path_bps[:mfe_idx + 1]
            mae_before_mfe_bps = np.min(ret_before_mfe)
            
            ret_60m_bps = ret_path_bps[min(59, n_bars - 1)]
            ret_eod_bps = ret_path_bps[-1]
            minutes_to_close = n_bars
            
            excursions.append({
                "mfe_bps": mfe_bps,
                "mfe_minute": mfe_min,
                "mae_bps": mae_bps,
                "mae_minute": mae_min,
                "mae_before_mfe_bps": mae_before_mfe_bps,
                "ret_60m_bps": ret_60m_bps,
                "ret_eod_bps": ret_eod_bps,
                "minutes_to_close": minutes_to_close
            })
            
        results_by_bucket[pct_label] = pd.DataFrame(excursions)
        
    return results_by_bucket

def format_distribution_table(df_ex, metric_name):
    vals = df_ex[metric_name].values
    if len(vals) == 0:
        return {}
    return {
        "Metric": metric_name,
        "Mean": round(np.mean(vals), 2),
        "P5": round(np.percentile(vals, 5), 2),
        "P25": round(np.percentile(vals, 25), 2),
        "Median": round(np.median(vals), 2),
        "P75": round(np.percentile(vals, 75), 2),
        "P95": round(np.percentile(vals, 95), 2),
    }

def print_bucket_summary(title, dict_excursions):
    print("\n" + "=" * 105)
    print(title)
    print("=" * 105)
    
    metrics = ["mfe_bps", "mfe_minute", "mae_bps", "mae_minute", "mae_before_mfe_bps", "ret_60m_bps", "ret_eod_bps", "minutes_to_close"]
    
    for bucket_label, df_ex in dict_excursions.items():
        if df_ex.empty:
            continue
        print(f"\n--- Bucket: {bucket_label} ({len(df_ex):,} signals) ---")
        
        med_mfe = np.median(df_ex["mfe_bps"])
        med_eod = np.median(df_ex["ret_eod_bps"])
        med_mae = abs(np.median(df_ex["mae_bps"]))
        giveback_bps = med_mfe - med_eod
        giveback_pct = (giveback_bps / med_mfe * 100) if med_mfe > 0 else 0.0
        mfe_mae_ratio = (med_mfe / med_mae) if med_mae > 0 else 0.0
        
        print(f"  Diagnostics: Give-back Gap = {giveback_bps:.2f} bps ({giveback_pct:.1f}% peak surrendered) | MFE/MAE Ratio = {mfe_mae_ratio:.2f}")
        
        rows = [format_distribution_table(df_ex, m) for m in metrics]
        df_summary = pd.DataFrame(rows)
        
        fmt = "  {:<22} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}"
        print(fmt.format("Metric", "Mean", "P5", "P25", "Median", "P75", "P95"))
        print("  " + "-" * 80)
        for _, r in df_summary.iterrows():
            print(fmt.format(r["Metric"], r["Mean"], r["P5"], r["P25"], r["Median"], r["P75"], r["P95"]))

def main():
    log("Loading true 5-fold OOF predictions...")
    df = pd.read_parquet(OOF_PATH)
    log(f"Loaded {len(df):,} rows.")

    build_group_index(df)

    df_valid = df.dropna(subset=["label_60m_1pct", "ltp"]).copy()

    # 1. OVERALL 5-FOLD LONG EXCURSIONS
    log("Extracting LONG excursion paths (Overall 5-Fold)...")
    long_overall = extract_excursion_paths(df_valid, "prob_long", "LONG")
    print_bucket_summary("OVERALL 5-FOLD: LONG EXCURSION & MAE/MFE PROFILE", long_overall)

    # 2. OVERALL 5-FOLD SHORT EXCURSIONS
    log("Extracting SHORT excursion paths (Overall 5-Fold)...")
    short_overall = extract_excursion_paths(df_valid, "prob_short", "SHORT")
    print_bucket_summary("OVERALL 5-FOLD: SHORT EXCURSION & MAE/MFE PROFILE", short_overall)

    # 3. FOLD 5 ONLY LONG EXCURSIONS
    log("Extracting LONG excursion paths (Fold 5 Only)...")
    df_f5 = df_valid[df_valid["fold"] == 5].copy()
    long_f5 = extract_excursion_paths(df_f5, "prob_long", "LONG")
    print_bucket_summary("FOLD 5 ONLY (JUL 15 - JUL 24): LONG EXCURSION & MAE/MFE PROFILE", long_f5)

    # 4. FOLD 5 ONLY SHORT EXCURSIONS
    log("Extracting SHORT excursion paths (Fold 5 Only)...")
    short_f5 = extract_excursion_paths(df_f5, "prob_short", "SHORT")
    print_bucket_summary("FOLD 5 ONLY (JUL 15 - JUL 24): SHORT EXCURSION & MAE/MFE PROFILE", short_f5)

if __name__ == "__main__":
    main()
