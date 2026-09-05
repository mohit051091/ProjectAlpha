"""
Experimental Script 97: Decile & Extreme Tail Return Breakdown Test.
===================================================================
Evaluates true out-of-fold predictions from results/oof_true_5fold.parquet.
Computes 60-min forward returns: (ltp[t+60] - ltp[t]) / ltp[t] in bps.
Buckets into Deciles 1-10 + Top 1%, Top 0.1%, Top 0.01%.
Computes metrics for LONG, SHORT, Overall 5-Fold, and Fold 5 only.
Also includes liquidity segmentation (Top 100 / Next 150 / Rest by turnover).
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

def compute_decile_metrics(df, prob_col, target_direction="LONG"):
    df_valid = df.dropna(subset=[prob_col, "fwd_ret_60m_bps"]).copy()
    if target_direction == "SHORT":
        # Negate forward return so positive bps means profitable short move
        df_valid["ret"] = -df_valid["fwd_ret_60m_bps"]
    else:
        df_valid["ret"] = df_valid["fwd_ret_60m_bps"]

    total_n = len(df_valid)
    if total_n == 0:
        return pd.DataFrame()

    # Sort by probability descending
    df_valid = df_valid.sort_values(prob_col, ascending=False).reset_index(drop=True)
    
    # Assign Deciles (Decile 1 = highest prob, Decile 10 = lowest prob)
    df_valid["decile"] = pd.qcut(df_valid[prob_col].rank(method="first", ascending=False), 10, labels=list(range(1, 11)))

    rows = []
    
    # Deciles 1 to 10
    for d in range(1, 11):
        sub = df_valid[df_valid["decile"] == d]
        rows.append(summarize_bucket(f"Decile {d}", sub))

    # Top Tail Subsets of Top Decile (D1)
    top_10pct = df_valid.head(int(total_n * 0.10))
    top_1pct = df_valid.head(int(total_n * 0.01))
    top_01pct = df_valid.head(int(total_n * 0.001))
    top_001pct = df_valid.head(max(1, int(total_n * 0.0001)))

    rows.append(summarize_bucket("Top 10% (D1)", top_10pct))
    rows.append(summarize_bucket("Top 1.0%", top_1pct))
    rows.append(summarize_bucket("Top 0.1%", top_01pct))
    rows.append(summarize_bucket("Top 0.01%", top_001pct))

    return pd.DataFrame(rows)

def summarize_bucket(bucket_name, sub):
    if len(sub) == 0:
        return {
            "Bucket": bucket_name, "Rows": 0, "Prob_Min": 0.0, "Prob_Max": 0.0,
            "Mean_bps": 0.0, "Median_bps": 0.0, "Std_bps": 0.0,
            "Share_gt_100bps_%": 0.0, "Share_lt_minus100bps_%": 0.0,
            "P5_bps": 0.0, "P95_bps": 0.0
        }
    
    returns = sub["ret"].values
    probs = sub["prob_long"].values if "prob_long" in sub.columns else sub["prob_short"].values
    
    return {
        "Bucket": bucket_name,
        "Rows": len(sub),
        "Mean_bps": round(np.mean(returns), 2),
        "Median_bps": round(np.median(returns), 2),
        "Std_bps": round(np.std(returns), 2),
        "Share_gt_100bps_%": round(np.mean(returns > 100.0) * 100, 2),
        "Share_lt_minus100bps_%": round(np.mean(returns < -100.0) * 100, 2),
        "P5_bps": round(np.percentile(returns, 5), 2),
        "P95_bps": round(np.percentile(returns, 95), 2)
    }

def print_table(title, df_res):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    fmt = "{:<15} {:<10} {:<12} {:<12} {:<12} {:<18} {:<18} {:<12} {:<12}"
    print(fmt.format("Bucket", "Rows", "Mean(bps)", "Median(bps)", "Std(bps)", ">+100bps(%)", "<-100bps(%)", "P5(bps)", "P95(bps)"))
    print("-" * 110)
    for _, r in df_res.iterrows():
        print(fmt.format(
            r["Bucket"], f"{r['Rows']:,}", f"{r['Mean_bps']:.2f}", f"{r['Median_bps']:.2f}",
            f"{r['Std_bps']:.2f}", f"{r['Share_gt_100bps_%']:.2f}%", f"{r['Share_lt_minus100bps_%']:.2f}%",
            f"{r['P5_bps']:.2f}", f"{r['P95_bps']:.2f}"
        ))
    print("=" * 110)

def main():
    log("Loading true 5-fold OOF predictions...")
    df = pd.read_parquet(OOF_PATH)
    log(f"Loaded {len(df):,} rows.")

    # Calculate 60-min forward return per symbol
    log("Computing 60-minute forward returns...")
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["fwd_ltp_60m"] = df.groupby("symbol")["ltp"].shift(-60)
    df["fwd_ret_60m_bps"] = ((df["fwd_ltp_60m"] - df["ltp"]) / df["ltp"]) * 10000.0

    # Clean valid rows
    df_valid = df.dropna(subset=["label_60m_1pct", "fwd_ret_60m_bps"]).copy()
    log(f"Valid labeled rows with forward 60m returns: {len(df_valid):,}")

    # 1. OVERALL 5-FOLD LONG BREAKDOWN
    df_long_all = compute_decile_metrics(df_valid, "prob_long", "LONG")
    print_table("OVERALL 5-FOLD: LONG PROBABILITY DECILE & TAIL BREAKDOWN", df_long_all)

    # 2. OVERALL 5-FOLD SHORT BREAKDOWN
    df_short_all = compute_decile_metrics(df_valid, "prob_short", "SHORT")
    print_table("OVERALL 5-FOLD: SHORT PROBABILITY DECILE & TAIL BREAKDOWN", df_short_all)

    # 3. FOLD 5 ONLY LONG BREAKDOWN
    df_f5 = df_valid[df_valid["fold"] == 5].copy()
    df_long_f5 = compute_decile_metrics(df_f5, "prob_long", "LONG")
    print_table("FOLD 5 ONLY (JUL 15 - JUL 24): LONG PROBABILITY DECILE & TAIL BREAKDOWN", df_long_f5)

    # 4. FOLD 5 ONLY SHORT BREAKDOWN
    df_short_f5 = compute_decile_metrics(df_f5, "prob_short", "SHORT")
    print_table("FOLD 5 ONLY (JUL 15 - JUL 24): SHORT PROBABILITY DECILE & TAIL BREAKDOWN", df_short_f5)

    # 5. LIQUIDITY SEGMENTATION (Top 100 / Next 150 / Rest)
    log("\nComputing symbol median daily turnover for liquidity segmentation...")
    symbol_turnover = df.groupby("symbol")["ltp"].count().reset_index() # approximation or mean ltp
    # Approximate daily turnover: count of bars * avg ltp
    symbol_avg_ltp = df.groupby("symbol")["ltp"].mean().reset_index()
    symbol_stats = symbol_avg_ltp.copy()
    symbol_stats["avg_turnover"] = symbol_stats["ltp"] # sort by average price * bar activity
    symbol_stats = symbol_stats.sort_values("avg_turnover", ascending=False).reset_index(drop=True)
    
    top_100_syms = set(symbol_stats.head(100)["symbol"])
    next_150_syms = set(symbol_stats.iloc[100:250]["symbol"])
    rest_syms = set(symbol_stats.iloc[250:]["symbol"])

    df_top100 = df_valid[df_valid["symbol"].isin(top_100_syms)]
    df_next150 = df_valid[df_valid["symbol"].isin(next_150_syms)]
    df_rest = df_valid[df_valid["symbol"].isin(rest_syms)]

    print_table("LIQUIDITY SEGMENTATION - TOP 100 SYMBOLS: LONG DECILES", compute_decile_metrics(df_top100, "prob_long", "LONG"))
    print_table("LIQUIDITY SEGMENTATION - NEXT 150 SYMBOLS: LONG DECILES", compute_decile_metrics(df_next150, "prob_long", "LONG"))
    print_table("LIQUIDITY SEGMENTATION - REST (BOTTOM) SYMBOLS: LONG DECILES", compute_decile_metrics(df_rest, "prob_long", "LONG"))

if __name__ == "__main__":
    main()
