"""
Debug Omitted Symbols — debug_omitted_symbols.py
=================================================
Custom debugging tool to trace why specific symbols on specific days did not
trigger trades in our backtest sweeps. Traces Daily Open filters, ROC confirmation filters,
model probabilities, and exceptions minute-by-minute.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

def parse_time(ist_time_str):
    return datetime.strptime(ist_time_str, "%H:%M:%S").time()

def audit_symbol_day(df_symbol: pd.DataFrame, theta: float, enable_exceptions: bool):
    df_sorted = df_symbol.sort_values("ts").reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        print("No data found for this symbol on this date.")
        return
        
    symbol = df_sorted["symbol"].iloc[0]
    date = df_sorted["date"].iloc[0]
    ltp = df_sorted["ltp"].values
    prob_long = df_sorted["prob_long"].values
    prob_short = df_sorted["prob_short"].values
    roc = df_sorted["roc_9"].values if "roc_9" in df_sorted.columns else (df_sorted["ltp"].pct_change(9).values * 100.0)
    ts_ist_str = df_sorted["ts_ist_str"].values
    
    daily_open = ltp[0]
    
    print(f"\n=======================================================")
    print(f"AUDITING SYMBOL: {symbol} ON DATE: {date}")
    print(f"Execution Mode: {'Exception-Enabled' if enable_exceptions else 'Standard'}")
    print(f"Probability Threshold: {theta:.2f} | Daily Open: {daily_open:.2f}")
    print(f"=======================================================")
    
    max_prob_l = df_sorted["prob_long"].max()
    max_prob_s = df_sorted["prob_short"].max()
    print(f"Max Probabilities: LONG={max_prob_l:.4f}, SHORT={max_prob_s:.4f}")
    
    # Check if threshold was ever breached
    any_alert = (max_prob_l >= theta) or (max_prob_s >= theta)
    if not any_alert:
        print(f"[OMITTED] Reason: Model probabilities never reached threshold {theta:.2f}.")
        return
        
    alert_long_active = False
    alert_long_time = -1
    alert_short_active = False
    alert_short_time = -1
    holding_until = -1
    
    has_trades = False
    
    for t in range(n):
        # Expiration check
        if alert_long_active and (t - alert_long_time > 60):
            print(f"[{ts_ist_str[t]}] LONG Alert from {ts_ist_str[alert_long_time]} expired after 60 mins without triggering.")
            alert_long_active = False
        if alert_short_active and (t - alert_short_time > 60):
            print(f"[{ts_ist_str[t]}] SHORT Alert from {ts_ist_str[alert_short_time]} expired after 60 mins without triggering.")
            alert_short_active = False
            
        # Alert generation
        if t > holding_until:
            p_l = prob_long[t]
            p_s = prob_short[t]
            if p_l >= theta and p_l > p_s and not alert_long_active:
                alert_long_active = True
                alert_long_time = t
                alert_short_active = False
                print(f"[{ts_ist_str[t]}] >>> Alert LONG generated (Prob: {p_l:.4f}, LTP: {ltp[t]:.2f})")
            elif p_s >= theta and p_s > p_l and not alert_short_active:
                alert_short_active = True
                alert_short_time = t
                alert_long_active = False
                print(f"[{ts_ist_str[t]}] >>> Alert SHORT generated (Prob: {p_s:.4f}, LTP: {ltp[t]:.2f})")
                
        # Trigger evaluation
        if t > holding_until and (alert_long_active or alert_short_active):
            direction = None
            reason = "Standard ROC"
            ltp_t = ltp[t]
            roc_t = roc[t]
            p_l = prob_long[t]
            p_s = prob_short[t]
            
            # Exceptions logic
            if enable_exceptions:
                t_time = parse_time(ts_ist_str[t])
                is_morning = parse_time("09:15:00") <= t_time <= parse_time("09:30:00")
                is_high_conf_long = p_l >= 0.35
                is_high_conf_short = p_s >= 0.35
                
                if alert_long_active and (is_morning or is_high_conf_long):
                    if ltp_t > daily_open:
                        direction = "LONG"
                        reason = "Morning Bypass" if is_morning else "High-Confidence Bypass"
                    else:
                        print(f"[{ts_ist_str[t]}] Exception check: LONG alert active but price <= open ({ltp_t:.2f} <= {daily_open:.2f})")
                elif alert_short_active and (is_morning or is_high_conf_short):
                    if ltp_t < daily_open:
                        direction = "SHORT"
                        reason = "Morning Bypass" if is_morning else "High-Confidence Bypass"
                    else:
                        print(f"[{ts_ist_str[t]}] Exception check: SHORT alert active but price >= open ({ltp_t:.2f} >= {daily_open:.2f})")
            
            # Standard ROC confirmation check (fallback)
            if not direction:
                if alert_long_active:
                    if pd.isna(roc_t):
                        print(f"[{ts_ist_str[t]}] ROC check: LONG standard trigger blocked. ROC is NaN")
                    elif roc_t < 1.0:
                        print(f"[{ts_ist_str[t]}] ROC check: LONG standard trigger blocked. ROC is {roc_t:.2f}% (need >= 1.0%)")
                    elif ltp_t <= daily_open:
                        print(f"[{ts_ist_str[t]}] ROC check: LONG standard trigger blocked. Price <= open ({ltp_t:.2f} <= {daily_open:.2f})")
                    else:
                        direction = "LONG"
                        reason = "Standard ROC"
                elif alert_short_active:
                    if pd.isna(roc_t):
                        print(f"[{ts_ist_str[t]}] ROC check: SHORT standard trigger blocked. ROC is NaN")
                    elif roc_t > -1.0:
                        print(f"[{ts_ist_str[t]}] ROC check: SHORT standard trigger blocked. ROC is {roc_t:.2f}% (need <= -1.0%)")
                    elif ltp_t >= daily_open:
                        print(f"[{ts_ist_str[t]}] ROC check: SHORT standard trigger blocked. Price >= open ({ltp_t:.2f} >= {daily_open:.2f})")
                    else:
                        direction = "SHORT"
                        reason = "Standard ROC"
                        
            if direction:
                print(f"[{ts_ist_str[t]}] SUCCESS! Triggered {direction} trade at {ltp_t:.2f} via {reason} (ROC: {roc_t:.2f}%)")
                alert_long_active = False
                alert_short_active = False
                holding_until = n - 1 # EOD hold
                has_trades = True
                
    if not has_trades:
        print("[OMITTED] Reason: Alerts were generated but never triggered due to the filters logged above.")

def main():
    parser = argparse.ArgumentParser(description="Trace signal gating for specific symbols and dates.")
    parser.add_argument("--target", type=str, default="label_60m_1pct", choices=["label_60m_1pct"])
    parser.add_argument("--mode", type=str, default="Exception-Enabled", choices=["Standard", "Exception-Enabled"])
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--date", type=str, required=True, help="Date to check, e.g. 10_Jun_26")
    parser.add_argument("--symbols", type=str, nargs="+", required=True, help="List of symbol names, e.g. ZEEL AARTIIND")
    args = parser.parse_args()
    
    parquet_file = RESULTS / f"oof_predictions_lgbm_{args.target}.parquet"
    if not parquet_file.exists():
        print(f"ERROR: OOF Parquet file {parquet_file} not found. Run sweeps first.")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_file)
    enable_exceptions = (args.mode == "Exception-Enabled")
    
    # Normalize inputs
    symbols = [s.upper() for s in args.symbols]
    date_str = args.date
    
    # Filter
    df_filtered = df[(df["symbol"].isin(symbols)) & (df["date"] == date_str)]
    
    if df_filtered.empty:
        print(f"No records found for symbols {symbols} on date {date_str} in the OOF dataset.")
        return
        
    for sym in symbols:
        df_sym = df_filtered[df_filtered["symbol"] == sym]
        if df_sym.empty:
            print(f"\nNo data found for symbol {sym} on date {date_str}.")
            continue
        audit_symbol_day(df_sym, args.theta, enable_exceptions)

if __name__ == "__main__":
    main()
