"""
SCRIPT: audit_target_stocks.py
==============================
Audits target gainers/losers of June 9, 10, and 11 to analyze:
  1. RF Alone predictions and times (IST)
  2. RF + ROC Entry confirmations and times (IST)
  3. Returns and P&L results.
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────

TARGET_LABEL = "label_60m_1pct"

FEATURES = [
    "large_trade_ratio",
    "delta_1m",
    "volume_burst",
    "aggressor_ratio",
    "trade_count_burst",
    "imbalance_top5",
    "spread",
    "depth_drop_bid",
    "depth_drop_ask",
    "vwap_distance",
    "volatility_5m",
    "price_acceleration",
    "iceberg_score",
    "bid_replenishment_rate",
    "absorption_buyer_1m",
    "absorption_buyer_5m",
    "absorption_seller_1m",
    "absorption_seller_5m",
]

N_FOLDS = 5
RANDOM_STATE = 42

PROC = Path("02_processed")
RESULTS = Path("results")
fold_path = RESULTS / "baseline_fold_assignments.json"

# Targets from user
TARGET_LONGS = {
    "09_Jun_26": ["CEMPRO", "CAPLIPOINT", "CERA", "LTTS", "DATAPATTNS", "OLAELEC", "JAINREC", "PINELABS"],
    "10_Jun_26": ["CARTRADE", "AKUMS", "CHAMBLFER", "AFCONS", "CONCORDBIO", "ABSLAMC", "COROMANDEL", "CCL", "CRISIL", "AAVAS"],
    "11_Jun_26": ["AEGISLOG", "ZEEL", "DOMS", "ABDL", "BLUEJET"]
}

TARGET_SHORTS = {
    "09_Jun_26": ["ZEEL", "ZYDUSWELL", "SCHNEIDER"],
    "10_Jun_26": ["JAINREC", "IFCI", "JINDALSAW", "NLCINDIA"],
    "11_Jun_26": ["JAINREC", "BALKRISIND", "CL", "CEMPRO", "MRPL"]
}

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def utc_to_ist_str(ts_val):
    ts = pd.to_datetime(ts_val)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ts_ist = ts.tz_convert("Asia/Kolkata")
    return ts_ist.strftime("%H:%M")

def get_both_slugs(proc_dir: Path):
    dom_slugs = {f.stem.replace("cleaned_dom_", "") for f in proc_dir.glob("cleaned_dom_*.parquet")}
    tick_slugs = {f.stem.replace("cleaned_ticks_", "") for f in proc_dir.glob("cleaned_ticks_*.parquet")}
    inf_slugs = {
        f.stem.replace("inferred_trades_", "")
        for f in proc_dir.glob("inferred_trades_*.parquet")
        if f.stem.replace("inferred_trades_", "").count("_") >= 2
    }
    both = sorted(dom_slugs & tick_slugs & inf_slugs)
    return both

def load_master_dataframe(slugs: list, proc_dir: Path, columns: list, label: str):
    all_dfs = []
    load_cols = columns + [label, "ltp", "ts"]

    for slug in slugs:
        fpath = proc_dir / f"labeled_features_1m_{slug}.parquet"
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath, columns=load_cols)
            df = df.dropna(subset=[label] + columns + ["ltp"])
            if len(df) == 0:
                continue
            df["slug"] = slug
            df["symbol"] = slug.split("_")[0].upper()
            all_dfs.append(df)
        except Exception as e:
            log(f"Error loading {slug}: {e}")

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def generate_oof_predictions(df_all: pd.DataFrame, fold_assignments: dict):
    symbol_to_fold = {}
    for key, fold in fold_assignments.items():
        sym = key.split("_")[0].upper()
        symbol_to_fold[sym] = fold

    def get_fold(slug):
        sym = slug.split("_")[0].upper()
        if sym in symbol_to_fold:
            return symbol_to_fold[sym]
        else:
            import hashlib
            h = int(hashlib.md5(sym.encode("utf-8")).hexdigest(), 16)
            return h % N_FOLDS

    df_all["fold"] = df_all["slug"].apply(get_fold)
    df_all["fold"] = df_all["fold"].astype(int)

    df_all["pred_rf"] = "NO_TRADE"
    X = df_all[FEATURES].values
    y = df_all[TARGET_LABEL].values

    for fold in range(N_FOLDS):
        train_mask = df_all["fold"] != fold
        val_mask = df_all["fold"] == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val = X[val_mask]

        if len(X_val) == 0:
            continue

        X_train_sub = X_train[::5]
        y_train_sub = y_train[::5]

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        rf.fit(X_train_sub, y_train_sub)
        df_all.loc[val_mask, "pred_rf"] = rf.predict(X_val)

    return df_all

def main():
    log("Starting target stock audit...")
    
    slugs = get_both_slugs(PROC)
    log(f"Found {len(slugs)} BOTH slugs in {PROC}")
    
    if not slugs:
        log("No data files found.")
        return
        
    df_all = load_master_dataframe(slugs, PROC, FEATURES, TARGET_LABEL)
    if df_all.empty:
        log("No rows loaded.")
        return
        
    if not fold_path.exists():
        log(f"Fold assignments not found. Run 50_baseline_model.py first.")
        return
    with open(fold_path, "r") as f:
        fold_assignments = json.load(f)
        
    df_all = generate_oof_predictions(df_all, fold_assignments)
    
    # Chronological sort
    df_all["ts"] = pd.to_datetime(df_all["ts"])
    df_all = df_all.sort_values(by=["symbol", "ts"]).reset_index(drop=True)
    df_all["roc_9pct"] = df_all.groupby("slug")["ltp"].pct_change(9) * 100.0
    
    # Audit list mapping
    results = []
    
    # Expiration and horizon settings
    exp_window = 60
    horizon = 180
    roc_thresh = 1.0 # 1%
    
    # Group by slug for simulation
    grouped = df_all.groupby("slug")
    
    for slug, df_sym in grouped:
        symbol = slug.split("_")[0].upper()
        date_str = slug.replace(f"{symbol}_", "")
        
        # Check if this stock-day is in the LONG or SHORT targets list
        is_target_long = (date_str in TARGET_LONGS) and (symbol in TARGET_LONGS[date_str])
        is_target_short = (date_str in TARGET_SHORTS) and (symbol in TARGET_SHORTS[date_str])
        
        if not (is_target_long or is_target_short):
            continue
            
        target_type = "LONG PROBABLE" if is_target_long else "SHORT PROBABLE"
        
        ltp = df_sym["ltp"].values
        ts = df_sym["ts"].values
        preds = df_sym[model_col].values if 'model_col' in locals() else df_sym["pred_rf"].values
        roc = df_sym["roc_9pct"].values
        n = len(df_sym)
        
        # Track raw alerts
        rf_long_times = [utc_to_ist_str(ts[t]) for t in range(n) if preds[t] == "LONG"]
        rf_short_times = [utc_to_ist_str(ts[t]) for t in range(n) if preds[t] == "SHORT"]
        
        # Simulate threshold gate alert-trigger trades
        holding_until = -1
        alert_long_active = False
        alert_long_time = -1
        alert_short_active = False
        alert_short_time = -1
        
        trades = []
        
        for t in range(9, n):
            # Expiration
            if alert_long_active and (t - alert_long_time > exp_window):
                alert_long_active = False
            if alert_short_active and (t - alert_short_time > exp_window):
                alert_short_active = False
                
            # Alerts
            if t > holding_until:
                pred = preds[t]
                if pred == "LONG":
                    alert_long_active = True
                    alert_long_time = t
                    alert_short_active = False
                elif pred == "SHORT":
                    alert_short_active = True
                    alert_short_time = t
                    alert_long_active = False
                    
            # Trigger
            if t > holding_until:
                direction = None
                if alert_long_active:
                    if roc[t] >= roc_thresh:
                        direction = "LONG"
                        alert_long_active = False
                elif alert_short_active:
                    if roc[t] <= -roc_thresh:
                        direction = "SHORT"
                        alert_short_active = False
                        
                if direction:
                    entry_price = ltp[t]
                    exit_idx = min(t + horizon, n - 1)
                    exit_price = ltp[exit_idx]
                    pnl = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price
                    
                    trades.append({
                        "direction": direction,
                        "entry_time_ist": utc_to_ist_str(ts[t]),
                        "exit_time_ist": utc_to_ist_str(ts[exit_idx]),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "roc_at_entry": roc[t]
                    })
                    holding_until = exit_idx
                    
        # Log result row
        long_alerts_summary = ", ".join(rf_long_times[:5]) + ("..." if len(rf_long_times) > 5 else "")
        short_alerts_summary = ", ".join(rf_short_times[:5]) + ("..." if len(rf_short_times) > 5 else "")
        
        if not trades:
            results.append({
                "Symbol": symbol,
                "Date": date_str,
                "Target Type": target_type,
                "RF Long Alerts (IST)": long_alerts_summary if long_alerts_summary else "None",
                "RF Short Alerts (IST)": short_alerts_summary if short_alerts_summary else "None",
                "RF+ROC Trade": "No Trade Triggered",
                "Entry Time": "-",
                "P&L": "-",
                "ROC at Entry": "-"
            })
        else:
            for trd in trades:
                results.append({
                    "Symbol": symbol,
                    "Date": date_str,
                    "Target Type": target_type,
                    "RF Long Alerts (IST)": long_alerts_summary if long_alerts_summary else "None",
                    "RF Short Alerts (IST)": short_alerts_summary if short_alerts_summary else "None",
                    "RF+ROC Trade": trd["direction"],
                    "Entry Time": trd["entry_time_ist"],
                    "P&L": f"{trd['pnl']:.2%}",
                    "ROC at Entry": f"{trd['roc_at_entry']:.2f}%"
                })
                
    df_res = pd.DataFrame(results)
    print("\n" + "="*115)
    print("AUDIT RESULTS FOR TARGETED STOCKS (IST TIMESTAMPS)")
    print("="*115)
    print(df_res.to_string(index=False))
    print("="*115)
    
    # Save results to CSV
    out_csv = RESULTS / "target_stocks_audit_report.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved detailed target stocks audit report to: {out_csv}")

if __name__ == "__main__":
    main()
