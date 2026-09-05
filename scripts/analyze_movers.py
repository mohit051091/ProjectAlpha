"""
Movers Analysis Script — analyze_movers.py
=========================================
Audits specific target stocks (gainers/losers of June 9 and 10) to trace:
  1. Did Random Forest generate alerts for them?
  2. Did the ROC confirmation trigger trades?
  3. What were the entry/exit times and returns?
  4. What features/behaviors are common among them?
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

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

# Target symbols from user
JUN9_GAINERS = ["CEMPRO", "CAPLIPOINT", "CERA", "LTTS", "DATAPATTNS", "OLAELEC", "JAINREC", "PINELAB"]
JUN9_LOSERS = ["ZEEL", "ZYDUSWELL", "SCHNEIDER"]
JUN10_GAINERS = ["CCL", "CARTRADE", "DOMS", "AKUMS"]
JUN10_LOSERS = ["IFCI", "JINDALSAW", "JAINREC"]
ALL_TARGETS = sorted(list(set(JUN9_GAINERS + JUN9_LOSERS + JUN10_GAINERS + JUN10_LOSERS + ["AARTIIND"])))

def main():
    print("Loading data...")
    dom_slugs = {f.stem.replace("cleaned_dom_", "") for f in PROC.glob("cleaned_dom_*.parquet")}
    tick_slugs = {f.stem.replace("cleaned_ticks_", "") for f in PROC.glob("cleaned_ticks_*.parquet")}
    both = sorted(dom_slugs & tick_slugs)
    
    all_dfs = []
    load_cols = FEATURES + [TARGET_LABEL, "ltp", "ts"]
    for slug in both:
        fpath = PROC / f"labeled_features_1m_{slug}.parquet"
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath, columns=load_cols)
            df = df.dropna(subset=[TARGET_LABEL] + FEATURES + ["ltp"])
            df["slug"] = slug
            df["symbol"] = slug.split("_")[0].upper()
            all_dfs.append(df)
        except:
            pass

    if not all_dfs:
        print("ERROR: No data found.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Load fold assignments
    with open(fold_path, "r") as f:
        fold_assignments = json.load(f)

    symbol_to_fold = {key.split("_")[0].upper(): fold for key, fold in fold_assignments.items()}

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

    # Train OOF RF model
    df_all["pred_rf"] = "NO_TRADE"
    X = df_all[FEATURES].values
    y = df_all[TARGET_LABEL].values

    for fold in range(N_FOLDS):
        train_mask = df_all["fold"] != fold
        val_mask = df_all["fold"] == fold
        if val_mask.sum() == 0:
            continue
        X_train, y_train = X[train_mask], y[train_mask]
        X_val = X[val_mask]
        
        # Subsample to speed up
        X_train_sub = X_train[::5]
        y_train_sub = y_train[::5]

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_train_sub, y_train_sub)
        df_all.loc[val_mask, "pred_rf"] = rf.predict(X_val)

    # Sort chronologically by symbol (across days!)
    df_all["ts"] = pd.to_datetime(df_all["ts"])
    df_all = df_all.sort_values(by=["symbol", "ts"]).reset_index(drop=True)
    
    # Compute 9-period ROC on LTP per symbol (carrying over from end of June 9 into June 10)
    df_all["roc_9pct"] = df_all.groupby("symbol")["ltp"].pct_change(9) * 100.0

    print("\n" + "="*80)
    print("AUDITING SPECIFIC TARGET MOVER SYMBOLS")
    print("="*80)
    
    # Filter for symbols present in our dataset
    targets_present = [s for s in ALL_TARGETS if s in df_all["symbol"].unique()]
    print(f"Target symbols present in data: {targets_present}")
    print(f"Target symbols missing from data: {set(ALL_TARGETS) - set(targets_present)}")

    all_trades = []
    
    # Run the user's specific state machine per symbol:
    # - Model: RF alone
    # - Wait for ROC Trigger (>= 1% for LONG, <= -1% for SHORT)
    # - New RF prediction overrides active alert of opposite direction
    # - Hold 180m
    
    horizon = 180
    roc_thresh = 1.0  # 1% since we multiplied by 100
    
    for symbol in targets_present:
        df_sym = df_all[df_all["symbol"] == symbol].copy()
        
        ltp = df_sym["ltp"].values
        ts = df_sym["ts"].values
        preds = df_sym["pred_rf"].values
        roc = df_sym["roc_9pct"].values
        n = len(df_sym)
        
        active_alert = None
        alert_time = None
        
        holding_until = -1
        entry_price = 0.0
        entry_time = None
        direction = None
        
        for t in range(1, n):
            # Check for exits if in trade
            if t <= holding_until:
                continue
                
            # We are currently flat, check for new RF prediction alerts
            pred = preds[t]
            if pred == "LONG":
                active_alert = "LONG"
                alert_time = ts[t]
            elif pred == "SHORT":
                active_alert = "SHORT"
                alert_time = ts[t]
                
            # Check if ROC trigger confirms the alert
            if active_alert == "LONG":
                triggered = roc[t] >= roc_thresh
                if triggered:
                    entry_price = ltp[t]
                    entry_time = ts[t]
                    exit_idx = min(t + horizon, n - 1)
                    exit_price = ltp[exit_idx]
                    pnl = (exit_price - entry_price) / entry_price
                    
                    all_trades.append({
                        "symbol": symbol,
                        "direction": "LONG",
                        "alert_time": alert_time,
                        "entry_time": entry_time,
                        "exit_time": ts[exit_idx],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "roc_at_entry": roc[t]
                    })
                    holding_until = exit_idx
                    active_alert = None
                    
            elif active_alert == "SHORT":
                triggered = roc[t] <= -roc_thresh
                if triggered:
                    entry_price = ltp[t]
                    entry_time = ts[t]
                    exit_idx = min(t + horizon, n - 1)
                    exit_price = ltp[exit_idx]
                    pnl = (entry_price - exit_price) / entry_price
                    
                    all_trades.append({
                        "symbol": symbol,
                        "direction": "SHORT",
                        "alert_time": alert_time,
                        "entry_time": entry_time,
                        "exit_time": ts[exit_idx],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "roc_at_entry": roc[t]
                    })
                    holding_until = exit_idx
                    active_alert = None

    df_trades = pd.DataFrame(all_trades)
    if not df_trades.empty:
        print("\n" + "="*80)
        print("TRADES GENERATED FOR MOVER STOCKS (RF + ROC TRIGGER, 180m HOLD)")
        print("="*80)
        print(df_trades[["symbol", "direction", "alert_time", "entry_time", "pnl", "roc_at_entry"]].to_string(index=False))
        
        # Analyze average stats for gainers and losers
        print("\nSummary statistics:")
        print(f"Total trades generated: {len(df_trades)}")
        print(f"Overall win rate: {len(df_trades[df_trades['pnl'] > 0]) / len(df_trades):.2%}")
        print(f"Average return: {df_trades['pnl'].mean():.4%}")
    else:
        print("\nNo trades were triggered for the target stocks in this period.")
        
    # Let's inspect raw RF alerts and ROC values for a target like IFCI and CEMPRO
    for sym in ["IFCI", "CEMPRO"]:
        if sym in targets_present:
            df_s = df_all[df_all["symbol"] == sym].copy()
            long_alerts = (df_s["pred_rf"] == "LONG").sum()
            short_alerts = (df_s["pred_rf"] == "SHORT").sum()
            roc_max = df_s["roc_9pct"].max()
            roc_min = df_s["roc_9pct"].min()
            print(f"\n{sym} Raw Stats: RF_LONG={long_alerts}, RF_SHORT={short_alerts}, Max_ROC={roc_max:.2f}%, Min_ROC={roc_min:.2f}%")

if __name__ == "__main__":
    main()
