"""
Utility script: export_predictions_csv.py
========================================
Exports minute-by-minute out-of-fold predictions, actual returns, and 9-period ROC
for selected symbols (e.g. IFCI, AARTIIND) to a human-readable CSV for manual audit.
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
RESULTS.mkdir(exist_ok=True)

fold_path = RESULTS / "baseline_fold_assignments.json"
output_csv = RESULTS / "symbol_predictions_audit.csv"

def main():
    print("Loading data...")
    # Load slugs
    dom_slugs = {f.stem.replace("cleaned_dom_", "") for f in PROC.glob("cleaned_dom_*.parquet")}
    tick_slugs = {f.stem.replace("cleaned_ticks_", "") for f in PROC.glob("cleaned_ticks_*.parquet")}
    both = sorted(dom_slugs & tick_slugs)
    
    # Load master DataFrame
    all_dfs = []
    load_cols = FEATURES + [TARGET_LABEL, "ltp", "ts", "return_60m"]
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

    # Map fold assignments (which are symbol_date keys) to base symbol keys
    symbol_to_fold = {}
    for key, fold in fold_assignments.items():
        sym = key.split("_")[0].upper()
        symbol_to_fold[sym] = fold

    # Assign fold to each row based on base symbol
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

    # Initialize prediction columns
    df_all["pred_lr"] = "NO_TRADE"
    df_all["pred_rf"] = "NO_TRADE"

    X = df_all[FEATURES].values
    y = df_all[TARGET_LABEL].values

    # Train OOF models
    for fold in range(N_FOLDS):
        train_mask = df_all["fold"] != fold
        val_mask = df_all["fold"] == fold
        if val_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_val = X[val_mask]

        X_train_sub = X_train[::5]
        y_train_sub = y_train[::5]

        scaler = RobustScaler()
        X_train_sub_lr = scaler.fit_transform(X_train_sub)
        X_val_lr = scaler.transform(X_val)

        lr = LogisticRegression(solver="lbfgs", C=1.0, class_weight="balanced", max_iter=100, random_state=RANDOM_STATE)
        lr.fit(X_train_sub_lr, y_train_sub)
        df_all.loc[val_mask, "pred_lr"] = lr.predict(X_val_lr)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_train_sub, y_train_sub)
        df_all.loc[val_mask, "pred_rf"] = rf.predict(X_val)

    # Compute 9-period ROC chronologically per symbol
    df_all["ts"] = pd.to_datetime(df_all["ts"])
    df_all = df_all.sort_values(by=["slug", "ts"]).reset_index(drop=True)
    df_all["roc_9pct"] = df_all.groupby("slug")["ltp"].pct_change(9) * 100.0

    # Save predictions for all symbols
    df_audit = df_all.copy()

    # Select columns for audit
    cols_to_save = [
        "ts", "symbol", "slug", "ltp", "roc_9pct", "pred_lr", "pred_rf", TARGET_LABEL, "return_60m"
    ]
    df_audit = df_audit[cols_to_save]
    
    # Save output
    df_audit.to_csv(output_csv, index=False)
    print(f"Saved audit CSV to {output_csv} (contains all symbols)")

if __name__ == "__main__":
    main()
