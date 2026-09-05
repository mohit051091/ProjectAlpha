"""
Experimental Script 98: Build True Out-Of-Fold (OOF) Predictions via 5-Block Purged K-Fold CV.
=============================================================================================
Does NOT touch any existing production pipeline files or existing scripts.
Saves output to results/oof_true_5fold.parquet.
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
import lightgbm as lgb
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "02_processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate", "absorption_buyer_1m",
    "absorption_buyer_5m", "absorption_seller_1m", "absorption_seller_5m",
]
TARGET_LABEL = "label_60m_1pct"
OLD_OOF_PATH = RESULTS_DIR / f"oof_predictions_lgbm_{TARGET_LABEL}.parquet"
NEW_OOF_PATH = RESULTS_DIR / "oof_true_5fold.parquet"

DATE_FILES = defaultdict(list)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def build_file_index():
    t0 = time.time()
    for fname in os.listdir(PROC_DIR):
        if fname.startswith("labeled_features_1m_") and fname.endswith(".parquet"):
            parts = fname.replace(".parquet", "").split("_")
            if len(parts) >= 3:
                d_str = "_".join(parts[-3:])
                DATE_FILES[d_str].append(str(PROC_DIR / fname))
    log(f"Indexed {sum(len(v) for v in DATE_FILES.values()):,} files across {len(DATE_FILES)} dates in {time.time()-t0:.2f}s")

def get_chronological_dates():
    date_strs = list(DATE_FILES.keys())
    parsed = []
    for d_str in date_strs:
        try:
            dt = datetime.datetime.strptime(d_str, "%d_%b_%y")
            parsed.append((dt, d_str))
        except Exception:
            pass
    parsed.sort()
    return [d_str for _, d_str in parsed]

def load_dates_data(dates_list):
    file_paths = []
    for d in dates_list:
        file_paths.extend(DATE_FILES.get(d, []))
    if not file_paths:
        return pd.DataFrame()
        
    load_cols = FEATURES + [TARGET_LABEL, "ltp", "ts", "symbol"]
    dataset = ds.dataset(file_paths, format="parquet")
    
    schema_names = set(dataset.schema.names)
    actual_cols = [c for c in load_cols if c in schema_names]
    
    table = dataset.to_table(columns=actual_cols, use_threads=True)
    df = table.to_pandas()
    
    if "close_price" not in df.columns and "ltp" in df.columns:
        df["close_price"] = df["ltp"]
        
    df["date"] = df["ts"].dt.strftime("%d_%b_%y")
    df["slug"] = df["symbol"] + "_" + df["date"]
    
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return df

def main():
    log("=" * 70)
    log("STEP 2: BUILD TRUE 5-FOLD PURGED OUT-OF-FOLD (OOF) PREDICTIONS")
    log("=" * 70)

    build_file_index()
    dates = get_chronological_dates()
    log(f"Found {len(dates)} total chronological dates: {dates[0]} to {dates[-1]}")

    # Split into 5 contiguous date blocks
    folds_dates = np.array_split(dates, 5)
    log(f"Created 5 contiguous date blocks:")
    for idx, f_dates in enumerate(folds_dates, 1):
        log(f"  Fold {idx}: {len(f_dates)} dates ({f_dates[0]} to {f_dates[-1]})")

    val_fold_dfs = []
    fold_metrics = []

    for fold_idx, val_dates in enumerate(folds_dates, 1):
        val_dates_set = set(val_dates)
        log(f"\n--- Processing Fold {fold_idx}/5 (Validation dates: {val_dates[0]} -> {val_dates[-1]}) ---")
        
        # Determine adjacent dates to purge
        val_indices = [dates.index(d) for d in val_dates]
        min_v_idx, max_v_idx = min(val_indices), max(val_indices)
        
        purge_dates = set()
        if min_v_idx > 0:
            purge_dates.add(dates[min_v_idx - 1])
        if max_v_idx < len(dates) - 1:
            purge_dates.add(dates[max_v_idx + 1])
            
        train_dates = [d for d in dates if d not in val_dates_set and d not in purge_dates]
        log(f"  Train dates: {len(train_dates)} dates | Purged boundary dates: {sorted(list(purge_dates))}")

        # Multi-threaded PyArrow loading
        log(f"  Loading training data ({len(train_dates)} dates)...")
        t_load = time.time()
        df_train = load_dates_data(train_dates)
        log(f"  Loaded {len(df_train):,} train rows in {time.time()-t_load:.1f}s")
        
        log(f"  Loading validation data ({len(val_dates)} dates)...")
        t_val = time.time()
        df_val = load_dates_data(list(val_dates_set))
        log(f"  Loaded {len(df_val):,} val rows in {time.time()-t_val:.1f}s")

        if df_train.empty or df_val.empty:
            log(f"  WARNING: Empty train or val data for fold {fold_idx}. Skipping.")
            continue

        # Drop unlabelled rows for training
        df_train_clean = df_train.dropna(subset=[TARGET_LABEL] + FEATURES).copy()
        
        X_train = df_train_clean[FEATURES].fillna(0.0).values
        y_train_raw = df_train_clean[TARGET_LABEL].values
        label_map = {"LONG": 0, "SHORT": 1, "NO_TRADE": 2}
        y_train = pd.Series(y_train_raw).map(label_map).values

        # Validation set
        df_val_clean = df_val.dropna(subset=[TARGET_LABEL] + FEATURES).copy()
        X_val = df_val_clean[FEATURES].fillna(0.0).values
        y_val_raw = df_val_clean[TARGET_LABEL].values
        y_val = pd.Series(y_val_raw).map(label_map).values

        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
        lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURES, reference=lgb_train)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.03,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 500,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 4,
        }

        # Train with early stopping (100 rounds patience)
        log("  Training LightGBM model (max_depth=6, n_jobs=4)...")
        t_tr = time.time()
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )

        best_iter = model.best_iteration
        log(f"  Trained in {time.time()-t_tr:.1f}s | Best Iteration: {best_iter}")

        # Predict on entire validation block
        for c in FEATURES:
            if c not in df_val.columns:
                df_val[c] = 0.0
        X_val_all = df_val[FEATURES].fillna(0.0).values
        preds_val = model.predict(X_val_all, num_iteration=best_iter)

        df_val["prob_long"] = preds_val[:, 0]
        df_val["prob_short"] = preds_val[:, 1]
        df_val["prob_no_trade"] = preds_val[:, 2]
        df_val["fold"] = fold_idx
        
        # Calculate PR-AUC (Average Precision)
        y_val_long_binary = (y_val == 0).astype(int)
        y_val_short_binary = (y_val == 1).astype(int)
        
        preds_val_clean = model.predict(X_val, num_iteration=best_iter)
        ap_long = average_precision_score(y_val_long_binary, preds_val_clean[:, 0])
        ap_short = average_precision_score(y_val_short_binary, preds_val_clean[:, 1])

        log(f"  Fold {fold_idx} PR-AUC -> LONG: {ap_long:.4f} | SHORT: {ap_short:.4f}")
        
        fold_metrics.append({
            "fold": fold_idx,
            "val_dates": f"{val_dates[0]}..{val_dates[-1]}",
            "val_rows": len(df_val),
            "best_iter": best_iter,
            "ap_long": ap_long,
            "ap_short": ap_short,
        })
        val_fold_dfs.append(df_val)
        
        del df_train, df_train_clean, X_train, y_train, lgb_train, lgb_val, model

    if not val_fold_dfs:
        log("ERROR: No validation folds built.")
        return

    df_oof_true = pd.concat(val_fold_dfs, ignore_index=True)
    df_oof_true = df_oof_true.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df_oof_true.to_parquet(NEW_OOF_PATH, index=False)
    log(f"\n[SUCCESS] Saved true 5-fold OOF predictions to {NEW_OOF_PATH} ({len(df_oof_true):,} rows)")

    # Compute overall true OOF PR-AUC
    df_oof_labeled = df_oof_true.dropna(subset=[TARGET_LABEL] + FEATURES).copy()
    y_true = df_oof_labeled[TARGET_LABEL].map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}).values
    y_true_long = (y_true == 0).astype(int)
    y_true_short = (y_true == 1).astype(int)

    overall_true_ap_long = average_precision_score(y_true_long, df_oof_labeled["prob_long"].values)
    overall_true_ap_short = average_precision_score(y_true_short, df_oof_labeled["prob_short"].values)

    # Load Old In-Sample OOF for side-by-side comparison
    old_ap_long, old_ap_short = np.nan, np.nan
    if OLD_OOF_PATH.exists():
        log(f"\nLoading old in-sample OOF file for comparison: {OLD_OOF_PATH}")
        df_old = pd.read_parquet(OLD_OOF_PATH)
        if "prob_long" in df_old.columns and "prob_short" in df_old.columns:
            if TARGET_LABEL not in df_old.columns:
                df_old = df_old.merge(df_oof_true[["symbol", "ts", TARGET_LABEL]], on=["symbol", "ts"], how="left")
            df_old_clean = df_old.dropna(subset=[TARGET_LABEL, "prob_long", "prob_short"]).copy()
            y_old = df_old_clean[TARGET_LABEL].map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}).values
            old_ap_long = average_precision_score((y_old == 0).astype(int), df_old_clean["prob_long"].values)
            old_ap_short = average_precision_score((y_old == 1).astype(int), df_old_clean["prob_short"].values)

    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE PR-AUC (AVERAGE PRECISION) COMPARISON: IN-SAMPLE vs TRUE OOF")
    print("=" * 90)
    print(f"{'Fold':<8} {'Val Dates':<25} {'Rows':<10} {'Best Iter':<10} {'True LONG PR-AUC':<18} {'True SHORT PR-AUC':<18}")
    print("-" * 90)
    for m in fold_metrics:
        print(f"{m['fold']:<8} {m['val_dates']:<25} {m['val_rows']:<10} {m['best_iter']:<10} {m['ap_long']:<18.4f} {m['ap_short']:<18.4f}")
    print("-" * 90)
    print(f"{'OVERALL TRUE OOF 5-FOLD':<55} {overall_true_ap_long:<18.4f} {overall_true_ap_short:<18.4f}")
    print(f"{'OLD IN-SAMPLE (LEAKED) FILE':<55} {old_ap_long:<18.4f} {old_ap_short:<18.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
