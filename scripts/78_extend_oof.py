"""
Extend OOF predictions parquet with Jun 15-16 walk-forward folds.

fold 8: train on Jun 3-12, predict Jun 15
fold 9: train on Jun 3-15, predict Jun 16
"""

import sys, time, warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "02_processed"
RESULTS = PROJECT_ROOT / "results"

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate",
    "absorption_buyer_1m", "absorption_buyer_5m",
    "absorption_seller_1m", "absorption_seller_5m",
]
TARGET = "label_60m_1pct"
RANDOM_STATE = 42
ROC_PERIOD = 3
HORIZON_MIN = 60

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

def load_day_data(proc_dir: Path, date_str: str):
    import pyarrow.parquet as pq
    all_dfs = []
    load_cols = FEATURES + [TARGET, "ltp", "ts"]
    labeled_files = list(proc_dir.glob(f"labeled_features_1m_*_{date_str}.parquet"))
    for f in labeled_files:
        slug = f.stem.replace("labeled_features_1m_", "")
        symbol = slug.split("_")[0].upper()
        try:
            schema = pq.read_schema(str(f))
            cols = load_cols.copy()
            if "close_price" in schema.names:
                cols.append("close_price")
            df = pd.read_parquet(f, columns=cols)
            if "close_price" not in df.columns:
                df["close_price"] = df["ltp"]
            df = df.dropna(subset=FEATURES + ["ltp", "ts"])
            if len(df) == 0:
                continue
            df["slug"] = slug
            df["symbol"] = symbol
            df["date"] = date_str
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def train_fold(df_train, df_val, fold_name):
    val_start = df_val["ts"].min()
    cutoff = val_start - timedelta(minutes=HORIZON_MIN)
    df_train_purged = df_train[df_train["ts"] < cutoff]
    df_clean = df_train_purged.dropna(subset=[TARGET])
    X_train = df_clean[FEATURES].values
    y_train = df_clean[TARGET].values
    X_val = df_val[FEATURES].values
    
    lgb_train = lgb.Dataset(X_train, label=pd.Series(y_train).map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}))
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
        "class_weight": "balanced", "random_state": RANDOM_STATE,
        "verbosity": -1, "n_jobs": -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=100)
    probs = model.predict(X_val)
    df_val = df_val.copy()
    df_val["prob_long"] = probs[:, 0]
    df_val["prob_short"] = probs[:, 1]
    df_val["prob_no_trade"] = probs[:, 2]
    log(f"  {fold_name}: train={len(df_clean)} val={len(df_val)} long_avg={probs[:,0].mean():.3f}")
    return df_val

def main():
    log("="*70)
    log("EXTEND OOF: Jun 15-16 walk-forward predictions")
    log("="*70)
    
    new_dates = ["15_Jun_26", "16_Jun_26"]
    
    # Load existing OOF
    oof_path = RESULTS / "oof_predictions_lgbm_label_60m_1pct.parquet"
    df_existing = pd.read_parquet(oof_path)
    log(f"Existing OOF: {len(df_existing)} rows, dates: {sorted(df_existing['date'].unique())}")
    
    # Load base training set (Jun 3-12)
    base_dates = sorted(df_existing["date"].unique())
    log(f"Loading base training days: {base_dates}")
    base_dfs = []
    for d in base_dates:
        df = load_day_data(PROC, d)
        if not df.empty:
            base_dfs.append(df)
    df_base = pd.concat(base_dfs, ignore_index=True)
    df_base["ts"] = pd.to_datetime(df_base["ts"])
    log(f"Base training data: {len(df_base)} rows from {len(base_dates)} days")
    
    def add_metadata(df):
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        ts_ist = df["ts"].dt.tz_convert("Asia/Kolkata")
        df["ts_ist_time"] = ts_ist.dt.time
        df["ts_ist_str"] = ts_ist.dt.strftime("%H:%M:%S")
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
        df["roc_3"] = df.groupby("symbol")["close_price"].pct_change(ROC_PERIOD) * 100.0
        return df
    
    new_oof_parts = []
    
    # Fold 8: train on Jun 3-12, predict Jun 15
    log(f"\n--- Fold 8: train on {base_dates[-1]}, val on 15_Jun_26 ---")
    df_val_15 = load_day_data(PROC, "15_Jun_26")
    if df_val_15.empty:
        log("ERROR: No Jun 15 data found!")
    else:
        df_val_15 = add_metadata(df_val_15)
        df_val_15 = train_fold(df_base, df_val_15, "Fold 8 (Jun3-12 -> Jun15)")
        new_oof_parts.append(df_val_15)
        log(f"  Jun15: {len(df_val_15)} rows, long_avg={df_val_15['prob_long'].mean():.3f}")
    
    # Fold 9: train on Jun 3-15, predict Jun 16
    log(f"\n--- Fold 9: train on Jun 3-15, val on 16_Jun_26 ---")
    df_val_16 = load_day_data(PROC, "16_Jun_26")
    if not df_val_16.empty:
        df_val_16 = add_metadata(df_val_16)
        
        # Combine base + Jun 15 for training (drop prediction columns, keep features)
        df_15_feats = new_oof_parts[0].drop(
            columns=["prob_long", "prob_short", "prob_no_trade"], errors="ignore"
        )
        df_train_9 = pd.concat([df_base, df_15_feats], ignore_index=True)
        df_train_9 = add_metadata(df_train_9)
        
        df_val_16 = train_fold(df_train_9, df_val_16, "Fold 9 (Jun3-15 -> Jun16)")
        new_oof_parts.append(df_val_16)
        log(f"  Jun16: {len(df_val_16)} rows, long_avg={df_val_16['prob_long'].mean():.3f}")
    
    # Combine and save
    df_new = pd.concat(new_oof_parts, ignore_index=True)
    
    # Ensure columns match existing OOF
    expected_cols = set(df_existing.columns) | set(df_new.columns)
    for col in expected_cols:
        if col not in df_new.columns:
            df_new[col] = np.nan
        if col not in df_existing.columns:
            df_existing[col] = np.nan
    
    df_combined = pd.concat([df_existing[df_new.columns], df_new[df_existing.columns]], ignore_index=True)
    df_combined.to_parquet(oof_path, index=False)
    log(f"\nSaved combined OOF: {len(df_combined)} rows ({len(df_existing)} existing + {len(df_new)} new)")
    log(f"Dates: {sorted(df_combined['date'].unique())}")

if __name__ == "__main__":
    main()
