"""
Train Final LightGBM Models — 80_train_final_models.py
======================================================
Trains the final LightGBM multiclass models on the entire 12-day dataset
(all available data, no train/validation splits) for:
  - label_60m_1pct (associated with 0.25 threshold Exception-Enabled mode)
Saves the model weights to models/ in both text and JSON formats.
"""

import os
import sys
import json
import warnings
from pathlib import Path
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Force paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Configuration ──────────────────────────────────────────────────────────

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

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "02_processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
TARGET_LABEL = os.getenv("AR_TARGET_LABEL", "label_60m_1pct").strip()
MODEL_PREFIX = os.getenv("AR_MODEL_PREFIX", f"lgbm_model_{TARGET_LABEL.replace('label_', '')}").strip()

def log(msg):
    print(f"[*] {msg}", flush=True)

def get_chronological_days(proc_dir: Path):
    labeled_files = list(proc_dir.glob("labeled_features_1m_*.parquet"))
    date_strs = set()
    for f in labeled_files:
        parts = f.stem.split("_")
        if len(parts) >= 7:
            date_strs.add("_".join(parts[-3:]))
            
    parsed = []
    from datetime import datetime
    for d_str in date_strs:
        try:
            dt = datetime.strptime(d_str, "%d_%b_%y")
            parsed.append((dt, d_str))
        except Exception:
            pass
    parsed.sort()
    return [d_str for _, d_str in parsed]

def load_day_data(proc_dir: Path, date_str: str, columns: list, label: str):
    all_dfs = []
    load_cols = columns + [label, "ltp", "ts"]
    import pyarrow.parquet as pq
    labeled_files = list(proc_dir.glob(f"labeled_features_1m_*_{date_str}.parquet"))
    for f in labeled_files:
        try:
            schema = pq.read_schema(str(f))
            current_cols = load_cols.copy()
            if "close_price" in schema.names:
                current_cols.append("close_price")
            df = pd.read_parquet(f, columns=current_cols)
            if "close_price" not in df.columns:
                df["close_price"] = df["ltp"]
            df = df.dropna(subset=columns + ["ltp", "ts"])
            if len(df) == 0:
                continue
            all_dfs.append(df)
        except Exception:
            pass
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def train_and_save_model(df_all: pd.DataFrame, target: str, name_prefix: str):
    log(f"Preparing training data for target: {target} (rows: {len(df_all)})...")
    
    # Drop rows with NaN labels or features
    df_clean = df_all.dropna(subset=[target] + FEATURES)
    
    X = df_clean[FEATURES].values
    y = df_clean[target].values
    
    # Map classes
    y_mapped = pd.Series(y).map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}).values
    
    # Create LightGBM Dataset
    lgb_train = lgb.Dataset(X, label=y_mapped, feature_name=FEATURES)
    
    # Parameters matching walk-forward sweeps
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "n_jobs": -1
    }
    
    log(f"Training LightGBM model for {target} on all available data (100 rounds)...")
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    
    # Save text model
    txt_path = MODELS_DIR / f"{name_prefix}_final.txt"
    lgb_model.save_model(str(txt_path))
    log(f"Saved text model weights to: {txt_path}")
    
    # Dump to JSON
    json_path = MODELS_DIR / f"{name_prefix}_final.json"
    model_json = lgb_model.dump_model()
    with open(json_path, "w") as f:
        json.dump(model_json, f, indent=2)
    log(f"Saved JSON model architecture to: {json_path}")

def main():
    days = get_chronological_days(PROC)
    log(f"Found {len(days)} chronological days for training.")
    log(f"\n--- Processing Target: {TARGET_LABEL} ---")
    all_dfs = []
    for d in days:
        df_d = load_day_data(PROC, d, FEATURES, TARGET_LABEL)
        if not df_d.empty:
            all_dfs.append(df_d)
    if not all_dfs:
        log(f"ERROR: No data found for target {TARGET_LABEL}")
        return
    df_all = pd.concat(all_dfs, ignore_index=True)
    train_and_save_model(df_all, TARGET_LABEL, MODEL_PREFIX)
    log("\n[SUCCESS] Final model training and weight export complete!")

if __name__ == "__main__":
    main()
