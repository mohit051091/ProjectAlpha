"""
OOF: train on existing OOF (Jun 3-24), predict Jun 25.
Then deduplicate (remove the first buggy run's predictions) and save.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROC = Path("02_processed")
RESULTS = Path("results")
FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate",
    "absorption_buyer_1m", "absorption_buyer_5m",
    "absorption_seller_1m", "absorption_seller_5m",
]
TARGET = "label_60m_1pct"
ROC_PERIOD = 3

def load_val_data(date_str: str):
    load_cols = FEATURES + [TARGET, "ltp", "ts", "close_price"]
    all_dfs = []
    for f in sorted(PROC.glob(f"labeled_features_1m_*_{date_str}.parquet")):
        try:
            tbl = pq.read_table(f, columns=load_cols)
            slug = f.stem.replace("labeled_features_1m_", "")
            df = tbl.to_pandas()
            if len(df) == 0:
                continue
            df["slug"] = slug
            df["symbol"] = slug.split("_")[0]
            df["date"] = date_str
            all_dfs.append(df)
        except Exception:
            pass
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def main():
    t0 = time.time()
    oof_path = RESULTS / "oof_predictions_lgbm_label_60m_1pct.parquet"

    # Load existing OOF (Jun 3-24 only — drop any buggy Jun 25 rows)
    df_orig = pd.read_parquet(oof_path)
    df_orig = df_orig[df_orig["date"] != "25_Jun_26"].copy()
    print(f"Existing OOF (without Jun 25): {len(df_orig)} rows, dates: {sorted(df_orig['date'].unique())}")

    # Load Jun 25
    print("Loading Jun 25 data...")
    df_val = load_val_data("25_Jun_26")
    print(f"Jun 25: {len(df_val)} rows")
    if df_val.empty:
        return

    # Prep training using existing OOF
    df_train = df_orig.dropna(subset=FEATURES + [TARGET]).copy()
    label_map = {"NO_TRADE": 0, "SHORT": 1, "LONG": 2}
    y_train = df_train[TARGET].map(label_map).values.astype(int)
    X_train = df_train[FEATURES].values

    # Prep validation
    df_val["ts"] = pd.to_datetime(df_val["ts"])
    df_val = df_val.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df_val["roc_3"] = df_val.groupby("symbol")["close_price"].pct_change(ROC_PERIOD) * 100.0
    X_val = df_val[FEATURES].values

    # Train
    dist = np.bincount(y_train)
    print(f"Training: X_train={X_train.shape}, y={dist}")
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
        "class_weight": "balanced", "random_state": 42,
        "verbosity": -1, "n_jobs": -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=100)
    probs = model.predict(X_val)

    # Build Jun 25 predictions
    df_new = df_val[["ts", "slug", "symbol", "date"] + FEATURES + [TARGET, "ltp", "close_price", "roc_3"]].copy()
    df_new["prob_long"] = probs[:, 2]
    df_new["prob_short"] = probs[:, 1]
    df_new["prob_no_trade"] = probs[:, 0]
    ts_ist = df_new["ts"].dt.tz_convert("Asia/Kolkata")
    df_new["ts_ist_time"] = ts_ist.dt.time
    df_new["ts_ist_str"] = ts_ist.dt.strftime("%H:%M:%S")

    # Combine and save
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined.to_parquet(oof_path, index=False)

    print(f"\nOOF: {len(df_orig)} -> {len(df_combined)} rows (+{len(df_new)})")
    print(f"Jun 25 rows: {len(df_new)}")
    print(f"  long_avg={probs[:,2].mean():.4f} >=0.30: {(probs[:,2]>=0.30).sum()} >=0.50: {(probs[:,2]>=0.50).sum()}")
    print(f"  short_avg={probs[:,1].mean():.4f} >=0.25: {(probs[:,1]>=0.25).sum()} >=0.50: {(probs[:,1]>=0.50).sum()}")
    print(f"  no_trade_avg={probs[:,0].mean():.4f}")
    print(f"  dates: {sorted(df_combined['date'].unique())}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
