"""
Rebuild OOF: load existing OOF (features), add Jul 24, predict with old model.
Uses DuckDB UNION to avoid pandas dtype conflicts.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
import duckdb

PROJECT = Path(__file__).parent.parent
TARGET_LABEL = "label_60m_1pct"
OOF_PATH = PROJECT / "results" / f"oof_predictions_lgbm_{TARGET_LABEL}.parquet"
MODEL_PATH = PROJECT / "models" / "lgbm_model_60m_1pct_final.txt"

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate", "absorption_buyer_1m",
    "absorption_buyer_5m", "absorption_seller_1m", "absorption_seller_5m",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_roc_3(closes):
    n = len(closes)
    roc = np.full(n, np.nan)
    for i in range(2, n):
        if closes[i - 2] > 0:
            roc[i] = (closes[i] - closes[i - 2]) / closes[i - 2] * 100.0
    return roc

def main():
    t0 = time.time()
    log(f"Loading model: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    log(f"Model loaded: {model.num_feature()} features")

    con = duckdb.connect()
    con.execute("SET TIME ZONE 'UTC'")
    
    # Step 1: Load existing OOF via DuckDB (drop prob columns)
    log(f"Loading existing OOF via DuckDB...")
    feats_expr = ", ".join(f'f."{c}"' for c in FEATURES)
    
    oof_str = str(OOF_PATH).replace("\\", "/")
    query = f"""
        SELECT f.slug, f.symbol, f.date, f.ts, 
               NULL AS open_price, f.close_price, f.ltp,
               f.delta_1m, f.roc_3,
               {feats_expr}
        FROM read_parquet('{oof_str}') f
        ORDER BY f.symbol, f.ts
    """
    df_all = con.execute(query).fetchdf()
    existing_dates = set(df_all["date"].unique())
    log(f"Loaded {len(df_all):,} rows, {len(existing_dates)} dates from OOF")

    # Step 2: Add Jul 24 if not present
    jul24 = "24_Jul_26"
    if jul24 not in existing_dates:
        log(f"Adding {jul24} from labeled features...")
        lf_path = str(PROJECT / "02_processed").replace("\\", "/")
        query_jul24 = f"""
            SELECT symbol, ts,
                   symbol || '_' || strftime(ts::DATE, '%d_%b_%y') AS slug,
                   strftime(ts::DATE, '%d_%b_%y') AS date,
                   ltp, close_price, open_price, delta_1m,
                   {feats_expr}
            FROM read_parquet('{lf_path}/labeled_features_1m_*_{jul24}.parquet',
                              union_by_name=true) f
            ORDER BY symbol, ts
        """
        df_jul24 = con.execute(query_jul24).fetchdf()
        log(f"Jul 24 raw: {len(df_jul24):,} rows, {df_jul24['symbol'].nunique()} symbols")
        
        # Compute roc_3 per symbol
        df_jul24["roc_3"] = np.nan
        for sym in df_jul24["symbol"].unique():
            mask = df_jul24["symbol"] == sym
            idxs = df_jul24.index[mask]
            closes = df_jul24.loc[idxs, "close_price"].values
            roc_vals = compute_roc_3(closes)
            df_jul24.loc[idxs, "roc_3"] = roc_vals
        
        # Append
        df_all = pd.concat([df_all, df_jul24], ignore_index=True)
        log(f"Combined: {len(df_all):,} rows, {df_all['date'].nunique()} dates")
    else:
        log(f"{jul24} already in OOF")
    
    con.close()
    
    log(f"Dates: {sorted(df_all['date'].unique())}")

    # Step 3: Predict with old model
    log("Predicting with old model...")
    X = df_all[FEATURES].fillna(0.0).values
    chunk_size = 100000
    n = len(X)
    preds = np.zeros((n, 3), dtype=np.float32)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        preds[i:end] = model.predict(X[i:end])
    df_all["prob_long"] = preds[:, 0]
    df_all["prob_short"] = preds[:, 1]
    df_all["prob_no_trade"] = preds[:, 2]

    # Step 4: Save
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OOF_PATH, index=False)
    elapsed = time.time() - t0
    log(f"Done in {elapsed:.0f}s. Saved {len(df_all):,} rows to {OOF_PATH}")

if __name__ == "__main__":
    main()