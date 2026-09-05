"""
Rebuild OOF predictions with vectorized ROC.

Modes:
  --incremental   Only predict (symbol, date) pairs not already in OOF, then append.
  (default)       Predict on ALL labeled features, overwrite existing OOF.
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT = Path(__file__).parent.parent
TARGET_LABEL = "label_60m_1pct"
OOF_PATH = PROJECT / "results" / f"oof_predictions_lgbm_{TARGET_LABEL}.parquet"
MODEL_PATH = Path(r"C:\ProjectAlpha\models\lgbm_model_60m_1pct_final.txt")  # SANDBOX: shared read-only weights
PROCESSED_DIR = Path(os.environ.get("PROJECTALPHA_PROCESSED", str(PROJECT / "02_processed")))

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate", "absorption_buyer_1m",
    "absorption_buyer_5m", "absorption_seller_1m", "absorption_seller_5m",
]

OOF_EXPLICIT_COLS = ["slug", "symbol", "date", "ts", "open_price",
                     "close_price", "ltp", "delta_1m", "roc_3"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_roc_3(closes):
    # SANDBOX FIX (align with live live_computer.py): 3-bar return, 0.0 for first 3 bars.
    # Original backtest used a 2-bar window with NaN for the first 2 bars.
    roc = np.zeros(len(closes))
    if len(closes) > 3:
        prev = closes[:-3]
        curr = closes[3:]
        mask = prev > 0
        roc_vals = np.zeros(len(curr))
        roc_vals[mask] = (curr[mask] - prev[mask]) / prev[mask] * 100.0
        roc[3:] = roc_vals
    return roc


def load_and_prepare(file_list):
    """Load labeled features from parquet files, compute slug/date/roc_3."""
    if not file_list:
        return pd.DataFrame()
    chunks = []
    for f in file_list:
        try:
            df = pd.read_parquet(f)
            if len(df) == 0:
                continue
            if "ts" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts"]):
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
            chunks.append(df)
        except Exception as e:
            log(f"  Skipping {f.name}: {e}")
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df["slug"] = df["symbol"] + "_" + df["ts"].dt.strftime("%d_%b_%y")
    df["date"] = df["ts"].dt.strftime("%d_%b_%y")
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    log(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols, {df['date'].nunique()} dates")

    log("Computing roc_3 (vectorized)...")
    df["roc_3"] = np.nan
    for sym in df["symbol"].unique():
        mask = df["symbol"] == sym
        closes = df.loc[mask, "close_price"].values.astype(float)
        df.loc[mask, "roc_3"] = compute_roc_3(closes)
    return df


def ensure_features(df):
    """Fill missing feature columns with 0 and return feature matrix."""
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURES].fillna(0.0).values


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild OOF predictions.")
    parser.add_argument("--incremental", action="store_true",
                        help="Only predict new (symbol,date) pairs, append to existing OOF.")
    args = parser.parse_args()

    t0 = time.time()
    log(f"Loading model: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    log(f"Model loaded: {model.num_feature()} features")

    all_files = sorted(PROCESSED_DIR.glob("labeled_features_1m_*.parquet"))
    if not all_files:
        log("No labeled features found. Run pipeline stages 1-4 first.")
        return

    # Determine new files
    if args.incremental and OOF_PATH.exists():
        log(f"Reading existing OOF: {OOF_PATH}")
        df_existing = pd.read_parquet(OOF_PATH, columns=["symbol", "date"])
        existing_pairs = set(zip(df_existing["symbol"], df_existing["date"]))
        log(f"Existing OOF: {len(existing_pairs):,} (symbol, date) pairs")

        new_files = []
        for f in all_files:
            stem = f.stem.replace("labeled_features_1m_", "")
            parts = stem.rsplit("_", 3)
            if len(parts) >= 4:
                symbol = parts[0]
                date_str = "_".join(parts[-3:])
            else:
                symbol = parts[0]
                date_str = "_".join(parts[1:])
            if (symbol, date_str) not in existing_pairs:
                new_files.append(f)
        log(f"New files: {len(new_files)} / {len(all_files)} total")
        if not new_files:
            log("Nothing new to predict. OOF is up to date.")
            return
    else:
        new_files = all_files
        log(f"Total files: {len(new_files)}")

    # Load and prepare new data
    df_new = load_and_prepare(new_files)
    if df_new.empty:
        log("No data loaded. Exiting.")
        return

    # Predict
    log("Predicting...")
    X = ensure_features(df_new)
    preds = model.predict(X)
    df_new["prob_long"] = preds[:, 0]
    df_new["prob_short"] = preds[:, 1]
    df_new["prob_no_trade"] = preds[:, 2]

    # Build output: explicit cols + FEATURES + prediction cols
    out_cols = OOF_EXPLICIT_COLS + [c for c in FEATURES if c not in OOF_EXPLICIT_COLS] + \
               ["prob_long", "prob_short", "prob_no_trade"]
    df_out = df_new[out_cols]

    # Save
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.incremental and OOF_PATH.exists():
        df_old = pd.read_parquet(OOF_PATH)
        log(f"Existing OOF: {len(df_old):,} rows")
        df_all = pd.concat([df_old, df_out], ignore_index=True)
        df_all = df_all.sort_values(["symbol", "ts"]).reset_index(drop=True)
        log(f"Appended {len(df_out):,} rows. Total: {len(df_all):,} rows")
    else:
        df_all = df_out.sort_values(["symbol", "ts"]).reset_index(drop=True)

    df_all.to_parquet(OOF_PATH, index=False)
    log(f"Saved {len(df_all):,} rows to {OOF_PATH}")
    log(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
