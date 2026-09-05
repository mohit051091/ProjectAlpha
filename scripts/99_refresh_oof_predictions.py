"""
Refresh OOF predictions using new model weights.
Reads existing OOF parquet (features intact), predicts with new model,
overwrites prob_* columns. Takes ~2 min.
"""
import os
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT = Path(__file__).parent.parent
TARGET_LABEL = os.getenv("AR_TARGET_LABEL", "label_60m_1pct").strip()
DEFAULT_SOURCE_OOF_PATH = PROJECT / "results" / "oof_predictions_lgbm_label_60m_1pct.parquet"
OOF_PATH = PROJECT / "results" / f"oof_predictions_lgbm_{TARGET_LABEL}.parquet"
MODEL_PATH = PROJECT / "models" / f"lgbm_model_{TARGET_LABEL.replace('label_', '')}_final.txt"

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate", "absorption_buyer_1m",
    "absorption_buyer_5m", "absorption_seller_1m", "absorption_seller_5m",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log(f"Loading model: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    log(f"Model loaded: {model.num_feature()} features")
    source_oof_path = Path(os.getenv("AR_OOF_SOURCE_PATH", "")).expanduser().resolve() if os.getenv("AR_OOF_SOURCE_PATH", "").strip() else None
    if source_oof_path is None:
        source_oof_path = OOF_PATH if OOF_PATH.exists() else DEFAULT_SOURCE_OOF_PATH

    log(f"Loading OOF source: {source_oof_path}")
    df = pd.read_parquet(source_oof_path)
    log(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols, {df['date'].nunique()} dates")

    log("Predicting with new model...")
    X = df[FEATURES].values
    chunk_size = 100000
    n = len(X)
    preds = np.zeros((n, 3), dtype=np.float32)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        preds[i:end] = model.predict(X[i:end])
        if (i // chunk_size) % 5 == 0:
            log(f"  predicted {end}/{n} rows")

    # LightGBM multiclass: output order is [prob_class0, prob_class1, prob_class2]
    # Label mapping was {"LONG": 0, "SHORT": 1, "NO_TRADE": 2}
    df["prob_long"] = preds[:, 0]
    df["prob_short"] = preds[:, 1]
    df["prob_no_trade"] = preds[:, 2]

    log("Saving...")
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OOF_PATH, index=False)
    elapsed = time.time() - t0
    log(f"Done in {elapsed:.0f}s. Saved {len(df)} rows to {OOF_PATH}")

if __name__ == "__main__":
    main()
