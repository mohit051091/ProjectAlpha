"""
Generate OOF predictions for unified-feed days (Jun 15-16).
Reads raw tick parquet (with depth 5), uses FeatureFactory to compute
1-minute features, predicts via trained model, appends to OOF parquet.

Usage:
    python scripts/77_generate_unified_predictions.py
"""

import sys, time, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from features.feature_factory import FeatureFactory
from utils.constants import DATA_DIR

PROJECT = Path(__file__).parent.parent
DATA = Path(DATA_DIR)
MODEL_PATH = PROJECT / "models" / "lgbm_model_60m_1pct_final.txt"
OOF_PATH = PROJECT / "results" / "oof_predictions_lgbm_label_60m_1pct.parquet"

DAYS = [
    ("15_Jun_26", DATA / "ticks_year=2026_month=06_day=15_ticks.parquet"),
    ("16_Jun_26", DATA / "ticks_year=2026_month=06_day=16_ticks.parquet"),
]

FEATURES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate", "absorption_buyer_1m",
    "absorption_buyer_5m", "absorption_seller_1m", "absorption_seller_5m",
]

_prev_closes = {}
MODEL = None


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_prev_closes():
    import duckdb
    con = duckdb.connect()
    con.execute("SET TIME ZONE 'UTC'")
    con.execute(f"CREATE VIEW oof_v AS SELECT * FROM read_parquet('{str(OOF_PATH)}')")
    cache = {}
    for sym in con.execute("SELECT DISTINCT symbol FROM oof_v ORDER BY symbol").fetchdf()["symbol"]:
        rows = con.execute(
            "SELECT close_price FROM oof_v WHERE symbol = ? AND date = '12_Jun_26' ORDER BY ts",
            [sym]
        ).fetchdf()
        if len(rows) >= 3:
            cache[sym] = rows["close_price"].values[-3:].tolist()
    con.close()
    log(f"Loaded {len(cache)} symbols' prev-day closes")
    return cache


def extract_trades(ticks: pd.DataFrame) -> pd.DataFrame:
    """Extract trade records from unified tick data (aggressor + trade_qty)."""
    has_trade = ticks["trade_qty"].fillna(0) > 0
    trades = ticks[has_trade].copy()
    if len(trades) == 0:
        return pd.DataFrame(columns=["ts", "symbol", "trade_qty", "trade_price", "direction"])
    trades["direction"] = trades["aggressor"].str.upper().map({"BUY": "BUY", "SELL": "SELL"}).fillna("BUY")
    trades["trade_price"] = trades["ltp"]
    return trades[["ts", "symbol", "trade_qty", "trade_price", "direction"]].reset_index(drop=True)


def compute_roc_3(closes, prev_closes):
    n = len(closes)
    padded = np.concatenate([prev_closes, closes])
    offset = len(prev_closes)
    roc = np.full(n, np.nan)
    for i in range(n):
        idx = i + offset
        if idx >= 3 and padded[idx - 3] > 0:
            roc[i] = (padded[idx] - padded[idx - 3]) / padded[idx - 3] * 100.0
    return roc


def process_day(day_str, path):
    global _prev_closes, MODEL
    import duckdb

    path_str = str(path).replace("\\", "/")
    con = duckdb.connect()
    con.execute("SET TIME ZONE 'UTC'")

    symbols = con.execute(
        f"SELECT DISTINCT symbol FROM read_parquet('{path_str}') ORDER BY symbol"
    ).fetchdf()["symbol"].tolist()
    log(f"Symbols: {len(symbols)}")

    all_rows = []
    total = len(symbols)
    start_t = time.time()

    for idx, sym in enumerate(symbols, 1):
        ticks = con.execute(f"""
            SELECT ts, symbol, ltp, trade_qty, aggressor,
                   bqty1,bqty2,bqty3,bqty4,bqty5,
                   aqty1,aqty2,aqty3,aqty4,aqty5,
                   bid1,ask1
            FROM read_parquet('{path_str}')
            WHERE symbol = '{sym}'
            ORDER BY ts
        """).df()

        if len(ticks) == 0:
            continue

        # Filter to trading session 09:15-15:29 IST = 03:45-09:59 UTC
        ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)
        day_norm = ticks["ts"].dt.normalize().iloc[0]
        open_utc = day_norm + pd.Timedelta(hours=3, minutes=45)
        close_utc = day_norm + pd.Timedelta(hours=9, minutes=59)
        ticks = ticks[(ticks["ts"] >= open_utc) & (ticks["ts"] <= close_utc)].copy()
        if len(ticks) == 0:
            continue

        # Build trades DataFrame (for FeatureFactory parameter)
        trades = extract_trades(ticks)

        # FeatureFactory.use ticks as both DOM and tick source (unified feed)
        features = FeatureFactory.build_feature_matrix(
            df_dom=ticks, df_tick=ticks, df_trades=trades
        )
        if features.empty:
            continue

        # Predict
        feat_vals = features[FEATURES].fillna(0.0).values
        preds = MODEL.predict(feat_vals)
        if preds.shape[1] == 3:
            prob_long = preds[:, 0]
            prob_short = preds[:, 1]
        else:
            prob_long = preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]
            prob_short = 1.0 - prob_long

        closes = features["close_price"].values
        roc = compute_roc_3(closes, _prev_closes.get(sym, []))

        date_label = day_str
        slug_base = f"{sym}_{date_label}"
        n = len(features)
        for i in range(n):
            all_rows.append({
                "slug": slug_base,
                "symbol": sym,
                "date": date_label,
                "ts": features["ts"].iloc[i],
                "open_price": float(features["open_price"].iloc[i]),
                "close_price": float(closes[i]),
                "prob_long": float(prob_long[i]),
                "prob_short": float(prob_short[i]),
                "roc_3": float(roc[i]) if not np.isnan(roc[i]) else None,
            })

        if idx % 100 == 0 or idx == total:
            log(f"  [{idx}/{total}] {sym} | {time.time()-start_t:.0f}s")

    con.close()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(PROJECT / "results" / f"unified_{day_str}.parquet", index=False)
        log(f"  Saved {len(df)} rows -> unified_{day_str}.parquet")
    else:
        log(f"  WARNING: No rows generated for {day_str}")

    # Update prev_closes for next day
    if all_rows:
        df = pd.DataFrame(all_rows)
        for sym in symbols:
            sub = df[df["symbol"] == sym].sort_values("ts")
            if len(sub) >= 3:
                _prev_closes[sym] = sub["close_price"].values[-3:].tolist()


def main():
    global MODEL
    log("=" * 60)
    log("Unified Feed Predictions - Jun 15/16")
    log("=" * 60)

    log(f"Loading model: {MODEL_PATH}")
    MODEL = lgb.Booster(model_file=str(MODEL_PATH))
    log(f"Model loaded: {MODEL.num_feature()} features")

    global _prev_closes
    _prev_closes = load_prev_closes()

    for day_str, path in DAYS:
        if not path.exists():
            log(f"SKIP: {path} not found")
            continue
        log(f"\n--- {day_str} ---")
        process_day(day_str, path)

    # Merge into main OOF
    log("\n--- Merging into OOF ---")
    oof = pd.read_parquet(OOF_PATH)
    log(f"Existing OOF: {len(oof)} rows ({sorted(oof['date'].unique())})")

    for day_str, _ in DAYS:
        p = PROJECT / "results" / f"unified_{day_str}.parquet"
        if p.exists():
            df_day = pd.read_parquet(p)
            log(f"  Adding {day_str}: {len(df_day)} rows ({df_day['symbol'].nunique()} symbols)")
            oof = pd.concat([oof, df_day], ignore_index=True)

    log(f"Final OOF: {len(oof)} rows ({sorted(oof['date'].unique())})")
    oof.to_parquet(OOF_PATH, index=False)
    log(f"Saved: {OOF_PATH}")

    # Cleanup temp files
    for day_str, _ in DAYS:
        p = PROJECT / "results" / f"unified_{day_str}.parquet"
        if p.exists():
            p.unlink()
    log("Done.")


if __name__ == "__main__":
    main()
