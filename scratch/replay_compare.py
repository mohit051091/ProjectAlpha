"""
Replay historical DOM data through both live and batch code paths,
compare resulting 1-min bar aggregates to verify pixel-perfect match.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from features.bar_accumulator import BarAccumulator
from features.trade_inference import TradeInferenceEngine

PROCESSED_DIR = Path("02_processed")
SYMBOLS = ["ZEEL", "MAPMYINDIA", "EXIDEIND", "SONACOMS", "TATATECH", "INFY", "AEGISLOG", "COFORGE"]

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

COMPARE_FIELDS = [
    ("buy_vol_1m", "buy_vol"),
    ("sell_vol_1m", "sell_vol"),
    ("volume_1m", "volume"),
    ("trade_count_1m", "trade_count"),
]


def run_one(symbol: str, date_str: str) -> dict:
    fname = f"cleaned_dom_{symbol}_{date_str}.parquet"
    fpath = PROCESSED_DIR / fname
    if not fpath.exists():
        return {"symbol": symbol, "date": date_str, "rows": 0, "error": "file not found"}

    df = pd.read_parquet(fpath)
    n_rows = len(df)
    if n_rows == 0:
        return {"symbol": symbol, "date": date_str, "rows": 0, "error": "empty file"}

    # Get date from data to make market-hours filter date-aware
    data_date = df["ts"].iloc[0].date()
    market_open = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=3, minutes=45)
    market_close = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=9, minutes=59, seconds=59)

    # Filter to market hours for BOTH paths (matches batch pipeline Stage 10)
    df = df[(df["ts"] >= market_open) & (df["ts"] <= market_close)].copy()
    n_market_rows = len(df)
    if n_market_rows == 0:
        return {"symbol": symbol, "date": date_str, "rows": n_rows, "error": "no market-hours rows"}

    # ── Live path: sequential BarAccumulator ──
    acc = BarAccumulator()
    live_bars = []
    for _, row in df.iterrows():
        ts_float = row["ts"].timestamp()
        tick = {"ts": ts_float, "ltp": float(row.get("ltp", 0))}
        for i in range(5):
            tick[f"bid{i+1}"] = float(row.get(f"bid{i+1}", 0))
            tick[f"bqty{i+1}"] = int(row.get(f"bqty{i+1}", 0))
            tick[f"ask{i+1}"] = float(row.get(f"ask{i+1}", 0))
            tick[f"aqty{i+1}"] = int(row.get(f"aqty{i+1}", 0))
        tick["total_bid_qty"] = float(row.get("total_bid_qty", 0))
        tick["total_ask_qty"] = float(row.get("total_ask_qty", 0))
        bar = acc.add_tick(tick)
        if bar is not None:
            live_bars.append(bar)

    while True:
        final_bar = acc.flush()
        if final_bar is None:
            break
        live_bars.append(final_bar)

    if not live_bars:
        return {"symbol": symbol, "date": date_str, "rows": n_rows, "error": "no live bars"}

    df_live = pd.DataFrame(live_bars)
    df_live["min_ts"] = pd.to_datetime(df_live["ts"], unit="s", utc=True)

    # ── Batch path: TradeInferenceEngine (same market-hours data) ──
    df_trades = TradeInferenceEngine.infer_trades(df)
    n_trades = len(df_trades)
    if n_trades == 0:
        return {
            "symbol": symbol, "date": date_str, "rows": n_rows,
            "n_bars_live": len(df_live), "n_trades": 0, "error": "no batch trades",
        }

    # Split buy/sell before aggregation
    df_trades["min_ts"] = df_trades["ts"].dt.floor("1min")
    df_trades["buy_qty"] = np.where(df_trades["direction"] == "BUY", df_trades["trade_qty"], 0)
    df_trades["sell_qty"] = np.where(df_trades["direction"] == "SELL", df_trades["trade_qty"], 0)
    df_trades["trade_value"] = df_trades["trade_price"] * df_trades["trade_qty"]
    agg = df_trades.groupby("min_ts").agg(
        buy_vol=("buy_qty", "sum"),
        sell_vol=("sell_qty", "sum"),
        volume=("trade_qty", "sum"),
        trade_count=("trade_qty", "count"),
        trade_value=("trade_value", "sum"),
    ).reset_index()
    agg["vwap"] = np.where(agg["volume"] > 0, agg["trade_value"] / agg["volume"], 0.0)

    # ── Merge ──
    merged = df_live.merge(agg, on="min_ts", how="inner", suffixes=("_live", "_batch"))
    n_match = len(merged)

    results = {
        "symbol": symbol, "date": date_str, "rows": n_rows,
        "n_market_rows": n_market_rows, "n_trades": n_trades,
        "n_bars_live": len(df_live), "n_bars_batch": len(agg), "n_match": n_match,
    }

    # Compare with % error (accepts ~2 lost comparisons per bar from boundary loss)
    max_pct_err = 0.0
    for live_field, batch_field in COMPARE_FIELDS:
        lv = merged[live_field].values.astype(np.float64)
        bv = merged[batch_field].values.astype(np.float64)
        diff = np.abs(lv - bv)
        max_abs = float(diff.max())
        # % error relative to batch volume
        denom = np.maximum(bv, 1.0)
        pct = 100.0 * diff / denom
        max_pct = float(pct.max())
        results[f"max_diff_{live_field}"] = max_abs
        results[f"max_pct_{live_field}"] = max_pct
        if max_pct > max_pct_err:
            max_pct_err = max_pct

    # VWAP comparison
    vwap_lv = merged["vwap_1m"].values.astype(np.float64)
    vwap_bv = merged["vwap"].values.astype(np.float64)
    vwap_diff = np.abs(vwap_lv - vwap_bv)
    vwap_max = float(vwap_diff.max())
    results["max_diff_vwap"] = vwap_max

    # Summary: max % error across all fields
    results["max_pct_err"] = max_pct_err
    return results


def print_result(r: dict):
    """Print one result line."""
    sym = r["symbol"]
    date = r["date"]
    if r.get("error"):
        print(f"  {sym:12s} {date}  SKIP  {r['error']}")
        return
    pct = r.get("max_pct_err", 0)
    diffs = []
    for lf, _ in COMPARE_FIELDS:
        md = r.get(f"max_diff_{lf}", 0)
        mp = r.get(f"max_pct_{lf}", 0)
        diffs.append(f"{lf}={md:.0f} ({mp:.1f}%)")
    vwap_md = r.get("max_diff_vwap", 0)
    if vwap_md:
        diffs.append(f"vwap={vwap_md:.4f}")
    diff_str = " | ".join(diffs) if diffs else ""
    print(f"  {sym:12s} {date}  rows={r['n_market_rows']}  bars={r['n_bars_live']}  "
          f"trades={r['n_trades']}  max_pct={pct:.1f}%")
    print(f"           {diff_str}")


if __name__ == "__main__":
    date_str = "02_Jul_26"
    print(f"\nReplay comparison: {len(SYMBOLS)} symbols, date={date_str}")
    print(f"Feeding cleaned_dom (market-hours filtered) through BOTH paths\n")
    print(f"{'SYMBOL':12s} {'DATE':12s}  DETAIL")
    print("-" * 80)

    for sym in SYMBOLS:
        r = run_one(sym, date_str)
        print_result(r)

    print(f"\n{'='*80}")
    print("KNOWN DIFFERENCE: ~2 lost comparisons per bar (boundary tick skipped)")
    print("Expected max % error: <5% for most bars")
    print()
