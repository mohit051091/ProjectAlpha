"""Final replay comparison — percentage-based tolerance."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from features.bar_accumulator import BarAccumulator
from features.trade_inference import TradeInferenceEngine

PROCESSED_DIR = Path("02_processed")
SYMBOLS = ["ZEEL", "MAPMYINDIA", "EXIDEIND", "SONACOMS", "TATATECH", "INFY", "AEGISLOG", "COFORGE"]

def run_one(symbol, date="02_Jul_26"):
    fname = f"cleaned_dom_{symbol}_{date}.parquet"
    fpath = PROCESSED_DIR / fname
    if not fpath.exists():
        return {"symbol": symbol, "error": "not found"}
    df = pd.read_parquet(fpath)
    data_date = df["ts"].iloc[0].date()
    mo = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=3, minutes=45)
    mc = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=9, minutes=59, seconds=59)
    df = df[(df["ts"] >= mo) & (df["ts"] <= mc)].copy()
    n = len(df)
    if n == 0:
        return {"symbol": symbol, "error": "no market rows"}

    # Live
    acc = BarAccumulator()
    lbars = []
    for _, row in df.iterrows():
        ts = row["ts"].timestamp()
        tick = {"ts": ts, "ltp": float(row.get("ltp", 0))}
        for i in range(5):
            tick[f"bid{i+1}"] = float(row.get(f"bid{i+1}", 0))
            tick[f"bqty{i+1}"] = int(row.get(f"bqty{i+1}", 0))
            tick[f"ask{i+1}"] = float(row.get(f"ask{i+1}", 0))
            tick[f"aqty{i+1}"] = int(row.get(f"aqty{i+1}", 0))
        bar = acc.add_tick(tick)
        if bar: lbars.append(bar)
    while True:
        bar = acc.flush()
        if bar is None: break
        lbars.append(bar)
    dl = pd.DataFrame(lbars)
    dl["min_ts"] = pd.to_datetime(dl["ts"], unit="s", utc=True)

    # Batch
    trades = TradeInferenceEngine.infer_trades(df)
    trades["min_ts"] = trades["ts"].dt.floor("1min")
    trades["buy_qty"] = np.where(trades["direction"] == "BUY", trades["trade_qty"], 0)
    trades["sell_qty"] = np.where(trades["direction"] == "SELL", trades["trade_qty"], 0)
    agg = trades.groupby("min_ts").agg(
        buy_vol=("buy_qty", "sum"),
        sell_vol=("sell_qty", "sum"),
        volume=("trade_qty", "sum"),
        trade_count=("trade_qty", "count"),
    ).reset_index()

    live_cols = {"buy_vol": "buy_vol_1m", "sell_vol": "sell_vol_1m", "volume": "volume_1m", "trade_count": "trade_count_1m"}
    m = dl.merge(agg, on="min_ts")
    results = {
        "symbol": symbol, "rows": n, "bars": len(m),
        "live_buy": m[live_cols["buy_vol"]].sum(), "batch_buy": m["buy_vol"].sum(),
        "live_sell": m[live_cols["sell_vol"]].sum(), "batch_sell": m["sell_vol"].sum(),
        "live_vol": m[live_cols["volume"]].sum(), "batch_vol": m["volume"].sum(),
        "live_tc": m[live_cols["trade_count"]].sum(), "batch_tc": m["trade_count"].sum(),
    }
    # Max absolute diffs
    for key in ["buy_vol", "sell_vol", "volume", "trade_count"]:
        diff = abs(m[live_cols[key]] - m[key]).max()
        results[f"max_{key}"] = int(diff)
    # Bars with >5% volume error
    denom = m["volume"].clip(1)
    pct = 100 * abs(m[live_cols["volume"]] - m["volume"]) / denom
    results["bars_over_5pct"] = int((pct > 5).sum())
    return results

results = []
for sym in SYMBOLS:
    r = run_one(sym)
    results.append(r)

print(f"{'Symbol':12s} {'Rows':6s} {'Bars':5s}  {'Buy_vol':>8s} {'Sell_vol':>8s} {'Volume':>8s} {'TC':>6s}")
print(f"{'':12s} {'':6s} {'':5s}  {'diff':>8s} {'diff':>8s} {'diff':>8s} {'diff':>6s}")
print("-" * 65)
for r in results:
    print(f"{r['symbol']:12s} {r['rows']:6d} {r['bars']:5d}  "
          f"{r['live_buy']-r['batch_buy']:8d} {r['live_sell']-r['batch_sell']:8d} "
          f"{r['live_vol']-r['batch_vol']:8d} {r['live_tc']-r['batch_tc']:6d}")
print()
print("Max absolute differences per bar:")
print(f"{'Symbol':12s}  {'buy_vol':>8s}  {'sell_vol':>8s}  {'volume':>8s}  {'tc':>6s}  {'bars>5%':>8s}")
print("-" * 60)
for r in results:
    print(f"{r['symbol']:12s}  {r['max_buy_vol']:8d}  {r['max_sell_vol']:8d}  {r['max_volume']:8d}  {r['max_trade_count']:6d}  {r['bars_over_5pct']:8d}")
