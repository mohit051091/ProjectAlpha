"""Analyze per-bar errors: are high % bars low-volume outliers?"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from features.bar_accumulator import BarAccumulator
from features.trade_inference import TradeInferenceEngine

df = pd.read_parquet("02_processed/cleaned_dom_ZEEL_02_Jul_26.parquet")
data_date = df["ts"].iloc[0].date()
mo = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=3, minutes=45)
mc = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=9, minutes=59, seconds=59)
df = df[(df["ts"] >= mo) & (df["ts"] <= mc)].copy()

acc = BarAccumulator()
live_bars = []
for _, row in df.iterrows():
    ts = row["ts"].timestamp()
    tick = {"ts": ts, "ltp": float(row.get("ltp", 0))}
    for i in range(5):
        tick[f"bid{i+1}"] = float(row.get(f"bid{i+1}", 0))
        tick[f"bqty{i+1}"] = int(row.get(f"bqty{i+1}", 0))
        tick[f"ask{i+1}"] = float(row.get(f"ask{i+1}", 0))
        tick[f"aqty{i+1}"] = int(row.get(f"aqty{i+1}", 0))
    bar = acc.add_tick(tick)
    if bar:
        live_bars.append(bar)
while True:
    bar = acc.flush()
    if bar is None:
        break
    live_bars.append(bar)
df_live = pd.DataFrame(live_bars)
df_live["min_ts"] = pd.to_datetime(df_live["ts"], unit="s", utc=True)

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

merged = df_live.merge(agg, on="min_ts", suffixes=("_live", "_batch"))
merged["vol_diff"] = abs(merged["volume_1m"] - merged["volume"])
denom = merged["volume"].clip(1)
merged["vol_pct"] = 100 * merged["vol_diff"] / denom

high = merged[merged["vol_pct"] > 10].sort_values("vol_pct", ascending=False)
print(f"Bars with >10% vol error: {len(high)} / {len(merged)}")
if len(high) > 0:
    cols = ["min_ts", "volume_1m", "volume", "vol_diff", "vol_pct"]
    print(high[cols].head(15).to_string())
    total_all = merged["volume"].sum()
    total_high = high["volume"].sum()
    print(f"\nHigh-error bars volume: {total_high:.0f} / {total_all:.0f} = {100*total_high/total_all:.2f}%")
