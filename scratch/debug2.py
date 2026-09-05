"""Debug replay with market-hours filter — find worst bars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import pandas as pd
import numpy as np
from features.bar_accumulator import BarAccumulator
from features.trade_inference import TradeInferenceEngine

df = pd.read_parquet("02_processed/cleaned_dom_ZEEL_02_Jul_26.parquet")
data_date = df["ts"].iloc[0].date()
market_open = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=3, minutes=45)
market_close = pd.Timestamp(data_date, tz="UTC") + pd.Timedelta(hours=9, minutes=59, seconds=59)

df = df[(df["ts"] >= market_open) & (df["ts"] <= market_close)].copy()
print(f"Rows: {len(df)}")

# Live
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
    bar = acc.add_tick(tick)
    if bar is not None:
        live_bars.append(bar)
while True:
    final = acc.flush()
    if final is None:
        break
    live_bars.append(final)
df_live = pd.DataFrame(live_bars)
df_live["min_ts"] = pd.to_datetime(df_live["ts"], unit="s", utc=True)

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

# Merge
merged = df_live.merge(agg, on="min_ts", suffixes=("_live", "_batch"))
merged["buy_diff"] = merged["buy_vol_1m"] - merged["buy_vol"]
merged["sell_diff"] = merged["sell_vol_1m"] - merged["sell_vol"]
merged["tc_diff"] = merged["trade_count_1m"] - merged["trade_count"]

print(f"Live total: buy={merged['buy_vol_1m'].sum():.0f} sell={merged['sell_vol_1m'].sum():.0f} vol={merged['volume_1m'].sum():.0f} tc={merged['trade_count_1m'].sum():.0f}")
print(f"Batch total: buy={merged['buy_vol'].sum():.0f} sell={merged['sell_vol'].sum():.0f} vol={merged['volume'].sum():.0f} tc={merged['trade_count'].sum():.0f}")

worst = merged.reindex(merged["buy_diff"].abs().sort_values(ascending=False).index)
cols = ["min_ts", "buy_vol_1m", "buy_vol", "buy_diff", "sell_vol_1m", "sell_vol", "sell_diff", "tc_diff"]
print("\nTop 5 bars by |buy_diff|:")
print(worst.head(5)[cols].to_string())

worst_tc = merged.reindex(merged["tc_diff"].abs().sort_values(ascending=False).index)
print("\nTop 5 by |tc_diff|:")
print(worst_tc.head(5)[cols].to_string())
