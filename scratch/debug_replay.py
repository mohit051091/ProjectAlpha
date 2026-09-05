"""Debug replay — compare live vs batch trade inference at per-bar detail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from features.bar_accumulator import BarAccumulator
from features.trade_inference import TradeInferenceEngine

PROCESSED_DIR = Path("02_processed")

symbol = "ZEEL"
date_str = "02_Jul_26"
fname = f"cleaned_dom_{symbol}_{date_str}.parquet"
df = pd.read_parquet(PROCESSED_DIR / fname)
print(f"Loaded {len(df)} DOM rows for {symbol}")

# First check: does the DOM data have consecutive timestamps?
df['ts_diff'] = df['ts'].diff().dt.total_seconds()
print(f"\nTimestamp gaps: min={df['ts_diff'].min():.3f}s max={df['ts_diff'].max():.1f}s median={df['ts_diff'].median():.3f}s")

# ── Live path ──
acc = BarAccumulator()
live_bars = []
live_inferred = []  # per-tick detail
for idx, (_, row) in enumerate(df.iterrows()):
    ts_float = row["ts"].timestamp()
    tick = {"ts": ts_float, "ltp": float(row.get("ltp", 0))}
    for i in range(5):
        tick[f"bid{i+1}"] = float(row.get(f"bid{i+1}", 0))
        tick[f"bqty{i+1}"] = int(row.get(f"bqty{i+1}", 0))
        tick[f"ask{i+1}"] = float(row.get(f"ask{i+1}", 0))
        tick[f"aqty{i+1}"] = int(row.get(f"aqty{i+1}", 0))
    tick["total_bid_qty"] = float(row.get("total_bid_qty", 0))
    tick["total_ask_qty"] = float(row.get("total_ask_qty", 0))
    
    # Track per-tick inference (before bar flush)
    n_trades_before = acc.trade_count
    bar = acc.add_tick(tick)
    trades_this_tick = acc.trade_count - n_trades_before
    if trades_this_tick > 0:
        live_inferred.append({
            "row_idx": idx,
            "ts": row["ts"],
            "n_trades": trades_this_tick,
            "buy_vol": acc.buy_vol - (live_bars[-1]["buy_vol_1m"] if live_bars else 0),
            "sell_vol": acc.sell_vol - (live_bars[-1]["sell_vol_1m"] if live_bars else 0),
        })
    if bar is not None:
        bar["row_idx_last"] = idx
        bar["row_idx_start"] = idx - (bar.get("trade_count_1m", 0) or 1)  # rough
        live_bars.append(bar)

while True:
    final_bar = acc.flush()
    if final_bar is None:
        break
    live_bars.append(final_bar)

df_live = pd.DataFrame(live_bars)
df_live["min_ts"] = pd.to_datetime(df_live["ts"], unit="s", utc=True)
print(f"\nLive: {len(df_live)} bars from {len(live_inferred)} inferred trade events")

# ── Batch path ──
df_trades = TradeInferenceEngine.infer_trades(df)
print(f"Batch: {len(df_trades)} inferred trades")

# Aggregate
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

# ── Merge and show first 5 bars with diffs ──
merged = df_live.merge(agg, on="min_ts", how="inner", suffixes=("_live", "_batch"))
merged["buy_diff"] = merged["buy_vol_1m"] - merged["buy_vol"]
merged["sell_diff"] = merged["sell_vol_1m"] - merged["sell_vol"]
merged["vol_diff"] = merged["volume_1m"] - merged["volume"]
merged["tc_diff"] = merged["trade_count_1m"] - merged["trade_count"]

print(f"\nFirst 10 bars with differences:")
cols = ["min_ts", "buy_vol_1m", "buy_vol", "buy_diff", "sell_vol_1m", "sell_vol", "sell_diff",
        "volume_1m", "volume", "vol_diff", "trade_count_1m", "trade_count", "tc_diff"]
print(merged.head(10)[cols].to_string())

# Show bars with biggest volume diff
print(f"\nTop 5 bars by volume diff:")
worst = merged.reindex(merged["vol_diff"].abs().sort_values(ascending=False).index)
print(worst.head(5)[cols].to_string())

# Check the worst bar in detail
worst_row = worst.iloc[0]
worst_ts = worst_row["min_ts"]
print(f"\n\n=== DEEP DIVE: worst bar at {worst_ts} ===")
print(f"Live: buy={worst_row['buy_vol_1m']:.0f} sell={worst_row['sell_vol_1m']:.0f} vol={worst_row['volume_1m']:.0f} tc={worst_row['trade_count_1m']:.0f}")
print(f"Batch: buy={worst_row['buy_vol']:.0f} sell={worst_row['sell_vol']:.0f} vol={worst_row['volume']:.0f} tc={worst_row['trade_count']:.0f}")

# Show DOM rows in that bar range
bar_data = df[(df["ts"] >= worst_ts) & (df["ts"] < worst_ts + pd.Timedelta(minutes=1))]
print(f"\nDOM rows in this bar: {len(bar_data)}")
print(bar_data[["ts", "ltp", "bid1", "bqty1", "ask1", "aqty1", "bid2", "bqty2", "ask2", "aqty2"]].to_string())

# Show batch inferred trades in this bar
batch_in_bar = df_trades[df_trades["min_ts"] == worst_ts]
print(f"\nBatch inferred trades in this bar: {len(batch_in_bar)}")
# Show top 10 by qty
print(batch_in_bar.sort_values("trade_qty", ascending=False).head(10).to_string())

# Show what live inferred in this bar range
# Find the start/end row indices for this bar
bar_start = worst_row.get("row_idx_start", 0)
bar_end = worst_row.get("row_idx_last", len(df))
live_trades_in_bar = [t for t in live_inferred if bar_start <= t["row_idx"] <= bar_end]
print(f"\nLive inferred events in this bar: {len(live_trades_in_bar)}")
if live_trades_in_bar:
    lt = pd.DataFrame(live_trades_in_bar)
    print(lt.sort_values("n_trades", ascending=False).head(10).to_string())
