"""Debug feature factory by monkey-patching the trades query."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import duckdb
from features.feature_factory import FeatureFactory, GARBAGE_COLUMNS
from features.event_alignment import EventAlignmentEngine
from features.window_validation import WindowValidationEngine
from features.tick_features import TickFeaturesCalculator

slug = '360ONE_25_Jun_26'
df_dom = pd.read_parquet(f'02_processed/cleaned_dom_{slug}.parquet')
df_trades = pd.read_parquet(f'02_processed/inferred_trades_{slug}.parquet')
df_tick = pd.read_parquet(f'02_processed/cleaned_ticks_{slug}.parquet')

# Replicate the exact flow from feature_factory.py
symbol = df_dom['symbol'].iloc[0]
print(f'Building feature matrix for {symbol}...')

tick_bqty = sorted([c for c in df_tick.columns if c.startswith('bqty') and c[4:].isdigit()], key=lambda c: int(c[4:]))
tick_aqty = sorted([c for c in df_tick.columns if c.startswith('aqty') and c[4:].isdigit()], key=lambda c: int(c[4:]))
is_unified_feed = len(tick_bqty) >= 5 and len(tick_aqty) >= 5
print(f'is_unified_feed={is_unified_feed}, tick_bqty={tick_bqty}, tick_aqty={tick_aqty}')

if not is_unified_feed:
    print('Aligning Tick + DOM events...')
    alignment_result = EventAlignmentEngine.align_tick_dom(df_tick, df_dom, tolerance_s=1.0, window='1min')
    window_metrics = alignment_result['window_metrics']
    overlap_start = max(df_tick['ts'].min(), df_dom['ts'].min())
    overlap_end = min(df_tick['ts'].max(), df_dom['ts'].max())
    eligible_start = overlap_start.floor('1min')
    eligible_end = overlap_end.floor('1min')
    eligible_windows = pd.date_range(start=eligible_start, end=eligible_end, freq='1min', tz='UTC')
    window_metrics = window_metrics[window_metrics['window_start'].isin(eligible_windows)].copy()
    print(f'Filtering to {len(window_metrics)} overlap windows...')
    window_metrics = WindowValidationEngine.compute_scores(window_metrics)
    valid_windows = window_metrics[window_metrics['window_valid']].copy()
    print(f'{len(valid_windows)} valid windows')
else:
    valid_windows = None
    print('Unified feed, skipping alignment')

print('Resampling DOM to 1m...')
con = duckdb.connect()
con.execute("SET TIME ZONE 'UTC'")
dom_source = df_tick if is_unified_feed else df_dom
con.register('df_dom_temp', dom_source)
num_cols = dom_source.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c not in GARBAGE_COLUMNS and c not in ('ts', 'symbol')]
avg_exprs = [f"AVG({col}) AS {col}" for col in num_cols if col != 'ltp']
dom_query = f"""SELECT date_trunc('minute', ts) AS ts, FIRST(symbol) AS symbol, AVG(ltp) AS ltp, LAST(ltp) AS close_price, {', '.join(avg_exprs)} FROM df_dom_temp GROUP BY 1 ORDER BY ts"""
dom_1m = con.execute(dom_query).df()
print(f'dom_1m: {len(dom_1m)} rows')

print('Aggregating trades...')
con.register('df_trades_temp', df_trades)
print(f'Registered df_trades_temp, columns: {df_trades.columns.tolist()}')
cols_check = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'df_trades_temp'").fetchall()]
print(f'DuckDB sees: {cols_check}')

median_size = df_trades['trade_qty'].median() if 'trade_qty' in df_trades.columns else 1.0
large_threshold = 10 * median_size

trades_query = f"""
WITH trades_tagged AS (
    SELECT
        ts, symbol, trade_qty, trade_price,
        CASE WHEN direction = 'BUY' THEN trade_qty ELSE 0 END AS buy_qty,
        CASE WHEN direction = 'SELL' THEN trade_qty ELSE 0 END AS sell_qty,
        trade_price * trade_qty AS trade_value,
        CASE WHEN trade_qty > {large_threshold} THEN 1 ELSE 0 END AS is_large
    FROM df_trades_temp
)
SELECT
    date_trunc('minute', ts) AS ts,
    SUM(buy_qty) AS buy_vol_1m, SUM(sell_qty) AS sell_vol_1m,
    SUM(trade_qty) AS volume_1m, COUNT(trade_qty) AS trade_count_1m,
    SUM(is_large) AS large_trade_count_1m, SUM(trade_value) AS trade_value_1m,
    CASE WHEN SUM(trade_qty) > 0 THEN SUM(trade_value) / SUM(trade_qty) ELSE 0.0 END AS vwap_1m
FROM trades_tagged GROUP BY 1
"""
trades_1m_raw = con.execute(trades_query).df()
print(f'trades_1m: {len(trades_1m_raw)} rows')
print('SUCCESS!')
