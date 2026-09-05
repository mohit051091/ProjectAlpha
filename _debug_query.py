"""Debug the trades query specifically."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import duckdb

slug = '360ONE_25_Jun_26'
df_trades = pd.read_parquet(f'02_processed/inferred_trades_{slug}.parquet')
print('trade_price exists:', 'trade_price' in df_trades.columns)
print('df_trades shape:', df_trades.shape)

con = duckdb.connect()
con.execute("SET TIME ZONE 'UTC'")
con.register('df_trades_temp', df_trades)
cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'df_trades_temp'").fetchall()]
print('registered cols:', cols)

tq = """
SELECT ts, symbol, trade_qty, trade_price
FROM df_trades_temp
LIMIT 3
"""
try:
    r = con.execute(tq).df()
    print('query worked:', r)
except Exception as e:
    print('query failed:', e)
