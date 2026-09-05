"""
MODULE: features/trade_inference.py
PURPOSE: Reconstruct trade events from successive order book (DOM) snapshots.
INPUT: Cleaned DOM snapshot DataFrame (with ts, ltp, bid1-bid20, bqty1-bqty20, ask1-ask20, aqty1-aqty20)
OUTPUT: DataFrame of inferred trades (ts, symbol, trade_price, trade_qty, direction)

WHY IT IS NEEDED:
Our tick-by-tick trade dataset is corrupted. To compute trade-flow features (such as delta_1m, delta_5m,
volume_burst, aggressor_ratio), we must infer when trades occurred and whether they were aggressive
buys or aggressive sells by looking at order book level crossings and quantity reductions.

HOW IT WORKS:
1. If the best ask price moves up, we infer that all ask levels below the new best ask price were consumed by aggressive BUY trades.
2. If the best ask price remains the same but its quantity decreases, we infer a BUY trade for the difference.
3. If the best bid price moves down, we infer that all bid levels above the new best bid price were consumed by aggressive SELL trades.
4. If the best bid price remains the same but its quantity decreases, we infer a SELL trade for the difference.
5. If the last traded price (LTP) changes, but no trade was captured by order book changes, we record a fallback trade at the new LTP.

AUTHOR: Senior Quant Researcher & Lead Developer
DATE: June 6, 2026
"""

import numpy as np
import pandas as pd
import duckdb
from typing import Dict, List, Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TradeInferenceEngine:
    """Infers trade executions from Depth of Market (DOM) state changes"""
    
    @staticmethod
    def _detect_depth(df: pd.DataFrame) -> int:
        """Detect available depth levels from DataFrame columns."""
        max_depth = 0
        for c in df.columns:
            if c.startswith("bid") and c[3:].isdigit():
                d = int(c[3:])
                if d > max_depth:
                    max_depth = d
        return max_depth

    @staticmethod
    def infer_trades(df_dom: pd.DataFrame) -> pd.DataFrame:
        """
        Infer trades from order book changes using DuckDB window functions
        
        Parameters
        ----------
        df_dom : pd.DataFrame
            Cleaned DOM DataFrame sorted by timestamp
            
        Returns
        -------
        pd.DataFrame
            DataFrame of inferred trades
        """
        if len(df_dom) < 2:
            logger.warning("Dataframe too small to infer trades.")
            return pd.DataFrame(columns=['ts', 'symbol', 'trade_price', 'trade_qty', 'direction'])
            
        max_depth = TradeInferenceEngine._detect_depth(df_dom)
        logger.info(f"Starting trade inference on {len(df_dom):,} order book snapshots (depth={max_depth}) using DuckDB...")
        
        symbol = df_dom['symbol'].iloc[0]
        
        # Connect to duckdb
        con = duckdb.connect()
        con.execute("SET TIME ZONE 'UTC'")
        con.register('df_dom_temp', df_dom)
        
        # Construct lag columns for levels 1..max_depth
        depth_cols = ["bid", "bqty", "ask", "aqty"]
        lag_cols = [
            "ts", "symbol", "ltp", "LAG(ltp) OVER (ORDER BY ts) AS prev_ltp",
            "bid1", "LAG(bid1) OVER (ORDER BY ts) AS prev_bid1",
            "bqty1", "LAG(bqty1) OVER (ORDER BY ts) AS prev_bqty1",
            "ask1", "LAG(ask1) OVER (ORDER BY ts) AS prev_ask1",
            "aqty1", "LAG(aqty1) OVER (ORDER BY ts) AS prev_aqty1"
        ]
        
        for i in range(1, max_depth + 1):
            lag_cols.append(f"bid{i}")
            lag_cols.append(f"LAG(bid{i}) OVER (ORDER BY ts) AS prev_bid_{i}")
            lag_cols.append(f"bqty{i}")
            lag_cols.append(f"LAG(bqty{i}) OVER (ORDER BY ts) AS prev_bqty_{i}")
            lag_cols.append(f"ask{i}")
            lag_cols.append(f"LAG(ask{i}) OVER (ORDER BY ts) AS prev_ask_{i}")
            lag_cols.append(f"aqty{i}")
            lag_cols.append(f"LAG(aqty{i}) OVER (ORDER BY ts) AS prev_aqty_{i}")
            
        # Build union query for levels 1..max_depth
        buy_clearing = []
        buy_partial = []
        sell_clearing = []
        sell_partial = []
        
        for i in range(1, max_depth + 1):
            buy_clearing.append(f"""
            SELECT ts, symbol, prev_ask_{i} AS trade_price, prev_aqty_{i} AS trade_qty, 'BUY' AS direction, 1 AS priority, {i} AS level_idx
            FROM lag_table
            WHERE prev_ask_{i} < ask1 AND prev_ask_{i} > 0 AND prev_aqty_{i} > 0 AND ask1 > prev_ask1 AND prev_ask1 > 0
            """)
            buy_partial.append(f"""
            SELECT ts, symbol, ask1 AS trade_price, (prev_aqty_{i} - aqty1) AS trade_qty, 'BUY' AS direction, 2 AS priority, {i} AS level_idx
            FROM lag_table
            WHERE prev_ask_{i} = ask1 AND ask1 > prev_ask1 AND prev_ask1 > 0 AND (prev_aqty_{i} - aqty1) > 0
            """)
            sell_clearing.append(f"""
            SELECT ts, symbol, prev_bid_{i} AS trade_price, prev_bqty_{i} AS trade_qty, 'SELL' AS direction, 1 AS priority, {i} AS level_idx
            FROM lag_table
            WHERE prev_bid_{i} > bid1 AND prev_bid_{i} > 0 AND prev_bqty_{i} > 0 AND bid1 < prev_bid1 AND prev_bid1 > 0
            """)
            sell_partial.append(f"""
            SELECT ts, symbol, bid1 AS trade_price, (prev_bqty_{i} - bqty1) AS trade_qty, 'SELL' AS direction, 2 AS priority, {i} AS level_idx
            FROM lag_table
            WHERE prev_bid_{i} = bid1 AND bid1 < prev_bid1 AND prev_bid1 > 0 AND (prev_bqty_{i} - bqty1) > 0
            """)
            
        union_parts = buy_clearing + buy_partial + sell_clearing + sell_partial
        
        # Add unchanged quantity decrease cases
        union_parts.append("""
        SELECT ts, symbol, prev_ask1 AS trade_price, (prev_aqty1 - aqty1) AS trade_qty, 'BUY' AS direction, 1 AS priority, 1 AS level_idx
        FROM lag_table
        WHERE ask1 = prev_ask1 AND prev_ask1 > 0 AND (prev_aqty1 - aqty1) > 0
        """)
        union_parts.append("""
        SELECT ts, symbol, prev_bid1 AS trade_price, (prev_bqty1 - bqty1) AS trade_qty, 'SELL' AS direction, 1 AS priority, 1 AS level_idx
        FROM lag_table
        WHERE bid1 = prev_bid1 AND prev_bid1 > 0 AND (prev_bqty1 - bqty1) > 0
        """)
        
        union_str = " UNION ALL ".join(union_parts)
        
        query = f"""
        WITH lag_table AS (
            SELECT {', '.join(lag_cols)}
            FROM df_dom_temp
        ),
        inferred_union AS (
            {union_str}
        ),
        trade_timestamps AS (
            SELECT DISTINCT ts FROM inferred_union
        ),
        fallback_trades AS (
            SELECT 
                ts,
                symbol,
                ltp AS trade_price,
                1 AS trade_qty,
                CASE 
                    WHEN prev_ask1 > 0 AND prev_bid1 > 0 THEN 
                        CASE WHEN ltp >= (prev_ask1 + prev_bid1) / 2.0 THEN 'BUY' ELSE 'SELL' END
                    ELSE 'BUY'
                END AS direction,
                3 AS priority,
                1 AS level_idx
            FROM lag_table
            WHERE ts NOT IN (SELECT ts FROM trade_timestamps) 
              AND prev_ltp IS NOT NULL 
              AND ltp <> prev_ltp 
              AND ltp > 0
        ),
        all_trades AS (
            SELECT ts, symbol, trade_price, trade_qty, direction, priority, level_idx FROM inferred_union
            UNION ALL
            SELECT ts, symbol, trade_price, trade_qty, direction, priority, level_idx FROM fallback_trades
        )
        SELECT ts, symbol, trade_price, CAST(trade_qty AS BIGINT) AS trade_qty, direction
        FROM all_trades
        ORDER BY ts, priority, level_idx
        """
        
        df_res = con.execute(query).df()
        
        # Ensure timestamp has UTC timezone set
        if len(df_res) > 0:
            if df_res['ts'].dt.tz is None:
                df_res['ts'] = df_res['ts'].dt.tz_localize('UTC')
            else:
                df_res['ts'] = df_res['ts'].dt.tz_convert('UTC')
            
        logger.info(f"Trade inference complete. Inferred {len(df_res):,} trade events.")
        return df_res


if __name__ == "__main__":
    # Example usage for testing
    import time
    import sys
    from pathlib import Path
    
    # Add root directory to python path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.logger import setup_logger
    logger = setup_logger("test_trade_inference")
    ts_now = pd.Timestamp.now(tz='UTC')
    dummy_dom = pd.DataFrame({
        'ts': [ts_now, ts_now + pd.Timedelta(microseconds=50)],
        'symbol': ['TEST', 'TEST'],
        'ltp': [100.0, 100.0],
        'total_bid_qty': [1000, 900],
        'total_ask_qty': [1000, 1000],
        'imbalance': [0.0, -0.05],
        'bid1': [99.9, 99.9],
        'bqty1': [150, 100],  # Bid quantity decreased by 50
        'ask1': [100.1, 100.1],
        'aqty1': [200, 200],
    })
    # Add dummy levels 2 to 5 to satisfy validation
    for i in range(2, 6):
        dummy_dom[f'bid{i}'] = 99.9 - (i - 1) * 0.1
        dummy_dom[f'bqty{i}'] = 100
        dummy_dom[f'ask{i}'] = 100.1 + (i - 1) * 0.1
        dummy_dom[f'aqty{i}'] = 100
        
    start_time = time.time()
    trades = TradeInferenceEngine.infer_trades(dummy_dom)
    print(f"Elapsed: {time.time() - start_time:.4f}s")
    print("\nInferred Trades:")
    print(trades)
