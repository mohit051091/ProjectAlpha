import pandas as pd
import numpy as np
from collections import deque
from typing import Tuple
from utils.logger import setup_logger
from features.tick_features import TickFeaturesCalculator
from features.alignment import EventAlignmentEngine, WindowValidationEngine
from utils.constants import FEATURE_RANGES, ALIGNMENT_CONFIG, GARBAGE_COLUMNS

logger = setup_logger(__name__)

class FeatureFactory:
    """Manages feature computation, synchronization, and validation"""
    
    @staticmethod
    def build_feature_matrix(
        df_dom: pd.DataFrame,
        df_tick: pd.DataFrame = None,
        df_trades: pd.DataFrame = None,
        tolerance_s: float = None,
    ) -> pd.DataFrame:
        if len(df_dom) == 0:
            logger.error("Empty DOM dataframe provided.")
            return pd.DataFrame()

        if df_tick is None or len(df_tick) == 0:
            logger.error("Tick data is required for feature generation. Skipping feature build.")
            return pd.DataFrame()

        for df in (df_dom, df_tick):
            if 'ts' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['ts']):
                    df['ts'] = pd.to_datetime(df['ts'], utc=True)
                if df['ts'].dt.tz is None:
                    df['ts'] = df['ts'].dt.tz_localize('UTC')
                else:
                    df['ts'] = df['ts'].dt.tz_convert('UTC')

        symbol = df_dom['symbol'].iloc[0]
        logger.info(f"Building feature matrix for {symbol}...")

        tolerance_s = tolerance_s if tolerance_s is not None else ALIGNMENT_CONFIG['tolerance_ms'] / 1000.0

        tick_bqty = sorted([c for c in df_tick.columns if c.startswith('bqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))
        tick_aqty = sorted([c for c in df_tick.columns if c.startswith('aqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))
        is_unified_feed = len(tick_bqty) >= 5 and len(tick_aqty) >= 5

        if is_unified_feed:
            logger.info("Detected unified marketfeed (DOM columns inside tick data). Skipping alignment.")
            valid_windows = None
        else:
            logger.info(f"Aligning Tick + DOM events with {tolerance_s:.3f}s tolerance...")
            alignment_result = EventAlignmentEngine.align_tick_dom(df_tick, df_dom, tolerance_s=tolerance_s, window='1min')
            window_metrics = alignment_result['window_metrics']

            if window_metrics.empty:
                logger.error("No alignment windows could be generated. Skipping feature build.")
                return pd.DataFrame()

            overlap_start = max(df_tick['ts'].min(), df_dom['ts'].min())
            overlap_end = min(df_tick['ts'].max(), df_dom['ts'].max())
            if overlap_start > overlap_end:
                logger.error("No overlapping Tick + DOM coverage period found. Skipping feature build.")
                return pd.DataFrame()

            eligible_start = overlap_start.floor('1min')
            eligible_end = overlap_end.floor('1min')
            eligible_windows = pd.date_range(
                start=eligible_start,
                end=eligible_end,
                freq='1min',
                tz='UTC',
            )
            window_metrics = window_metrics[window_metrics['window_start'].isin(eligible_windows)].copy()
            logger.info(
                f"Filtering validation to {len(window_metrics):,} overlap 1-minute windows "
                f"from {eligible_start} to {eligible_end}."
            )

            if window_metrics.empty:
                logger.error("No eligible overlap windows remain after filtering. Skipping feature build.")
                return pd.DataFrame()

            window_metrics = WindowValidationEngine.compute_scores(window_metrics)
            valid_windows = window_metrics[window_metrics['window_valid']].copy()

            if valid_windows.empty:
                logger.error("No valid 1-minute windows passed alignment validation. Skipping feature build.")
                return pd.DataFrame()

            logger.info(
                f"{len(valid_windows):,} valid windows selected out of {len(window_metrics):,} total windows."
            )

        import duckdb
        con = duckdb.connect()
        con.execute("SET TIME ZONE 'UTC'")

        dom_source = df_tick if is_unified_feed else df_dom
        con.register('df_dom_temp', dom_source)
        
        num_cols = dom_source.select_dtypes(include='number').columns.tolist()
        num_cols = [c for c in num_cols if c not in GARBAGE_COLUMNS and c not in ('ts', 'symbol')]
        
        avg_exprs = [f"AVG({col}) AS {col}" for col in num_cols if col != 'ltp']
        
        dom_query = f"""
        SELECT 
            date_trunc('minute', ts) AS ts,
            FIRST(symbol) AS symbol,
            FIRST(ltp) AS open_price,
            AVG(ltp) AS ltp,
            LAST(ltp) AS close_price,
            {', '.join(avg_exprs)}
        FROM df_dom_temp
        GROUP BY 1
        ORDER BY ts
        """
        dom_1m = con.execute(dom_query).df()
        if len(dom_1m) > 0 and dom_1m['ts'].dt.tz is None:
            dom_1m['ts'] = dom_1m['ts'].dt.tz_localize('UTC')

        avail_bqty = sorted([c for c in dom_1m.columns if c.startswith('bqty') and c[4:].isdigit()],
                            key=lambda c: int(c[4:]))
        avail_aqty = sorted([c for c in dom_1m.columns if c.startswith('aqty') and c[4:].isdigit()],
                            key=lambda c: int(c[4:]))
        b_depth = min(len(avail_bqty), 5)
        a_depth = min(len(avail_aqty), 5)
        b5_cols = avail_bqty[:b_depth]
        a5_cols = avail_aqty[:a_depth]
        dom_1m['total_bid_qty'] = dom_1m[b5_cols].sum(axis=1) if b5_cols else 0.0
        dom_1m['total_ask_qty'] = dom_1m[a5_cols].sum(axis=1) if a5_cols else 0.0

        dom_1m['imbalance_top5'] = FeatureFactory._compute_imbalance(dom_1m, levels=5)
        dom_1m['spread'] = dom_1m['ask1'] - dom_1m['bid1']
        dom_1m['depth_drop_bid'] = dom_1m['total_bid_qty'].diff().fillna(0.0)
        dom_1m['depth_drop_ask'] = dom_1m['total_ask_qty'].diff().fillna(0.0)

        dom_1m['bid_replenishment_rate'] = FeatureFactory._compute_bid_replenishment_rate(dom_1m)

        if valid_windows is not None:
            dom_1m = dom_1m[dom_1m['ts'].isin(valid_windows['window_start'])].copy()
        if dom_1m.empty:
            logger.error("No DOM 1-minute windows available after alignment filtering.")
            return pd.DataFrame()

        if df_trades is None:
            logger.info("Inferring trades from DOM snapshots...")
            from features.trade_inference import TradeInferenceEngine
            df_trades = TradeInferenceEngine.infer_trades(df_dom)

        if len(df_trades) == 0:
            logger.warning("No trades found after inference. Creating dummy trade flow features with default values.")
            features_df = dom_1m.copy()
            features_df['buy_vol_1m'] = 0
            features_df['sell_vol_1m'] = 0
            features_df['volume_1m'] = 0
            features_df['trade_count_1m'] = 0
            features_df['large_trade_count_1m'] = 0
            features_df['delta_1m'] = 0.0
            features_df['delta_5m'] = 0.0
            features_df['volume_burst'] = 0.0
            features_df['aggressor_ratio'] = 0.5
            features_df['trade_count_burst'] = 0.0
            features_df['large_trade_ratio'] = 0.0
            features_df['iceberg_score'] = 0.0
            features_df['order_cancel_rate'] = FeatureFactory._compute_order_cancel_rate(features_df)
            features_df['bid_replenishment_rate'] = features_df['bid_replenishment_rate'].fillna(0.0)
        else:
            logger.info("Tagging large trades via expanding running median (no lookahead)...")
            df_trades = df_trades.sort_values('ts').reset_index(drop=True)
            qty_vals = df_trades['trade_qty'].values.astype(np.int64)
            n_trades = len(qty_vals)
            qtys = deque(maxlen=10000)
            threshold = float('inf')
            is_large_arr = np.zeros(n_trades, dtype=np.int64)
            for idx in range(n_trades):
                qty = int(qty_vals[idx])
                qtys.append(qty)
                n = len(qtys)
                if n >= 3 and (n <= 50 and n % 5 == 0 or n > 50 and n % 50 == 0):
                    threshold = 10.0 * float(np.median(np.fromiter(qtys, dtype=np.int64, count=n)))
                if threshold != float('inf') and qty > threshold:
                    is_large_arr[idx] = 1
            df_trades['is_large'] = is_large_arr
            median_size = float(np.median(qty_vals))
            con.register('df_trades_temp', df_trades)
            
            trades_query = """
            SELECT 
                date_trunc('minute', ts) AS ts,
                SUM(CASE WHEN direction = 'BUY' THEN trade_qty ELSE 0 END) AS buy_vol_1m,
                SUM(CASE WHEN direction = 'SELL' THEN trade_qty ELSE 0 END) AS sell_vol_1m,
                SUM(trade_qty) AS volume_1m,
                COUNT(trade_qty) AS trade_count_1m,
                SUM(is_large) AS large_trade_count_1m,
                SUM(trade_price * trade_qty) AS trade_value_1m,
                CASE 
                    WHEN SUM(trade_qty) > 0 THEN SUM(trade_price * trade_qty) / SUM(trade_qty)
                    ELSE 0.0
                END AS vwap_1m
            FROM df_trades_temp
            GROUP BY 1
            """
            trades_1m_raw = con.execute(trades_query).df()
            if len(trades_1m_raw) > 0 and trades_1m_raw['ts'].dt.tz is None:
                trades_1m_raw['ts'] = trades_1m_raw['ts'].dt.tz_localize('UTC')

            logger.info("Merging DOM and trades onto synchronized time grid using DuckDB...")
            con.register('df_dom_1m_duck', dom_1m)
            con.register('df_trades_1m_duck', trades_1m_raw)
            
            trade_cols = ['buy_vol_1m', 'sell_vol_1m', 'volume_1m', 'trade_count_1m', 'large_trade_count_1m']
            select_fields = ["d.*"]
            for col in trade_cols:
                select_fields.append(f"COALESCE(t.{col}, 0)::BIGINT AS {col}")
            select_fields.append("COALESCE(t.vwap_1m, 0.0)::DOUBLE AS vwap_1m")
            
            merge_query = f"""
            SELECT {', '.join(select_fields)}
            FROM df_dom_1m_duck d
            LEFT JOIN df_trades_1m_duck t ON d.ts = t.ts
            ORDER BY d.ts
            """
            merged = con.execute(merge_query).df()
            if len(merged) > 0 and merged['ts'].dt.tz is None:
                merged['ts'] = merged['ts'].dt.tz_localize('UTC')

            merged['iceberg_score'] = FeatureFactory._compute_iceberg_score(merged)
            merged['order_cancel_rate'] = FeatureFactory._compute_order_cancel_rate(merged)
            merged['bid_replenishment_rate'] = FeatureFactory._compute_bid_replenishment_rate(merged)

            logger.info("Computing Stage 1 & 2 features...")
            features_df = TickFeaturesCalculator.compute_features(merged, median_size)

        if valid_windows is not None:
            features_df = features_df[features_df['ts'].isin(valid_windows['window_start'])].copy()

        FeatureFactory.validate_feature_ranges(features_df)

        return features_df
        
    @staticmethod
    def _compute_imbalance(df: pd.DataFrame, levels: int) -> pd.Series:
        avail_bid = sorted([c for c in df.columns if c.startswith('bqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))[:levels]
        avail_ask = sorted([c for c in df.columns if c.startswith('aqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))[:levels]
        if not avail_bid or not avail_ask:
            return pd.Series(0.0, index=df.index)
        bid_sum = df[avail_bid].sum(axis=1)
        ask_sum = df[avail_ask].sum(axis=1)
        total = bid_sum + ask_sum
        return np.where(total != 0, (bid_sum - ask_sum) / total, 0.0)

    @staticmethod
    def _compute_iceberg_score(df: pd.DataFrame, levels: int = 5) -> pd.Series:
        avail_bid = sorted([c for c in df.columns if c.startswith('bqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))[:levels]
        avail_ask = sorted([c for c in df.columns if c.startswith('aqty') and c[4:].isdigit()],
                           key=lambda c: int(c[4:]))[:levels]
        if not avail_bid or not avail_ask:
            return pd.Series(0.0, index=df.index)
        displayed_qty = df[avail_bid + avail_ask].sum(axis=1)
        return np.where(
            displayed_qty > 0,
            np.minimum(100.0, df['volume_1m'] / displayed_qty * 100.0),
            0.0,
        )

    @staticmethod
    def _compute_order_cancel_rate(df: pd.DataFrame) -> pd.Series:
        depth_change = df['total_bid_qty'].diff().fillna(0.0) + df['total_ask_qty'].diff().fillna(0.0)
        placed_qty = np.where(depth_change > 0, depth_change, 0.0)
        removed_qty = np.where(depth_change < 0, -depth_change, 0.0)
        cancelled_qty = np.maximum(0.0, removed_qty - df['volume_1m'].fillna(0.0))
        return np.where(
            placed_qty > 0,
            np.minimum(1.0, cancelled_qty / placed_qty),
            0.0,
        )

    @staticmethod
    def _compute_bid_replenishment_rate(df: pd.DataFrame) -> pd.Series:
        prev_bid_qty = df['total_bid_qty'].shift(1).fillna(0.0)
        bid_qty_change = df['total_bid_qty'] - prev_bid_qty
        return np.where(
            (df['bid1'] >= df['bid1'].shift(1)) & (prev_bid_qty > 0),
            np.minimum(1.0, np.maximum(0.0, bid_qty_change / prev_bid_qty)),
            0.0,
        )

    @staticmethod
    def validate_feature_ranges(df: pd.DataFrame) -> bool:
        passed = True
        logger.info("Running quality gates on computed features...")
        
        nans = df.isnull().sum()
        nan_cols = nans[nans > 0]
        if len(nan_cols) > 0:
            logger.warning(f"Found NaN values in feature columns:\n{nan_cols}")
            passed = False
        else:
            logger.info("Zero NaN values in feature matrix.")
            
        for feature, (min_val, max_val) in FEATURE_RANGES.items():
            if feature in df.columns:
                actual_min = df[feature].min()
                actual_max = df[feature].max()
                
                if actual_min < min_val or actual_max > max_val:
                    logger.warning(
                        f"Outlier detected in feature '{feature}': "
                        f"Expected range ({min_val}, {max_val}), got actual range ({actual_min:.4f}, {actual_max:.4f})"
                    )
                    passed = False
                    
        if passed:
            logger.info("[OK] All feature quality gates passed successfully.")
        return passed
