"""
MODULE: features/tick_features.py
PURPOSE: Compute tick/trade-flow features (Stage 1 & 2) aggregated to 1-minute bars.
INPUT: Inferred trades DataFrame (ts, symbol, trade_price, trade_qty, direction)
OUTPUT: DataFrame aggregated to 1-minute bars containing the calculated features

FEATURES:
- Stage 1:
  - delta_1m: Buy volume - Sell volume (1 min)
  - delta_5m: Buy volume - Sell volume (5 min)
- Stage 2:
  - volume_burst: Current 1m volume / rolling average volume (20 mins)
  - aggressor_ratio: Aggressive buy trades / total trades (1m window)
  - trade_count_burst: Number of trades in last 1m vs rolling average (20 mins)
  - large_trade_ratio: Trades above 10x median size / total trades

AUTHOR: Senior Quant Researcher & Lead Developer
DATE: June 6, 2026
"""

import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TickFeaturesCalculator:
    """Calculates trade flow features from inferred trade history"""
    
    @staticmethod
    def aggregate_trades_1min(df_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trade records into 1-minute bins
        
        Parameters
        ----------
        df_trades : pd.DataFrame
            Inferred trade records
            
        Returns
        -------
        pd.DataFrame
            Aggregated 1-minute trade metrics
        """
        if len(df_trades) == 0:
            logger.warning("No trades to aggregate.")
            return pd.DataFrame(columns=[
                'ts', 'symbol', 'buy_vol_1m', 'sell_vol_1m', 
                'volume_1m', 'trade_count_1m', 'large_trade_count_1m'
            ])
            
        # Ensure timestamp is datetime and set as index
        df = df_trades.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            
        symbol = df['symbol'].iloc[0]
        
        # Calculate median trade size for large trade logic
        median_size = df['trade_qty'].median()
        large_threshold = 10 * median_size
        logger.info(f"Symbol {symbol}: Median trade size = {median_size:.1f}, Large trade threshold = {large_threshold:.1f}")
        
        # Tag large trades
        df['is_large'] = df['trade_qty'] > large_threshold
        
        # Split into buys and sells
        df['buy_qty'] = np.where(df['direction'] == 'BUY', df['trade_qty'], 0)
        df['sell_qty'] = np.where(df['direction'] == 'SELL', df['trade_qty'], 0)
        df['trade_value'] = df['trade_price'] * df['trade_qty']
        
        # Resample to 1-minute bars
        # Note: resample label='left' puts the label at the start of the minute bar (standard in trading)
        resampler = df.resample('1min', on='ts')
        
        df_1m = pd.DataFrame()
        df_1m['buy_vol_1m'] = resampler['buy_qty'].sum()
        df_1m['sell_vol_1m'] = resampler['sell_qty'].sum()
        df_1m['volume_1m'] = resampler['trade_qty'].sum()
        df_1m['trade_count_1m'] = resampler['trade_qty'].count()
        df_1m['large_trade_count_1m'] = resampler['is_large'].sum()
        df_1m['trade_value_1m'] = resampler['trade_value'].sum()
        df_1m['vwap_1m'] = np.where(
            df_1m['volume_1m'] > 0,
            df_1m['trade_value_1m'] / df_1m['volume_1m'],
            0.0,
        )
        
        df_1m = df_1m.reset_index()
        df_1m.insert(1, 'symbol', symbol)
        
        return df_1m, median_size
        
    @staticmethod
    def compute_features(df_1m_trades: pd.DataFrame, median_size: float) -> pd.DataFrame:
        """
        Compute Stage 1 & 2 trade flow features from 1-minute trade aggregates
        
        Parameters
        ----------
        df_1m_trades : pd.DataFrame
            1-minute trade aggregates
        median_size : float
            Median trade size of the underlying trades (for reference/metadata)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all tick/trade flow features
        """
        if len(df_1m_trades) == 0:
            return df_1m_trades.copy()
            
        df = df_1m_trades.copy()
        
        # Ensure sorted by timestamp
        df = df.sort_values(by='ts').reset_index(drop=True)
        
        # ============================================================================
        # STAGE 1 FEATURES
        # ============================================================================
        
        # delta_1m: Buy volume - Sell volume (1 min)
        df['delta_1m'] = df['buy_vol_1m'] - df['sell_vol_1m']
        
        # delta_5m: Buy volume - Sell volume (5 min rolling sum)
        # rolling(5) sums the last 5 1-minute bins (min_periods=1 ensures no NaNs at start)
        rolling_buy_5m = df['buy_vol_1m'].rolling(window=5, min_periods=1).sum()
        rolling_sell_5m = df['sell_vol_1m'].rolling(window=5, min_periods=1).sum()
        df['delta_5m'] = rolling_buy_5m - rolling_sell_5m
        
        # ============================================================================
        # STAGE 2 FEATURES
        # ============================================================================
        
        # volume_burst: Current 1m volume / rolling average volume (20 mins)
        rolling_avg_vol_20m = df['volume_1m'].rolling(window=20, min_periods=1).mean()
        # Handle divide by zero: if rolling avg is 0, set burst to 1.0 (no burst)
        df['volume_burst'] = np.where(
            rolling_avg_vol_20m > 0,
            df['volume_1m'] / rolling_avg_vol_20m,
            1.0
        )
        
        # aggressor_ratio: Aggressive buy trades volume / total trades volume (1m window)
        # Handle divide by zero: if volume is 0, default to 0.5 (balanced)
        df['aggressor_ratio'] = np.where(
            df['volume_1m'] > 0,
            df['buy_vol_1m'] / df['volume_1m'],
            0.5
        )
        
        # trade_count_burst: Number of trades in last 1m vs rolling average (20 mins)
        rolling_avg_count_20m = df['trade_count_1m'].rolling(window=20, min_periods=1).mean()
        df['trade_count_burst'] = np.where(
            rolling_avg_count_20m > 0,
            df['trade_count_1m'] / rolling_avg_count_20m,
            1.0
        )
        
        # large_trade_ratio: Trades above 10x median size / total trades
        df['large_trade_ratio'] = np.where(
            df['trade_count_1m'] > 0,
            df['large_trade_count_1m'] / df['trade_count_1m'],
            0.0
        )
        
        # ============================================================================
        # STAGE 3 FEATURES
        # ============================================================================
        df['vwap_distance'] = np.where(
            df['vwap_1m'] > 0,
            (df['ltp'] - df['vwap_1m']) / df['vwap_1m'],
            0.0
        )
        
        # Compute volatility_5m on ltp instead of vwap_1m (Task 0.3)
        df['volatility_5m'] = df['ltp'].rolling(window=5, min_periods=1).std().fillna(0.0)
        df['price_acceleration'] = df['ltp'].diff().diff().fillna(0.0)
        
        # --- Order Flow Absorption Features ---
        df['return_1m'] = ((df['ltp'] - df['ltp'].shift(1)) / df['ltp'].shift(1).replace(0, np.nan)).fillna(0.0)
        df['return_5m'] = ((df['ltp'] - df['ltp'].shift(5)) / df['ltp'].shift(5).replace(0, np.nan)).fillna(0.0)
        
        df['delta_1m_ratio'] = (df['delta_1m'] / (df['volume_1m'] + 1e-8)).fillna(0.0)
        rolling_vol_5m = df['volume_1m'].rolling(window=5, min_periods=1).sum()
        df['delta_5m_ratio'] = (df['delta_5m'] / (rolling_vol_5m + 1e-8)).fillna(0.0)
        
        df['absorption_buyer_1m'] = np.where((df['delta_1m_ratio'] < 0) & (df['return_1m'] > 0), -df['delta_1m_ratio'] * df['return_1m'], 0.0)
        df['absorption_buyer_5m'] = np.where((df['delta_5m_ratio'] < 0) & (df['return_5m'] > 0), -df['delta_5m_ratio'] * df['return_5m'], 0.0)
        
        df['absorption_seller_1m'] = np.where((df['delta_1m_ratio'] > 0) & (df['return_1m'] < 0), df['delta_1m_ratio'] * -df['return_1m'], 0.0)
        df['absorption_seller_5m'] = np.where((df['delta_5m_ratio'] > 0) & (df['return_5m'] < 0), df['delta_5m_ratio'] * -df['return_5m'], 0.0)
        
        logger.info(f"Computed {len(df)} 1-minute tick/trade flow features.")
        return df
