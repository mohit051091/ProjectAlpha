"""
UNIT TESTS: tests/test_walk_forward.py
PURPOSE: Validate the correct behavior of the walk-forward split engine, purging logic, and trend filters.
HOW TO RUN:
    python -m unittest tests/test_walk_forward.py
"""

import sys
import unittest
import pandas as pd
import numpy as np
import importlib
from pathlib import Path
from datetime import datetime, timedelta

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import module starting with a digit using importlib
wf = importlib.import_module("scripts.70_walk_forward_validation")
get_train_val_split_purged = wf.get_train_val_split_purged
simulate_trades = wf.simulate_trades


class TestWalkForward(unittest.TestCase):
    """Test suite for walk-forward validation helper functions"""
    
    def setUp(self):
        """Prepare base dataframes and timestamps for testing"""
        # Create a sample dataframe with 2 days
        self.ts1 = pd.Timestamp("2026-06-01 10:00:00+00:00")
        self.ts2 = pd.Timestamp("2026-06-01 15:29:00+00:00")
        self.ts3 = pd.Timestamp("2026-06-02 09:15:00+00:00")
        self.ts4 = pd.Timestamp("2026-06-02 10:00:00+00:00")
        
        # 4 rows: 2 on day 1, 2 on day 2
        self.df_test = pd.DataFrame({
            "ts": [self.ts1, self.ts2, self.ts3, self.ts4],
            "date": ["01_Jun_26", "01_Jun_26", "02_Jun_26", "02_Jun_26"],
            "symbol": ["TEST", "TEST", "TEST", "TEST"],
            "ltp": [100.0, 101.5, 102.0, 103.0],
            "vwap_1m": [100.0, 101.5, 102.0, 103.0],
            "volume_1m": [100, 100, 100, 100],
            "pred_rf": ["NO_TRADE", "LONG", "SHORT", "NO_TRADE"],
            "label_60m_1pct": ["LONG", "NO_TRADE", "NO_TRADE", "NO_TRADE"]
        })
        
        # Add all configuration features
        for f in wf.FEATURES:
            self.df_test[f] = 1.0

    def test_purging_logic(self):
        """Verify that training samples close to validation start are purged"""
        # Set up a split where validation date is June 2.
        # Val start is 2026-06-02 09:15:00.
        # A train sample at 2026-06-01 15:29:00 with 60-min horizon finishes at 16:29.
        # This is < val start, so it should NOT be purged.
        df_train, df_val = get_train_val_split_purged(
            self.df_test, 
            train_dates=["01_Jun_26"], 
            val_date="02_Jun_26", 
            horizon_min=60
        )
        self.assertEqual(len(df_train), 2)
        self.assertEqual(len(df_val), 2)
        
        # Create a sample where train and val overlap on the same day to verify the purge condition.
        # Train timestamp = 10:00. Val start = 10:30.
        # Horizon = 60 mins. Train ends at 11:00.
        # 11:00 is >= 10:30 (val_start), so it must be purged.
        df_custom = pd.DataFrame({
            "ts": [pd.Timestamp("2026-06-01 10:00:00+00:00"), pd.Timestamp("2026-06-01 10:30:00+00:00")],
            "date": ["01_Jun_26", "02_Jun_26"], # Treat 10:30 as val day
            "ltp": [100.0, 101.0],
            "label_60m_1pct": ["LONG", "NO_TRADE"]
        })
        for f in wf.FEATURES:
            df_custom[f] = 1.0
            
        df_train_purged, df_val_purged = get_train_val_split_purged(
            df_custom,
            train_dates=["01_Jun_26"],
            val_date="02_Jun_26",
            horizon_min=60
        )
        # Train row must be purged since 10:00 + 60m = 11:00 >= 10:30 (val_start)
        self.assertEqual(len(df_train_purged), 0)
        self.assertEqual(len(df_val_purged), 1)

    def test_daily_open_filter_long(self):
        """Verify that the daily open price filter permits LONG only when LTP > open"""
        # Create a day sequence where open = 100.0
        # At t=9, we issue a LONG alert.
        # Case A: Price is 102.0 (above open). Trade should trigger.
        df_long_ok = pd.DataFrame({
            "ts": [pd.Timestamp("2026-06-01 09:15:00+00:00") + timedelta(minutes=i) for i in range(15)],
            "ltp": [100.0] * 10 + [102.0] * 5,
            "vwap_1m": [100.0] * 15,
            "volume_1m": [100] * 15,
            "pred_rf": ["NO_TRADE"] * 9 + ["LONG"] + ["NO_TRADE"] * 5
        })
        trades = simulate_trades(df_long_ok, "pred_rf", "RF", filter_type="open")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["direction"], "LONG")
        
        # Case B: Price is 98.0 (below open). Trade should be blocked by trend filter.
        df_long_blocked = pd.DataFrame({
            "ts": [pd.Timestamp("2026-06-01 09:15:00+00:00") + timedelta(minutes=i) for i in range(15)],
            "ltp": [100.0] * 10 + [98.0] * 5,
            "vwap_1m": [100.0] * 15,
            "volume_1m": [100] * 15,
            "pred_rf": ["NO_TRADE"] * 9 + ["LONG"] + ["NO_TRADE"] * 5
        })
        trades_blocked = simulate_trades(df_long_blocked, "pred_rf", "RF", filter_type="open")
        self.assertEqual(len(trades_blocked), 0)

    def test_spread_segmentation(self):
        """Verify that symbols are correctly segmented into low and high spread groups"""
        df_spreads = pd.DataFrame({
            "slug": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "spread": [0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35]
        })
        spread_profile = df_spreads.groupby("slug")["spread"].mean()
        median_spread = spread_profile.median()
        
        low_spread_slugs = spread_profile[spread_profile <= median_spread].index.tolist()
        high_spread_slugs = spread_profile[spread_profile > median_spread].index.tolist()
        
        self.assertEqual(median_spread, 0.20)
        self.assertIn("A", low_spread_slugs)
        self.assertIn("B", low_spread_slugs)
        self.assertIn("C", high_spread_slugs)
        self.assertIn("D", high_spread_slugs)



if __name__ == "__main__":
    unittest.main()
