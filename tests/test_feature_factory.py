"""
UNIT TESTS: tests/test_feature_factory.py
PURPOSE: Validate calculations of Stage 1 & 2 features, 1-minute aggregations,
         and quality gate checks.
HOW TO RUN:
    python -m unittest tests/test_feature_factory.py
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.tick_features import TickFeaturesCalculator
from features.feature_factory import FeatureFactory

class TestFeatureFactory(unittest.TestCase):
    """Test suite for FeatureFactory and TickFeaturesCalculator"""
    
    def setUp(self):
        """Prepare base dataframes for tests"""
        self.symbol = 'TEST'
        
        # Create dummy 1-minute trade aggregate data
        self.ts_base = pd.Timestamp("2026-06-01 10:00:00", tz='UTC')
        self.timestamps = [self.ts_base + pd.Timedelta(minutes=i) for i in range(25)]
        
        # 1-minute aggregates template
        self.df_1m_template = pd.DataFrame({
            'ts': self.timestamps,
            'symbol': [self.symbol] * 25,
            'buy_vol_1m': [100] * 25,
            'sell_vol_1m': [50] * 25,
            'volume_1m': [150] * 25,
            'trade_count_1m': [10] * 25,
            'large_trade_count_1m': [1] * 25,
            'ltp': [100.0] * 25,
            'vwap_1m': [100.0] * 25,
        })
        
    def test_stage1_features(self):
        """Verify Stage 1 delta_1m and delta_5m calculations"""
        df_feat = TickFeaturesCalculator.compute_features(self.df_1m_template, median_size=10.0)
        
        # delta_1m should be buy - sell = 100 - 50 = 50
        self.assertTrue((df_feat['delta_1m'] == 50.0).all())
        
        # delta_5m: rolling 5-period buy sum - rolling 5-period sell sum
        # At index 0, it should be 100 - 50 = 50 (since lookback window is 1 period due to min_periods=1)
        self.assertEqual(df_feat.loc[0, 'delta_5m'], 50.0)
        # At index 4 and beyond, it should be 500 - 250 = 250
        self.assertEqual(df_feat.loc[4, 'delta_5m'], 250.0)
        self.assertEqual(df_feat.loc[24, 'delta_5m'], 250.0)
        
    def test_stage2_features(self):
        """Verify Stage 2 volume burst, aggressor ratio, trade count burst, and large trade ratio"""
        df_feat = TickFeaturesCalculator.compute_features(self.df_1m_template, median_size=10.0)
        
        # volume_burst: 150 / 150 (since all are 150, rolling mean is 150) = 1.0
        self.assertTrue((df_feat['volume_burst'] == 1.0).all())
        
        # aggressor_ratio: buy / total = 100 / 150 = 2/3 ≈ 0.6667
        np.testing.assert_allclose(df_feat['aggressor_ratio'].values, 2.0/3.0)
        
        # trade_count_burst: 10 / 10 = 1.0
        self.assertTrue((df_feat['trade_count_burst'] == 1.0).all())
        
        # large_trade_ratio: large / total = 1 / 10 = 0.1
        self.assertTrue((df_feat['large_trade_ratio'] == 0.1).all())

    def test_division_by_zero_safety(self):
        """Verify division-by-zero protection when trade volume/counts are zero.

        BEHAVIOUR CLARIFICATION:
        - volume_burst = current_vol / rolling_avg_vol.
          When current_vol=0 and rolling_avg>0 → burst=0.0  (no burst, correct).
          The default 1.0 only fires when rolling_avg is *itself* 0 (first bar edge case).
        - aggressor_ratio defaults to 0.5 when volume_1m=0 (neutral/balanced).
        - trade_count_burst = current_count / rolling_avg_count.
          When current_count=0 → burst=0.0 (correct).
        - large_trade_ratio defaults to 0.0 when trade_count_1m=0.
        """
        df_zero = self.df_1m_template.copy()
        df_zero.loc[5, ['buy_vol_1m', 'sell_vol_1m', 'volume_1m', 'trade_count_1m', 'large_trade_count_1m']] = 0

        df_feat = TickFeaturesCalculator.compute_features(df_zero, median_size=10.0)

        # volume_burst: 0 / rolling_avg(150) = 0.0  (no volume spike)
        self.assertEqual(df_feat.loc[5, 'volume_burst'], 0.0)
        # aggressor_ratio: volume=0 → default 0.5 (neutral)
        self.assertEqual(df_feat.loc[5, 'aggressor_ratio'], 0.5)
        # trade_count_burst: 0 / rolling_avg(10) = 0.0  (no trades)
        self.assertEqual(df_feat.loc[5, 'trade_count_burst'], 0.0)
        # large_trade_ratio: 0 / 0 → default 0.0
        self.assertEqual(df_feat.loc[5, 'large_trade_ratio'], 0.0)
        
    def test_validation_gates_nan_checks(self):
        """Validation gate should fail if NaNs are present in features"""
        df_nan = self.df_1m_template.copy()
        df_nan.loc[5, 'buy_vol_1m'] = np.nan
        df_feat = TickFeaturesCalculator.compute_features(df_nan, median_size=10.0)
        
        # Check range validation fails due to NaNs
        passed = FeatureFactory.validate_feature_ranges(df_feat)
        self.assertFalse(passed)

    def test_validation_gates_outliers(self):
        """Validation gate should log warning if a feature exceeds physical ranges"""
        df_outlier = self.df_1m_template.copy()
        df_feat = TickFeaturesCalculator.compute_features(df_outlier, median_size=10.0)
        
        # Inject an outlier for aggressor_ratio (range is 0 to 1)
        df_feat.loc[5, 'aggressor_ratio'] = 5.0
        
        passed = FeatureFactory.validate_feature_ranges(df_feat)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
