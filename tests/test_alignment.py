"""
UNIT TESTS: tests/test_alignment.py
PURPOSE: Validate Tick + DOM event-time alignment and window validation.
HOW TO RUN:
    python -m unittest tests/test_alignment.py
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.alignment import EventAlignmentEngine, WindowValidationEngine
from features.feature_factory import FeatureFactory

class TestEventAlignmentEngine(unittest.TestCase):
    def setUp(self):
        self.symbol = 'TEST'
        self.base_ts = pd.Timestamp('2026-06-01 10:00:00', tz='UTC')

        self.df_tick = pd.DataFrame({
            'ts': [
                self.base_ts + pd.Timedelta(milliseconds=100),
                self.base_ts + pd.Timedelta(milliseconds=450),
                self.base_ts + pd.Timedelta(milliseconds=900),
            ],
            'symbol': [self.symbol] * 3,
            'trade_qty': [100, 150, 120],
            'trade_price': [100.5, 100.7, 100.8],
        })

        self.df_dom = pd.DataFrame({
            'ts': [
                self.base_ts + pd.Timedelta(milliseconds=80),
                self.base_ts + pd.Timedelta(milliseconds=470),
                self.base_ts + pd.Timedelta(milliseconds=915),
            ],
            'symbol': [self.symbol] * 3,
            'ltp': [100.5, 100.7, 100.8],
            'total_bid_qty': [1000, 980, 970],
            'total_ask_qty': [1200, 1180, 1170],
            'imbalance': [0.1, 0.08, 0.07],
        })
        for i in range(1, 21):
            self.df_dom[f'bid{i}'] = 100.5 - i * 0.01
            self.df_dom[f'bqty{i}'] = 100
            self.df_dom[f'ask{i}'] = 100.7 + i * 0.01
            self.df_dom[f'aqty{i}'] = 100

    def test_alignment_within_tolerance(self):
        result = EventAlignmentEngine.align_tick_dom(self.df_tick, self.df_dom, tolerance_s=1.0, window='1min')

        self.assertEqual(result['global_metrics']['total_tick_count'], 3)
        self.assertEqual(result['global_metrics']['total_dom_count'], 3)
        self.assertEqual(result['global_metrics']['matched_tick_count'], 3)
        self.assertGreaterEqual(result['global_metrics']['matched_dom_count'], 3)
        self.assertAlmostEqual(result['global_metrics']['unmatched_ratio'], 0.0)
        self.assertLessEqual(result['global_metrics']['avg_alignment_distance_ms'], 50.0)

        window_metrics = result['window_metrics']
        self.assertEqual(len(window_metrics), 1)
        self.assertEqual(window_metrics.loc[0, 'tick_count'], 3)
        self.assertEqual(window_metrics.loc[0, 'dom_count'], 3)
        self.assertEqual(window_metrics.loc[0, 'tick_matched_count'], 3)

    def test_alignment_out_of_tolerance(self):
        result = EventAlignmentEngine.align_tick_dom(self.df_tick, self.df_dom, tolerance_s=0.001, window='1min')

        self.assertEqual(result['global_metrics']['matched_tick_count'], 0)
        self.assertEqual(result['global_metrics']['matched_dom_count'], 0)
        self.assertEqual(result['global_metrics']['unmatched_ratio'], 1.0)

    def test_window_validation_scores(self):
        result = EventAlignmentEngine.align_tick_dom(self.df_tick, self.df_dom, tolerance_s=1.0, window='1min')
        window_metrics = result['window_metrics']
        scored = WindowValidationEngine.compute_scores(window_metrics)

        self.assertTrue('aqs' in scored.columns)
        self.assertTrue('dcs' in scored.columns)
        self.assertTrue('wvs' in scored.columns)
        self.assertTrue(scored.loc[0, 'window_valid'])

    def test_feature_factory_requires_tick(self):
        # Create a minimal feature factory scenario with valid tick and DOM data.
        features = FeatureFactory.build_feature_matrix(self.df_dom, self.df_tick)
        self.assertFalse(features.empty)
        self.assertIn('delta_1m', features.columns)

    def test_feature_factory_rejects_missing_tick(self):
        features = FeatureFactory.build_feature_matrix(self.df_dom, None)
        self.assertTrue(features.empty)

if __name__ == '__main__':
    unittest.main()
