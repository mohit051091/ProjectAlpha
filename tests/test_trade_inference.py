"""
UNIT TESTS: tests/test_trade_inference.py
PURPOSE: Validate the correct behavior of the TradeInferenceEngine under different order book scenarios.
HOW TO RUN:
    python -m unittest tests/test_trade_inference.py
"""

import sys
import unittest
import pandas as pd
from pathlib import Path

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.trade_inference import TradeInferenceEngine

class TestTradeInference(unittest.TestCase):
    """Test suite for the TradeInferenceEngine class"""
    
    def setUp(self):
        """Prepare timestamps and base order book structures for tests"""
        self.ts1 = pd.Timestamp("2026-06-01 10:00:00.000000", tz='UTC')
        self.ts2 = pd.Timestamp("2026-06-01 10:00:00.000100", tz='UTC')
        
        # Create a helper method to construct a standard template DOM snapshot
        self.base_dom = {
            'symbol': 'TEST',
            'ltp': 100.0,
            'total_bid_qty': 1000,
            'total_ask_qty': 1000,
            'imbalance': 0.0,
        }
        for i in range(1, 21):
            self.base_dom[f'bid{i}'] = 100.0 - i * 0.1
            self.base_dom[f'bqty{i}'] = 100
            self.base_dom[f'ask{i}'] = 100.0 + i * 0.1
            self.base_dom[f'aqty{i}'] = 100

    def test_no_change_no_trades(self):
        """No changes in DOM or LTP should infer 0 trades"""
        dom1 = pd.DataFrame([dict(ts=self.ts1, **self.base_dom)])
        dom2 = pd.DataFrame([dict(ts=self.ts2, **self.base_dom)])
        df = pd.concat([dom1, dom2], ignore_index=True)
        
        trades = TradeInferenceEngine.infer_trades(df)
        self.assertEqual(len(trades), 0)

    def test_ask_qty_decrease_buy_trade(self):
        """A decrease in ask1 quantity (price unchanged) should infer a BUY trade"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        
        dom2 = dict(ts=self.ts2, **self.base_dom)
        dom2['aqty1'] = 70  # Quantity decreased from 100 to 70 (-30)
        
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]['trade_qty'], 30)
        self.assertEqual(trades.iloc[0]['trade_price'], 100.1)  # ask1 price
        self.assertEqual(trades.iloc[0]['direction'], 'BUY')

    def test_bid_qty_decrease_sell_trade(self):
        """A decrease in bid1 quantity (price unchanged) should infer a SELL trade"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        
        dom2 = dict(ts=self.ts2, **self.base_dom)
        dom2['bqty1'] = 60  # Quantity decreased from 100 to 60 (-40)
        
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]['trade_qty'], 40)
        self.assertEqual(trades.iloc[0]['trade_price'], 99.9)  # bid1 price
        self.assertEqual(trades.iloc[0]['direction'], 'SELL')

    def test_ask_price_crossing_multiple_levels(self):
        """An increase in ask1 price should infer BUY trades clearing intermediate levels"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        
        dom2 = dict(ts=self.ts2, **self.base_dom)
        # ask1 shifts from 100.1 to 100.3. 
        # Levels at 100.1 (qty 100) and 100.2 (qty 100) are cleared.
        # Level 100.3 (qty 100) is now the new ask1, let's say it has 80 shares remaining (20 consumed).
        dom2['ask1'] = 100.3
        dom2['aqty1'] = 80
        # Re-index levels 2 to 20 for dom2 to be consistent
        for i in range(2, 21):
            dom2[f'ask{i}'] = 100.3 + (i - 1) * 0.1
            dom2[f'aqty{i}'] = 100
            
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        # We expect 3 inferred trade records:
        # 1. 100 shares at 100.1 (cleared)
        # 2. 100 shares at 100.2 (cleared)
        # 3. 20 shares at 100.3 (partially cleared)
        self.assertEqual(len(trades), 3)
        self.assertTrue((trades['direction'] == 'BUY').all())
        self.assertEqual(list(trades['trade_price']), [100.1, 100.2, 100.3])
        self.assertEqual(list(trades['trade_qty']), [100, 100, 20])

    def test_bid_price_crossing_multiple_levels(self):
        """A decrease in bid1 price should infer SELL trades clearing intermediate levels"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        
        dom2 = dict(ts=self.ts2, **self.base_dom)
        # bid1 shifts from 99.9 to 99.7.
        # Levels at 99.9 (qty 100) and 99.8 (qty 100) are cleared.
        # Level 99.7 (qty 100) has 90 shares remaining (10 consumed).
        dom2['bid1'] = 99.7
        dom2['bqty1'] = 90
        for i in range(2, 21):
            dom2[f'bid{i}'] = 99.7 - (i - 1) * 0.1
            dom2[f'bqty{i}'] = 100
            
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        # We expect 3 inferred trade records:
        # 1. 100 shares at 99.9 (cleared)
        # 2. 100 shares at 99.8 (cleared)
        # 3. 100 - 90 = 10 shares at 99.7 (partially cleared)
        self.assertEqual(len(trades), 3)
        self.assertTrue((trades['direction'] == 'SELL').all())
        self.assertEqual(list(trades['trade_price']), [99.9, 99.8, 99.7])
        self.assertEqual(list(trades['trade_qty']), [100, 100, 10])

    def test_fallback_ltp_change(self):
        """If LTP changes but no quantities decrease, record fallback trade"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        
        dom2 = dict(ts=self.ts2, **self.base_dom)
        # LTP moves to 100.2 (which is >= midpoint 100.0) without changing quantities
        dom2['ltp'] = 100.2
        
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]['trade_price'], 100.2)
        self.assertEqual(trades.iloc[0]['trade_qty'], 1)
        self.assertEqual(trades.iloc[0]['direction'], 'BUY')

    def test_empty_book_ignored(self):
        """Ask or bid initialization from 0.0 price should not trigger crossings"""
        dom1 = dict(ts=self.ts1, **self.base_dom)
        dom1['ask1'] = 0.0  # Empty ask book initially
        dom1['aqty1'] = 0
        
        dom2 = dict(ts=self.ts2, **self.base_dom)  # ask1 is now 100.1, aqty1 is 100
        
        df = pd.DataFrame([dom1, dom2])
        trades = TradeInferenceEngine.infer_trades(df)
        
        # Should NOT trigger any crossings at 0.0 or from 0.0 to 100.1
        # It should show 0 trades because ask1 went from empty (0) to 100.1 (not a crossing trade)
        self.assertEqual(len(trades), 0)


if __name__ == "__main__":
    unittest.main()
