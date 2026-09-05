"""
Tests: ML Scanner — bar accumulator, live computer, state machine, engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from features.bar_accumulator import BarAccumulator
from features.live_computer import LiveFeatureComputer, RingBuffer
from features.state_machine import DoubleROCStateMachine, Signal
from features.ml_engine import MLEngine


# ===== Ring Buffer =====

class TestRingBuffer:
    def test_push_and_mean(self):
        b = RingBuffer(3)
        for v in [1.0, 2.0, 3.0]:
            b.push(v)
        assert b.filled == 3
        assert b.mean() == 2.0

    def test_eviction(self):
        b = RingBuffer(3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            b.push(v)
        assert b.filled == 3
        assert b.mean() == 3.0

    def test_get_offset(self):
        b = RingBuffer(5)
        for v in range(5):
            b.push(float(v))
        assert b.get(-1) == 4.0
        assert b.get(-3) == 2.0

    def test_get_beyond_limit(self):
        b = RingBuffer(3)
        b.push(1.0)
        assert np.isnan(b.get(-5))

    def test_std_single(self):
        b = RingBuffer(3)
        b.push(5.0)
        assert b.std() == 0.0


# ===== Bar Accumulator =====

class TestBarAccumulator:
    def test_single_tick(self):
        acc = BarAccumulator()
        ts = int(datetime(2026, 6, 25, 3, 45, 30, tzinfo=timezone.utc).timestamp())
        tick = {"ts": ts, "ltp": 38.0, "bid1": 38.0, "ask1": 38.1,
                "bqty1": 1000, "aqty1": 800, "total_bid_qty": 5000, "total_ask_qty": 4000}
        result = acc.add_tick(tick)
        assert result is None

    def test_pre_market_ignored(self):
        acc = BarAccumulator()
        ts = int(datetime(2026, 6, 25, 3, 39, 0, tzinfo=timezone.utc).timestamp())
        tick = {"ts": ts, "ltp": 38.0, "bid1": 38.0, "ask1": 38.1}
        result = acc.add_tick(tick)
        assert result is None
        assert acc.ltp_count == 0

    def test_flush_at_minute_boundary(self):
        acc = BarAccumulator()
        ts1 = int(datetime(2026, 6, 25, 3, 45, 30, tzinfo=timezone.utc).timestamp())
        tick1 = {"ts": ts1, "ltp": 38.0, "bid1": 38.0, "ask1": 38.1,
                 "bqty1": 1000, "aqty1": 800,
                 "total_bid_qty": 5000, "total_ask_qty": 4000}
        acc.add_tick(tick1)

        # Same minute — DOM change triggers inference
        ts2 = int(datetime(2026, 6, 25, 3, 45, 55, tzinfo=timezone.utc).timestamp())
        tick2 = {"ts": ts2, "ltp": 38.0, "bid1": 38.0, "ask1": 38.1,
                 "bqty1": 900, "aqty1": 800,  # bqty1 decrease → SELL 100 inferred
                 "total_bid_qty": 4900, "total_ask_qty": 4000}
        acc.add_tick(tick2)

        # Next minute — triggers flush (bar returned on next tick)
        ts3 = int(datetime(2026, 6, 25, 3, 46, 5, tzinfo=timezone.utc).timestamp())
        tick3 = {"ts": ts3, "ltp": 38.0, "bid1": 38.0, "ask1": 38.1,
                 "bqty1": 900, "aqty1": 800,
                 "total_bid_qty": 4900, "total_ask_qty": 4000}
        bar = acc.add_tick(tick3)
        assert bar is not None
        assert bar["open"] == 38.0
        assert bar["close"] == 38.0
        assert bar["buy_vol_1m"] == 0
        assert bar["sell_vol_1m"] == 100  # bid qty decreased 1000→900 at same price

    def test_bar_has_dom_fields(self):
        acc = BarAccumulator()
        ts = int(datetime(2026, 6, 25, 3, 45, 30, tzinfo=timezone.utc).timestamp())
        tick = {"ts": ts, "ltp": 38.0,
                "bid1": 38.0, "bid2": 37.9, "bqty1": 1000, "bqty2": 500,
                "ask1": 38.1, "ask2": 38.2, "aqty1": 800, "aqty2": 400,
                "total_bid_qty": 5000, "total_ask_qty": 4000}
        acc.add_tick(tick)
        ts2 = int(datetime(2026, 6, 25, 3, 46, 5, tzinfo=timezone.utc).timestamp())
        tick2 = {"ts": ts2, "ltp": 38.5,
                 "bid1": 38.5, "bid2": 38.4, "bqty1": 900, "bqty2": 600,
                 "ask1": 38.6, "ask2": 38.7, "aqty1": 700, "aqty2": 500,
                 "total_bid_qty": 4500, "total_ask_qty": 3800}
        acc.add_tick(tick2)
        bar = acc.flush()
        assert bar is not None
        assert bar["total_bid_qty"] > 0
        assert bar["bid1"] > 0
        assert bar["ask1"] > 0
        assert bar["bqty1"] > 0


# ===== Live Feature Computer =====

class TestLiveFeatureComputer:
    def test_volume_burst_constant(self):
        fc = LiveFeatureComputer()
        for _ in range(20):
            bar = self._make_bar(close=100.0, buy=50, sell=50, vol=100, tc=10)
            fc.push_bar(bar)
        bar20 = self._make_bar(close=100.0, buy=50, sell=50, vol=100, tc=10,
                               bid1=99, ask1=101, bqty1=500, aqty1=500,
                               total_bid=1150, total_ask=1150, vwap=100,
                               bqty2=300, bqty3=200, bqty4=100, bqty5=50,
                               aqty2=300, aqty3=200, aqty4=100, aqty5=50)
        feat = fc.push_bar(bar20)
        assert feat.shape == (18,)
        assert feat[2] == 1.0

    def test_roc_3(self):
        fc = LiveFeatureComputer()
        for i, price in enumerate([100.0, 101.0, 102.0, 103.0]):
            bar = self._make_bar(close=price, buy=50, sell=50, vol=100, tc=10)
            fc.push_bar(bar)
        assert abs(fc.roc_3 - 3.0) < 0.01

    def test_imbalance_top5(self):
        fc = LiveFeatureComputer()
        for _ in range(5):
            fc.push_bar(self._make_bar(close=100, buy=50, sell=50, vol=100, tc=10))
        bar = self._make_bar(close=100, buy=50, sell=50, vol=100, tc=10,
                             bqty1=2000, bqty2=0, bqty3=0, bqty4=0, bqty5=0,
                             aqty1=1000, aqty2=0, aqty3=0, aqty4=0, aqty5=0)
        feat = fc.push_bar(bar)
        assert abs(feat[5] - 0.3333) < 0.01

    def test_spread(self):
        fc = LiveFeatureComputer()
        for _ in range(5):
            fc.push_bar(self._make_bar(close=100, buy=50, sell=50, vol=100, tc=10))
        bar = self._make_bar(close=100, buy=50, sell=50, vol=100, tc=10,
                             bid1=99.5, ask1=100.5)
        feat = fc.push_bar(bar)
        assert feat[6] == 1.0

    def _make_bar(self, close=100.0, buy=50, sell=50, vol=100, tc=10,
                  bid1=100, ask1=101, bqty1=500, aqty1=500,
                  bqty2=300, bqty3=200, bqty4=100, bqty5=50,
                  aqty2=300, aqty3=200, aqty4=100, aqty5=50,
                  total_bid=1150, total_ask=1150, vwap=100,
                  large_tc=0):
        return {
            "close": close, "ltp": close, "buy_vol_1m": buy, "sell_vol_1m": sell,
            "volume_1m": vol, "trade_count_1m": tc, "large_trade_count_1m": large_tc,
            "bid1": bid1, "ask1": ask1,
            "bqty1": bqty1, "bqty2": bqty2, "bqty3": bqty3, "bqty4": bqty4, "bqty5": bqty5,
            "aqty1": aqty1, "aqty2": aqty2, "aqty3": aqty3, "aqty4": aqty4, "aqty5": aqty5,
            "total_bid_qty": total_bid, "total_ask_qty": total_ask,
            "vwap_1m": vwap, "ts": 1750806900,
        }


# ===== State Machine =====

class TestStateMachine:
    def test_idle_to_arming(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        bar = self._make_bar()
        result = sm.step(bar, "LONG", 0.25, 0.5)
        assert result is None
        assert sm.state == "ARMING"

    def test_arming_to_mark1(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        sm.state = "ARMING"
        bar = self._make_bar()
        result = sm.step(bar, "LONG", 0.25, 1.5)
        assert result is None
        assert sm.state == "MARK1"

    def test_mark1_to_between(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        sm.state = "MARK1"
        sm.mark1_ts = "09:30:00"
        bar = self._make_bar()
        result = sm.step(bar, "LONG", 0.25, 0.5)
        assert result is None
        assert sm.state == "BETWEEN"

    def test_full_trigger_long(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        bar1 = self._make_bar(close=100, delta=1000, ts=1750806900,
                              prob_long=0.25, prob_short=0.0)
        sm.step(bar1, "LONG", 0.25, 0.5)

        bar2 = self._make_bar(close=101, delta=1000, ts=1750806960,
                              prob_long=0.25, prob_short=0.0)
        sm.step(bar2, "LONG", 0.25, 2.0)

        bar3 = self._make_bar(close=100, delta=1000, ts=1750807020,
                              prob_long=0.25, prob_short=0.0)
        sm.step(bar3, "LONG", 0.25, 0.5)

        bar4 = self._make_bar(close=102, delta=1000, ts=1750807080,
                              prob_long=0.25, prob_short=0.0, daily_open=99)
        sig = sm.step(bar4, "LONG", 0.25, 2.0, daily_open=99,
                      prob_long=0.25, prob_short=0.0)
        assert sig is not None
        assert sig.direction == "LONG"
        assert sig.entry_price == 102

    def test_full_trigger_short(self):
        sm = DoubleROCStateMachine(long_th=0.25, short_th=0.20)
        bar1 = self._make_bar(close=500, delta=-1000, ts=1750806900,
                              prob_long=0.0, prob_short=0.25)
        sm.step(bar1, "SHORT", 0.25, -0.5)

        bar2 = self._make_bar(close=498, delta=-1000, ts=1750806960,
                              prob_long=0.0, prob_short=0.25)
        sm.step(bar2, "SHORT", 0.25, -2.0)

        bar3 = self._make_bar(close=499, delta=-1000, ts=1750807020,
                              prob_long=0.0, prob_short=0.25)
        sm.step(bar3, "SHORT", 0.25, -0.5)

        bar4 = self._make_bar(close=496, delta=-1000, ts=1750807080,
                              prob_long=0.0, prob_short=0.25, daily_open=500)
        sig = sm.step(bar4, "SHORT", 0.25, -2.0, daily_open=500,
                      prob_long=0.0, prob_short=0.25)
        assert sig is not None
        assert sig.direction == "SHORT"
        assert sig.entry_price == 496

    def test_reset_daily(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        sm.state = "MARK1"
        sm.reset_daily()
        assert sm.state == "IDLE"

    def test_daily_open_filter(self):
        sm = DoubleROCStateMachine(long_th=0.20, short_th=0.25)
        bar1 = self._make_bar(close=100, ts=1750806900)
        sm.step(bar1, "LONG", 0.25, 0.5)

        bar2 = self._make_bar(close=101, ts=1750806960)
        sm.step(bar2, "LONG", 0.25, 2.0)

        bar3 = self._make_bar(close=100, ts=1750807020)
        sm.step(bar3, "LONG", 0.25, 0.5)

        bar4 = self._make_bar(close=102, daily_open=103, ts=1750807080)
        sig = sm.step(bar4, "LONG", 0.25, 2.0, daily_open=103)
        assert sig is not None
        assert sig.daily_open_passed is False

    def _make_bar(self, close=100.0, delta=0, ts=1750806900,
                  prob_long=0.0, prob_short=0.0, prob_no_trade=0.0,
                  daily_open=0, symbol="TEST"):
        return {
            "ts": ts, "close": close, "ltp": close, "symbol": symbol,
            "delta_1m": delta, "daily_open": daily_open,
            "prob_long": prob_long, "prob_short": prob_short,
            "prob_no_trade": prob_no_trade,
            "buy_vol_1m": 100, "sell_vol_1m": 100, "volume_1m": 200,
            "trade_count_1m": 10, "large_trade_count_1m": 0,
            "bid1": 99.5, "ask1": 100.5, "vwap_1m": close,
            "bqty1": 500, "bqty2": 300, "bqty3": 200, "bqty4": 100, "bqty5": 50,
            "aqty1": 500, "aqty2": 300, "aqty3": 200, "aqty4": 100, "aqty5": 50,
            "total_bid_qty": 1150, "total_ask_qty": 1150,
        }


# ===== MLEngine (no-model test) =====

class TestMLEngine:
    def test_start_without_model(self):
        engine = MLEngine()
        engine.model_path = ""
        engine.start()
        assert engine.model_loaded is False

    def test_start_with_bad_path(self):
        engine = MLEngine()
        engine.model_path = "nonexistent.txt"
        engine.start()
        assert engine.model_loaded is False

    def test_on_tick_no_model(self):
        engine = MLEngine()
        engine.start()
        engine.on_tick({"ts": 1750806900, "ltp": 100}, "TEST")
