"""
Per-symbol 1-minute bar accumulator from live ticks.
Matches DuckDB aggregation: AVG(ltp), LAST(ltp), AVG(DOM levels), SUM(trade flow).
Time-filtered to 09:15:00-15:29:59 IST.
"""

import time
import logging
import statistics
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

MARKET_OPEN_IST = "09:15:00"
MARKET_CLOSE_IST = "15:29:59"


class BarAccumulator:
    """Per-symbol 1-minute bucket from tick data."""

    def __init__(self):
        # Persistent state across resets
        self._recent_trade_qtys = deque(maxlen=2000)
        self._large_threshold = float('inf')
        self._prev_dom = {}
        self._prev_ltp = 0.0
        self._prev_ts = 0.0
        self._pending_bar = None
        self.reset()

    def reset(self):
        self.min_ts: int = 0
        self.ltp_sum = 0.0
        self.ltp_count = 0
        self.first_ltp = 0.0
        self.last_ltp = 0.0
        self.high_ltp = 0.0
        self.low_ltp = float('inf')
        self.bid_sum = [0.0] * 5
        self.ask_sum = [0.0] * 5
        self.bqty_sum = [0.0] * 5
        self.aqty_sum = [0.0] * 5
        self.dom_count = 0
        self.total_bid_qty_sum = 0.0
        self.total_ask_qty_sum = 0.0
        self.buy_vol = 0
        self.sell_vol = 0
        self.volume = 0
        self.trade_count = 0
        self.large_trade_count = 0
        self.vwap_sum = 0.0
        self.vwap_qty = 0.0
        self._prev_dom = {}
        self._prev_ltp = 0.0

    @staticmethod
    def _ts_ist_str(ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(IST)
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def _is_market_hours(ist_str: str) -> bool:
        return MARKET_OPEN_IST <= ist_str <= MARKET_CLOSE_IST

    def add_tick(self, tick: dict) -> Optional[dict]:
        """Process one tick. Returns 1-min bar if minute boundary crossed, else None.
        tick keys: ts(float), ltp, bid1..bid5, bqty1..bqty5, ask1..aqty5,
                   total_bid_qty, total_ask_qty
        Trades are inferred from DOM level changes (matches batch TradeInferenceEngine).
        """
        ts = tick.get("ts", 0.0)
        ist_str = self._ts_ist_str(ts)
        if not self._is_market_hours(ist_str):
            return None

        # Gap >1h means overnight/weekend — clear prev state to avoid stale inference
        if self._prev_ts > 0 and ts - self._prev_ts > 3600.0:
            self._prev_dom = {}
            self._prev_ltp = 0.0
        self._prev_ts = ts

        # Return pending bar from previous minute boundary - REMOVED pending bar logic
        min_ts = int(ts) // 60

        flushed_bar = None
        if self.ltp_count > 0 and min_ts != self.min_ts:
            # Save prev BEFORE flush so cross-boundary comparison is preserved
            saved_prev = dict(self._prev_dom) if self._prev_dom else {}
            saved_prev_ltp = self._prev_ltp
            flushed_bar = self.flush()
            self._prev_dom = saved_prev
            self._prev_ltp = saved_prev_ltp

        if self.ltp_count == 0:
            self.min_ts = min_ts
            self.first_ltp = tick.get("ltp", 0.0)

        ltp = tick.get("ltp", 0.0)
        self.ltp_sum += ltp
        self.ltp_count += 1
        self.last_ltp = ltp
        if ltp > self.high_ltp:
            self.high_ltp = ltp
        if ltp < self.low_ltp:
            self.low_ltp = ltp

        # DOM level accumulation (no bid1 != 0.0 filter)
        for i in range(5):
            self.bid_sum[i] += tick.get(f"bid{i+1}", 0.0)
            self.ask_sum[i] += tick.get(f"ask{i+1}", 0.0)
            self.bqty_sum[i] += tick.get(f"bqty{i+1}", 0)
            self.aqty_sum[i] += tick.get(f"aqty{i+1}", 0)
        self.total_bid_qty_sum += tick.get("total_bid_qty", 0)
        self.total_ask_qty_sum += tick.get("total_ask_qty", 0)
        self.dom_count += 1

        # ── DOM-based trade inference (matches batch TradeInferenceEngine exactly) ──
        trades = []  # list of (price, qty, direction)

        if self._prev_dom:
            c_bid1 = tick.get("bid1", 0.0)
            c_ask1 = tick.get("ask1", 0.0)
            p_bid1 = self._prev_dom.get("bid1", 0.0)
            p_ask1 = self._prev_dom.get("ask1", 0.0)

            # ── BUY detection (ask side consumption) ──
            if c_ask1 == p_ask1 and p_ask1 > 0:
                dec = self._prev_dom.get("aqty1", 0) - tick.get("aqty1", 0)
                if dec > 0:
                    trades.append((p_ask1, int(dec), "BUY"))
            elif c_ask1 > p_ask1 and p_ask1 > 0:
                for i in range(5):
                    p_price = self._prev_dom.get(f"ask{i+1}", 0.0)
                    p_qty = int(self._prev_dom.get(f"aqty{i+1}", 0))
                    if p_price > 0 and p_qty > 0 and p_price < c_ask1:
                        trades.append((p_price, p_qty, "BUY"))
                    elif p_price == c_ask1 and p_qty > 0:
                        # Partial: old level i is now best ask — remaining qty is at aqty1
                        dec = p_qty - int(tick.get("aqty1", 0))
                        if dec > 0:
                            trades.append((p_price, dec, "BUY"))

            # ── SELL detection (bid side consumption) ──
            if c_bid1 == p_bid1 and p_bid1 > 0:
                dec = self._prev_dom.get("bqty1", 0) - tick.get("bqty1", 0)
                if dec > 0:
                    trades.append((p_bid1, int(dec), "SELL"))
            elif c_bid1 < p_bid1 and p_bid1 > 0:
                for i in range(5):
                    p_price = self._prev_dom.get(f"bid{i+1}", 0.0)
                    p_qty = int(self._prev_dom.get(f"bqty{i+1}", 0))
                    if p_price > 0 and p_qty > 0 and p_price > c_bid1:
                        trades.append((p_price, p_qty, "SELL"))
                    elif p_price == c_bid1 and p_qty > 0:
                        # Partial: old level i is now best bid — remaining qty is at bqty1
                        dec = p_qty - int(tick.get("bqty1", 0))
                        if dec > 0:
                            trades.append((p_price, dec, "SELL"))

            # Fallback: LTP changed with no DOM inference
            if not trades and ltp != self._prev_ltp and self._prev_ltp > 0 and ltp > 0:
                mid = (p_bid1 + p_ask1) / 2.0 if p_bid1 > 0 and p_ask1 > 0 else 0
                if mid > 0:
                    trades.append((ltp, 1, "BUY" if ltp >= mid else "SELL"))

        # Apply inferred trades to accumulator
        for price, qty, direction in trades:
            if direction == "BUY":
                self.buy_vol += qty
            else:
                self.sell_vol += qty
            self.volume += qty
            self.trade_count += 1
            self.vwap_sum += price * qty
            self.vwap_qty += qty
            self._recent_trade_qtys.append(qty)
            n = len(self._recent_trade_qtys)
            if n >= 3 and (n % 50 == 0 if n > 50 else n % 5 == 0):
                self._large_threshold = 10 * statistics.median(self._recent_trade_qtys)
            if self._large_threshold != float('inf') and qty > self._large_threshold:
                self.large_trade_count += 1

        # Save current DOM + LTP for next tick
        self._prev_dom = {k: tick.get(k, 0.0) for k in
            ["bid1","bqty1","bid2","bqty2","bid3","bqty3","bid4","bqty4","bid5","bqty5",
             "ask1","aqty1","ask2","aqty2","ask3","aqty3","ask4","aqty4","ask5","aqty5"]}
        self._prev_ltp = ltp

        return flushed_bar

    def flush(self) -> Optional[dict]:
        """Close current bar and return it. Resets accumulator."""
        if self.ltp_count == 0:
            return None
        n = self.ltp_count
        dom_n = self.dom_count or 1
        bar = {
            "ts": self.min_ts * 60,
            "open": self.first_ltp,
            "high": self.high_ltp if self.high_ltp != float('inf') else self.first_ltp,
            "low": self.low_ltp if self.low_ltp != float('inf') else self.first_ltp,
            "close": self.last_ltp,
            "ltp": self.ltp_sum / n,
            "buy_vol_1m": self.buy_vol,
            "sell_vol_1m": self.sell_vol,
            "volume_1m": self.volume,
            "trade_count_1m": self.trade_count,
            "large_trade_count_1m": self.large_trade_count,
            "vwap_1m": self.vwap_sum / self.vwap_qty if self.vwap_qty > 0 else 0.0,
            "total_bid_qty": self.total_bid_qty_sum / dom_n,
            "total_ask_qty": self.total_ask_qty_sum / dom_n,
            "dom_count": dom_n,
        }
        for i in range(5):
            bar[f"bid{i+1}"] = self.bid_sum[i] / dom_n
            bar[f"bqty{i+1}"] = self.bqty_sum[i] / dom_n
            bar[f"ask{i+1}"] = self.ask_sum[i] / dom_n
            bar[f"aqty{i+1}"] = self.aqty_sum[i] / dom_n
        self.reset()
        return bar
