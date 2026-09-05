"""
Per-symbol rolling feature computer — 18 microstructure features.
Mirrors batch TickFeaturesCalculator + FeatureFactory exactly.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "large_trade_ratio", "delta_1m", "volume_burst", "aggressor_ratio",
    "trade_count_burst", "imbalance_top5", "spread", "depth_drop_bid",
    "depth_drop_ask", "vwap_distance", "volatility_5m", "price_acceleration",
    "iceberg_score", "bid_replenishment_rate",
    "absorption_buyer_1m", "absorption_buyer_5m",
    "absorption_seller_1m", "absorption_seller_5m",
]


class RingBuffer:
    """Fixed-size ring buffer for rolling windows."""
    def __init__(self, size: int):
        self.size = size
        self._buf = np.zeros(size, dtype=np.float64)
        self._idx = 0
        self._count = 0

    def push(self, val: float):
        self._buf[self._idx] = val
        self._idx = (self._idx + 1) % self.size
        self._count = min(self._count + 1, self.size)

    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return float(self._buf[:self._count].mean())

    def std(self) -> float:
        if self._count < 2:
            return 0.0
        return float(self._buf[:self._count].std(ddof=1))

    def sum(self) -> float:
        return float(self._buf[:self._count].sum())

    def get(self, offset: int) -> float:
        if abs(offset) > self._count:
            return float('nan')
        return float(self._buf[(self._idx + offset) % self.size])

    def serialize(self) -> dict:
        return {
            "buf": self._buf[:self._count].tolist() if self._count > 0 else [],
            "idx": self._idx,
            "count": self._count
        }

    def deserialize(self, d: dict):
        buf = d.get("buf", [])
        self._count = d.get("count", 0)
        self._idx = d.get("idx", 0)
        self._buf = np.zeros(self.size, dtype=np.float64)
        for i, val in enumerate(buf):
            if i < self.size:
                self._buf[i] = val

    @property
    def filled(self) -> int:
        return self._count


class LiveFeatureComputer:
    """Per-symbol state. Rolling windows + 18 features at each bar."""

    def __init__(self):
        self.volume_1m_20 = RingBuffer(20)
        self.trade_count_1m_20 = RingBuffer(20)
        # bar["ltp"] (avg) ring — size 6 so get(-6) gives 5-bar gap for return_5m
        self.ltp_avg_6 = RingBuffer(6)
        # bar["ltp"] (avg) ring — size 5 for volatility_5m (rolling(5).std)
        self.ltp_avg_5 = RingBuffer(5)
        # bar["close"] (last) ring — for ROC_3
        self.ltp_close_5 = RingBuffer(5)
        self.buy_vol_1m_5 = RingBuffer(5)
        self.sell_vol_1m_5 = RingBuffer(5)
        self.volume_1m_5 = RingBuffer(5)
        self.prev_total_bid_qty = 0.0
        self.prev_total_ask_qty = 0.0
        self.prev_bid1 = 0.0
        self.prev_bar = None
        self.roc_3 = 0.0

    def push_bar(self, bar: dict) -> np.ndarray:
        """Push new 1-min bar. Return 18-feature vector. Updates roc_3."""
        self.volume_1m_20.push(float(bar["volume_1m"]))
        self.trade_count_1m_20.push(float(bar["trade_count_1m"]))
        self.ltp_avg_6.push(float(bar["ltp"]))
        self.ltp_close_5.push(float(bar["close"]))
        self.buy_vol_1m_5.push(float(bar["buy_vol_1m"]))
        self.sell_vol_1m_5.push(float(bar["sell_vol_1m"]))
        self.volume_1m_5.push(float(bar["volume_1m"]))
        self.ltp_avg_5.push(float(bar["ltp"]))

        tc = max(bar["trade_count_1m"], 1)
        large_trade_ratio = bar["large_trade_count_1m"] / tc

        delta_1m = bar["buy_vol_1m"] - bar["sell_vol_1m"]
        avg_vol_20 = self.volume_1m_20.mean()
        volume_burst = bar["volume_1m"] / avg_vol_20 if avg_vol_20 > 0 else 1.0
        current_vol = bar["volume_1m"]
        if current_vol > 0:
            aggressor_ratio = bar["buy_vol_1m"] / current_vol
        else:
            aggressor_ratio = 0.5
        avg_tc_20 = self.trade_count_1m_20.mean()
        trade_count_burst = bar["trade_count_1m"] / avg_tc_20 if avg_tc_20 > 0 else 1.0

        # Imbalance/depth/replen from per-level sums (matches batch: sum(bqty1..5))
        total_bid = sum(float(bar.get(f"bqty{i+1}", 0)) for i in range(5))
        total_ask = sum(float(bar.get(f"aqty{i+1}", 0)) for i in range(5))
        denom = total_bid + total_ask
        imbalance_top5 = (total_bid - total_ask) / denom if denom > 0 else 0.0

        spread = float(bar.get("ask1", 0) - bar.get("bid1", 0))
        depth_drop_bid = 0.0 if self.prev_total_bid_qty == 0.0 else (total_bid - self.prev_total_bid_qty)
        depth_drop_ask = 0.0 if self.prev_total_ask_qty == 0.0 else (total_ask - self.prev_total_ask_qty)

        # vwap_distance uses avg LTP (batch: df['ltp'])
        vwap = float(bar.get("vwap_1m", bar["ltp"]))
        vwap_distance = (bar["ltp"] - vwap) / vwap if vwap > 0 else 0.0

        # volatility_5m uses 5-bar rolling std on avg LTP (matches batch rolling(5).std())
        volatility_5m = self.ltp_avg_5.std()
        prev_ltp = self.ltp_avg_6.get(-2)
        prev_prev_ltp = self.ltp_avg_6.get(-3)
        price_acceleration = bar["ltp"] - 2 * prev_ltp + prev_prev_ltp

        # Iceberg score — return 0.0 when displayed = 0 (matches batch _compute_iceberg_score)
        displayed = 0.0
        for i in range(5):
            displayed += float(bar.get(f"bqty{i+1}", 0))
            displayed += float(bar.get(f"aqty{i+1}", 0))
        iceberg_score = min(100.0, bar["volume_1m"] / displayed * 100.0) if displayed > 0 else 0.0

        # Bid replenishment rate (matches batch _compute_bid_replenishment_rate)
        bid1 = float(bar.get("bid1", 0))
        pbq = self.prev_total_bid_qty
        if bid1 >= self.prev_bid1 and pbq > 0:
            bid_replenishment_rate = min(1.0, max(0.0, (total_bid - pbq) / pbq))
        else:
            bid_replenishment_rate = 0.0

        # ── Absorption features (match tick_features.py exactly) ──

        # delta_1m_ratio = delta_1m / volume_1m  (current bar volume, not rolling avg)
        vol_1m = max(bar["volume_1m"], 1e-8)
        delta_1m_ratio = delta_1m / vol_1m

        # return_1m = (ltp[t] - ltp[t-1]) / ltp[t-1]  — uses avg LTP
        return_1m = 0.0
        if not np.isnan(prev_ltp) and prev_ltp > 0:
            return_1m = (bar["ltp"] - prev_ltp) / prev_ltp

        # 5-bar sums (batch uses rolling(5).sum(), not .mean())
        sum_buy_5 = self.buy_vol_1m_5.sum()
        sum_sell_5 = self.sell_vol_1m_5.sum()
        delta_5m = sum_buy_5 - sum_sell_5
        sum_vol_5 = self.volume_1m_5.sum()
        delta_5m_ratio = delta_5m / max(sum_vol_5, 1e-8)

        # return_5m = (ltp[t] - ltp[t-5]) / ltp[t-5]  — uses avg LTP, 5-bar gap
        ltp_5ago = self.ltp_avg_6.get(-6)
        return_5m = 0.0
        if not np.isnan(ltp_5ago) and ltp_5ago > 0:
            return_5m = (bar["ltp"] - ltp_5ago) / ltp_5ago

        # Continuous absorption magnitudes (matching batch exactly)
        absorption_buyer_1m = -delta_1m_ratio * return_1m if (delta_1m_ratio < 0 and return_1m > 0) else 0.0
        absorption_seller_1m = delta_1m_ratio * -return_1m if (delta_1m_ratio > 0 and return_1m < 0) else 0.0
        absorption_buyer_5m = -delta_5m_ratio * return_5m if (delta_5m_ratio < 0 and return_5m > 0) else 0.0
        absorption_seller_5m = delta_5m_ratio * -return_5m if (delta_5m_ratio > 0 and return_5m < 0) else 0.0

        # ROC_3 (uses close = LAST(ltp), matching batch's close_price)
        ltp_minus_3 = self.ltp_close_5.get(-4)
        if not np.isnan(ltp_minus_3) and ltp_minus_3 > 0:
            self.roc_3 = (bar["close"] - ltp_minus_3) / ltp_minus_3 * 100.0
        else:
            self.roc_3 = 0.0

        # Update prev state
        self.prev_total_bid_qty = total_bid
        self.prev_total_ask_qty = total_ask
        self.prev_bid1 = bid1
        self.prev_bar = bar

        return np.array([
            large_trade_ratio, delta_1m, volume_burst, aggressor_ratio,
            trade_count_burst, imbalance_top5, spread, depth_drop_bid,
            depth_drop_ask, vwap_distance, volatility_5m, price_acceleration,
            iceberg_score, bid_replenishment_rate,
            absorption_buyer_1m, absorption_buyer_5m,
            absorption_seller_1m, absorption_seller_5m,
        ], dtype=np.float64)

    def serialize(self) -> dict:
        return {
            "volume_1m_20": self.volume_1m_20.serialize(),
            "trade_count_1m_20": self.trade_count_1m_20.serialize(),
            "ltp_avg_6": self.ltp_avg_6.serialize(),
            "ltp_avg_5": self.ltp_avg_5.serialize(),
            "ltp_close_5": self.ltp_close_5.serialize(),
            "buy_vol_1m_5": self.buy_vol_1m_5.serialize(),
            "sell_vol_1m_5": self.sell_vol_1m_5.serialize(),
            "volume_1m_5": self.volume_1m_5.serialize(),
            "prev_total_bid_qty": self.prev_total_bid_qty,
            "prev_total_ask_qty": self.prev_total_ask_qty,
            "prev_bid1": self.prev_bid1,
            "roc_3": self.roc_3,
            "prev_bar": self.prev_bar,
        }

    def deserialize(self, d: dict):
        self.volume_1m_20.deserialize(d.get("volume_1m_20", {}))
        self.trade_count_1m_20.deserialize(d.get("trade_count_1m_20", {}))
        self.ltp_avg_6.deserialize(d.get("ltp_avg_6", {}))
        self.ltp_avg_5.deserialize(d.get("ltp_avg_5", {}))
        self.ltp_close_5.deserialize(d.get("ltp_close_5", {}))
        self.buy_vol_1m_5.deserialize(d.get("buy_vol_1m_5", {}))
        self.sell_vol_1m_5.deserialize(d.get("sell_vol_1m_5", {}))
        self.volume_1m_5.deserialize(d.get("volume_1m_5", {}))
        self.prev_total_bid_qty = d.get("prev_total_bid_qty", 0.0)
        self.prev_total_ask_qty = d.get("prev_total_ask_qty", 0.0)
        self.prev_bid1 = d.get("prev_bid1", 0.0)
        self.roc_3 = d.get("roc_3", 0.0)
        self.prev_bar = d.get("prev_bar")

